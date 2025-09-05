# pip install xgboost scikit-learn pandas joblib matplotlib
import os
import json
import math
import random
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Dict, Any, Tuple
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
XGB_LOG_EVERY = int(os.getenv("XGB_LOG_EVERY", "0"))  # 0이면 조용, 50이면 매 50라운드 로그
XGB_LEADER_N  = int(os.getenv("XGB_LEADER_N", "3"))   # 리더보드 상위 몇 개 보여줄지
# =========================
# 0) 유틸
# =========================
def safe_strip_str_col(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype("string").str.strip()

def print_head(df: pd.DataFrame, n=5, title=None):
    if title:
        print(title)
    with pd.option_context("display.max_colwidth", 120):
        print(df.head(n))

def predict_proba_with_best_iter(booster: xgb.Booster, dmat: xgb.DMatrix) -> np.ndarray:
    """XGBoost 버전별 best-iter 안전 예측."""
    # xgboost>=1.6: best_iteration 존재, best_ntree_limit는 deprecated일 수 있음
    if hasattr(booster, "best_iteration") and booster.best_iteration is not None:
        return booster.predict(dmat, iteration_range=(0, booster.best_iteration + 1))
    # 구버전 호환 (일부 버전에선 best_ntree_limit가 있을 수 있음)
    if getattr(booster, "best_ntree_limit", 0):
        return booster.predict(dmat, ntree_limit=booster.best_ntree_limit)
    # fallback
    return booster.predict(dmat)

def classification_report_sane(y_true_enc: np.ndarray,
                               y_pred_enc: np.ndarray,
                               le: LabelEncoder) -> str:
    """검증셋에 실제로 등장한 라벨만 대상으로 리포트 생성 (경고 제거)."""
    present = np.unique(y_true_enc)
    target_names = [le.classes_[i] for i in present]
    rep = classification_report(
        y_true_enc,
        y_pred_enc,
        labels=present,
        target_names=target_names,
        digits=4,
        zero_division=0
    )
    return rep

# =========================
# 1) 데이터 로드/정리
# =========================
train_df = pd.read_csv("./person_mood_filled.csv")
safe_strip_str_col(train_df, ["mood_parent","mood_child","mood_code","utterance_1","utterance_2","utterance_3"])

# wide -> long
long_df = train_df.melt(
    id_vars=["person_id","person","mood_parent","mood_child","mood_code"],
    value_vars=["utterance_1","utterance_2","utterance_3"],
    var_name="slot", value_name="text"
)
long_df["text"] = long_df["text"].astype("string").str.strip()
long_df = long_df[long_df["text"].notna() & (long_df["text"] != "")]
long_df = long_df.drop_duplicates(subset=["text", "mood_code"]).reset_index(drop=True)

X_train_text = long_df["text"].values
y_train_raw = long_df["mood_code"].values

print(f"train samples: {len(long_df)}  | #labels(train): {long_df['mood_code'].nunique()}")
print_head(long_df, title="(train) head")

# 외부 검증셋
valid_df = pd.read_csv("./ok_labels_text.csv")
safe_strip_str_col(valid_df, ["label","text"])
valid_df = valid_df[valid_df["text"].notna() & (valid_df["text"] != "")]
valid_df = valid_df[valid_df["label"].notna() & (valid_df["label"] != "")]
valid_df = valid_df.drop_duplicates(subset=["label","text"]).reset_index(drop=True)

# 학습에 없는 라벨 제거
train_label_set = set(pd.Series(y_train_raw).unique())
before = len(valid_df)
only_in_valid = sorted(list(set(valid_df["label"].unique()) - train_label_set))
if only_in_valid:
    print("[Info] 검증에만 있는 라벨 수:", len(only_in_valid))
valid_df = valid_df[valid_df["label"].isin(train_label_set)].copy()
dropped = before - len(valid_df)
print(f"[Info] 검증 라벨 정리: {before} -> {len(valid_df)} (학습에 없는 라벨 {dropped}건 제거)")

X_valid_text = valid_df["text"].values
y_valid_raw = valid_df["label"].values

print(f"valid samples: {len(valid_df)}  | #labels(valid after filter): {valid_df['label'].nunique()}")
print_head(valid_df, title="(valid) head")

# =========================
# 2) 벡터라이저 (문자 n-gram)
# =========================
vec = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2,5),
    min_df=2,
    max_features=200_000,
    lowercase=False
)
X_train_vec = vec.fit_transform(X_train_text)
X_valid_vec = vec.transform(X_valid_text)

# =========================
# 3) 라벨 인코딩
# =========================
le = LabelEncoder()
le.fit(y_train_raw)
y_train_enc = le.transform(y_train_raw)
y_valid_enc = le.transform(y_valid_raw)
n_classes = len(le.classes_)
print("#classes(train-only):", n_classes)

# =========================
# 4) 클래스 가중치 & DMatrix
# =========================
train_unique = np.unique(y_train_enc)
cw = compute_class_weight(class_weight="balanced", classes=train_unique, y=y_train_enc)
cw_map = {cls: w for cls, w in zip(train_unique, cw)}
sample_weight_train = np.array([cw_map[c] for c in y_train_enc], dtype=float)

dtrain_full = xgb.DMatrix(X_train_vec, label=y_train_enc, weight=sample_weight_train)
dvalid = xgb.DMatrix(X_valid_vec, label=y_valid_enc)

# =========================
# 5) 랜덤 서치 기반 오토 하이퍼파라미터
#    - 조기종료는 valid mlogloss
#    - 최적 선택은 macro F1
# =========================
SEARCH_TRIALS = int(os.getenv("XGB_TRIALS", "30"))
NUM_BOOST_ROUND = int(os.getenv("XGB_N_EST", "2000"))
EARLY_STOP = int(os.getenv("XGB_EARLY", "100"))

def sample_params() -> Dict[str, Any]:
    return {
        "objective": "multi:softprob",
        "num_class": n_classes,
        "eta": 10 ** np.random.uniform(-2.0, -0.7),          # ~0.01~0.2
        "max_depth": int(np.random.choice([4,5,6,7,8,9])),
        "min_child_weight": float(np.random.choice([1,2,3,5,7,10])),
        "subsample": float(np.random.uniform(0.6, 1.0)),
        "colsample_bytree": float(np.random.uniform(0.6, 1.0)),
        "colsample_bylevel": float(np.random.uniform(0.6, 1.0)),
        "lambda": 10 ** np.random.uniform(-2.0, 1.0),        # L2
        "alpha": 10 ** np.random.uniform(-4.0, 0.5),         # L1
        "min_split_loss": float(np.random.uniform(0.0, 5.0)),# gamma
        "tree_method": "hist",
        "eval_metric": "mlogloss",
        "nthread": -1,
        "seed": RANDOM_SEED,
        "verbosity": 1,
    }

def train_one(params: Dict[str, Any]) -> Tuple[xgb.Booster, Dict[str, Any], Dict[str, list]]:
    evals_result = {}
    evals = [(dtrain_full, "train"), (dvalid, "valid")]
    booster = xgb.train(
        params=params,
        dtrain=dtrain_full,
        num_boost_round=NUM_BOOST_ROUND,
        evals=evals,
        early_stopping_rounds=EARLY_STOP,
        verbose_eval=(XGB_LOG_EVERY if XGB_LOG_EVERY > 0 else False),  # ★ 추가
        evals_result=evals_result
    )
    return booster, params, evals_result

def evaluate_booster(booster: xgb.Booster) -> Dict[str, Any]:
    proba_valid = predict_proba_with_best_iter(booster, dvalid)
    pred_valid = np.argmax(proba_valid, axis=1)

    macro_f1 = f1_score(y_valid_enc, pred_valid, average="macro", zero_division=0)
    micro_f1 = f1_score(y_valid_enc, pred_valid, average="micro", zero_division=0)
    acc = accuracy_score(y_valid_enc, pred_valid)
    return {
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "accuracy": float(acc),
        "pred_valid": pred_valid,
        "proba_valid": proba_valid
    }
leader = []  # 진행 중 성능 기록용
def print_topk(leader_sorted, k=3):
    print("\n[Leaderboard - Top {}] (by macro F1 then valid mlogloss)".format(k))
    for i, r in enumerate(leader_sorted[:k], 1):
        print(f" {i:>2}. macro_f1={r['macro_f1']:.4f} | mlogloss={r['best_valid_mlogloss']} | iter={r['best_iteration']}")
    print()
best = {
    "macro_f1": -1.0,
    "booster": None,
    "params": None,
    "evals_result": None,
    "metrics": None,
}

print(f"\n[Search] Trials = {SEARCH_TRIALS}, num_boost_round = {NUM_BOOST_ROUND}, early_stopping = {EARLY_STOP}\n")

for t in range(1, SEARCH_TRIALS + 1):
    params = sample_params()
    booster, used_params, evals_result = train_one(params)
    metrics = evaluate_booster(booster)
    leader.append({
        "trial": t,
        "macro_f1": metrics["macro_f1"],
        "best_iteration": getattr(booster, "best_iteration", None),
        "best_valid_mlogloss": getattr(booster, "best_score", None)
    })
    leader_sorted = sorted(leader, key=lambda r: (-r["macro_f1"], float(r["best_valid_mlogloss"] or np.inf)))

    # 진행 요약 한 줄 + 현재까지 최고
    print(f" ↳ best_so_far macro_f1={best['macro_f1']:.4f} | "
        f"best_valid_mlogloss={getattr(best['booster'],'best_score', None)}")

    # 5트라이얼마다(또는 마지막) 상위 N개 보여주기
    if (t % 5 == 0) or (t == SEARCH_TRIALS):
        print_topk(leader_sorted, k=min(XGB_LEADER_N, len(leader_sorted)))

    best_iter = getattr(booster, "best_iteration", None)
    best_score = getattr(booster, "best_score", None)

    print(f"[Trial {t:02d}] macro_f1={metrics['macro_f1']:.4f} | micro_f1={metrics['micro_f1']:.4f} "
          f"| acc={metrics['accuracy']:.4f} | best_iter={best_iter} | best_valid_mlogloss={best_score}")

    # 선택 기준: macro F1 우선, 동률이면 valid mlogloss 더 낮은 쪽
    is_better = False
    if metrics["macro_f1"] > best["macro_f1"]:
        is_better = True
    elif math.isclose(metrics["macro_f1"], best["macro_f1"], rel_tol=1e-6):
        # 동률이면 mlogloss 비교 (낮을수록 좋음)
        curr_loss = float(getattr(booster, "best_score", np.inf) or np.inf)
        prev_loss = float(getattr(best["booster"], "best_score", np.inf) or np.inf) if best["booster"] else np.inf
        if curr_loss < prev_loss:
            is_better = True

    if is_better:
        best.update({
            "macro_f1": metrics["macro_f1"],
            "booster": booster,
            "params": used_params,
            "evals_result": evals_result,
            "metrics": metrics
        })

assert best["booster"] is not None, "모든 시도 실패: 설정/데이터를 확인하세요."

print("\n[Best Model]")
print(json.dumps({
    "macro_f1": best["metrics"]["macro_f1"],
    "micro_f1": best["metrics"]["micro_f1"],
    "accuracy": best["metrics"]["accuracy"],
    "best_iteration": getattr(best["booster"], "best_iteration", None),
    "best_valid_mlogloss": getattr(best["booster"], "best_score", None),
    "params": best["params"]
}, ensure_ascii=False, indent=2))

# =========================
# 6) 최적 모델 성능 리포트/혼동행렬
# =========================
best_booster: xgb.Booster = best["booster"]
best_pred_valid = best["metrics"]["pred_valid"]

report_text = classification_report_sane(y_valid_enc, best_pred_valid, le)
print("\n[Validation Classification Report]\n", report_text)

# 파일로 저장
with open("valid_report.txt", "w", encoding="utf-8") as f:
    f.write(report_text)

cm = confusion_matrix(y_valid_enc, best_pred_valid, labels=np.unique(y_valid_enc))
cm_labels = [le.classes_[i] for i in np.unique(y_valid_enc)]

plt.figure(figsize=(10, 10))
plt.imshow(cm, aspect="auto")
plt.title("Confusion Matrix (Valid)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(ticks=np.arange(len(cm_labels)), labels=cm_labels, rotation=90, fontsize=7)
plt.yticks(ticks=np.arange(len(cm_labels)), labels=cm_labels, fontsize=7)
plt.colorbar()
plt.tight_layout()
plt.savefig("confusion_matrix_valid.png", dpi=150)
print("saved -> confusion_matrix_valid.png")

# =========================
# 7) 러닝 커브(mlogloss) 그리기 (최적 trial의 기록 사용)
# =========================
res = best["evals_result"]  # booster.evals_result() 아님! (버전 호환)
train_loss = res["train"]["mlogloss"]
valid_loss = res["valid"]["mlogloss"]

plt.figure(figsize=(7,4))
plt.plot(train_loss, label="train mlogloss")
plt.plot(valid_loss, label="valid mlogloss")
plt.xlabel("Boosting round"); plt.ylabel("mlogloss")
plt.title("Best XGBoost Learning Curves (External Validation)")
plt.legend(); plt.tight_layout()
plt.savefig("best_learning_curve.png", dpi=150)
print("saved -> best_learning_curve.png")

# =========================
# 8) 아티팩트 저장 (모델/벡터라이저/라벨인코더/메타)
# =========================
ARTIFACT_PATH = "mood_clf_xgb_best.joblib"
joblib.dump({
    "vectorizer": vec,
    "label_encoder": le,
    "booster": best_booster,
    "best_iteration": getattr(best_booster, "best_iteration", None),
    "params": best["params"]
}, ARTIFACT_PATH)
print(f"saved -> {ARTIFACT_PATH}")

with open("best_params.json", "w", encoding="utf-8") as f:
    json.dump(best["params"], f, ensure_ascii=False, indent=2)
print("saved -> best_params.json")

# =========================
# 9) 추론 함수
# =========================
def predict_mood(texts, topk=3):
    art = joblib.load(ARTIFACT_PATH)
    vec_local = art["vectorizer"]
    le_local: LabelEncoder = art["label_encoder"]
    booster_local: xgb.Booster = art["booster"]
    best_iter_local = art.get("best_iteration", None)

    if isinstance(texts, str):
        texts = [texts]

    Xv = vec_local.transform([str(t).strip() for t in texts])
    dX = xgb.DMatrix(Xv)

    # best-iter 안전 예측
    if best_iter_local is not None:
        proba = booster_local.predict(dX, iteration_range=(0, best_iter_local + 1))
    elif getattr(booster_local, "best_ntree_limit", 0):
        proba = booster_local.predict(dX, ntree_limit=booster_local.best_ntree_limit)
    else:
        proba = booster_local.predict(dX)

    top_idx = np.argsort(-proba, axis=1)[:, :topk]
    out = []
    for i, row in enumerate(top_idx):
        items = [(le_local.classes_[j], float(proba[i, j])) for j in row]
        out.append(items)
    return out

# quick test
print("\n[Quick Test]")
print(predict_mood("더 많은 조개를 가져가고 싶어.", topk=3))
