# pip install xgboost scikit-learn pandas joblib matplotlib
import re, json, joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import xgboost as xgb
from xgboost import XGBClassifier
# 콜백은 버전에 따라 위치가 다를 수 있어 try/except
try:
    from xgboost.callback import EarlyStopping, EvaluationMonitor
    HAVE_EVAL_MON = True
except Exception:
    from xgboost.callback import EarlyStopping
    EvaluationMonitor = None
    HAVE_EVAL_MON = False

import matplotlib.pyplot as plt

# ============ 1) 데이터 로드 & 정리 ============
df = pd.read_csv("./person_mood_filled.csv")

for c in ["mood_parent","mood_child","mood_code","utterance_1","utterance_2","utterance_3"]:
    if c in df.columns:
        df[c] = df[c].astype("string").str.strip()

# wide -> long (문장 3개를 각각 한 행으로)
long_df = df.melt(
    id_vars=["person_id","person","mood_parent","mood_child","mood_code"],
    value_vars=["utterance_1","utterance_2","utterance_3"],
    var_name="slot", value_name="text"
)

# 빈 문장/결측/중복 제거
long_df["text"] = long_df["text"].astype("string").str.strip()
long_df = long_df[long_df["text"].notna() & (long_df["text"] != "")]
long_df = long_df.drop_duplicates(subset=["text", "mood_code"]).reset_index(drop=True)

print(long_df.head())

# ============ 2) 학습/검증 분할 ============
X = long_df["text"].values
y = long_df["mood_code"].values

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

# ============ 3) TF-IDF(char n-gram) ============
vec = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2,5),
    min_df=2,
    max_features=200_000,
    lowercase=False
)
X_train_vec = vec.fit_transform(X_train)
X_valid_vec = vec.transform(X_valid)

# ============ 4) 라벨 인코딩 ============
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_valid_enc = le.transform(y_valid)
n_classes = len(le.classes_)

# 클래스 불균형 보정(옵션)
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(n_classes),
    y=y_train_enc
)
cw_map = {i:w for i,w in enumerate(class_weights)}
sample_weight = np.vectorize(cw_map.get)(y_train_enc)

# ============ 5) 모델(XGBoost) ============
eval_set = [(X_train_vec, y_train_enc), (X_valid_vec, y_valid_enc)]

model = xgb.XGBClassifier(
    objective="multi:softprob",
    num_class=n_classes,           # sklearn 래퍼는 없어도 되지만 명시해도 OK
    n_estimators=800,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    tree_method="hist",
    eval_metric="mlogloss",
    n_jobs=-1
)

callbacks = [EarlyStopping(rounds=50, save_best=True, maximize=False)]
if HAVE_EVAL_MON:
    callbacks.append(EvaluationMonitor(period=20))  # 20라운드마다 로그

model.fit(
    X_train_vec, y_train_enc,
    eval_set=eval_set,
    sample_weight=sample_weight,
    verbose=False,
    callbacks=callbacks
)

# ============ 6) 검증 성능 ============
pred_valid = model.predict(X_valid_vec)
print(classification_report(y_valid_enc, pred_valid, target_names=le.classes_, digits=4))
cm = confusion_matrix(y_valid_enc, pred_valid)
print("Confusion matrix shape:", cm.shape)

# ============ 7) 학습 곡선(mlogloss) ============
# 버전에 따라 evals_result 접근이 다를 수 있어 안전하게 처리
try:
    res = model.evals_result()
except Exception:
    res = model.get_booster().evals_result()

# 키 이름은 보통 validation_0/validation_1
train_key = [k for k in res.keys() if "validation_0" in k or k=="validation_0"]
valid_key = [k for k in res.keys() if "validation_1" in k or k=="validation_1"]
train_key = train_key[0] if train_key else list(res.keys())[0]
valid_key = valid_key[0] if valid_key else list(res.keys())[-1]

train_loss = res[train_key]["mlogloss"]
valid_loss = res[valid_key]["mlogloss"]

plt.figure(figsize=(7,4))
plt.plot(train_loss, label="train mlogloss")
plt.plot(valid_loss, label="valid mlogloss")
plt.xlabel("Boosting round"); plt.ylabel("mlogloss")
plt.title("XGBoost Learning Curves")
plt.legend(); plt.tight_layout()
plt.savefig("learning_curve.png", dpi=150)
print("saved -> learning_curve.png")

# 베스트 이터레이션/스코어
if hasattr(model, "best_iteration"):
    print("best_iteration:", model.best_iteration)
if hasattr(model, "best_score"):
    print("best_valid_mlogloss:", model.best_score)

# ============ 8) 저장 ============
joblib.dump({"vectorizer": vec, "label_encoder": le, "model": model}, "mood_clf_xgb.joblib")
print("saved -> mood_clf_xgb.joblib")

# ============ 9) 추론 함수 ============
def predict_mood(texts, topk=3):
    if isinstance(texts, str):
        texts = [texts]
    Xv = vec.transform([t.strip() for t in texts])
    proba = model.predict_proba(Xv)  # (N, C)
    top_idx = np.argsort(-proba, axis=1)[:, :topk]
    out = []
    for i, row in enumerate(top_idx):
        items = [(le.classes_[j], float(proba[i, j])) for j in row]
        out.append(items)
    return out

print(predict_mood("더 많은 조개를 가져가고 싶어.", topk=3))
