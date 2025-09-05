import os, re, json, time, pathlib
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ====== 파일 경로 ======
INPUT_CSV  = r"C:\AI\Sentiment_Analysis\person_mood_template.csv"
OUTPUT_CSV = r"C:\AI\Sentiment_Analysis\person_mood_filled.csv"

# ====== 데이터 로드 ======
df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
persons   = df[["person_id","person"]].drop_duplicates().sort_values("person_id")
all_moods = list(df["mood_code"].drop_duplicates())
SLEEP_BETWEEN = 0.6

# ====== OpenAI 설정 ======
MODEL = "gpt-4o-mini"
TEMPERATURE = 0.7
MAX_RETRY = 3
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

PROMPT_TEMPLATE = """
[역할] 당신은 감정 라벨링 데이터 작가입니다.

[대상 인물]
- {person}

[요구사항]
- 아래 mood_code 목록(예: "기쁨_반가움")의 각 항목마다 그 감정이 잘 드러나는 한글 문장 3개를 만드세요.
- 문장 길이: 5~40자, 구어체, 1~2문장씩.
- 인물/상황 맥락 반영.
- 반드시 JSON으로만 출력.

[출력 형식(JSON)]
{{
  "person": "<원본 person 문자열>",
  "moods": [
    {{"mood_code": "기쁨_반가움", "utterances": ["...", "...", "..."]}}
  ]
}}

[mood_code 목록]
{mood_codes}
""".strip()

def _extract_text_from_responses(resp) -> str:
    t = getattr(resp, "output_text", None)
    if t:
        return t.strip()
    out = []
    for item in getattr(resp, "output", []) or []:
        if getattr(item, "type", None) == "message":
            for c in getattr(item, "content", []) or []:
                if getattr(c, "type", None) in ("output_text", "text"):
                    out.append(getattr(c, "text", ""))
    return "".join(out).strip()

def _parse_json_strict_or_loose(text: str) -> dict:
    s = text.strip()
    if s.startswith("```"):
        parts = s.split("```", 2)
        s = parts[1] if len(parts) >= 2 else text
        s = "\n".join(s.splitlines()[1:]) if s.lower().startswith("json") else s
    s = s.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", s)
    if not m:
        raise json.JSONDecodeError("No JSON object found", s, 0)
    return json.loads(m.group(0))

def call_model(person: str, mood_codes: list[str]) -> dict | None:
    prompt = PROMPT_TEMPLATE.format(
        person=person,
        mood_codes="\n".join(f"- {m}" for m in mood_codes),
    )
    logs_dir = pathlib.Path("logs"); logs_dir.mkdir(exist_ok=True)
    for attempt in range(1, MAX_RETRY + 1):
        raw_text = ""
        try:
            resp = client.responses.create(
                model=MODEL,
                input=prompt,
                temperature=TEMPERATURE,
                max_output_tokens=4096,
            )
            print(resp)
            raw_text = _extract_text_from_responses(resp)
            if not raw_text:
                raise ValueError("Empty model output")
            data = _parse_json_strict_or_loose(raw_text)
            if "moods" not in data or not isinstance(data["moods"], list):
                raise ValueError("Invalid JSON shape: 'moods' missing")
            return data
        except Exception as e:
            safe = re.sub(r"[^\w가-힣]+", "_", person)[:30]
            (logs_dir / f"{safe}_attempt{attempt}.txt").write_text(raw_text or "", encoding="utf-8")
            if attempt == MAX_RETRY:
                print(f"[ERROR] {person} 실패: {e}")
                return None
            time.sleep(1.2 * attempt)

# ====== 보조 유틸 ======
def canon(code: str) -> str:
    # 공백 전부 제거(띄어쓰기 오탈자 보정)
    return re.sub(r"\s+", "", str(code).strip())

def normalize_utt(u: str) -> str:
    if not isinstance(u, str): return ""
    s = re.sub(r"\s+", " ", u.strip())
    return s[:50]  # 너무 길면 잘라내기(필요시 조정)

def chunks(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i+size]

# ====== 실행: 사람별 호출(감정 2분할) → CSV 채우기 ======
filled = df.copy()

for _, row in persons.iterrows():
    pid, person = int(row.person_id), row.person
    print(f"[RUN] person_id={pid} :: {person}")

    store = {}  # canon(mood_code) -> [utt1, utt2, utt3]

    # 69개 감정 → 35/34로 분할 호출 (토큰/안정성)
    for part in chunks(all_moods, 35):
        out = call_model(person, part)
        time.sleep(SLEEP_BETWEEN)
        if not out: 
            continue
        for m in out.get("moods", []):
            mc = canon(m.get("mood_code", ""))
            utts = [normalize_utt(x) for x in m.get("utterances", []) if isinstance(x, str)]
            utts = (utts + ["", "", ""])[:3]
            store[mc] = utts

    # 해당 person의 모든 mood 행 채우기
    mask = (filled["person_id"] == pid)
    miss = 0
    for idx in filled[mask].index:
        mcode = filled.at[idx, "mood_code"]
        key = canon(mcode)
        utts = store.get(key)
        if not utts:
            miss += 1
            continue
        filled.at[idx, "utterance_1"] = utts[0]
        filled.at[idx, "utterance_2"] = utts[1]
        filled.at[idx, "utterance_3"] = utts[2]

    print(f"  ↳ filled: {len(filled[mask]) - miss} / {len(filled[mask])}, missed: {miss}")

    # 안정성 위해 사람 단위로 중간 저장
    filled.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print(f"[DONE] 저장됨: {OUTPUT_CSV}")
