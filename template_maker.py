from pathlib import Path
import pandas as pd
from pandas.errors import EmptyDataError

# === 1) 경로 지정 (프로젝트에 맞게 수정) ===
INPUT_CSV = r"C:\AI\Sentiment_Analysis\person_mood_template.csv"

# === 2) person_type / mood_type 는 기존 코드 그대로 위에 정의되어 있다고 가정 ===
person_type = [
  "영화관에서 영화를 보고 나온 여자",
  "길에서 아는 사람을 만난 남자",
  "꽃을 사서 집에 가는 남자",
  "장례식에 간 여자",
  "도서관에서 책을 읽는 남자",
  "카페에서 커피를 마시는 여자",
  "지하철을 기다리는 남자",
  "마트에서 장을 보는 여자",
  "헬스장에서 운동하는 남자",
  "공원에서 산책하는 여자",
  "회사에서 야근하는 남자",
  "해변에서 사진 찍는 여자",
  "버스를 타고 출근하는 남자",
  "미술관을 관람하는 여자",
  "친구와 술집에 간 남자",
  "강아지를 산책시키는 여자",
  "편의점에서 물건을 고르는 남자",
  "길거리에서 버스킹을 보는 여자",
  "공항에서 비행기를 기다리는 남자",
  "백화점에서 쇼핑하는 여자",
  "학교에서 시험을 보는 남자",
  "놀이공원에서 놀이기구를 타는 여자",
  "병원에서 진료를 기다리는 남자",
  "음악회를 감상하는 여자",
  "등산을 하는 남자",
  "요가 학원에서 수업 듣는 여자",
  "콘서트에 간 남자",
  "미용실에서 머리하는 여자",
  "경기장에서 축구를 보는 남자",
  "바닷가에서 조개를 줍는 여자"
]
mood_type = [
  # 기쁨
  "기쁨_감동","기쁨_고마움","기쁨_공감","기쁨_기대감","기쁨_놀람","기쁨_만족감","기쁨_반가움",
  "기쁨_신뢰감","기쁨_신명남","기쁨_안정감","기쁨_자랑스러움","기쁨_자신감","기쁨_즐거움",
  "기쁨_통쾌함","기쁨_편안함",

  # 두려움
  "두려움_걱정","두려움_공포","두려움_놀람","두려움_위축감","두려움_초조함",

  # 미움(상대방)
  "미움(상대방)_경멸","미움(상대방)_냉담","미움(상대방)_반감","미움(상대방)_불신감",
  "미움(상대방)_비위상함","미움(상대방)_시기심","미움(상대방)_외면","미움(상대방)_치사함",

  # 분노
  "분노_날카로움","분노_발열","분노_불쾌","분노_사나움","분노_원망","분노_타오름",

  # 사랑
  "사랑_귀중함","사랑_너그러움","사랑_다정함","사랑_동정(슬픔)","사랑_두근거림",
  "사랑_매력적","사랑_아른거림","사랑_호감",

  # 수치심
  "수치심_미안함","수치심_부끄러움","수치심_죄책감",

  # 슬픔
  "슬픔_고통","슬픔_그리움","슬픔_동정(슬픔)","슬픔_무기력","슬픔_수치심","슬픔_실망",
  "슬픔_아픔","슬픔_억울함","슬픔_외로움","슬픔_절망","슬픔_허망","슬픔_후회",

  # 싫어함(상태)
  "싫어함(상태)_난처함","싫어함(상태)_답답함","싫어함(상태)_불편함",
  "싫어함(상태)_서먹함","싫어함(상태)_싫증","싫어함(상태)_심심함",

  # 욕망
  "욕망_갈등","욕망_궁금함","욕망_기대감","욕망_불만","욕망_아쉬움","욕망_욕심",

  # 중립(요청에서 빼지 않은 것만 유지)
  "중립_공감","중립_놀람","중립_다정함","중립_동정(슬픔)","중립_만족감","중립_안정감",
]

def build_template_df():
    rows = []
    for pid, p in enumerate(person_type, start=1):
        for m in mood_type:
            parent, child = m.split("_", 1)
            rows.append({
                "person_id": pid,
                "person": p,
                "mood_parent": parent,
                "mood_child": child,
                "mood_code": m,
                "utterance_1": "",
                "utterance_2": "",
                "utterance_3": ""
            })
    return pd.DataFrame(rows, columns=[
        "person_id","person","mood_parent","mood_child","mood_code",
        "utterance_1","utterance_2","utterance_3"
    ])

path = Path(INPUT_CSV)
need_create = (not path.exists()) or (path.stat().st_size == 0)

if need_create:
    df = build_template_df()
    df.to_csv(INPUT_CSV, index=False, encoding="utf-8-sig")
else:
    try:
        df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    except EmptyDataError:
        # 파일은 있는데 내용이 비어있을 때
        df = build_template_df()
        df.to_csv(INPUT_CSV, index=False, encoding="utf-8-sig")

print(df.head())