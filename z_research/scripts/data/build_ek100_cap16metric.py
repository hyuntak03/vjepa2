"""EK100 action anticipation 의 `cap16metric` train / val CSV 를 만든다 (2026-10-21).

⚠️ **사후 복원이다.** 이 부분집합은 2026-10-17 에 즉석으로 만들어졌고 생성 스크립트가 남지 않았다
(`z_research/scripts/data/build_ek100_subset.py` 는 `1of{N}_{후보}` 계열만 만든다 — cap16metric 은 거기 없다).
2026-10-21 에 저장된 CSV 두 개에서 규칙을 역으로 맞추고, 이 스크립트의 출력이 **sha256 바이트 단위로 같은지**
확인했다 (`--verify`, 아래 `EXPECT_SHA256`). 그러니 규칙은 추정이 아니라 재현된 것이다.

규칙
----
지표 클래스 (metric class) = **full train 과 (공식 filter 를 거친) val 에 둘 다 있는** 클래스.
  action 1,114 · verb 73 · noun 201.  `filter_annotations` = train 에 있는 (verb, noun) 조합만 남김
  (공식 규약. EK100 README §4 — val 9,668 -> 9,296 / 138 video).

  **train** `EPIC_100_train_cap16metric_seed0.csv`
      지표 action 클래스마다 full train 에서 **min(그 클래스 빈도, 16) 개**를 무작위로 뽑는다.
      지표 클래스가 아닌 action 은 통째로 버린다 (val 에 없으니 지표에 안 들어간다).
      -> 11,470 행 / 470 video / 32 participant, full (67,217) 의 1/5.9.
      클래스 1,114 개 전부 포함 · 490 개가 상한 16 에 걸리고 96 개는 표본이 1 개뿐이다.
      난수: `np.random.default_rng(0)` 하나로 `groupby("action")` 순회하며 그룹마다
      `random_state=int(rng.integers(1<<30))` — 그룹 순서 (문자열 정렬) 에 의존한다.

  **val** `EPIC_100_validation_cap16metric_seed0.csv`
      ⚠️ **부분집합이 아니다.** 공식 filter 를 건 val 전수 9,296 행 / 138 video 그대로다.
      이름에 cap16metric 이 붙은 건 train 과 짝을 맞추려는 것뿐이고 **val 은 한 행도 안 버렸다.**
      -> "train 만 1/5.9 로 줄이고 평가는 전수" 가 이 설정의 성격이다.

왜 16 인가 — 2026-10-17 사용자 결정. 긴 꼬리를 평탄화해 **mean class recall@5** (클래스 평균) 에서
한 번도 안 나온 클래스가 생기지 않게 한다. 영상 단위로 1/3 을 자른 옛 `third` 는 action 클래스를 34 % 잃어
실측 action 21.8 (full 35.3) 로 무너졌다 (`build_ek100_subset.py` 각주).

⚠️ full train (67,217) 기준선 action 35.34 와 **직접 비교 금지** — train 이 1/5.9 다.
  cap16metric 기준선: `ar_vast_ep70` verb 48.88 / noun 49.01 / action 33.10 (g4, 2026-10-18).

⚠️ `.gitignore` 가 `*csv` 를 무시한다 — 만들어진 CSV 는 **git 에 없다.** 이 스크립트가 정본이고,
  CSV 를 잃으면 여기서 다시 만든다 (바이트 동일).

  python z_research/scripts/data/build_ek100_cap16metric.py --verify   # 기존 CSV 와 sha256 대조만 (쓰지 않음)
  python z_research/scripts/data/build_ek100_cap16metric.py            # 없으면 쓰고, 있으면 대조
"""
from __future__ import annotations

import argparse
import hashlib
import io
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
ANN = Path("/data/hyuntak/project/2026/2027_cvpr/epic-kitchens-100-annotations")
OUT = REPO / "data_csv/ek100"
CAP, SEED = 16, 0

# 2026-10-21 에 실물 CSV 에서 읽은 값. 이 스크립트가 같은 바이트를 내는지의 기준이다
EXPECT_SHA256 = {
    "EPIC_100_train_cap16metric_seed0.csv": "354af52ed8f0bffc" ,  # 앞 16 자리만 (아래 full 과 대조)
    "EPIC_100_validation_cap16metric_seed0.csv": "62163c6b4cf95cc2",
}
EXPECT_ROWS = {"EPIC_100_train_cap16metric_seed0.csv": 11470,
               "EPIC_100_validation_cap16metric_seed0.csv": 9296}


def act(d):
    return d.verb_class.astype(str) + "_" + d.noun_class.astype(str)


def build():
    tr = pd.read_csv(ANN / "EPIC_100_train.csv")
    va = pd.read_csv(ANN / "EPIC_100_validation.csv")

    # 공식 filter_annotations — train 에 있는 (verb, noun) 조합만
    combo = set(zip(tr.verb_class, tr.noun_class))
    val = va[[(v, n) in combo for v, n in zip(va.verb_class, va.noun_class)]]

    tr_a, val_a = act(tr), act(val)
    metric = set(tr_a) & set(val_a)                      # 지표 action 클래스 1,114

    rng = np.random.default_rng(SEED)
    tr = tr.assign(action=tr_a)
    parts = [g.sample(n=min(CAP, len(g)), random_state=int(rng.integers(1 << 30)))
             for k, g in tr.groupby("action") if k in metric]
    sub = pd.concat(parts).drop(columns=["action"]).sort_values(["video_id", "start_frame"])

    print(f"지표 클래스  action {len(metric)} · verb {len(set(tr.verb_class) & set(val.verb_class))}"
          f" · noun {len(set(tr.noun_class) & set(val.noun_class))}")
    print(f"train  {len(sub):,} 행 / video {sub.video_id.nunique()} / participant {sub.participant_id.nunique()}"
          f"   (full {len(tr):,} 의 1/{len(tr)/len(sub):.1f})")
    print(f"val    {len(val):,} 행 / video {val.video_id.nunique()}   (공식 filter 전수 — 부분집합 아님)")
    return {"EPIC_100_train_cap16metric_seed0.csv": sub,
            "EPIC_100_validation_cap16metric_seed0.csv": val}


def sha_of(df):
    b = io.StringIO()
    df.to_csv(b, index=False)
    return hashlib.sha256(b.getvalue().encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="쓰지 않고 기존 CSV 와 sha256 만 대조")
    a = ap.parse_args()
    bad = False
    for name, df in build().items():
        p, h = OUT / name, sha_of(df)
        assert len(df) == EXPECT_ROWS[name], f"{name}: {len(df)} 행 (기대 {EXPECT_ROWS[name]})"
        if p.exists():
            got = hashlib.sha256(p.read_bytes()).hexdigest()
            ok = got[:16] == h[:16] == EXPECT_SHA256[name]
            print(f"  {'OK  ' if ok else 'DIFF'} {name}  재현 {h[:16]} / 실물 {got[:16]} / 기대 {EXPECT_SHA256[name]}")
            bad |= not ok
        elif a.verify:
            print(f"  없음 {name}")
            bad = True
        else:
            OUT.mkdir(parents=True, exist_ok=True)
            df.to_csv(p, index=False)
            print(f"  쓰기 {p}  sha {h[:16]}")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
