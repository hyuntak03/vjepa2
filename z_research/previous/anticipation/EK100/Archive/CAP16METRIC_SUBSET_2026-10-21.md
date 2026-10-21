# EK100 `cap16metric` — train / val 이 정확히 무엇인가 (2026-10-21, 사후 복원)

2026-10-17 사용자 결정으로 쓰기 시작한 EK100 부분집합의 **생성 규칙이 어디에도 기록돼 있지 않았다.**
`GROUP4_CAP16METRIC_2026-10-18.md` · `GROUP5_VAST_VARIANTS_2026-10-18.md` · `eval_ek100.sh` 는 결과 행 수
(train 11,470 / val 9,296) 만 적었고, `build_ek100_subset.py` 는 `1of{N}_{후보}` 계열만 만든다 — cap16metric 은 거기 없다.

2026-10-21 에 저장된 CSV 두 개에서 규칙을 역으로 맞췄다. 복원한 생성기의 출력이 실물 CSV 와 **sha256 바이트 단위로 같다**
(train `354af52e…` · val `62163c6b…`). 그러니 아래는 추정이 아니라 재현된 사실이다.

생성기 **`z_research/scripts/data/build_ek100_cap16metric.py`** (git 추적).

---

## 1. 한 줄

> **train 만 1/5.9 로 줄였고, val 은 공식 전수다.** train 은 action 클래스마다 최대 16 개씩 (긴 꼬리 평탄화).

| | 행 | video | participant | 성격 |
|---|---:|---:|---:|---|
| full train (공식) | 67,217 | 495 | — | — |
| **`EPIC_100_train_cap16metric_seed0.csv`** | **11,470** | **470** | **32** | 클래스당 상한 16 **부분집합** |
| full val (공식 원본) | 9,668 | 138 | — | — |
| **`EPIC_100_validation_cap16metric_seed0.csv`** | **9,296** | **138** | — | **공식 filter 전수 — 안 줄였다** |

⚠️ **val 의 `cap16metric` 은 이름뿐이다.** train 과 짝을 맞추려 같은 접미사를 붙였을 뿐,
`filter_annotations` 를 건 공식 val 전수와 **한 행도 다르지 않다.** "val 도 샘플링했다" 고 읽으면 틀린다.

## 2. 지표 클래스 (metric class)

지표 = **mean class recall@5**, 클래스 평균이다. 그래서 어떤 클래스가 train 에 한 번도 안 나오면
그 클래스는 구조적으로 0 점이고, 부분집합의 성능 하한이 **클래스 덮음**으로 정해진다.

지표 클래스 = **full train 과 (공식 filter 를 건) val 에 둘 다 있는** 클래스:

| | 개수 |
|---|---:|
| action | **1,114** |
| verb | **73** |
| noun | **201** |

⚠️ 이 숫자는 **filter 를 건 val (9,296)** 기준이다. filter 전 val (9,668) 로 재면 verb 78 · noun 207 이 나온다
(action 은 1,114 로 같다). `GROUP4` 문서의 "1,114 / 73 / 201" 이 맞는 쪽이다.

⚠️ 이것은 **논문의 라벨 공간 (action 3,568 · verb 97 · noun 289)** 과 다르다. 논문 숫자는
"train 에 있는 (verb, noun) 조합" 전체이고, 지표 클래스는 거기서 다시 **val 에도 있는 것**만 남긴 것이다.
EK100 README §4 와 혼동하지 말 것.

## 3. train 부분집합을 뜨는 규칙

```
지표 action 클래스 1,114 개 각각에 대해:
    full train 에서 min(그 클래스의 빈도, 16) 개를 무작위로 뽑는다
지표 클래스가 아닌 action 은 통째로 버린다   # val 에 없으니 지표에 못 들어간다
```

- 난수: `np.random.default_rng(0)` **하나**로 `groupby("action")` 을 순회하며 그룹마다
  `random_state = int(rng.integers(1 << 30))`. **그룹 순서 (action 문자열 정렬) 에 의존한다** —
  `random_state=0` 을 그룹마다 고정하거나 `groupby.sample` 을 쓰면 **다른 CSV 가 나온다** (실측, 둘 다 불일치)
- 저장: `sort_values(["video_id", "start_frame"])`, 공식 CSV 와 같은 15 컬럼 (파생 `action` 컬럼은 뺀다)

### 클래스당 표본 분포 (1,114 클래스)

| 표본 수 | 1 | 2 | 3 | 4 | 5 | 6~10 | 11~15 | **16 (상한)** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 클래스 수 | 96 | 87 | 66 | 52 | 38 | 170 | 115 | **490** |

- **1,114 클래스 전부 ≥ 1 개** (덮음 100 %) — 이게 상한 16 을 쓴 이유다
- 490 개 (44 %) 가 상한에 걸려 잘렸다. 96 개는 full train 에도 1 개뿐이라 **학습 1 개 · 평가 다수**다
- verb 73 / noun 201 도 **전부 덮인다**

### 왜 클래스 상한인가 — 영상 단위로 자르면 무너진다

옛 `EPIC_100_train_third_seed0.csv` (영상 단위 1/3) 는 action 클래스를 **34 % 잃었고**
실측 action **21.8** 로 떨어졌다 (full 35.3). 같은 크기라도 어떻게 자르느냐가 지표를 지배한다.
후보 다섯 (`video` · `segment` · `strat` · `strat_min3` · `capped`) 의 대리 지표 비교는
`z_research/scripts/data/build_ek100_subset.py` 가 만든다.

## 4. 읽을 때의 금지 사항

- ⚠️ **full train 기준선 action 35.34 와 직접 비교 금지.** train 이 1/5.9 다
- cap16metric 기준선 (g4, 2026-10-18, `ar_vast_ep70`): **verb 48.88 · noun 49.01 · action 33.10**.
  cap16metric 안에서만 비교한다
- ⚠️ `anticipation_point_mode` 를 같이 밝히지 않은 숫자는 비교 금지 — **기준선 규약은 `released`** 다.
  `paper` 로 재면 action 35.34 → 14.14 로 바뀐다 (`protocol.yaml:62` 기본값이 `paper` 라서 실제로 한 번 틀렸다,
  2026-10-21 g6 실행 폐기)
- val 은 전수이므로 "val 도 1/5.9" 같은 말은 틀렸다

## 5. CSV 는 git 에 없다

`.gitignore` 가 `*csv` 를 무시한다. `data_csv/ek100/*.csv` 는 **untracked 가 정상**이고
`git checkout` 으로 못 되돌린다. **정본은 생성 스크립트**이고, CSV 를 잃으면 거기서 바이트 동일하게 다시 만든다.

## 재현

```bash
python z_research/scripts/data/build_ek100_cap16metric.py --verify   # 기존 CSV 와 sha256 대조만 (쓰지 않음)
python z_research/scripts/data/build_ek100_cap16metric.py            # 없으면 쓰고, 있으면 대조
```
2026-10-21 실행 결과 (vll7, `vjepa2_bw`):
```
지표 클래스  action 1114 · verb 73 · noun 201
train  11,470 행 / video 470 / participant 32   (full 67,217 의 1/5.9)
val    9,296 행 / video 138   (공식 filter 전수 — 부분집합 아님)
  OK   EPIC_100_train_cap16metric_seed0.csv       재현 354af52ed8f0bffc / 실물 354af52ed8f0bffc
  OK   EPIC_100_validation_cap16metric_seed0.csv  재현 62163c6b4cf95cc2 / 실물 62163c6b4cf95cc2
```

쓰는 곳: `z_vll7_training/eval_ek100.sh:54-55` (`TRAIN_CSV` / `VAL_CSV` 기본값).
결과 문서: `GROUP4_CAP16METRIC_2026-10-18.md` (predictor 4종) · `GROUP5_VAST_VARIANTS_2026-10-18.md`.
