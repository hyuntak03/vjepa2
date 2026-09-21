# 학습 데이터 레지스트리 (predictor 학습용)

> **2026-09-10: 학습 데이터는 아직 정하지 않았다.** 아래는 로컬에 **있는 것**을 파악해 적어 둔 후보 목록이다.
> 어느 것도 기본값이 아니고, config 의 `data.datasets` 는 비어 있다. 돌릴 때 `SET="data.datasets=[이름]"` 으로 준다.

`configs/training/*.yaml` 의 `data.datasets: [이름, ...]` 이 여기 `## 이름` 섹션으로 풀린다
(`z_training/harness/resolve_train.py`). 파서는 `configs/protocols/datasets.md` 와 같다 —
`## 이름` 아래의 `key: value` 줄만 읽고 `type` 이 없는 섹션은 설명으로 버린다.
리스트/딕셔너리 값은 한 줄 flow 문법으로 쓴다 (`[1, 2]`, `{a: 1}`).

공통 키 (`app/vjepa_frozen/data.py` 참고)
- `type: frames_index` — PNG 프레임 폴더 + index.csv (world_model_analysis 스키마)
  `root` `index_csv` `frames_root` `frames_pattern` `frames_start` `frames_stride`
  `frames_start_choices` (시간 jitter: 샘플마다 여기서 시작 프레임을 고른다)
  `include` / `exclude` (`{컬럼: [값]}` 행 필터, 기본 include={plausible: ["1"]}) `raw_frames` `weight`
- `type: video_csv` — `"<mp4> <label>"` 공백 구분 csv. `fps` / `duration` / `frame_step` 중 하나.

⚠️ **가능(plausible=1) 변이만 학습에 쓴다.** 불가능 변이는 라벨 없는 JEPA 손실에서도
"미래가 문맥과 안 맞는" 표본이라 predictor 에게 위반을 정상으로 가르친다.
⚠️ **채점 세트와 문맥을 공유하는 데이터로 학습하면 그 세트의 점수는 오염된다.** v11 계열은
아래에 두되 v11 채점에는 쓰지 말 것 (`v11_possible` 주석).

---

## intphys1_train

IntPhys 2019 **train 분할** (가능 영상만, 4중항 없음). 로컬에 3,750 scene
(`/local_datasets/world/IntPhys1/<id>/scene/scene_001..100.png`, 288×288). 원 train 은 15,000 인데
그중 3,750 만 받아 뒀다 (id 가 띄엄띄엄). 인덱스는 `z_training/data/build_intphys1_train_index.py`.
**IntPhys1 dev (intphys1_dev) 와 분할이 다르므로 dev 채점이 오염되지 않는다.** v11 과도 무관.
파일 번호가 1부터라 start 1..7 이 전부 예산(100) 안이다 (7 + 31×3 = 100).

**sliding 채점(`intphys1_sliding`, 보고 지표 `skip2_w32/avg`)에 맞춘 값이다**: raw 프레임 2칸 간격 32장 창을 영상
어디서든(시작 1~38) 자르고, 문맥 길이는 config 의 `mask.context_frames: [4, 8, 12, 16, 20]` (채점의 C 집합) 으로 준다.
v11 류(stride 3, 문맥 16 고정) 와 다르니 섞어 쓸 때 주의. 인덱스: `python z_training/data/build_intphys1_train_index.py`.

type: frames_index
root: /data/hyuntak/project/2026/2027_cvpr/vjepa2/data_csv/intphys1_train
index_csv: index.csv
frames_root: /local_datasets/world/IntPhys1
frames_pattern: "{file_name}/scene/scene_{frame:03d}.png"
frames_start: 1
frames_stride: 2
frames_start_choices: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38]
raw_frames: 100

## rollout_v2_training

RollOut_v2 의 학습셋 (무중력 직선 운동, 896 clip, 전부 가능). `configs/protocols/datasets.md ## rollout_v2_training`
과 같은 프레임. 저장 프레임이 0,3,...,99 (34장) 라 start 는 0/3/6 만 가능하다.

type: frames_index
root: /data/hyuntak/project/2026/2027_cvpr/vjepa2/data_csv/rollout_v2_training
index_csv: index.csv
frames_root: /local_datasets/world/world_analysis/RollOut_v2_training
frames_pattern: "{file_name}/{frame:06d}.png"
frames_start: 0
frames_stride: 3
frames_start_choices: [0, 3, 6]
raw_frames: 100

## rollout_v2_possible

RollOut_v2 (운동 법칙 7종, 5,488 clip) 의 **가능 변이만**. ledge/wall 의 불가능 짝은 include 필터로 빠진다.
`rollout_v2` 위치 readout 결과와 비교할 때 이걸로 학습하면 그 세트가 오염된다 — readout 은 held-out 이 필요하다.

type: frames_index
root: /data/hyuntak/project/2026/2027_cvpr/vjepa2/data_csv/rollout_v2
index_csv: index.csv
frames_root: /local_datasets/world/world_analysis/RollOut_v2
frames_pattern: "{file_name}/{frame:06d}.png"
frames_start: 0
frames_stride: 3
frames_start_choices: [0, 3, 6]
raw_frames: 100

## v11_possible

IntPhysGen v11 본체 6조건의 **가능 변이** (10,752 clip). ⚠️ **v11 / v11_full 채점과 문맥이 픽셀 단위로
같다** — 이걸로 학습한 predictor 의 v11 점수는 학습셋 점수다. 전이(IntPhys1 dev, rollout) 를 볼 때만 쓸 것.
`exclude` 로 조건을 뺄 수 있다: 예) `SET='data.datasets=[{"name":"v11_possible","exclude":{"condition":["static_occlusion"]}}]'`.

type: frames_index
root: /data/hyuntak/project/2026/2027_cvpr/vjepa2/data_csv/intphysgen_v11
index_csv: index.csv
frames_root: /local_datasets/world/world_analysis/IntPhysGen_v11
frames_pattern: "{file_name}/{frame:06d}.png"
frames_start: 0
frames_stride: 3
frames_start_choices: [0, 3, 6]
raw_frames: 100

## intphys2_train_mp4

IntPhys 2 Main 의 mp4 (808 clip, 2way train 분할, 가능·불가능 섞임). **미검증** — `frame_step` 은 추정값이고
라벨 컬럼(가능/불가능)을 VideoDataset 이 필터하지 않는다. 형식 예시로만 둔다.

type: video_csv
csv: /data/hyuntak/project/2026/2027_cvpr/vjepa2/data_csv/IntPhys2/IntPhys2_2way_train.csv
frame_step: 3

## intphys2_main_possible

**IntPhys 2 Main 의 가능 영상 506 개** (253 장면 × {1_Possible, 2_Possible}), mp4 직독 (2026-09-19 결정: 학습 = Main 가능, test = HeldOut 리더보드).
`z_training/data/build_intphys2_main_index.py --check --write` 산출. 512×512 · 60 fps, 길이 635–938 (전수 확인). `frame_step: 10` = 채점과 같은 6 fps.
⚠️ 논문 A.3 이 Main 메타데이터로 학습하지 말라고 적었다 — **학습 후 Main 점수는 논문 Table 2 와 비교 불가.** 유효한 test 는 HeldOut 뿐.
⚠️ `window_grid` 는 frames_index 전용이라 여기서는 못 쓴다 — 창 분포는 `n_frames` + 무작위 시작 + `mask.context_frames` 목록으로 맞춘다.

type: video_csv
csv: /data/hyuntak/project/2026/2027_cvpr/vjepa2/data_csv/IntPhys2/main_possible.csv
frame_step: 10

## intphys2_main_split_train

**IntPhys 2 Main 장면 split 의 train 절반, 가능 영상만** — 254 clip / 127 장면 (2026-09-19 결정: test 도 Main → 장면 단위 반반).
`z_training/data/build_intphys2_main_split.py --write --link` 산출 (seed 0, condition × Difficulty × Camera 층화). test = 126 장면 504 영상 (252 쌍),
채점기 `split: MainTest` (`/local_datasets/world/IntPhys2/MainTest/metadata.csv`, `Videos -> ../Main/Videos`).
⚠️ Main 은 논문이 학습 금지한 세트 — test 수치는 같은 test 장면의 릴리즈 predictor 값과만 비교한다.

type: video_csv
csv: /data/hyuntak/project/2026/2027_cvpr/vjepa2/data_csv/IntPhys2/main_split/train_possible.csv
frame_step: 10

## v11_split_train

**IntPhysGen v11 (12조건) 의 block 단위 train 절반, 가능 변이만** — 10,752 clip / 5,376 block.
`z_training/data/build_v11_split_index.py` 산출 (`data_csv/intphysgen_v11_split/`, seed 0, 50/50,
(condition × violation_type × sym_k) 117 셀 층화). 짝인 test 절반은 `configs/protocols/datasets.md ## v11_split_test`.
⚠️ **v11 / v11_full 전체 채점과는 여전히 문맥을 공유한다** — 학습 후 점수는 `v11_split_test` 로만 읽는다.

type: frames_index
root: /data/hyuntak/project/2026/2027_cvpr/vjepa2/data_csv/intphysgen_v11_split
index_csv: index_train.csv
frames_root: /local_datasets/world/world_analysis/IntPhysGen_v11
frames_pattern: "{file_name}/{frame:06d}.png"
frames_start: 0
frames_stride: 3
frames_start_choices: [0, 3, 6]
raw_frames: 100

## predictor_v1_training

**predictor 학습셋 v1** (2026-09-17 설계, `UnrealEngine/gen/PREDICTOR_V1_TRAINING_DESIGN.md`). 세 팔: `line` 무중력 등속 (속도·방향·위치 연속),
`occ` 지면 등속 + 트랩도어 가림 k=4 (속도·경계 위치 연속, 가림막 폭이 속도를 따름), `ramp` 10° 쐐기 등가속 (a 5–60 cm/s²·경계 속도 연속).
20,000 clip 렌더, 그중 **학습 index 14,250** (`index_train.csv`; held-out 모양 torus·cone, 색 cyan·purple, 배경 hex_rust, 속도 0.45–0.55 칸/튜블릿, 가속 25–35 는 `index_holdout.csv`).
raw 100 장 전부 저장이라 start 0…6 이 다 된다. README: `/data2/.../Predictor_v1_training/README.md`. index: `UnrealEngine/gen/build_predictor_v1_index.py --write`.
⚠️ 테스트(RollOut_v2, v11, IntPhys1, EK100)와 문맥을 공유하지 않는다.

type: frames_index
root: /data/hyuntak/project/2026/2027_cvpr/vjepa2/data_csv/predictor_v1_training
index_csv: index_train.csv
frames_root: /local_datasets/world/world_analysis/Predictor_v1_training
frames_pattern: "{file_name}/{frame:06d}.png"
frames_start: 0
frames_stride: 3
frames_start_choices: [0, 3, 6]
raw_frames: 100
