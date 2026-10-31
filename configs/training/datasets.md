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
- `type: video_window` — mp4 + **행마다 고정된 창** csv (`path,start,stride,n_frames,...`).
  무작위 샘플링이 아니라 인덱스가 지정한 프레임을 정확히 읽는다. `csv` `limit` `include`/`exclude` `decord_threads`.

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

## ssv2_min48

**Something-Something v2 에서 48 프레임 이상만** — **97,416 train / 984 val** (220,847 편 중 98,400 = 44.6 %).
`n_frames: 48` × `fps: 12` (fstp 1, 창 4.0 초) 용 — `natural_prefix` / `natural_ar` 의 데이터 (2026-09-27 Ariel 이식).
원본: vll5 로컬 HDD `/data2/local_datasets/something-something/something-something-v2-mp4` (심볼릭 `/local_datasets/something-something-v2-mp4`),
12 fps · 높이 320 · 라벨 없음 (csv 둘째 칸은 전부 0 — frozen-encoder JEPA 손실은 라벨을 안 쓴다).
⚠️ **vll5 에서만 돈다** (경로가 vll5 로컬).
인덱스: `python z_training/data/build_video_index.py ssv2 --probe --min-frames 48 --workers 16 --write`
→ `data_csv/ssv2/{lengths.tsv, train_min48.csv, val_min48.csv}` (gitignore — 이 명령으로 다시 만든다. vll5 CPU 16 스레드 약 2 분).
decord 헤더 길이 ≥ 48 로 거르고 seed 0 으로 1 % 를 val 로 뗀다. **Ariel 의 수 (97,416 / 984) 와 정확히 같다** (2026-09-27 빌드, probe 실패 0 편).

⚠️ **왜 미리 거르나** — `filter_short_videos` 는 짧은 영상을 만나면 **무작위 인덱스로 재추첨**한다.
통과율이 낮으면 성공 1 건당 영상을 여러 번 열게 되고, rank 하나가 배치를 못 채우면
**DDP 전체가 멈춘다** (Ariel 2026-09-22 실측: skipping 1,368 회 / step 0 회, GPU 7 장 100 % 공회전).
`train_min<N>.csv` 를 쓰면 런타임 거부가 **0** 이다. N = config 의 `n_frames × fstp`.
⚠️ val 분할은 **공식 SSv2 val 이 아니라** seed 0 무작위 1 % 다 (in-loop 감시용).
`decord_threads: 1` — 리더당 decord 스레드 (Ariel 기본). 안 주면 -1 (리더마다 전 코어) 이라 rank × worker 개 리더가 과다구독한다.

type: video_csv
csv: /data/hyuntak/project/2026/2027_cvpr/vjepa2/data_csv/ssv2/train_min48.csv
fps: 12
filter_short_videos: true
decord_threads: 1

## ssv2_min48_val

위 `ssv2_min48` 의 val 분할 (984 편, 같은 빌드). 같은 규약. 감시용이고 보고 지표가 아니다.

type: video_csv
csv: /data/hyuntak/project/2026/2027_cvpr/vjepa2/data_csv/ssv2/val_min48.csv
fps: 12
filter_short_videos: true
decord_threads: 1

## k400_train_40k

**Kinetics-400 train 무작위 40,000 영상** (2026-09-25, auto_research reach 학습). vll6 로컬 `/data2/local_datasets/Kinetics-400/videos_train` (전체 240,436 개 중 seed 0 섞기).
원본 320 × (426–682) · 30 fps. `fps: 12` → VideoDataset 이 frame step 30//12 = 2 (실효 15 fps, Ariel 학습과 같은 설정값).
⚠️ 경로가 vll6 로컬이라 **vll6 에서만** 돈다. csv: `z_training` 밖의 `data_csv/k400_train/train_40k_abs.csv` (gitignore; seed 0 로 다시 만든다).
평가 세트 (K400 val 인공 팬 · SSv2 · EK100 · v3) 와 영상이 겹치지 않는다 (val 은 videos_val).

type: video_csv
csv: /data/hyuntak/project/2026/2027_cvpr/vjepa2/data_csv/k400_train/train_40k_abs.csv
fps: 12

## ssv2_min48_vll6

**`ssv2_min48` 과 같은 97,416 편, 경로만 vll6 로컬** (`/data2/local_datasets/something-something-v2-mp4`, 2026-09-29 확인 — vll5 의 `something-something/something-something-v2-mp4` 와 같은 mp4 세트). ctx_ar 이어 돌리기 (vll6) 용.
⚠️ **vll6 에서만 돈다.** csv 는 `train_min48.csv` 의 경로 치환 (`data_csv/ssv2/train_min48_vll6.csv`, gitignore).

type: video_csv
csv: /data/hyuntak/project/2026/2027_cvpr/vjepa2/data_csv/ssv2/train_min48_vll6.csv
fps: 12

## pusht_noise

DINO-WM 의 Push-T 궤적 (OSF, 2026-10-09 수신). `${WORLD_ROOT}/action_planning/dino_wm/pusht_noise/{train,val}` —
train 18,685 episode (2,336,736 스텝), val 21. `obses/episode_%03d.mp4` 224px (프레임 = sim 스텝) + `states.pth` (agent xy, block xy, 각도)
+ `rel_actions.pth` (agent 목표 Δxy, px; PD 제어라 실제 이동의 약 40 %) + `seq_lengths.pkl`.
`type: pusht` 는 `app/vjepa_frozen/pusht_data.py` 가 읽는다 — frameskip 마다 관측 1 장을 2 번 복제해 tubelet 으로,
action 은 frameskip 개를 이어 붙여 2×frameskip 차원 (DINO-WM 규약), train 평균/표준편차로 정규화 (`action_stats_train.json`).
`data.n_frames = 2 × 관측 수`. aug 없음 (action 이 화면 좌표에 묶여 있다). 시뮬레이터·planner: `benchmarks/dino_wm`.

type: pusht
root: ${WORLD_ROOT}/action_planning/dino_wm/pusht_noise
frameskip: 5

## ssv2_fs1_min32

SSv2 webm 원본을 **frame_step 1** (12 fps 그대로) 로 — vast 브랜치 `natural_ctx_state` 와 같은 항목 (2026-10-10 이식).
0.083 s 간격, 32 장 = 2.7 s, 1 블록 0.17 s. 원본 32 장 이상만 (`build_video_index.py ssv2 --min-frames 32 --write`, train 187,724 줄).

type: video_csv
csv: ${DATA_CSV}/ssv2/train_min32.csv
frame_step: 1
filter_short_videos: true

## k400_320_fps4_f32

K400 320p 사본 (`k400_320_min96`) 을 **fps 4 × 32 장 (8 초)** 가 실제로 나오는 클립만 남긴 것 (2026-10-10) — vast `k710_320_fps4_min224` 의 K400 판.
VideoDataset 규칙 그대로 잰다: fstp = ceil(avg_fps) // 4, 필요 프레임 = 32 × fstp (30 fps → 224, 25 fps → 192).
`python z_training/data/filter_by_clip_len.py --csv data_csv/k400_320/train_min96.csv --fps 4 --n-frames 32 -o data_csv/k400_320/train_fps4_f32.csv`

type: video_csv
csv: ${DATA_CSV}/k400_320/train_fps4_f32.csv
fps: 4
filter_short_videos: true

## k400_320_fps6_f32

`k400_320_min96` 에서 **fps 6 × 32 장 (5.3 초)** 이 실제로 나오는 클립 (2026-10-10). 30 fps → fstp 5 → 160 장, 25 fps → 128 장 필요.
train 38,517 / 53,117 (72.5 %), val 383 / 536. fps 4 판 (`k400_320_fps4_f32`, 21,114 = 39.7 %) 은 K400_320 이 대부분 5~7 초 서브클립이라 너무 적다 —
602 만 clip 예산을 1 : 1 로 섞으면 클립당 143 번을 본다 (fps 6 은 78 번).

type: video_csv
csv: ${DATA_CSV}/k400_320/train_fps6_f32.csv
fps: 6
filter_short_videos: true

## wall_single

DINO-WM 의 Wall 궤적 (OSF, 2026-10-09 수신). `${WORLD_ROOT}/action_planning/dino_wm/wall_single` — 1,920 episode × 50 step, 점이 벽의 문을 지나 반대편으로.
`type: wall` 은 `app/vjepa_frozen/pusht_data.py:WallDataset` 이 읽는다. 관측은 `obses_u8.npy` (uint8 memmap, `convert_wall_obs.py`, 반올림 오차 ≤ 0.5/255),
split = DINO-WM 과 같은 무작위 90/10 (randperm seed 42) → train 1,728 / val 192. 문/벽 위치는 전부 30/32.
⚠️ 공개 env 는 action 을 2 배로 적용한다 (`wall.py:_generate_transition`) — 데이터는 1 배. planner 는 action / 2 를 넣는다 (`WallDataset.ACTION_SCALE = 2`).
   안쪽에서는 데이터 재생과 정확히 일치, 바깥 테두리에서만 env 가 벽을 따라 ≤ 0.6 미끄러진다 (데이터는 멈춤).

type: wall
root: ${WORLD_ROOT}/action_planning/dino_wm/wall_single
frameskip: 5

## vobench_split_train

> ⚠️ **누수 (2026-10-14)** — test 쪽 가능 영상 52 개와 같은 설정 · 궤적의 영상이 들어 있다. 새 학습은 `vobench_split2_train`.

**VOBench_v1 block 단위 train 절반, 가능 변이만** (2026-10-12) — 720 clip / 360 block (primitive · realistic).
`z_training/data/build_vobench_split_index.py` 산출 (`data_csv/vobench_v1_split/index_train.csv`, seed 0, objset × violation × occlusion × motion 층화).
짝인 test 절반은 `configs/protocols/datasets.md ## vobench_v1_split_test`. ⚠️ vobench_v1 전체 채점은 train block 을 포함한다 — 학습 뒤 점수는 split_test 로만.
원본 96 장 30 fps → stride 3 의 32 장. 시작 0 / 1 / 2 중 무작위 (raw 0–95 안).

type: frames_index
root: /data/hyuntak/project/2026/2027_cvpr/vjepa2/data_csv/vobench_v1_split
index_csv: index_train.csv
frames_root: /data2/local_datasets/world/Benchmark/VOBench_v1
frames_pattern: "{file_name}/{frame:06d}.png"
frames_start: 0
frames_stride: 3
frames_start_choices: [0, 1, 2]

## vobench_allpos_train

**VOBench_v1 전체의 가능 변이 (상한선용, 2026-10-14)** — 1,440 clip / 720 block. held-out 없음 — 채점 영상의 가능 arm 을 전부 본다.
`python z_training/data/build_vobench_split_index.py --version all`. 같은 test 쌍 (split v2 test) 에서 held-out 판 (`vobench_split2_train`) 과 견주는 데 쓴다.

type: frames_index
root: /data/hyuntak/project/2026/2027_cvpr/vjepa2/data_csv/vobench_v1_allpos
index_csv: index_train.csv
frames_root: /data2/local_datasets/world/Benchmark/VOBench_v1
frames_pattern: "{file_name}/{frame:06d}.png"
frames_start: 0
frames_stride: 3
frames_start_choices: [0, 1, 2]

## vobench_split2_train

**VOBench_v1 장면 묶음 단위 train 절반, 가능 변이만** (split v2, 2026-10-14) — 720 clip / 360 block.
`z_training/data/build_vobench_split_index.py` (기본 v2): 가능 arm 설정 (objset · 운동 · 배경 · 가림 · k · 물체) 을 공유하는 block 을 union-find 로 묶어 묶음째 배정 — **test 와 공유 설정 0**, (objset × 위반 × 가림) 셀마다 반반.
짝인 test 는 `configs/protocols/datasets.md ## vobench_v1_split2_test`.

type: frames_index
root: /data/hyuntak/project/2026/2027_cvpr/vjepa2/data_csv/vobench_v1_split_v2
index_csv: index_train.csv
frames_root: /data2/local_datasets/world/Benchmark/VOBench_v1
frames_pattern: "{file_name}/{frame:06d}.png"
frames_start: 0
frames_stride: 3
frames_start_choices: [0, 1, 2]

## blender_track_v0_train

**BlenderTrack v0 학습 4,000 clip (seed 0)** (2026-10-31, z_analysis/release/06_blender_track) — 눈높이 Blender 렌더, 연속 무작위 궤적 · 가림 사건,
30 fps × 180 장 · 256². type `blender_track` = clip 번호만 내고 창 · 읽기 · 라벨은 `app/vjepa_frozen/blender_track_data.py` 의 수집기 (data.blender_track 노브).
메타는 `z_training/data/build_blender_track_index.py` (clip 이름 + target 보이는 비율). 검증 400 clip (seed 1) 은 `val_meta.npz`.

type: blender_track
meta: /data/hyuntak/project/2026/2027_cvpr/vjepa2/data_csv/blender_track_v0/train_meta.npz

## vobench_v2_split_train

**VOBench_v2 궤적 묶음 단위 train 절반의 가능 변이 + 물체 궤적 라벨** (2026-10-31, z_analysis/release binding 학습) — 2,160 clip.
`z_training/data/build_vobench_v2_split_index.py`: 묶음 = (objset, 배경, 운동, 가림, k) — VOBench 의 궤적 · 가림막을 정하는 다섯 —
(objset, 가림, k) 층마다 seed 0 으로 절반. **test 와 공유 궤적 0.** 짝인 test 는 `data_csv/vobench_v2_split/index_test.csv` (1,080 block, 네 변이).
type `frames_index_track` = frames_index + 블록 폴더의 `label.json` 궤적 (app/vjepa_frozen/where_head.py). 라벨을 안 쓰는 팔도 같은 type 으로 읽어
데이터 순서를 맞춘다.

type: frames_index_track
root: /data/hyuntak/project/2026/2027_cvpr/vjepa2/data_csv/vobench_v2_split
index_csv: index_train.csv
frames_root: /data2/local_datasets/world/Benchmark/VOBench_v2
frames_pattern: "{file_name}/{frame:06d}.png"
frames_start: 0
frames_stride: 3
frames_start_choices: [0, 1, 2]

## vobench_split2g_train

**VOBench split v2 train (720) + VOBench gravity train 의 가능 'pos' (108) = 828 clip** (2026-10-15, 사용자: PFT 에 gravity 포함).
`z_training/data/build_vobench_gravity_split_index.py`: gravity 를 (objset, law, 배경, 물체) 묶음 단위로 반반 — arc 의 apex66 · 48 · 30 은 같은 포물선을 시간만 밀어 자른 것이라 한 묶음.
셀 (objset × law × cell) 18 개 모두 train 6 · test 6 block. 짝인 gravity test 는 `data_csv/vobench_v1_gravity_split/index_test.csv` (108 block). gravity file_name 앞에 `gravity/` 를 붙여 한 frames_root 로 읽는다.

type: frames_index
root: /data/hyuntak/project/2026/2027_cvpr/vjepa2/data_csv/vobench_v1_split_v2g
index_csv: index_train.csv
frames_root: /data2/local_datasets/world/Benchmark/VOBench_v1
frames_pattern: "{file_name}/{frame:06d}.png"
frames_start: 0
frames_stride: 3
frames_start_choices: [0, 1, 2]

## intphys2_main_possible_nfs

`intphys2_main_possible` 과 같은 506 clip (IntPhys 2 Main 가능 영상 전부) 을 **NFS 사본** (`/data/dataset/world/IntPhys2/Main/Videos`) 에서 읽는 판 (2026-10-15) — 노드 로컬 사본이 없는 노드 (vll6 등) 용.
용도: 학습 가능 여부 (학습한 Main 을 그대로 채점하는 상한) — `configs/training/intphys2_postft_all.yaml`.

type: video_csv
csv: /data/hyuntak/project/2026/2027_cvpr/vjepa2/data_csv/IntPhys2/main_possible_nfs.csv
frame_step: 10

## intphys2_pair_train

**IntPhys 2 Main 샘플 (쌍) 단위 split 의 train 쌍 가능 영상 253 clip** — 6 fps 프레임을 채점의 Resize + CenterCrop 으로 미리 뽑은 256 PNG (2026-10-15).
`z_training/data/build_intphys2_pair_split.py` 산출. 시작점 (0, 2, …, 46) 마다 `index_s{start}.csv` 가 있어 config 에서 `index_csv` · `frames_start` 를 덮어 쓴다 (`configs/training/intphys2_postft_pair.yaml`).

type: frames_index
root: /data/hyuntak/project/2026/2027_cvpr/vjepa2/data_csv/IntPhys2/pair_split
index_csv: index_s0.csv
frames_root: /data/hyuntak/project/2026/2027_cvpr/vjepa2/data_gen/IntPhys2_pair
frames_pattern: "{file_name}/{frame:06d}.png"
frames_start: 0
frames_stride: 1

---

# vll7 (6 x RTX PRO 6000 Blackwell) — 노드 로컬 학습 영상

⚠️ **vll7 에서만 돈다.** 경로가 이 노드 로컬 NVMe 다 (`/local_datasets/...`). 다른 노드의 `ssv2_min48` · `k400_train_40k`
csv 는 `/data2/...` 를 가리켜 여기서 안 맞는다. 하네스는 `z_vll7_training/`, 경로 정본은 `z_vll7_training/paths.env`.

인덱스 만드는 법 (gitignore — 아래 명령으로 다시 만든다):
```
python z_vll7_training/data/build_vll7_index.py --dataset k400_320      # 240,436 편 -> data_csv/vll7_k400_320/all.csv
python z_vll7_training/data/build_vll7_index.py --dataset ssv2          # 146,773 편 -> data_csv/vll7_ssv2/all.csv
python z_training/data/build_fps4_table.py --csv data_csv/vll7_k400_320/all.csv -o data_csv/vll7_k400_320/fps4_table.tsv --workers 48
python z_training/data/build_fps4_table.py --csv data_csv/vll7_ssv2/all.csv      -o data_csv/vll7_ssv2/fps4_table.tsv      --workers 48
python z_vll7_training/data/derive_fixed_csv.py --table data_csv/vll7_k400_320/fps4_table.tsv --mode fps4 --min 32 -o data_csv/vll7_k400_320/fps4_min32.csv
python z_vll7_training/data/derive_fixed_csv.py --table data_csv/vll7_ssv2/fps4_table.tsv      --mode raw  --min 32 -o data_csv/vll7_ssv2/fs1_min32.csv
```

**fps 4 로 창 W 장을 패딩 없이 채울 수 있는 편수** (2026-10-16 전수 실측, `fps4_table.tsv`):

| 창 W | 4 | 8 | 16 | 24 | 32 | 40 | 48 |
|---|---:|---:|---:|---:|---:|---:|---:|
| K400 320p (240,436) | 100 % | 99.8 % | 99.0 % | 97.1 % | 92.1 % | 87.7 % | **0** |
| SSv2 (146,773) | 100 % | 98.1 % | **50.1 %** | **1.9 %** | 7 편 | 6 편 | 1 편 |

⚠️ K400 은 10 초 클립이라 fps 4 에서 **최대 약 41 장**, SSv2 는 fps 4 에서 **W 16 을 넘으면 사라진다.**
그래서 (가) 긴 창은 K400 전용이 되어 **명목 가중치 != 실제 비율**이고 (나) 고정 32 장 창을 fps 4 로 잡으면 SSv2 가 7 편만 남는다.
vast 의 `natural_ctx_state` 가 **SSv2 만 `frame_step: 1` (12 fps)** 로 돌린 이유가 이것이다 — 창 텐서 모양은 같고 **시간 척도가 섞인다**
(같은 "K 블록 뒤" 가 K400 은 2~4 초, SSv2 는 0.7~1.3 초. RoPE 는 블록 번호만 안다).
실제 비율은 `python z_vll7_training/data/var_window_report.py` 로 칸마다 본다.

## vll7_k400_320_fps4_min32

K400 320p train 에서 **fps 4 로 32 장 이상** 뽑히는 221,361 / 240,436 편 (92.1 %). 블록 0.5 초, 32 장 = 8 초.
`n_frames: 32` 고정 창 팔용 (vast `natural_ctx_state` 의 `k710_320_fps4_min224` 자리).

type: video_csv
csv: /data/hyuntak/project/2026/2027_cvpr/vjepa2/data_csv/vll7_k400_320/fps4_min32.csv
fps: 4
filter_short_videos: true
decord_threads: 1

## vll7_ssv2_fs1_min32

SSv2 에서 **원본 32 프레임 이상** 126,001 / 146,773 편 (85.8 %). `frame_step: 1` = 12 fps, 블록 0.17 초, 32 장 = 2.7 초.
⚠️ 위 K400 (fps 4) 과 **시간 척도가 섞인다** — 의도된 선택이다 (fps 4 로는 SSv2 가 7 편밖에 안 남는다). 섞지 않으려면 둘 다 fps 4 + 창 16 으로.

type: video_csv
csv: /data/hyuntak/project/2026/2027_cvpr/vjepa2/data_csv/vll7_ssv2/fs1_min32.csv
frame_step: 1
filter_short_videos: true
decord_threads: 1

## vll7_k400_320_fps4_min16

위와 같되 **fps 4 로 16 장 이상** 237,925 편 (99.0 %). 16 장 = 4 초. 짧은 고정 창 (n_frames 16) 팔용.

type: video_csv
csv: /data/hyuntak/project/2026/2027_cvpr/vjepa2/data_csv/vll7_k400_320/fps4_min16.csv
fps: 4
filter_short_videos: true
decord_threads: 1

## vll7_ssv2_fps4_min16

SSv2 를 **fps 4** 로 (블록 0.5 초 = K400 과 같은 시간 척도). 16 장 = 4 초를 채우는 편은 73,544 / 146,773 (50.1 %).
시간 척도를 맞추고 싶을 때 `vll7_ssv2_fs1_min32` 대신 쓴다 — 대신 편수가 절반이다.
⚠️ csv 는 `derive_fixed_csv.py --mode fps4 --min 16` 으로 만든다 (아직 안 만들었으면 resolve 가 죽는다).

type: video_csv
csv: /data/hyuntak/project/2026/2027_cvpr/vjepa2/data_csv/vll7_ssv2/fps4_min16.csv
fps: 4
filter_short_videos: true
decord_threads: 1

## phys2d_v1_future

**phys2d_v1 의 "사건이 예측 구간에만 들어가는" 창** (2026-10-18 사용자 지시로 만든 세트).
원본 `/local_datasets/world/training_data/phys2d_v1` — 2D 물리 sandbox 28,900 clip (train),
256×256 · **4 fps** · 64~128 프레임, 시나리오 17 종 × 카테고리 3 (solidity 6 · permanence 5 · gravity 6, 각 1,700 clip).

창 규칙 (`z_training/data/build_phys2d_window_index.py`): T=32 · C=16 · stride 1 에서
**문맥 [start, start+16) 안에 clip 의 어떤 event_frame 도 없고**, 예측 [start+16, start+32) 안에 하나 이상 있다.
clip 당 최대 4 창을 고르게 솎아 **98,298 창** (clip 25,112 / 28,900 = 86.9 % 가 창을 가짐; 카테고리 solidity 34,736 · gravity 33,325 · permanence 30,237).

⚠️ 세트의 `boundary.tsv:boundary_frame` 은 **쓰지 않았다** — 실측하면 `event_frame + 1~3` 이라
그대로 자르면 사건이 문맥 끝 1~3 프레임에 들어온다 (지시의 반대). 대신 `event_frame` 을 썼고,
그 컬럼이 실제 운동 불연속인지 따로 쟀다 (단일 사건 clip 60 개, 가속도 백분위 중앙값 0.92 vs 무작위 대조군 0.74).
⚠️ T·C·stride 를 바꾸면 **인덱스를 다시 만들어야 한다** (행의 `n_frames` 가 `data.n_frames` 와 다르면 로더가 죽는다).

type: video_window
csv: /data/hyuntak/project/2026/2027_cvpr/vjepa2/data_csv/phys2d_v1/train_T32C16s1.csv
decord_threads: 1

## phys2d_v1_future_val

위의 val 분할 (clip 1,020 → **1,774 창**, clip 당 최대 2). 학습에 쓰지 않는다.

type: video_window
csv: /data/hyuntak/project/2026/2027_cvpr/vjepa2/data_csv/phys2d_v1/val_T32C16s1.csv
decord_threads: 1

## phys2d_v1_clips

**phys2d_v1 clip 전체, 창 제약 없음** (28,900 clip). 릴리즈 사전학습과 같은 tubelet(block3d) 마스크로
post-FT 할 때 쓴다 — 그 목적함수는 "사건을 예측 구간에" 라는 제약이 없다 (시공간 블록을 무작위로 가린다).
원본이 **4 fps** 라 `frame_step: 1` 이 사전학습의 `fps: 4` 와 같다. 영상이 64 프레임 이상이라 16 프레임 창은 항상 든다.

type: video_csv
csv: /data/hyuntak/project/2026/2027_cvpr/vjepa2/data_csv/phys2d_v1/train_clips.csv
frame_step: 1
decord_threads: 1

## phys2d_sup_v1_future

**phys2d_sup_v1 — 2D 물리 sandbox + 프레임별 상태 라벨** (2026-10-18, vll5 생성). 원본 `/local_datasets/world/training_data/phys2d_sup_v1`
(→ `/data2/...` 심볼릭 링크). train 19,953 clip · 시나리오 20 종 (gravity 7 · permanence 6 · solidity 6 · mixed 1), 256×256 · **4 fps** (렌더 1 장 = 시뮬 1/12 초) · 64~128 프레임.
vast `phys2d_v2` 와 같은 영상 관례에 **물체마다 장마다 위치 · 속도 · 가속도 · 각도 · 보임 · 정체 (모양 6 × 색 8) 라벨** (`<clip>_states.npz`, 읽기 `z_training/data/phys2d_sup_labels.py`).
문서 `z_training/docs/PHYS2D_SUP_DATASET_2026-10-18.md`.

창 규칙은 `phys2d_v1_future` 와 같다 (`build_phys2d_window_index.py --root <위> --out data_csv/phys2d_sup_v1/...`): T=32 · C=16 · stride 1,
문맥에 어떤 event_frame 도 없고 예측에 하나 이상. clip 당 최대 4 창 → **67,334 창** (clip 17,311 / 19,953 = 86.8 %).
⚠️ 이 세트의 사건에는 `reappear` (다시 보이기 시작한 장) 가 더 있다 — 같은 규칙이라도 phys2d_v1 과 창 분포가 다르다.

type: video_window
csv: /data/hyuntak/project/2026/2027_cvpr/vjepa2/data_csv/phys2d_sup_v1/train_T32C16s1.csv
decord_threads: 1

## phys2d_sup_v1_future_val

위의 val 분할 (clip 996 → **1,701 창**, clip 당 최대 2, seed 2 — train 과 겹침 0). 학습에 쓰지 않는다.

type: video_window
csv: /data/hyuntak/project/2026/2027_cvpr/vjepa2/data_csv/phys2d_sup_v1/val_T32C16s1.csv
decord_threads: 1

## phys2d_sup_v1_clips

**phys2d_sup_v1 clip 전체, 창 제약 없음** (19,953 clip). 4 fps 원본이라 `frame_step: 1`.

type: video_csv
csv: /data/hyuntak/project/2026/2027_cvpr/vjepa2/data_csv/phys2d_sup_v1/train_clips.csv
frame_step: 1
decord_threads: 1

## phys2d_sup_v1_state

**phys2d_sup_v1 창 + 상태 라벨** (2026-10-19) — `phys2d_sup_v1_future` 와 같은 창 (67,334, T32 C16) 을 `type: phys2d_sup` 로 읽어
배치마다 정체 48 칸 라벨 (존재 · 위치 · 속도, 블록마다) 을 같이 낸다 (`app/vjepa_frozen/data.py` Phys2DSupDataset → `state_head`).
좌우 반전은 영상 · 라벨에 같이 걸린다. `random_resized_crop` 은 금지 (resolve_train 이 죽인다). 다른 데이터셋과 섞지 못한다.

type: phys2d_sup
csv: /data/hyuntak/project/2026/2027_cvpr/vjepa2/data_csv/phys2d_sup_v1/train_T32C16s1.csv
decord_threads: 1

## phys2d_sup_v1_state_clips

위와 같되 **clip 전체에서 연속 n_frames 장을 무작위 시작으로** (19,953 clip). 사건이 문맥 안에 들어가는 창도 나온다.

type: phys2d_sup
csv: /data/hyuntak/project/2026/2027_cvpr/vjepa2/data_csv/phys2d_sup_v1/train_clips.csv
decord_threads: 1
