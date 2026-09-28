# 학습 데이터 레지스트리 (predictor 학습용)

> **2026-09-22 (기계 이전): 이 기계(`ariel-k2`)에 실물이 있는 학습 데이터는 `ssv2` · `k400` 둘이다.**
> 나머지 섹션(v11 · rollout · predictor_v1 · intphys1_train · intphys2_*)은 **프레임/인덱스가 아직 없다** —
> `${WORLD_ROOT}/world_analysis/...` 와 `${DATA_CSV}/...` 가 비어 있어서 `train.sh` 가 실물 검사에서 죽는다.
> 설계 기록으로 남겨 둔 것이고, 쓰려면 데이터를 `${TRAIN_DATA_ROOT}` 로 옮기고 인덱스를 다시 만들어야 한다.
> 경로 정본은 `z_research/scripts/harness/paths.env`. 값 안의 `${VAR}` 는 resolve 단계에서 풀린다.
>
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

## ssv2

**Something-Something v2** (220,847 편, 31 GB, 12 fps · 높이 320). **이 기계에 실물이 있는 유일한 학습 데이터다.**
라벨은 안 쓴다 (frozen-encoder JEPA 손실은 라벨 없음) — csv 의 둘째 칸은 전부 0 이다.
인덱스: `python z_training/data/build_ssv2_index.py --write` (train 218,639 / val 2,208, seed 0).

**왜 후보인가** — v11 류 합성 세트는 속도가 1 종·궤적이 고정이라 post-FT 가 *궤적 prior* 를 외운다
(`z_research/predictor_training/predictor_IntPhysGenV11_PFT/Archive/RESULTS_2026-09-16.md` §3 정정,
RollOut β≈2). "속도가 다양한 데이터로 학습해 β→1 이 나오는가" 가 다음 검정이고, 실사 영상은
속도·동역학 분포가 넓다는 점에서 그 축에 맞는다.

⚠️ **미검증 항목** (돌리기 전에 정할 것):
- `frame_step: 1` 은 12 fps 원본에서 **32 장 = 2.7 초**다. v11 채점(stride 3 @ 30 fps ≈ 3.2 초) 과 비슷하게
  맞춘 값이지 재 본 값이 아니다. `n_frames` 를 바꾸면 여기도 같이 본다.
- **길이 >= 32 인 영상이 85.5 %** 다 (400 편 표본). 나머지는 마지막 프레임 반복으로 패딩돼
  **미래가 정지 화면**이 되므로 `filter_short_videos: true` 로 버린다.
  아예 인덱스에서 빼려면 `build_ssv2_index.py --write --probe --min-frames 32` (220k 편 probe, 느리다).
- val 분할은 **공식 SSv2 val 이 아니라** seed 0 무작위 1 % 다. loss 감시용으로만 읽을 것.
- SSv2 는 손 조작 영상이라 **카메라가 흔들리고 장면이 가득 차 있다.** v11/RollOut 위치 자(readout)를
  그대로 걸 수 없다 — 전이 검정은 RollOut_v2 / IntPhys1 처럼 **자가 이미 검증된 세트**에서 한다.

type: video_csv
csv: ${DATA_CSV}/ssv2/train.csv
fps: 12
filter_short_videos: true

## k400

**Kinetics-400** (약 138,000 clip, ~150 GB). `${TRAIN_DATA_ROOT}/K400/videos/<클래스>/<영상>/<clip>.mp4`.
인덱스: `python z_training/data/build_video_index.py k400 --write`.

**왜 SSv2 와 같이 쓰나** — SSv2 는 **최대 76장(6.3초)** 이라 긴 창을 못 준다 (실측 2026-09-22).
K400 은 median 112장 (p95 300) 이라 창·해상도를 넓게 고를 수 있다. 둘을 섞으면
**블록당 시간** 과 **장면 종류** 가 동시에 다양해진다.

⚠️ **`aug.square_crop: center` 또는 `random_resized_crop` 을 반드시 켠다.** K400 은 해상도·화면비가
   섞여 있다 (720×1280 / 360×480 / 360×640 …). `ClipTransform` 은 (r,r) 로 바로 리사이즈하므로
   그냥 두면 **16:9 가 1:1 로 눌려 가로 속도가 1.78배 압축**된다 — "스텝당 이동 거리" 를 재는
   이 프로젝트에서는 치명적이다.

⚠️ `fps` 로 주면 원본 fps 가 섞여 있어도 (30 주류, 25·24 섞임) `VideoDataset` 이
   `fstp = 원본fps // fps` 로 맞춘다. **`frame_step` 대신 `fps` 를 쓸 것.**
   프레임 예산 = `n_frames × fstp` 이고, 짧은 영상은 `filter_short_videos` 가 버린다:

| fps | fstp(30fps) | n_frames 32 필요 | 사용률 | 블록당 |
|---:|---:|---:|---:|---:|
| 12 | 2 | 64 | 96.2% | 0.17s |
| 6 | 5 | 160 | 31.3% | **0.33s** (IntPhys2 와 동일) |
| 4 | 7 | 224 | ~15% | 0.50s (릴리즈 사전학습과 동일) |

type: video_csv
csv: ${DATA_CSV}/k400/train.csv
fps: 12
filter_short_videos: true

## k400_val

위 `k400` 의 val 분할. in-loop 감시용이고 보고 지표가 아니다.

type: video_csv
csv: ${DATA_CSV}/k400/val.csv
fps: 12
filter_short_videos: true

## ssv2_val

위 `ssv2` 의 val 분할 (2,208 편). 같은 규약. in-loop 감시용이고 보고 지표가 아니다.

type: video_csv
csv: ${DATA_CSV}/ssv2/val.csv
fps: 12
filter_short_videos: true

## ssv2_min48

`ssv2` 에서 **48 프레임 이상만** 남긴 것 — 97,416 train / 984 val (44.6 %).
`n_frames: 48` × `fps: 12` (fstp 1) 용. 인덱스: `build_video_index.py ssv2 --probe --min-frames 48 --write`.

⚠️ **왜 미리 거르나** — `filter_short_videos` 는 짧은 영상을 만나면 **무작위 인덱스로 재추첨**한다.
통과율이 낮으면 성공 1 건당 영상을 여러 번 열게 되고, rank 하나가 배치를 못 채우면
**DDP 전체가 멈춘다** (2026-09-22 실측: skipping 1,368 회 / step 0 회, GPU 7 장 100 % 공회전).
`train_min<N>.csv` 를 쓰면 런타임 거부가 **0** 이다. N = config 의 `n_frames × fstp`.

type: video_csv
csv: ${DATA_CSV}/ssv2/train_min48.csv
fps: 12
filter_short_videos: true

## k400_min96

`k400` 에서 96 프레임 이상 — 88,930 train / 898 val (64.4 %). `n_frames: 48` × `fps: 12` (fstp 2) 용.

type: video_csv
csv: ${DATA_CSV}/k400/train_min96.csv
fps: 12
filter_short_videos: true

## k400_320_min96

**`k400_min96` 를 짧은 변 320 으로 재인코딩한 사본** (`${TRAIN_DATA_ROOT}/K400_320/videos`, 원본 불변).
`z_research/scripts/data/resize_videos.sh` (2026-09-22): 비율 유지, `-g 24`(키프레임 1초), 오디오 제거.
csv 는 `z_training/data/finalize_k400_320.sh` 가 원본 csv 의 경로만 바꿔 만든다 (실재 파일만).

**왜** — 720p 디코드가 clip당 0.6~1 s 라 8 GPU step 의 40 % 가 데이터 대기였다 (bench_g 실측
iter 18~44 s 중 data 5~40 s). 320p + keyint 24 로 디코드 5~8배 ↓. 재인코딩되면 **헤더 길이 = 실제
길이** 라 "헤더만 긴 서브클립에서 decord 무한 spin → rank 정지" 클래스도 사라진다.
⚠️ 이 K400 사본은 clip 이 최대 10 초라 **`fps 6 × 48 장`(8 초 창) 팔은 못 만든다** (2.8~10 초 서브클립). 6 fps 규격은 SSv2 로만.

type: video_csv
csv: ${DATA_CSV}/k400_320/train_min96.csv
fps: 12
filter_short_videos: true

## k400_min240

`k400` 에서 240 프레임 이상 — 21,645 train / 218 val (15.7 %). `n_frames: 48` × `fps: 6` (fstp 5) 용.
**블록당 0.33 s · 창 8.0 s 로 IntPhys 2 채점 규격과 정확히 같다.** 표본이 적으니 weight 로 조절한다.

type: video_csv
csv: ${DATA_CSV}/k400/train_min240.csv
fps: 6
filter_short_videos: true

## intphys1_train

IntPhys 2019 **train 분할** (가능 영상만, 4중항 없음). 로컬에 3,750 scene
(`${BENCH_ROOT}/IntPhys1/<id>/scene/scene_001..100.png`, 288×288). 원 train 은 15,000 인데
그중 3,750 만 받아 뒀다 (id 가 띄엄띄엄). 인덱스는 `z_training/data/build_intphys1_train_index.py`.
**IntPhys1 dev (intphys1_dev) 와 분할이 다르므로 dev 채점이 오염되지 않는다.** v11 과도 무관.
파일 번호가 1부터라 start 1..7 이 전부 예산(100) 안이다 (7 + 31×3 = 100).

**sliding 채점(`intphys1_sliding`, 보고 지표 `skip2_w32/avg`)에 맞춘 값이다**: raw 프레임 2칸 간격 32장 창을 영상
어디서든(시작 1~38) 자르고, 문맥 길이는 config 의 `mask.context_frames: [4, 8, 12, 16, 20]` (채점의 C 집합) 으로 준다.
v11 류(stride 3, 문맥 16 고정) 와 다르니 섞어 쓸 때 주의. 인덱스: `python z_training/data/build_intphys1_train_index.py`.

type: frames_index
root: ${DATA_CSV}/intphys1_train
index_csv: index.csv
frames_root: ${BENCH_ROOT}/IntPhys1
frames_pattern: "{file_name}/scene/scene_{frame:03d}.png"
frames_start: 1
frames_stride: 2
frames_start_choices: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38]
raw_frames: 100

## rollout_v2_training

RollOut_v2 의 학습셋 (무중력 직선 운동, 896 clip, 전부 가능). `configs/protocols/datasets.md ## rollout_v2_training`
과 같은 프레임. 저장 프레임이 0,3,...,99 (34장) 라 start 는 0/3/6 만 가능하다.

type: frames_index
root: ${DATA_CSV}/rollout_v2_training
index_csv: index.csv
frames_root: ${WORLD_ROOT}/world_analysis/RollOut_v2_training
frames_pattern: "{file_name}/{frame:06d}.png"
frames_start: 0
frames_stride: 3
frames_start_choices: [0, 3, 6]
raw_frames: 100

## rollout_v2_possible

RollOut_v2 (운동 법칙 7종, 5,488 clip) 의 **가능 변이만**. ledge/wall 의 불가능 짝은 include 필터로 빠진다.
`rollout_v2` 위치 readout 결과와 비교할 때 이걸로 학습하면 그 세트가 오염된다 — readout 은 held-out 이 필요하다.

type: frames_index
root: ${DATA_CSV}/rollout_v2
index_csv: index.csv
frames_root: ${WORLD_ROOT}/world_analysis/RollOut_v2
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
root: ${DATA_CSV}/intphysgen_v11
index_csv: index.csv
frames_root: ${WORLD_ROOT}/world_analysis/IntPhysGen_v11
frames_pattern: "{file_name}/{frame:06d}.png"
frames_start: 0
frames_stride: 3
frames_start_choices: [0, 3, 6]
raw_frames: 100

## intphys2_train_mp4

IntPhys 2 Main 의 mp4 (808 clip, 2way train 분할, 가능·불가능 섞임). **미검증** — `frame_step` 은 추정값이고
라벨 컬럼(가능/불가능)을 VideoDataset 이 필터하지 않는다. 형식 예시로만 둔다.

type: video_csv
csv: ${DATA_CSV}/IntPhys2/IntPhys2_2way_train.csv
frame_step: 3

## intphys2_main_possible

**IntPhys 2 Main 의 가능 영상 506 개** (253 장면 × {1_Possible, 2_Possible}), mp4 직독 (2026-09-19 결정: 학습 = Main 가능, test = HeldOut 리더보드).
`z_training/data/build_intphys2_main_index.py --check --write` 산출. 512×512 · 60 fps, 길이 635–938 (전수 확인). `frame_step: 10` = 채점과 같은 6 fps.
⚠️ 논문 A.3 이 Main 메타데이터로 학습하지 말라고 적었다 — **학습 후 Main 점수는 논문 Table 2 와 비교 불가.** 유효한 test 는 HeldOut 뿐.
⚠️ `window_grid` 는 frames_index 전용이라 여기서는 못 쓴다 — 창 분포는 `n_frames` + 무작위 시작 + `mask.context_frames` 목록으로 맞춘다.

type: video_csv
csv: ${DATA_CSV}/IntPhys2/main_possible.csv
frame_step: 10

## intphys2_main_split_train

**IntPhys 2 Main 장면 split 의 train 절반, 가능 영상만** — 254 clip / 127 장면 (2026-09-19 결정: test 도 Main → 장면 단위 반반).
`z_training/data/build_intphys2_main_split.py --write --link` 산출 (seed 0, condition × Difficulty × Camera 층화). test = 126 장면 504 영상 (252 쌍),
채점기 `split: MainTest` (`${BENCH_ROOT}/IntPhys2/MainTest/metadata.csv`, `Videos -> ../Main/Videos`).
⚠️ Main 은 논문이 학습 금지한 세트 — test 수치는 같은 test 장면의 릴리즈 predictor 값과만 비교한다.

type: video_csv
csv: ${DATA_CSV}/IntPhys2/main_split/train_possible.csv
frame_step: 10

## v11_split_train

**IntPhysGen v11 (12조건) 의 block 단위 train 절반, 가능 변이만** — 10,752 clip / 5,376 block.
`z_training/data/build_v11_split_index.py` 산출 (`data_csv/intphysgen_v11_split/`, seed 0, 50/50,
(condition × violation_type × sym_k) 117 셀 층화). 짝인 test 절반은 `configs/protocols/datasets.md ## v11_split_test`.
⚠️ **v11 / v11_full 전체 채점과는 여전히 문맥을 공유한다** — 학습 후 점수는 `v11_split_test` 로만 읽는다.

type: frames_index
root: ${DATA_CSV}/intphysgen_v11_split
index_csv: index_train.csv
frames_root: ${WORLD_ROOT}/world_analysis/IntPhysGen_v11
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
root: ${DATA_CSV}/predictor_v1_training
index_csv: index_train.csv
frames_root: ${WORLD_ROOT}/world_analysis/Predictor_v1_training
frames_pattern: "{file_name}/{frame:06d}.png"
frames_start: 0
frames_stride: 3
frames_start_choices: [0, 3, 6]
raw_frames: 100
