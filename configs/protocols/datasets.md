# 데이터셋 레지스트리

`z_research/scripts/run.sh <프로토콜> <데이터셋> [모델]` 의 두 번째 인자.
`## 이름` 아래의 `key: value` 만 읽는다 (그 밖의 줄은 전부 설명으로 무시된다).

## 규칙

- 여기 값은 최종 config 의 `data:` 블록으로 들어간다. **프로토콜 yaml 의 `data:` 가 이긴다** —
  프로토콜이 소유한 키(`n_frames`, `resolution`, probing 의 `index_csv`/`type_column` 등)는
  여기 뭘 적든 프로토콜 값이 쓰인다.
- 메타 키 3개는 `data:` 로 안 들어간다:
  - `raw_frames`  원본 프레임 수. 프로토콜이 `n_frames: RAW` 로 적으면 이 값으로 치환된다
  - `cache_tag`   토큰 캐시 이름의 앞부분. 최종 `tag` = `<cache_tag>_<모델>`
  - `results_root` 결과가 쌓일 자리. 최종 `output_dir` = `<results_root>/<프로토콜>__<데이터셋>_<모델>`.
    없으면 `z_exp/world_model_analysis/results` 로 간다. 데이터셋별 아카이브(`z_research/<셋>/exp_results`)를
    쓰려고 둔 키다 — ⚠️ `*.json`/`*.png` 는 `.gitignore` 대상이라 결과 자체는 추적되지 않는다
  - `available`   `false` 면 실행을 막고 이유를 찍는다
- `frames_pattern` 의 `{...}` 는 index.csv 의 **컬럼 이름**이다 (`{frame}` 만 예외 — 프레임 번호).
- **프레임은 mp4 가 아니라 원본 PNG 직독이 원칙**이다 (코덱 손실·yuv420 크로마 서브샘플링 차단).
  `frames_root` 가 없는 데이터셋만 `file` 컬럼의 영상을 직접 읽는다.

## 새 데이터셋 추가하기

아래 형식으로 `## 이름` 섹션 하나만 더 쓰면 된다. yaml 은 건드리지 않는다.
`run.sh` 가 실행 전에 `root` / `index_csv` / `frames_root` 실물과
`resolution == model.img_size` 를 검사하고, 안 맞으면 모델을 로드하기 전에 죽는다.

---

## intphys1_dev

외부 벤치마크. 100프레임 360영상 = 90 4중항. `pair_id` 는 build_intphys1_pairs.py 산출.
`type_column` 은 O1(Object Permanence) / O2(Shape Constancy) / O3(Continuity).
⚠️ 원본은 288x288 이라 256 으로 리사이즈된다 (antialias=False bilinear — 공식과 같은 커널).

raw_frames: 100
cache_tag: intphys1_dev
results_root: /data/hyuntak/project/2026/2027_cvpr/vjepa2/z_research/IntPhys/exp_results
root: /local_datasets/world/world_analysis/IntPhys1_dev_videos
index_csv: index.csv
frames_root: /local_datasets/world/world_analysis/IntPhys1_dev_frame_png
frames_pattern: "{block}/{quadruplet}/{run}/scene/scene_{frame:03d}.png"
frames_start: 1
frames_stride: 3
block_column: block_id
pair_column: pair_id
variant_column: variant
plausible_column: plausible
type_column: block_type

## v8

정식 IntPhysGen. 2048영상 / 512 block. `condition` 4분할 균등(512씩),
`violation_type` = vanish 1024 / color 512 / shape 512.
가림은 4+4 대칭 (fully hidden raw 36~57 = context 끝 4장 + future 앞 4장).
기존 41GB 토큰 캐시는 `cache/v8_vith -> attn_probe_v8_vith` 심볼릭 링크로 그대로 재사용된다.

raw_frames: 100
cache_tag: v8
results_root: /data/hyuntak/project/2026/2027_cvpr/vjepa2/z_research/IntPhysGenV8/exp_results
root: /data/hyuntak/project/2026/2027_cvpr/vjepa2/data_csv/intphysgen_v8
index_csv: index.csv
frames_root: /data/hyuntak/project/2026/2027_cvpr/UnrealEngine/IntPhysGen_v8
frames_pattern: "{file_name}/{frame:06d}.png"
frames_start: 0
frames_stride: 3
block_column: block_id
pair_column: pair_id
variant_column: variant
plausible_column: plausible
type_column: condition

## v8_halfsize

v8 과 같은 장면을 물체 겉보기 크기 28.1px 로 줄여 렌더한 것. 크기 효과 통제용. fixed 74.71%.

raw_frames: 100
cache_tag: v8_halfsize
results_root: /data/hyuntak/project/2026/2027_cvpr/vjepa2/z_research/IntPhysGenV8/exp_results
root: /data/hyuntak/project/2026/2027_cvpr/vjepa2/data_csv/intphysgen_v8_halfsize
index_csv: index.csv
frames_root: /data/hyuntak/project/2026/2027_cvpr/UnrealEngine/IntPhysGen_v8_halfsize
frames_pattern: "{file_name}/{frame:06d}.png"
frames_start: 0
frames_stride: 3
block_column: block_id
pair_column: pair_id
variant_column: variant
plausible_column: plausible
type_column: condition

## v11

**본 실험 세트.** 21,504 clip / 5,376 block / 10,752 matched pair.
`시나리오 6 x 위반 3 x k 4 x 84 block` — **k 를 데이터셋 내부 축으로** 가져간 것이 v10 과의
가장 큰 차이다 (v10 은 k=4 고정, `v10_occ_low` 가 k=1 을 별도 셋으로 뒀다).

| 축 | 값 |
|---|---|
| `condition` | `static/moving_flat/moving` x `visible/occlusion` — **6개, k 를 접지 않았다** |
| `sym_k` | 0(가림 없음) / 1 / 2 / 3 / 4 — 가려지는 **샘플 프레임 수**(한쪽당) |
| `violation_type` | vanish / shape / color |
| shape | 7종, **21쌍 전부** 커버 (v10 은 8종 중 4쌍뿐이었다) |
| color | 8종, **28쌍 전부** |

⚠️ **`condition` 에 k 를 접어 넣지 않은 것이 중요하다.** `attn_probe` 의 `fit_groups_sweep: auto`
가 `group_column` 값을 읽으므로, k 를 접었다면 24 그룹 x 3 target x 3 run = 216 head 가 됐다.
지금은 7 x 3 x 3 = 63 이다.

**속도 고정, 가림막 폭이 k 와 함께 변한다** (v0 116 cm/s 고정, 폭 17.5% -> 36.9%).
"가림 시간"을 늘리려면 고정 속도에서는 폭을 늘릴 수밖에 없다 — 조작의 구현이지 교란이 아니다.

**silhouette 을 7종 전부 동일하게 맞췄다** (렌더된 픽셀 기준, `shape_scale_fix`). v10 에서는
모양이 폭과 교락돼 clean frame 수로 샜다 (cone 5 vs cylinder 8). 보정은 메시에만 걸고
`EXTENTS`(가림막을 푸는 기준)에는 걸지 않는다 — 그러면 모양마다 다른 가림막이 생겨
가시성 단서를 되돌려준다.

⚠️ `visible` 은 가림막이 장면에 없다. "가림 유무" 대비의 정의는 CLAUDE.md §8-5 를 볼 것.
프레임은 34장이 저장돼 있고 프로토콜은 32장(0,3,..,93)만 읽는다. 96·99 는 창 밖이다.

설계 문서: `/data/hyuntak/project/2026/2027_cvpr/UnrealEngine/gen/V11_DESIGN.md`

raw_frames: 100
cache_tag: v11
results_root: /data/hyuntak/project/2026/2027_cvpr/vjepa2/z_research/IntPhysGenV11/exp_results
root: /data/hyuntak/project/2026/2027_cvpr/vjepa2/data_csv/intphysgen_v11
index_csv: index.csv
frames_root: /local_datasets/world/world_analysis/IntPhysGen_v11
frames_pattern: "{file_name}/{frame:06d}.png"
frames_start: 0
frames_stride: 3
block_column: block_id
pair_column: pair_id
variant_column: variant
plausible_column: plausible
type_column: condition

## v11_earlymid

**v11 의 가림 타이밍 팔.** 21,504 clip / 5,376 block / 10,752 matched pair.
2026-09-01 에 v11 본체(`IntPhysGen_v11`)에 **추가로 렌더된** 6개 조건만 담는다.
기존 6조건은 프레임이 그대로라 **다시 채점하지 않는다** — `v11` 결과를 그대로 쓴다.

`k` 장을 문맥 **어디에** 두느냐만 바꾼다 (문맥 = raw 0–45, 미래 = 48–93):

| timing | 가리는 raw 프레임 (k=4) | 문맥 숨김 | 미래 숨김 | 총 |
|---|---|---:|---:|---:|
| `early` | 12–21 | k | 0 | k |
| `mid` | 21–30 | k | 0 | k |
| `late` (= **v11 본체**, 별도 셋 아님) | 36–57 | k | **k** | **2k** |

⚠️ **`late` 는 이 셋에 없다.** v11 의 `static_occlusion` 등이 그대로 `late` 다.
⚠️ **`early`/`mid` vs `late` 는 "문맥 안 위치" 말고 두 가지가 더 바뀐다** —
   미래 가림 유무, 총 가림량 2배. `early` vs `mid` 만 순수한 위치 비교다.
   총량을 맞추려면 `early k=2` ↔ `late k=1`, `early k=4` ↔ `late k=2` 로 짝지어 본다.
⚠️ **타이밍이 `condition` 에 접혀 있다** (`*_early`, `*_mid`).
   `v11` 과 합치면 12 조건이 되어 `fit_groups_sweep: auto` 가 12 그룹을 만든다 — probing 은 쪼갤 것.

문맥 무결성 전수 감사 **172,032쌍, mismatch 0**
(`data_csv/intphysgen_v11_earlymid/context_integrity.json`).

raw_frames: 100
cache_tag: v11_earlymid
results_root: /data/hyuntak/project/2026/2027_cvpr/vjepa2/z_research/IntPhysGenV11/exp_results
root: /data/hyuntak/project/2026/2027_cvpr/vjepa2/data_csv/intphysgen_v11_earlymid
index_csv: index.csv
frames_root: /local_datasets/world/world_analysis/IntPhysGen_v11
frames_pattern: "{file_name}/{frame:06d}.png"
frames_start: 0
frames_stride: 3
block_column: block_id
pair_column: pair_id
variant_column: variant
plausible_column: plausible
type_column: condition

## v13_black

**가림막이 검정인 v11.** 4,608 clip / 1,152 block / 2,304 matched pair.
조건 구성은 `v11_full` 과 같다 (12 조건 = visible 3 + early/mid/late 각 3), 규모만 1/9 이다.

**왜 만들었나** — v11 은 가림막이 전부 `wood` 한 색이라 "가림막 자체가 색면으로 들어와
색 채점을 죽이는가" 를 **검정할 수 없었다** (변이가 0). v13 은 `occ_color: black` 이라
같은 설계에서 가림막 색만 바꾼 대조군이 된다.

| | v11_full | v13_black |
|---|---|---|
| 가림막 색 | `wood` (3,456 clip 전부) | **`black`** |
| 규모 | 43,008 clip | 4,608 clip |
| 조건 | 12 | 12 (같음) |

⚠️ **규모가 1/9 이다.** 조건×위반×k 로 쪼개면 셀당 16쌍이라 v11 처럼 잘게 못 쪼갠다.
   타이밍×위반 정도까지만 읽고, k 별은 선 모양만 볼 것.
⚠️ 캐시 태그가 다르므로 probing 을 돌리려면 **새로 추출**해야 한다 (~90G).

문맥 무결성 전수 감사 **36,864쌍, mismatch 0**
(`data_csv/intphysgen_v13_black/context_integrity.json`).

raw_frames: 100
cache_tag: v13_black
results_root: /data/hyuntak/project/2026/2027_cvpr/vjepa2/z_research/IntPhysGenV13_Black_occluder/exp_results
root: /data/hyuntak/project/2026/2027_cvpr/vjepa2/data_csv/intphysgen_v13_black
index_csv: index.csv
frames_root: /local_datasets/world/world_analysis/IntPhysGenV13_occluder_black
frames_pattern: "{file_name}/{frame:06d}.png"
frames_start: 0
frames_stride: 3
block_column: block_id
pair_column: pair_id
variant_column: variant
plausible_column: plausible
type_column: condition

## rollout_v1

**등속 직선 운동만 있는 세트.** 7,392 clip / 7,392 block. **위반도 쌍도 없다** (클립 하나가 block 하나,
`is_possible` 전부 1, `violation_type` 전부 none). 물리는 한 가지 법칙뿐이다 — 평지 등속.

| 축 | 값 |
|---|---|
| 속도 | 11 레벨 (부호 포함): -160 -135 -110 -85 -60 **0** 60 85 110 135 160 (cm/s) |
| anchor | -100 / 0 / +100 (cm) = **문맥 마지막 프레임(45)에서의 위치** |
| condition | `roll_v{속도}_a{anchor}` — **33종** (11 x 3), 셀당 정확히 224 |
| shape | **7종** (v11 과 같음, `narrowcap` 없음) · color 8종 · env 4종 |

**왜 만들었나** — `PAPER_STORY_2026-09-06.md` beat 4(a): *미래 토큰에서 물체 위치를 튜블릿별로
읽으면 진전하는가.* 미래 8칸에서 위치를 읽어 `x(t) = a + b*t` 를 적합하고 **`b` 가 실제 속도인지**
본다. 수평선이면 "조회기", 대각선이면 "이어간다".

**설계의 핵심은 속도와 anchor 의 직교다.** 같은 마지막 관측 위치에서 11개의 다른 미래가 나오므로
**"마지막 프레임 복사" 가설이 데이터 수준에서 차단된다.** `v=0` 은 별도 조건이 아니라 속도 레벨의
하나로 들어가 있어 회귀 안에 대조군이 내장된다.

**위치 라벨은 해석식이다** (렌더 메타데이터만 쓰고 픽셀 측정이 없다):

```
x_cm(raw f) = flat_anchor_x_cm + flat_v_cm_s * (f - 45) / 16
x_norm      = x_cm / 769.0909          # frame_half_width_cm, 정규화 이미지 좌표
```

`obj_depth_cm`(1410) · 카메라 · fov 가 전부 상수라 cm -> 정규화 좌표가 **선형**이다.
예측 끝 `x_norm` 범위는 [-0.754, +0.754] 로 물체가 항상 화면 안에 있다.
전수 검증: `python z_research/scripts/data/build_rollout_index.py` (라벨 공식이 metadata 를
`max|Δ| = 0` 으로 재현, 프레임 실물 표본 200클립 x 32장).

⚠️ **`surprise_c16t32` 는 못 돌린다** — matched pair 가 없다. 이 세트는 토큰 캐시 위의 회귀 전용이다.
⚠️ **`attn_probe` 는 캐시 추출용으로만 쓴다.** `condition` 이 33종이라 `fit_groups_sweep: auto` 를
   그대로 두면 33 x 3 target x 3 run = **297 head** 가 된다. `[null]` + `num_epochs=1` 로 접을 것
   (`runs` 는 절대 줄이지 말 것 — z/p/h base 3종이 다 필요하다, CLAUDE.md §7-1).
⚠️ **`shape` 이 7종**이라 `probing.targets.shape.classes` 를 덮어써야 죽지 않는다 (v11 과 같다).
⚠️ **원본이 288px** 이라 256 으로 리사이즈된다 (crop 아님, bilinear/antialias off — IntPhys1 과 같은 경로).
   v11 도 288 렌더라 같은 경로를 탄다. 정규화 좌표가 보존된다.
⚠️ **프레임과 캐시가 vll6 로컬(`/data2`)이다.** vll5 에서는 안 보인다. `-w vll6` 로 제출할 것.
   `/local_datasets/world/world_analysis/{cache,RollOut_v1}` 는 vll6 에서 `/data2` 로 가는 심볼릭이다
   (vll5 의 관례를 그대로 만들어 둬서 프로토콜 yaml 을 안 고쳐도 된다).
   토큰 캐시 **약 145 GiB** — 실측 클립당 20.97 MB = (2048 ctx + 4096 target + 2048 pred) x 1280 x 2byte.
   ⚠️ base 3종의 **토큰 수 합**이 8192 이지 각각이 8192 가 아니다 (`target` 만 32프레임 전부라 4096).

설계 문서: `<원본>/dataset.json` · `provenance.json`. 결과: `z_research/RollOutV1/`.

raw_frames: 100
cache_tag: rollout_v1
results_root: /data/hyuntak/project/2026/2027_cvpr/vjepa2/z_research/RollOutV1/exp_results
root: /data/hyuntak/project/2026/2027_cvpr/vjepa2/data_csv/rollout_v1
index_csv: index.csv
frames_root: /local_datasets/world/world_analysis/RollOut_v1
frames_pattern: "{file_name}/{frame:06d}.png"
frames_start: 0
frames_stride: 3
block_column: block_id
pair_column: pair_id
variant_column: variant
plausible_column: plausible
type_column: condition

## rollout_v2

**운동 법칙 7종, 위치 readout 전용.** 5,488 clip / 4,704 block. `flat_v`(등속) `flat_a`(등가속) `ramp_a`(경사)
`arc`(포물선) `fall`(자유낙하) `ledge`(선반 끝 낙하, **pos/imp 쌍**) `wall`(벽 정지, **pos/imp 쌍**).
시나리오마다 primary 7 × secondary 2 셀, 셀당 56 (ledge/wall 은 28 × 2 변이). `holdout` = primary 2·6번째 레벨.
가림 없음. 프레임·캐시 **vll5 로컬** (`/data2`).

⚠️ **metadata 의 `object_*_by_sample` 은 믿지 말 것.** 09-09 17:17 수정본도 wall 불가능 클립(통과)에 정지 궤적을 쓴다.
   라벨은 `build_rollout2_index.py` 가 plan(`UnrealEngine/gen/plans/blocks_rollout2.json`)에서 가져와 index 에 싣는다
   (`x_cm_by_sample` 등 32 샘플). 검증 14항목 (flat 에서 metadata 와 일치, 투영식 0.07 px, 예측 16장 in_frame).
⚠️ `surprise_c16t32` 는 ledge/wall 에서만 의미가 있고 pairing 이 v11 과 다르다 (block 당 pos 1 + imp 1). 이 세트는 캐시 회귀 전용.
⚠️ `attn_probe` 는 캐시 추출용. v1 과 같은 SET (`fit_groups_sweep=[null]`, `num_epochs=1`, shape 7종) 으로 돌린다.
   캐시 약 115 GiB (5,488 × 20.97 MB).

raw_frames: 100
cache_tag: rollout_v2
results_root: /data/hyuntak/project/2026/2027_cvpr/vjepa2/z_research/RollOutV2/exp_results
root: /data/hyuntak/project/2026/2027_cvpr/vjepa2/data_csv/rollout_v2
index_csv: index.csv
frames_root: /local_datasets/world/world_analysis/RollOut_v2
frames_pattern: "{file_name}/{frame:06d}.png"
frames_start: 0
frames_stride: 3
block_column: block_id
pair_column: pair_id
variant_column: variant
plausible_column: plausible
type_column: condition

## rollout_v2_training

**RollOut_v2 위치 readout 의 학습셋. 그것 말고는 용도가 없다.** 896 clip / 896 block (`line_x` 448 + `line_z` 448).
무중력 직선 운동 — x 축 또는 z 축 하나만 움직이고 아무것도 받치지 않는다 (물체는 떠 있다). primary 8 × secondary 8 셀,
7 shape × 8 color × 4 env 를 완전 교차. `holdout` 전부 0, 불가능 변이 없음. 가림 없음. 32 샘플 전부 in_frame.
readout 은 여기서 fit 하고 `rollout_v2` 에는 **test 로만** 건다 (`rollout_v2` 의 p 를 fitting 에 쓰지 않는다).
카메라·물체 크기·깊이·env·shape·color 는 `rollout_v2` 와 동일 (2026-09-10 실측). 프레임·캐시 **vll5 로컬** (`/data2`).

⚠️ 라벨 커버는 x 25~267 px, **y 24~149 px** — `rollout_v2` 의 바닥 (151.7 px, flat_a/flat_v/wall) 은 3 px 바깥이고
   "바닥에 놓인 물체" 외형은 학습셋에 없다. 그 세 시나리오의 y 는 따로 적을 것.
⚠️ 그림자는 물체 바로 아래 바닥에 떨어진다 — x 의 두 번째 단서, y 에는 정보가 없다 (line_z 가 y 를 물체에서 읽게 강제).
   metadata `object_*_by_sample` 은 plan 과 0.05 cm 안에서 일치하지만 라벨은 v2 와 같은 경로(plan) 로 만든다
   (`build_rollout2_index.py --set training`).
⚠️ `attn_probe` 는 캐시 추출용 (v2 와 같은 SET). 캐시 약 19 GiB (896 × 20.97 MB).

raw_frames: 100
cache_tag: rollout_v2_training
results_root: /data/hyuntak/project/2026/2027_cvpr/vjepa2/z_research/RollOutV2/exp_results
root: /data/hyuntak/project/2026/2027_cvpr/vjepa2/data_csv/rollout_v2_training
index_csv: index.csv
frames_root: /local_datasets/world/world_analysis/RollOut_v2_training
frames_pattern: "{file_name}/{frame:06d}.png"
frames_start: 0
frames_stride: 3
block_column: block_id
pair_column: pair_id
variant_column: variant
plausible_column: plausible
type_column: condition

## v11_full

**v11 전체 12조건.** 43,008 clip / 10,752 block / 21,504 matched pair.
`visible 3 + late 3(= v11 본체) + early 3 + mid 3`.

**캐시를 이어 붙여서 만든다 — 다시 뽑지 않는다.**
`v11_vith`(21,504, 이미 있음) 뒤에 `v11_earlymid_vith`(21,504, 새로 뽑음)를 append 한다.
`.npy` 는 C-order 라 axis 0 이어붙이기가 바이트 이어붙이기와 같고,
`21504 -> 43008` 은 둘 다 5자리라 헤더 길이(118B)가 안 변한다.

```bash
# 1) early/mid 캐시만 새로 뽑는다 (~32분, 8 GPU)
GPUS=8 SET="probing.fit_groups_sweep=[null] probing.optims.attn_30.num_epochs=1" \
  bash z_research/scripts/run.sh attn_probe v11_earlymid vith

# 2) 캐시 병합 (~20분, GPU 불필요).  --dry-run 으로 먼저 확인할 것
python z_research/scripts/harness/merge_token_cache.py \
  --into /local_datasets/world/world_analysis/cache/v11_vith \
  --from /local_datasets/world/world_analysis/cache/v11_earlymid_vith \
  --rename-to v11_full_vith --free-as-you-go

# 3) 인덱스도 **같은 순서로** 이어 붙인다 (metadata 순서로 다시 만들면 캐시가 안 맞는다)
python z_research/scripts/data/concat_index.py \
  --out data_csv/intphysgen_v11_full/index.csv \
  data_csv/intphysgen_v11/index.csv data_csv/intphysgen_v11_earlymid/index.csv
python z_scripts/world_model_analysis/build_probe_index.py v11_full
```

⚠️ **인덱스 순서가 `v11 + v11_earlymid` 여야 한다.** 캐시의 `video_ids` 가 그 순서다.
⚠️ **병합하면 `v11` 데이터셋의 캐시가 사라진다** (`v11_vith` → `v11_full_vith`).
   v11 의 채점·probing 결과는 이미 저장돼 있으니 문제 없지만, v11 을 다시 돌리면 재추출한다.
⚠️ 12 조건 × 3 target × 3 run = **108 head**. `SPLIT`/`GSPLIT` 으로 12개 job 으로 쪼갠다.

캐시 840 GiB (병합 후). 병합 중 순간 증가는 base 하나치(target 210 GiB)뿐이다
(`--free-as-you-go` 가 붙인 즉시 원본을 지운다).

raw_frames: 100
cache_tag: v11_full
results_root: /data/hyuntak/project/2026/2027_cvpr/vjepa2/z_research/IntPhysGenV11_occlusion_timing_ablation/exp_results
root: /data/hyuntak/project/2026/2027_cvpr/vjepa2/data_csv/intphysgen_v11_full
index_csv: index.csv
frames_root: /local_datasets/world/world_analysis/IntPhysGen_v11
frames_pattern: "{file_name}/{frame:06d}.png"
frames_start: 0
frames_stride: 3
block_column: block_id
pair_column: pair_id
variant_column: variant
plausible_column: plausible
type_column: condition

## v11_timing

> ⚠️ **`v11_full` 로 대체됐다 (2026-09-02).** 인덱스는 만들어 뒀지만(감사 통과) **캐시는 뽑지 말 것** —
> `v11_full` 이 같은 9조건에 `late` 까지 포함하고, 이미 있는 캐시를 재사용해 더 싸다.
> 이 셋은 `late` 가 없고 `visible` 을 중복 추출해야 했다.

**probing 용 통합 셋.** 32,256 clip / 8,064 block / 16,128 matched pair.
`visible 3조건 + early 3조건 + mid 3조건` = 9 조건.

**왜 따로 뜨나** — `attn_probe` 의 이식(transfer)은 **한 run 안의 group 끼리만** 된다
(`fit_groups_sweep` 이 group 마다 head 를 만들고 `evals[fit].per_group` 으로 전 group 평가).
그래서 `visible → early/mid` 를 보려면 셋이 **같은 인덱스·같은 캐시**에 있어야 한다.

⚠️ **`late` 는 뺐다.** 넣으면 12 조건(43,008 clip, 캐시 ~900G)이 되어 디스크에 안 들어간다.
   `visible → late` 이식은 기존 `attn_probe__v11_vith` 결과에 이미 있다.
⚠️ **`visible` 은 v11 과 같은 clip 이라 캐시가 중복된다.** `TokenCache.matches()` 가
   `video_ids` **완전 일치**를 요구해서 상위집합 재사용이 안 된다 (`cache.py:57`).
   재추출을 피할 방법이 지금은 없다.
⚠️ 9 조건 × 3 target × 3 run = **81 head** 다. 8시간 벽을 넘으므로 `SPLIT`/`GSPLIT` 으로 쪼갠다.

캐시 예상 크기: 32,256 clip × 8,192 token × 1,280 dim × 2 byte ≈ **676G**
(`/data2` 여유 867G — 들어가지만 남는 게 190G 뿐이다. 돌리기 전에 `df -h` 할 것).

raw_frames: 100
cache_tag: v11_timing
results_root: /data/hyuntak/project/2026/2027_cvpr/vjepa2/z_research/IntPhysGenV11_occlusion_timing_ablation/exp_results
root: /data/hyuntak/project/2026/2027_cvpr/vjepa2/data_csv/intphysgen_v11_timing
index_csv: index.csv
frames_root: /local_datasets/world/world_analysis/IntPhysGen_v11
frames_pattern: "{file_name}/{frame:06d}.png"
frames_start: 0
frames_stride: 3
block_column: block_id
pair_column: pair_id
variant_column: variant
plausible_column: plausible
type_column: condition

## v11_occtiming

> ⚠️ **available: false (2026-09-01).** 원본 렌더가 v11 본체로 병합되면서 사라졌다.
> 후속은 아래 `## v11_earlymid` 다. 이 셋의 결과는 기록으로만 남는다
> (`z_research/IntPhysGenV11_occlusion_timing_ablation/`).
> ⚠️ pilot 은 early/mid 가 문맥을 **2k** 장 가렸다. 후속은 **k** 로 맞췄다 — 수치를 섞지 말 것.

available: false

**v11 의 가림 타이밍 절제 (pilot).** 12,096 clip / 3,024 block / 6,048 matched pair
= `가림조건 3 × k 4 × 타이밍 3 × 84 block × 4 clip`.

v11 은 가림이 **문맥 끝 k 장 + 미래 앞 k 장** 으로 `semantic_event_frame` 에 대칭 고정이었다.
이 셋은 그 창을 **앞뒤로 옮긴다** — `k` 는 그대로 두고 **언제 가려지는가**만 바꾼다.

| 축 | 값 |
|---|---|
| `condition` | `static_occlusion` / `moving_occlusion_flat` / `moving_occlusion` — **가림 조건만 3개** |
| `sym_k` | 1 / 2 / 3 / 4 (**k=0 없음** — 비가림 대조군은 v11 것을 쓴다) |
| `occ_timing` | `early` / `mid` / `late` |
| `violation_type` | vanish / shape / color |

⚠️ **비가림 대조군이 이 셋에는 없다.** `k=0` 비교가 필요하면 v11 의 `*_visible` 을 쓴다.
⚠️ `condition` 에 타이밍을 접지 않았다 — `fit_groups_sweep: auto` 가 3 그룹만 만든다.
   타이밍별 분해는 `report.py` 의 축(`occ_timing`)으로 한다.

`index.csv` 는 v11 과 같은 14컬럼이고 `index_probe.csv` 가 `sym_k` · `occ_timing` 을 싣는다.
문맥 무결성 전수 감사 통과 — 96,768 쌍 검사, **mismatch 0**
(`data_csv/intphysgen_v11_occtiming/context_integrity.json`).

raw_frames: 100
cache_tag: v11_occtiming
results_root: /data/hyuntak/project/2026/2027_cvpr/vjepa2/z_research/IntPhysGenV11_occlusion_timing_ablation/exp_results
root: /data/hyuntak/project/2026/2027_cvpr/vjepa2/data_csv/intphysgen_v11_occtiming
index_csv: index.csv
frames_root: /local_datasets/world/world_analysis/IntPhysGenV11_occlusion_timing_ablation
frames_pattern: "{file_name}/{frame:06d}.png"
frames_start: 0
frames_stride: 3
block_column: block_id
pair_column: pair_id
variant_column: variant
plausible_column: plausible
type_column: condition

## v10_flat

**v10 의 flat 팔.** 2026-08-28 에 v10 에 `moving_occlusion_flat` / `moving_visible_flat`
두 조건(각 2048 clip)이 추가됐다. 이 셋은 **그 둘만** 담는다 — 4096 clip / 1024 block /
2048 matched pair. `--strict-context` 로 32,768 문맥 프레임 쌍 전수 검사, mismatch 0.

⚠️ **v10 본체와 짝지어 읽는 것이 요점이다.** v10 은 `moving`=ramp / `static`=flat 으로
`surface` 와 `motion` 이 **완전 공선**이었다 (조건 4개가 surface 값을 하나씩만 가졌다).
그래서 "움직여서 무너지는가, 경사면이라 무너지는가"를 원리적으로 못 갈랐다.
flat 팔이 `moving x flat` 칸을 채워 그 교락을 끊는다. 비교할 짝:
`moving_visible`(ramp) vs `moving_visible_flat` · `moving_occlusion`(ramp) vs `moving_occlusion_flat`.

프레임·metadata 는 v10 본체 디렉토리를 그대로 쓴다. index 는 flat 만 걸러 만든 것이라
v10 의 index.csv(8192행)·토큰 캐시(v10_vith, 161 GiB)를 건드리지 않는다.
probing 캐시는 `v10_flat_vith` 로 따로 생기고 약 80 GiB 다.

raw_frames: 100
cache_tag: v10_flat
results_root: /data/hyuntak/project/2026/2027_cvpr/vjepa2/z_research/IntPhysGenV10/exp_results
root: /data/hyuntak/project/2026/2027_cvpr/vjepa2/data_csv/intphysgen_v10_flat
index_csv: index.csv
frames_root: /local_datasets/world/world_analysis/IntPhysGen_v10
frames_pattern: "{file_name}/{frame:06d}.png"
frames_start: 0
frames_stride: 3
block_column: block_id
pair_column: pair_id
variant_column: variant
plausible_column: plausible
type_column: condition

## v10_occ_low

**가림 길이 통제 세트.** v10 과 모든 축이 같고 **가림 길이만 짧다** — `sym_k: 1` 이라
context 끝 1장(raw 45) / predictor 앞 1장(raw 48)만 가린다. v10 은 4+4 (36~45 | 48~57).
1024 clip / 256 block / 512 matched pair. `--strict-context` 로 8,192 쌍 전수 검사, mismatch 0.

CLAUDE.md §10-5 의 미해결 항목("가림 길이 통제 세트: L1 이 IntPhys 수준까지 오르는가")이 이 셋이다.
v10 과 짝지어 읽는 게 요점이다 — 물체 크기(31.8px)·ramp·배경 교차가 전부 같으므로
**두 셋의 차이는 가림 길이 하나뿐**이다.

`occ_height_cm` 이 250~450 으로 블록마다 다르다(v10 은 고정). 가림막 크기로 길이를 맞춘
것이라 §11 의 "가림 길이는 가림막 크기가 아니라 속도/타이밍으로 조절할 것" 과 어긋난다 —
높이와 길이가 함께 움직이니 둘을 분리해 해석하지 말 것.

raw_frames: 100
cache_tag: v10_occ_low
results_root: /data/hyuntak/project/2026/2027_cvpr/vjepa2/z_research/IntPhysGenV10_low_occlusion
root: /data/hyuntak/project/2026/2027_cvpr/vjepa2/data_csv/intphysgen_v10_occlusion_low
index_csv: index.csv
frames_root: /local_datasets/world/world_analysis/IntPhysGen_v10_occlusion_low
frames_pattern: "{file_name}/{frame:06d}.png"
frames_start: 0
frames_stride: 3
block_column: block_id
pair_column: pair_id
variant_column: variant
plausible_column: plausible
type_column: condition

## jongseo_physv3

외부 대조군. swapshape 50쌍. **유일하게 PNG 가 아니라 mp4 직독**이다
(`file` 컬럼이 /data/jongseo/... 절대경로). 그래서 frames_root 가 없다.

raw_frames: 100
cache_tag: jongseo_physv3
root: /data/hyuntak/project/2026/2027_cvpr/vjepa2/data_csv/jongseo_physv3_swapshape
index_csv: index.csv
block_column: block_id
pair_column: pair_id
variant_column: variant
plausible_column: plausible
type_column: condition

## 2d_v8_transit

available: false
note: 원본 프레임(195M)이 2026-08-28 z_research 정리 때 지워졌고 생성 코드도 함께 사라졌다. 토큰 캐시(/local_datasets/world/world_analysis/cache/attn_probe_2d_v8_transit_vith, 41GB)는 남아 있어 캐시 기반 사후 분석은 계속 가능하지만 재추출·재렌더는 불가능하다.

raw_frames: 32
cache_tag: 2d_v8_transit
root: /data/hyuntak/project/2026/2027_cvpr/vjepa2/data_csv/2d_intphysgen_v8_transit
index_csv: index.csv
frames_root: /data/hyuntak/project/2026/2027_cvpr/vjepa2/z_research/IntPhys-Like_data/2D_video/data/2D_IntPhysGen_v8_transit
frames_pattern: "{file_name}/{frame:06d}.png"
frames_start: 0
frames_stride: 1
block_column: block_id
pair_column: pair_id
variant_column: variant
plausible_column: plausible
type_column: condition

## v10

정식 IntPhysGen. **8192영상 / 2048 block** (v8 의 4배). `condition` 4분할 균등(2048씩),
`violation_type` = vanish 4096 / color 2048 / shape 2048.
`build_intphysgen_v6_index.py --strict-context` 로 **65,536 문맥 프레임 쌍 전수 검사, mismatch 0.**

v8 대비 바뀐 것 (`dataset.json.changes`):
- **`surface` 축 신설** — flat 4096 / ramp 4096. moving 이 평지를 미끄러지는 대신 쐐기를
  내려온다. **가림이 감추는 연속을 중력이 강제**하므로, predictor 가 커밋하지 않던 문제
  (α≈0.5, `CLAUDE.md` §5-4b)를 데이터 쪽에서 건드리는 축이다.
  ⚠️ index.csv 에는 안 들어간다 — 쓰려면 `metadata.csv` 에서 조인해야 한다.
- **가림 대칭 정정** — `sym_k: 4` 로 context 4장(36,39,42,45) / predictor 4장(48,51,54,57)을
  정확히 가린다. **v8 은 4/5 로 비대칭이었다** (패널을 선언된 plateau + 고정 마진으로 잡아서).
- **배경이 완전 교차 요인** — 모든 shape-color 조합이 배경 4종에 전부 나타난다.
  §5-4c 의 probing 클래스 불균형 교란이 설계 단계에서 제거됐다. 실측:
  `index_probe.csv` 의 obj 3072개가 shape/color **클래스당 정확히 384개**,
  surface flat/ramp 1536, env 4종 768. block 단위 split 후에도 train/class 가
  shape 167~207 · color 175~204 로 v8(28~68, 2.4배)보다 훨씬 고르다.
- 물체 300 → 150 cm (`obj_apparent_px` 31.8)

프로토콜 상수는 v8 과 같다: `splice 48`, `stride 3`, `n_context 16`, `n_predict 16`, `n_target 32`,
원본 288x288 → 256 리사이즈.

⚠️ **`attn_probe` 를 걸면 토큰 캐시가 약 160 GiB 다** (영상당 20 MiB × 8192).
`/data2` 여유가 185G 뿐이라 거의 꽉 찬다. `surprise_c16t32` 는 토큰 캐시를 안 쓰므로 무관하다.
probing 은 `LIMIT=` 나 부분집합 index 로 줄여서 돌릴 것.
`index_probe.csv` 는 만들어져 있다 (`build_probe_index.py v10`, 8192행 37컬럼).
**`surface` 는 index.csv 엔 없고 `index_probe.csv` 에만 있다** — surface 축을 쓰는
분석은 그쪽을 읽을 것.

raw_frames: 100
cache_tag: v10
results_root: /data/hyuntak/project/2026/2027_cvpr/vjepa2/z_research/IntPhysGenV10/exp_results
root: /data/hyuntak/project/2026/2027_cvpr/vjepa2/data_csv/intphysgen_v10
index_csv: index.csv
frames_root: /local_datasets/world/world_analysis/IntPhysGen_v10
frames_pattern: "{file_name}/{frame:06d}.png"
frames_start: 0
frames_stride: 3
block_column: block_id
pair_column: pair_id
variant_column: variant
plausible_column: plausible
type_column: condition
