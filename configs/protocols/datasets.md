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

## rollout_v2

**운동 법칙 7종, 위치 readout 전용.** 5,488 clip / 4,704 block. `flat_v`(등속) `flat_a`(등가속) `ramp_a`(경사)
`arc`(포물선) `fall`(자유낙하) `ledge`(선반 끝 낙하, **pos/imp 쌍**) `wall`(벽 정지, **pos/imp 쌍**).
시나리오마다 primary 7 × secondary 2 셀, 셀당 56 (ledge/wall 은 28 × 2 변이). `holdout` = primary 2·6번째 레벨.
가림 없음. 프레임·캐시 **vll5 로컬** (`/data2`).

⚠️ **2026-09-10 재생성 (plan 14:14, 프레임 14:44, metadata 15:19).** metadata `object_*_by_sample` 이 가능·불가능 모두 plan 과
   일치한다 (불가능 1 px 안). 라벨은 그래도 `build_rollout2_index.py` 가 plan(`UnrealEngine/gen/plans/blocks_rollout2.json`)에서
   만들고 metadata 와 대조한다. `in_frame` 은 metadata ("물체 전체가 화면 안", 가장자리 ~20 px 부터 0). arc 정점이 y 25 px 로
   내려와 가능 변이의 미래 슬롯 off-screen 은 0. (기록: 09-09 판은 wall 불가능이 정지 궤적을 복사하고 있었다.)
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

## rollout_v2_training_v5

**RollOut_v2 위치 readout 의 학습셋 (v5, 2026-09-11 19:34, 폴더 `RollOut_v2_training`).** 8,064 clip
= `rollout2_training_v5` (사물 없음: line_x 1,344 / line_z 896 / line_xz 896 / still 1,344) + `rollout2_training_props` (사물 1개: prop_x 1,792 / prop_still 1,792, 절반은 가림).
무중력 등속. 위치 커버 544/544 셀 (x 19~269, y 20~152). **`in_frame_by_sample` 0 인 샘플 (19.6%) 은 라벨에서 뺀다** (화면 밖).
사물은 `prop_*` 컬럼 (전부 0 = 사물 없음). 클립 폴더는 두 소스로의 심볼릭 링크. README: `/data2/.../RollOut_v2_training/README.md`.
스크립트는 `ROLLOUT2_TRAIN=v5`, 결과 `RollOutV2/exp_results/v5`, `figures/v5`. index: `build_rollout2_index.py --set training_v5 --write` (plan 두 개 합침).
이전 학습셋 v1~v4 (직선만 / kink·고속 / 바닥 셀) 는 2026-09-11 에 프레임·캐시·index·결과를 전부 지웠다. 왜 v5 가 필요했는지는 `RollOutV2/figures/v5/summary/POSITION_READOUT_2026-09-12.md` §6·§8 에만 남긴다.

raw_frames: 100
cache_tag: rollout_v2_training_v5
results_root: /data/hyuntak/project/2026/2027_cvpr/vjepa2/z_research/RollOutV2/exp_results/v5
root: /data/hyuntak/project/2026/2027_cvpr/vjepa2/data_csv/rollout_v2_training_v5
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

## v11_vanish_all

**v11 vanish 의 가능(pos_a) 클립 전부 — 위치 readout 의 v11 전수 검증용.** 2,688 clip = visible 3 조건 × 224 (k=0)
+ occlusion {late, early, mid} × {flat, ramp, static} × k=1..4 × 56. `data_csv/intphysgen_v11_full/index_probe.csv` 에서 그대로 골라낸 행이라 라벨·순서 규약이 같다.
용도는 **context encoder 32 frames (`isolated_ctx:0_32`) 캐시 추출뿐** — p/h 는 `v11_full_vith` 캐시를 그대로 쓴다.
캐시 tag `v11_vanish_all_ctx32_vith` (28 GiB). 만든 곳: `plot_v11_vanish_readout.py` docstring (2026-09-11).

raw_frames: 100
frames_root: /local_datasets/world/world_analysis/IntPhysGen_v11
frames_pattern: "{file_name}/{frame:06d}.png"
frames_start: 0
frames_stride: 3
block_column: block_id
pair_column: pair_id
variant_column: variant
plausible_column: plausible
type_column: condition
cache_tag: v11_vanish_all
results_root: /data/hyuntak/project/2026/2027_cvpr/vjepa2/z_research/RollOutV2/exp_results/v11_vanish_all
root: /data/hyuntak/project/2026/2027_cvpr/vjepa2/data_csv/intphysgen_v11_vanish_all
index_csv: index.csv

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

## v11_split_test

**v11 (12조건) 의 block 단위 test 절반** — 21,504 clip / 5,376 block / **10,752 matched pair**, 가능·불가능 전부.
predictor 학습(`z_training/`)의 held-out 채점용. 짝인 train 절반(가능만)은 `configs/training/datasets.md ## v11_split_train`.
분할은 `z_training/data/build_v11_split_index.py` (seed 0, 50/50, condition × violation_type × sym_k 층화, `split_report.json`).
프레임은 v11 과 같고 채점은 캐시를 안 쓴다. 릴리즈 predictor 기준선: `bash z_training/eval.sh release_vith v11_split_test`.
⚠️ `attn_probe` 를 여기 걸면 캐시(`v11_split_test_<모델>`)를 새로 뽑는다 — v11_full 캐시의 부분집합 재사용은 안 된다 (§7-1).

raw_frames: 100
cache_tag: v11_split_test
results_root: /data/hyuntak/project/2026/2027_cvpr/vjepa2/z_research/IntPhysGenV11/exp_results
root: /data/hyuntak/project/2026/2027_cvpr/vjepa2/data_csv/intphysgen_v11_split
index_csv: index_test.csv
frames_root: /local_datasets/world/world_analysis/IntPhysGen_v11
frames_pattern: "{file_name}/{frame:06d}.png"
frames_start: 0
frames_stride: 3
block_column: block_id
pair_column: pair_id
variant_column: variant
plausible_column: plausible
type_column: condition
