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
