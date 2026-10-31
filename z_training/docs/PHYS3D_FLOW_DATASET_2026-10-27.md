# phys3d_flow — 오래 이어지는 운동이 기본인 3D 합성 학습 세트 (2026-10-27, v1 완성 2026-10-27 14:42 UTC)

**왜 (사용자 2026-10-27)**: "데이터셋 새로 설계 … 지금 데이터셋 보니까 정지해 있는 경우도 있고" · "오래 이어지는 운동이 기본이 되도록. 15 % 괜찮네 정지 …
시나리오 최대한 많이 그리고 물체 종류도 최대한 많이 물체 texture 도 다양하게 배경도 다양하게" · "배경에도 살짝 노이즈 시간에 따라 계속 변하게" ·
"물체가 좀 시원시원하게 움직였으면" · "fps 10 으로 뽑을 거고 context 최대 16 block, prediction 최대 8 block" · "물체의 움직임 궤적 depth 도 다양하게" ·
"ledge 가 평면에서 좀 운동하다가 떨어지는 게 아니라 그냥 바로 떨어지고 좀 어색한 게 많은 거 같은데 … 물리적으로 좀 생각해봐".
원칙: [`DATA_DESIGN_PRINCIPLES_2026-10-26.md`](DATA_DESIGN_PRINCIPLES_2026-10-26.md) (정본 — 원칙 A: 물리적인 흐름이 그 상황의 흔한 결과) ·
[`SYNTHETIC_TRAINSET_PRINCIPLES_2026-10-25.md`](SYNTHETIC_TRAINSET_PRINCIPLES_2026-10-25.md) (장면 세부).

| | |
|---|---|
| 위치 | `/workspace/dataset/training_data/phys3d_flow_v1/{train,val}` — 구성은 phys3d_multi_v1 과 같다 (`clips/` · `boundary.tsv` · `uniform.tsv` · `manifest.tsv` · `audit.md`) |
| 형식 | 256 × 256, **10 fps, 48 장 (4.8 초)** = 문맥 최대 32 장 (16 블록) + 예측 최대 16 장 (8 블록). 시뮬 10 fps · 부분 단계 12 (120 Hz) |
| 앵커 | **늘 저장 장 32** — 사건 (보임) · 사건 가림 · 그냥 가림 1/3 씩. 로더가 C ∈ {2, 4, 8, 16, 32} 를 고르면 문맥 `[32 − C, 32)` 은 늘 앵커 바로 앞에서 끝난다 (`boundary_frame 32`, `ctx_start 0`) |
| 생성기 | `z_training/data/gen_phys3d_flow_blender.py` (phys3d_multi 를 import, 갈고리로만 바꿈 — v1 그대로 재현) · `gen_phys3d.py --multi --flow --defer-render` · `run_gen_phys3d_flow_v1.sh` |
| 자산 | Poly Haven CC0 — HDRI 1k 400 장 (`/workspace/dataset/assets/hdri_polyhaven_1k/`), 사진 질감 1k 309 장 (`/workspace/dataset/assets/tex_polyhaven_1k/`, wood · rock · brick · fabric · metal …). 각 폴더에 `selection.tsv` · `fetch.sh` · `LICENSE_NOTE.txt` |

## 1. phys3d_multi_v1 과 다른 것

1. **오래 이어지는 운동** — 창 검사 (`window_hook`): 영상의 85 % 는 주 물체가 창의 ≥ 85 % 장에서 움직이고 장면 정지 ≤ 5 %. 15 % 는 멈춤 허용 (장면 정지 ≤ 35 %, 멈춤은 예측 안에서만).
   계속 들어오는 · 쌓이는 장면 (`active_any`) 은 장면 안 아무 물체나 움직이면 된다. 움직이는 동안 화면 속력 중앙 ≥ 12 px / 초.
2. **물리는 재질답게** (⚠️ 정정 2026-10-27, §3): 마찰 · 반발은 방 · 시나리오의 보통 값, 얼음 바닥은 curling 만, 중력 4~12 m/s² (로그 균등), 속도 배율 1.0~2.2.
   오래 가는 운동은 미끄럼이 아니라 **구조**로 — 구르는 물체 (축 대칭 모양은 60 % 눕힘), 경사 · 매끄러운 둔덕, 진자, 반원통, 당구, 계속 들어오는 물체, 시간을 맞춘 접근.
3. **깊이** — 물체 길 y −1.0 ~ 2.3 m (v1 ±0.8), 구조물 (passive 상자 · 쐐기) 을 그 깊이로 늘려 뒷벽에 붙임, 묶이지 않은 물체의 출발 깊이 −0.6 ~ +1.4 m,
   초속도에 앞뒤 성분 (70 %, 수평 속력 ±35 %). 서로 부딪혀야 하는 장면은 같은 깊이로. 시험: 주 물체 평균 깊이 −0.7 ~ 2.0 m, 54 % 가 0.5 m 넘게 앞뒤로.
4. **시나리오 33 → 45** — 새로 plinko (못 판) · bowling · curling · bounce_ball · newton_cradle · double_pendulum · rolling_hills · tumble_slope · pendulum_ball ·
   ball_rain · sweeper (회전 막대) · cart_shuttle. 고친 것: ledge_fall · ledge_to_shelf (탁자 위 3.4~6 초 뒤 떨어짐) · tower_collapse (굴러온 공이 3.4~5 초 뒤 침) ·
   block_stack (6~10 개가 0.6~1.4 초 간격으로 떨어져 쌓임) · shelf_catch · container · seesaw · stack_topple (계속 들어오는 물체) · moving_platform (왕복) · flat_decel (멈춤이 결과).
5. **모양 18 → 30** — rod · plank · plate · can · frustum · barrel · egg · diamond · half_sphere · star · wedge · gem.
6. **물체 질감 10 종** — 한 색 · 체크 · 줄무늬 · 대리석 · 점박이 · 나이테 · magic · 벽돌 · 사진 질감 (바탕색으로 물들임) · 금속. 라벨 `pattern` = 종류 번호.
7. **배경** — 벽 · 바닥 40 % 에 사진 질감, HDRI 400 장. **시간에 따라 천천히 변하는 노이즈**: 벽 · 바닥 무늬 흐름 (4D 노이즈, 세기 3~12 %), 빛 세기 · 색 흔들림, 하늘 회전 (≤ 0.15° / 장). 물체 라벨에는 영향 없음.
8. 가림 4 종 · 문맥 중 가림 · 들어오는 물체 · 라벨 / 렌더 분리는 multi_v1 그대로.

## 2. 시험 (라벨만, 2026-10-27)

| 판 | 통과 | 앵커 때 움직임 | 장면 정지 | 지속률 0.5 · 1 · 1.6 · 2 초 | 비고 |
|---|---:|---:|---:|---|---|
| 4 fps · 64 장 (옛 사양) | 305 / 450 | 89~98 % | 1~7 % | 0.94 · 0.78 · — · 0.17 (4 초 0.00) | 첫 판 |
| 10 fps · 48 장 · C* 층화 | 416 / 450 | 93~96 % | 0~1 % | 0.84 · 0.43 · 0.07 · 0.01 | |
| + 깊이 | 331 / 360 | 90~100 % | 0~2 % | 0.87 · 0.52 · 0.11 · 0.01 | |
| **+ 물리 정정 · 앵커 32** | 350 / 450 | 89 % | 0 % | **0.84 · 0.53 · 0.21 · 0.12** | 문맥 속력 0.98 m/s, 안 보임 23 %, 문맥 중 가림 45 % |
| **v1 train (완성)** | **19,825 / 25,200** | 89 % | 1 % | **0.82 · 0.48 · 0.18 · 0.09** | 앵커 event 5,226 · occ_event 1,359 · occ_any 13,240, 안 보임 22.8 %, 문맥 중 가림 45 %, 시나리오 45 개 (199~560 편), 모양 30, HDRI 400 |
| v1 val (완성) | 799 / 1,125 | 89 % | — | 0.84 · 0.52 · 0.20 · 0.11 | |

- 지속률 = 지금 장 화면 속도가 L 초 뒤에 그대로 이어지는 몫 (라벨, 주 물체, `audit.md` §4-1). 도구 검사: 등속 1.00 · 멈춘 뒤 0.00 · 무작위 ≈ 0.
  자연 영상 (0.5 초 뒤, 광학 흐름): K710 0.20 · SSv2 0.10 · HowTo 0.06. ⚠️ 라벨 (물체) 과 광학 흐름 (칸) 은 같은 자는 아니다.
- 2 초 쪽이 낮은 건 튕김 · 포물선 · 진자처럼 방향이 바뀌는 운동이 많아서다 (원칙 D — 변화도 흔한 결과).

## 3. ⚠️ 정정 — 첫 flow 판은 물리가 어색했다 (2026-10-27)

사용자: "ledge 가 평면에서 좀 운동하다가 떨어지는 게 아니라 그냥 바로 떨어지고 좀 어색한 게 많은 거 같은데". 실측 (첫 판 train ledge_fall 300 편):
**57 % 는 영상 시작 전에 이미 떨어졌고**, 나머지는 영상 안 0.2~0.8 초에 떨어졌다. 원인 다섯:
1. 영상마다 C* (2~32) 에 앵커를 둬 C* 가 작으면 영상이 사건 0.2 초 전에 시작 → **앵커를 늘 32 번째 장에** (C 는 로더가 고른다).
2. 속도 × 1~3 → 탁자 (2~3 m) 를 1 초 안에 지남 → 1.0~2.2 + **시간을 맞춘 접근** (탁자 3.2~4.6 m, 끝까지 3.4~6 초).
3. 바닥 마찰 0.05~0.5 · 물체 마찰 × 0.3~1 → 얼음 위처럼 미끄러짐 → 보통 값.
4. 벽 반발 0.8~0.97 · 물체 ≥ 0.4~0.85 → 고무공처럼 튐 → 보통 값.
5. 중력 2~16 → g ≈ 2 가 달처럼 둥둥 → 4~12.
첫 판 (렌더 일부) 은 `/workspace/dataset/training_data/_phys3d_flow_v1_fastslide_20261027` · 깊이 전 판은 `_phys3d_flow_v1_nodepth_20261027` — 쓰지 말 것 (지워도 된다).

고치는 동안 금방 멈추던 시나리오: tower_collapse 0 / 10 → 7 · block_stack 2 → 7 · moving_platform 1 → 9 · shelf_catch 3 → 9 (같은 시험 크기).

## 4. 단서

- 앵커가 늘 32 장이라 영상의 앞 3.2 초는 앵커 전, 뒤 1.6 초가 예측이다. C = 2 로 학습하면 마지막 2 장만 문맥으로 쓴다 (앞부분은 다른 C · uniform 행이 쓴다).
- 통과율 ~78 %, 평균 시도 ~7 번 — 시나리오마다 통과율이 달라 (block_stack · container · seesaw 등이 낮다) **시나리오 구성이 작업 수보다 덜 고르다** — `audit.md` §2 에서 볼 것.
- 학습이 GPU 를 쓰는 동안 만들어 렌더는 GPU 마다 2 개 + 라벨이 끝난 뒤 CPU 렌더 230 개. train 은 2026-10-27 09:07 → 14:42 UTC (5.6 시간).

## 재현

```bash
cd /workspace/world_model/vjepa2
# 라벨만 시험 (렌더 생략)
PHYS3D_NORENDER=1 PYTHONHASHSEED=0 python3 z_training/data/gen_phys3d.py --multi --flow --cpu-workers 240 --gpus= --out /tmp/flowpilot --per-scenario 10
python3 z_training/data/audit_phys3d_multi.py --root /tmp/flowpilot > /tmp/flowpilot/audit.md
# 한 편
/workspace/tools/blender-4.2.3-linux-x64/blender -t 1 -b -noaudio -P z_training/data/gen_phys3d_flow_blender.py -- --scenario ledge_fall --seed 0 --out /tmp/f
# 전체 (val → audit → train → audit). 끊기면 같은 명령으로 이어 한다
TRAIN_PER=560 VAL_PER=25 CPU_WORKERS=200 RENDER_PER=2 bash z_training/data/run_gen_phys3d_flow_v1.sh
```
