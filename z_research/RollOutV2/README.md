# RollOutV2 — predictor 의 미래 8 tubelet 에서 물체 위치를 읽는다 (시작점)

> **정본 결과 문서: [`figures/v5/summary/POSITION_READOUT_2026-09-12.md`](figures/v5/summary/POSITION_READOUT_2026-09-12.md)** — 모든 수치·정정·재현 명령이 거기 있다.
> 이 README 는 세트의 구조와 **읽는 규칙**, 결론 요약만 담는다. 2026-09-12 저녁 판 (decoder 검증 · 정정 반영).
> **2026-09-19 종합: [`Archive/PREDICTOR_DISTANCE_LIMIT_2026-09-19.md`](Archive/PREDICTOR_DISTANCE_LIMIT_2026-09-19.md)** — p 가 만드는 것, 거리 한계, predictor attention·knockout·속도 probe, 가설 판정 (이 README §4c).

## 0. 한 줄

별도 학습셋 (`RollOut_v2_training` v5, 무중력 등속 8,064 clip) 에서 **위치 자 (attentive readout, 3,842 파라미터)** 를 p / z / h 각각에 정하고,
운동 법칙 7종의 test 셋 `RollOut_v2` (가능 4,704 clip) 와 IntPhysGen v11 가림 셀 (2,688 clip) 에 **test 로만** 건다.
encoder 자는 어디서나 물체를 읽고 (≤ 0.6 칸), p 는 등속에서만 물체 위에 있으며, ledge 에서는 선반 높이를 유지하고, wall 은 2 슬롯 통과한 뒤 물체를 잃고,
v11 가림이 경계에 걸리면 p 의 미래에는 처음부터 물체다운 토큰이 없다.

## 1. 읽는 규칙 (반드시)

1. **자는 물체가 없어도 위치를 낸다.** attention 이 퍼지면 Linear(토큰 평균) = 기본값이 나온다. v11 은 기본값 ≈ 마지막 관측 위치 (화면 중심), wall 은 기본값 ≈ 정지 자리 옆.
   → **위치 주장은 그 슬롯의 진실/후보 3×3 attention 질량이 균등 (9/256 = 0.035) 을 넘을 때만** 한다. `attn_diag.md`, `two_futures_attn.md` 가 그 표다.
2. **8 슬롯 평균 지표 (L2, 두 미래 비율, 슬롯 7 ratio) 는 기본값을 섞는다.** 슬롯별 표와 같이 읽는다.
3. **decoder 검증을 먼저 보인다** (§2-3): held-out 0.87 칸 (학습 0.83), encoder 대조 ≤ 0.6 칸, 자 없는 검사 (v11 h_pos/h_imp 토큰 L1 비율) 와 일치. 이 셋이 있어야 "p 의 세계" 를 말할 수 있다.
4. **"p 가 물체를 X 에 둔다" 와 "p 에 물체가 없다" 는 다른 도구로 말한다.** 전자 = 자의 위치 (질량 조건 하), 후자 = 질량·기본값 거리·자 없는 검사.
5. 학습셋에 "물체 없는 p" 는 없다 (화면 밖·판 뒤는 손실에서 뺌). presence head 는 보류 (2026-09-12, 사용자 결정).

## 2. 데이터와 캐시

| 이름 | 용도 | clip | 프레임 | 캐시 (`/local_datasets/world/world_analysis/cache/`) |
|---|---|---|---|---|
| `rollout_v2` | test (**9 시나리오**: flat_v, flat_a, **flat_d**, ramp_a, **ramp_d**, arc, fall, ledge, wall; pos/imp 쌍). 감속 두 개는 2026-09-19 에 뒤에 붙였다 (이 README §4b) | **7,056** (= 5,488 + 1,568) | `RollOut_v2` | `rollout_v2_vith` (p, h) · `rollout_v2_ctx32_vith` (z = context encoder 32 frames) · `rollout_v2_z16_vith` (ctx_masked 16 frames, probe 용) |
| `rollout_v2_training_v5` | 자 학습 (line_x/z/xz, still, prop_x, prop_still; `visible_by_sample` 마스크) | 8,064 | `RollOut_v2_training` | `rollout_v2_training_v5_vith` (p, h) · `rollout_v2_training_v5_ctx32_vith` (z) |
| `v11_vanish_all` | v11 pos_a 전부 (visible k=0 + late/early/mid × flat/ramp/static × k=1..4) | 2,688 | `IntPhysGen_v11` | `v11_vanish_all_ctx32_vith` (z 만; p/h 는 `v11_full_vith`) |

1 토큰 칸 = 18 px = 물체 반폭. (이전 학습셋 v1~v4 와 late 만의 v11 부분집합은 삭제됐다 — 기록은 정본 §6·§8.)

## 3. 폴더

```
Archive/                            날짜별 분석 문서 (PREDICTOR_DISTANCE_LIMIT_2026-09-19.md)
exp_results/
  v5/attentive_pooling/{p,z,h}/     attn.pt, fit.json, test.json, preds.npz (pred·truth·attn)   ← 본체 (학습셋 v5 자)
  v5/attentive_pooling/p_holdout50/ 학습셋 절반 held-out 검증 (holdout.json)
  v5/spatial_pooling/{z,h}/         256 토큰 평균 → OLS 자 (대조)
  v5/locality/                      거리 × 슬롯 (distance_decay.md), predictor attention·knockout·속도 probe (report.md, exp{1,2,3}_*.json) — 2026-09-19
  probe_pos_imp/                    ledge/wall pos·imp attentive probe (json, log)
  v11_vanish_all/                   v11 z 캐시 추출 기록 + readout_runs.log
  attn_probe*__rollout_v2*_vith/    캐시 추출 실행 기록 (_resolved.yaml)
figures/
  v5/summary/POSITION_READOUT_2026-09-12.md   정본. fig_l2 / fig_motion_gain / fig_motion_xy, two_futures_attn.{md,json}
  v5/summary/fig_xy_velocity · fig_xy_profile · fig_step_profile   운동 9 종 속도-시간 / 위치-시간 / 한 축 걸음 (2026-09-19)
  v5/attentive_pooling/{p,z,h}/{overlay,traj}/  v2 클립 overlay·궤적
  v5/attentive_pooling/p/traj_gif/             위 traj 의 GIF 판 (32 샘플 프레임, 정답·읽기 궤적 누적, time bar)
  v5/attentive_pooling/pair_gif/               같은 21 clip 의 context encoder z vs predictor p 나란히 GIF (`plot_v11_vanish_pair_gif.py --dataset v2 --out`)
  v5/v11_vanish/{p,z}/<motion>[_early|_mid]/k<k>/  GIF·overlay·fig_gain·readout.npz
  v5/v11_vanish/pair/                          encoder vs predictor 나란히 GIF (발표용; <motion>_<timing>/k1~k4/ 하위, 가림막 없는 k=0 은 <motion>_visible/)
  v5/v11_vanish/timing/                        fig_timing, timing_summary, attn_diag, token_test
  v5/v11_vanish/emergence/fig_position_k1      visible / early k=1 / mid k=1 / late k=1 미래 위치 (진실·z·p 평균 ± SD, 출구 모서리 점선) — 가장 읽기 쉬운 판
  v5/v11_vanish/emergence/                     물체가 가림막 밖으로 나오는가: z vs p. by_condition/fig_<late|mid|early>_k<1..4> (위치 + 튜블릿 상태), data/*.npz, emergence.{md,json}, _superseded/ (첫 판)
  probe_pos_imp/                               fig_confusion_all_{h_ctx,z}
```

## 4. 결론 요약 (수치는 정본 문서 §번호)

| 주장 | 근거 | 위치 |
|---|---|---|
| 자는 믿을 수 있다 | held-out 0.87 칸; encoder 자 v2 전 시나리오 ≤ 0.61 칸, v11 어디서나 추적; 자 없는 검사 (r = \|p−h_imp\|/\|p−h_pos\|) 와 슬롯 단위 일치; v11 만으로 학습한 자는 자리 prior 를 외움 (기각) | §2 |
| p 는 등속만 물체 위 | flat_v 0.71 칸 gain 0.85; flat_a 1.07 / 0.57; ramp·arc·fall 0.9~1.4 칸, gain 0.4~0.5 | §3 |
| ledge: 선반 높이 유지 (부유) | 슬롯 3~4 부유 궤적 질량 0.43 / 0.39 (낙하 0.17 / 0.19), 기본값은 낙하 쪽, 392 쌍 100 %; probe P(imp) 0.91~0.96 | §4-1, 4-3, 4-4 |
| wall: 2 슬롯 통과 뒤 물체 소실 (멈춤 없음) | 슬롯 1~2 통과 질량 0.61 / 0.48; 슬롯 3~7 질량 ≤ 0.18, 읽기 = 기본값; probe P(imp) 0.70~0.90. ("+29 px 에서 멈춤" 은 정정) | §4-2 → 4-3, 4-4 |
| v11: 물체다운 토큰이 남아 있는 길이 = late 0 < early/mid 3 < 가림막 없음 4~5 슬롯 | 자 질량 (k=0 0.74→0.06, early 0.59→0.03, late 0.03), 자 없는 검사 frac r>1 (k=0 1.00→0.05 슬롯 5, early 슬롯 3 0.59, late k4 슬롯 0 0.54) | §5-3, 5-4, 2-3 C |
| v11: "마지막 관측에 머문다" 는 기본값 | 마지막 관측 3×3 질량 0.01~0.03, 읽기−기본값 19~36 px, 기본값−마지막관측 14~25 px | §5-4 정정 |
| ramp late 만 k 에 따라 악화, static 도 late 만 1 칸 넘게 | ratio 0.21→0.04; static p late 20~24 px vs early/mid 12~15, k=0 9.6 | §5-4 |
| 채점과 위치는 별개 축 | vanish 채점 early 97.8 / mid 95.8 / late 57.5 vs 위치는 셋 다 3 슬롯 안 소실 | §5-4 |

폐기·정정한 해석 전체 목록: 정본 §8.

## 4b. 감속 시나리오 추가 (2026-09-19)

그 전까지 이 세트에는 **수평 감속이 없었다** (flat_a 는 정지에서 출발해 빨라지기만 하고, ramp_a 는 내리막뿐; 감속은 arc·fall 상승 구간의 수직 성분과 wall 충돌뿐).
렌더에 두 시나리오를 더했다 (가능 영상만, 각 784 clip, 예측 16 샘플 안에서 방향 반전 없음):

| 시나리오 | 운동 | primary | secondary |
|---|---|---|---|
| `flat_d` | 지면 위 수평 감속 | `flat_v` 초기 속도 200–340 cm/s | `flat_a` −20 / −30 cm/s² |
| `ramp_d` | 쐐기 오르막 감속 | `ramp_v0` −200 … −300 cm/s (오르막) | `ramp_acc` 15 / 30 |

**캐시는 새 clip 만 뽑아 기존 세 캐시 뒤에 붙였다** (`rollout_v2_vith`, `_ctx32_vith`, `_z16_vith` 모두 7,056; 붙인 행이 원본과 바이트 동일,
인덱스 순서 일치 확인). 자는 **다시 학습하지 않았다** — 저장된 `attn.pt` / `w.npy` 로 test 만 돌렸고, 기존 7 시나리오의 예측 차이는 최대 4.8e-7 이다.
옛 인덱스 `data_csv/rollout_v2/_before_decel_20260919/`, 옛 예측 `exp_results/v5/attentive_pooling/_before_decel_20260919/`, 옛 요약 그림 `figures/v5/summary/_superseded/before_decel_20260919/`.

결과 (attentive 자, 진행 방향 x, 가능 clip 784 개씩; `v5_decel_summary_fig.log` 와 아래 재현의 계산):

| 시나리오 | 진실 변위 (px, 슬롯 0→7) | p | h | 등속 외삽 | p gain (corr) | h gain | 물체 칸 attention 질량 p 슬롯 0 / 7 |
|---|---:|---:|---:|---:|---:|---:|---|
| flat_v | 44.8 | 39.3 | 43.9 | 44.8 | 0.85 (0.98) | 1.00 | 0.75 / 0.37 |
| flat_a (가속) | 49.4 | 28.2 | 48.7 | 28.4 | 0.57 (0.97) | 1.01 | 0.72 / 0.25 |
| **flat_d (감속)** | 78.4 | 71.0 | 79.8 | 101.3 | **0.26 (0.49)** | 1.00 | 0.75 / 0.17 |
| ramp_a (가속) | 68.4 | 38.2 | 69.4 | 47.8 | 0.40 (0.67) | 1.03 | 0.57 / 0.11 |
| **ramp_d (감속)** | 74.6 | 62.2 | 74.4 | 95.0 | **0.26 (0.34)** | 1.13 | 0.67 / 0.15 |

- **자는 감속도 읽는다**: h / z 의 gain 1.00–1.15, 위치 오차 ≤ 0.44 칸.
- **가속에서 p 의 평균 변위는 등속 외삽과 같다** (flat_a 28.2 vs 28.4) — 가속을 이어가지 않는다.
- **감속에서 p 의 평균 변위는 등속 외삽보다 작고 진실에 가깝다** (flat_d 71 / 진실 78 / 등속 101). 그러나 clip 사이 gain 0.26, corr 0.49 로
  **clip 마다의 감속량을 따라가지 않는다** — 평균적으로 비슷한 거리를 가는 것이다. flat_v 의 p 도 진실의 0.88 만 가므로 "짧게 가는 경향" 과 섞여 있다.
- ⚠️ 슬롯 7 의 물체 칸 질량이 감속에서 0.15–0.17 로 flat_v (0.37) 보다 낮다 (균등 0.035 보다는 크다). 뒤 슬롯일수록 기본값이 섞인다 (§1-2).
  그래서 **"p 가 감속을 만든다" 는 이 결과로 못 쓴다.** 쓸 수 있는 건 "p 는 가속·감속 어느 쪽도 clip 별로 이어가지 않는다" 까지다.

**슬롯별로 보면 (그림 `figures/v5/summary/fig_step_profile.png`, `plot_rollout2_step_profile.py`)** p 의 걸음 모양이 세 시나리오에서 같다:
0→1 에서 문맥 끝 속도의 1.7~1.9 배로 한 번 뛰고, 슬롯 3 까지 문맥 속도를 유지하다가, 슬롯 4 이후 거의 멈춘다 (5→6 은 셋 다 0 근처 또는 음수).
진실은 평평 / 증가 / 감소인데 p 의 모양은 그대로다 → flat_a 는 후반에 뒤처지고 (−26 px), flat_d 는 초반에 앞선다 (+25 px, 슬롯 3).
flat_d 슬롯 0 은 가장 빠른데 (13.8 px/튜블릿) 오차가 가장 작고 (6.9 px), flat_d 후반은 느린데 (10 px/튜블릿) p 가 거의 안 움직인다
→ **"빨라서 못한다" 가 아니라 "문맥 속도로 조금 가다 멈추는 고정된 동작" 이다.** 물체 칸 attention 은 슬롯과 함께 떨어진다
(회귀 표준화 계수 슬롯 −0.17 / 속도 −0.08). ⚠️ 슬롯 4 이후의 "멈춤" 은 p 가 물체를 제자리에 둔 것인지 표현이 흐려져 자가 기본값으로 끌린 것인지
이 도구로 가르지 못한다. flat_a 는 정지 출발, flat_d 는 200–340 cm/s 출발이라 속도 범위가 겹치지 않는다 (경계 속도를 맞춘 가속/감속 쌍이 없다).

> ⚠️ 정정 (2026-09-19 같은 날, §4c) — 위 문단의 두 해석은 틀렸다. 지우지 않고 남긴다.
> 1. **"슬롯 −0.17 / 속도 −0.08, 슬롯 탓"** → 변수가 속도였다. 속도 × 시간 = **마지막 관측 자리에서의 이동 거리** 로 재면 거리가 주 변수다
>    (자 attention: 슬롯 −0.26 / 거리 100 px 당 −0.43; 자 없는 검사: −0.28 / −0.47).
> 2. **"문맥 속도로 조금 가다 멈추는 고정된 동작"** → p 는 멈추는 것이 아니라 **물체를 놓친다** (후반 속도 0 은 자의 기본값). flat_v 는 슬롯 5 까지 진실을 따른다.
>    "빨라서 못한다" 는 **이동 거리로 바꿔 말하면 맞다**. 초반 1.7~1.9 배 과속은 1 칸 안의 차이라 주장하지 않는다.

## 4c. p 가 만드는 것과 그 한계 (2026-09-19) — 정본 [`Archive/PREDICTOR_DISTANCE_LIMIT_2026-09-19.md`](Archive/PREDICTOR_DISTANCE_LIMIT_2026-09-19.md)

| 주장 | 근거 | 문서 § |
|---|---|---|
| p 의 물체다운 토큰은 마지막 관측 자리에서 약 2~3 칸 (40~60 px) 안에서만 유지 — 슬롯보다 거리 | 자 없는 적중 슬롯 7: 0–20 px 0.87 / 40–60 px 0.57 / 80–120 px 0.24; 회귀 거리 −0.47, 슬롯 −0.28 (자 attention 과 같은 계수) | §4 |
| 등속 외삽이 아니다 | arc 수평 (진실 = 등속 92 px) 에서 p 53 px 에서 멈춤 | §3 |
| 문맥에 원인이 안 보이는 사건은 없음, 보인 곡률은 일부 따름 | ledge p +8 vs 진실 −60 (자 없이 낙하 자리 적중 0.00); wall 46 vs 17; arc 상승 clip p −27 (등속 +38) | §3, §4 |
| 속도 정보는 입력에 있다 (H3 기각) | z → 문맥 끝 속도 ridge R² 0.996 / 0.989 | §5-3 |
| 멀면 물체의 과거 자리를 찾아가지 않는다 | 진실 칸 미래 query 의 물체 궤적 attention 0.073 (20–40 px) → 0.021 ≈ 균등 (80+ px); 자기 자리 기둥은 균등의 4.6–11 배 | §5-1 |
| 가져올 능력은 4~5 칸까지 있다 (딱 잘린 창 아님) | knockout r=3 → 5: 40–60 px 적중 0.38 → 0.72 (원래 0.80) | §5-2 |
| 재현 | hook 단 predictor 가 캐시 p 를 800 clip 전부 재현 (상대 L1 0.0000) | §5 |

## 5. 스크립트 (`z_research/scripts/`)

| 파일 | 역할 |
|---|---|
| `data/build_rollout2_index.py` | v2 / training_v5 index (plan 에서 라벨, `visible_by_sample`) |
| `analysis/rollout2_test_readout.py` | 공통 설정 (`ROLLOUT2_TRAIN=v5`, 경로, 지표 함수) |
| `analysis/rollout2_attn_readout.py` | p 자 학습·v2 test (`--holdout 0.5` 로 검증 A). GPU 2 장 샤딩 |
| `analysis/rollout2_encoder_readout.py` | 같은 자를 z / h 에 (`--encoder z\|h`) |
| `analysis/rollout2_fit_readout.py`, `rollout2_ceiling.py` | spatial OLS 자 (대조) |
| `analysis/rollout2_two_futures_attn.py` | ledge/wall 슬롯별 attention 질량·기본값 거리 (§4-3) |
| `analysis/rollout2_probe_pos_imp.py` | pos/imp attentive probe (§4-4) |
| `analysis/v11_readout_attn_diag.py` | v11 슬롯별 attention 진단 (§5-4 정정) |
| `analysis/v11_token_object_test.py` | 자 없는 검사 C (§2-3) |
| `figures/plot_rollout2_readout.py`, `plot_rollout2_summary.py` | v2 overlay·궤적, fig_l2 등 |
| `figures/plot_v11_vanish_readout.py` | v11 자 적용 (`--rep --motion --timing --k`), GIF·readout.npz |
| `figures/plot_v11_vanish_timing.py` | timing × k 표·그림 |
| `figures/plot_v11_position_k1.py` | visible / early·mid·late k=1 미래 위치 한 장 (y = 진행 방향 화면 중심 기준 px) |
| `figures/plot_v11_emergence.py` | 미래에서 물체가 가림막 밖으로 나오는가 (z vs p, 출구 모서리 거리 + attention gate). `--plot-only` 로 그림만 |
| `figures/plot_v11_vanish_pair_gif.py` | encoder vs predictor 나란히 GIF (`--dataset v11\|v2`, `--timing`) |
| `figures/plot_probe_pos_imp_confusion.py` | probe confusion 2×4 |
| `figures/plot_rollout2_xy_profile.py` | 운동 9 종 화면 x·y 속도-시간 (기본) / 위치-시간 (`--pos`), 문맥 끝 등속 대조 (§4c) |
| `figures/plot_rollout2_step_profile.py` | 한 축 사영 걸음 모양 (§4b) |
| `analysis/rollout2_distance_decay.py` | 거리 × 슬롯: 자 attention + 자 없는 검사 (§4c) |
| `analysis/rollout2_predictor_locality.py` | predictor attention hook · 공간 반경 knockout · z/h 속도 probe (§4c) |

## 6. 재현

정본 문서 `## 재현` 절 (캐시 추출 → 자 학습 → test → v11 → 검증 → 그림, 전부 명령줄). 데이터셋 등록: `configs/protocols/datasets.md` `## rollout_v2`, `## rollout_v2_training_v5`, `## v11_vanish_all`.

감속 추가 (2026-09-19):
```bash
python z_research/scripts/data/build_rollout2_index.py --set v2_decel --write          # flat_d / ramp_d 인덱스 (검증 18항목)
# 새 clip 만 캐시 추출 (기존 캐시를 만든 resolved config 를 그대로 복제한 프로토콜)
GPUS=8 BATCH_SIZE=8 OUTDIR=z_research/RollOutV2/exp_results/attn_probe__rollout_v2_decel_vith \
  bash z_research/scripts/run.sh z_research/RollOutV2/exp_results/decel_ph_protocol.yaml rollout_v2_decel vith
GPUS=8 BATCH_SIZE=8 TAG=rollout_v2_decel_ctx32_vith bash z_research/scripts/run.sh z_research/RollOutV2/exp_results/decel_ctx32_protocol.yaml rollout_v2_decel vith
GPUS=8 BATCH_SIZE=8 TAG=rollout_v2_decel_z16_vith   bash z_research/scripts/run.sh z_research/RollOutV2/exp_results/decel_z16_protocol.yaml rollout_v2_decel vith
#   (셋 다 캐시를 쓴 뒤 버리는 probe head 단계에서 "학습셋이 비어 있다" 로 exit 1 — 캐시는 완성된다)
python z_research/scripts/harness/merge_token_cache.py --into <cache>/rollout_v2_vith --from <cache>/rollout_v2_decel_vith   # ctx32, z16 도 같이
# 인덱스: data_csv/rollout_v2/index{,_probe}.csv = 옛 5,488 행 + rollout_v2_decel 1,568 행 (block_id 보존)
ROLLOUT2_TRAIN=v5 python z_research/scripts/analysis/rollout2_attn_readout.py --test-only
ROLLOUT2_TRAIN=v5 python z_research/scripts/analysis/rollout2_encoder_readout.py --encoder z --test-only    # h 도
ROLLOUT2_TRAIN=v5 python z_research/scripts/figures/plot_rollout2_summary.py
ROLLOUT2_TRAIN=v5 python z_research/scripts/figures/plot_rollout2_readout.py --pooling attentive_pooling --rep p --scenarios flat_d ramp_d   # z, h 도
ROLLOUT2_TRAIN=v5 python z_research/scripts/figures/plot_rollout2_traj_gif.py --pooling attentive_pooling --rep p --clips <flat_d/ramp_d clip id>
ROLLOUT2_TRAIN=v5 python z_research/scripts/figures/plot_rollout2_step_profile.py            # 걸음 모양 그림
```

§4c (2026-09-19): 명령은 [`Archive/PREDICTOR_DISTANCE_LIMIT_2026-09-19.md`](Archive/PREDICTOR_DISTANCE_LIMIT_2026-09-19.md) `## 재현`.

## 7. 이 세트에서 걸렸던 함정

- 자의 기본값이 데이터의 "의미 있는 자리" (v11 마지막 관측, wall 정지점) 와 겹쳐 하루 동안 잘못 읽었다 (§8). → 규칙 1·2.
- v11 k=0 p 만으로 자를 학습하면 자리 prior 를 외운다 (오차 3 px 인데 물체 질량 0.1). 위치를 고루 덮는 학습셋이 필요하다.
- 학습셋 판 뒤 튜블릿을 자가 16~20 px 로 읽는 것은 판 자리를 읽는 것과 구분이 안 된다 — 증거로 쓰지 말 것.
- v3 (kink·고속) 학습셋은 자를 흐리게 했다 — 재시도 금지. v11 속도 (18 px/tubelet) 는 학습셋 상한 (12) 밖이지만 k=0 슬롯 0~4 는 12 px 로 읽힌다.
- 캐시는 `video_ids` 완전 일치 — 부분집합으로 먼저 뽑은 캐시를 전체에 재사용 못 해 v11 z 를 두 번 뽑았다. **묶어서 뽑을 범위를 index 를 만들 때 정한다.**
- 캐시 `ctx_masked` 는 LN(z) 다 — predictor 입력으로 쓰면 캐시 p 와 상대 L1 0.60 으로 어긋난다. predictor 를 다시 돌리려면 문맥 encoder 를 다시 돌린다 (2026-09-19).
- bf16 predictor 출력은 batch 구성에 따라 달라진다 (같은 clip, batch 1 vs 4 에서 상대 L1 0.049). 캐시 재현 검사는 batch 를 맞춘다.
- `datasets.md` 에 같은 섹션이 두 번 들어간 적이 있다 (2026-09-12 제거). `sed -i` 금지 — Python 으로 고친다.
