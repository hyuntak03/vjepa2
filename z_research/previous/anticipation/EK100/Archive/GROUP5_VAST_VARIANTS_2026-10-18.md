# EK100 action anticipation — vast predictor 변종 4종 + release (5 팔, group `vith_ek100_g5`)

2026-10-18. 한 번의 run 안에서 **frozen ViT-H encoder 하나를 공유**하고 predictor 만 바꾼 5 팔을
동시에 attentive probe 로 학습·채점했다. 앞선 `vith_ek100_g4` (4 팔) 와 같은 프로토콜이다.

## 1. 결과 (EK100 **subset**, mean class recall@5, epoch 20)

| 팔 | predictor 체크포인트 | action | verb | noun | best lr |
|---|---|---:|---:|---:|---:|
| `ar_vast_op` | `z_training/vast/ar_block_ctxt_block_target_op/latest.pt` | **32.56** | 47.78 | 48.05 | 1.2e-3 |
| `ar_vast_vob` | `…/ar_block_ctxt_block_target_vob/latest.pt` | 32.24 | 46.85 | 48.08 | 1.2e-3 |
| `ctx_vast_video` | `…/ar_video_ctxt_block_target/latest.pt` | 32.19 | 47.82 | 48.79 | 6e-4 |
| `release` (기준팔) | 릴리즈 predictor | 32.13 | 48.85 | 49.32 | 1.2e-3 |
| `ar_vast_gres` | `…/ar_block_ctxt_block_target_residual/latest.pt` | 32.00 | 47.68 | 47.46 | 1.2e-3 |
| *chance* | | *0.45* | *6.8* | *2.5* | |

궤적 (action):

| 팔 | ep5 | ep10 | ep15 | ep20 |
|---|---:|---:|---:|---:|
| `ar_vast_op` | 5.63 | 19.04 | 28.68 | 32.56 |
| `ar_vast_vob` | 4.75 | 18.22 | 28.02 | 32.24 |
| `ctx_vast_video` | 6.28 | 19.34 | 28.43 | 32.19 |
| `release` | 5.66 | 20.11 | 29.00 | 32.13 |
| `ar_vast_gres` | 4.88 | 18.60 | 28.03 | 32.00 |
| **폭** | 1.53 | 1.89 | 0.98 | **0.56** |

## 2. ⚠️ 결론 — 이 probe 는 다섯 predictor 를 가르지 못한다

**같은 팔의 run 간 차이가 팔 사이 차이보다 크다.**

| | action |
|---|---:|
| `release`, `vith_ek100_g4` | 33.11 |
| `release`, `vith_ek100_g5` | 32.13 |
| **동일 팔 run 간 차이** | **0.98** |
| g5 안 5 팔의 폭 | 0.56 |
| g4 안 4 팔의 폭 | 0.36 |

두 run 의 차이는 head 개수뿐이다 (g4 12 = 4 팔 × 3 lr, g5 15 = 5 팔 × 3 lr).
데이터 · seed · encoder · epoch 수 · lr 격자가 모두 같다. head init 의 RNG 소비 순서가 달라진 것으로 본다.
따라서:

- ✅ "다섯 predictor 모두 EK100 anticipation subset 에서 32.0 ~ 32.6 에 들어온다" — 말할 수 있다
- ✅ "chance (0.45) 의 70 배로, 다섯 모두 유효한 문맥을 낸다" — 말할 수 있다
- ❌ "`ar_vast_op` 가 `ar_vast_gres` 보다 낫다" (Δ 0.56) — **말할 수 없다.** 동일 팔 run 노이즈 0.98 안이다
- ❌ "vast 변종이 release 보다 낫다/못하다" — 같은 이유로 말할 수 없다

**g4 의 값과 g5 의 값을 한 표에서 직접 비교하지 말 것.** 각 run 의 `release` anchor 대비로만 읽는다.
시트 `Overall` 의 `EK100 subset` 열에는 두 run 의 값이 섞여 있다 (행 12 · 18 · 21 은 g4, 행 22 · 23 · 25 · 27 은 g5).

**다음에 갈라야 한다면** 필요한 것은 더 긴 학습이 아니라 **seed 반복**이다 — 같은 설정 3 seed 로
run 간 표준편차를 재고, 그보다 큰 간격만 주장한다. 지금 비용은 run 당 7.4 시간 (GPU 6 장) 이다.

## 3. 배선 검증

새로 이식한 두 경로가 제대로 물렸는지를 수치가 아니라 **구조**로 확인했다 (이식 내역은 §5).

| 확인 | 결과 |
|---|---|
| `ar_vast_gres` 의 `keep_gate.{weight,bias}` 가 로드됐나 | ✅ wrapper 의 unexpected-key 가드 통과 (`_load_state_dict` 는 모르는 키를 **조용히 버린다** — 가드가 없으면 gating 없이 5 시간을 태운다) |
| `ctx_vast_video` 가 online encoder 문맥을 쓰나 | ✅ 로그 `ctx_encoder <- 체크포인트의 'encoder' 키` · group 안에서 객체 공유 확인 |
| gating off 가 비회귀인가 | ✅ 실 `ar_state` 가중치로 `forward_seq` · `rollout` · `rollout_kv` · `rollout_state` 전부 `max|Δ| = 0` |
| `rollout_state` 와 `rollout_kv` 가 같나 (gating on) | ✅ Δ = 0 (CPU 테스트 `gating_residual_cpu_test.py` 21/21) |
| ctx_state 분기가 정의와 같나 | ✅ `ar_scoring.ctx_state_from_window` 와 Δ = 0 (`ek100_ctx_state_cpu_test.py` 7/7) |
| ep5 에서 chance 근처가 아닌가 | ✅ gres 4.88 · ctx_video 6.28 (chance 0.45) — 다른 세 팔과 같은 구간 |

## 4. 설정

| | |
|---|---|
| 명령 | `GPUS=6 bash z_vll7_training/eval_ek100.sh vith_ek100_g5` |
| 출력 | `cvpr/results/eval/ek100/action_anticipation_frozen/ek100_vith_ek100_g5` |
| encoder | ViT-H, frozen, 5 팔 공유 (`predictor_group` / `MultiAnticipative`) |
| global batch | 96 (BS 16 × 6 GPU), `ipe` 119 |
| head | `cap16b96` — lr {3e-4, 6e-4, 1.2e-3} × wd 1e-4, 5 팔 = 15 head |
| epoch | 20, `val_every` 5 |
| 데이터 | train `EPIC_100_train_cap16metric_seed0.csv` · val `EPIC_100_validation_cap16metric_seed0.csv` (9,296 clip) |
| subset 클래스 수 | action 1,114 · verb 73 · noun 201 → chance 0.45 / 6.8 / 2.5 |
| `anticipation_point_mode` | `released` |
| 소요 | 14:36:31 → 21:58, **7 시간 22 분**. epoch 19.7 분 · val (15 head) 11 분 |
| 에러 | 0 |

⚠️ **val clip 9,295 / 9,296** (ep10 · ep20). decode 실패로 1 개가 빠졌다 (ep5 · ep15 는 9,296).
mean-class-recall 이라 최악의 경우 (singleton class 가 빠지면) 0.09 pt 까지 움직일 수 있다 —
팔 사이 간격 0.56 보다는 작지만, run 간 비교에서는 이것도 노이즈의 일부다.

## 5. 이 run 을 위해 고친 코드 (origin/vast 이식)

| 파일 | 무엇 |
|---|---|
| `src/models/rollout_predictor.py` | `gating_residual` (vast 416f284) + `gating_anchor` (2910622). ar 출력 = `g·A + proj(st)`, `g = σ(keep_gate(st))/σ(2)`, `A` = 마지막 **관측** 블록 LN(h). vast 는 KV rollout 만 지원하지만 EK100 wrapper 는 `rollout_state` 를 타므로 **거기에도 구현**했다 (KV 와 Δ=0 검증). wrapper 를 KV 로 바꾸면 이미 보고한 `ar_vast_ep70` 33.10 과 비교가 깨지기 때문 |
| `analysis/predictors/__init__.py` | `KIND_ONLY_KEYS` 에 `gating_residual` · `gating_anchor` 추가 |
| `evals/.../vit_encoder_predictor_concat_ar_nonsquare.py` | ① unexpected-key 가드 (위 §3) ② `ctx_encoder` 빌드·group 공유 ③ `ctx_state` forward 분기 (문맥 = online encoder window z, 되먹임 = predictor state, 출력 = `render(state)`) |
| `evals/action_anticipation_frozen/models.py` | **호스트 RAM OOM 수정.** `shared["ckpt"]` (릴리즈 체크포인트 9.66 GB) 를 run 내내 붙들고 있어 rank 6 개 = 58 GB 였다 → `shared.pop("ckpt")` + `gc.collect()` |
| `z_vll7_training/eval_ek100.sh` | `NW` 6 → 4 (속도 동일, data time 350 → 94 ms), 메모리 가드 상수 75 → **150 MB/clip** (실측 OOM 으로 재보정) |
| `cvpr/registry/models.yaml` | `vith_ek100_g5` 엔트리 |

테스트: `z_vll7_training/tests/gating_residual_cpu_test.py` (21/21) · `z_vll7_training/tests/ek100_ctx_state_cpu_test.py` (7/7). 둘 다 CPU, GPU 불필요.

## 재현

```bash
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
GPUS=6 bash z_vll7_training/eval_ek100.sh vith_ek100_g5
# 결과: cvpr/results/eval/ek100/action_anticipation_frozen/ek100_vith_ek100_g5/val_metrics__<팔>.jsonl
# 배선 검사 (GPU 불필요)
python z_vll7_training/tests/gating_residual_cpu_test.py
python z_vll7_training/tests/ek100_ctx_state_cpu_test.py
```

앞선 4 팔 run 은 `GROUP4_CAP16METRIC_2026-10-18.md`.
