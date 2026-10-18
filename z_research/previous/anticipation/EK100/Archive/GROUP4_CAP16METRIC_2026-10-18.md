# EK100 action anticipation — predictor 4종 동시 비교 (cap16metric, 2026-10-18)

frozen encoder 하나 위에 predictor 4개를 **동시에** 올리고 attentive probe 를 20 epoch 학습한 뒤 val.
`val_metrics__<arm>.jsonl` 에서 다시 읽어 확인했다 (문서 수치 아님).

## 결과 — mean class recall@5, head 3개 중 best

| arm | predictor 체크포인트 | ep5 | ep10 | ep15 | **ep20 action** | verb | noun |
|---|---|---:|---:|---:|---:|---:|---:|
| `release` | (릴리즈) | 6.42 | 20.73 | 29.64 | **33.11** | 48.73 | 49.46 |
| `release_vobftg` | `z_training/runs/vobench_release_postft_g/latest.pt` | 5.88 | 20.86 | 30.56 | **32.75** | 49.47 | 48.61 |
| `ar_vast_ep70` | `z_training/vast/ar_state/latest.pt` | 5.81 | 19.29 | 30.24 | **33.10** | 48.88 | 49.01 |
| `ar_vast_vobftg` | `z_training/runs/vobench_vast_ar_state_postft_g/latest.pt` | 5.70 | 19.68 | 30.07 | **32.75** | 47.01 | 47.00 |

**네 arm 의 폭은 0.36 pt 다 — 구분되지 않는다.** post-FT 도 ar(vast) 도 EK100 anticipation 을 올리지 않는다.
⚠️ 단서: 이건 **동일 encoder** 위의 probe 성능이라 predictor 가 내놓는 미래 표현의 차이만 잰다.
probe (AttentivePooler, depth 2) 가 네 표현 모두에서 비슷한 지점까지 학습된 것일 수 있으므로
"네 predictor 가 같다" 가 아니라 "**이 probe 로는 갈리지 않는다**" 가 맞는 문장이다.

## lr — head 3개 (ep20 action, release 기준)

| head | lr | 결과 |
|---|---|---:|
| 0 | 3e-4 (batch 48 검증값) | 29.08 |
| 1 | 6e-4 + warmup 1 (선형) | 32.77 |
| 2 | **1.2e-3 + warmup 2 (2× 선형)** | **33.11** |

global batch 를 48 → 96 으로 키우면 **lr 선형 스케일링이 필요하다** (3e-4 그대로 쓰면 −4.0 pt).
head 2 가 격자 끝에서 이겼으므로 최적이 더 위에 있을 수 있다 (안 쟀다).

## global batch — 측정으로 갈린 것

| | ipe | 20 epoch step | 분/epoch | ep5 | ep10 | ep15 |
|---|---:|---:|---:|---:|---:|---:|
| 기준 `ek100_rel_cap16metric_32f_e20_g2` (global 48) | 238 | 4,760 | — | 10.25 | 23.62 | 30.52 |
| 이 실행 (global 96) | 119 | 2,380 | 14.5 | 6.42 | 20.73 | **29.64~30.56** |

- **step 절반으로 같은 자리에 도착한다.** sample 당 격차가 ep5 0.63배 → ep10 0.88배 → ep15 ~1.00배로 닫혔다
- 전력 208–410 W → **505–536 W**, SM 40–100 % → **100 % 고정** (batch 48 → 96)
- ⚠️ **batch 126 (6 × 21) 은 나빴다** (ep5 2.32). 96 과 126 사이에 선이 있다 — 추론이 아니라 지점마다 다르다
- ⚠️ 중간에 "step/분 으로는 batch 48 이 1.5배 빠르다" 고 적었던 것은 **틀렸다** (step 당 동률을 가정했는데 실제로는 2배였다).
  같은 정확도까지의 wall-clock 은 batch 96 이 더 짧다

## 설정

- `GPUS=6 bash z_vll7_training/eval_ek100.sh vith_ek100_g4` · `HEADS=cap16b96` · `BS=16` (global 96) · `NW=6`
- `val_every 5` · `ckpt_every 5` · `classifier.use_activation_checkpointing: false`
- `anticipation_point_mode=released` (기준선 규약. `paper` 로 재면 action 35.34 → 14.14)
- train `EPIC_100_train_cap16metric_seed0.csv` (11,470 행 / 470 편) · val `EPIC_100_validation_cap16metric_seed0.csv` (9,296 행 / 138 편)
- 1,114 action / 73 verb / 201 noun. action chance ≈ 0.45 %
- ⚠️ full train CSV (67,217 행) 기준선은 action 35.34 다. 여기 수치와 **직접 비교 금지** (train 이 1/5.9)

## 공유 encoder 가 정확한가

`z_vll7_training/tests/ek100_group_equiv.py` (CPU, 모델 없이 몇 초):
1. **memo 동치** — 공유 경로 출력이 단독 실행과 `torch.equal` 비트 동일
2. **가중치 비오염** — arm 마다 predictor state_dict 의 sha256 이 단독 빌드와 일치, `ek_kind` 도 일치

(2) 는 실제 버그를 잡았다 — `init_module` 이 `predictor_checkpoint` 로드 후 `checkpoint[prd_ckp_key] = predictor.state_dict()`
로 공유 dict 를 덮어써서, 뒤에 오는 릴리즈 arm 이 **앞 arm 의 가중치**를 읽을 뻔했다. 일반 로드 루프를 `else:` 로 옮겨 고쳤다.

## 비용

4 arm × 3 lr = 12 head 를 **5 h 25 min** (02:17:33 → 07:42:15). 순차 추정 22.8 h 대비 4.2배.
기여: encoder 공유 · `val_every 5` · `ckpt_every 5` · activation checkpointing off (4.2 %) · global batch 96.
VRAM 34.7 / 96 GB. 호스트 RAM 가드 (`clips = BS × NW × 2 × GPUS`, 75 MB/clip) 통과.

## 재현

```bash
GPUS=6 bash z_vll7_training/eval_ek100.sh vith_ek100_g4
# 산출물 cvpr/results/eval/ek100/action_anticipation_frozen/ek100_vith_ek100_g4/
#   val_metrics__{release,release_vobftg,ar_vast_ep70,ar_vast_vobftg}.jsonl · _resolved.yaml
python z_vll7_training/tests/ek100_group_equiv.py     # 공유 encoder 동치 검사
```
