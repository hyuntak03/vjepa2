# Residual gating — 구조 · 학습 · 결과 (2026-10-17 ~ 10-18, vast)

> 한 줄 요약: **AR 에 gating 하나만 얹은 판 (`natural_ar_vw_state_gres`) 은 예측 오차와 margin 은 좋아졌지만,
> 판별 정확도는 네 벤치마크 모두 gating 없는 AR 이하다.** 가장 크게 떨어진 것은 VOBench late 영속 (93–96 → 60).
> 출력이 쪼그라드는 것 (LN 없음) 은 측정됐지만 원인은 아니었다 (채점 때 LN 을 씌워도 그대로).

수치는 전부 산출물 (`summary.json` · `per_block.json` · `per_window.json`) 에서 다시 계산했다.
측정과 해석을 나눠 적는다 — **[측정]** / **[해석]**.

---

## 1. 왜 gating 인가 (출발점)

진단 문서 `z_research/VOBench_v1/Archive/DIAG_CTXT_VIDEO_VS_AR_2026-10-17.md` §7 의 측정:

| 걸음 0 배경 오차 (L1, 배경 247 토큰) | late | visible |
|---|---:|---:|
| 블록 복사 (마지막 문맥 블록 LN(h)) | **0.366** | **0.397** |
| AR | 0.472 | 0.460 |
| ctxt (영상 문맥) | 0.569 | 0.548 |

- 배경은 그냥 복사하는 게 두 모델보다 정확하고, 물체 자리는 복사가 가장 나쁘다 (걸음 7 0.84).
- 그래서 "배경은 복사, 움직이는 물체는 predictor" 로 나누는 gate 를 출력에 거는 것이 목표였다.

## 2. 구조

### 2-1. 출력식

```
ŷ_{k,i} = g_{k,i} · A_i + f_{k,i}          (기본 출력은 LN(proj(s)))
g = σ(wᵀ s + b) / σ(2)    — 토큰마다, 0 ~ 1.135     s = predictor_norm 출력 (384)
f = predictor_proj(s)
```

- 시작 시 `w = 0 · b = 2 · predictor_proj = 0` → **ŷ = A (복사) 정확히**. 추가 파라미터는 `keep_gate` 385 개.
- 출력 LN 은 뺐다 (`LN(g·A) = A` 라 초기에는 g 가 지워진다는 이유 — §5 에서 다시 본다).
- gate 는 출력에만 걸린다. rollout 되먹임 (상태 `s`) 은 그대로다.
- 끄면 (`gating_residual: false`, 기본) 파라미터 · 출력이 기존과 비트 단위로 같다 (CPU 검사 1 · 6).

### 2-2. anchor A — 무엇을 복사하나

| 판 | anchor | 비고 |
|---|---|---|
| ariel 첫 구현 (7417f68) | LN(문맥 **encoder** z 의 마지막 블록) | **[측정]** 쓸 수 없다 — 아래 표 |
| vast 구현 (cfd9f6d 부터) | LN(**target_encoder**(마지막 문맥 블록)) — 타깃 공간 | ariel 도 이후 target 으로 고쳤다 (사용자) |
| kind=ar (416f284) | TF 위치 b = 입력 블록 b · rollout 전 걸음 = 마지막 문맥 블록 | "그 자리의 마지막 관측 블록" |
| `gating_anchor: prev` (2910622, 미학습) | TF 같음 · rollout = LN(직전 걸음 예측) | TF 와 rollout 규칙을 맞춘다 |

**[측정] anchor 후보와 다음 블록 정답 LN(h) 의 L1** (`gating_anchor_gap.py`, VOBench 각 60 clip):

| 후보 | visible 배경 (걸음 0 · 3 · 7) | late 배경 (걸음 0 · 3 · 7) |
|---|---|---|
| LN(z) 마지막 문맥 블록 (encoder, 영상) | 0.821 · 0.827 · 0.834 | 0.819 · 0.842 · 0.849 |
| LN(h) 마지막 문맥 블록 (target, 블록) | **0.374** · 0.445 · 0.487 | **0.454** · 0.590 · 0.599 |

- LN(z) 와 LN(h) 는 같은 블록끼리도 배경 L1 0.81 — 두 encoder 공간이 거의 무관하다. 영상 z 를 anchor 로 쓰면 학습 시작점이 학습된 ctxt (0.55) 보다도 나쁘다.

### 2-3. 조립 축 (전부 config 키, `SET` 한 줄로 ablation)

| 키 | 값 | 뜻 |
|---|---|---|
| `model.predictor.kind` | `ar` \| `ctx_state` | ar = 문맥 · 미래 블록별 LN(target_encoder), 전부 block-causal, TF 전 위치 / ctx_state = 문맥 prefix 양방향, TF 예측 구간만 |
| `model.predictor.gating_residual` | `true` \| `false` | 위 출력식 |
| `model.predictor.gating_anchor` | `last_obs` \| `prev` | (ar) rollout anchor |
| `model.predictor.gating_out_ln` | `true` \| `false` | (ar) 출력 `LN(g·A + f)` — 2026-10-18 추가, 미학습 |
| `model.predictor.ctx_encoder` | `encoder` \| `target_encoder` | (ctx_state) 문맥 encoder. `context_encoder_key` 를 덮는다 |
| `model.predictor.ctx_input` | `video` \| `block` | (ctx_state) 문맥 인코딩 |
| `model.predictor.ctx_target` | `window` \| `block` | (ctx_state) 타깃 |
| `loss.tf_span` | `all` \| `predict` | (ar) TF 손실 범위 |

- 이 키들을 `predictor` 블록 안에 둔 이유: 평가 프로토콜이 `context_encoder_key: encoder` 를 고정해서 덮는다.
  `predictor` 아래 키는 `eval_bench.sh` · `chain_bench.sh` · 레지스트리가 run config 에서 그대로 넘긴다.
- 채점기 (`analysis/intphys2/model.py`) 는 ctx_state 영상 문맥 + gating 일 때 `_AnchorCtxEncoder` 로 타깃 블록 anchor 를 따로 만들고,
  ar 은 `_ArScorer → rollout_kv` 안에서 처리한다.

## 3. 학습한 것

| run | 구성 | 학습 | 상태 |
|---|---|---|---|
| `gres_smoke` | ctx_state · target_encoder × block · gating | 1 epoch | 스모크 (경로 검증용) |
| `natural_ar_vw_state_ctxt_block_gres` | ctx_state · **target_encoder × block** · 문맥 **양방향** · TF 예측 구간만 · gating · ctxt_video 샘플링 (문맥 4–32 장) | e14 / 70 | **중단** (사용자: AR 위에 gating 만 얹은 판으로) |
| **`natural_ar_vw_state_gres`** | **AR (natural_ar_vw_state) + gating** — 병합 config 가 AR 과 `gating_residual` 하나만 다름 | **70 / 70** | 완료 (10-18 12:22 KST) |
| `natural_ar_vw_state_gres_prev` | gres + `gating_anchor: prev` (gres 와 키 하나만 다름) | — | config 만, 미학습 |

- `natural_ar_vw_state_gres` 는 AR 과 **로그에 찍힌 batch 순서가 줄마다 같다** (seed · 샘플링 동일) → 같은 clip 을 같은 순서로 봤다.
  본 clip 약 1,042 만 (rank 당 평균 B 63.3 × 8 GPU × 20,580 step). 데이터 · 창 (C · K 2–32 장) · lr · 70 epoch 동일.

## 4. 결과

### 4-1. 최종 — AR + gating vs 기준 모델

| | IntPhys1 A.8 w16 / w32 | VOBench late P / R | late 영속 P / R | VOBench gravity P / R | IntPhys2 | InfLevel macro |
|---|---:|---:|---:|---:|---:|---:|
| AR (`natural_ar_vw_state`) | 90.00 / 88.33 | 74.1 / 77.8 | 93.1 / 95.8 | 20.4 / 8.3 | 51.8 | 58.4 |
| AR + op | **92.78 / 91.67** | **76.9 / 80.6** | **97.2 / 100** | 21.3 / 8.3 | 51.8 | **58.7** |
| ctxt (영상 문맥) | 76.11 / 70.00 | 52.3 / 54.6 | 52.8 / 54.2 | **33.3 / 20.4** | — | (채점 중) |
| **AR + gating** | 87.22 / 84.44 | 58.3 / 60.6 | 59.7 / 59.7 | 18.5 / 5.6 | **52.4** | 58.3 |

세부 (AR + gating):
- VOBench late — primitive 영속 · 모양 · 색 59.7 · 56.9 · 58.3 / realistic 59.7 · 55.6 · 66.7
- gravity — ledge 15.3 / 2.8, arc 25.0 / 11.1 (AR 12.5 / 1.4, 36.1 / 22.2)
- IntPhys1 (O1 · O2 · O3) 83.3 · 93.3 · 85.0 (AR 83.3 · 98.3 · 88.3)
- IntPhys2 (w48, Overall 최고 창) 영속 · 불변 · 연속 · 고체 53.3 · 54.2 · 51.7 · 50.7
- InfLevel (Garrido 규칙, skip10_w16) continuity · solidity · gravity 76.1 · 49.3 · 49.4

**[측정]** gating 하나만 다른 대조에서 IntPhys1 −2.8, VOBench late −16 / −17 (영속 −33 / −36), gravity −1.9 / −2.7.
IntPhys2 (+0.6) · InfLevel (−0.1) 은 둘 다 우연 근처라 차이가 없다.

### 4-2. 중단한 블록 문맥 판 (참고, e11 스냅샷)

`natural_ar_vw_state_ctxt_block_gres` e11: IntPhys1 85.56 / 85.00 · VOBench late 57.9 / 63.4 · gravity 18.5 / 4.6.
AR 과 네 가지 (문맥 attention · TF 범위 · 샘플링 · gating) 가 같이 달라 원인을 가를 수 없어 중단했다.

### 4-3. 학습 중 val (IntPhys1 dev 60 쌍)

| | e0 | e10 | e20 | e40 | e70 | 최종 surprise (pos) | 최종 margin |
|---|---:|---:|---:|---:|---:|---:|---:|
| AR | 50.0 | 80.0 | — | — | **86.7** | 0.493 | +0.0105 |
| AR + gating | **76.7** | 76.7 | 81.7 | 80.0 | 80.0 | **0.406** | **+0.0219** |

- **[측정] epoch 0 val 은 학습 전에 잰 값이다.** 두 gating run 의 e0 가 소수 넷째 자리까지 같다 (76.7 %, 0.4471, +0.0378) —
  초기 출력이 정확히 복사라서. 즉 **순수 복사만으로 IntPhys1 val 76.7 %** 이고, gating 판은 70 epoch 동안 그 위로 두 쌍 (80.0) 올랐다.
- margin 은 복사 (0.038) 가 가장 크고 학습하며 줄었다 (0.022). 예측 오차는 줄지만 가능 / 불가능 차이는 커지지 않는다.

## 5. 진단

### 5-1. gate 값

최종 (rollout): 평균 0.21 · 움직이는 칸 0.155 / 정지 칸 0.331 · 첫 걸음 0.277 / 마지막 0.176 · >0.95 비율 0.01.
epoch 별 평균 1.0 → 0.50 (e1) → 0.24 (e5) → 0.21 (e70), e5 이후 평평.

- **[측정]** 방향은 의도대로 (움직임 < 정지, 뒤 걸음 < 첫 걸음) 지만 **정지 칸도 0.33** — "배경은 복사 (≈ 1)" 가 아니다.
- **[해석]** gate 값만으로 실효 복사량을 말할 수 없다. f 도 attention 으로 마지막 블록을 보고 A 와 닮은 값을 낼 수 있기 때문이다.
  실효 복사량은 칸별 `a = |ŷ − A|`, `b = |h − A|`, `c = |ŷ − h|` 로 재야 한다 (a/b = 바꾼 양 / 바뀌어야 할 양, c/b < 1 이면 복사보다 낫다). **미측정.**
- ⚠️ 모니터의 ✓ 는 대소 비교만 본다 — 위 절대 수준은 판정하지 않는다.

### 5-2. 출력 크기 (`gating_out_scale.py`, VOBench 80 clip, rollout 걸음 1 · 3 · 5 · 8)

| | 걸음 1 | 3 | 5 | 8 |
|---|---:|---:|---:|---:|
| 출력 std (토큰별 D 방향, 타깃 = 1) AR | 1.000 | 1.000 | 1.000 | 1.000 |
| 〃 AR + gating | **0.875** | **0.816** | **0.783** | **0.745** |
| L1 오차 AR | 0.457 | 0.478 | 0.491 | 0.507 |
| L1 오차 AR + gating | 0.330 | 0.371 | 0.394 | 0.424 |
| L1 오차 복사 | 0.373 | 0.427 | 0.455 | 0.485 |

- **[측정]** LN 이 없으니 불확실할수록 출력이 쪼그라든다 (중앙값 쪽). 그 덕에 L1 은 AR · 복사보다 낮다.

### 5-3. 채점 때만 출력 LN 을 씌우면 (가설 검사)

| AR + gating 최종 | IntPhys1 A.8 w16 | VOBench late P / R | late 영속 P / R |
|---|---:|---:|---:|
| 그대로 | 87.22 | 58.3 / 60.6 | 59.7 / 59.7 |
| 채점 때 LN | 86.67 | 57.4 / 61.1 | 59.7 / 61.1 |

- **[측정] 거의 그대로.** "쪼그라든 출력이 판별을 망쳤다" 는 가설은 **지지되지 않는다.** 문제는 크기가 아니라 예측의 내용 쪽이다.
- 한계: 채점 때 LN 은 LN 을 걸고 학습한 것과 다르다.

## 6. 해석과 남은 가설

- **[해석] TF / rollout anchor 어긋남.** 손실 대부분은 TF (전 위치) 에서 나오고 거기서 anchor 는 0.23 s 전 블록 — gate 는 "직전 블록 복사 + 보정" 으로 배운다.
  rollout 에서는 걸음마다 낡아지는 고정 anchor 에 그 gate 를 쓴다 (gate k1 0.28 > kR 0.18 와 맞는다). → `gating_anchor: prev` 로 검사 (gres 와 키 하나 차이).
- **[해석] 먼 배경 신호 소거.** AR 의 VOBench late 정답 중 71 % 는 먼 배경 토큰에서 나왔다 (진단 §7-3). 배경을 복사하면 그 신호가 지워져 late 가 떨어질 수 있다.
  → 물체 자리 / 먼 배경 분리 채점 (`obj_region_surprise.py`) 으로 판정 가능. **미측정.** 단, AR 의 물체 자리 late 도 우연 근처였으므로 gating 이 물체 영속을 키우지 못한 것은 분명하다.
- **[측정] 출력 LN 부재는 원인이 아니다** (§5-3).
- 움직임 (gravity · 미래 위치) 은 개선되지 않았다. ctxt 의 움직임 장점 (gravity 33.3 / 20.4, 걸음 7 위치 오차 2.87) 은 gating 으로 얻어지지 않는다.

## 7. 다음 후보 (미실행)

| 후보 | gres 와의 차이 | 묻는 것 |
|---|---|---|
| `natural_ar_vw_state_gres_prev` | `gating_anchor: prev` | TF / rollout anchor 규칙을 맞추면 나아지나 |
| + `gating_out_ln: true` | 키 하나 더 | (prev 판에서) anchor 사슬의 크기 떠다님 차단 — 원인 분리를 위해 prev 다음에 따로 |
| a · b · c 모니터 | 코드 (학습 로그) | 실효 복사량 — 다음 run 부터, 지금 run 은 체크포인트로 |
| 물체 자리 / 먼 배경 채점 | 채점만 | late 하락이 배경 신호 소거인가 |

## 8. 함정 기록

- **VOBench gravity 는 block 정의** (가능 미래가 그 block 의 모든 불가능 미래보다 낮아야 정답, 우연 37 %). `per_block.json` 의 `acc` (쌍 정확도) 를 쓰면 arc 가 85 처럼 부풀려진다 — 2026-10-17 에 한 번 잘못 보고했다.
- **실행 중인 bash 스크립트를 고치지 말 것.** 10-18 에 IntPhys2 채점이 도는 동안 `eval_bench.sh` 를 고쳤다가 마지막 요약이 `unexpected EOF` 로 실패했다 (bash 는 스크립트를 읽어 가며 실행한다). 채점 산출물은 멀쩡했고 `eval_bench_summary.py` 로 요약만 다시 만들었다.
- in-loop val 의 epoch 0 은 학습 전 값이다 (gating 판에서는 = 순수 복사 기준선).
- InfLevel 시트 값은 `garrido_rescore.collect_inflevel` (property 별 최고 축약 → 3 property macro, 두 창을 한 탐색 공간으로) — AR + op 58.7 (75.7 / 50.0 / 50.3) 을 정확히 재현했다.

## 재현

```bash
cd /workspace/world_model/vjepa2
# CPU 검사 (13 항목, 모델 없이 몇 초)
CUDA_VISIBLE_DEVICES= /venv/vjepa/bin/python z_training/tests/gating_ctx_block_cpu_test.py

# 학습
GPUS=8 bash z_training/tmux_train.sh natural_ar_vw_state_gres          # configs/training/natural_ar_vw_state_gres.yaml (맨 위 조립 표)
GPUS=8 bash z_training/tmux_train.sh natural_ar_vw_state_gres_prev     # 미학습

# 채점
WINDOWS_I2='' bash z_training/eval_bench.sh natural_ar_vw_state_gres latest.pt        # IntPhys1 A.8 (w32 · w16)
bash z_training/eval_bench.sh natural_ar_vw_state_gres latest.pt                      # + IntPhys2 (w16 · 32 · 48, 이미 있는 칸은 건너뜀)
for ds in vobench_v1_late vobench_v1_gravity; do RESULTS_ROOT=legacy GPUS=8 bash cvpr/eval/testbed/run.sh $ds vast_ar_vw_state_gres; done
NOWAIT=1 RUNS=natural_ar_vw_state_gres BENCHES="inflevel_continuity inflevel_solidity inflevel_gravity" bash z_training/chain_bench.sh

# 진단
CUDA_VISIBLE_DEVICES=7 /venv/vjepa/bin/python z_research/scripts/analysis/gating_anchor_gap.py    # §2-2
CUDA_VISIBLE_DEVICES=7 /venv/vjepa/bin/python z_research/scripts/analysis/gating_out_scale.py     # §5-2
# §5-3 채점 때 LN: SET 에 model.predictor.gating_out_ln=True 를 더해 run.sh intphys1_sliding / testbed vobench_v1_late
```

- 표 계산: gravity = `data_csv/vobench_v1_gravity/index.csv` + `per_block.json["per_video_surprise"]` (block 정의) ·
  VOBench late = `per_block.json` 쌍 정확도 · IntPhys1 유형별 = `summary.json["surprise"]["skip2_w16/avg"]["by_block_type"]` ·
  IntPhys2 유형별 = Overall 최고 창의 `breakdown` condition 합산 · InfLevel = `garrido_rescore.collect_inflevel(<run>/eval, "latest")`.
- 시트: Overall 24 행 · Intuitive Physics 22 행 (`AR predictor w/ residual gating`).
- 코드: `src/models/rollout_predictor.py` (`render`, `_ar_out`, `rollout_kv`), `app/vjepa_frozen/{train,ar,utils}.py`, `analysis/intphys2/model.py` (`_CtxStateScorer`, `_AnchorCtxEncoder`).
  커밋 cfd9f6d · 57b3a56 · 416f284 · 2910622 (vast).
