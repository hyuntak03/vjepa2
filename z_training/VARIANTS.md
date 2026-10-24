> 📥 **vast 브랜치에서 가져온 문서 (2026-10-24, `origin/vast` c05c259 의 `z_training/VARIANTS.md`, 원문 그대로).**
> 본문의 run 이름 · `/workspace/...` 경로 · 시트 언급은 vast 노드 기준이다 — 이 노드 (vll5) 에 그 산출물이 없을 수 있고, 수치는 그쪽 산출물에서 계산된 것이다.

# 학습 variant 전체 기록 — 철학 · 구조 · run · 결과 (2026-10-18 기준)

> **이 파일 하나로 "지금까지 무엇을 왜 학습했고 어떻게 됐나" 를 추적한다.** 새 variant 를 만들면 여기 §5 에 카드 하나, §4 표는 도구로 다시 뽑는다.
> 표의 숫자는 손으로 옮기지 않는다 — `python z_training/tools/variants_table.py` 가 run 산출물에서 다시 계산한다.

---

## 0. 30 초 요약

- **목표:** 실제 영상만 보고, **미래 예측 목적**으로 intuitive physics 를 배워 그럴듯한 미래를 내는 predictor.
- **고정한 것:** V-JEPA 2 ViT-H encoder 는 **얼린다** (학습 안 함). 바꾸는 것은 predictor (22 M) 뿐.
- **지금 가장 좋은 것:** AR (블록 문맥) 계열. `natural_ar_vw_state` IntPhys1 90.0, `+op` 92.8.
- **가장 큰 문제:** 예측이 **문맥의 평균**으로 무너진다 (움직임 0.3–0.5 배, rollout ≈ 복사). → [`docs/PROBLEMS_AND_DIRECTION_2026-10-18.md`](docs/PROBLEMS_AND_DIRECTION_2026-10-18.md)
- **지금 도는 것:** `natural_ar_vw_state_vidctx` (영상 문맥 · 영상 정답 · TF 문맥만), 10-18 14:40 시작.

---

## 1. 철학 — 왜 이렇게 실험하나

| 원칙 | 뜻 | 어디서 왔나 |
|---|---|---|
| **실제 영상에서 자연스럽게** | 합성 데이터 · 정답 라벨 · 사람이 짠 구조로 물리를 "가르치지" 않는다. 목적함수와 데이터 분포로 배우게 한다 | 사용자 목표 |
| **단순한 해법 먼저** | 모듈 추가 (codebook · CE · 슬롯) 보다 손실 · 입력 형식 · 학습 신호를 바꾼다 | 사용자 (CE · codebook 은 너무 복잡하다고 기각) |
| **한 번에 하나만 바꾼다** | 새 variant 는 기준 run 과 키 하나 차이가 이상적. 병합 config 를 기준과 diff 해서 확인한다 | gating · gres_prev 에서 실행 |
| **학습 전에 싼 진단** | 10 시간 학습 전에 기존 체크포인트 · 몇 분짜리 측정으로 가설을 거른다 | 메모리 `separate-measured-from-speculation` |
| **측정과 추측을 나눈다** | 문서마다 **[측정]** / **[해석]** | 같음 |
| **벤치마크 점수만 믿지 않는다** | 점수에 배경 · 복사가 섞인다. 물체 자리 채점 · 복사 기준선 · 전이 지표를 같이 본다 | 진단 (AR late 정답 71 % 가 먼 배경) |
| **도메인 안 상승은 일반화가 아니다** | 학습 분포와 같은 벤치마크가 오르는 건 증거가 아니다. 외부 벤치마크 (IntPhys1 · GRASP · InfLevel) 로 판정 | RESULTS_2026-09-19 (post-FT 세 번 모두 도메인 안에서만) |

---

## 2. 공통 골격 (모든 variant 가 공유)

```
릴리즈 V-JEPA 2 ViT-H (fpc64-256)          ── 얼린다 (no_grad, bf16)
   ├─ encoder        (online)  ─ 문맥 인코딩에 쓰는 판도 있다
   └─ target_encoder (EMA)     ─ 정답 · 블록 인코딩
predictor (ViT, 384 dim, depth 12, 22 M)   ── 이것만 학습 (scratch)

데이터   K710 + SSv2 + HowTo100M 1 %, 4 fps 가변 창 (step 마다 문맥 C · 예측 K 장을 고른다, token budget 950)
손실     L1 (예측 vs affine 없는 LN 한 정답)
학습     70 epoch × 294 step = 20,580 step, lr 5.2e-4 WSD (마지막 7 ep 감소), 8 GPU
채점     IntPhys1 A.8 · IntPhys 2 · VOBench (late · gravity) · GRASP · InfLevel — 정의는 §6
```

- 체크포인트는 predictor 만 (`runs/<run>/latest.pt`, 2 epoch 마다 `e{N}.pt`).
- 학습 진입점 `z_training/tmux_train.sh <config>` → `train.sh` → `app/vjepa_frozen/train.py`.

---

## 3. 구조 — predictor 의 네 가지 형태

### 3-1. oneshot (tube) — 릴리즈 V-JEPA 2 와 같은 규약

```
clip ─► encoder(문맥 토큰만) ─► z ─┐
                                   ├─► predictor (mask token 자리 한 번에) ─► p ─┐
                                   │                                             ├─ L1
clip ─► target_encoder(전체) ─► LN ─► h[미래] ─────────────────────────────────┘
```
- 미래를 **한 번에** 낸다 (rollout 없음). tube 마스크, 16 장 → 64 장 cooldown.
- 대표: `k710_ssv2_how_ep70_cooldown64` IntPhys1 82.2.

### 3-2. kind=ar — 블록 단위 자기회귀 (지금의 주력)

```
clip ─► target_encoder 를 블록(2 장)마다 따로 ─► LN ─► h_0 h_1 … h_{C-1} | h_C … h_{C+K-1}
                                                       문맥              미래 (정답)
predictor: 전부 block-causal (블록 b 는 블록 ≤ b 만 본다)
  TF      : 위치 b 의 입력 h_b → 출력 = 블록 b+1 예측       (전 위치, 문맥 포함)
  rollout : 문맥 C 블록 → 걸음마다 자기 상태 s 를 되먹여 R (≤16) 블록    (ar_feedback=state)
손실 = L1(TF) + L1(rollout)
```
- 입력 · 출력 · 정답이 **같은 공간** (블록별 LN(target_encoder)) → 정밀도 · margin 이 좋다.
- 약점: 마지막 블록 토큰이 자기 2 장만 알아서 시간 통합을 predictor 가 해야 하고, 그걸 **평균** 으로 푼다 (rollout ≈ 복사).

### 3-3. kind=ctx_state — 문맥 prefix + 상태 rollout

```
문맥 C 블록 ─► (ctx_encoder × ctx_input) ─► z ─┐ 문맥끼리 양방향 (prefix)
미래        ─► target_encoder 블록별 LN(h)     ─┤ TF: 예측 구간만 (관측 토큰), 미래는 block-causal
                                                 └ rollout: z 에서 상태 되먹임
정답 = 블록별 LN(h) (ctx_target=block) | 창 전체 LN(target_encoder) (window)
```
- 대표: `natural_ar_vw_state_ctxt_video` (encoder × 영상 문맥, 블록 정답).
- 장점: 영상 문맥은 **움직임을 더 잘 옮긴다** (위치 오차 걸음 7 2.87 vs AR 4.65, gravity 33.3).
- 약점: 영상 토큰과 블록 토큰은 **다른 공간** (같은 블록끼리도 배경 L1 0.81, encoder 를 target_encoder 로 바꿔도 0.815) → 배경 오차 +0.09, margin 1/8, IntPhys1 76.1.

### 3-4. kind=ar · ar_input=video — 영상 문맥 · 영상 정답 (10-18 신설, 학습 중)

```
문맥 pass : target_encoder(문맥 프레임만, 영상)  ─► LN ─► zc   (미래를 안 본다)
전체 pass : target_encoder(클립 전체, 영상)      ─► LN ─► zf   (창 정답 = 릴리즈 V-JEPA 2 정의)
predictor : AR 그대로 (block-causal, 상태 되먹임)
  TF      : 문맥 자리만 — zc[b] → zf 블록 b+1   (zc[b] 는 문맥 끝까지 본 토큰: 의도된 약간의 누수)
  rollout : 예측 구간 전부 (TF 없음) → zf 블록 C..C+R-1
```
- 채점기는 문맥 pass 를 따로 돌린다 (`_CtxPassEncoder` — IntPhys2 하네스가 문맥을 전체 pass 에서 잘라 쓰는 누수를 막는다).

### 3-5. 옵션 (위 구조에 얹는 것)

| 옵션 | 무엇 | 대상 | 결과 한 줄 |
|---|---|---|---|
| `ar_feedback: state` | rollout 때 출력 대신 상태 (384) 를 되먹임 | ar | 주력 기본값 (`output` 보다 val 안정) |
| hint (`hint_cross_dim`) | TF 때 다음 블록 patch 일부를 cross-attn 으로 보여 줌 (학습 전용) | ar | 끊는 순간 손실이 짝과 같아짐 — 효과 없음 |
| `loss.delta_weight` | 시간 변화량 L1 \|Δp − Δh\| | ar | IntPhys1 같음, VOBench late −1.8 |
| `loss.dyn_weight` + `dyn_head` | 움직임 크기 스칼라 예측 손실 | ar | config 만 (이 기계 미실행) |
| `gating_residual` | 출력 = g·A + proj(s), A = 마지막 관측 블록 | ar · ctx_state | 네 벤치마크 모두 AR 이하 (late 영속 93 → 60) |
| `gating_anchor: prev` · `gating_out_ln` | rollout anchor = 직전 예측 / 출력 LN | ar + gating | 구현만, 미학습 |
| `ctx_hole` (+ `recon_head`) | 문맥에 encoding 전 구멍 + 복원 손실 | ctx_state | 복원이 미래 예측과 충돌 (사용자) — 중단 |
| 데이터 `boundary: op320` | 물체 영속 합성 영상을 경계 정렬로 10 % | ar · ctx_state | AR: 정지 영속 ↑, IntPhys1 +2.8 · GRASP +4.1 / ctxt: 효과 없음 |
| 데이터 `fixed: vob` | VOBench 가능 arm 10 % | ar | VOBench (도메인 안) 만 ↑, 외부 벤치 그대로 |
| `loss.tf_span: predict` | TF 손실을 예측 구간만 | ar | 옵션만 |

---

## 4. 전체 run 표 [측정]

`python z_training/tools/variants_table.py` 출력 (2026-10-18 14:50, `_aborted_*` · `gres_smoke` 제외). — 는 미측정.

| run | kind | 구성 | 창 (문맥 · 예측 장) | epoch | 완료 | 본 clip (M) | val 최종 / 최고 | IntPhys1 A.8 | VOBench late P / R | gravity P / R | IntPhys2 | GRASP | InfLevel |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `k710_ssv2_how_ep70` | oneshot | tube | 고정 창 | 63/63 | ✅ | — | 50.0 / 53.3 | — | — | — | — | — | — |
| `k710_ssv2_how_ep70_cooldown32` | oneshot | tube + 32 장 cooldown | 고정 창 | 7/7 | ✅ | — | 66.7 / 71.7 | — | — | — | — | — | — |
| `k710_ssv2_how_ep70_cooldown64` | oneshot | tube + 64 장 cooldown | 고정 창 | 7/7 | ✅ | — | — / — | 82.2 | 56.0 / 65.7 | — | 54.7 | — | — |
| `natural_tube_k710` | oneshot | tube, K710 만 | 고정 창 | 40/40 | ✅ | — | 50.0 / 51.7 | — | — | — | — | — | — |
| `natural_ctx_state` | ctx_state | encoder×video / 창 정답 · rollout 만 | 고정 창 | 40/40 | ✅ | — | 68.3 / 68.3 | — | — | — | — | — | — |
| `natural_ar_vw` | ar | 출력 되먹임 | 2–32 · 2–32 | 28/70 | 중단 | 3.91 | 78.3 / 83.3 | — | — | — | — | — | — |
| `natural_ar_vw_hint` | ar | 출력 되먹임 · hint | 2–32 · 2–32 | 70/70 | ✅ | 10.42 | 78.3 / 81.7 | 91.7 | 75.0 / 76.9 | — | 52.4 | — | — |
| **`natural_ar_vw_state`** | ar | **상태 되먹임 (기준)** | 2–32 · 2–32 | 70/70 | ✅ | 10.42 | 81.7 / 86.7 | **90.0** | 74.1 / 77.8 | 20.4 / 8.3 | 51.8 | 67.2 | 58.4 |
| `natural_ar_vw_state_hint` | ar | + hint | 2–32 · 2–32 | 70/70 | ✅ | 10.42 | 81.7 / 85.0 | 91.1 | 70.4 / 76.9 | — | 51.4 | — | — |
| `natural_ar_vw_state_delta` | ar | + 변화량 L1 | 2–32 · 2–32 | 70/70 | ✅ | 10.42 | 81.7 / 83.3 | 90.6 | 74.1 / 75.0 | — | 51.6 | — | — |
| `natural_ar_vw_state_vob` | ar | + VOBench 10 % | 2–32 · 2–32 | 70/70 | ✅ | 9.81 | 80.0 / 83.3 | 91.7 | (held-out, RESULTS_2026-10-15) | — | 52.2 | 68.9 | 58.2 |
| **`natural_ar_vw_state_op`** | ar | **+ op320 10 %** | 2–32 · 2–32 | 70/70 | ✅ | 10.41 | 81.7 / 85.0 | **92.8** | **76.9 / 80.6** | 21.3 / 8.3 | 51.8 | **71.3** | **58.7** |
| `natural_ar_vw_state_gres` | ar | + gating | 2–32 · 2–32 | 70/70 | ✅ | 10.42 | 80.0 / 81.7 | 87.2 | 58.3 / 60.6 | 18.5 / 5.6 | 52.4 | — | 58.3 |
| `natural_ar_vw_state_ctxt_video` | ctx_state | encoder×video / 블록 정답 | 4–32 · 4–32 | 70/70 | ✅ | 6.86 | 63.3 / 63.3 | 76.1 | 52.3 / 54.6 | **33.3 / 20.4** | — | — | 56.2 |
| `natural_ar_vw_state_ctxt_video_op` | ctx_state | 위 + op320 10 % | 4–32 · 4–32 | 50/70 | 중단 | 4.96 | 61.7 / 65.0 | (e20 74.4) | (e20 51.9 / 54.6) | — | — | — | — |
| `natural_ar_vw_state_ctxt_video_c4b_lasthole` | ctx_state | 문맥 ≥ 8 장 + 마지막 블록 구멍 | 8–32 · 4–32 | 6/70 | 중단 | 0.40 | 58.3 / 58.3 | — | — | — | — | — | — |
| `natural_ar_vw_state_ctxt_block_gres` | ctx_state | target×block · 양방향 · gating | 4–32 · 4–32 | 14/70 | 중단 | 1.24 | 81.7 / 81.7 | (e11 85.6) | (e11 57.9 / 63.4) | (e11 18.5 / 4.6) | — | — | — |
| `natural_ar_vw_state_vidctx` | ar | 영상 문맥 · 영상 정답 · TF 문맥만 | 4–32 · 4–32 | 학습 중 | — | — | — | — | — | — | — | — | — |

참고 (같은 채점 정의):
- 릴리즈 V-JEPA 2 ViT-H: IntPhys1 88.9 · VOBench late 57.9 / 65.7 · gravity 0 / 0 · IntPhys2 54.3 · GRASP 70.3 · InfLevel 58.3.
- 순수 복사 (마지막 문맥 블록 그대로): IntPhys1 val 76.7 (gating run 의 epoch 0 val).
- 우연: 50 (gravity 37).
- 다른 서버 (ariel) 의 `AR predictor w/ pixel diff loss` 는 시트에만 있다 (VOBench 73.1 / 77.3 · gravity 16.7 / 9.3 · IntPhys1 91.1 · IntPhys2 50.8). 이 기계에 run 없음.

---

## 5. variant 카드 (시간 순)

각 카드: **무엇** · **왜** · **결과** · **문서**. ✅ 완료 / ⏸ 중단 / ▶ 진행 중 / ○ 미실행.

### 5-1. 1 기 — 릴리즈 predictor post-FT (ariel/vll5, 2026-09) ✅
- **무엇:** 릴리즈 predictor 를 v11 · IntPhys1 · Predictor_v1 로 이어 학습.
- **결과:** 셋 다 **학습한 도메인 안에서만** 오른다 (v11 75.83 → 91.35 인데 IntPhys1 88.89 → 77.22).
- **교훈:** 합성 · 벤치마크 데이터로 post-FT 는 일반화 증거가 아니다.
- 문서: [`docs/RESULTS_2026-09-19.md`](docs/RESULTS_2026-09-19.md)

### 5-2. tube 기준선 (Vast, 09-29 ~ 10-01) ✅
- **무엇:** 릴리즈 레시피 (tube 마스크, oneshot) 로 predictor scratch. K710 + SSv2 + HowTo.
- **결과:** cooldown 이 결정적 (69.4 → 32 장 73.3 → 64 장 **82.2**). 같은 때 ctx_state rollout 만 학습은 **우연 수준**.
- 문서: [`docs/RESULTS_2026-10-01.md`](docs/RESULTS_2026-10-01.md)

### 5-3. kind=ar 가변 창 · 힌트 · 상태 되먹임 (10-11 ~ 12) ✅
- **무엇:** V-JEPA 2-AC 를 action 없이 — 블록별 인코딩 + block-causal AR, 가변 창, 상태 되먹임, 힌트 cross-attn.
- **왜:** oneshot 은 한 번에 미래를 내 "굴리는" 능력을 안 배운다. AC 방식이 rollout 을 직접 학습한다.
- **결과:** `natural_ar_vw_state` IntPhys1 **90.0** (릴리즈 88.9 를 처음 넘음). 힌트는 효과 없음.
- 문서: [`docs/RESULTS_2026-10-12.md`](docs/RESULTS_2026-10-12.md)

### 5-4. AR 후속 — 차이 L1 · VOBench 학습 · GRASP · InfLevel (10-12 ~ 15) ✅
- **delta:** 변화량 L1 — IntPhys1 같음, VOBench late −1.8.
- **vob:** VOBench 가능 arm 10 % — held-out VOBench 는 크게 ↑ (gravity 75 / 58) 인데 외부 벤치 그대로 → in-distribution.
- **진단:** AR 의 실효 문맥은 마지막 1–2 블록. late 가림에서 가려진 마지막 블록만 본다.
- 문서: [`docs/RESULTS_2026-10-15.md`](docs/RESULTS_2026-10-15.md)

### 5-5. op 데이터 · ctxt_video (10-15 ~ 16) ✅
- **op:** 물체 영속 합성 영상 (op320) 을 경계 정렬로 10 %. IntPhys1 92.8 · GRASP 71.3 (현재 최고). 정지 영속만 ↑.
- **ctxt_video:** 문맥을 영상으로 (encoder, 한 번). 움직임 ↑ (gravity 33.3), 정밀도 ↓ (IntPhys1 76.1).
- 문서: [`docs/RESULTS_2026-10-16.md`](docs/RESULTS_2026-10-16.md) · 진단 [`z_research/VOBench_v1/Archive/DIAG_CTXT_VIDEO_VS_AR_2026-10-17.md`](../z_research/VOBench_v1/Archive/DIAG_CTXT_VIDEO_VS_AR_2026-10-17.md)

### 5-6. ctxt_video 변형들 (10-16 ~ 17) ⏸
- **tgtvideo (영상 정답):** 다른 서버에서 — "성능이 너무 안 나옴" (사용자). 이 기계 run 없음.
- **hole · lasthole:** encoding 전 문맥 구멍 + 복원. 복원 손실이 마지막 위치의 미래 예측과 충돌 → 복원 항 제거 후 중단 (e6).
- **ctxt_video + op:** e20 에서 IntPhys1 74.4 · VOBench late 51.9 / 54.6 — ctxt 와 같음, op 효과 없음 → e50 에서 중단.
- 문서: [`z_training_config/CTXT_VIDEO_TGTVIDEO_HOLE_2026-10-16.md`](../z_training_config/CTXT_VIDEO_TGTVIDEO_HOLE_2026-10-16.md)

### 5-7. residual gating (10-17 ~ 18) ✅
- **무엇:** 출력 = g · (마지막 관측 블록) + f. 배경은 복사, 움직임은 predictor 가 맡게.
- **결과:** AR 과 키 하나 차이 판 `natural_ar_vw_state_gres` 가 **네 벤치마크 모두 AR 이하** (IntPhys1 87.2, late 영속 93 → 60).
  예측 오차는 낮아졌지만 판별이 나빠짐. 출력 크기 축소는 측정됐지만 원인 아님.
- 중간 판 `ctxt_block_gres` (블록 문맥 · 양방향 · gating) 은 AR 과 네 가지가 같이 달라 중단.
- 문서: [`docs/RESULTS_GATING_2026-10-18.md`](docs/RESULTS_GATING_2026-10-18.md)

### 5-8. 영상 문맥 AR (10-18 ~) ▶
- **무엇:** target_encoder 로 문맥 pass · 전체 pass 를 따로 (영상), TF 는 문맥 자리만, 예측 구간은 rollout 만. gating 없음.
- **왜:** 영상 문맥의 전이 장점을 살리되 문맥 · 정답을 같은 (영상) 공간에 둔다. 예측 구간 TF 의 복사 지름길을 없앤다.
- **사전 측정:** 영상 공간에서는 복사 anchor 가 0.81–0.87 (블록 0.37–0.60) → gating 은 근거 없음 (`video_space_copy_gap.py`).
- **걱정:** 10-01 `natural_ctx_state` (영상 문맥 + 창 정답 + rollout 만) 은 우연 수준이었다. 차이는 TF (문맥) · AR 구조 · target_encoder 문맥.
- config: `configs/training/natural_ar_vw_state_vidctx.yaml`

### 5-9. 준비만 된 것 ○

| config | 무엇 |
|---|---|
| `natural_ar_vw_state_gres_prev` | gres + rollout anchor = 직전 예측 (gres 와 키 하나) |
| `natural_ar_ms` | multistart — 여러 시작점 rollout, 공식 데이터 비율 (`z_training_config/NATURAL_AR_MS_2026-10-14.md`) |
| `natural_ar_vw_state_dyn` | 움직임 크기 헤드 손실 |
| `natural_ar_vw_state_ctxt_video_c4b` | 문맥 ≥ 8 장 (gravity 개선 가설) |
| `natural_ar_vw_state_ctxt_video_tgtvideo(_hole)` | 영상 정답 ctx_state (다른 서버에서 돌린 판) |
| `natural_prefix` · `natural_ar` · `natural_ctx_ar` · `natural_ctx_state_vw` | 9 월 말 팔 A/C/C' 설계 (현재 계열의 조상) |

---

## 6. 채점 정의 (표의 숫자가 무엇인가)

| 지표 | 정의 | 계산 위치 |
|---|---|---|
| IntPhys1 A.8 | Garrido sliding, 창 16 · 32 × skip 2 · 5 · 10, Filtered, 격자 최고 | `eval_bench.sh` → `BENCH_latest.txt` |
| IntPhys 2 | Main, 창 16 · 32 · 48, 열별 최고 (Overall 최고 창) | 같음 |
| VOBench late P / R | late 가림만, 쌍 정확도 (primitive / realistic) | `cvpr/eval/testbed/run.sh vobench_v1_late` |
| VOBench gravity P / R | **block 정의** — 가능 미래가 그 block 의 모든 불가능 미래보다 낮아야 정답 (우연 37) | per_video_surprise + `data_csv/vobench_v1_gravity/index.csv` |
| GRASP · InfLevel | Garrido 공식 규칙 (`garrido_rescore`): property 별 최고 축약 → macro, 두 창을 한 탐색 공간으로 | `chain_bench.sh` |
| val | **2026-10-18 부터 VOBench late 432 쌍** (정식 채점과 같은 정의 — AR 최종으로 76.4 vs 정식 76.0 확인), epoch 마다. 그 전 run 은 IntPhys1 dev 60 쌍. **epoch 0 = 학습 전** | `metrics.jsonl` |

⚠️ gravity 를 `per_block.json` 의 `acc` 로 읽으면 arc 가 85 처럼 부풀려진다 (쌍 정확도). 반드시 block 정의로.

---

## 7. 조립 키 (config, `SET` 한 줄로 ablation)

| 키 | 값 |
|---|---|
| `model.predictor.kind` | `oneshot` · `ar` · `ctx_state` (+ 옛 `prefix` · `ctx_ar`) |
| `model.predictor.ar_feedback` | `state` · `output` |
| `model.predictor.ar_input` / `ar_target` | `block`/`block` · `video`/`window` |
| `model.predictor.ctx_encoder` × `ctx_input` × `ctx_target` | `encoder`·`target_encoder` × `video`·`block` × `window`·`block` |
| `model.predictor.gating_residual` · `gating_anchor` · `gating_out_ln` | `true`/`false` · `last_obs`/`prev` · `true`/`false` |
| `loss.tf_span` · `loss.tf_weight` · `loss.rollout_steps` | `all`/`predict` · 수 · 정수 |
| `loss.delta_weight` · `loss.dyn_weight` · `loss.ctx_hole` | 수 · 수 · 블록 |
| `data.var_window.{context_frames, predict_frames, datasets, fixed, boundary}` | 창 · 데이터 · 섞을 데이터 |

predictor 블록 안 키는 `eval_bench.sh` · `chain_bench.sh` · 레지스트리가 run config 에서 그대로 채점기로 넘긴다 (프로토콜이 `context_encoder_key` 를 고정으로 덮기 때문).

---

## 재현

```bash
cd /workspace/world_model/vjepa2
/venv/vjepa/bin/python z_training/tools/variants_table.py          # §4 표 다시 뽑기 (GPU 불필요)
CUDA_VISIBLE_DEVICES= /venv/vjepa/bin/python z_training/tests/gating_ctx_block_cpu_test.py   # 구조 검사 15 항목
GPUS=8 bash z_training/tmux_train.sh <config>                       # 학습
watch -c -n 2 bash z_training/monitor_ar.sh <run>                   # 감시
bash z_training/eval_bench.sh <run> latest.pt                       # IntPhys1 · IntPhys2
NOWAIT=1 RUNS=<run> bash z_training/chain_bench.sh                  # InfLevel · GRASP
RESULTS_ROOT=legacy GPUS=8 bash cvpr/eval/testbed/run.sh vobench_v1_late <레지스트리 모델>   # VOBench
```
- 시트: https://docs.google.com/spreadsheets/d/1b2CyYWqIFUNbPUB12Tx_BxtagImkLwQB1F1dNQ4iIiU
