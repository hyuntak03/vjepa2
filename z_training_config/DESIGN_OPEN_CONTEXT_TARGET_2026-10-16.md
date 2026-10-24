> 📥 **vast 브랜치에서 가져온 문서 (2026-10-24, `origin/vast` c05c259 의 `z_training_config/DESIGN_OPEN_CONTEXT_TARGET_2026-10-16.md`, 원문 그대로).**
> 본문의 run 이름 · `/workspace/...` 경로 · 시트 언급은 vast 노드 기준이다 — 이 노드 (vll5) 에 그 산출물이 없을 수 있고, 수치는 그쪽 산출물에서 계산된 것이다.

# 설계 고민 정리 — 문맥을 영상으로 줄 것인가, 정답 (타깃) 을 어떻게 만들 것인가 (2026-10-16)

> 다른 에이전트에게 넘기려고 쓴 자기완결 문서. 측정한 것과 추측을 나눠 적었다 (추측은 **[추측]**).
> 레포: `/workspace/world_model/vjepa2` (branch `vast`). 기존 기록: `z_training/docs/RESULTS_2026-10-12/15/16.md`, `z_training_config/*.md`, 레포 `CLAUDE.md`.

## 0. 한 문장

frozen V-JEPA 2 ViT-H encoder 위에서 predictor 만 학습해 **latent state 를 시간 방향으로 굴리는 (rollout) world model** 을 만들고 싶다.
지금 막힌 질문은 **"문맥을 영상 (encoder 한 번) 으로 줄지 블록 단위로 줄지"** 와 **"그때 정답을 어떻게 만들어야 미래가 새지 않고 입력과 같은 공간이 되는지"** 다.

## 1. 배경과 사용자 원칙

- 목적: state evolution 네 능력 — 읽기 · 지속 · 전이 (문맥에 따라 미래가 바뀜) · 영속. 물리 장면은 측정 도구 (CLAUDE.md §0)
- 사용자 선호: **자연스러운 학습** (L1 위주, 메모리 · 레지스터 토큰 · warp/flow 헤드 · VAE/ODE 같은 공학적 구조는 싫어함), 단순한 해법
- 리뷰 우려 (사용자): "블록 단위 (2 장 tubelet) 로만 encoding 할 거면 왜 video encoder 를 쓰냐, 이미지 encoder 와 뭐가 다르냐" → **문맥은 영상으로 주는 게 이야기가 선다**는 쪽으로 기울어 있다
- 동시에: **정답 (GT) 에 미래가 섞이는 것이 걸린다** (사용자)
- 작업 규칙: 학습은 사용자가 "올려" 할 때만. 원인 설명은 측정 / 추측을 나눈다. 긴 학습 전에 싼 진단부터

## 2. 공통 설정 (모든 run)

- encoder: 릴리즈 V-JEPA 2 ViT-H (`vjepa2-vith-fpc64-256`), frozen. `encoder` (online) / `target_encoder` (EMA) 둘 다 있음. 시간 **양방향** attention. 사전학습: fps 4, 16 장 (cooldown 64 장), tubelet 2 장, 마스크 = clip 전체 시간에 걸친 tube (70 % × 2 / 15 % × 8)
- predictor: 22 M (384 차원, 12 층), 처음부터 학습. 블록 = tubelet 2 장 = 256 토큰 (16 × 16)
- 데이터: K710 + SSv2 + HowTo100M 1 %, fps 4, 가변 창 (`app/vjepa_frozen/var_window.py`), 패딩 없음, 20,580 step, Vast 8 × 96 GB
- 손실: L1. TF (teacher forcing, 진짜 앞 블록을 보고 다음 블록) + rollout (자기 상태를 되먹이며 최대 16 블록)
- 평가: IntPhys1 A.8 (Garrido 격자), VOBench (자체, late 가림이 주 지표), VOBench gravity, IntPhys 2, GRASP, InfLevel

## 3. 지금까지 판과 결과 (측정)

| 판 | 문맥 입력 | TF 입력 | 정답 | 한 번에 / rollout | IntPhys1 A.8 | VOBench late P / R | VOBench gravity P / R |
|---|---|---|---|---|---:|---|---|
| 릴리즈식 tube (`k710_ssv2_how_ep70_cooldown64`) | 영상 (`encoder`) | — | 창 전체 양방향 | 한 번에 (마스크 토큰) | 82.2 | 56.0 / 65.7 | — |
| **AR** (`natural_ar_vw_state`) | **블록 단위** (`target_encoder` 블록마다) | 블록 단위 | 블록 단위 | rollout (상태 되먹임) | **90.0** | **74.1 / 77.8** | 20.4 / 8.3 |
| **ctxt_video** (`..._ctxt_video`, kind=ctx_state) | 영상 (`encoder`) | 블록 단위 | 블록 단위 | rollout | 76.1 | **52.3 / 54.6** | **33.3 / 20.4** |
| 10-01 `natural_ctx_state` | 영상 | — (TF 없음) | 창 전체 | rollout 만 | 우연 수준 (46–56) | — | — |
| (참고) 릴리즈 V-JEPA 2 predictor | 영상 | — | 창 | 한 번에 | 88.9 | 57.9 / 65.7 | 0 / 0 |

ctxt_video 의 추가 측정:
- VOBench 전체 57.1 / 62.1 — **블록 복사 기준선 (77.8 / 81.1, 마지막 문맥 블록을 미래에 복사) 보다 20 pt 낮다**. visible (가림 없음) 사라짐도 66.7 (A 100 / B 33). surprise margin 이 AR 의 1/8–1/6
- 학습 손실은 AR 과 거의 같다 (e70 TF 0.531 vs 0.534, rollout 0.559 vs 0.550) — L1 의 대부분이 배경이라 손실로는 진단이 안 된다
- gravity 는 AR 보다 높다 (네 칸 모두 ↑, 그래도 전체는 우연 37 % 아래)
- ⚠️ ctxt_video 는 샘플링도 AR 과 다르다 (문맥 ≥ 2 블록, 데이터 33 / 10 / 57 % vs AR 38 / 20 / 42 %) — 원인 분리 불가

AR 진단 (측정): 마지막 1–2 문맥 블록만 본다 (마지막 4 블록만 남겨도 예측 6–9 % 변화). late 가림에서 가려진 마지막 블록 (attention 0.25) 을 보고 앞의 보이던 블록 (0.056) 은 거의 안 본다.

**[추측]** ctxt_video 실패 원인 후보 — (a) predictor 가 z 를 안 쓴다 (b) 영상 z → 블록 정답 공간 변환을 못 배워 평균으로 간다 (c) 문맥 (영상) 과 정답 (블록) 종류 불일치. 10-01 판 (z 만 입력) 도 실패했으므로 "TF 지름길" 만으로는 설명이 안 된다. 검증 안 함 — §6 의 진단이 먼저.
**[추측]** gravity 상승은 영상 z 가 운동 정보 (속도 · 가속) 를 넘겨서일 수도 (z 는 문맥 끝 속도를 R² 0.996 으로 담는다, CLAUDE.md §5-5), 예측이 둔해져 AR 의 편향이 묽어진 것일 수도.

## 4. 핵심 제약 — 왜 어려운가

1. **encoder 가 시간 양방향이다.** 창 (문맥 + 미래) 을 한 번에 encoding 하면 블록 b 토큰에 b+1 이후 정보가 섞인다
2. 그래서 **TF 입력** (predictor 에게 보여 주는 진짜 앞 블록) 은 창에서 잘라 쓰면 **누수**다 (답을 보고 맞힘)
3. **정답** 을 창에서 잘라 쓰면 누수 (부정행위) 는 아니지만 (손실에만 쓰임):
   - 블록 t 의 정답에 맞힐 수 없는 **더 먼 미래 정보가 섞인다**
   - 같은 블록의 정답이 **창 길이 (= 어디까지 예측하느냐) 에 따라 바뀐다** — 우리는 몇 블록을 예측할지 정해 두지 않으므로 일관성이 없다 (사용자 지적)
   - 릴리즈 V-JEPA 2 는 창 정답을 쓰지만 사전학습은 **미래 예측이 아니라** 시간 전체 tube 를 채우는 과제라 문제가 없다 (사용자 지적). **Meta 의 미래 예측판 V-JEPA 2-AC 는 프레임마다 따로 encoding 한 정답**을 쓴다
4. ctx_state 구조 (영상 문맥 + rollout) 에서는 **마지막 문맥 자리 출력이 다음 블록 예측의 시작 상태**이기도 하다 → 문맥 자리에 다른 손실 (예: 구멍 복원) 을 걸면 마지막 블록에서 두 정답이 충돌한다 (§5)
5. TF 첫 걸음과 rollout 첫 걸음은 같은 계산 (둘 다 진짜 문맥에서 출발) → 첫 예측 블록 손실이 두 번 들어간다 (AC 도 같다). TF 를 둘째 블록부터 걸면 중복이 빠진다

## 5. 설계 선택지

| | 문맥 | TF 입력 | 정답 | 누수 | 입력·정답 공간 | 비용 (encoder, ctxt_video = 1) | 비고 |
|---|---|---|---|---|---|---|---|
| A. 블록 단위 (AR, AC 방식) | 블록 | 블록 | 블록 | 없음 | 같음 | ≈ 1 | 지금 최고 성능. "왜 video encoder 냐" 에 약함 |
| B. 릴리즈식 한 번에 | 영상 | — | 창 | 없음 | 같음 | 낮음 | rollout 없음 (상태 진화를 못 봄). tube 판 82.2 / late 56–66 |
| C. 섞은 판 (ctxt_video · tgtvideo) | 영상 | 블록 | 블록 또는 창 | 정답 창이면 미래 섞임 | 다름 | 1 (블록 정답) / 2.2 (창 정답) | **일관성 없음** — 결과 해석이 어렵다 |
| **D. 인과 prefix 영상** | 영상 (블록 0..C−1 을 한 번) | 블록 t = encode(블록 0..t) 의 마지막 블록 slice | 같은 slice | 없음 | **같음 (인과 영상)** | **≈ 4.7** (희소 감독 시, §5-D) | 사용자가 제안. 가장 깔끔. 비용이 큼 |
| E. encoder 에 블록 causal 마스크 | 한 번 encoding (블록 b 는 ≤ b 만 봄) | slice | slice | 없음 | 같음 | ≈ 1 | 가장 쌈. 사용자가 "별로" — encoder 가 못 본 입력 분포 |

### D 상세 (가장 유력, 미구현)
- 미래 블록 j 마다 `target_encoder(블록 0..j)` 한 번 → 마지막 블록 slice 가 블록 j 의 **정답**이자 블록 j+1 을 맞힐 때의 **TF 입력**. 정답이 예측 길이에 따라 안 바뀐다
- 비용을 줄이려면 희소 감독: TF 는 첫 몇 블록 (예: 4), rollout 은 끝까지 굴리되 손실은 1 · 2 · 4 · 8 · 16 걸음에서만 → prefix forward 약 6–7 번. encoder 일 ≈ 4.7 배 (창 정답판 2.2 배), 학습 약 20–33 h **[추정, 미측정 — encoder 가 step 시간의 몇 %인지 모름]**
- 채점도 같은 인과 정의로 바꿔야 한다 → 릴리즈 (창 정답) 와 surprise 숫자를 직접 비교하기 어려워진다
- 문맥은 "블록 0..C−1 을 한 번" encoding (양방향) — 마지막 문맥 블록 표현은 prefix 정의와 같다 (encode(0..C−1) 의 마지막 slice)

## 6. 함께 걸린 "문맥 구멍" 아이디어 (사용자 결정, 진행 중)

- 목적: 읽기 · 추정 능력. AR 이 먼 과거를 안 보는 문제를 겨냥 — 문맥 **끝 k 블록** (k ≤ min(4, C−1)) 에 V-JEPA 2 tube 마스크 (70 % × 2, 같은 자리) 를 **encoding 전에** 뚫어 (encoder 에 보이는 토큰만), predictor 가 mask token 으로 채우고 복원 손실 (가린 자리) 을 건다. step 의 75 %
- 구현됨 (`loss.ctx_hole`, kind=ctx_state 전용): `app/vjepa_frozen/ar.py make_ctx_hole`, `train.py`, `src/models/rollout_predictor.py` (commit `85d214d`)
- ⚠️ **발견된 결함 (미수정)**: 마지막 문맥 블록 출력이 "다음 블록 예측 상태" 이자 "자기 블록 복원" 을 동시에 요구받는다 (§4-4). 후보: (i) 복원에서 마지막 블록 제외 (k = 1 이면 복원 0, 구멍 step 의 46 %) (ii) **복원 전용 선형 머리** (작은 파라미터, 채점 때 버림) — 권장
- ZWM (Aw et al. 2026, Yamins lab): 문맥 프레임은 다 보이고 미래 프레임 90 % 를 무작위로 가림 (픽셀 L2). 문맥까지 가린 **대칭 마스킹이 나빴다** → 보이는 문맥 블록을 최소 하나 남긴다
- ZWM 식 "미래 일부 패치 힌트" 는 우리가 이미 해 봤고 (힌트 10 %, TF 만) 효과가 없었다 (힌트를 끊자 TF 손실이 기준선과 0.5308 vs 0.5309)

## 7. 지금 상태 (2026-10-16 저녁)

- 학습 중: `natural_ar_vw_state_op` (AR + object_permanence 10 %, 20:49 끝)
- `tmux chain_hole`: op 채점 → **`natural_ar_vw_state_ctxt_video_tgtvideo_hole`** (C 판 + 창 정답 + 구멍) 학습이 21:30 쯤 자동 시작 — **§4-3 의 미래 섞인 정답, §4-4 의 마지막 블록 충돌이 그대로 들어 있다**. 멈출지 결정 필요
- 다른 서버: `natural_ar_vw_state_ctxt_video_tgtvideo` (C 판, 창 정답, 구멍 없음)

## 8. 다른 에이전트에게 묻고 싶은 것

1. 문맥을 영상으로 주면서 **누수 없고, 입력 · 정답이 같은 공간이고, 정답이 예측 길이에 안 흔들리는** 설계로 D 말고 더 싼 방법이 있나? (E 는 사용자가 꺼림 — encoder 분포 이탈이 실제로 얼마나 문제인지 근거가 있으면 재고 가능)
2. D 를 한다면 희소 감독 (어느 블록에 정답을 둘지) 을 어떻게 고르는 게 좋은가? 비용 대비 rollout 감독 밀도
3. "왜 video encoder 냐" 리뷰 질문에 A (블록 단위, 지금 최고 성능) 를 유지하면서 답할 방법은? (예: 이미지 encoder 대조군, V-JEPA 2-AC 선례)
4. ctxt_video 실패의 원인을 학습 없이 가르는 진단 설계 (§3 의 추측 a/b/c) — z 바꿔치기 · 예측의 평균 근접도 · 물체 / 배경 손실 분해 외에 더 좋은 게 있나
5. 문맥 구멍 복원과 rollout 시작 상태가 같은 자리를 쓰는 문제 (§4-4) 의 더 자연스러운 해법

## 재현 · 참고 경로

- config: `configs/training/natural_ar_vw_state{,_ctxt_video,_ctxt_video_tgtvideo,_ctxt_video_tgtvideo_hole}.yaml`, 릴리즈식: `k710_ssv2_how_ep70_cooldown64`
- 코드: `app/vjepa_frozen/{train,ar,var_window}.py`, `src/models/rollout_predictor.py` (kind=ar · ctx_state), 채점 `analysis/intphys2/model.py`
- 결과: `z_training/runs/<run>/eval/BENCH_latest.txt`, VOBench `z_research/VOBench_v1/<model>/surprise/tables.md`
- 비용 측정: encoder forward (ViT-H, batch 2, bf16) — 창 타깃 추가 시 encoder ×1.81 (C8K8) ~ ×2.37 (C32K32)
