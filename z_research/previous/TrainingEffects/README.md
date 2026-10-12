# TrainingEffects — 학습하면 무엇이 좋아지나 (시작점)

> 마지막 갱신 2026-09-25 (세션 `trained_model_eval`). 처음 작성.
> - 14 모델 × (IntPhys 1 · IntPhysGen v11) 채점, 그룹 A 22 체크포인트 학습 곡선, 복사 기준선, 네 능력 (읽기 · 지속 · 전이 · 영속) 분석
> - 세부 문서 다섯 개는 적대 검증을 거쳤고, 틀린 문장은 정정했다 → [`Archive/VERIFICATION_2026-09-25.md`](Archive/VERIFICATION_2026-09-25.md)
>
> 모델 목록 → [`MODELS.md`](MODELS.md) · 점수표 (자동 생성) → [`scores/SCORES.md`](scores/SCORES.md)

## 0. 결론

**predictor 만 다시 학습하면 v11 점수는 다 오르지만, IntPhys 1 은 IntPhys 1 로 학습한 것만 오른다 — 학습한 도메인 밖으로는 안 간다.**
**v11 에서 오른 능력 가운데 확실한 것은 "가려졌다 다시 나오는 물체를 제자리에 두는 것" 이고, 그것은 v11 궤적을 그대로 학습한 predictor 에서만 깨끗하다.**
속도 · 가속이 다양한 합성 데이터로 학습한 Predictor_v1 은 그 가운데 상당 부분 (flat 약 40 %, ramp 약 26 %) 이 "가림막 출구에서 물체가 나온다" 는, 문맥과 무관한 prior 다.
**문맥에 없던 운동 변화 (가속 · 감속 · 중력) 를 미래에 넣게 됐다고 확인되는 predictor 는 없다.** 확실한 변화는 Predictor_v1 이 포물선에서 **덜** 떨어지는 것 (진실에서 멀어짐) 하나다.
**예외 하나 — 보이는 물체를 등속에 가깝게 더 멀리 옮기는 것 (지속 · 도달) 은 도메인 밖으로도 간다.** 자연 영상으로만 학습한 Ariel 이 처음 보는 RollOut_v3 에서
마지막으로 본 물체를 진실 자리까지 옮긴다 (파라미터 없는 판독, auto_research 교차 확인; 외형 템플릿 적중 t10 릴리즈 0.04 → Ariel 0.73, encoder 천장 0.80). 그런데 같은 Ariel 의 IntPhys 1 은 69.4 로 떨어진다.
사전학습부터 다시 한 모델 (그룹 B) 에서는 데이터가 IntPhys 1 을 좌우한다 — 합성 물리 ViT-B 96.1 vs K400 ViT-B ~50.
그런데 v11 에서는 그룹 B 의 **predictor 여덟 개 중 누구도 자기 encoder 의 "마지막 관측 복사" 를 의미 있게 넘지 못한다** (−22.3 ~ +1.4).

근거 세 줄:
1. IntPhys 1: **IntPhys 1 로 학습한 predictor 만** 오른다 (88.9 → 93.9). v11 로 학습하면 **첫 epoch 에** 88.9 → 79.4 로 떨어지고, skip2_w16 칸에서는 epoch 가 늘수록 더 떨어진다.
   v11: 다섯 predictor 가 다 오른다. 하지만 v11 post-FT (+15.5) 는 같은 두 궤적의 재생이고, 도메인 밖 셋 (+2.7 ~ +3.7) 은 48 칸 중 5 ~ 7 칸을 잃는다.
2. 문맥 끝에 가려진 물체의 진실 칸 attention: v11 post-FT 는 물체 clip 0.66 / 0.57, 빈 장면 같은 자리 0.06 / 0.04 로 **깨끗하다**.
   Predictor_v1 은 0.50 / 0.35 인데 빈 장면에서도 0.21 / 0.09 를 둔다 (순 0.29 / 0.26). 이 귀무는 epoch 마다 커진다.
   도메인 밖 셋은 0.04 ~ 0.13 으로 릴리즈 (0.06 / 0.04) 수준이다.
3. RollOut_v3 (8 운동 법칙, 궤적 112 개): 가속 · 감속을 진실 쪽으로 옮겼다고 두 자 모두에서 확인되는 predictor 가 없다. 다만 학습한 predictor 는 모두 "궤적 속도를 따르는 기울기 ≈ 1 + 고정 앞섬 (+6 ~ +50 px)" 이라 이득 비율 (ρ) 판정이 확정적이지 않다 — **"넣지 못한다" 가 아니라 "넣는다는 증거가 없다"** 까지다.

미결: 그룹 B 의 IntPhys 1 복사 기준선은 뽑지 않았다 (합성 물리 ViT-B 96.1 이 predictor 몫인지 모른다). 위치 판독의 주 자 (`p@h`) 는 p 토큰에서 따로 검증되지 않았다 (§5).

## 1. 성능

### 1-1. 그룹 A — encoder 는 릴리즈 ViT-H 고정, predictor 만 다르다

v11 = `v11_split_test` (학습 안 한 block 절반, 10,752 matched pair, block bootstrap).
IntPhys 1 = Garrido A.8 격자 최고 칸 (180 쌍, 창 16/32 실행을 합쳐 Filtered · property macro). 격자 최고는 descriptive 다 (CLAUDE.md §1-3).
복사 = 예측 없이 문맥 마지막 latent 를 미래에 그대로 둔 채점 (같은 쌍, 같은 encoder).

| predictor (학습 데이터) | v11 | Δ vs 릴리즈 | p − 복사 (v11) | IntPhys 1 | p − 복사 (IntPhys 1 skip2_w32, 표준 표적) |
|---|---:|---|---|---:|---|
| **릴리즈** | 75.8 | — | +2.5 [+1.6, +3.4] ⚠️ 색 빼면 −1.7 | **88.89** | +3.9 [−0.6, +8.9] |
| post-FT **v11** (v11 train 절반) | **91.4** | +15.5 [+14.8, +16.3] | +17.9 | 77.22 ↓ | −6.7 (CI 0 포함) |
| post-FT **IntPhys 1** (IntPhys 1 train) | 78.5 | +2.7 [+2.0, +3.4] | +5.5 ⚠️ 색 빼면 −0.4 | **93.89** | **+8.9 [+4.4, +13.9]** |
| post-FT IntPhys 2 (254 clip) | 79.6 | +3.7 [+3.1, +4.3] | +6.3 | 85.56 | +0.6 (CI 0 포함) |
| post-FT **Predictor_v1** (등속 · 가림 · 경사, 속도 다양) | 81.6 | +5.8 [+5.1, +6.5] | +8.1 | 80.56 ↓ | −4.4 (CI 0 포함) |
| Ariel block-causal (scratch, SSv2+K400) | 79.4 | +3.6 [+2.9, +4.2] | +6.1 | 69.44 ↓ | −16.1 [−23.3, −8.9] (인과 표적 −7.8) |
| *복사 기준선 (예측 없음)* | *73.2* | | | *85.00* | |

- v11 p − 복사 는 `scores/ANALYSIS.md`, IntPhys 1 p − 복사 는 `ip1_copy/IP1_COPY.md` 에서 왔다. 추출 배열과 하네스는 fp16 동점 근처 쌍 뒤집힘만큼 다르다 (v11 ±0.24 pt, IntPhys 1 ±1 쌍).
- ⚠️ **릴리즈가 v11 에서 복사를 이기는 +2.5 는 전부 색 위반 (+9.3) 에서 나온다.** 모양 (−1.6 [−2.9, −0.2]) 과 vanish '물체→빈' (−6.8) 에서는 복사보다 CI 로 낮고, 색을 빼면 −1.7 [−2.6, −0.7] 이다.
- **IntPhys 1 은 복사만으로 85.0 이다.** 릴리즈 88.89 도 그 칸에서 복사와 CI 로 구분되지 않는다 (skip2_w16 에서는 +9.4, 인과 표적 skip2_w32 에서는 +6.7).
  skip2_w32 에서 모든 방법이 '나타남' 방향 92 ~ 100 %, '사라짐' 방향 36 ~ 64 % 라 방향 편향이 크다 (O1 '사라짐' 은 15 쌍 — 한 쌍이 6.7 pt).
- Ariel 이 복사보다 낮은 것은 표준 표적에서다 (인과 표적에서는 skip2_w32 한 칸만). Ariel 은 예측 8 slot 이하로 학습했는데 skip2_w32 (C=4) 는 14 slot 을 묻는다.

### 1-2. 그룹 B — encoder · predictor 전부 새로 사전학습 (jongseo, V-JEPA 2 abbreviated recipe, 16f @ 4 fps)

| 모델 | v11 (own p) | v11 own 복사 | own p − own 복사 | IntPhys 1 | 칸 |
|---|---:|---:|---|---:|---|
| ViT-B K400 72K | 53.6 | 61.1 | **−7.6 [−8.2, −7.1]** | 52.8 | skip5_w16 |
| ViT-B K400 30K\* | 54.7 | **76.9** | **−22.3 [−23.1, −21.4]** | 49.4 | skip2_w32 (skip2_w16 **22.8** — chance 밑) |
| **ViT-B 합성 물리 30K** | 69.3 | 69.0 | +0.3 [−0.5, +1.1] | **96.11** [92.9, 98.6] | skip2_w32 (w16 95.6 · skip5 79.4) |
| Tiny K400 5K | 67.2 | 66.8 | +0.4 [−0.0, +0.8] | 55.6 | skip2_w16 |
| Tiny 합성 물리 5K | 67.1 | 67.4 | −0.3 [−0.8, +0.1] | 53.9 | skip2_w16 |
| Tiny K400 30K | 68.7 | 69.6 | −0.9 [−1.4, −0.4] | 62.2 | skip2_w16 |
| Tiny 합성 물리 22.5K\* | 67.4 | 70.2 | −2.9 [−3.5, −2.3] | 72.2 | skip2_w32 |
| Tiny K400+SSv2 22.5K\* | 72.4 | 71.0 | +1.4 [+0.8, +2.0] (visible 에서는 −2.2) | 63.3 | skip2_w32 |

\* 중간 체크포인트 (lr 미감소 또는 학습 중). 복사 절대값은 encoder 마다 다르므로 행끼리 비교하지 않는다.

- **합성 물리 ViT-B 는 IntPhys 1 에서 릴리즈 ViT-H 를 넘는다** (세 칸 95.6 / 96.1 / 79.4). v11 은 릴리즈보다 낮다 (69.3 vs 75.8).
  합성 데이터에 가림 · 지지 · 경사 · 포물선 장면이 있고, 렌더 도메인도 IntPhys (Unreal) 와 가깝다. 이 모델의 IntPhys 1 복사 기준선은 뽑지 않았다.
- **ViT-B K400 이 chance 근처인 이유는 predictor 다.** 같은 encoder 의 복사가 76.9 (30K) · 61.1 (72K) 인데 own p 는 54.6 · 53.6 이다. 원인은 안 쟀다.
- Tiny 의 IntPhys 1 은 step 이 늘면 오른다 (K400 55.6 → 62.2, 합성 53.9 → 72.2). v11 은 67 ~ 72 에 머문다. 복사를 넘는 유일한 모델 (K400+SSv2) 은 학습 중인 체크포인트다.

## 2. 무엇이 좋아졌나 — 네 능력 (그룹 A)

정의는 §4. 수치는 전부 추출 배열에서 스크립트로 다시 냈고, 세부 문서마다 적대 검증을 거쳤다 (`Archive/VERIFICATION_2026-09-25.md`).
자 (readout) 는 **릴리즈에서 학습 · 보정한 것을 모든 predictor 에 같은 문턱으로** 건다. 위치는 진실 칸 3×3 attention 질량이 균등 (0.035) 을 넘을 때만 읽는다.
⚠️ 주 판독 h 자 (`p@h`) 는 p 토큰에서 따로 검증되지 않았다 (평지 장면에서 릴리즈 p 를 19 ~ 40 px 위로 읽는다). **위치 · 운동 수치는 행동 수준의 비교로만 읽는다.**

| 능력 → 측정 | 릴리즈 | v11 | Predictor_v1 | IntPhys 1 | IntPhys 2 | Ariel |
|---|---|---|---|---|---|---|
| **영속** — v11 문맥 끝 가림, 진실 칸 attention (flat / ramp) | 0.06 / 0.04 | **0.66 / 0.57** | 0.50 / 0.35 | 0.08 / 0.04 | 0.08 / 0.07 | 0.13 / 0.10 |
| 영속 — 같은 자리, 물체 없던 빈 장면 (귀무) | | 0.06 / 0.04 | **0.21 / 0.09** (epoch 마다 ↑) | | | |
| 영속 — vanish '물체→빈' 채점 (late k≥2 flat / ramp, %, 자 없음; 복사 26 / 5) | 0 / 0 | 100 / 100 | 96 / 92 ⚠️ | 1 / 0 | 48 / 26 | 27 / 5 (= 복사) |
| **지속** — v11 보임, 진실 칸 attention (flat / ramp) | 0.17 / 0.05 | 0.75 / 0.66 | 0.72 / 0.62 | 0.55 / 0.15 | 0.26 / 0.12 | 0.64 / 0.37 |
| 지속 — v11 보임 flat t4 x 오차 (진실 전진 86 px; 궤적 2 개) | −72 px | −17 | −16 | −28 | −64 | −36 |
| **지속** — v3 C16/P32, 두 정의 ('있다' 비율 · '모임') 에서 같은 방향 | — | **↓** (T50 10 → 5; 과잉 외삽 — §2-1b) | **↑** (T50 → 16, 안 떨어짐) | 정의에 따라 다름 | 자에 따라 다름 | 정의에 따라 반대 |
| 지속 — v3 **파라미터 없는 판독** (auto_research): 외형 템플릿 적중 t10 / 진행률 t10 (진실 1; encoder 천장 0.80) | 0.04 / 0.42 | 0.35 / 0.87 (t4 **1.92** 앞지름) | **0.67 / 0.92** | 안 잼 | 안 잼 | **0.73 / 0.93** |
| 도달 거리 — v3 flat_v, 마지막 on-GT 튜블릿의 진실 이동 (p@h / p@p, px) | 82 / 82 | 43 / 47 | **127 / 127** (하한) | 64 / 115 | 36 / 82 | 104 / 65 |
| **전이** — v3 가속 · 감속 (ρ = 이득 비, 서술적) | 두 자 불일치 | 궤적 속도 따라감 (기울기 0.86) + 고정 앞섬 +37 px | 등속 외삽 쪽 | 등속 외삽 쪽 | 섞임 | 섞임 |
| 전이 — 중력 (arc y, 부호 수정 뒤 두 자 × 두 보정 일치) | — | — | **덜 떨어짐** (Δ −17.6 / −14.5, 진실에서 멂) | 더 떨어짐 (Δ +2.6 / +7.0, 작음 · 보정 가정 약함) | — | — |

⚠️ Predictor_v1 의 vanish '물체→빈' 상승은 **교환**이다. 같은 칸의 '빈→물체' 가 100 → 73 (flat) 으로 epoch 마다 내려간다 — 빈 장면에서도 물체가 나오는 쪽으로 기운 것이다.
⚠️ Predictor_v1 은 RollOut_v3 와 **같은 10° 쐐기 · 등속 팔**로 학습했다 — v3 에서의 지속 · 도달 거리 향상은 근접 도메인의 이득이다.
세부: 영속 · 지속 [`v11_presence/PRESENCE.md`](v11_presence/PRESENCE.md) · 전이 · v3 지속 [`v3_motion/MOTION.md`](v3_motion/MOTION.md).

### 2-1. 영속 · 지속 — 그 상황을 학습한 predictor 에서, 첫 epoch 에

- 영속 배치 (flat) 학습 곡선: v11 e0 0.06 → **e1 0.51** → e10 0.66 · Predictor_v1 e0 0.06 → **e1 0.53** → e15 0.50.
  Predictor_v1 은 같은 기간 빈 장면 귀무가 0.12 → 0.21 로 커져 **순 배치가 0.41 → 0.29 로 줄어든다.** IntPhys 1 · IntPhys 2 · Ariel 은 0.08 ~ 0.13 에 머문다.
- 채점: 늦은 가림에서 슬롯 수를 맞춰 비교하면 v11 만 "다시 나온 뒤" 에 이득이 몰린다 (숨음 +5.9 / 재등장 +12.3 / 그 뒤 +20.3). 다른 predictor 는 숨은 슬롯 ≤ +0.7 이다.
- 보이는 물체는 어떤 predictor 도 잃지 않는다 (AUROC ≥ 0.97). 하지만 복사 기준선도 1.00 이라 **예측 능력의 증거가 아니다.**
  바뀌는 것은 자리 (릴리즈 −72 px → v11 · Predictor_v1 −16 px) 이고, 이것은 v11 의 두 궤적 안에서만 일관된다.

### 2-1b. 교차 확인 — 자 없는 판독 (auto_research, 1차 · 적대 검증 전)

`auto_research/Archive/TRAINED_PREDICTORS_SIGNATURES_2026-09-25.md` 가 같은 체크포인트를 **학습된 자 없이** (진실 칸 템플릿 argmax − 다른 궤적 귀무 · 문맥 마지막 물체 토큰 외형 템플릿) RollOut_v3 에서 읽었다.
- **Predictor_v1 · Ariel 은 물체 내용을 진실 자리까지 옮기고 거의 뒤처지지 않는다** (t10 외형 적중 0.67 / 0.73, 진행률 0.92 / 0.93). 릴리즈는 0.04 / 0.42.
  → 이 문서의 attn 자 판독에서 Ariel 이 "정의에 따라 반대" 였던 것은 **자의 '있다' 판정 (물체다움) 이 약해지는 것** (공통 문턱 t10 0.49 · t15 0.25) 과 **위치 내용이 옮겨지는 것** 이 갈렸기 때문이다.
- **v11 post-FT 의 '일찍 잃음' 은 과잉 외삽이다** — stride 1 에서 너무 빨리 옮긴다 (t4 진행률 1.92). raw stride 3 으로 학습해 튜블릿당 이동을 3 배로 배운 것과 맞다 (§2-2 의 기울기 0.86 + 고정 앞섬과 같은 그림).
- 네 predictor 모두 **앞선 문맥 역사를 다른 궤적으로 바꿔도 거의 안 변한다** (≈ 0) — 마지막 상태만 쓴다.
- 가속 반영은 재지 않았다. 향상은 "등속에 가까운 외삽을 더 멀리" 까지다 — §2-2 와 같은 결론.

### 2-2. 전이 — 넣게 됐다는 증거가 없다

- 가속 (flat_a) · 감속 (flat_d) 을 진실 쪽으로 옮겼다고 **두 자 모두에서** 확인되는 predictor 가 없다. Predictor_v1 · IntPhys 1 은 ρ 로 보면 등속 외삽 쪽이다.
- 하지만 학습한 predictor 는 모두 "궤적 속도를 따르는 기울기 ≈ 1 + 고정 앞섬 (+6 ~ +50 px)" 이다. 이득 비 ρ 가 이 덧셈꼴 앞섬을 지우지 못하므로 ρ 판정은 서술적이다.
  (v11 post-FT 가 "문맥 속도를 무시한다" 던 초판 문장은 **틀렸다** — 기울기 0.86 [0.53, 1.13] 으로 궤적 속도를 따른다.)
- 중력: 부호를 바로잡은 보정에서 두 자 × 두 방식 모두 같은 방향인 것은 **Predictor_v1 이 포물선에서 덜 떨어진다** (Δ −17.6 / −14.5, 진실에서 멀어지는 쪽) 와
  IntPhys 1 post-FT 가 조금 더 떨어진다 (Δ +2.6 / +7.0; 이 predictor 는 t0–1 이 흔들려 보정 가정이 약하다) 둘이다.
- 릴리즈 자신의 "가속 뒤처짐 · 감속 앞섬 · 벽 통과" 서명도 p 자에서만 재현되고 h 자에서는 안 된다 — 이 판독 체계로는 전이를 가르는 힘이 약하다.

### 2-3. 채점이 무엇을 올렸나 (v11, 48 세부 칸)

- 릴리즈가 복사를 못 넘는 칸이 비천장 31 개 중 20 개 (8 개는 CI 로 아래: 늦은 가림 vanish '물체→빈' 2.7 vs 복사 39.3, 보임 ramp shape 70 vs 79).
  학습 뒤 그중 복사를 넘게 된 칸 (bootstrap seed 에 따라 ±1): v11 16 · Predictor_v1 12 · IntPhys 2 10 ~ 11 · Ariel 9 · IntPhys 1 4.
- **도메인 밖 predictor 는 올린 만큼 다른 칸을 잃는다.** IntPhys 1 은 10 칸 ↑ / 7 칸 ↓ (움직이는 vanish '물체→빈', early ramp −49) 이고, 복사보다 새로 낮아진 칸이 3 개다 (48 칸 전부 기준; Predictor_v1 2 · IntPhys 2 2 · Ariel 1).
  IntPhys 2 · Ariel 이 잃는 5 칸은 전부 정지 물체 (주로 색) 다.
- 학습 곡선: v11 e1 89.4 (e0 75.7), 향상의 90 % 가 e2 까지. Predictor_v1 은 e1 최고 (82.4) 뒤 81.4. IntPhys 1 은 checkpoint 사이 ±2 pt 로 흔들린다 (e5 80.3 · e10 77.9 · e20 80.0 · e40 78.8).
  이 흔들림이 도메인 밖 이득 (+2.7 ~ +3.7) 과 같은 크기라, **도메인 밖 셋끼리 순위는 매길 수 없다** (seed 1 개).

### 2-4. 읽기 · 배치 — 정체는 모두 남고, 같은 생성기로 학습한 둘만 p 가 target 쪽으로 온다

[`v11_identity/IDENTITY.md`](v11_identity/IDENTITY.md) — v11_split test block, 튜블릿 평균 풀링 특징 위 선형 probe (logistic + ridge).
- **읽기**: 학습한 다섯 모두 p 의 조건 안 정체가 천장 근처로 남는다 (late 머리 → late: shape 96.1 ~ 97.8 %, color 94.0 ~ 97.5 %; 릴리즈 96.8 / 97.1; 우연 14.3 / 12.5).
- **배치**: v11 · Predictor_v1 (같은 생성기; v11 은 궤적 재생) 은 p 를 target (h_fut) 쪽으로 크게 옮긴다.
  - cos(p_fut, h_fut) late: 0.931 → 0.996 / 0.991.
  - 고정 h_fut 머리를 p 에 그대로 건 visible 정확도: shape 40.4 → 86.9 / 69.6 %, color 47.3 → 85.3 / 75.3 % (두 속성 모두).
  - 모양 클래스 평균 부분공간 z–p: 0.794 → 0.881 / 0.896 (encoder 끼리 0.943). 색은 추정 잡음이 크다.
  - IntPhys 1 은 일부만 움직이고 (cos 0.974), IntPhys 2 와 Ariel 은 거의 안 움직인다.
- **정렬**: visible 짝으로 맞춘 직교 사상을 late 에 걸면 어느 predictor 에서도 45 ~ 66 % 이고, late 짝으로 맞춘 91 ~ 97 % 에 못 미친다.
  → 어떤 학습도 가려진 조건의 p 를 보이는 조건과 같은 자리에 놓게 만들지 않았다.
- (궤적 × k) 단위 CI 에서 약 5 pt 이하 정확도 차와 0.01 이하 공유 몫 차는 대부분 유의하지 않다 — 주장하지 않는다.

## 3. 학습만으로 해결되나 — 판정

| 질문 | 판정 | 근거 |
|---|---|---|
| 점수가 학습 도메인 밖으로 가나 | **아니다** | IntPhys 1 은 IntPhys 1 학습만 오른다. v11 · Predictor_v1 학습은 첫 epoch 에 IntPhys 1 을 내리고 (88.9 → 79.4 / 83.9), skip2_w16 에서는 epoch 마다 더 내린다. v11 은 다 오르지만 도메인 밖 셋은 5 ~ 7 칸을 잃는다 |
| 복사보다 나은 예측이 되나 | **도메인 안에서만, 그리고 위반 종류에 따라** | v11: 전체로는 모두 릴리즈 (+2.5) 보다 폭이 크지만, 릴리즈 · IntPhys 1 post-FT 의 우위는 색 위반에서만 나온다. IntPhys 1: 복사를 CI 로 넘는 것은 IntPhys 1 학습뿐 (+8.9) |
| 영속 (가려진 물체를 제자리에) | **v11 궤적을 그대로 학습한 predictor 만 깨끗하게** | v11 post-FT 순 배치 0.60 / 0.53. Predictor_v1 은 순 0.29 / 0.26 이고 빈 장면에도 물체를 둔다. 도메인 밖 셋은 릴리즈 수준 |
| 지속 · 도달 (보이는 물체를 더 멀리) | **도메인 밖으로도 일부 간다** | 자 없는 판독에서 Predictor_v1 (근접 도메인) 과 **Ariel (자연 영상 scratch, 도메인 밖)** 이 물체를 진실 자리까지 옮긴다 (t10 0.67 / 0.73 vs 릴리즈 0.04). v11 post-FT 는 과잉 외삽. 단 학습 자로는 Ariel 의 '있음' 이 약해진다 |
| 전이 (문맥에 없던 변화) | **넣게 됐다는 증거 없음** | 가속 · 감속을 두 자 모두에서 진실 쪽으로 옮긴 predictor 없음. 확실한 변화는 Predictor_v1 의 "포물선에서 덜 떨어짐" (진실에서 멂). IntPhys 1 post-FT 의 "조금 더 떨어짐" 은 작고 보정 가정이 약하다 |
| 정체 · 배치 | **정체는 원래 남아 있다. 배치는 같은 생성기로 학습하면 target 쪽으로** | 모든 p 에서 조건 안 정체 ≥ 94 %. v11 · Predictor_v1 만 cos 0.93 → 0.99, 고정 머리 이식 40 → 87 / 70 % |
| 사전학습 데이터를 바꾸면 | **IntPhys 1 은 크게 바뀐다. v11 에서 predictor 가 복사를 넘지는 못한다** | 합성 물리 ViT-B IntPhys 1 96.1. 그룹 B own p − own 복사 −22.3 ~ +1.4 |

**한 문장**: 학습은 "보이던 물체를 등속에 가깝게 더 멀리 옮기는 것" (도메인 밖으로도 일부) 과 "이 도메인에서 가려진 물체가 어디 있을지" (도메인 안) 를 가르친다. "문맥에 없던 변화 (가속 · 중력) 를 미래에 넣는 것" 을 가르쳤다는 증거는 없다 — 이 자료가 닿는 범위 (합성 장면 · predictor 5 개 + 사전학습 8 개 · seed 1 개) 안에서.

## 4. 무엇을 쟀나 (정의)

기호: 문맥 encoder 출력 z (online encoder, 앞 16 장), target h = LN(target encoder(32 장 전부)) — affine 없는 토큰 LN,
predictor 출력 p (미래 8 튜블릿), 튜블릿 t = 0…7.

| 축 | 정의 | 데이터 | 비교 기준 |
|---|---|---|---|
| **점수 v11** | 쌍마다 hit = 1[S(가능) < S(불가능)] (+0.5 동점), S = mean_t,token \|p_t − h_t\| | `v11_split_test` 10,752 matched pair (학습 안 한 block 절반) | chance 50 · 릴리즈 |
| **점수 IntPhys1** | Garrido A.8: 창마다 S, 시작점마다 C 최소 (Filtered), 창 평균, 쌍 비교, property macro, 격자 최고 칸 | dev 180 쌍 | 88.89 (릴리즈) |
| **복사 기준선** | S_copy = mean_t \|LN(z)_{문맥 마지막 튜블릿} − h_t\| — **예측 없이** 마지막 관측을 미래로 복사 | 같은 쌍 | p − copy (같은 쌍 위의 차이) |
| **학습 곡선** | 위 점수를 epoch 체크포인트마다 (v11 e1…e10, Predictor_v1 e1…e15, IntPhys 1 e5…e40, IntPhys 2 e10…e80, Ariel ep5…ep43) | 같은 쌍 | e0 = 릴리즈 |
| **영속** (가려진 물체가 미래에 남나) | 튜블릿마다 '있다' = 자의 logit > 고정 문턱 (h 자 −0.961 / p 자 0.8307, 릴리즈에서 정한 값을 **모든 predictor 에 그대로**), 물체 clip 비율 − 빈 장면 비율, AUROC (물체 vs 빈), 진실 칸 3×3 attention 질량 | v11 문맥 끝 가림 (late, k ≥ 2), test 절반 | 가림 없음 (visible) · 릴리즈 · h@h (천장) |
| **지속** (보이는 물체가 미래 끝까지 남나) | 같은 '있다' 비율의 튜블릿 곡선, T50 = 처음으로 50 % 아래 | v11 visible · RollOut_v3 C16/P32 | 릴리즈 |
| **전이** (문맥에 없던 변화를 미래에 넣나) | 부호 오차 e_a(t) = (x̂_a − x_a) · s_a(t) (+ = 운동 방향으로 앞섬), 이득 β = 읽은 이동 / 진실 이동, ρ = β(법칙) / β(flat_v) | RollOut_v3 8 법칙 (등속 · 가속 · 감속 · 경사 · 포물선 · 선반 낙하 · 벽) | **복사** (멈춤) · **등속 외삽** · 릴리즈 |
| **읽기 · 배치** | 평균 풀링 특징 위 선형 probe (같은 조건 안 / 조건 사이 이식), encoder 머리 → p 이식, 직교 Procrustes 뒤 이식, 클래스 평균 부분공간 공유 몫 | v11 가능 물체 clip | encoder (z · h) · 릴리즈 |

### 읽는 규칙 (먼저 정해 두었다)

- **자는 릴리즈에서 학습 · 보정한 것을 모든 predictor 에 그대로 건다.** 주 판독은 h 자 (`p@h`) — 모든 predictor 가 LN(h) 를 맞추도록 학습됐으므로 predictor 와 무관한 자다.
  p 자 (릴리즈 p 로 학습) 는 이식 판독이라 보조. predictor 마다 문턱을 다시 잡지 않는다. 두 자에서 방향이 같을 때만 '견고' 라고 쓴다.
- **위치 주장은 attention 이 물체에 모일 때만** (CLAUDE.md §5-5). 퍼지면 자는 기본값을 낸다 — "멈춘다" 로 읽지 않는다.
- **CI 단위** — v11 점수: block · IntPhys 1: 장면 (principle 층화) · RollOut_v3: **궤적** (자유 운동 84 + ledge 14 + wall 14 개, clip 은 외형 복사) · v11 readout: (궤적, k).
- **v11 은 `v11_split_test` 로만 읽는다.**
- **"학습 도메인 안"** 을 표마다 밝힌다: v11 → v11 post-FT · Predictor_v1 (가림 k=4 팔), IntPhys 1 → IntPhys 1 post-FT.

## 5. 단서 (결론과 같은 무게로)

1. **v11 은 운동 조건마다 궤적이 2 개 (좌→우 · 우→좌) 뿐이다.** test 절반에도 같은 두 궤적이 있다.
   v11 post-FT 의 향상은 **궤적 재생**이다. v11 readout CI 단위 (궤적, k) 는 늦은 가림 6 개 · 보임 2 개라, 외형 (모양 · 색 · 배경) 을 가로지른 일반화만 말할 수 있다.
   궤적만으로 묶으면 (2 개) v11 post-FT 의 영속 flat Δ 도 CI 가 0 을 걸친다 ([−0.01, +0.32]).
2. **자 이식.** 고정 문턱 '있다' 는 predictor 의 척도 이동을 섞는다. IntPhys 2 는 두 자에서 영속 Δ 방향이 반대이고, Ariel 은 빈 장면 바닥이 0.6 이다.
   그래서 영속 판단은 **진실 칸 attention 질량** (문턱 없음) 과 채점 (자 없음) 을 같이 본다.
3. **RollOut_v3 는 어떤 predictor 의 학습 시간 척도와도 다르다** (stride 1 = 튜블릿 67 ms; v11 · Predictor_v1 post-FT 는 200 ms).
   P32 t8 ~ t15 는 v11 · Predictor_v1 post-FT 의 학습 지평 (8 튜블릿) 밖이다. IntPhys 1 post-FT (≤ 14) · IntPhys 2 post-FT (≤ 18) 에게는 일부 안이다.
   Predictor_v1 은 v3 와 같은 10° 쐐기 · 등속 팔로 학습했다 — 근접 도메인이다.
4. **한 번씩만 학습했다 (seed 0).** 학습 분산을 모른다. IntPhys 1 180 쌍의 95 % CI 는 ±5 ~ 7 pt 다.
5. **Ariel 은 네 가지가 동시에 다르다** (attention 규칙 · 데이터 · scratch · epoch). 원인을 attention 규칙으로 좁힐 수 없다.
6. **목적함수를 원인으로 걸지 않는다.** 여기서 쓰는 것은 "무엇이 바뀌었나" 까지다.
7. **표적 번짐.** h 는 창 전체를 본 양방향 target encoder 출력이다. '숨은 슬롯' 점수에도 재등장 정보가 섞일 수 있다.
   IntPhys 1 은 인과 표적 (그 튜블릿까지만 본 target) 으로도 냈다 (`ip1_copy/`).
8. **그룹 B 의 IntPhys 1 복사 기준선은 없다** (v11 만 뽑았다). 그룹 B 의 학습 입력은 16f @ 4 fps 로 채점 시간 간격과 다르다.
9. 여러 칸을 한꺼번에 비교했다 (v11 48 칸 × 5 predictor 등). 한 칸의 CI 하나보다 **반복되는 패턴**을 읽는다.
10. **위치 판독의 주 자 `p@h` 는 p 토큰에서 따로 검증되지 않았다.** 평지 장면에서 릴리즈 p 를 19 ~ 40 px 위로 읽는다. 위치 · 운동 비교는 행동 수준으로만 읽고, x 방향은 두 자가 같은 쪽일 때만 견고하다고 쓴다.
11. **이득 비 ρ 는 덧셈꼴 앞섬을 지우지 못한다.** 학습한 predictor 는 모두 기울기 ≈ 1 + 고정 앞섬이다. 전이 판정은 기울기 + 절편 적합으로 다시 내야 확정된다.
12. **IntPhys 1 은 방향 편향이 크다** (skip2_w32 '나타남' 92 ~ 100 %, '사라짐' 36 ~ 64 %; O1 '사라짐' 은 15 쌍). 칸 하나의 차이는 한두 쌍이다.
13. **v11 readout 의 CI 단위 (궤적, k) 에서 작은 효과는 대부분 유의하지 않다** — 5 pt · 0.01 이하 차이는 주장하지 않는다.

## 6. 다음 (제안)

- **외형 hold-out 학습** (`build_v11_split_index.py --holdout-shape/--holdout-color`) — v11 post-FT 가 외형을 가로질러서도 영속 배치를 내는지.
- **궤적 · 속도 hold-out** — 지금 자료로는 궤적 일반화를 말할 수 없다 (§5-1).
- **전이를 직접 가르치는 데이터** — Predictor_v1 도 등속 외삽이었다. 문맥 안에 가속이 보이는 clip 을 학습시키면 ρ 가 움직이는가.
- **Ariel full-attention (oneshot) 팔** — attention 규칙 효과를 가르는 유일한 대조.
- 그룹 B 의 IntPhys 1 복사 기준선 · synphys Tiny 30K 완주 뒤 재채점 (09-26 예정).

## 폴더

| 자리 | 무엇 | 만드는 스크립트 |
|---|---|---|
| `MODELS.md` | 모델 목록 · 학습 설정 | (손으로) |
| `scores/SCORES.md` | 하네스 점수표 14 모델 (v11 · IntPhys 1, CI) | `training_effects_scores.py` |
| `scores/ANALYSIS.md` | v11 복사 기준선 · 학습 곡선 · 슬롯별 · 그룹 B own vs 복사 | `te_analyze_scores.py` |
| `ip1_copy/IP1_COPY.md` | IntPhys 1 복사 기준선 · 인과 표적 · 튜블릿별 · 학습 곡선 | `te_analyze_ip1.py` |
| `v11_presence/PRESENCE.md` | 영속 · 지속 (v11 readout) | `te_analyze_presence.py` |
| `v3_motion/MOTION.md` | 전이 · 지속 · 도달 거리 (RollOut_v3 readout) | `te_analyze_v3.py` |
| `v11_identity/IDENTITY.md` | 읽기 · 배치 (정체 probe · 기하) | `te_analyze_identity.py` |
| `v11score/`, `v11readout/`, `ip1score/` | 추출 검증 json | 추출 스크립트 `--validate` |
| `logs/` | 실행 로그 | — |

큰 배열 (추출): `/data2/local_datasets/world/world_analysis/cache/training_effects/{v11score_*, v11readout_curves, v3readout_curves, ip1score_{final,curves}, v11pooled_*}` (vll5 로컬).

## 재현

```bash
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
PY=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
# 1. 표준 하네스 채점 (14 모델 × v11_split_test · IntPhys 1 창 16/32)
bash z_research/scripts/analysis/training_effects_run.sh
$PY z_research/scripts/analysis/training_effects_scores.py                     # → scores/SCORES.md
# 2. 다중 predictor 추출 (encoder 한 번, predictor 만 바꿔 끼움; 전부 --validate 로 릴리즈 · 하네스 대조)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 $PY z_research/scripts/analysis/te_v11_score.py --model vith --preset curves --gpus 8
for m in vitb_k400_e240 vitb_k400_e100 vitb_synphys_30k vittiny_k400_5k vittiny_synphys_5k vittiny_k400 vittiny_synphys_30k_e225 vittiny_k400ssv2_30k_e225; do
  $PY z_research/scripts/analysis/te_v11_score.py --model $m --gpus 8; done
$PY z_research/scripts/analysis/te_intphys1_score.py --preset curves --gpus 8 && $PY z_research/scripts/analysis/te_intphys1_score.py --preset curves --validate --score
$PY z_research/scripts/analysis/te_v11_readout.py --preset curves --gpus 8
$PY z_research/scripts/analysis/te_v3_readout.py --preset curves --gpus 8 --validate
# 3. 분석 (CPU; identity 는 GPU probe 포함)
$PY z_research/scripts/analysis/te_analyze_scores.py
$PY z_research/scripts/analysis/te_analyze_ip1.py
$PY z_research/scripts/analysis/te_analyze_presence.py
$PY z_research/scripts/analysis/te_analyze_v3.py
$PY z_research/scripts/analysis/te_analyze_identity.py
# 추출은 2026-09-25 에 전부 끝났다 (13 GB). 다시 돌릴 때 SLURM 으로: sbatch z_research/scripts/analysis/training_effects_queue.sbatch (모든 단계 resume)
```
config: `configs/protocols/{surprise_c16t32,intphys1_sliding}.yaml` + `configs/protocols/models.md` (그룹 A `vith_pft_*` · `vith_ariel_bc_ep43`, 그룹 B `vitb_*` · `vittiny_*`).
