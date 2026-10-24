# 07 — P3 (iv) encoder 에는 정보가 있다 · 실패는 predictor 의 미래 예측이다

> ⚠️ **2026-10-21 갱신 → [`13_SINCE_2026-10-10.md`](13_SINCE_2026-10-10.md) §4.** 이 장 머리 · README §3 · 정본 §2-3 (iv) 의 **"z 자가 가림 뒤 미래 위치를 5–10 px 로 맞힌다" 에는 가림막 교락 단서가 붙는다** (v11 가림막이 경계 순간 물체 자리에 서 있다; 10-13 정정) — 쓸 수 있는 문장은 "predictor 입력에 물체 자리를 정할 단서가 있다" 까지. 새 근거: encoder 선형 probe VOBench 99.7 (late 100) · IntPhys 1 98.9 · **IntPhys 2 74** (천장 아님); 가린 순간의 encoder 중간층은 '물체 없음'; 정확한 위치는 물체 토큰과 이웃에 쓰여 있고 표적 차의 물체 자리 몫은 3–5 % (L1 은 '무엇' 을 크게, '어디' 를 작게 본다).

> **결론** — frozen V-JEPA 2 ViT-H 의 문맥 encoder 출력 (z, predictor 의 입력) 에서 속도 · 위치 · 정체 · 가려진 물체의 정체, 그리고 과거만 보고도 미래 위치가 **선형에 가깝게 읽힌다**. predictor 출력 (p) 에도 정체는 남지만, p 는 그 정보를 **맞는 미래 자리에 놓지 않는다** (채점 민감도 4.46 · 이식 17.3 · 문맥 끝 가림에서 미래 위치 12–30 px). 쓸 문장은 정본 그대로: **"미래를 만드는 데 필요한 정보는 입력에 선형으로 있고, predictor 는 그것을 미래로 옮기지 않는다."**
> **근거** — (1) z → 문맥 끝 속도 ridge R² **0.996 / 0.989** (`z_research/RollOutV2/Archive/PREDICTOR_DISTANCE_LIMIT_2026-09-19.md` §5-3, 확정).
> (2) 과거 16 장만 본 z 위 12.8k 자가 v11 late 가림 k=1–4 미래 위치를 **5.4–10.1 px** (미래를 본 h 4.6–8.2 와 같은 급), 같은 자를 p 에 걸면 **16.9–29.9 px** (flat · ramp; static k1–4 는 11.8–15.4) (`auto_research/Archive/FUTURE_FROM_CONTEXT_2026-10-02.md` §2, 1 차).
> (3) 등가속 + 가림 p attentive probe shape **98.46** / color **99.49** / env 100 인데 같은 조건 채점 민감도 shape **4.46**; 비가림 → 가림 이식 color z 97.4 · h 99.2 · **p 17.3** (`z_research/IntPhysGenV11/Archive/surprising_score/RESULTS_2026-08-30.md` §7 · §8 · §11).
> **미결** — "encoder 충분 · predictor 병목" 이라는 **분업 귀속** 은 09-25 에 철회된 뒤 요인 분리로 다시 닫힌 적이 없다. 10-08 정본은 그것을 **정보 문장** (입력에 선형으로 있다) 으로 좁혀 다시 세웠고, 그 근거 (`FUTURE_FROM_CONTEXT`) 는 적대 검증 전 · 지도 자 · 파라미터 없는 교차검증 없음이다.
> 이 장이 받치는 논문 문장: P3 (iv) "encoder 에는 정보가 있다 — 실패는 predictor 의 미래 예측이다." · 원재료: G12 · G06 · G07 · G01 · G01b · G02a · G02b · G03 · G03b · G08b · G09 · G11 추출문 → 원 문서는 각 절 끝 `출처`.

---

## 0. 읽는 법 — 쓸 문장 · 못 쓸 문장 · 등급

| 구분 | 문장 | 근거 · 이유 | 출처 |
|---|---|---|---|
| ✅ 쓸 문장 (정본) | "미래를 만드는 데 필요한 정보는 입력에 선형으로 있고, predictor 는 그것을 미래로 옮기지 않는다." | z 자 ≈ h 자, 안 본 속도 · 가림 문맥에서도 | `auto_research/paper/PAPER_STORY_2026-10-08.md` §2-3 (iv) |
| ✅ | "정보 부족이 아니라 규칙 미적용" (memory 프레임 아님) | FFC 결론 2 | `auto_research/Archive/FUTURE_FROM_CONTEXT_2026-10-02.md` §3 |
| ✅ | "p 에도 정체는 남는다 — 정보는 있는데 맞는 미래로 놓지 않는다" | probe 98–100 vs 채점 4.46 | 정본 §2-3 (iv) |
| ✅ (범위 한정) | "encoder 가 충분하다" 는 **color · env · 위치 · 속도** 로 쓴다 | shape 은 이식에서 흔들림 (§1-5) | 정본 §2-3 (iv) |
| ❌ 금지 | "encoder 가 미래를 표현한다" · "낙하 시각을 안다" | 데이터에서 미래가 현재 상태의 결정적 함수 (ledge 낙하 시각 모든 clip 샘플 22 고정, wall 정지 자리 고정) → 자가 지도 라벨의 곡선을 붙인다 | `FUTURE_FROM_CONTEXT` §3 · 정본 §7 |
| ❌ 금지 | "p 에 정보가 없다" · "predictor 가 X 를 못 만든다" | p probe 98–100 (§2-1) | CLAUDE.md §6 · 정본 §7 |
| ❌ 철회 유지 | "encoder 는 충분하고 predictor 가 병목이다" (분업 귀속) | 충분성 대조가 진실 위치를 썼다; 요인 미분리 | `auto_research/README.md` §2 · `auto_research/Archive/TRAINED_PREDICTORS_SIGNATURES_2026-09-25.md` 정정 2-1 |
| ❌ 금지 | "목적함수 때문이다" · "기억 (memory) 문제다" · "state 를 잃는다" | discussion 한 문단 / 09-30 프레임 금지 | 정본 §7 · `auto_research/paper/RETHINK_PREDICTOR_ROLE_2026-09-29.md` §0-c |

- 등급 표기: **확정** (적대 검증 통과) · **1 차** (검증 전) · **시사** · **판정 보류** · **철회** · **기록** (원 문서에 등급 없음, 대체된 판 아님).
- 표현 약어: **z** = 문맥 encoder (`encoder`, online) 출력, 문맥 16 장만 봄 (캐시 `ctx_masked` 는 LN(z)) · **h** = target encoder (EMA) 32 장 전부 = 천장 대조 · **p** = predictor 미래 출력. 1 칸 = 16 px (합성 렌더 18 px).

---

## 1. encoder 정보 표 — 무엇이 z (와 h) 에서 읽히나

### 1-1. 합성 운동 — 속도 · 가속 · 위치

| 무엇 | 표현 | 수치 | 자 · probe | 검증 (held-out · 파라미터 없는 판) | 등급 | 출처 |
|---|---|---|---|---|---|---|
| 문맥 끝 속도 (RollOut_v2, 9 시나리오) | z / h | R² vx **0.996** / vy **0.989** / 속력 0.985 (h 0.997 / 0.993 / 0.990); MAE vx 0.51 px | 마지막 두 문맥 튜블릿 물체 3×3 평균 → ridge | block 반반 train 3,136 / test 3,136 | 확정 (정본 인용) | `z_research/RollOutV2/Archive/PREDICTOR_DISTANCE_LIMIT_2026-09-19.md` §5-3 |
| 경계 튜블릿 속도 (flat_v vx) | z t7 물체 3×3 (F1) / h / p 슬롯 0 | 시나리오 내 fit F1 **0.988** · h 0.989 · P0 0.992 (`EXPERIMENTS_2026-09-16.md` §2 표; RESULTS.md per-scenario 표 원값은 F1 0.989 · F1h 0.989 · P0 0.992, pooled fit 의 flat_v 줄은 0.988 · 0.990 · 0.989); flat_a 0.985 · fall vy 0.987 | 닫힌 형식 ridge, n=300/시나리오, OOF 5-fold block | 위치 지름길 POS7 −0.015 (flat_v); **파라미터 없는 쌍 검사** 같은 위치 쌍 centered cos 속도 비슷 **+0.498** [0.491, 0.505] vs 다름 **−0.177** (3,163 vs 12,777 쌍) | 기록 (정본 E1) | `z_research/context_encoder_analysis/exp_results/ctxenc_boundary_kinematics/RESULTS.md` · `z_research/context_encoder_analysis/Archive/EXPERIMENTS_2026-09-16.md` §2 |
| 배경 창 속도 | z16_bg (물체에서 ≥ 4 칸) | F1_bg 0.871 (flat_v); 물체 창 속도 부분공간 ~ 배경 창 속도 부분공간 겹침 **0.043** (랜덤 0.002, 천장 0.294) | ridge 가중치 열 span, 주각 mean cos² | split-half 천장 · 랜덤 하한 | 기록 | `…/ctxenc_state_subspaces_kinematics/RESULTS.md` (a)(c) |
| 문맥 끝 속도 (RollOut_v3, 궤적 84) | z_last | pooled R² vx .983 / vy .976 (\|v\| .967); **(법칙 × secondary) 그룹 안 vx .802 · \|v\| .949** (검증자 .955); LOSO .986 / .978; 기준선 법칙 one-hot .925 / .877 | ridge, 궤적 group 6 겹 | 궤적 군집 bootstrap; 물체 없는 창 t14 \|v\| **.905** → 물체 비특이 | 확정 (VERIFY_0925 M13, "읽힌다 ≠ 쓰인다") | `auto_research/Archive/M13_MEMORY_VELOCITY_SUFFICIENCY_2026-09-25.md` · `auto_research/Archive/VERIFY_0925_2026-09-25.md` §1 |
| 국소 속력 \|v_T\| (v3 평지 3 법칙) | z | R² 0.976–0.984 | ridge, 궤적 단위 split | 궤적 bootstrap | 확정 (H1 정정판 "읽기 ✅") | `auto_research/Archive/H1_ACCEL_IN_INPUT_2026-09-24.md` §4 |
| 가속 잔차 r8 (v3 flat_v/a/d) | z | **C8 0.01** [−0.22, 0.13] · **C16 0.51** [0.25, 0.65] · **C32 0.76** [0.62, 0.84]; 국소 가속 a_T C8 −0.02 / C16 −0.03 / C32 +0.33 | ridge, **궤적 단위 split** (법칙당 12 궤적) | 기준선 B_xfirst (창 시작 위치) C16 **0.97** → r8 은 "역사" 와 못 가름 | ◐ 1 읽기 (가속) — 긴 문맥에서만 | `H1_ACCEL_IN_INPUT_2026-09-24.md` §4 · 정정 |
| 가속 (v2 ramp_a · flat_a 교차 쌍) | z | ramp_a ax7 F1 0.825 이나 **POS67 → ax7 0.991** (경사각 = 장면 지름길); flat_v × flat_a 교차 쌍 has_a F1 1.00, **둘 다 정지인 대조 쌍도 AUC 0.20** 로 갈림 | ridge / 쌍 분류 | 정지 대조 쌍 | **무효 (못 잼)** — (v0 × a) 세트 필요 | `…/ctxenc_boundary_kinematics/RESULTS.md` · `z_research/context_encoder_analysis/Archive/EXPERIMENTS_2026-09-16.md` §2-1 |
| 위치 | z16_obj | x 0.99 / y 1.00 (pooled R²) | ridge | — | 기록 | `…/ctxenc_state_subspaces_kinematics/RESULTS.md` (a) |
| 위치 (자) | z · h (encoder 자) | v2 전 시나리오 ≤ 0.61 칸; v3 16 창 z · h 자 recall 96.7–99.6 %, 위치 8.0–10.0 px | attentive 위치 자 (3,842) / presence 자 | held-out 0.87 칸 (자 자체); '없다' 오작동 z 0.0–5.7 % | 확정 (자 검증) / v3 줄은 **대체된 판** (옛 presence 자 — 지금 자는 `identity_r8`) | CLAUDE.md §5-5 · `z_research/RollOutV3/Archive/RESULTS_2026-09-25.md` §1 |
| 위치 · 속도 · 시각 · 장면 얽힘 | z16_obj | 위치~속도 **0.019** (랜덤 0.002 / 천장 0.389) · 속도~시각 0.004 · 시나리오~시각 0.001 · 위치~시나리오 0.088 (유일한 약한 얽힘) | 부분공간 주각 | 랜덤 · split-half | 기록 | `…/ctxenc_state_subspaces_kinematics/RESULTS.md` (b) |
| 시각 서명 (튜블릿 index) | z16_obj · z16_bg · p · h · z32 | 모두 acc **1.000** (chance 0.125) | 분류 | — | 기록 — **단서로 읽는다** (§1-5) | 같은 곳 (a) |

⚠️ 0.996 / 0.989 (RollOutV2, 두 튜블릿 평균 특징) 와 0.988 / 0.989 (ctxenc E1, t7 3×3 특징 · F1 vs h) 는 **fit · 특징이 다른 두 수치** 다 — 모순 아님 (`z_research/context_encoder_analysis/Archive/STATE_EVOLUTION_CAPABILITIES_2026-09-19.md` §2 vs `ctxenc_boundary_kinematics`).
⚠️ RollOutV2 속도 probe 는 **물체 칸을 알려 줬다** — 두 튜블릿 위치 차로 속도를 계산했을 수도 있다. "predictor 입력에 그 정보가 있다" 는 결론은 같다 (`PREDICTOR_DISTANCE_LIMIT_2026-09-19.md` §5-3).
⚠️ 속도는 배경 창 (0.871) · 물체 없는 창 (.905) 에서도 읽힌다 → "물체 상태" 의 증거로는 약하다; "속도가 물체 창에 국소적" 은 n=30 표본 artifact 로 철회됐다 (`ctxenc_boundary_kinematics/n30/RESULTS.md` → n=300).

### 1-2. 정체 — 보이는 물체 · 가려진 물체

| 무엇 | 표현 | 수치 | 자 · probe | 검증 | 등급 | 출처 |
|---|---|---|---|---|---|---|
| 정체 (v1, 가림 없음) | z / p / h | 층별 선형 probe shape z **87.5** · predictor block 1 68.4 → block 12 85.5 → **proj (p) 88.7** · h 91.8 | 선형, held-out block | — | 기록 (08-20) | `z_world_model_analysis/MECHANISM_2026-08-20.md` §2.1 |
| 정체 · 장면 (v11 6 조건) | z / h / p | 가림 팔 OCC self z 99.9–100; color VIS→OCC 이식 3 팔 평균 **z 97.4 · h 99.2** (p 17.3 → §2-2); env self · 이식 · k 72 칸 **전부 100** | attentive (`attn_30`), block split 0.5 | 팔 **안** 이식 (팔 가로지르는 이식은 기대값이라 안 씀) | 기록 (정본 인용) | `RESULTS_2026-08-30.md` §7 · §8 · CLAUDE.md §5-2 · `auto_research/Archive/H7_V11_LAYERWISE_2026-09-24.md` 정정 (4) |
| 정체 · 타이밍 (v11_full 12 조건) | z / h | 비가림 head → late 이식 z · h **93–100** | attentive (`attn_probe_xfer`, 108 head) | — | 기록 | `z_research/IntPhysGenV11_occlusion_timing_ablation/README.md` §probing 전수 7 |
| 가려진 문맥 튜블릿의 정체 (z_hid) | z (late k ≥ 2, 물체 완전히 가려진 튜블릿) | shape **95.7** [94.6, 96.6] / color **93.9** [92.5, 95.0] (같은 clip 보이는 튜블릿 대비 −2.7 / −4.3) | 튜블릿 평균 풀링 + logistic (+ 독립 ridge 96.8 / 92.9) | 라벨 섞기 12 %; **외형 쌍둥이 없는 clip color 83.7 · 장면 hold-out 72.0** | 기록 (감사 통과, 자 무관) | `z_research/RollOutV3/Archive/RESULTS_2026-09-25.md` §6 · `z_research/RollOutV3/audit/v11_identity/AUDIT.md` |
| 경계 튜블릿의 정체 (가려진 경계) | z B7 (t7 진실 칸 3×3) / C (문맥 2048 토큰) | self late B7 87.2 / 78.9 / 74.7 · **C 98.2 / 99.3 / 99.3** (static / flat / ramp); **visible → late 이식 B7 20.2 [17, 24] / 16.0 [13, 19] / 32.9** (chance 14.3); 역이식 late → early 35.8 / 40.9 / 72.8 | 닫힌 형식 ridge, block held-out, CI 1000 | 라벨 셔플; 옛 6 셀 bit-identical 재현 | 기록 (E2 ◐) | `z_research/context_encoder_analysis/exp_results/ctxenc_boundary_identity_v2/RESULTS.md` (i)(iii)(iv) |

⚠️ z_hid self 95.7 과 경계 B7 이식 20.2 는 **모순이 아니다** — 앞은 가려진 조건 안에서 학습한 probe (정보 존재), 뒤는 보이는 조건 probe 를 가린 조건에 옮긴 것 (코드가 다르다). 맞는 문장: "가려진 물체의 정체는 z 에 남지만, 보이던 때와 **다른 코드** 로 · 문맥 전체에 퍼져 있다" (`ctxenc_boundary_identity_v2` 결론 · `EXPERIMENTS_2026-09-16.md` §3).
⚠️ h 도 가려진 창에서 z 보다 높지 않다 (static k3/k4 h:B7 52.7 / 58.5 vs z 71.6 / 74.3) — amodal 완성의 증거는 없다 (`ctxenc_boundary_identity_occlusion/RESULTS.md`).

### 1-3. 과거만 본 z 에서 미래 위치 — `FUTURE_FROM_CONTEXT` (10-01 ~ 02, 1 차)

설계: z (`ctx_masked`, LN, 과거 16 장) 위 attentive 자 — **자 A** 3,842 (튜블릿당 쿼리 1) · **자 B** 12,802 (슬롯별 쿼리 8, 토큰 전체 attend) → 미래 8 슬롯 (x, y). 같은 자를 p · h 에도. (1) RollOut_v2 가림 없음, holdout = **안 본 속도 레벨 2 개**, 학습 4,480 / test 1,792; (2) v11 late vanish pos_a 1,344 clip, **fit_all** = block 절반 학습 (condition × k 층화).

**(a) RollOut_v2 (MAE px, y / x)**

| 표현 · 자 | ledge | wall | arc | flat_a | ramp_a |
|---|---|---|---|---|---|
| CV (문맥 끝 속도 외삽, **진실 좌표**) | 24.8 / 0.2 | 0.0 / **79.9** (벽 뚫음) | 29.7 / 0.4 | 0 / 10.6 | 1.6 / 8.8 |
| TEMPLATE_SCEN (시나리오 oracle) | 8.4 / 21.1 | 0 / 0 | 39.4 / 10.5 | 0 / 33.9 | 1.5 / 8.6 |
| **z 자 B** | **3.3 / 5.1** | 2.7 / 4.7 | 4.4 / 4.5 | 2.4 / 4.0 | 2.9 / 4.0 |
| p 자 A | 8.2 / 8.2 | 3.2 / 5.3 | 9.2 / 5.7 | 3.2 / 8.4 | 3.0 / 6.5 |
| **p 자 B** | **5.4 / 5.7** | 4.0 / 4.8 | 7.3 / 4.5 | 4.2 / 7.2 | 3.3 / 4.7 |
| h 자 B (미래를 본 encoder) | 2.7 / 2.2 | 5.1 / 1.7 | 2.9 / 1.8 | 3.9 / 2.3 | 3.3 / 1.6 |

- ledge 슬롯 7 낙하량 (진실 60.1 px): z-B **60.8** (clip 상관 .90) · p-B 52.8 (.76) · p-A 41.4; flat 에서 같은 자는 낙하 −1 ~ +3 px (`FUTURE_FROM_CONTEXT` §1).
- **자 검증**: h 자는 **불가능 ledge (떠 있음)** 에서 낙하 13–16 px 만 (가능 57–59) — 자가 "ledge 면 무조건 떨어뜨림" 이 아니다 (같은 곳).

**(b) v11 late 가림 (유클리드 px, 288 px 화면, fit_all)**

| 표현 · 자 | stat k0 | flat k0 | ramp k0 | flat k1 · k2 · k3 · k4 | ramp k1 · k2 · k3 · k4 |
|---|---|---|---|---|---|
| CV (마지막 **보인** 슬롯 둘, 진실 좌표) | 0 | 0 | 15.5 | 0 · 0 · 0 · 0 | 21.9 · 21.9 · 30.1 · 30.1 |
| COPY (마지막 보인 자리) | 0 | 72.8 | 84.4 | 91 · 91 · 109 · 109 | 102 · 102 · 118 · 118 |
| **z 자 B** | 10.4 | 5.5 | 5.3 | **7.4 · 6.9 · 6.3 · 10.1** | **5.9 · 5.4 · 5.6 · 6.9** |
| **p 자 B** | 14.7 | 10.4 | 12.1 | 16.9 · 19.8 · 22.2 · **29.9** | 19.2 · 19.1 · 20.1 · 19.1 |
| h 자 B (상한) | 8.1 | 5.1 | 4.4 | 5.2 · 5.0 · 4.9 · 8.2 | 5.9 · 5.4 · 5.5 · 6.5 |

- 가려진 미래 슬롯만: z-B 5–15 · p-B 11–27 · h-B 4–10 · COPY 36–64 px. CI (ramp k4): z-B [5.8, 8.3] · p-B [17.8, 20.5] (`FUTURE_FROM_CONTEXT` §2).
- flat k4 슬롯별 p-B 10.8 → 61.1 px — p 가 물체를 잃은 뒤 자의 기본값 (같은 곳).
- **결론 (원문)**: "입력 (z) 에 선형으로 있는 것이 출력 (p) 에서 사라지는 조건은 **문맥 끝 가림**이다." 가림 없는 v2 에서는 p 자 B 로 대부분 나온다 (10-02 정정).

⚠️ **지도 자다** — 자는 장면 종류 · 상태를 읽고 라벨이 가르친 곡선 (템플릿) 을 붙인다. ledge 낙하 시각 · wall 정지 자리가 모든 clip 에서 같아, "encoder 가 담은 것" 과 "자가 계산한 것" 을 못 가른다 (§3). 변동 시각 데이터가 필요하다.
⚠️ **fit_vis 이식 (visible 로만 학습 → 가림 문맥) 은 z · p · h 셋 다 깨진다** (late 15–58 px, h 도 17–49) — 미래를 본 h 까지 깨지므로 정보가 아니라 자의 attention 문제 (CLAUDE.md §7-3). 결론은 fit_all 로만 읽는다.
⚠️ CV · COPY 는 **진실 좌표를 쓰는 참조선** 이지 경쟁 모델이 아니다. 지도 자 수치를 VoE · 자 없는 검사와 직접 비교하지 않는다.
⚠️ wall 에서 p 도 멈춤을 읽는다 — RollOutV2 위치 자 ("2 슬롯 통과 뒤 물체 잃음") 와 **다른 자** (장면 → 템플릿). 두 결과는 06 장 조건 표로 같이 낸다 (정본 §2-3 iii 조건 3).
⚠️ ramp 에서 z 가 CV 를 이기는 몫 (22–30 → 5–7) 은 자가 **가속** 을 반영한다는 뜻이지만, 문맥 8 튜블릿의 속도 변화인지 **경사 장면 읽기** 인지 안 갈랐다 (§1-1 가속 줄과 같은 교란).
⚠️ 원 문서에 **파라미터 없는 교차검증이 없다** — 정본 §4-5 의 "파라미터 없는 교차검증" 은 이 문서 밖 (§1-1 E1 쌍 검사 · §1-4 뒤집힘 비율) 을 가리키는 것으로 읽힌다 (확인 못 함, §4-4).

### 1-4. 실영상 · 외부 벤치마크

| 무엇 | 표현 | 수치 | 자 · probe | 검증 | 등급 | 출처 |
|---|---|---|---|---|---|---|
| 좌/우 운동 방향 (ssv2 722 · ntu 416) | z (문맥 encoder, 마스크 없음, 32 장) | ssv2 attentive 정방향 test **99.5** · **뒤집힘 비율 98.1**; tiny (3,842) 91.2 / 85.2; ntu attentive · tiny **100.0 / 96.2** | 정방향 clip 으로만 학습 → 역재생 시험 | **뒤집힘 비율 = 라벨 없는 교차검증**; 연기자 (P###) 단위 split (ntu) | 기록 (정본) | `z_research/context_encoder_analysis/Archive/ENCODER_TEMPORAL_DYNAMICS_2026-09-20.md` §3 |
| 좌/우 방향 (SSv2 677 clip) | z / p | z 0.89–0.97 (정) / 0.91–0.95 (역) vs **픽셀 기준선 0.41–0.70**; p 0.83–0.94, 문맥 끝 복사 수준 (0.92–0.97) 을 넘지 않음 | 로지스틱 probe (clip 7:3) | 픽셀 기준선 | 확정 (VERIFY_0925 E-SSv2) | `auto_research/Archive/E_SSV2_ENCODER_READS_PREDICTOR_STALLS_2026-09-25.md` §3-1 |
| 전역 ego-motion (EK100) | z 문맥 마지막 튜블릿 256 토큰 평균 | 전역 x / y R² **.69 / .63** (10-fold .709 / .651 · 영상 단위 split .685 / .614 · 64 토큰 부표본 .682 / .613) | ridge, clip 5-fold CV, 표적 Farneback flow | 셔플 ≈ 0; flow 표적 자기 일관성 clip-CV **.53 / .36** 보다 높다 | 1 차 → R² 는 VERIFY_R3 **확정** | `auto_research/Archive/ENCODER_MOTION_READOUT_2026-09-29.md` §2 · `auto_research/Archive/VERIFY_R3_2026-09-29.md` B |
| EK100 미래 변위 중 카메라 몫 | — | 슬롯 0 / 3 / 7 = .42 / .65 / .74 → **67–75 %** (화면 안 출처만 .67) | 같은 판 | VERIFY_R3 B3 정정 ("74 %" → 범위) | 1 차 (정정 반영) | 같은 곳 |
| 국소 · 국소 잔차 (물체 운동) | z | 국소 x / y EK100 .42 / .33 (SSv2 .29 / .05); **국소 잔차 EK100 .04 / .09** · SSv2 .22 / .01 | 토큰 하나 → 칸별 flow | — | 1 차 — encoder 한계인지 표적 한계인지 미결 | `ENCODER_MOTION_READOUT_2026-09-29.md` §2 단서 |
| IntPhys 2 Main 위반 (506 쌍) | z T×N 평균 1280-d | 쌍 **73.7** [69.9, 77.5] · 영상 정확도 62.5 (라벨 섞기 48.5–53.6) · AUC 0.63 vs **VoE 릴리즈 쌍 54.3**; immutability 84.2 / permanence 78.3 / continuity 73.3 / **solidity 61.6** | 표준화 + L2 로지스틱, 장면 단위 외부 5-fold | 라벨 섞기 10 회; 풀링 변형 4 개 ±4 pt | 기록 (단일 문서, 지도 자) | `auto_research/Archive/INTPHYS2_ENCODER_LINEAR_PROBE_2026-09-30.md` |
| IntPhys 1 dev 층별 linear probe (Joseph 2026 재현) | ViT-**L** target encoder 층별 raw `x_N` mean pool | peak **L15 80.83 ± 3.98** (논문 chart ~95–100 @ L8–10), L0 58.06, L23 77.22 | Linear(1024, 2), 5-fold quadruplet CV, 20 HP | — | 기록 (ViT-H 기준 아님) | `analysis/intphys1/PAPER_VS_OURS.md` §2 |
| IntPhys 1 마지막 층 probe (07-12 감사) | ViT-L L23 / ViT-H L31 | **73.61 / 80.56** (같은 하네스 surprise 와 격차 27.2 / 20.6) | linear, 288/72 split, 20 epoch | — | 기록 (IntPhys 2 식 프로토콜, 현 88.89 와 자가 다름) | `analysis/intphys2/ANALYSIS_NOTES.md` §9 · §13 |

⚠️ ENCODER_TEMPORAL_DYNAMICS 는 **"방향" 과 "끝 자리" 를 못 가른다** — 마지막 시점 자리만 읽어도 역재생하면 뒤집힌다. 가르는 대조 (`static_last` · `static_first` · `shuffle`) 는 구현돼 있으나 2026-09-20 사용자 결정으로 **돌리지 않았다**. 쓸 수 있는 문장은 "좌/우 라벨을 가르는 신호가 encoder 표현에 있고, 그 신호는 시간 순서를 뒤집으면 부호가 바뀐다" 까지 (같은 문서 §4-2).
⚠️ 역재생 정확도는 독립된 수가 아니다 (ssv2 tiny 93.1 = 84.7 + 8.3) — 믿을 수는 정방향 정확도와 뒤집힘 비율 둘뿐 (같은 문서 §3 정정, 09-20).
⚠️ ENCODER_MOTION 의 "표적 신뢰도 천장까지 읽는다" 는 **정정** — 원 값 .70 / .50 은 train = test 최소제곱이었다. "flow 표적의 자기 일관성보다 높게 읽는다" 로 쓴다 (`VERIFY_R3_2026-09-29.md` B2). 토큰 평균 판독은 자리를 버린 것이라 predictor 입력에 **쓰기 좋은 형태** 인지는 별개다.
⚠️ IntPhys 2 probe 는 impossible 라벨을 보고 배운 지도 자 — VoE 와 같은 능력이 아니고 "위반 후 장면의 통계" 를 배울 수 있다. VoE 와 프레임 배치도 다르다 (probe 9.3 s 한 창 vs VoE 슬라이딩 9 창). "encoder 가 물리를 이해한다" 는 문서도 금지로 명시 (`INTPHYS2_ENCODER_LINEAR_PROBE_2026-09-30.md` 단서).
⚠️ SSv2 는 V-JEPA 2 사전학습 데이터에 포함 — 실영상 결과는 분포 안 확증일 뿐이다 (`E_SSV2_…` 단서).

### 1-5. encoder 정보의 단서 (결론과 같은 무게)

| 단서 | 내용 | 그래서 못 쓰는 문장 | 출처 |
|---|---|---|---|
| shape 은 encoder 도 흔들린다 | v11 VIS→OCC **이식** shape 3 팔 평균 z **64.8** (등속 팔 42.4) · h 75.7 · p 27.3 (color z 97.4). ⚠️ 이 64.8 은 **이식** 수치다 — 가림 조건 안 self 는 z 99.9–100 (H7 정정 4: "§5-2 를 잘못 인용했다") | "encoder 는 조건과 무관하게 정체를 같은 코드로 싣는다" 는 color · env 로만 | `RESULTS_2026-08-30.md` §8 (l.1199 · l.1467) · `H7_V11_LAYERWISE_2026-09-24.md` 정정 (4) · CLAUDE.md §5-2 · 정본 §2-3 (iv) |
| 가속은 ◐ | 국소 가속 a_T ≈ 0 (C8 · C16); r8 은 C32 에서만 0.76 이고 창 시작 위치 기준선이 0.97 → "가속 표현" 과 "역사" 미분리; v2 가속은 시나리오 지름길 | "가속은 z 에 있다" (철회 — H1 첫 판, clip 단위 split 누수) | `H1_ACCEL_IN_INPUT_2026-09-24.md` 정정 · `ctxenc_boundary_kinematics/RESULTS.md` |
| 속도는 물체 비특이 | 배경 창 0.871 · 물체 없는 창 .905 · p 비물체 창 .94 | "물체 상태를 받아 둔다" (state 프레임 — 현재 금지) | `M13_…` 정정 (3) · `ctxenc_state_subspaces_kinematics` (c) |
| 시각 서명 | 튜블릿 index 가 모든 표현에서 100 % — 배경 토큰만으로도 시간 정렬 argmin 이 슬롯 5 까지 대각선 | 시간 정렬 대각선 = 물체 증거 (gated Δc 로만) | `z_research/context_encoder_analysis/exp_results/ctxenc_time_alignment_gated/RESULTS.md` · `EXPERIMENTS_2026-09-16.md` §1 |
| 지도 자 | FFC · IntPhys 2 probe · attentive probe 는 라벨이 가르친 곡선 / 통계를 배울 수 있다 | "encoder 가 미래를 표현한다" | `FUTURE_FROM_CONTEXT` §3 |
| 외형 쌍둥이 · 장면 누수 | z_hid color 93.9 → 쌍둥이 없는 clip 83.7 → 장면 hold-out 72.0 | 정체 수치를 쌍둥이 · 장면 단서 없이 단독 인용 | `RollOutV3/Archive/RESULTS_2026-09-25.md` §6 |
| 읽힌다 ≠ 쓰인다 | 선형 readout 은 정보 존재만 보인다 (Hewitt & Liang) | "predictor 가 그 정보를 써야 하는데 안 쓴다" 를 readout 만으로 | `M13_…` 못 하는 주장 · `auto_research/Archive/M5_VELOCITY_STEERING_2026-09-25.md` |

출처 (§1): `z_research/RollOutV2/Archive/PREDICTOR_DISTANCE_LIMIT_2026-09-19.md` · `z_research/context_encoder_analysis/exp_results/{ctxenc_boundary_kinematics,ctxenc_state_subspaces_kinematics,ctxenc_boundary_identity_v2,ctxenc_boundary_identity_occlusion,ctxenc_time_alignment_gated}/RESULTS.md` · `z_research/context_encoder_analysis/Archive/{EXPERIMENTS_2026-09-16,STATE_EVOLUTION_CAPABILITIES_2026-09-19,ENCODER_TEMPORAL_DYNAMICS_2026-09-20}.md` · `z_research/IntPhysGenV11/Archive/surprising_score/RESULTS_2026-08-30.md` · `z_research/RollOutV3/Archive/RESULTS_2026-09-25.md` · `z_research/RollOutV3/audit/v11_identity/AUDIT.md` · `z_world_model_analysis/MECHANISM_2026-08-20.md` · `auto_research/Archive/{H1_ACCEL_IN_INPUT_2026-09-24,H7_V11_LAYERWISE_2026-09-24,M13_MEMORY_VELOCITY_SUFFICIENCY_2026-09-25,E_SSV2_ENCODER_READS_PREDICTOR_STALLS_2026-09-25,ENCODER_MOTION_READOUT_2026-09-29,VERIFY_R3_2026-09-29,INTPHYS2_ENCODER_LINEAR_PROBE_2026-09-30,FUTURE_FROM_CONTEXT_2026-10-02}.md` · `analysis/intphys1/PAPER_VS_OURS.md` · `analysis/intphys2/ANALYSIS_NOTES.md`

---

## 2. p 에도 정보는 남는다 — 그러나 맞는 미래로 놓지 않는다

### 2-1. 해리: p probe self 는 천장, 채점은 바닥

| 판 | 조건 | p probe self (정체) | 같은 조건 채점 | 자 · probe | 등급 | 출처 |
|---|---|---|---|---|---|---|
| v11 (`surprise_c16t32` / `attn_probe`) | 등가속 + 가림 (ramp+OCC) | shape **98.46** · color **99.49** · env 100 | shape 민감도 **4.46** · color 21.43 · vanish 0.00 | attentive `attn_30`, 가능 · 물체 clip 만 | 기록 (정본 인용) | `RESULTS_2026-08-30.md` §11-(1) · CLAUDE.md §5-2 |
| 〃 | 등속 + 가림 (flat+OCC) | shape 98.48 · color 98.98 | shape 민감도 6.85 | 〃 | 기록 | 같은 곳 §7 · §11 |
| 〃 | 정지 + 가림 (sta+OCC) | shape 85.13 · color 87.55 (head 미수렴 train_acc 0.954 / 0.931) | shape 4.76 · vanish **+20.54** (유일하게 살아남음) | 〃 | 기록 — **미해결** | 같은 곳 §7 · §11-(5) |
| v11_full (타이밍 12 조건) | late (경계 가림) | shape 95.2–98.9 · color 86.3–99.2 · env 36 칸 100 (vis / early / mid shape 100) | late 전체 **59.5** (visible 87.2) | `attn_probe_xfer` 108 head | 기록 | `IntPhysGenV11_occlusion_timing_ablation/README.md` 전수 6 |
| v10 | 이동 + 가림 | shape **98.96** · color 93.52 · env 전 조건 100 | shape 51.6 · color 73.4 | attentive | 기록 | `z_research/IntPhysGenV10/Archive/V10_2026-08-28.md` §2 |
| IntPhysGen v1 (가림 없음) | — | 선형 probe 를 채점 규칙으로: shape p **100.0** (164 쌍) · color 98.3 | L1 채점 shape 52.9 · color 65.3 | 층별 선형 probe | 기록 (08-20) | `MECHANISM_2026-08-20.md` §2.2 |
| v11_full (풀링 ridge, 층 12 개) | late | shape 77.9–96.3 · color 69.3–95.5 (최저는 L0–L2) | late L11 59.4 | 슬롯 0/3/7 풀링 ridge | 기록 (H7 정정판) | `H7_V11_LAYERWISE_2026-09-24.md` §3-B1 |

⚠️ 용어 — v11 timing 의 `shape consistency / colour consistency` 는 **채점 축** 이고 probing 의 `shape / color` target 과 다르다 (같은 조건 p probe 98.2 vs 채점 54.5) (`IntPhysGenV11_occlusion_timing_ablation/README.md` 단서).
⚠️ probe head 는 과대용량 (49.3 M) 이라 self 는 포화한다 — 대조는 **천장에 안 닿은 이식 줄** 에서 읽는다 (CLAUDE.md §7-3). probing 은 가능 변이만 쓴다 (불가능 변이를 넣으면 z · p 가 비트 동일해 오염, CLAUDE.md §1-4).

### 2-2. 이식: 같은 팔 안에서 비가림 → 가림 readout 이 p 에서만 깨진다

| 판 | 비교 | z | h | **p** | 등급 | 출처 |
|---|---|---:|---:|---:|---|---|
| v11 팔 안 VIS → OCC (3 팔 평균) | color (chance 12.5) | 97.4 | 99.2 | **17.3** | 기록 (정본 인용) | `RESULTS_2026-08-30.md` §8 |
| 〃 | shape (chance 14.3) | 64.8 | 75.7 | 27.3 | 기록 | 같은 곳 |
| 〃 역방향 OCC → VIS | shape / color | — | — | 82.7 / 88.8 | 기록 | 같은 곳 |
| v11 k 분해 (등가속 팔) | shape p k1 → k4 | — | — | 45.0 → 15.5; color 고유 예측 클래스 7 → 5 → **1 → 1** (전부 orange) | 기록 | 같은 곳 §10 |
| v11_full 타이밍 | color 비가림 head → early / mid / **late** (static / flat / ramp) | late 93–100 | late 93–100 | early 98.6 / 99.9 / 100 · mid 98.2 / 99.4 / 98.9 · **late 15.1 / 20.6 / 16.4** | 기록 | `IntPhysGenV11_occlusion_timing_ablation/README.md` 전수 7 · 9 (붕괴는 k=0 → 1 한 칸) |
| v11_full h → p (가림 없음에서도) | color visible (static / flat / ramp) | — | (h→h ~100) | **94.9 / 79.0 / 57.3** | 기록 | 같은 곳 전수 8 |
| v10 h → p | ridge (ALL) | — | self 98.6–99.9 | **15.08** (조건별 15–25); 개념 벡터 cos diag 0.70–0.79 (off −0.06 ~ −0.08, ≈ 28σ) | 기록 | `z_research/IntPhysGenV10/Archive/GEOMETRY_2026-08-29.md` §4 |
| v11 풀링 · 중심화 이식 (비가림 → late) | shape h_t7 / z / p_t7 | 52.4 | 56.5 | 35.2 → **h − p +21.3** [+19.6, +23.1]; early 에서도 +13.3 | 기록 (late 전용 아님) | `H7_V11_LAYERWISE_2026-09-24.md` §3-B2 |
| v11 풀링 (RollOutV3 감사) | encoder 머리 → p raw (shape) | — | — | logistic 41.9 / ridge 18.0; **clip 짝 Procrustes 로 99.8 / 96.4 회복** | 기록 (감사 통과) | `z_research/RollOutV3/audit/v11_identity/AUDIT.md` §2 (6) |

⚠️ **이식 실패는 정보 부재가 아니다** — clip 짝 직교 사상으로 95–99 % 회복되고 (RollOutV3 감사), p self 는 98–100 이다. 맞는 문장은 "p 는 같은 정보를 encoder 와 **다른 자리 · 좌표계** 에 둔다" (정본 §2-3 iv, `IntPhysGenV11_occlusion_timing_ablation/README.md` 결론 9).
⚠️ 팔을 **가로지르는** 이식 실패는 증거가 아니다 — 기대값 (attention query 가 위치 특이적) 이다. 정지 → 이동 70.1 vs 이동 → 정지 17.3 도 같은 이유 (`RESULTS_2026-08-30.md` §12 철회).
⚠️ 이식 수치는 probe 종류에 크게 흔들린다 — 독립 ridge 는 방향만 같고 크기가 다르다 ((5) visible → late 18.9 / 40.9), λ 를 바꾸면 5–13 pt 움직인다 (`RollOutV3/figures/v11_probe/PROBE.md` §6 (λ 민감도) · §8 (독립 ridge 18.9 / 40.9) · 감사). 크기보다 **p 만 깨진다는 방향** 을 읽는다.

### 2-3. p 의 기하 — 저차원 부분집합 · 토큰 배치 · 걸음 방향

| 측정 | 수치 | 해석 (기술까지) | 등급 | 출처 |
|---|---|---|---|---|
| participation ratio | p **15.4–18.7** vs z / h 59.6–65.9; p top-16 분산 0.723–0.782 (h 0.406–0.422) — v11_full 9 set (static · flat · ramp × visible · late · early); ramp 빼고 6 set 인 `ctxenc_subspace_overlap` 은 p 15.8–18.7 · z / h 62.2–65.9 | p 는 h 보다 훨씬 적은 방향을 쓴다 | 기록 | `z_research/context_encoder_analysis/exp_results/ctxenc_subspace_v2/RESULTS.md` §4 · `ctxenc_subspace_overlap/RESULTS.md` · `CLAIM_NO_STATE_EVOLUTION_2026-09-15.md` §2-6 |
| p–h 주각 겹침 (top-16) | **0.619–0.667** (split-half 0.93–0.99, 랜덤 0.013); visible / late / early 무관; p top-16 의 h top-256 포함 0.896–0.907 | "p ∉ span(h)" 가 아니라 "h 상위 공간 안의 **저차원 부분집합**" (가림 무관) | 기록 | 같은 곳 · `ctxenc_subspace_v2/RESULTS.md` §4 |
| CKA (p, h) | 같은 clip · 토큰 0.475–0.551 vs 다른 clip 같은 토큰 index 0.403–0.469 | 위치 (토큰 index) 가 지배 | 기록 | `ctxenc_subspace_v2` §4 |
| 채점 거리 안 운동 부분공간 몫 | Σ‖P_S(p−h)‖² 몫 — 운동 · 시각 · 장면 합집합 16-d (`union_kin_time_scen`, h8 에서 맞춘 부분공간, 전 미래 토큰) **0.0213** (랜덤 0.0125, PCA top-16 0.3409); 위치 2-d 0.0028 · 속도 2-d 0.0023 | 채점 L1 의 대부분은 운동 상태가 아닌 방향의 차 | 기록 | `ctxenc_state_subspaces_kinematics/RESULTS.md` (e) |
| 정체 부분공간 (shape) | encoder 0.94–0.96 vs p 0.78–0.81 — **visible 에서 이미** | 가림이 만든 차이가 아니다 | 기록 (감사) | `RollOutV3/audit/v11_identity/AUDIT.md` §4 |
| 토큰 배치 (v1) | corr(Mh, Mp) shape **+0.002** / color −0.056; Mh 상위 1 % 집중 1.9 % (물체 위치에 안 몰림); ‖Ep‖/‖Mh‖ **1.150**; 개념 벡터 cos(c_p, c_h) +0.823 | 방향은 정렬, **토큰별 배치가 무관** → 토큰 단위 L1 에서 신호가 오차에 묻힌다; 토큰 평균 L1 만으로 shape 52.9 → **88.2**, 중심화 · 표준화 L2 97.1 | 기록 (08-20) | `MECHANISM_2026-08-20.md` §2.3 · §4 · §5 |
| 걸음 방향 (v10) | cos(step = p − z, right) − cos(step, wrong) **+0.000 ~ +0.011** (걸음의 1 % 미만); vanish 이동 + 가림 −0.007; \|p − z\| / \|h − z\| **79 %** | p 는 문맥 복사가 아니다 (79 % 움직인다) — 그러나 옳은 / 틀린 미래 쪽을 거의 가르지 않는다 | 기록 | `GEOMETRY_2026-08-29.md` §2 · §3 |
| margin 효과크기 (v11_full 12 셀) | late margin (% of base) 0.08–0.77 vs visible 0.27–3.73; ρ(효과크기, acc) **0.986** | early / mid 는 신호가 묽어짐, late 는 거기에 자리 변화가 더해진 다른 고장 | 기록 (α 가 아니라 margin) | `IntPhysGenV11_occlusion_timing_ablation/README.md` 전수 10 |

⚠️ "표현이 회전했다" 는 **측정했고 아니었다** → "저차원 부분집합" 으로 쓴다 (`z_research/IntPhysGenV11/Archive/PAPER_STORY_2026-09-22.md` · `STATE_EVOLUTION_HYPOTHESES_2026-09-15.md` §7).
⚠️ 토큰 평균은 시간 · 공간 축을 버려 **위치 · 궤적 위반은 원리적으로 못 잡는다** — 88.2 / 97.1 은 정체 위반에서만 (`MECHANISM_2026-08-20.md` §5 단서; `auto_research/Archive/COPY_VS_POSITION_2026-10-07.md` 가 실측).
⚠️ v10 GEOMETRY · MECHANISM 의 α 분해 · 보정 · 헤지 절은 **인용 금지** — 위 표에는 α 가 아닌 양만 옮겼다.

### 2-4. frozen 정렬 · 증폭으로는 작게만 돌아온다

| 개입 | 수치 | 대조군 | 등급 | 출처 |
|---|---|---|---|---|
| 전역 선형 정렬 사상 W: p → h (ridge, visible possible 960 clip 맞춤, held-out 288 block/조건) | POOLED visible 88.9 → **91.9** (+3.0 [2.1, 4.0]) · **late 61.2 → 64.4 (+3.2 [1.8, 4.8])** · early 81.1 → 87.8 (+6.7 [5.4, 8.1]); 가족별 ridge late 66.7 (+5.6) | 다른 clip 의 h 로 맞춘 ridge / z → h ridge: late +0.9 [−0.8, 2.7] / +1.4 [−0.4, 3.4] (0 포함), visible −13.4 / −8.9 | 기록 (09-16 "기각이 아니라 부분 효과") | `ctxenc_subspace_v2/RESULTS.md` §0 |
| 같은 W 의 토큰 L1 감소 | ≤ 9–11 % (z → h 대조 31 %) | — | 기록 | 같은 곳 |
| 재학습 대비 | post-FT `v11_postft` late 이득 +25 ~ +30 → 정렬 사상은 그 **약 1/5** | — | 기록 | 같은 곳 · `PAPER_STORY_2026-09-22.md` §4-1 |
| 균등 증폭 p' = mu + γ(p − mu) | 천장 51–65 % (v10 이동 + 가림 shape 64.8); z 기준 외삽은 잘 맞던 static_visible **98.4 → 59.4** 로 부숨 | — | 기록 — CLAUDE.md §6 "세기를 키우면 된다" 기각 | `GEOMETRY_2026-08-29.md` §6a |

⚠️ 정렬 재계산은 bf16 probing 캐시 위라 raw 정확도가 fp32 채점기와 다르다 (static late 63.4 vs 58.2). **Δ 는 믿되 정식 수치는 fp32 재추출 후** (`ctxenc_subspace_v2` §1 · `CLAIM_…` §2-7 단서).
⚠️ 정렬 사상은 visible clip 짝으로 맞췄다 — "자리 하나 사상으로 가림이 고쳐진다" 는 서지 않는다 (late +3.2 뿐).

### 2-5. p 의 정체 · 방향 정보는 복사 수준이다

| 판 | p | 복사 기준선 (문맥 끝 z) | 차 | 등급 | 출처 |
|---|---|---|---|---|---|
| v11 late (k ≥ 2) 풀링 정체 probe | p_fut shape 97.3 · color 97.2 (튜블릿별 94.2–97.0 평평) | z_all 99.0 · 98.2 | p − z_all **−1.6 / −0.8** (CI < 0); [z_all, p] − z_all −0.2 ~ +0.5 (더해도 안 는다) | 기록 (감사) | `RollOutV3/Archive/RESULTS_2026-09-25.md` §6 · `audit/v11_identity/AUDIT.md` §3 |
| 〃 튜블릿 하나끼리 | p_fut − z_t7 color +3.0 [1.9, 4.1] | — | 풀링 크기 다름 (8 vs 1) — 같은 크기면 p ≈ z_t7 이거나 낮음 | 기록 | 같은 곳 |
| SSv2 좌/우 방향 | p 0.83–0.94 | 문맥 끝 복사 0.92–0.97 | 넘지 않음 | 확정 (E-SSv2) | `E_SSV2_…` §3-1 |
| 고리 판정 (VERIFY 1 차) | "(b) p 정보 — 약하게 (복사 수준)" | — | — | 확정 (판정) | `VERIFY_0925_2026-09-25.md` 결론 |

### 2-6. 이 절이 세우는 문장

- ✅ "p 에 정체 · 장면 정보는 남는다 (self 98–100, env 100). 그러나 p 는 그것을 h 와 같은 자리 · 좌표계에 두지 않고 (이식 17.3, h → p 57–95, 저차원 부분집합), 맞는 미래 쪽으로 기울이지 않는다 (걸음 방향 차 < 1 %, 채점 민감도 4.46)."
- ✅ "p 의 정보는 문맥 끝 복사보다 많지 않다" (복사 기준선 대비 −1.6 / −0.8, SSv2 복사 넘지 않음).
- ❌ "predictor 가 정보를 잃는다" (층별 probe 단조 증가, proj 88.7) · ❌ "p 에 정보가 없다" (철회) — §5.
- ❌ "가림이 표현을 망가뜨린다" — env 72 칸 100, z · h 이식 93–100.

출처 (§2): `z_research/IntPhysGenV11/Archive/surprising_score/RESULTS_2026-08-30.md` · `z_research/IntPhysGenV11_occlusion_timing_ablation/README.md` · `z_research/IntPhysGenV10/Archive/{V10_2026-08-28,GEOMETRY_2026-08-29}.md` · `z_world_model_analysis/MECHANISM_2026-08-20.md` · `z_research/context_encoder_analysis/exp_results/{ctxenc_subspace_overlap,ctxenc_subspace_v2,ctxenc_state_subspaces_kinematics}/RESULTS.md` · `z_research/context_encoder_analysis/Archive/CLAIM_NO_STATE_EVOLUTION_2026-09-15.md` · `z_research/RollOutV3/Archive/RESULTS_2026-09-25.md` · `z_research/RollOutV3/audit/v11_identity/AUDIT.md` · `z_research/IntPhysGenV11/Archive/PAPER_STORY_2026-09-22.md` · `auto_research/Archive/{H7_V11_LAYERWISE_2026-09-24,E_SSV2_ENCODER_READS_PREDICTOR_STALLS_2026-09-25,VERIFY_0925_2026-09-25}.md`

---

## 3. predictor 기제 — 상관에서 인과 순으로

"predictor 가 실패한다" 를 받치는 기제 측정. 위에서 아래로 갈수록 개입 (인과) 에 가깝다. **어느 것도 "그 간선 · 층이 X 를 나른다 / 정한다" 를 세우지 않는다** — knockout · bias 는 분포 밖 개입이다 (`H8_KNOCKOUT_PERSISTENCE_2026-09-24.md` §5).

### 3-1. 상관 ① — attention 관찰

| 측정 | 판 | 수치 | 읽기 | 등급 | 출처 |
|---|---|---|---|---|---|
| 층별 질량 | v11 ViT-H, 6 조건 × 32 clip | **layer 0 에서 미래 토큰 → 문맥 질량 0.885**, 문맥 마지막 프레임에 몰림 (부호 Δt −12.4 프레임); 층이 깊을수록 미래 쪽으로 퍼짐; 6 조건이 거의 완전히 겹침 | 집계 수준에서 attention 은 가림 · 운동과 무관 | 기록 (관찰) | `z_research/IntPhysGenV11/figures/attention/README.md` · `analysis/attention/README.md` |
| **질량 최대 층 ≠ 피해 최대 층** | 같은 판 | 질량 최대 = 층 **0** (0.885) · 끊었을 때 가장 아픈 층 = 층 **8** (Δacc −18.06) | 관찰 도구와 개입 도구가 다른 답 — attention 질량으로 기제를 말하지 않는다 | 기록 | `analysis/attention/README.md` |
| 국소 헤드 위치 | 같은 판 | 층 0–2 에 국소 헤드 (0.4–2.3 패치), 층 8–11 은 5–8 패치로 균질 | 인코더의 "Physics Emergence Zone" (Joseph 2026, 중간층) 과 같은 현상으로 읽지 말 것 | 기록 | `z_research/IntPhysGenV11/figures/attention/README.md` |
| 진실 칸 미래 질의 → 문맥 물체 궤적 질량 (RollOut_v2 flat, 거리별) | 200 clip × 4 시나리오 | 0–20 px 0.040 · **20–40 px 0.073** · 40–60 0.056 · 60–80 0.032 · **80–200 px 0.021** (원시 균등 ≈ 0.018); 자기 자리 기둥 0.084–0.203 (균등의 4.6–11 배); arc 80–200 px **0.006** | 멀면 물체의 과거 자리를 **균등 수준** 으로만 본다 | 기록 (정본 인용) | `PREDICTOR_DISTANCE_LIMIT_2026-09-19.md` §5-1 · CLAUDE.md §5-5 |
| **attention 거리 감쇠 (v3, 회귀)** | release vs pv1, 법칙 · 팔 고정효과 | log ×U **칸당 (거리만) release −0.42** [−0.45, −0.40] vs pv1 −0.30 (e-folding 2.4 → 3.3 칸); 슬롯당 release −0.096 vs pv1 −0.014 (0 포함) | reach 가 긴 predictor 는 attention 이 거리로 더 천천히 준다 | **1 차 · 상관** | `auto_research/Archive/SC1_REACH_NOT_HORIZON_2026-09-25.md` §3-5 · `auto_research/README.md` 논리 사슬 4 행 |
| 같은 감쇠 (거리 + 슬롯 동시) | release, 3 stride | log ×U 칸당 **−0.293** [−0.319, −0.260] · 튜블릿당 −0.096 (R² 0.821); 팔 안 거리–슬롯 상관 0.89–0.92 | 거리와 슬롯이 얽혀 있다 | 시사 (M4 정정판) | `auto_research/Archive/M4_RETRIEVAL_RADIUS_2026-09-25.md` |
| ring 귀무 대비 물체 선택성 | v3 s1 / s2 / s4 | obj − ring (×U): Chebyshev d ≤ 3 에서 **+2.3 ~ +11.7**; s1 d 6+ **+0.03** [−0.03, 0.08] | 가까운 곳에서는 방향을 골라 보고, 멀면 고르지 않는다 | 시사 | 같은 곳 · `VERIFY_0925_R2_2026-09-25.md` (M4 정정판) |
| 장면 단위 출처 attention (K400 인공 팬) | release / Ariel | release p_src 1–2 칸 .0171 (p_src/p_ring 5.1) → 9–13 칸 .0011 (2.4), **약 15 배 감소**; 6–9 칸부터 p_copy > p_src (교차점 ≈ release R50); Ariel p_src 3–6 배 큼 | 출처 조회 강도가 거리로 준다; 넘어가면 복사 쪽 | 확정 (C4, VERIFY_ISCENE) | `auto_research/Archive/I_SCENE_CAUSAL_BOTTLENECK_2026-09-25.md` §4-1 |
| 실영상 방향 선택성 | SSv2 · EK100 | p_src/p_ring 팬 4.6–9.0 · SSv2 1.2–2.3 · EK100 1.3–1.5; **토큰별 중앙값 SSv2 0.8–1.2 · EK100 0.6–0.9** | 실영상에서는 "일부 토큰만" 출처 방향을 고른다 | 확정 + 정정 (평균만이 아님) | 같은 곳 §4-2 · 정정 (7) |

⚠️ attention 은 head 평균 질량 — 정보 흐름이 아니다 (`PREDICTOR_DISTANCE_LIMIT_2026-09-19.md` §7). 질의 0..7 은 문맥, 8..15 는 mask token — 섞어 읽지 않는다 (`analysis/attention/predictor_attn/README.md`).
⚠️ 문서의 "방향은 안다 / 모른다" 는 attention 선택성의 기술이다 (내부 귀속 표현 — 현재 금지 소지, `I_SCENE` 단서).

### 3-2. 상관 ② — 층별 렌즈 (파라미터 0)

| 측정 | 수치 | 읽기 | 등급 | 출처 |
|---|---|---|---|---|
| v11_full 12 층 렌즈 채점 (S_l = mean \|proj(norm(q_l)) − LN(h)\|) | 전체 L0 54.6 … L10 75.6 / **L11 75.9**; late L11 59.4; **late L10 − L11 +0.1** [−0.4, +0.6], held-out 층 선택 이득 +0.0 | late (경계 가림) 의 무너짐은 **층을 바꿔 읽어도 돌아오지 않는다** | 기록 (H7 정정판) | `H7_V11_LAYERWISE_2026-09-24.md` §3-A1 · A3 |
| 예외 static × color | early L1 85.9 vs L11 58.6 → +27.2 [+23.5, +31.1] (held-out +27.5); mid +28.3 | 초기층 렌즈가 크게 낫다 — prior 아님은 시사까지 | 기록 | 같은 곳 |
| 원판 "단조 증가, L11 최고" | 비단조 (L1 57.1 → L2 55.2, L3 58.2 → L4 57.2) | 원판은 L0/4/6/8/10/11 만 골라 dip 을 건너뛰었다 → **정정** | 정정 | 같은 곳 정정 (1) |
| RollOut_v3 먼 슬롯 위치 적중 (귀무 보정) | C16/P32 슬롯 10: L11 적중 − 귀무 **−0.13**; L0–L4 는 슬롯 9–11 에서 +0.03 ~ +0.09; L6 이후 귀무 이하 | 먼 슬롯에서 귀무를 넘는 것은 초기층뿐 | 기록 (h23b) | `auto_research/exp_results/h23b/REPORT_h23b.md` §2 · `H23_PREDICTOR_LAYERS_TRAJECTORY_2026-09-24.md` |
| 공통 문턱 물체다움 층 차이 | C16 θ_L6 슬롯 10 L6 0.47 / L11 0.44; θ_L11 0.09 / 0.11 — **층 차이 없음** | "뒤층이 먼 물체를 지운다" 는 층별 자기 문턱의 인공물 → **철회** | 철회 (CROSS_REVIEW A1) | `REPORT_h23b.md` §5 · `auto_research/Archive/CROSS_REVIEW_2026-09-24.md` A1 |
| (참고) 렌즈 수렴 | cos L0 0.31–0.38 → L6 0.69–0.73 → L11 1.00 | 렌즈는 반사실 — "그 층이 안다 / 모른다" 로 읽지 않는다 | 기록 | `REPORT_h23b.md` §6 · `H7` §5 |
| (참고, v1) 층별 정체 probe | z 87.5 → block 1 **68.4** → block 4 72.7 → block 8 86.7 → block 12 85.5 → proj 88.7 | mask token 은 block 1 에서 이미 문맥에서 정체를 끌어온다; `predictor_proj` 는 정보를 안 버린다 → "predictor 내부가 정보를 잃는다" **기각** | 기록 (CLAUDE.md §6) | `MECHANISM_2026-08-20.md` §2.1 |

### 3-3. 인과 ① — attention knockout (간선 · 층 · 반경)

| 개입 | 판 | 수치 | 읽기 | 등급 | 출처 |
|---|---|---|---|---|---|
| 간선 전 층 차단 | v11_full 21,504 쌍, 기준선 `clean_null` 75.86 | **`mask→ctx` 38.3 (−37.5)** · `mask→mask` 74.4 (**−1.4**) · `ctx→ctx` 76.4 (+0.5) | 채점은 미래 → 문맥 경로 하나에 기댄다; 미래끼리는 채점에 거의 기여 안 함 | 기록 (정본 인용) | `IntPhysGenV11_occlusion_timing_ablation/README.md` 전수 11 · `PAPER_STORY_2026-09-06.md` §2 |
| 같은 차단의 출력 변화 (pred_drift) | 같은 판 | `mask→mask` rel_l1 **0.419** / cos 0.931 · `ctx→ctx` 0.129 / 0.993 · `mask→ctx` 1.897 / 0.143 | `mask→mask` 를 끊어도 **예측 자체는 약 40 % 움직인다** — 채점 Δ = 0 이 "아무 일 없음" 이 아니다 | 기록 | 같은 곳 · `CLAIM_…` §2-1 |
| 층 창 프로파일 (폭 7) | 같은 판 | `mask→ctx` 창 중심 0–4 평평 (63.9–66.4) → 5 (55.3) → 6 (47.8) → **8 이후 포화** (37.6–38.7; 전 층 38.3); k · 타이밍 무관 (k=1–4 41.4–42.5) | 층 5–11 에서 문맥을 못 읽으면 앞 층으로 메워지지 않는다 | 기록 | `IntPhysGenV11_occlusion_timing_ablation/README.md` 전수 11 |
| 단일 층 L8 | v11_full (`knockout_w1`) / 144 block 표본 | **−14.4** / **−18.06** (이웃 층의 6 배) | 층 8 이 꺾이는 자리 — 두 수는 표본이 다르다 (§6) | 기록 | `PAPER_STORY_2026-09-06.md` 재현절 · `z_research/IntPhysGenV11/figures/knockout/README.md` |
| 가려진 문맥 튜블릿으로 가는 간선 차단 (`mask→hidden`) | v11_full late, 1,792 쌍 / 조건 | static late **+15.4** (대조군 `hidden_ctrl` −0.6); moving late −2.0 / −2.6 (대조군 −6.8 … +3.6 안 → **해석 불가**) | 정지 물체에서는 가림막 토큰을 보지 않게 하면 오른다 | 기록 | `CLAIM_…` §2-4 · `EXPERIMENTS_2026-09-16.md` §6-2 |
| 조회 반경 제한 (미래 → 문맥을 Chebyshev r 칸 안으로) | RollOut_v2 flat, 거리별 | 40–60 px: r=1 0.03 · r=3 0.38 · **r=5 0.72** · 원래 0.80; 0–20 px 는 r=1 로도 0.96 (p 39 % 바뀌어도) | 성공할 때는 **4–5 칸 떨어진 과거 자리에서 물체를 실제로 가져온다** — 딱 잘린 국소 창이 아니다 | 기록 (정본 인용) | `PREDICTOR_DISTANCE_LIMIT_2026-09-19.md` §5-2 |
| 같은 반경 제한 (v3, 3 stride) | 588 clip | r = d 에서 손실 Cheb 2 s1 −0.23 · Cheb 3 s1 −0.28; 회복은 대략 **r ≥ d + 2**; r=∞ 적중 − 귀무 Cheb 4–5 s1 −0.05 · s2 +0.19 · s4 +0.32 | 2–3 칸 떨어진 진실 칸에 물체가 나오려면 문맥 물체 토큰 **직접 조회가 필요** (미래 ↔ 미래 경로가 대체 못 함); 닿는 거리는 칸 수로 고정되지 않는다 | **시사** (M4 정정판) | `M4_RETRIEVAL_RADIUS_2026-09-25.md` |
| 먼 슬롯 물체다움 knockout 8 명세 (v3, P32) | 2,352 clip | clean 0.97 → 0.11 (슬롯 0 → 10); 어느 명세도 되돌리지 못함 (mm_L0-6 슬롯 10 +0.054, 소실 폭의 6–7 %); mc_L7-11 은 L1 **+64 %** (목표 공간 이탈) | 먼 슬롯 소실은 간선 한 종류로 좁혀지지 않는다 | 기록 (H8 정정판, 부정 결과로만) | `H8_KNOCKOUT_PERSISTENCE_2026-09-24.md` §3 |

⚠️ knockout 38 % 는 "정보가 없다" 가 아니다 — chance 아래 (visible / vanish 13.2) = 문맥 무관 고정 예측의 외형 편향. **꺾이는 자리** 를 읽는다; 개입 뒤 정확도가 clean 수준에 끌려가 조건 간 Δ 비교는 안 된다 (`IntPhysGenV11_occlusion_timing_ablation/README.md` 단서 · `figures/knockout/README.md`).
⚠️ `static_occlusion` 은 clean 54.17 인데 `mask→ctx` 절단 후 **6.25** (체계적 역전) — "가림 조건은 덜 다친다" 금지 (`figures/knockout/README.md`).
⚠️ knockout 은 분포 밖 개입 — zero-ablation 과대평가 경고 (`M4_…` 못 하는 주장). "이 간선이 X 를 나른다" 금지 (`H8_…` §5).

### 3-4. 인과 ② — 조회 강도 · attention 규칙 · 질의 위치 · RoPE

| 개입 | 판 | 수치 (ΔA, 4–6 / 6–9 칸) | 읽기 | 등급 | 출처 |
|---|---|---|---|---|---|
| **I1 조회 보조: 진짜 출처 key logit + β** | K400 인공 팬 1,600 clip (출처 정확) | release β = +1 / +2 / +4: **+0.03 / +0.07 / +0.22** (R50 4.8 → 6.2); Ariel +4 **+0.22 / +0.43** (9–13 칸 +0.36); pv1 +4 +0.23 / +0.17 | reach 가 **조회 세기에 비례** 해 늘어난다 → 진짜 출처 내용을 끌어오면 먼 자리 내용이 맞다 = encoder 충분성 (출처 조회 범위) | **확정** (C3) | `I_SCENE_CAUSAL_BOTTLENECK_2026-09-25.md` §4-1 · `auto_research/Archive/VERIFY_ISCENE_2026-09-25.md` |
| 반대 방향 bias +4 | 같은 판 | −0.05 / −0.01 (1–2 칸 −0.08, 망가뜨림 문턱 초과) | 방향 대조 | 확정 (정정 3: 근거리 손상) | 같은 곳 |
| 필요성 (진짜 출처 −4) | 같은 판 | release −0.02 / −0.00 · Ariel −0.06 / −0.02 | 필요성은 약하다 (중복 경로) | 확정 (약함) | 같은 곳 §4-1 단서 |
| 매개 — Ariel − release 이득 중 log p_src 로 설명되는 몫 | 팬 · SSv2 | 1 차 95.5 % / 99 % → **거리 구간 층화 시 팬 4–6 칸 56 % · 6–9 칸 44 % · SSv2 67 % / 41 %** → "약 절반 (44–67 %)"; EK100 · v3 해석 불가 | 조회 강도는 Ariel 우위의 약 절반을 설명 | **시사** (강등) | 같은 곳 정정 (8) |
| 실영상 bias4 (flow 출처 = **미래를 본 것, 상한**) | SSv2 / EK100 | SSv2 release +0.13 / +0.06 (R50 2.7 → 3.3), Ariel +0.20 / +0.20; EK100 +0.11 / +0.03 | 상한 | 1 차 | 같은 곳 §4-2 |
| 실영상 bias4_cv (문맥 끝 flow 등속 외삽 = 인과 추정) | SSv2 / EK100 | −0.05 ~ +0.01 **무효** (추정 출처 일치 45 %); v3 에서는 유효 (+0.05 / +0.06) | 학습 없는 출처 추정은 실영상에서 못 쓴다 → 출처 추정은 학습된 모듈이어야 (설계 후보 DC1(b), 09 장) | 1 차 | 같은 곳 · 정정 (6) |
| I3 attention 규칙 교체 (full → prefix / isolated) | 팬 · SSv2 · EK100 | release prefix **+0.02 / +0.01**; isolated −0.06 / −0.02; Ariel 에 full −0.21 / −0.12 | **양방향 규칙은 병목 아님 — 팬 · SSv2 · EK100 한정**; v3 에서는 prefix −0.06 ~ −0.10 (2–6 칸), isolated v=12 −0.128 | 시사 · 범위 한정 (C2) | 같은 곳 · 정정 (5) |
| RoPE 축척 (미래 행 ×0.5 / ×0.75) | 팬 | 근거리까지 −0.3 ~ −0.6 (망가뜨림) | 해석 불가 → "RoPE 가 거리 감쇠 원인" 판정 보류 | 판정 보류 | 같은 곳 §4-1 단서 |
| qshift (미래 질의 행의 공간 RoPE 를 출처 − 자기 칸만큼 이동) | 팬 / SSv2 / EK100 / v3 | 팬 release .32 / .07 → **.36 / .38**; EK100 .03 / .02 → .47 / .45; v3 .17 / .01 → .06 / .02 (해로움; 2–3 칸 .77 → .16); 인과 추정판 qshift_cv 무효 | "위치를 맞춰 주면 한계가 사라진다" · "근거리와 같은 수준" **철회** (근거리 비용 −0.12 ~ −0.25, 정확 칸 .08–.09) | **시사** (2 차 강등) | 같은 곳 §4-6 · `auto_research/Archive/VERIFY_ISCENE2_2026-09-26.md` |
| 질의 위치 평행이동 (C16 single, 인덱스 +4 / +8 튜블릿) | v3 2,352 clip | 공통 문턱 물체다움 k 8–10: 0.51 / 0.68 / 0.76 (원 / +4 / +8); k 14–15 0.20 / 0.24 / 0.21 (차 없음); L1 은 이동 쪽이 모든 띠에서 나쁨 (+0.006 ~ +0.018) | 대비는 이동량에 단조로 오르지만 사라지는 k 는 당겨지지 않는다 → 절벽은 "절대 인덱스 상한" 이 아니다 | **시사** (분포 밖 — 인코더 · predictor 위치 어긋남) | `auto_research/Archive/H4_QUERY_AND_POSITION_2026-09-24.md` |
| RoPE 주파수 "subtle bug" | 릴리즈 코드 | `rotate_queries_or_keys` 가 회전쌍 두 성분에 서로 다른 주파수 (`repeat(1,1,1,2)`, 저장소 주석 "subtle bug", 릴리즈가 그대로 학습) → 위치 인덱스를 4 / 8 튜블릿 밀면 p **19 % / 26 %** 변화 (첫 배치 16 clip); 토이 58 % 변화, 고치면 1.8e-15 | "RoPE 라 절대 위치 효과는 구조상 0" 은 **틀렸다** — 릴리즈 RoPE 는 순수 상대가 아니다 | 확정 (CROSS_REVIEW 확립 5) | `auto_research/Archive/PLAN_2026-09-24.md` 정정 · `H4_…` · `CROSS_REVIEW_2026-09-24.md` 결론 |
| 단일 질의 vs 한 번에 (single vs joint) | v3 | 공통 문턱 single − joint 진실 템플릿 +0.11 ~ +0.20 (k ≥ 6) 이나 **L1 (p, h) 은 모든 C · k 에서 joint 가 낮다** | 단일 질의라고 예측이 나아지지 않는다 | 기록 (H4 정정판) | `H4_…` 결론 |

⚠️ I1 은 **진짜 출처를 알려 주는** 개입이다 — "옮겨 준 것" 반론이 남는다. 그래서 이 줄이 세우는 것은 "encoder 토큰 안에 먼 자리를 맞힐 내용이 있다" 까지이고 "predictor 가 그 출처를 찾아야 한다" 의 방법은 아니다 (`I_SCENE` §4-1 단서).
⚠️ "거리가 주 변수" 는 Chebyshev ≤ 1 판독에서만 — 정확 칸 판독에서 release 슬롯 계수 −0.074 > 거리 −0.049; 실영상 연속 판독에서는 시간 효과가 같거나 크다 (`I_SCENE` 정정 (1) · §4-7). 거리 vs 시간 문제는 04 장.

### 3-5. 인과 ③ — 입력 토큰 교체 · 조향

| 개입 | 판 | 수치 | 읽기 | 등급 | 출처 |
|---|---|---|---|---|---|
| 경계 1–2 튜블릿 토큰 교체 (late ↔ early 쌍둥이) | v11, late 쌍 2,821 | **있음** (t2–4 연속 contrast) 움직임 회복 **1.10** [1.08, 1.12] · 파괴 1.19; **자리** 회복 **0.15** [0.09, 0.21] (등속 0.03); 역사 대비 (L+Eb) − E 자리 −0.198 (등속 −0.274); 채점 회복 0.41 | 있음은 경계 토큰을 따라가고 자리는 따라가지 않는다; 정지 물체의 있음은 역사 토큰에도 기댄다 (파괴 0.53–0.71) | 확정 (M2 정정판) | `auto_research/Archive/M2_BOUNDARY_PATCH_2026-09-25.md` §7 |
| 역사 토큰 제거 · 교체 (자연 조건) | v3 784 clip | hist_none 진행 비율 **0.547** vs base 0.542 (차 −0.004 [−0.014, +0.005]); 물체다움 t4 / t6 / t8 차 +0.16 / +0.28 / +0.20; hist_swap 적중 − 귀무 Δ ±0.01, 물체다움 t6 −0.136 | 역사 토큰은 **자리가 아니라 있음** 에 기여 (v3 가시 자유 운동 · 릴리즈 한정) | 확정 (범위 한정) | `auto_research/Archive/M6_HISTORY_SHIFT_2026-09-25.md` · `M13_…` M1 |
| 따로 인코딩한 역사 시간 이동 | v3 | 먼 슬롯 argmax 이동 −0.33 ~ +0.34 칸 (진실 진행 약 ±5 %); 인코딩 방식만으로 t4 +0.56 칸 | 이음 · 화면 밖 교란과 공선 | 시사 (분포 밖) | `M6_…` |
| 속도 방향 조향 (경계 18 토큰에 s·u) | v3 784 clip | t0 반대칭 이득 **0.392** [0.308, 0.476] → t1 0.089 → t4–7 0.014–0.025; 멈춘 자리 b 불변 (base 4.60 vs vx+3 4.74) | 첫 튜블릿 자리만 옮겨진다 — 이를 "속도 사용" 으로 읽는 것은 판정 보류 (조향 방향이 법칙 정체와 얽힘, 법칙 안 held-out R² 0.518) | 확정 (관측) · 판정 보류 (귀속) | `M5_VELOCITY_STEERING_2026-09-25.md` §4 |
| 자연 속도 차이의 누적 | 같은 clip base | 자연 이득 t1–t5 **0.72–1.02** (진실 0.76–0.98) → t6–7 0.40 / 0.38; 멈춘 자리 b **4.60** [4.01, 5.14] vs 진실 6.07 | 자연 속도 차이는 one-shot 에서 약 5 튜블릿까지 자리에 쌓이고 멈춘다 | 확정 (행동) | 같은 곳 D |

### 3-6. 인과 ④ — 되먹임 (닫힘 검정)

| 팔 | 판 | j3–7 전진 기울기 (진실 0.440) | 읽기 | 등급 | 출처 |
|---|---|---|---|---|---|
| ar_z (매 걸음 **참 인과 관측** z 를 넣음, 탐침) | v3 784 clip, C16, 궤적 56 | **+0.361** · 새 칸 착지 .78–.96 | frozen predictor 는 관측 한 튜블릿을 받으면 물체를 한 칸 옮기는 계산을 가진다 (한 걸음 전이) | 확정 | `auto_research/Archive/P3_CLOSURE_TEST_2026-09-25.md` · `H5_AUTOREGRESSIVE_FEEDBACK_2026-09-24.md` |
| ar_hcA (인과 표적 + 같은 어댑터 A) | 같은 판 | +0.361 (ar_z 와 짝 차 −0.000 [−0.006, +0.005]) | 어댑터 · target encoder 출처는 원인 아님 | 확정 | `P3_…` |
| ar_h (양방향 표적 참 토큰) | 같은 판 | +0.212 [+0.179, +0.248]; 형식 효과 −0.148 · 내용 (+누적) 효과 −0.224 | 양방향 표적 형식은 참 토큰이어도 j ≥ 4 부터 늦다 (진실 칸 템플릿 한정) | 확정 (범위 한정) | 같은 곳 · `README_UNTIL_2026-09-25.md` 정정 2-3 |
| **ar_pA (자기 예측 되먹임)** | 같은 판 | **−0.012** [−0.042, +0.018]; j7 적중 − 귀무 −0.205 | 자기 예측을 이어붙이면 j3 부터 멈춘다 — one-shot 의 짧은 지평 (약 3 튜블릿) 을 늘리지 않는다 | 확정 (행동) | 같은 곳 |
| tf_first / tf_last (역사 · 마지막 중 한쪽만 참) | v3 784 clip | 격차 회복 (외형) 0.79 / 0.78; 2×2 상호작용 −0.294 (진실 칸) | 한쪽만 참이어도 약 78 % 회복; 누적 vs 마지막 토큰 질은 판정 보류 | 시사 | `P3B_LAST_TOKEN_DRIVES_TRANSITION_2026-09-25.md` |
| 장면 closure (K400 팬, 400 clip) | release / Ariel / N1 / pv1 | Δ(ar_pA − os) 슬롯 3–7 release −0.03 ~ +0.01 · **Ariel −0.12 ~ −0.16 · N1 −0.15 ~ −0.22**; Δ(ar_z − os) release +0.10 ~ +0.22 | 진짜 관측을 붙이면 계속 맞히고, 자기 예측을 붙이면 one-shot 보다 낫지 않다 | 확정 (2 차 검증) | `I_SCENE_…` §4-5 · `VERIFY_ISCENE2_2026-09-26.md` §2 |

⚠️ 되먹임은 학습된 사용법이 아니다 (stress test) — one-shot 결론으로 옮기지 않는다 (`P3B_…` 단서). ar_z 는 진짜 다음 프레임을 넣은 **탐침** 이라 "encoder 충분" 의 근거로 쓰지 않는다 (`README_UNTIL_2026-09-25.md` 정정 1-7).

### 3-7. 인과 ⑤ — frozen 채점 개입은 교환이지 상승이 아니다

| 개입 | 판 | 수치 | 읽기 | 등급 | 출처 |
|---|---|---|---|---|---|
| single (미래 튜블릿마다 256 mask 토큰만 질의) · mmx_all (전 층 미래 → 다른 튜블릿 미래 차단) | v11_full block 1/4, 5,376 쌍 | 정지 76.3 → 70.9 (**−5.4**) · 등속 +3.7 · 등가속 +4.4; 전체 +0.9; 가능 clip L1 +3.1 % (분포 밖) | 움직이는 조건은 오르고 정지는 내린다 — **교환** | 기록 (H9 정정판) | `H9_FROZEN_INTERVENTION_SCORING_2026-09-24.md` §3 |
| 같은 개입의 late 물체 → 빈 | 같은 판 | 22.6 → 52.4 (+29.8) 이나 등속 3.6 → 46.4 vs **복사 44.6**, 등가속 0.0 → 21.4 vs **복사 16.1** | 움직이는 물체에서 복사 기준선을 넘는다는 증거 없음 | 기록 | 같은 곳 |
| 같은 개입의 IntPhys 1 | skip2_w32 | clean **88.89** → mmx_all 87.22 / single 86.67; 사라짐 방향 56.0 불변 | 벤치마크를 올리지 못한다 | 기록 | 같은 곳 |
| 전역 정렬 사상 W · 균등 증폭 | v11 / v10 | late +3.2 (post-FT 의 1/5) · 증폭 천장 51–65 | §2-4 | 기록 | `ctxenc_subspace_v2` · `GEOMETRY_2026-08-29.md` |

→ 08-31 의 "frozen 위에서 닫힌다" 는 **09-06 에 폐기** 됐다 (`PAPER_STORY_2026-09-06.md`). frozen 개입은 진단 도구로만 쓴다.

### 3-8. 기제 요약 — 무엇이 서고 무엇이 철회됐나

| 문장 | 근거 | 등급 |
|---|---|---|
| 채점은 미래 → 문맥 조회 (`mask→ctx`) 에 기댄다; 층 5–11 에서 끊으면 메워지지 않는다 | §3-3 | 기록 (개입) |
| 미래 토큰은 먼 출처를 약하게 조회한다 — 조회 강도가 거리로 준다 (칸당 log −0.42), 세기를 올리면 reach 가 비례해 는다 | §3-1 · §3-4 I1 | 상관 1 차 + 개입 확정 (충분성) · 매개 시사 |
| 가까운 곳에서는 물체를 선택적으로 보고, 2–3 칸에서는 직접 조회가 필요하다 | §3-3 M4 · §3-1 ring | 시사 |
| 양방향 attention 규칙은 병목이 아니다 | §3-4 I3 | 시사 · 팬 · SSv2 · EK100 한정 (v3 반례) |
| 층을 바꿔 읽어도 late 는 돌아오지 않는다 | §3-2 | 기록 |
| 경계 토큰은 있음을, 역사 토큰은 (v3) 있음을 · (v11) 자리를 바꾼다 — 데이터셋마다 반대 축 | §3-5 | 확정 (범위 한정) — 서로 "역사 불사용" 근거로 쓰지 않음 |
| 자기 예측을 되먹이면 j3 부터 멈춘다; 참 관측이면 전진한다 | §3-6 | 확정 (행동) |
| frozen 개입으로는 교환 · 작은 상승뿐 | §3-7 · §2-4 | 기록 |
| ~~"운반되는 상태가 아니라 공간 조회 · ≈ 4–6 칸 고정 반경"~~ · ~~"공간 반경 2.6 칸 + 2 걸음"~~ · ~~"마지막 토큰 하나가 정한다"~~ · ~~"역사 토큰이 뒤로 끈다"~~ · ~~"뒤층이 지운다"~~ | §5 | 철회 |

출처 (§3): `analysis/attention/README.md` · `analysis/attention/predictor_attn/README.md` · `z_research/IntPhysGenV11/figures/{attention,knockout}/README.md` · `z_research/IntPhysGenV11_occlusion_timing_ablation/README.md` · `z_research/IntPhysGenV11/Archive/PAPER_STORY_2026-09-06.md` · `z_research/RollOutV2/Archive/PREDICTOR_DISTANCE_LIMIT_2026-09-19.md` · `z_research/context_encoder_analysis/Archive/{CLAIM_NO_STATE_EVOLUTION_2026-09-15,EXPERIMENTS_2026-09-16}.md` · `z_world_model_analysis/MECHANISM_2026-08-20.md` · `auto_research/Archive/{PLAN_2026-09-24,H4_QUERY_AND_POSITION_2026-09-24,H5_AUTOREGRESSIVE_FEEDBACK_2026-09-24,H7_V11_LAYERWISE_2026-09-24,H8_KNOCKOUT_PERSISTENCE_2026-09-24,H9_FROZEN_INTERVENTION_SCORING_2026-09-24,H23_PREDICTOR_LAYERS_TRAJECTORY_2026-09-24,CROSS_REVIEW_2026-09-24,SC1_REACH_NOT_HORIZON_2026-09-25,M2_BOUNDARY_PATCH_2026-09-25,M4_RETRIEVAL_RADIUS_2026-09-25,M5_VELOCITY_STEERING_2026-09-25,M6_HISTORY_SHIFT_2026-09-25,M13_MEMORY_VELOCITY_SUFFICIENCY_2026-09-25,P3_CLOSURE_TEST_2026-09-25,P3B_LAST_TOKEN_DRIVES_TRANSITION_2026-09-25,I_SCENE_CAUSAL_BOTTLENECK_2026-09-25,VERIFY_ISCENE_2026-09-25,VERIFY_ISCENE2_2026-09-26,VERIFY_0925_R2_2026-09-25,README_UNTIL_2026-09-25}.md` · `auto_research/exp_results/h23b/REPORT_h23b.md` · `auto_research/README.md`

---

## 4. "충분성" 주장의 이력 — 08-20 → 09-25 철회 → 10-02 → 10-08

### 4-1. 판마다 무엇을 말했나

| 날짜 | 문서 | 그때의 문장 | 근거 | 무엇이 그것을 세웠나 / 무너뜨렸나 | 지금 상태 |
|---|---|---|---|---|---|
| 08-20 | `z_world_model_analysis/MECHANISM_2026-08-20.md` | "identity 52.9 / 65.3 은 모델의 한계가 아니라 채점 지표의 한계" — 원문 "V-JEPA 2 는 알고 있는데" (내부 귀속 표현 — 현재 금지) | 층별 probe z 87.5 → p 88.7 (h 91.8), probe 채점 100 / 98.3, 토큰 배치 corr 0.002 | "p 에 정보가 없다" · "물체를 엉뚱한 자리에" **철회** 를 같은 문서가 기록 | 수치 유효 · 귀속 문장 금지 |
| 08-30 / 08-31 | `RESULTS_2026-08-30.md` · `IntPhysGenV11/Archive/PAPER_STORY_2026-08-31.md` | "정보는 남고 배치가 무너진다, frozen 위에서 닫힌다" | p self 98.46 vs 채점 4.46, 이식 p 17.3, env 100 | "frozen 위에서 닫힌다" 는 09-06 에 **폐기** (knockout · AC `load_predictor: false`) | 앞 절반 (정보 · 배치) 유효 |
| 09-15 / 09-16 | `context_encoder_analysis/Archive/CLAIM_NO_STATE_EVOLUTION_2026-09-15.md` v0.1 · v0.2 | "encoder 를 바꾸지 않고 v11 채점을 세우기에 충분한 정보가 encoder 에 있다" — "이어갈 재료는 encoder 에 있었다" 는 **추론으로 하향** | 경계 토큰 속도 R² 0.988, post-FT 가 encoder 그대로 late 57 → 87 회복, 정렬 사상 +3 | post-FT 가 배운 것은 "v11 궤적 prior" (EXPERIMENTS §6-1) → 회복이 encoder 충분성의 증거인지 미결 | 기록 (state 프레임 — 현재 금지) |
| 09-19 | `RollOutV2/Archive/PREDICTOR_DISTANCE_LIMIT_2026-09-19.md` · `STATE_EVOLUTION_CAPABILITIES_2026-09-19.md` | "원인은 정보 부족이 아니다" — H3 (z 가 속도를 약하게 담는다) **기각**; 능력 1 읽기 ✅ | z 속도 R² 0.996 / 0.989; knockout 으로 4–5 칸 출처에서 가져올 능력 확인 | — | **현행** (정본 (iv) 첫 줄) |
| 09-25 오전 | `auto_research/Archive/PLAN_ENCODER_PREDICTOR_2026-09-25.md` | 사용자 프레임: "(1) encoder is enough → (2) 실패를 predictor 병목으로 인과적으로 → (3) 새 predictor" — 사슬 L1–L9 사전 등록 | — | 결과 뒤 L1 "읽힌다" 만 확정, L2 (표적 공간) 불지지, L3 은 "predictor 한 걸음 전이" 로 이동 | 계획서 (기록) |
| 09-25 | `M13_MEMORY_VELOCITY_SUFFICIENCY_2026-09-25.md` | 충분성 대조: z_cv (진실 마지막 위치 + z 에서 읽은 속도 × 경과) 가 먼 슬롯에서 p 를 이김 — t10 적중 − 귀무 **+.24 vs p −.14** → "encoder 는 충분하다" | 위와 같음 | **철회** — z_cv 가 **진실 위치를 썼다** (위치는 encoder 로 안 읽음), 손으로 짠 등속 규칙, 지도 학습 속도; 라벨 평균 속도만으로도 .56–.61 (z 고유 몫 +.14–.19); arc 에서 역전 | 철회 (M13 정정 2) |
| 09-25 | `auto_research/Archive/README_UNTIL_2026-09-25.md` 정정 1 · `TRAINED_PREDICTORS_SIGNATURES_2026-09-25.md` 정정 2-1 · `auto_research/README.md` §2 | 제목 "encoder 는 충분하고, predictor 가 병목이다" **철회** | "충분성 대조가 진실 위치를 써서 이를 재지 못했다. 분업 귀속은 닫히지 않았다" | — | **철회 유지** (README §2 철회표 1 행) |
| 09-25 | `VERIFY_0925_R2_2026-09-25.md` X5 · `TRAINED_…` §4 | 좁힘: "릴리즈의 3–4 튜블릿 한계는 **encoder 입력만으로 정해지지 않는다**" (존재 증명) | 같은 frozen encoder 위 학습된 pv1 · Ariel 은 가시 물체를 약 8 튜블릿 진실 기울기로 (t3–7 0.52 / 0.48 vs 릴리즈 0.07; t10 적중 − 귀무 +0.24 / +0.26 vs −0.14) | 요인 (초기값 · 데이터 · 질의 · attention · fps · epoch) 미분리 → "encoder 충분 · predictor 병목" 은 그래도 철회 유지 | 확정 (존재 증명, 가시 · 등속 근처 · stride 1–2 · arc 제외) |
| 09-25 저녁 | `I_SCENE_CAUSAL_BOTTLENECK_2026-09-25.md` C3 | "**encoder 도 병목 아님** — 진짜 출처 내용을 끌어오면 맞는 내용" | I1 bias +4 로 reach 가 세기별 증가 (release +0.22, Ariel +0.43) | 범위는 **출처 조회 개입** — 진짜 출처를 알려 준 개입이라 "옮겨 준 것" 반론 | 확정 (C3), 범위 한정 |
| 09-29 | `auto_research/paper/RETHINK_PREDICTOR_ROLE_2026-09-29.md` §1-b · `ENCODER_MOTION_READOUT_2026-09-29.md` | "encoder 충분성 표" (정체 98–100 · 운동 R² 0.996 · 출처 = 천장 · EK100 ego-motion .69 / .63) → EK100 "encoder 가 카메라 운동을 못 담아서" **기각**, "encoder 충분" (RETHINK §1-b 확정) | 위 수치들 | VERIFY_R3: "천장까지 읽는다" 는 정정 (자기 일관성보다 높게); 토큰 평균은 자리 정보를 버림 | RETHINK 는 대체됨 (10-08), 표는 유효 |
| 09-30 | `INTPHYS2_ENCODER_LINEAR_PROBE_2026-09-30.md` | "encoder 는 정보를 놓치지 않는다 쪽의 증거이고, 그 위에 예측이 얹히지 않는다는 것이 논지의 빈자리" | IntPhys 2 probe 쌍 73.7 vs VoE 54.3 | 지도 자 (impossible 라벨을 보고 배움) — VoE 와 같은 능력 아님 | 기록 |
| 10-02 | `FUTURE_FROM_CONTEXT_2026-10-02.md` | "**predictor 가 미래를 만드는 데 필요한 정보는 입력에 선형으로 다 있고** (안 본 속도 · 가림 문맥에서도), predictor 는 그것을 미래 자리로 옮기지 않는다" | z 자 B v11 late 5.4–10.1 px ≈ h 4.6–8.2; p 자 B 16.9–29.9; v2 ledge z-B 3.3 px | 10-02 정정: v2 (가림 없음) 에서는 p 자 B 로 대부분 나온다 → p 가 z 에 못 미치는 곳은 **v11 late 가림만** | 정본 (1 차) |
| 10-08 | `auto_research/paper/PAPER_STORY_2026-10-08.md` §2-3 (iv) · §4-5 | P3 (iv) "encoder 에는 정보가 있다 — 실패는 predictor 의 미래 예측이다"; 쓸 문장 "미래를 만드는 데 필요한 정보는 입력에 선형으로 있고, predictor 는 그것을 미래로 옮기지 않는다" | z 속도 R² (확정) · FFC (1 차) · p probe vs 채점 · ENCODER_MOTION (1 차) | 리뷰 공격 4-5 "encoder 충분성은 지도 자의 산물" 에 "z 자 ≈ h 자, 파라미터 없는 교차검증, 안 본 속도 · 가림 문맥에서도" 로 답함 | **현행** |

### 4-2. 09-25 철회의 정확한 이유 (지우지 않는다)

| 항목 | 내용 | 출처 |
|---|---|---|
| 무엇을 재려 했나 | "encoder 에서 읽은 상태만으로 먼 슬롯을 맞힌다" → 먼 슬롯에서 encoder 판독이 p 를 이기면 병목은 predictor | `PLAN_ENCODER_PREDICTOR_2026-09-25.md` L2 · `M13_…` |
| 무엇이 잘못됐나 (1) | z_cv 의 **위치는 진실 픽셀** 이었다 — 위치는 encoder 에서 읽지 않았다 | `M13_…` 정정 (2) · `README_UNTIL_2026-09-25.md` 정정 1-1 |
| (2) | 속도는 z 에서 **지도 학습** 으로 읽었고, 미래는 **손으로 짠 등속 규칙** 이 만들었다 — "읽힌다 ≠ 쓰인다" (Hewitt & Liang) | `M13_…` 못 하는 주장 |
| (3) | 라벨 평균 속도 (encoder 없음) 만으로도 t10 적중 .56–.61 → z 고유 몫은 +.14–.19 뿐 | `M13_…` 핵심 수치 · `VERIFY_0925_2026-09-25.md` |
| (4) | arc 에서 역전 (t8 z_cv .00 vs p .55) — 등속 규칙이 맞는 법칙에서만 이긴다 | `M13_…` |
| (5) | "표적 공간 (apph 0.75–0.91)" 은 진짜 미래를 본 표적의 **천장** 이었다; "관측이면 된다 (ar_z)" 는 진짜 다음 프레임을 넣은 탐침 | `README_UNTIL_2026-09-25.md` 정정 1-6 · 1-7 |
| 남긴 것 | "읽기" 고리 (z 에서 가시 방향 · 속력 선형) 만 확정 | `README_UNTIL_2026-09-25.md` 결합 이야기 1 행 |

### 4-3. 10-08 정본은 무엇으로 다시 세웠나 — 무엇이 달라졌나

| 09-25 결함 | `FUTURE_FROM_CONTEXT` 에서는 | 남은 단서 |
|---|---|---|
| 위치에 진실 좌표를 썼다 | 자의 **입력은 z 하나** — 진실 좌표는 라벨 (지도 학습 표적) 과 참조선 (CV · COPY) 에만 쓰인다 | 라벨이 곡선을 가르친다 (지도 자) |
| 미래를 손으로 짠 규칙이 만들었다 | 미래 8 슬롯 위치를 **자가 직접 회귀** — 낙하 · 정지 · 포물선 · 가속을 규칙 없이 | 그 곡선은 데이터의 템플릿일 수 있다 (낙하 시각 고정, wall 자리 고정) → "encoder 가 미래를 표현한다" 금지 |
| 같은 궤적 · 분포 안 | v2 는 **안 본 속도 레벨 2 개** holdout; v11 은 block 절반 held-out (condition × k 층화) | v11 은 같은 장면 · 같은 궤적 계열 — 장면 hold-out 판 없음 |
| encoder 판독 vs p 의 비교가 공정하지 않았다 (규칙 vs 출력) | **같은 자 구조 (A · B) 를 z · p · h 에 똑같이** 걸었다 → z ≈ h, p 만 v11 late 에서 멀다 | 자 A 는 p 에 불리 (p 정보가 튜블릿 자리에 국소적이지 않음) — 자 B 로 정정 |
| 천장 대조 없음 | h (미래를 본 encoder) 자 = 상한, 불가능 ledge 에서 h 자 낙하 13–16 px (가능 57–59) 로 자 검증 | 파라미터 없는 교차검증은 이 문서에 없다 |

→ **달라진 것**: 주장의 크기가 "encoder 는 충분하다 (분업 귀속)" 에서 "필요한 정보는 입력에 선형으로 있다 (정보 문장)" 로 줄었다. 정본은 분업 귀속을 다시 쓰지 않고 정보 문장을 쓴다 (§0 ✅ 줄).
→ **같이 세우는 다른 판**: 속도 R² 0.996 (확정) · I_SCENE I1 출처 조회 충분성 (확정, 범위 한정) · ENCODER_MOTION EK100 (R² 확정, 해석 1 차) · IntPhys 2 linear probe (지도 자, 기록) · ENCODER_TEMPORAL_DYNAMICS (뒤집힘 비율 = 라벨 없는 교차검증).

### 4-4. 아직 못 닫힌 것

| # | 미결 | 왜 중요한가 | 출처 |
|---|---|---|---|
| 1 | **FFC 는 적대 검증 전 (1 차)** — 정본 표의 "확정 (합성)" 과 등급이 다르다 | P3 (iv) 의 핵심 수치 (5–10 vs 12–30 px) 가 검증 전 | 정본 §2-3 (iv) "(`FUTURE_FROM_CONTEXT`, 1 차)" |
| 2 | 정본 §4-5 의 "**파라미터 없는 교차검증**" 이 FFC 원문에 없다 — 있는 것은 E1 쌍 검사 (속도) · 뒤집힘 비율 (실영상 방향) · h 자의 불가능 ledge (자 검증) | v11 late 미래 위치 (가려진 물체) 에 대한 파라미터 없는 판은 아직 없다 | `FUTURE_FROM_CONTEXT` 전문 · `ctxenc_boundary_kinematics` · `ENCODER_TEMPORAL_DYNAMICS` |
| 3 | 미래가 현재 상태의 결정적 함수인 데이터 — 낙하 시각이 변하는 세트 (선반 끝까지 거리 × 속도 독립) 가 필요 | "언제 떨어질지를 안다" 를 가르려면 | `FUTURE_FROM_CONTEXT` §3 |
| 4 | 분업 귀속 (encoder 충분 · predictor 병목) 은 요인 분리로 닫힌 적 없다 — 같은 encoder 위 학습 predictor 가 더 간다는 존재 증명뿐, 그 predictor 들도 전이 (가속) · 사건은 확인 안 됨 | "단순히 predictor 가 실패한다" 의 "단순히" 를 받치는 대조는 08 장 (재학습해도 같다) 에 걸려 있다 | `VERIFY_0925_R2_2026-09-25.md` X5 · `TRAINED_…` 결론 · 정본 §2-3 (v) |
| 5 | 가속 정보는 ◐ — 국소 가속 a_T ≈ 0, r8 은 긴 문맥 · 역사와 미분리, v2 가속은 시나리오 지름길, FFC ramp 는 경사 장면 단서 미분리 | 3a (가속) 실패를 "정보는 있는데" 로 쓰려면 (v0 × a) 교차 세트가 먼저 | `H1_…` · `EXPERIMENTS_2026-09-16.md` §2-1 · 06 장 |
| 6 | "encoder 가 여러 프레임을 경계 토큰에 적분한다" 는 판정 보류 (bndM 은 있음 붕괴로 위치를 못 가름) | encoder 쪽 기제 서술 금지 | `M6_…` 정정 (5) · `README_UNTIL_2026-09-25.md` 정정 2-6 |
| 7 | shape 은 이식에서 encoder 도 흔들린다 (z 64.8) — self 는 99.9–100 | "encoder 충분" 을 shape 으로는 안 쓴다 | `H7_…` 정정 (4) · 정본 §2-3 (iv) |
| 8 | 실영상 국소 물체 운동 (EK100 국소 잔차 .04 / .09) 은 거의 안 읽힌다 — encoder 한계인지 flow 표적 한계인지 미결 | 실영상 "encoder 에 정보" 는 전역 · 방향까지 | `ENCODER_MOTION_READOUT_2026-09-29.md` 단서 |
| 9 | p 가 z 에 못 미치는 조건은 **v11 late 가림** 하나로 좁혀졌다 (가림 없는 v2 는 p 자 B 로 대부분 나옴) — 안 보이는 것 (05 장) 에는 강하고, 보이는 것 · 없는 것 (04 · 06 장) 에서는 FFC 는 p 자가 z 자보다 분명히 먼 결과를 내지 않았다 (v2 p-B 와 z-B 차 2–4 px) | 정본 문장의 "옮기지 않는다" 는 조건 표와 같이 써야 한다 | `FUTURE_FROM_CONTEXT` §1 정정 |

출처 (§4): 위 표의 각 문서 + `auto_research/README.md` §2 · `auto_research/paper/PAPER_STORY_2026-10-08.md` §2-3 · §4

---

## 5. 철회 · 기각 (이 장 주제) → 통합 표는 10 장

| 옛 문장 | 왜 무너졌나 | 지금 쓸 문장 / 상태 | 날짜 · 출처 |
|---|---|---|---|
| "p 에 정보가 없다" (원 문서 표현 — 현재 금지) | v11 등가속 + 가림 p probe shape 98.46 / color 99.49 / env 100; v1 proj 88.7 | "정보는 있는데 맞는 미래로 놓지 않는다" | 08-20 `MECHANISM` §6 · CLAUDE.md §6 |
| "물체를 엉뚱한 자리에 그린다" | Mh 가 물체 위치에 안 몰림 (상위 1 % = 1.9 %) | 토큰 배치가 무관 (corr 0.002) | 08-20 `MECHANISM` §6 |
| "predictor 내부가 정보를 잃는다" | 층별 probe 68.4 → 85.5 단조 증가, proj 후 88.7 | 기각 (다시 시도하지 말 것) | 08-20 `MECHANISM` §6 · CLAUDE.md §6 |
| "예측 축이 직교 (cos 0.085) = 원인" | 개념 벡터 cos +0.823 — 방향은 정렬, 토큰 배치가 무관 (측정 단위가 달랐다) | 정교화 | 08-16 → 08-20 `MECHANISM` §2.3 |
| "표현이 회전했다" / "가림이 표현을 회전시킨다" | 측정했고 아니었다 — h 상위 공간 안 저차원 부분집합 (가림 무관) | "저차원 부분집합" | `RESULTS_2026-08-30.md` §12 · `PAPER_STORY_2026-09-22.md` |
| "z 와 h 가 같다" | 천장 (포화) 비교 | 쓰지 않는다 | `RESULTS_2026-08-30.md` §12 · 정본 §7 |
| "frozen 위에서 닫힌다" | knockout `mask→mask` −1.4 · 정렬 사상 +3 (재학습의 1/5) | 진단 도구로만 | 09-06 폐기 · `PAPER_STORY_2026-09-06.md` |
| "전역 Procrustes 사실상 기각" (토큰 L1 ≤ 9 %) | 채점 재계산 late +3 ~ +6 | "기각이 아니라 부분 효과" | 09-15 → 09-16 `ctxenc_subspace_v2` |
| "속도가 경계 토큰 하나에 국소적" | n=30 표본 artifact (배경 0.39 → n=300 0.871) | "배경에도 속도를 맞히는 다른 코드가 있다" (겹침 0.043) | 09-16 `ctxenc_boundary_kinematics/n30` → `RESULTS.md` |
| "경계에 정체성 없음, 원리적으로 못 푼다" | ridge 이식 실패이지 없음이 아님 (static B7 − BG +35 ~ 42; late → early 36 / 41 / 73) | E2 ◐ — "다른 코드로 남는다" | 09-16 `STATE_EVOLUTION_HYPOTHESES` blockquote |
| "채점기가 보는 것 = 토큰 평균" | 채점기는 토큰별 \|p − h\| 평균 — 등치 불가 | 삭제 | 09-16 같은 곳 |
| "가속은 입력 (z) 에 있다 (held-out 0.77) · 전이 실패는 predictor 미래 계산 탓" | clip 단위 split 이 같은 궤적을 train · test 에 넣음 (궤적 복사본 누수) → 궤적 split C16 0.51, 국소 a_T ≈ 0 | "1 읽기 (가속) ◐ — 긴 문맥에서만" | 09-24 `H1_ACCEL_IN_INPUT` 정정 |
| "encoder 는 충분하고 predictor 가 병목이다" | 충분성 대조 (z_cv) 가 진실 위치를 썼다; 요인 미분리 | 정보 문장으로 좁힘 (§4) | 09-25 `README_UNTIL` 정정 1 · `TRAINED` 정정 2-1 · `auto_research/README.md` §2 |
| "속도가 물체 자리로 적분되지 않는다 / 정보는 있고 진화가 없다" | 절대 위치 R² 무효 대조 (진실 칸도 R² .23 / .15) | 철회 (state 프레임 — 현재 금지) | 09-25 `M13` 정정 (1) |
| "마지막 토큰 하나가 정한다 · 역사 무관" | tf_first ≈ tf_last (+0.003) — 한쪽만 참이어도 약 78 % 회복; 2×2 깨끗하지 않음 | 누적 vs 토큰 질 판정 보류 | 09-25 `P3B` 정정 2 · `VERIFY_0925_R2` X3 |
| "역사 토큰이 예측을 과거 쪽으로 끌어 뒤처짐을 만든다" | 자연 역사 제거 진행 0.547 vs 0.542 (차 없음); M2 −0.231 은 이어붙인 역사 | "역사 토큰은 있음에 기여" | 09-25 `M6` 정정 2 · X2 |
| "속도 부호를 첫 튜블릿에서만 쓰고 누적하지 않는다 / 적분하지 못한다" | 자연 속도 이득은 t5 까지 자리에 쌓인다 (0.72–1.02); 조향 방향이 법칙 정체와 얽힘 | "이 조향 방향의 성질" | 09-25 `M5` 정정 2 · X4 |
| "최종층이 먼 미래 물체를 지운다" (L6 0.88 → L11 0.30) | 층별 자기 문턱 인공물 — 공통 문턱에서 층 차이 없음 | 철회 | 09-24 `H23` 정정 · `CROSS_REVIEW` A1 |
| "주된 경로는 초기층 미래 ↔ 미래 섞임" · "뒤층 문맥 재조회가 외형을 만든다" | 명세별 문턱 인공물; mc_L7-11 은 목표 공간 이탈 (L1 +64 %) | 부정 결과로만 | 09-24 `H8` 정정 1 · 6 |
| "미래 mask 토큰끼리 상호작용이 먼 미래 물체를 지운다" · "단일 질의 ≫ joint (k10 0.43 vs 0.10)" | 모드별 문턱 인공물 (공통 문턱 0.43 vs 0.32) · 인과 문장 | 철회 | 09-24 `H4` 정정 |
| "RoPE 라 절대 위치 효과는 구조상 0" | 회전쌍 주파수 불일치 ("subtle bug") — 인덱스 이동 4 / 8 튜블릿에 p 19 / 26 % 변화 | 정정 (릴리즈 RoPE 는 순수 상대 아님) | 09-24 `PLAN_2026-09-24` 정정 · `H4` |
| "미래의 물체는 운반된 상태가 아니라 문맥 과거 자리에서 공간적으로 가져온 것 · ≈ 4–6 칸 고정 반경 · 둘 방법이 없다" | 고정 반경 없음 (stride · 슬롯 의존), 상태 부재 안 잼, 불가능 주장 = 금지 표현 유형 | "2–3 칸에서 직접 조회가 필요" (시사) | 09-25 `M4` 정정 · `VERIFY_0925` |
| "공간 조회 반경 약 2.6 칸 + 약 2 걸음" | 멈춤 예측기 (freeze K=3) R² 0.991 이 더 잘 맞음 | "몇 걸음 외삽 (stride 1 약 3 튜블릿)" | 09-25 `P2_STRIDE_SWEEP` · `VERIFY_0925` (c) |
| "멈춤의 원인은 경로가 아니라 p 가 담은 내용" · "한 걸음 전이에 필요한 것은 관측 토큰에 있고 predictor 가 만든 토큰에는 없다" | 양방향 참 토큰도 j ≥ 4 멈춤 (형식 교락) · 금지 표현 계열 | 형식 −0.148 / 내용 (+누적) −0.224 로 기술 | 09-25 `P3_CLOSURE_TEST` 정정 |
| "위치를 맞춰 주면 한계가 사라진다" (qshift) · "매개 95 % / 99 %" | 근거리 비용 · 정확 칸 .08–.09; 층화하면 44–67 % | qshift 시사 강등 · 매개 "약 절반" 시사 | 09-25 / 09-26 `I_SCENE` 정정 · `VERIFY_ISCENE2` |
| "encoder 가 경계 토큰에 운동을 요약해 넣는다" | bndM 은 있음 붕괴 (물체다움 0.63 → 0.23) 가 먼저 — 위치로 못 가름 | 판정 보류 (가설) | 09-25 `M6` 정정 (5) · `M2B` |
| "p 는 어디서 바뀔지 안다" · "predictor 의 미래는 시간을 따라 진전하지 않는다 (기울기 0.01)" (SSv2) | 공유 기준 인공물 · 두 방향 평균 (정 −0.244 / 역 +0.269) | 철회 | 09-25 `E_SSV2` 정정 1 |
| EK100 "encoder 가 ego-motion 을 표적 신뢰도 천장까지 읽는다" · "74 %" | train = test 최소제곱 값을 천장으로 씀 | "flow 자기 일관성보다 높게" · 67–75 % | 09-29 `VERIFY_R3` B |
| "v11 모양 self 는 인코더도 흔들린다" | z 64.8 / 42.4 는 VIS → OCC **이식**, OCC self 는 z 99.9–100 | "shape 은 이식에서 encoder 도 흔들린다" 로 정확히 | 09-24 `H7` 정정 (4) |
| "v2 ledge 에서 p 는 안 된다" | 자 B 로 p 5.4 px · 낙하량 52.8 | p 가 z 에 못 미치는 곳은 v11 late 가림만 | 10-02 `FUTURE_FROM_CONTEXT` 정정 |
| 캐시 `ctx_masked` 를 predictor 입력으로 | 캐시는 LN(z) (`forward.py:163`) — 캐시 p 와 상대 L1 0.60 | predictor 를 다시 돌리려면 문맥 encoder 를 다시 돌린다 (readout 특징으로 쓰는 것은 무방 — FFC 의 z) | CLAUDE.md §6 · `PREDICTOR_DISTANCE_LIMIT_2026-09-19.md` §5 |

---

## 6. 문서끼리 어긋나는 수치 — 조건 차이

| 무엇 | 값 A | 값 B | 조건 차이 | 출처 |
|---|---|---|---|---|
| z 속도 R² | 0.996 / 0.989 (RollOutV2) | 0.988 (F1) / 0.989 (h) (ctxenc E1) · .983 / .976 pooled, 그룹 안 .949 (v3 M13) | 특징 (두 튜블릿 평균 vs t7 3×3 vs z_last) · 데이터 (v2 9 시나리오 vs 6 시나리오 n=300 vs v3 궤적 84) · 그룹 안 분산 | `PREDICTOR_DISTANCE_LIMIT` §5-3 · `ctxenc_boundary_kinematics` · `M13` |
| 층 8 단일 knockout | −18.06 | −14.4 | 144 block 표본 (`knockout_window__v11_vith`, n_pair 288, 기준 72.92) vs v11_full 전수 (`knockout_w1`) | `figures/knockout/README.md` · `PAPER_STORY_2026-09-06.md` 재현절 |
| knockout 기준선 | 75.86 (`clean_null`) | 76.00 (공식 v11_full) · 73.37 (v11 6 조건) | SDPA 커널 변화 0.14 pp · 데이터셋 다름 | `IntPhysGenV11_occlusion_timing_ablation/README.md` · `CROSS_REVIEW` A6 |
| 비가림 → 가림 p 이식 (color) | 17.3 (3 팔 평균) | late 15.1 / 20.6 / 16.4 | `attn_probe` (v11) vs `attn_probe_xfer` (v11_full) — 같은 self 칸이 108 중 1 개에서 6.1 pt 벌어짐 | `RESULTS_2026-08-30.md` §8 · timing README 단서 |
| p participation ratio | 15.4–18.7 (v2, ramp 포함 9 set) · 15.8–18.7 (overlap, 6 set) | 16–19 | set 범위 (ramp 포함 여부) 와 반올림 요약 — 같은 측정 | `ctxenc_subspace_v2` §4 · `ctxenc_subspace_overlap` · `CLAIM` §2-6 |
| encoder 충분성 | "철회" (README §2, 09-25) | "encoder 도 병목 아님" (I_SCENE C3, 09-25 저녁) · "encoder 충분 (확정)" (RETHINK §1-b · ENCODER_MOTION, 09-29) | 철회된 것은 **분업 귀속** (충분성 대조가 진실 위치 사용); 다시 선 것은 **범위가 좁은 정보 문장** (출처 조회 개입 · 전역 ego-motion · 미래 위치 자) | `auto_research/README.md` §2 · `I_SCENE` · `RETHINK` · G03 추출 (c) |
| 가려진 물체 정체 | z_hid self 95.7 / 93.9 | 경계 B7 visible → late 이식 20.2 / 16.0 | self (가린 조건 안 학습) vs 이식 (보이는 조건 학습) — 둘 다 맞다 | `RollOutV3/Archive/RESULTS_2026-09-25.md` §6 · `ctxenc_boundary_identity_v2` |
| p 가 미래 위치를 담나 | v2 (가림 없음) p 자 B ledge 5.4 px (대부분 남음) | v11 late p 자 B 16.9–29.9 px | 가림 유무 · 자 B — RollOutV2 위치 자 (등속 학습셋) 는 p 가 선반 높이 유지 · wall 2 슬롯 뒤 소실 | `FUTURE_FROM_CONTEXT` §1 · §2 · CLAUDE.md §5-5 |
| 정체 · 위치 (v1 층별 probe) "단조 증가" | MECHANISM (v1, 선형 probe 정확도) 68.4 → 85.5 → 88.7 | H7 (v11, 렌즈 채점) 비단조 (L1 57.1 → L2 55.2) | 다른 양 (정체 probe vs 채점 정확도) · 다른 데이터 — 둘 다 유효 | `MECHANISM` §2.1 · `H7` 정정 (1) |
| SSv2 방향 probe n | 698 | 677 | 원본 ≥ 32 프레임 clip 만 (정정) | `E_SSV2` 정정 1 |

---

## 출처

정본 · 지도
- `auto_research/paper/PAPER_STORY_2026-10-08.md` (§2-3 (iv) · §2-3 (v) · §4-5 · §7)
- `auto_research/README.md` (§2 철회표 · 논리 사슬)
- `auto_research/paper/RETHINK_PREDICTOR_ROLE_2026-09-29.md` (대체됨, §1-b 표 유효)
- `CLAUDE.md` (§1-4 · §5-2 · §5-5 · §6 · §7-3)

encoder 정보
- `z_research/RollOutV2/Archive/PREDICTOR_DISTANCE_LIMIT_2026-09-19.md`
- `z_research/context_encoder_analysis/Archive/STATE_EVOLUTION_CAPABILITIES_2026-09-19.md`
- `z_research/context_encoder_analysis/Archive/CLAIM_NO_STATE_EVOLUTION_2026-09-15.md`
- `z_research/context_encoder_analysis/Archive/STATE_EVOLUTION_HYPOTHESES_2026-09-15.md`
- `z_research/context_encoder_analysis/Archive/EXPERIMENTS_2026-09-16.md`
- `z_research/context_encoder_analysis/Archive/ENCODER_TEMPORAL_DYNAMICS_2026-09-20.md`
- `z_research/context_encoder_analysis/exp_results/ctxenc_boundary_kinematics/RESULTS.md` (+ `n30/RESULTS.md`, 대체됨)
- `z_research/context_encoder_analysis/exp_results/ctxenc_state_subspaces_kinematics/RESULTS.md`
- `z_research/context_encoder_analysis/exp_results/ctxenc_boundary_identity_occlusion/RESULTS.md`
- `z_research/context_encoder_analysis/exp_results/ctxenc_boundary_identity_v2/RESULTS.md`
- `z_research/context_encoder_analysis/exp_results/ctxenc_subspace_overlap/RESULTS.md`
- `z_research/context_encoder_analysis/exp_results/ctxenc_subspace_v2/RESULTS.md`
- `z_research/context_encoder_analysis/exp_results/ctxenc_time_alignment_gated/RESULTS.md`
- `z_research/RollOutV3/Archive/RESULTS_2026-09-25.md` (§6 · §7 정본)
- `z_research/RollOutV3/audit/v11_identity/AUDIT.md`
- `z_research/RollOutV3/figures/v11_probe/PROBE.md`
- `auto_research/Archive/FUTURE_FROM_CONTEXT_2026-10-02.md`
- `auto_research/Archive/H1_ACCEL_IN_INPUT_2026-09-24.md`
- `auto_research/Archive/M13_MEMORY_VELOCITY_SUFFICIENCY_2026-09-25.md`
- `auto_research/Archive/E_SSV2_ENCODER_READS_PREDICTOR_STALLS_2026-09-25.md`
- `auto_research/Archive/ENCODER_MOTION_READOUT_2026-09-29.md`
- `auto_research/Archive/VERIFY_R3_2026-09-29.md`
- `auto_research/Archive/INTPHYS2_ENCODER_LINEAR_PROBE_2026-09-30.md`
- `analysis/intphys1/PAPER_VS_OURS.md`
- `analysis/intphys2/ANALYSIS_NOTES.md`

p 의 정보 · 기하
- `z_research/IntPhysGenV11/Archive/surprising_score/RESULTS_2026-08-30.md`
- `z_research/IntPhysGenV11_occlusion_timing_ablation/README.md`
- `z_research/IntPhysGenV10/Archive/V10_2026-08-28.md`
- `z_research/IntPhysGenV10/Archive/GEOMETRY_2026-08-29.md` (α 절 인용 금지)
- `z_world_model_analysis/MECHANISM_2026-08-20.md` (§7 α 인용 금지)
- `z_research/IntPhysGenV11/Archive/PAPER_STORY_2026-09-06.md` · `PAPER_STORY_2026-09-22.md` (대체됨, 수치 유효)

predictor 기제
- `analysis/attention/README.md` · `analysis/attention/predictor_attn/README.md` · `analysis/attention/knockout/README.md`
- `z_research/IntPhysGenV11/figures/attention/README.md` · `z_research/IntPhysGenV11/figures/knockout/README.md`
- `auto_research/Archive/PLAN_2026-09-24.md`
- `auto_research/Archive/H4_QUERY_AND_POSITION_2026-09-24.md`
- `auto_research/Archive/H5_AUTOREGRESSIVE_FEEDBACK_2026-09-24.md`
- `auto_research/Archive/H7_V11_LAYERWISE_2026-09-24.md`
- `auto_research/Archive/H8_KNOCKOUT_PERSISTENCE_2026-09-24.md`
- `auto_research/Archive/H9_FROZEN_INTERVENTION_SCORING_2026-09-24.md`
- `auto_research/Archive/H23_PREDICTOR_LAYERS_TRAJECTORY_2026-09-24.md` · `auto_research/exp_results/h23b/REPORT_h23b.md`
- `auto_research/Archive/CROSS_REVIEW_2026-09-24.md`
- `auto_research/Archive/SC1_REACH_NOT_HORIZON_2026-09-25.md`
- `auto_research/Archive/M2_BOUNDARY_PATCH_2026-09-25.md` · `auto_research/Archive/M2B_BOUNDARY_V2_2026-09-25.md` (1 차, 인용 안 함)
- `auto_research/Archive/M4_RETRIEVAL_RADIUS_2026-09-25.md`
- `auto_research/Archive/M5_VELOCITY_STEERING_2026-09-25.md`
- `auto_research/Archive/M6_HISTORY_SHIFT_2026-09-25.md`
- `auto_research/Archive/P2_STRIDE_SWEEP_HORIZON_UNIT_2026-09-25.md`
- `auto_research/Archive/P3_CLOSURE_TEST_2026-09-25.md` · `auto_research/Archive/P3B_LAST_TOKEN_DRIVES_TRANSITION_2026-09-25.md`
- `auto_research/Archive/I_SCENE_CAUSAL_BOTTLENECK_2026-09-25.md` · `auto_research/Archive/VERIFY_ISCENE_2026-09-25.md` · `auto_research/Archive/VERIFY_ISCENE2_2026-09-26.md`

충분성 이력 · 적대 검증
- `auto_research/Archive/PLAN_ENCODER_PREDICTOR_2026-09-25.md`
- `auto_research/Archive/README_UNTIL_2026-09-25.md`
- `auto_research/Archive/VERIFY_0925_2026-09-25.md` · `auto_research/Archive/VERIFY_0925_R2_2026-09-25.md`
- `auto_research/Archive/TRAINED_PREDICTORS_SIGNATURES_2026-09-25.md`
- `auto_research/Archive/COPY_VS_POSITION_2026-10-07.md` (토큰 평균 L1 이 위치를 못 봄)
