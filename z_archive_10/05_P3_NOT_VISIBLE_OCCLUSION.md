# 05 — P3 (ii) 마지막 관측에 안 보이는 것: 가림 (네 능력 4 영속)

> **결론** — 가림이 **마지막 관측 (문맥 끝)** 에 걸리면 V-JEPA 2 ViT-H 의 vanish 채점은 '물체→빈' 0 / '빈→물체' 100 으로 갈라져 50 % 가 되고, 학습된 자로 읽은 p 의 미래에는 처음부터 물체다운 토큰이 없다. 문맥 가운데 가림은 거의 처리한다. 정체 정보는 입력 (z) 과 p 에 남아 있지만 p 는 그것을 가려진 물체의 미래 자리에 놓지 않는다.
> **근거** — (1) v11 등가속 (ramp) 가림 행 vanish 50.0 = A 0.00 / B 100.00 (k=1–4 전부; 등속 flat 가림 52.0 = A 0–12.5 / B 100), 같은 행 shape 54.5, k=1–4 평평 (`z_research/IntPhysGenV11/Archive/surprising_score/RESULTS_2026-08-30.md` 표 A·B·D). (2) 가림 타이밍: visible 87.2 / early 79.0 / mid 78.2 / **late 59.5**, 총 가림 장수를 맞춰도 +20.2 / +17.3 (`z_research/IntPhysGenV11_occlusion_timing_ablation/README.md`). (3) 실물 메시 A 0–10 / B 100 (`z_research/v11_realistic/README.md` §1) · K400 팬 가림 띠 hid .21 vs vis .42, 같은 칸 비가림 대비 약 1/3 (`auto_research/Archive/I_SCENE_CAUSAL_BOTTLENECK_2026-09-25.md` §4-4, 2 차 검증 확정).
> **미결** — late 가 "문맥 끝을 가려서" 인지 "미래 앞을 가려서" 인지 가르는 팔이 없다 (대칭 가림). 자 없는 vanish 비교 (이동 k≥2 98–100 % 빈 미래 쪽) 와 identity_r8 자 (ramp k=2 t1 '있음' 69 %) 사이의 긴장도 남는다.
> 이 장이 받치는 논문 문장: P3 (ii) "마지막 관측에 **안 보이는 것** (가려진 물체) 은 예측하지 못한다." ("context 에서 안 보이면" 이 아니라 **마지막 관측에** — 문맥 가운데 가림은 처리한다)  ·  원재료: G06 · G07 · G11 · G08 · G08b · G09 · G03 · G03b · G02a · G10b · G10c · G12 추출문 → 원 문서는 각 절 끝 `출처`.

---

## 0. 범위 · 읽는 법

| 항목 | 이 장에서의 뜻 | 출처 |
|---|---|---|
| "안 보이는 것" | 마지막 관측 (문맥 끝 튜블릿) 에서 물체가 가림막 뒤에 있음. v11 `late` = **문맥 끝 k 장 + 미래 앞 k 장 대칭 가림** (`sym_k` 1–4; `n_context_hidden == k` 실측) | CLAUDE.md §1-7 · `z_research/IntPhysGenV11/README.md` §3 |
| 네 능력 | **4 영속** (안 보이는 것). 현재 V-JEPA 2 판정 ❌ (late 0 슬롯) | `z_research/context_encoder_analysis/Archive/STATE_EVOLUTION_CAPABILITIES_2026-09-19.md` §2 |
| 도구 넷 | ① 채점 (`surprise_c16t32`, matched pair, A/B 방향 병기) · ② 학습된 자 (위치 · 있음 · 정체) · ③ 자 없는 검사 (파라미터 0, margin 지도) · ④ probe (정보 존재) | CLAUDE.md §1-2 · §1-4 · §7-3 |
| 읽기 규칙 | 자는 물체가 없어도 위치를 낸다 → 위치 주장은 그 슬롯 3×3 질량 > 균등 0.035 일 때만 · 자 판독에는 held-out · encoder 대조 · 자 없는 교차검증을 같이 | `z_research/RollOutV2/figures/v5/summary/POSITION_READOUT_2026-09-12.md` §읽는 규칙 |
| 설계 결정 | 가림 유무 = "물체가 가려지는가" (가림막 존재는 구현). 가림막 폭이 k 와 함께 커지는 것도 설계 | CLAUDE.md §8-5 |

---

## 1. v11 채점 — 가림이 문맥 끝에 걸리면 무너진다

v11: 21,504 clip / 5,376 block / **10,752 matched pair**, ViT-H `surprise_c16t32`, `report.json.verified = true`, overall **73.37 %** (45 cell). 등급: 확정 (2026-10-08 `per_block.json` 에서 matched 쌍을 다시 셌다 — `auto_research/paper/PAPER_STORY_2026-10-08.md` §재현).

### 1-1. 조건 × 위반 (k 합산, acc %)

| condition | vanish | shape | color | 출처 |
|---|---:|---:|---:|---|
| `static_visible` | 100.0 | 94.0 | 81.8 | RESULTS 표 A |
| `static_occlusion` | 70.5 | 54.8 | 55.4 | 〃 |
| `moving_visible_flat` | 100.0 | 89.9 | 86.0 | 〃 |
| `moving_occlusion_flat` | 52.0 | 56.8 | 68.5 | 〃 |
| `moving_visible` (ramp) | 100.0 | 69.0 | 76.8 | 〃 |
| `moving_occlusion` (ramp) | **50.0** | 54.5 | 71.4 | 〃 |

가림 행만 sensitivity / bias 로 풀면 (`sens = (acc(A→B)+acc(B→A))/2 − 50`, `bias = |acc(A→B) − acc(B→A)|/2`):

| condition | vanish sens / bias | shape sens / bias | color sens / bias | n pair / cell |
|---|---|---|---|---|
| `static_occlusion` | +20.54 / 3.57 | +4.76 / 27.68 | +5.36 / 9.23 | vanish 448 · shape/color 672 |
| `moving_occlusion_flat` | +2.01 / 47.99 | +6.85 / 26.19 | +18.45 / 14.58 | 〃 |
| `moving_occlusion` | **+0.00 / 50.00** | +4.46 / 35.42 | +21.43 / 13.39 | 〃 |

(출처: `z_research/IntPhysGenV11/Archive/surprising_score/RESULTS_2026-08-30.md` 표 A)

- 가림 행은 전부 바닥, 비가림 행은 전부 높다 (vanish 100 ×3). 색만 가림에서 sens 18–21 로 일부 남는다.
- ⚠️ 가림 행끼리 등속 vs 등가속 (shape 56.85 vs 54.46) 은 **바닥 효과** — "가림 하에서 가속 무관" 금지 (RESULTS §3-(2) 정정 08-30).
- ⚠️ `static_occlusion` vanish 만 sens +20.54 로 살아남는데, 같은 조건 `p` self probe 가 가장 낮다 (85.13 / 87.55, 미수렴 head train_acc 0.931 / 0.954) — **미해결** (RESULTS §11-(5)).

### 1-2. k — 0→1 절벽, 1–4 평평

| 위반 | k=0 | k=1 | k=2 | k=3 | k=4 | 출처 |
|---|---:|---:|---:|---:|---:|---|
| vanish sens | 50.0 | 11.3 | 7.4 | 6.9 | 4.5 | RESULTS 표 B |
| shape sens | 34.3 | **5.0** | 5.2 | 6.2 | **5.2** | 〃 |
| color sens | 31.6 | 13.1 | 12.9 | 13.5 | 20.8 | 〃 |
| permanence bias | 0.0 | 31.0 | 29.5 | 33.5 | **40.8** | 〃 |

- acc 로: k=0→1 영속 100.0 → 61.3 (−38.7) · 모양 84.3 → 55.0 (−29.4) · 색 81.5 → 63.1 (−18.5). k=1→4: 영속 −6.85 [−14.3, +0.6] · 모양 **+0.20 [−5.9, +6.3]** · 색 **+7.74 [+1.9, +13.5]** (`z_research/IntPhysGenV11/Archive/SYNTHESIS_2026-08-30.md` §①).
- ⚠️ "k=1–4 평평" 이 깨끗한 것은 **모양** 이다. 색은 k 와 함께 오르고 (k=4 sens 20.8, 설명 미상 — RESULTS §3-(4)), 영속성은 약하게 내려간다 (CI 0 걸침). "가려진 **시간** 은 무관" 은 이 범위로 쓴다.
- 재해석 (timing ablation): v11 의 모든 k 는 가림이 예측 경계에 닿아 **이미 바닥** 이라 평평하다 (§2).

### 1-3. vanish 50 % = A 0 / B 100

| condition | k | 물체→빈 (A) | 빈→물체 (B) | n / 방향 | 출처 |
|---|---|---:|---:|---:|---|
| `static_occlusion` | 1 | 78.57 | 76.79 | 56 | RESULTS 표 D |
| `static_occlusion` | 4 | **41.07** | 85.71 | 56 | 〃 |
| `moving_occlusion_flat` | 1 | 12.50 | 100.00 | 56 | 〃 |
| `moving_occlusion_flat` | 4 | **0.00** | 100.00 | 56 | 〃 |
| `moving_occlusion` (ramp) | 1–4 전부 | **0.00** | **100.00** | 56 | 〃 |

- 움직이는 물체가 문맥 끝에서 가려지면 채점은 **"물체가 있는 미래" 를 항상 덜 놀랍다고 본다** — 가능 미래가 비었을 때 (A) 0 %, 가능 미래에 물체가 있을 때 (B) 100 %. 50 % 는 탐지가 아니라 상쇄다 (CLAUDE.md §1-7 "A/B 방향별 정확도를 항상 병기").
- **permanence 편향은 k 와 함께 커진다** (bias 31.0 → 40.8, 탐지력과 별개): 정지+가림 A k1 78.6 / k2 82.1 / k3 66.1 / k4 41.1 (Δ −37.5, SE 8.6, −4.38σ) (`SYNTHESIS_2026-08-30.md` §⑧ H2).
- ⚠️ "물체가 있는 미래 쪽 편향" 은 채점 (scalar surprise) 의 행동이지 p 가 물체를 그린다는 뜻이 아니다 — p 의 미래에는 물체다운 토큰이 없다 (§4). 둘의 조화는 §6.

### 1-4. 방향 비대칭 (짧게 — 가림 특이가 아님)

| 항목 | 값 | 출처 |
|---|---|---|
| 등가속+가림 shape 지킴 순서 | `cylinder 90 > cube 86 > pyramid 59 > capsule 58 > cone 48 > sphere 36 > torus 3` | `z_research/IntPhysGenV11/Archive/SYNTHESIS_2026-08-30.md` §5 |
| 순서 일치 | 42 방향 중 40 · Hodge R² 0.82–0.91 (무작위 기준선 6/21 = 0.286) | 〃 §5-3 |
| 순서는 가림이 만들지 않는다 | 등가속·비가림 R² **0.816** (mean\|b\| 26.8) · 정지/등속 비가림 2.7 / 4.2 < 잡음바닥 7.1 | 〃 §6 |
| precision vs recall | recall 폭 **86** vs precision 폭 **24** (cylinder R 89.6 / P 51.5 · torus R 3.1 / P 75.0); 대조군 정지·비가림 6 / 12 | `z_research/IntPhysGenV11/Archive/surprising_score/VOTE_GRID_2026-08-31.md` §4 |
| probe vs 채점 | torus probe recall 100 % 인데 채점 지킴 3 % | 〃 · CLAUDE.md §5-3 |

- 결론: 능력 차이가 아니라 과생산 / 과소생산 (편향). 탐지 신호를 죽이는 조건이면 (가림 · 가속) 같은 순서가 드러난다 → 이 장의 주장 (가림) 과 06 장 (가속) 이 같은 출력 편향을 공유한다.
- ⚠️ "특정 물체로 몰린다" 성립 안 함 (최다 목적지 열 질량 21–40 %, 균등 16.7 %) → "순서 위쪽으로 흘러내린다". (조건, k) 로 쪼개면 비대각 방향당 n=4 — 칸 단독 인용 금지.

출처 (§1): `z_research/IntPhysGenV11/README.md` §4 · `z_research/IntPhysGenV11/Archive/surprising_score/RESULTS_2026-08-30.md` · `z_research/IntPhysGenV11/Archive/SYNTHESIS_2026-08-30.md` · `z_research/IntPhysGenV11/Archive/surprising_score/VOTE_GRID_2026-08-31.md`

---

## 2. 언제 무너지나 — 가림이 예측 경계 (마지막 관측) 에 닿을 때

`v11_full` = v11 6 조건 + `*_early` / `*_mid` 6 조건 → 43,008 clip / **21,504 pair** (overall 76.00), ViT-H, `report.json verified=true` (채점 cell 117). 설계: 문맥 raw 0–45 / 미래 48–93. `early` 가림 raw 12–21 (문맥 k 장), `mid` 21–30 (문맥 k 장), `late` = v11 본체 (문맥 끝 k + 미래 앞 k = 2k 장). 등급: 정본 세트 README (등급 문서에 없음, 산출물 대조 검증 4 층 불일치 0).

### 2-1. 타이밍 × 위반

| timing | vanish | shape | color | 전체 | n pair |
|---|---:|---:|---:|---:|---:|
| visible (가림막 없음) | 100.0 | 84.3 | 81.5 | **87.2** | 5,376 |
| early | 97.8 | 78.1 | 67.4 | **79.0** | 5,376 |
| mid | 95.8 | 76.0 | 68.7 | **78.2** | 5,376 |
| **late (= v11)** | 57.5 | 55.4 | 65.1 | **59.5** | 5,376 |

| 운동 | visible | early | mid | late |
|---|---:|---:|---:|---:|
| static | 91.0 | 79.0 | 77.4 | **58.9** |
| moving_flat | 91.0 | 83.0 | 84.2 | **60.0** |
| moving (ramp) | 79.7 | 75.1 | 73.1 | **59.7** |

(출처: `z_research/IntPhysGenV11_occlusion_timing_ablation/README.md` §채점 결과)

### 2-2. 총 가림 장수를 맞춰도 남는 격차

| 총 가림 장수 | early/mid | late | 격차 |
|---|---|---|---:|
| 2 장 | k=2 **79.8** | k=1 **59.6** | **+20.2** |
| 4 장 | k=4 **75.9** | k=2 **58.6** | **+17.3** |

- 단계별 하락 (visible→early / early→mid / mid→late): vanish −2.2 / −1.9 / **−38.3** · shape −6.2 / −2.1 / **−20.7** · colour **−14.1** / +1.3 / −3.6.
- k × 타이밍 (전체): early 81.7 / 79.1 / 78.9 / 76.4 · mid 80.5 / 80.5 / 76.6 / 75.4 · late 59.6 / 58.6 / 59.1 / 60.9 (k=1..4).
- 영속성의 k 반응은 ramp 에서만: early 97.3 → 89.3, mid 92.9 → 76.8 (static · flat 100 유지).
- 효과크기 (margin / base |p − h|) 하나가 12 셀 정확도를 설명 (Spearman ρ **0.986**); late 효과크기 0.44–0.22 vs visible 1.30–0.76. early/mid 는 "신호가 묽어진 것", late 는 거기에 "자리 변화" 가 더해진 다른 고장 (README 결론 10).

문서가 세운 문장: "문맥 안 어디를 가리든 거의 같다 (early 79.0 vs mid 78.2) · 양도 거의 무관 · **결정적인 것은 가림이 예측 경계에 닿는가** · 문맥만 가리면 8 pt 손해, 경계에 닿으면 27.7 pt · 세 운동이 late 에서 한 점으로 모인다."

- ⚠️ **"미래를 가려서" 인지 "문맥 끝을 가려서" 인지 못 가린다** (late 는 둘 다; 문맥 끝만 가리는 팔이 없다). P3 (ii) 의 "마지막 관측에" 는 이 대칭 설계 위에서의 문장이다.
- ⚠️ "늦게 가릴수록 나쁘다" 못 쓴다 (early ≈ mid). 채점만으로 predictor 내부를 말할 수 없다.
- ⚠️ 계단은 위반마다 다르다 — 영속성만 깨끗한 계단, 모양은 두 계단, 색은 첫 계단만. k × 위반 셀 504 쌍, 운동까지 쪼개면 168 쌍 (선 모양만).
- ⚠️ 폐기된 pilot (6,048 pair, 68.11 %, early/mid 가 문맥을 2k 장 가림) **인용 금지** (`datasets.md` `v11_occtiming` `available: false`).
- ⚠️ 원 문서 간 작은 차: `STATUS_AND_CRITIQUE_2026-09-03.md` §1-1 의 colour 행은 81.5 / 67.4 / **67.6 / 65.3** 으로 README (68.7 / 65.1) 와 다르다 — 이 장은 README (정본) 값을 쓴다.

### 2-3. 가림막 색 가설 — v13_black 으로 기각

| | v11_full | v13_black |
|---|---|---|
| 가림막 색 | `wood` | **`black`** |
| 규모 | 43,008 clip | 4,608 clip / 2,304 pair (1/9) |
| colour visible→early 하락 | −14.1 | **−13.5** (같음 → 색면 가설 기각) |
| shape visible→early 하락 | −6.2 | **0.0** (나뭇결이 모양을 깎고 있었다) |

(출처: `z_research/IntPhysGenV11_occlusion_timing_ablation/README.md` "⚠️ 가림막 색 가설은 기각됐다 (2026-09-03)" · `configs/protocols/datasets.md` `## v13_black`)
- ⚠️ 원 README 가 가리키는 `z_research/IntPhysGenV13_Black_occluder/README.md` 는 **폴더가 없다** (CLAUDE.md §6). 결과는 레지스트리와 실행 산출물에만. 1/9 규모라 타이밍 × 위반까지만 읽는다.
- ⚠️ 원 README 본문에 "아직 안 했다" 는 옛 경쟁 가설 문단이 그대로 남아 있고 그 위에 기각 문단이 붙어 있다.

### 2-4. 옛 데이터의 같은 현상과, v10 · v11 이 뒤집은 것

| 날짜 · 데이터 | 문서가 세운 것 | 지금 상태 | 출처 |
|---|---|---|---|
| 08-16 blender occlusion_v2 재렌더 (ViT-L bf16, 문맥일치 2 쌍) | presence **50.00** = `pos_empty<imp_appear` 100.00 / `pos_obj<imp_vanish` **0.00** (131 block 전부, Cohen d −2.83) | 같은 A/B 0/100 패턴의 가장 이른 기록. 유효 (기록) | `z_world_model_analysis/RESULTS_2026-08-16.md` §3.2–3.3 |
| 08-19 blender occlusion_v3 (ViT-H, C 4/8/12/16) | `pos_obj<imp_vanish` 0–3 % 가 7 설정 전부 · context 길이 가설 기각 (50.9–54.3) | 기각 유지 (CLAUDE.md §6) | `z_world_model_analysis/FRAME_ABLATION_2026-08-19.md` §4 |
| 08-20 blender v4 자연 변동 (ViT-H) | **"가림 길이가 원인 ✅"** — n_hidden 3–7 64 % → 10–14 53.9 % (단 가림 길이 · 증거 프레임 · 가림막 크기가 얽혀 분리 불가라고 스스로 적음) | **뒤집힘** | `z_world_model_analysis/OCCLUSION_LENGTH_2026-08-20.md` §1–2 |
| 08-28 v10_occ_low (가림 4 → 1 프레임) | moving_occlusion 56.25 → 57.03, static_occlusion 60.25 → 60.16 — **변화 없음**; mov+occ permanence 50.00 | 가림 길이 무관 (첫 반증) | `z_research/IntPhysGenV10/Archive/V10_2026-08-28.md` §1a |
| 08-30 v11 (k 1–4 데이터셋 축) | k=1–4 평평 (shape sens 5.0 → 5.2) — "가림 길이 통제로 L1 이 IntPhys 수준까지 오르나" → **아니오** | 유효 | RESULTS §3-(1) |
| 09-01~03 v11_full timing | 양 · 문맥 안 위치 무관, **경계에 닿을 때만** (+20.2 / +17.3) | 정본 | timing README |

- ⚠️ `OCCLUSION_LENGTH` 머리에는 대체 표시가 없다. 그 문서의 설계 원칙 (distractor `|v| > 100 cm/s`, 가림 길이는 가림막 크기가 아니라 속도 · 타이밍으로) 은 CLAUDE.md §11 에 계승됐고, 결론 "가림 길이가 레버" 만 뒤집혔다.

출처 (§2): `z_research/IntPhysGenV11_occlusion_timing_ablation/README.md` · `z_research/IntPhysGenV11/Archive/STATUS_AND_CRITIQUE_2026-09-03.md` · `configs/protocols/datasets.md` · `z_world_model_analysis/RESULTS_2026-08-16.md` · `z_world_model_analysis/FRAME_ABLATION_2026-08-19.md` · `z_world_model_analysis/OCCLUSION_LENGTH_2026-08-20.md` · `z_research/IntPhysGenV10/Archive/V10_2026-08-28.md`

---

## 3. 실물 메시 · 실영상 · 벤치마크

### 3-1. v11_realistic — 실물 메시에서도 같다

`v11_realistic`: 실물 메시 15 종, 1,440 clip / 360 block / **720 쌍**, vanish 만, ViT-H `surprise_c16t32`, 문맥 무결성 mismatch 0. overall **79.58 % [77, 82]** (기하 v11 vanish 행 78.8 [77, 80], 2,688). 등급: 정본 세트 README · 2026-10-08 `tables.json` 재확인 ✓.

| condition | 실물 전체 | 실물 A 물체→빈 | 실물 B 빈→물체 | 기하 v11 전체 | 기하 A | 기하 B |
|---|---:|---:|---:|---:|---:|---:|
| `static_visible` | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| `static_occlusion` | 72.5 [66, 79] | 61.7 | 83.3 | 70.5 [67, 74] | 67.0 | 74.1 |
| `moving_visible_flat` | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| `moving_occlusion_flat` | 55.0 [52, 59] | **10.0** | 100.0 | 52.0 [51, 53] | 4.0 | 100.0 |
| `moving_visible` (ramp) | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| `moving_occlusion` (ramp) | 50.0 [50, 50] | **0.0** | **100.0** | 50.0 | 0.0 | 100.0 |

(출처: `z_research/v11_realistic/README.md` §1 · `z_research/v11_realistic/exp_results/tables.md`)

| 가림 × k (실물; A 칸 n 15 · 전체 칸 n 30) | k=1 | k=2 | k=3 | k=4 |
|---|---:|---:|---:|---:|
| `moving_occlusion_flat` A | 20.0 | 20.0 | 0.0 | 0.0 |
| `moving_occlusion` A | 0.0 | 0.0 | 0.0 | 0.0 |
| `static_occlusion` 전체 | 80.0 | 66.7 | 76.7 | 66.7 |

- 물체 15 종 비가림 100 · 가림 A 8–42 % (물체당 n 12, CI ±25) — 물체 간 차이 주장 불가.
- 토큰별 surprise 지도 (k=2): `static_occlusion A` 는 물체 자리 |Δ| **0.042** ≈ 배경 0.043 — p 가 "물체 있음" 과 "빈 바닥" 에서 똑같이 멀다; `moving_occlusion A` 는 물체 · 배경 기여 둘 다 100 % 틀린 방향 (`z_research/v11_realistic/figures/token_surprise/README.md`, `background_checks.md`).
- ⚠️ 규모 v11 의 1/4 (조건 × 방향 60 쌍, k 칸 15 쌍 — 선 모양만). A/B 방향 항상 같이.

### 3-2. 실영상 — K400 팬 + 마지막 k 프레임 가림 띠

묶음 Q: K400 팬 (카메라 운동) 900 clip 의 마지막 k 프레임에 화면 고정 가림 띠. 판독 = 파라미터 없는 출처 판독 (가려지기 전 튜블릿 L−m 출처 템플릿 argmax 가 출처 1 칸 안). hid = 프레임 15 에서 띠 안, vis = 같은 clip 띠 밖. 등급: **2 차 검증 확정** (`auto_research/Archive/VERIFY_ISCENE2_2026-09-26.md` §1, 표 전 칸 재계산 일치).

| k · 변위 | release hid / vis | Ariel | N1 (Ariel + 팬 post-FT) | pv1 |
|---|---|---|---|---|
| k=2 · 0–1 칸 | .26 / .36 | .29 / .42 | .29 / .43 | .27 / .38 |
| **k=4 · 0–1 칸** | **.21 / .42** | .21 / .45 | .21 / .48 | .19 / .39 |
| k=4 · 1–2 칸 | .21 / .38 | .20 / .42 | .18 / .45 | .17 / .33 |
| k=4 · 2–3 칸 | .17 / .25 | .17 / .31 | .15 / .36 | .15 / .23 |
| k=8 · 0–1 칸 | .19 / .38 | .19 / .39 | .21 / .39 | .19 / .36 |

(출처: `auto_research/Archive/I_SCENE_CAUSAL_BOTTLENECK_2026-09-25.md` §4-4)

| 2 차 검증이 더한 것 | 값 | 출처 |
|---|---|---|
| 같은 800 clip 비가림 판 (pan_cm) 같은 칸 대조 | hid 칸 **.63 → .21** (약 1/3 만 남음), vis 칸 .45 → .43 | `VERIFY_ISCENE2_2026-09-26.md` §1 |
| "가림막을 그린다" | release k=4 0–1 칸 cos(원래 내용) .39 vs cos(가림막) .57. encoder 진짜 미래 토큰은 회색 .28 / 내용 .48, predictor 출력은 .53 / .42 (뒤집힘) | 〃 |
| hid/vis 비 (d < 2) k 단조 | k2 .76 > k4 .53 ≈ k8 .49 | 〃 공격 4 |
| (hid − vis) clip 평균 차 | k4 −0.26 [−0.28, −0.24] · k8 −0.22 · k2 −0.12 | 〃 |
| 정확 칸 판독 | hid .03–.04 vs vis .10–.16 → "역사 경로 약 .2" 는 **추정** (1 칸 허용폭 누출 미배제) | 〃 공격 5 |

- 문서의 결론: 마지막 관측에서 가려진 내용은 제자리 복원이 같은 칸 비가림 대비 약 1/3, **네 predictor 가 같다** (미래 예측 학습 Ariel · N1 도 영속을 만들지 않음), ≥ 3 칸에서 hid ≈ vis.
- ⚠️ 정본 스토리 (`PAPER_STORY_2026-10-08.md` §2-3 ii) 는 "보이는 내용의 **절반**" 이라 쓰는데, 2 차 검증은 그것을 "보수적 (자리 편향 섞여 과소)" 이라 하고 "같은 칸 비가림 대비 약 1/3" 을 권고했다 — 두 수치는 기준 (vis 칸 vs 같은 칸 비가림) 이 다르다. ⚠️ "절반은 같다" 는 절대값 (hid .17–.21) 기준이고, 비율은 학습한 쪽 (N1 · T1) 이 조금 낮다 (vis 가 오른 탓). 팬은 카메라 운동 하나.

### 3-3. IntPhys 2 permanence — 우연과 구분 안 됨

IntPhys 2 Main 506 type-matched 쌍, Bordes 프로토콜 (`analysis/intphys2`), scene bootstrap 95 % CI, run 의 best C 하나로 읽음. 등급: 1 차 (본문에 2 차 정정 반영).

| | n | release w48 (C24) | Ariel43 w32 (C8) | 복사 w48 (C42) | release − 복사 |
|---|---:|---|---|---|---|
| overall | 506 | 54.3 [50.4, 58.1] | 53.8 | 55.5 [51.8, 59.1] | −1.2 [−5.5, +3.0] |
| permanence · Fixed | 52 | 59.6 [44.2, 75.0] | 71.2 [59.6, 80.8] | 63.5 [51.9, 75.0] | −3.8 [−17.3, +9.6] |
| permanence · Moving | 68 | **45.6 [35.3, 54.4]** | 39.7 [29.4, 51.5] | 51.5 | — |
| permanence · all | 120 | 51.7 | 53.3 | 56.7 | −5.0 [−13.3, +3.3] |

(출처: `auto_research/Archive/INTPHYS2_PERMANENCE_AUDIT_2026-09-26.md` §3)

- 결론 (문서): release 의 permanence 는 Fixed · Moving · 전체 모두 CI 가 50 포함. 복사 Fixed 63.5 CI 하한 > 50 → "가장 쉬운 조건" 의 신호는 예측 없이 나온다; permanence 점수는 사건 순간 예측이 아니라 재등장 뒤 장면 vs 마지막 관측의 전역 불일치.
- 같은 C = 24 로 맞추면 release − 복사 +1.8 [−1.8, +5.5] (전체), permanence +2.5 [−4.2, +10.0] — 구분 안 됨 (`VERIFY_ISCENE2` §8).
- ⚠️ 2 차 정정: 원문 "복사가 같거나 조금 낫다" 는 best-C 선택 산물 → "구분 안 됨". n 52–94 (CI ±15). Ariel Fixed 71.2 는 Moving 39.7 과 짝 · 창 민감 (w16 59.6) · max 집계에서 사라짐. 하네스는 IntPhys 2 dtype 수정 미반영 상태 (CLAUDE.md §2-7). IntPhys 2 메타데이터에 사건 프레임이 없어 "문맥 경계" 가설 (H2) 은 이 하네스로 검정 불가 → v11 이 그 자리.
- IntPhys 2 copy-hard 에서 release 21.2 [16.0, 26.6], permanence copy-hard 61 쌍 19.7 (`auto_research/Archive/COPY_HARD_SUBSET_2026-09-29.md`) — 02 장 참조.

### 3-4. IntPhys 1 — 움직이는 물체의 가림은 0.2 초 (corner case 방어)

IntPhys 1 dev 180 쌍, mask 정답, 공식 88.89 실행 (`skip2_w32`) `per_window.json` 과 대조. 등급: 정본 (같은 날 두 번 정정 후 현행).

| | 이동 + 가림 | 정지 + 가림 | 이동 + 보임 | 정지 + 보임 |
|---|---|---|---|---|
| 쌍 수 | 60 | 30 | 60 | 30 |
| 사건 가림 길이 | **2–6 프레임, 중앙 3 = 0.2 초** | 27–89, 중앙 73 = **4.9 초** | 없음 | 없음 |
| skip-2 로 가려진 장수 | 1 (28) · 2 (14) · 3 (3) | 13–44, 중앙 36 | — | — |
| 가려진 동안 이동 | 중앙 **4.6 물체 폭** | 0.14 폭 | — | — |
| skip2_w32 창 중 context 끝 가림 비율 | 12.9 % | **89.6 %** | 0.2 % | 0 % |
| V-JEPA 2 정답 (skip2_w32) | **75.0** | 100 | 91.7 | 100 |

(출처: `z_research/OcclusionStats/Archive/INTPHYS1_OCCLUSION_2026-09-26.md` §2-1)

- 이동+가림 쌍 세부 (§2-2): 가림 안 갈림 81.8 (33) / 끝 직후 66.7 (12) · 사건 전 미관측 15 쌍 66.7 (판별은 '나타남'). ⚠️ 3 장 칸 n=3, 쌍 비독립 — **가림 길이 효과로 읽지 않는다.**
- 정지+가림은 context 끝 가림 창이 90 % 인데도 100 % — "그 자리에 있다" 만 알면 된다.
- 셀 분해 (Garrido sliding 88.89, ViT-H): 정지+눈앞 100 · 정지+가림 100 · 이동+눈앞 91.67 · **이동+가림 75.0** (O1 영속 65 · O2 모양 95 · O3 연속 65; 셀당 n 10–20) — 정지 두 셀이 천장, 이동+가림 한 셀만 크게 떨어진다 (`z_research/IntPhysGenV11/Archive/TALK_2026-08-30.md` §1).
- 사건이 예측 구간 안인 창만 쓰면 (V1, 분석용) 이동+가림 75.0 → 80.0 (`z_research/Benchmarks/Archive/INTPHYS1_EVENT_POSITION_2026-09-25.md` §2).
- 문서 결론: "IntPhys 1 의 88.89 는 움직이는 물체를 오래 가려도 상태를 이어 가는가를 재지 않는다" (state 프레임 표현 — 현재 금지; 뜻은 "이동 물체의 긴 가림을 묻지 않는다"). IntPhys 2 Fig. 2 의 5 초 봉우리는 정지 물체.
- ⚠️ "IntPhys 2 는 움직이는 물체를 오래 가린다" — **안 쟀다** (추적기 필요, `OcclusionStats` 미결).

### 3-5. corner case · use case 판정 (하네스 9)

| 질문 | 답 | 출처 |
|---|---|---|
| 문맥 끝 가림은 corner case 인가 | 09-03 외부 비판 "가림이 사건 순간에 걸리는 건 corner case" → 가림은 **testbed** 로. 같은 성질 (마지막 관측 너머를 못 이음) 이 가림 없는 등속 (reach 2–3 칸, 04 장) 에서도 성립 | `z_research/IntPhysGenV11/Archive/STATUS_AND_CRITIQUE_2026-09-03.md` §2 · `z_research/IntPhysGenV11/Archive/PAPER_STORY_2026-09-22.md` §3 |
| 벤치마크는 이 상황을 내나 | 거의 안 낸다 — IntPhys 1 이동 물체 가림 0.2 초 (skip-2 1–3 장) | §3-4 |
| 실영상에서 닿나 | K400 팬 가림 띠에서 같은 칸 비가림 대비 약 1/3 (§3-2). 실영상 움직이는 내용의 SSv2 46 % · EK100 51 % 가 reach 밖 (`I_SCENE` §4-3, 1 차) | 04 장 |
| 어느 use case 에 닿나 | 벤치마크 갭 (leakage — 02 장) · 평가 방법론 (A/B 병기, copy-hard) · planning (안 쟀다) | PAPER_STORY §4-3 |

출처 (§3): `z_research/v11_realistic/README.md` · `z_research/v11_realistic/exp_results/tables.md` · `z_research/v11_realistic/figures/token_surprise/README.md` · `z_research/v11_realistic/figures/token_surprise/background_checks.md` · `auto_research/Archive/I_SCENE_CAUSAL_BOTTLENECK_2026-09-25.md` · `auto_research/Archive/VERIFY_ISCENE2_2026-09-26.md` · `auto_research/Archive/INTPHYS2_PERMANENCE_AUDIT_2026-09-26.md` · `z_research/OcclusionStats/Archive/INTPHYS1_OCCLUSION_2026-09-26.md` · `z_research/Benchmarks/Archive/INTPHYS1_EVENT_POSITION_2026-09-25.md` · `z_research/IntPhysGenV11/Archive/TALK_2026-08-30.md` · `z_research/IntPhysGenV11/Archive/STATUS_AND_CRITIQUE_2026-09-03.md` · `auto_research/paper/PAPER_STORY_2026-10-08.md`

---

## 4. p 의 미래에 물체가 있나 — 자 판독 세대별

채점은 scalar 하나라 "p 가 가려진 물체를 미래에 두는가" 를 직접 말하지 않는다. 그래서 학습된 자 (readout) 로 p 를 읽었고, 자는 세대마다 바뀌었다. **자 검증 수치를 결과와 같은 무게로** 적는다.

### 4-0. 자 세대표

| 자 (지문) | 학습셋 | 자 검증 | v11 가림에 쓸 수 있나 | 상태 | 출처 |
|---|---|---|---|---|---|
| v11 선형 회귀 (공간평균) | v11_full 자체 (조건별 self) | p 위치 R² 0.96–0.99 · MAE 20–50 cm | ❌ "슬롯 번호 + 방향" 지름길 배제 불가 (문서 스스로 적음) | 기록 — RollOutV2 와 충돌 (§4-2) | `z_research/v11_roll_out/README.md` |
| **RollOutV2 위치 자 v5** (attentive, 3,842 파라미터) | `RollOut_v2_training_v5` 무중력 등속 8,064 clip | held-out **0.87 칸** (학습 0.83) · encoder 자 v2 전 시나리오 ≤ 0.61 칸 · v11 에서 z 는 가림 타이밍 무관 추적 · 자 없는 검사와 슬롯 단위 일치 | ✅ 위치 + attention 질량 (있음 판정은 질량 문턱으로만) | 정본 | `z_research/RollOutV2/figures/v5/summary/POSITION_READOUT_2026-09-12.md` |
| v11 k=0 p 로 학습한 자 | v11 k=0 p 슬롯 0–2 (336 clip) | held-out 3–5 px 인데 물체 칸 질량 0.10–0.16 → **자리 prior 외움**; early 슬롯 0–2 에서 47–63 px | ❌ | 기각 (재시도 금지) | 〃 §2-3 E · CLAUDE.md §6 |
| presence 자 (`dcd24d8a8a47`) | `training_v8` 14,360 clip | v11 판 빈 장면 p '있음' **7–45 %** (가림막을 물체로 읽음) | ❌ static 전부 · moving mid 판정 불가 | 물러남 (09-25) | `z_research/RollOutV3/audit/v11_presence/AUDIT.md` · `audit/training_v8/AUDIT.md` |
| 옛 identity 자 (`aeda55874c1c`) | `training_v8` | v11 판 빈 장면 '있음' **56–99 %** | ❌ | 물러남 (09-26) | `z_research/RollOutV3/Archive/IDENTITY_R8_2026-09-26.md` §1 |
| **identity_r8** (`01c3afaac37f`, 위치 + 57-way) | `RollOut_v2_training_v8` 8,640 clip (v11 판 arm · 빈 짝 1,920, 감사 3 차 통과) | test 조합 p 84.0 / z 90.5 / h 90.9 % · '있음' 오탐 p 0.27 % · 위치 p 13.0 px · **v11 판 빈 장면 '있음' 0 %** · 다시 보인 칸 h '있음' 97–100 % | ✅ (반쯤 가린 칸 · 출구 칸은 정의 밖) | 정본 | 〃 §3 · `z_research/RollOutV3/README.md` |
| identity_ar40 | 같은 학습셋, Ariel AR ep40 p 로 학습 | 조합 91.1 · 위치 중심 1.16 칸 (릴리즈 자 0.85) · 빈 장면 오탐 AR p 2.5 % | ◐ (AR moving 0–8 % 는 오탐과 구분 안 됨) | 1 차 + 적대 검증 | `auto_research/Archive/AR40_READOUT_2026-09-30.md` · `VERIFY_AR40RD_2026-09-30.md` |

### 4-1. RollOutV2 위치 자 — 물체다운 토큰이 남는 길이 late 0 < early/mid 3 < 없음 4–5

(데이터: `v11_vanish_all` pos_a 2,688 clip, 셀당 n 56; 1 칸 = 18 px = 물체 반폭; 균등 질량 0.035)

진실 물체 3×3 attention 질량 (슬롯 0 → 7):

| 조건 | s0 | s1 | s2 | s3 | s4 | s5 | s6 | s7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| flat k=0 | 0.74 | 0.65 | 0.73 | 0.68 | 0.53 | 0.34 | 0.22 | 0.06 |
| flat early k=4 | 0.59 | 0.61 | 0.51 | **0.24** | … | | | |
| flat mid k=4 | (0.5–0.6) | | | **0.30** | … | | | |
| **flat late k=4** | **0.03** | 0.10 | 0.15 | 0.12 | 0.16 | 0.13 | 0.08 | 0.04 |
| ramp k=0 | 0.74 | 0.65 | 0.56 | 0.20 | 0.05 | … | | |
| **ramp late k=4** | 0.01–0.05 (전 슬롯) | | | | | | | |

(출처: `z_research/RollOutV2/figures/v5/v11_vanish/timing/attn_diag.md` · POSITION_READOUT §5-4)

자 없는 검사 (파라미터 0) — 진실 물체 3×3 에서 r = |p − h_imp| / |p − h_pos| > 1 (p 가 그 자리에 물체를 둠) 인 비율:

| 조건 | s0 | s1 | s2 | s3 | s4 | s5 | 대조 r_bg |
|---|---:|---:|---:|---:|---:|---:|---|
| flat k=0 | 1.00 | 1.00 | 1.00 | 0.93 | 0.79 | 0.05 | 0.97–1.05 |
| flat early k=4 | 1.00 | 1.00 | 1.00 | 0.59 | 0.36 | 0.00 | 〃 |
| flat late k=1 | 0.80 | 0.79 | 0.38 | 0.21 | 0.09 | 0 | 〃 |
| **flat late k=4** | 0.54 | 0.39 | 0.09 | 0.05 | 0.11 | 0 | 〃 |
| **ramp late k=4** | 0.52 | 0.29 | 0.00 | … | | | 〃 |

(출처: `z_research/RollOutV2/figures/v5/v11_vanish/timing/token_test.md`)

- 출구 판정 (emergence, gate 0.15): p 가 출구 모서리 밖으로 나오는 비율 flat late k=4 t8–t15 0.00 0.14 0.05 0.00 … · ramp late k=3·4 ≈ 0.00 · flat mid/early ≈ 0.93–1.00. z 는 모든 조건에서 진실을 1 튜블릿 늦게 따라 나온다 (`.../v11_vanish/emergence/emergence.md`). ⚠️ gate 민감 (flat late k=1 마지막 튜블릿 out 0.50 / 0.45 / **0.07** at gate 0.1 / 0.15 / 0.25).
- 문서의 문장: "가림이 경계에 걸리면 **p 의 미래에는 처음부터 물체다운 토큰이 없다**." ramp late 는 k 에 따라 악화 (슬롯 7 진행 비율 0.21 → 0.04), static 도 late 만 1 칸 넘게 이탈 (L2 20–24 vs early·mid 12–15 · k=0 9.6 px; z 7–10).
- **철회 (09-12)**: "v11 가림에서 p 는 물체를 마지막 관측 자리에 둔 채 머문다" — 마지막 관측 칸 질량 0.01–0.03 (슬롯 ≥ 2), 읽기는 자 기본값 ± 19–36 px, v11 기본값 ≈ 마지막 관측 위치 ≈ 화면 중심. 맞는 문장은 위.
- ⚠️ v11 속도 18 px / 튜블릿은 학습셋 상한 12 밖. "슬롯 7 비율" 한 숫자는 물체가 사라진 뒤의 기본값을 섞어 early/mid 와 late 의 차이를 지운다 — 슬롯별 질량과 같이만.

### 4-2. 모순 — v11_roll_out (09-07) "late 에서도 위치가 있다"

| | v11_roll_out (09-07) | RollOutV2 (09-12, 정본) |
|---|---|---|
| 자 | 공간평균 → 선형 회귀, **v11 자체로 조건별 학습** | 등속 학습셋 attentive 자, v11 은 test 만 |
| 결과 | late k=4 가려진 슬롯 0·1 오차 flat 50/38 · ramp 59/49 cm (보이는 슬롯과 같은 급) → "가림이 경계에 걸려도 거의 안 변한다" | late 슬롯 0 부터 질량 0.03 · 자 없는 검사 r>1 0.54 → 0.09 |
| 해석 | 문서 스스로 "슬롯 번호 + 방향" 지름길을 못 막는다고 적음 (v11 은 조건마다 궤적 하나) | v11 로 학습한 자는 자리 prior 를 외운다 (§4-0 기각 행) |
| 판정 | 지름길 미배제 → 현 정본은 RollOutV2 | — |

(출처: `z_research/v11_roll_out/README.md` · POSITION_READOUT §2-3 E · §5-4)

### 4-3. presence 자 — 감사 결과와 물러난 이유

| 감사 항목 | 결과 | 출처 |
|---|---|---|
| last 움직이는 물체, 깨끗한 칸 (flat t3, ramp t3–t5) | p '있다' 0.23 / 0.28 / 0.07 / 0.24 (CI 상한 ≤ 0.39–0.47) vs h 0.86–1.00 · visible p 0.83–0.96 · 빈 장면 p 0.00 → holds_weaker | `z_research/RollOutV3/audit/v11_presence/AUDIT.md` |
| 자 없는 vanish 쌍 A | p 가 물체 미래 쪽인 block flat 1.8 / 1.8 / 0 %, ramp 0 / 0 / 0 % (visible 100 %) | 〃 |
| 문턱 아래 신호 | AUROC 0.86–0.96 → **문턱 아래 물체 신호** 있음 | 〃 |
| static | 판정 불가 — trapdoor 빈 장면 오탐 0.23–0.92 (AUROC 0.67–0.89); 복사 z_ctx 0.96 / h_ctx 0.94 도 '있다' | 〃 |
| 복사 기준선 (문맥 16 샘플만 encoder 재실행) | h_ctx '있다' flat 0.20 (빈 0.14) · ramp 0.51 (빈 0.00) → "복사는 '없다' 가 아니다"; "p = 복사" 도 "p > 복사" 도 성립 안 함 | 〃 |
| 물러난 원인 | v11 가림막은 학습셋에 없는 물건 (재질 `MI_OccDark` vs `M_Wood_Pine`, 깊이 1,190 cm 물체 앞 vs 1,530–1,563 cm 물체 뒤, 누운·움직이는 판 0 장). 빈 장면 오탐이 실제 프레임을 본 h 에서도 뜸 (static late h 2–95 %) → 자가 처음 보는 판에 내는 반응 | `z_research/RollOutV3/audit/training_v8/AUDIT.md` §E |
| 사용자 결정 (09-25) | 자를 위치 + 정체 (57 분류) 로 교체 · **가림 음성은 자 학습셋에 넣지 않는다** (가림 판정은 자 없는 vanish 짝 비교로) | `z_research/RollOutV3/_superseded/README_2026-09-25.md` §4-2 |

- 감사 뒤에도 서는 것: **moving last 영속 ❌** (p 7–28 % vs h 86–100 %; 빈 장면 p 0 %, 자 없는 vanish 98–100 % 와 일치).
- 물러난 문장: "멈춘 물체 (static) 는 가려져도 대부분 남는다" (판정 불가) · "p 는 t8 근처 절벽처럼 물체를 놓는다" (presence 자의 성질 — identity_r8 에서 절벽 없음) — §8.

### 4-4. identity_r8 — k 별 '있음', 모양이 가른다, 짧은 가림

v11 문맥 끝 가림, **다시 보인 튜블릿 t3–t6** 에서 p '있음' (%) — 같은 조건 빈 장면 p · h 0 %:

| | k=0 | k=1 | k=2 | k=3 | k=4 | 문맥 중간 가림 |
|---|---:|---:|---:|---:|---:|---:|
| 정지 | 100 | 28 | 18 | 16 | 17 | 100 |
| flat | 81 | 62 | 49 | 40 | 37 | 75–79 |
| ramp | 100 | 60 | 48 | 34 | 25 | 75–92 |

(출처: `z_research/RollOutV3/Archive/IDENTITY_R8_2026-09-26.md` §5-1) — "가려졌었나" 보다 "문맥 **끝** 에 보였나" 가 가른다. 문맥 중간 가림은 k 무관.

| 세부 | 값 | 출처 |
|---|---|---|
| ramp k=2 짧은 가림 ('있음' 칸) | t1 '있음' **69 %**, 그 중 55 % 조합까지 맞음 (우연 1.8 %), 진실에서 24 px; t3 이후 마지막 본 자리에서 67–70 px (≈ 3.8 칸) 에 머묾 | §5-2 |
| 정체 유지하며 '있음' (분모 물체 clip 전부, t3–t6) | 문맥 끝 가림 정지 / flat / ramp '있음' 20 / 47 / 42, **둘 다 유지 4 / 14 / 10 %** (가림 없음 77 / 39 / 38, 중간 가림 69 / 31 / 23) | §5-4 |
| 모양이 가른다 (t1–t6 '있음', 괄호 모양 맞음) | flat 끝 가림 sphere 93 (93) · torus 90 (61) · 나머지 20–35; ramp sphere 87 (96) · torus 74 (47) · 나머지 21–41; 정지 sphere 52 · torus 48 · 나머지 1–8 | §5-4 |
| 모양 범위 vs 다른 축 | 모양 67–71 %p ≫ 색 20–24 · 배경 12–27 %p | §5-4 |
| 자 탓 아님 | 학습셋 p '있음' recall 모양마다 99.1–99.8 %; h 모든 모양 99–100 % | §5-4 |
| k 경향 | 이동 '있음' k 따라 꾸준히 줄음 (flat 62 → 37, ramp 60 → 25); sphere·torus 색 유지 k1→4 −10.6 %p [−15.2, −5.9], 모양 유지 −1.9 %p [−5.8, +1.4] | §5-4b |

- 결론 (문서): 네 능력 영속 **대부분 ❌** — 짧게 가려진 이동 물체만 1–2 튜블릿. 가려진 물체가 남는지는 모양에 달렸다 (p 의 성질).
- ⚠️ **sphere · torus 가 남는 이유는 모른다** (회전 불변 외형은 추측). CLAUDE.md §5-3 채점 순서 (sphere 36 · torus 3 최저) 와 **거의 반대** — 재는 양이 달라 잇는 주장 안 함.
- ⚠️ **t7 은 표에서 뺐다** — 이동 물체 화면 나가는 중 (h '있음' 11–31 %) 인데 p '있음' flat 78–95 · ramp 40–100 %; 학습셋에서도 p 오탐 t7 최고 → 예측 창 마지막 자리 성질일 수 있음.
- ⚠️ 자는 보이는 물체만 읽는다 — 정지 '없음' 은 "판을 세운 채 두어 가렸다" 일 수도 (가르는 검사 미실행). 학습 판은 물체 뒤, v11 판은 물체 앞. 정체 정확도는 옛 자보다 낮다 (v11 가림 없음 정지 p 조합 56–81 vs 옛 92–97).
- ⚠️ "모양은 k 와 무관" 을 전체로 넓히지 말 것 (구성 변화 섞임: capsule 37 → 10, cone 31 → 12, pyramid 71 → 97 칸 적음). v11 이동 팔 궤적 둘 (l2r · r2l), 거리 크기 방향에 따라 2 배.

### 4-5. AR (Ariel ep40) 를 같은 꼴의 자로 읽으면

| v11 late, 이동 '있음' | k1 | k2 | k4 | 출처 |
|---|---|---|---|---|
| AR40 moving_flat | 10 → 1 % | 0 % | 0 % | `AR40_READOUT_2026-09-30.md` §2 |
| AR40 moving_ramp | 3 → 0 % | 0 % | 2 → 7 % | 〃 |
| 릴리즈 moving_flat (identity_r8) | 56 → 93 % | 33 → 87 % | 6 → 78 % (자리 77–88 px 틀림) | 〃 |
| 릴리즈 moving_ramp (identity_r8) | 87 → 78 % | 41 → 68 % | 8 → 40 % (자리 87–100 px 틀림) | 〃 |
| 빈 장면 오탐 | AR p 2.5 % · 릴리즈 0.0 · h 0.0 | | | `VERIFY_AR40RD_2026-09-30.md` §4 |

- 적대 검증 정정: AR moving k1–4 '있음' 0–10 % 는 빈 장면 오탐 2.5 % 와 **"0 과 구분 안 됨"**; 릴리즈 "있지만 자리 틀림" 77–100 px 확정 (마지막 본 자리에서 53–86 px = 출구 prior 정합); static k1–2 t5–7 30–41 % 는 실재; AR 의 드문 '있음' (flat k1 4 %) 은 마지막 본 자리 20 px 안 — 이어 간 것이 아니라 마지막 자리에 남긴 것.
- ⚠️ 위 표의 "a → b" 는 `AR40_READOUT` §2 표 머리대로 **튜블릿 t0 → t7 의 '있음' %** 이고 (자리가 틀린 칸 = 출구 prior 포함), identity_r8 §4-4 표 (flat k1 62 · k2 49 · k4 37) 는 **t3–t6 평균** 이다 — 구간이 달라 두 표를 섞어 읽지 않는다. (검증 10-10: 앞 판의 릴리즈 행은 flat 값만 적고 "flat / ramp" 로 묶었었다 → 두 행으로 나눔, `AR40_READOUT_2026-09-30.md` §2.)

### 4-6. 채점과 자는 다른 축이다

| 측정 A | 측정 B | 어긋남 | 출처 |
|---|---|---|---|
| 채점 vanish early **97.8** / mid **95.8** / late **57.5** | 위치 자: 세 타이밍 모두 3 슬롯 안에 물체다운 토큰 소실 | 채점은 타이밍을 가르는데 위치는 셋 다 낮다 → **별개 축** | POSITION_READOUT §5-4 · 결론 6 |
| 채점 k=1–4 평평 (shape sens 5.0 / 5.2) | identity_r8 '있음' · 색 유지가 k 따라 계속 줄음 (flat 62 → 37) | "채점이 k 에 평평 ≠ p 미래가 k 에 평평" | IDENTITY_R8 §5-4b |
| 채점 shape 순서 sphere 36 · torus 3 (최저) | identity_r8 가려져도 sphere · torus 가 가장 잘 남음 (74–93 %) | 거의 반대 — 재는 양이 다름 | IDENTITY_R8 §5-4 |
| 채점 ramp late vanish A 0 % (Ariel AR 은 late vanish→빈 72–99 %, §6) | AR40 자: late 뒤 이동 물체 토큰 0 과 구분 안 됨 | AR 채점 정답은 물체 토큰이 아니라 전역 오프셋 (§6-2) | `AR40_READOUT` §결론 3 · `VERIFY_PRESENCE` |

- ⚠️ 채점 (토큰 평균 latent L1) 은 시간축 · 공간축을 모두 버린다 — "어디에 놓았나" 를 원리적으로 못 본다 (CLAUDE.md §10 축별 풀링 메모, `auto_research/Archive/COPY_VS_POSITION_2026-10-07.md`). v11_realistic 지도에서도 쌍 Δ 의 합은 배경 토큰 (물체 토큰 3–6 %) 이 정한다 (§3-1).

출처 (§4): `z_research/RollOutV2/figures/v5/summary/POSITION_READOUT_2026-09-12.md` · `z_research/RollOutV2/figures/v5/v11_vanish/timing/attn_diag.md` · `z_research/RollOutV2/figures/v5/v11_vanish/timing/token_test.md` · `z_research/RollOutV2/figures/v5/v11_vanish/emergence/emergence.md` · `z_research/v11_roll_out/README.md` · `z_research/RollOutV3/README.md` · `z_research/RollOutV3/Archive/IDENTITY_R8_2026-09-26.md` · `z_research/RollOutV3/audit/v11_presence/AUDIT.md` · `z_research/RollOutV3/audit/training_v8/AUDIT.md` · `z_research/RollOutV3/_superseded/README_2026-09-25.md` · `auto_research/Archive/AR40_READOUT_2026-09-30.md` · `auto_research/Archive/VERIFY_AR40RD_2026-09-30.md`

---

## 5. 정체 · 정보는 남는다 (→ 07 장)

"p 에 정보가 없다" 는 철회된 문장이다. 정보는 encoder 와 p 에 있고, 깨지는 것은 **비가림에서 정한 자리 (좌표) 가 late 가림에서 안 통하는 것** 이다.

| 측정 | 값 | 무엇을 말하나 | 등급 | 출처 |
|---|---|---|---|---|
| p self probe, 등가속+가림 | shape **98.46** / color **99.49** / env **100** · 같은 조건 채점 sens shape **4.46** · vanish 0.00 | 정보는 p 에 있다, 채점이 못 쓴다 | 확정 (10-08 재확인 `per_group`) | `RESULTS_2026-08-30.md` §7 · §11-(1) |
| env 대조군 | self · 이식 · k 분해 72 칸 전부 100.0 | 출력 전체가 망가진 것 아님 | 〃 | 〃 · CLAUDE.md §5-2 |
| 비가림 → 가림 이식 (팔 안, 3 팔 평균) | color z 97.4 / h 99.2 / **p 17.3** · shape z 64.8 / h 75.7 / p 27.3 | 같은 팔 가림에서 **p 만** 비가림 head 가 안 통함 | 〃 | `RESULTS_2026-08-30.md` §8 |
| 이식 × k (등가속 팔 p) | shape 45.0 / 40.9 / 22.2 / 15.5 · color 고유 클래스 7 → 5 → 1 → 1 (전부 orange) | 실패는 k 에 비례, 끝은 단일 클래스 | 〃 | 〃 §10 |
| timing 이식 (`attn_probe_xfer`, p colour) | early 98.6 / 99.9 / 100.0 · mid 98.2 / 99.4 / 98.9 · **late 15.1 / 20.6 / 16.4** (static / flat / ramp); z · h late 93–100 | **late 에서만** p 의 자리가 바뀐다 (k=0→1 한 칸) | 정본 README | timing README §probing |
| p self (timing) | shape vis/early/mid 100, late 95.2–98.9 · colour late 86.3–99.2 · env 36 칸 100 | 정보는 어디서도 안 없어진다 | 〃 | 〃 |
| h → p (비가림에서도) | colour vis 94.9 / 79.0 / 57.3 | p 와 h 는 비가림에서도 같은 좌표계가 아니다 | 〃 | 〃 |
| 경계 토큰 B7 (z, shape) visible → late | static **20.2** [17, 24] / flat **16.0** [13, 19] / ramp 32.9 (chance 14.3) · 9 토큰 concat 20.6 / 19.5 / 28.1 | 가려진 경계 토큰은 "보이던 때의 코드" 가 아니다 | 정본 (E2) | `z_research/context_encoder_analysis/exp_results/ctxenc_boundary_identity_v2/RESULTS.md` (i)(iii) |
| 같은 셀 self late | B7 87.2 / 78.9 / 74.7 · **C (문맥 2048 토큰) 98.2 / 99.3 / 99.3** | 정보는 문맥 전체에 있다 (다른 코드) | 〃 | 〃 |
| late → early 역이식 (B7) | static 35.8 / flat 40.9 / ramp 72.8 (CI 하한 > chance) | 가려진 자리 코드는 부분적 정체성 코드 | 〃 | 〃 (iv) |
| v11_identity 감사 — 복사 기준선 | p_fut − z_t7 shape +2.1 [+0.9, +3.4] / +0.2 (n.s.) · color +3.0 / +2.6; p_fut − z_all shape −1.6 / −3.5; [z_all, p_fut] − z_all ≈ 0 | **"p 가 정체를 지킨다" 는 문맥 복사 수준 이상의 증거가 아니다** | 정본 (감사) | `z_research/RollOutV3/audit/v11_identity/AUDIT.md` §3 |
| 과거만 본 z 위 자 B (12,802 파라미터) — v11 late 미래 위치 | z-B flat k1–4 7.4 / 6.9 / 6.3 / 10.1 · ramp 5.9 / 5.4 / 5.6 / 6.9 px (k0 5.5 / 5.3; h-B 상한 4.4–8.2) | 가림 문맥에서도 미래 위치 정보는 입력에 있다 | 1 차 | `auto_research/Archive/FUTURE_FROM_CONTEXT_2026-10-02.md` |
| 같은 자를 p 에 | p-B flat 16.9 / 19.8 / 22.2 / **29.9** · ramp 19.2 / 19.1 / 20.1 / 19.1 px; 가려진 미래 슬롯만 z-B 5–15 · p-B 11–27 · COPY 36–64 | 입력에 있는 것이 출력에서 사라지는 조건 = 문맥 끝 가림 | 1 차 | 〃 |
| M2 경계 패칭 (v11 late ↔ early) | 움직이는 물체의 **있음** 신호는 경계 1–2 튜블릿을 따라 넘어감 (연속 contrast 회복 1.10 [1.08, 1.12] · 파괴 1.19); **자리** 회복은 0.15 [0.09, 0.21]; 역사만 다른 대비 자리 −0.231; 정지 물체 있음은 역사에도 기댐 (파괴 0.53–0.71) | p 의 "있음" 은 경계 토큰에 묶인다 | 적대 검증 정정 반영 | `auto_research/Archive/M2_BOUNDARY_PATCH_2026-09-25.md` |

- ⚠️ shape 은 encoder 도 흔들린다 (z 3 팔 평균 64.8, 등속 팔 42.4) — "encoder 는 가림과 무관" 은 **color · env** 로만 (CLAUDE.md §5-2).
- ⚠️ 팔을 가로지르는 이식 실패는 증거가 아니다 (기대값) — 위 이식은 전부 **팔 안** (같은 운동, 비가림 → 가림).
- ⚠️ v11_identity 감사: "가림 조건에서 사상이 한 번 더 바뀐다 / late 의 p 는 또 다른 자리" (PROBE.md §5-3) **철회** — encoder 쌍 (h_fut → z 65.0, z → h 42.5) 도 late 에서 같이 무너진다. 서는 것은 "p < h_fut" (모든 운동). 외형 쌍둥이를 빼면 z_hid color 83.7 / 82.7 (−12.4 pt).
- ⚠️ FUTURE_FROM_CONTEXT: "encoder 가 미래를 표현한다" **금지** (이 데이터에서 미래는 현재 상태의 결정적 함수, 자는 지도 라벨의 곡선을 붙인다). 쓸 문장: "미래를 만드는 데 필요한 정보는 입력에 선형으로 있고, predictor 는 그것을 미래 자리로 옮기지 않는다." fit_vis 이식은 z · p · h 셋 다 깨져 (h 도 17–49) 증거 아님.
- ⚠️ M2: 짝은 "가림 타이밍만 다른 쌍둥이" 가 아니다 (sym_k 41 % 다름, 등가속 짝 가림막 높이 다름, E clip 최대 11 회 재사용) — 경계 가림은 corner case, 닿는 곳은 VoE 평가 방법론.

출처 (§5): `z_research/IntPhysGenV11/Archive/surprising_score/RESULTS_2026-08-30.md` · `z_research/IntPhysGenV11_occlusion_timing_ablation/README.md` · `z_research/context_encoder_analysis/exp_results/ctxenc_boundary_identity_v2/RESULTS.md` · `z_research/RollOutV3/audit/v11_identity/AUDIT.md` · `auto_research/Archive/FUTURE_FROM_CONTEXT_2026-10-02.md` · `auto_research/Archive/M2_BOUNDARY_PATCH_2026-09-25.md`

---

## 6. "presence-without-placement" 와 다시 학습해도 (→ 08 장)

### 6-1. 복사가 틀리는 late 쌍 — AR 만 맞힌다 (copy-hard)

복사 = 문맥 마지막 튜블릿 LN(z) 를 미래 전 슬롯에 (H6 관례). 판: `v11_split_test` 10,752 쌍. cluster bootstrap B 2,000, 원배열 재계산. 등급: 1 차 + `VERIFY_R3` 확정 (읽기 문장 교체).

late vanish→빈 (물체→빈, A) 정답 % (112 쌍씩):

| predictor | static | flat | ramp |
|---|---:|---:|---:|
| 복사 (LN) | 67.9 | 39.3 | 9.8 |
| release | 65.2 | **2.7** | **0.0** |
| Ariel prefix ep45 | 49.1 | 37.5 | 14.3 |
| Ariel full ep40 | 50.9 | 15.2 | 8.9 |
| **Ariel AR ep38** | **99.1** | **85.7** | **72.3** |
| Ariel AR ep18 | 99.11 | 94.64 | 86.61 |
| pv1_e15 (post-FT) | 52.7 | 88.4 | 77.7 |
| v11_e10 (v11 post-FT, 도메인 안) | 86.6 | 100 | 100 |

(출처: `auto_research/Archive/COPY_HARD_SUBSET_2026-09-29.md` §2-1 · `auto_research/Archive/VERIFY_R3_2026-09-29.md` C)

- copy-hard 전체 (2,877 쌍): release **48.4** [46.6, 50.2] · prefix 51.0 · full 49.6 · **AR ep38 73.8** [72.2, 75.4] · v11_e10 82.5. copy-hard 안 late: release 38.3 · prefix 37.8 · AR 65.8.
- ⚠️ **`VERIFY_R3` C3 정정**: 무작위 겹침 귀무에서 copy-hard 정확도 = 전체 정확도 → 모든 predictor 가 그 아래 (release −27 · prefix −28 · full −28 · **AR −10** · v11_e10 −9). AR 73.8 은 "복사가 못 푸는 것을 푼다" 가 **아니라** "AR 의 오류가 복사 오류와 덜 묶여 있다".
- ⚠️ AR 표적 공간이 블록별이라 표준 복사와 덜 상관될 뿐일 수 있다. AR late·static 99.1 은 블록별 마지막 블록에 물체가 없어 AR 공간 복사로도 맞을 수 있다 (이동 칸 72–86 은 그 설명 불가). copy_blk-hard (45 쌍, 전부 late) AR 82.2 는 시사 이상 아님.

### 6-2. AR 이 맞히는 방식 — 자리 없는 있음 (VERIFY_PRESENCE)

v11 split_test late 가림 · vanish 672 쌍, margin M = |p − h_imp| − |p − h_pos| (+ = 가능 쪽), Dh = |h_pos − h_imp| (두 미래가 갈리는 칸). **학습된 자 없음** (파라미터 0). 등급: 적대 검증 결과 (독립 코드 전수 재계산).

| 항목 | AR (ep40) | release | ctx_ar (CX16) | 출처 |
|---|---|---|---|---|
| moving 정답 | .670 [.58, .76] (flat .857 · static 1.000) | **.000** (복사 .054) | — | `VERIFY_PRESENCE` §1 |
| margin @ Dh 상위 5 % 칸 (재등장 자리) | **−.0405** [−.0456, −.0352] | 음수 | — | 〃 |
| margin @ 나머지 칸 | **+.0108** / +.0159 | — | — | 〃 |
| 링 (재등장 핵에서 Chebyshev 거리, moving) | 0: −.057 · 1: −.038 · 2: −.004 · 3: +.011 · 4: +.014 · 5: +.014 · 6+: +.018 | −.021 … −.004 (전부 음수) | 전 링 양수 (+.013 ~ +.037) | 〃 §3 |
| 복사 지도 상관 r(M, Mc) | **.592** [.585, .599] | .258 | — | 〃 §5 |
| empty 쌍 정답 | .994 | .917 | **.214** (편향) | 〃 §1 |
| sym_k 별 정답 | k1 .941 → k2 .917 → k3 .798 → k4 .714 | — | — | 〃 §7 |

- 정확한 문장 (문서 §8 정정 후): AR 의 정답은 **재등장 자리 (반경 ≤ 1 칸) 에서 '사라진' 미래 쪽으로, 3 칸 밖 원거리에서 균일하게 '돌아온' 미래 쪽으로 기우는 두 성분의 합이고 후자가 이긴다**; 릴리즈 · prefix 는 둘 다 음수라 틀린다. 근접 빗맞힘이 아니라 **자리 정보 부재** — 이 아카이브는 이것을 "presence-without-placement" 로 부른다.
- 정합: AR40 자 판독에서도 late 뒤 AR 미래에 이동 물체 토큰은 0 과 구분 안 됨 (§4-5).
- ⚠️ 문서 요약 문장 "AR 은 돌아온다는 것은 알지만 어디로 오는지는 모른다" (내부 귀속 표현 — 현재 금지). ⚠️ "전 장면에 퍼진" → "재등장 반경 2 칸 밖 원거리, 중앙 띠 (행 6–9) 제외". ⚠️ Dh 상위 칸은 encoder 가 두 미래를 가르는 자리이지 GT 물체 자리가 아니다 (상위 13 칸 중 24 % 만 최상위 칸 5×5 안). ⚠️ presence 에 문맥 편향 성분이 섞인다 (시사). ⚠️ AR 은 k 에 따라 정답이 준다 (.94 → .71) — CLAUDE.md §5-1 "k 평평" 은 릴리즈 채점 이야기. ⚠️ 무엇이 돌아오나 (정체) 는 재지 않음.

### 6-3. 다시 학습해도 — 도메인 밖에서는 영속이 안 오른다

v11 split_test held-out block 절반 8,064 clip. 영속 = late 가림 k ≥ 2. 등급: 정본 (적대 검증 9 항목 반영).

| predictor | vanish A 물체→빈 (late k≥2, flat / ramp) | 영속 순 배치 (flat / ramp; 배치 − 빈 장면 귀무) | 해석 | 출처 |
|---|---|---|---|---|
| 복사 기준선 | **26 / 5 %** | — | — | `z_research/TrainingEffects/v11_presence/PRESENCE.md` 표 7 |
| release | 0 / 0 | 0.06 / 0.04 (배치) | 퍼진 attention 의 기본값 읽기 | 〃 |
| v11_e10 (v11 궤적 재생 post-FT) | 100 / 100 | **+0.60 / +0.53** | 도메인 안 (궤적 prior) | 〃 |
| pv1_e15 | 96 / 92 | +0.29 / +0.26 (귀무 0.21 / 0.09) | 문맥 무관 "출구에서 물체가 나온다" prior, epoch 따라 커짐 | 〃 |
| ip2_e80 | 48 / 26 | +0.04 / +0.04 | release 수준 | 〃 |
| **ariel_ep43** | **27 / 5** (복사 수준) | +0.09 / +0.07 | release 수준 | 〃 |
| ip1_e40 | 1 / 0 | +0.03 / +0.00 | release 수준 | 〃 |
| 참조 h@h · z@z | — | +0.45 / +0.39 · +0.38 / +0.35 | 천장 | 〃 |

- ⚠️ v11_e10 · pv1_e15 가 맨 위인 것은 **k ≥ 2 에서만** — k=1 flat 에서 pv1_e15 64 % < 복사 79 · ip2_e80 75 · ariel_ep43 68. late static A 는 pv1_e15 39 · ariel_ep43 43 · release 58 % 로 복사 64 % 아래. 궤적이 둘 (l2r · r2l) 이라 궤적만 cluster 하면 v11_e10 flat Δ CI [−0.01, +0.32] 로 0 을 걸친다. 고정 문턱 판독은 자에 따라 부호가 뒤집힌다 (ip2 p@h −0.24 vs p@p +0.13) — 단독 인용 금지.

| 다른 재학습 근거 | 값 | 출처 |
|---|---|---|
| Ariel full / prefix v11 채점 | late 정지 51.9 / 51.2 (가림 끝 칸은 릴리즈처럼 바닥) | `z_training/ARIEL_CHECKPOINTS.md` |
| Ariel AR ep18 v11 채점 | vanish late 95.8 (릴리즈 56.7) · 물체→빈 98.4 / 빈→물체 99.6 (릴리즈 78.0 / 97.6) — 단 §6-2 의 자리 없는 있음 | 〃 |
| v11 post-FT (`v11_postft` e10) 채점 | late release 52.7–61.2 (flat 57.6–61.2 · ramp 57.1–60.7 · static 52.7–59.8) → flat 82.1–85.7 / ramp 84.4–89.7 / static 84.4–89.3 (k=1–4 같이) · 비가림 · early · mid 도 오름 → **v11 도메인 전체 적응** | `z_research/predictor_training/predictor_IntPhysGenV11_PFT/Archive/SCORING_2026-09-16.md` §2 |
| v11 post-FT 위치 자 | flat late k=4 질량 0.00 0.19 0.58 0.69 0.84 0.97 0.92 0.80 (릴리즈 0.03–0.15) — 재출현 슬롯 2 부터 제자리; 도메인 밖 RollOut 에서 β 1.8–2.2, ramp_a R² −4.5, wall 70 px 통과 | `…/predictor_IntPhysGenV11_PFT/Archive/RESULTS_2026-09-16.md` §1 · §2-1 |
| IntPhys 1 post-FT (IP1-FT) | 정지 물체 late +8.5 ~ +15.2 오르나 moving 은 안 오름; ramp (`moving_occlusion_early` / `_mid`) vanish A **39.3 / 35.7** (−49.1 / −45.5) | SCORING §2 · §3 |
| 실영상 가림 띠 | Ariel · N1 (팬 증강 post-FT) hid .21 — release 와 같음 (§3-2) | `I_SCENE` §4-4 |

- ⚠️ **철회 (2026-09-17, 사용자)**: v11 post-FT "학습 도메인 안에서는 진짜 상태 진화처럼 보인다" — v11 은 속도 1 종, held-out block 도 궤적이 학습셋과 같다 → 학습 분포 재생 (state 프레임 표현 — 현재 금지).
- 합격선 (정본 §2-4) 의 한 줄 "v11 late vanish '물체→빈' > 50 **(v11 로 학습하지 않고)**" 는 이 표에서 나온다 — v11 학습은 100 을 내고 Ariel 은 복사 수준 (27 / 5). 08 · 09 장.

출처 (§6): `auto_research/Archive/COPY_HARD_SUBSET_2026-09-29.md` · `auto_research/Archive/VERIFY_R3_2026-09-29.md` · `auto_research/Archive/VERIFY_PRESENCE_2026-09-29.md` · `auto_research/Archive/AR_PREDICTOR_2026-09-27.md` §2-6 · `z_research/TrainingEffects/v11_presence/PRESENCE.md` · `z_training/ARIEL_CHECKPOINTS.md` · `z_research/predictor_training/predictor_IntPhysGenV11_PFT/Archive/SCORING_2026-09-16.md` · `z_research/predictor_training/predictor_IntPhysGenV11_PFT/Archive/RESULTS_2026-09-16.md` · `z_training/RESULTS_2026-09-19.md`

---

## 7. 문서 간 모순 · 미결 (조건 차이)

| # | A 쪽 | B 쪽 | 조건 차이 | 지금 판정 | 출처 |
|---|---|---|---|---|---|
| 1 | 자 없는 vanish block 비교: 이동 k=2–4 가 98–100 % 빈 미래 쪽 | identity_r8: ramp k=2 t1 '있음' 69 % (조합 55 %, 진실 24 px), flat 34–59 % | 쌍 판정 (토큰 평균 L1) vs 튜블릿별 자 분류 | **미결** — "짧은 가림을 이어 붙인다" 는 조건부 | `IDENTITY_R8_2026-09-26.md` §7-1 |
| 2 | 채점 k=1–4 평평 | identity_r8 '있음' k 따라 감소 · AR 정답 k 따라 감소 (.94 → .71) | 채점 (릴리즈 scalar) vs 자 / AR margin | 둘 다 유효 — 채점 평평 ≠ p 미래 평평 | §4-6 · `VERIFY_PRESENCE` §8-6 |
| 3 | v11_roll_out "late 에서도 위치 있음" | RollOutV2 "late 슬롯 0 부터 없음" | v11 자체 학습 선형 회귀 vs 등속 학습셋 attentive 자 | RollOutV2 정본 (지름길 미배제) | §4-2 |
| 4 | 정본 스토리 "보이는 내용의 절반만 제자리에" | 2 차 검증 "같은 칸 비가림 대비 약 1/3" | 기준 = 같은 clip vis 칸 vs 같은 칸 비가림 판 | 둘 다 수치는 맞음 — 1/3 이 엄격 | §3-2 |
| 5 | 채점 static_occlusion vanish 만 살아남음 (sens +20.54) | 같은 조건 p self probe 가장 낮음 (85.13, 미수렴 head) | 채점 vs probe 학습 | **미해결** | RESULTS §11-(5) |
| 6 | 채점 지킴 순서 sphere 36 · torus 3 (최저) | identity_r8 가려져도 sphere 93 · torus 90 (최고) | 2 지선다 채점 vs 자 '있음' | 잇는 주장 안 함 (재는 양 다름) | IDENTITY_R8 §5-4 |
| 7 | late = 문맥 끝 가림의 효과 | late = 미래 앞 가림의 효과 | 대칭 설계라 분리 팔 없음 | **미결** (문맥 끝만 가리는 팔 필요) | timing README ❌ 줄 |
| 8 | knockout 층 8 −18.06 | −14.4 | `knockout_window__v11_vith` 288 쌍 표본 vs `knockout_w1` v11_full | 같은 실행 안 비교만 (07 장) | `z_research/IntPhysGenV11/Archive/PAPER_STORY_2026-09-06.md` 재현절 |
| 9 | 정지 물체 '없음' (identity_r8, 1–8 %) | "판을 세운 채 가렸다" 일 수 있음 | 자는 보이는 물체만 읽는다 | 가르는 검사 미실행 | IDENTITY_R8 §7-2 |
| 10 | IntPhys 2 이동 물체 가림 길이 | — | 추적기 필요 | **안 쟀다** ("IntPhys 2 는 움직이는 물체를 오래 가린다" 금지) | `z_research/OcclusionStats/README.md` |

---

## 8. 이 장 주제의 철회 · 정정 (요약 → 10 장)

| 옛 문장 | 왜 물러났나 | 날짜 | 출처 |
|---|---|---|---|
| "predictor 가 (v11 가림에서) 물체를 마지막 관측 자리에 둔 채 머문다" | 자의 기본값 (토큰 평균 읽기 ≈ 화면 중심 ≈ 마지막 관측). 마지막 관측 칸 질량 0.01–0.03 → "처음부터 물체다운 토큰이 없다" | 09-12 | POSITION_READOUT · CLAUDE.md §6 |
| "가림막이 색면이라 색 채점을 죽인다" | v13_black 에서 colour visible→early −14.1 → −13.5 같음. 대신 모양 −6.2 → 0.0 | 09-03 | timing README |
| "가림 길이가 원인 (레버)" (blender v4 64 → 53.9) | 세 변수 얽힘 · v10_occ_low 4 → 1 무관 · v11 k 평평 · timing 경계 문제 | 08-28 ~ 09-03 | `OCCLUSION_LENGTH_2026-08-20.md` · V10 §1a |
| "가림 하에서 가속 무관" | 둘 다 바닥 (54.46 / 56.85) 이라 잴 수 없음 | 08-30 | RESULTS §3-(2) |
| "멈춘 물체 (static) 는 가려져도 대부분 남는다" | presence 자에서 판정 불가 (trapdoor 빈 장면 '있다' 0.23–0.92) | 09-25 | `RollOutV3/audit/v11_presence/AUDIT.md` (b) |
| "p 는 t8 근처 절벽처럼 물체를 놓고, 자리는 튜블릿 수가 정하며, 깜빡인다" | presence 자의 성질 — identity_r8 에서 '있음' 84–100 %, 절벽 없음 | 09-26 | G09 철회표 #2 · `RollOutV3/Archive/_superseded/RECALL_TUBELET_LIMIT_2026-09-24.md` |
| "가림 조건에서 사상이 한 번 더 바뀐다 / late 의 p 는 또 다른 자리" | encoder 쌍 (h_fut → z, z → h) 도 late 에서 같이 무너짐 | 09-25 | `RollOutV3/audit/v11_identity/AUDIT.md` §0-5 |
| "p 가 정체를 지킨다" 를 영속 증거로 | 복사 수준 — z_all 보다 낮고 [z_all, p] 로 안 늚 | 09-25 | 〃 §3 |
| "물체의 정체성을 잃는다" | 잰 것은 자리의 물체다운 토큰; p 에 shape 98.46 / color 99.49 | 09-12 ~ 09-22 | POSITION_READOUT · `PAPER_STORY_2026-09-22.md` 금지표 |
| "p 에 정보가 없다" | 등가속+가림 p probe 98–100 | 08-20 · 08-30 | `z_world_model_analysis/MECHANISM_2026-08-20.md` · CLAUDE.md §6 |
| "가림이 표현을 망가뜨린다" | env 72 칸 100 | 08-30 | RESULTS §12 |
| "t0–t4 전부 '없음'" 을 영속 증거로 | t0 (k ≥ 3 의 t1) 은 물체가 실제로 가려진 칸 (h 도 0.11 / 0.37) | 09-25 | G09 철회표 #11 |
| "파괴 1.22 — 경계만 바꾸면 물체 신호가 통째로 넘어간다" (M2) | 빈 장면 섞인 문턱 0.234 에서만; 넘어가는 것은 있음뿐, 자리 회복 0.15 | 09-25 | `M2_BOUNDARY_PATCH_2026-09-25.md` 정정 1·2 |
| v11 post-FT "학습 도메인 안에서는 진짜 상태 진화처럼 보인다" | 속도 1 종 · 궤적 같음 → 학습 분포 재생 | 09-17 | PFT `RESULTS_2026-09-16.md` §3-1 |
| "IntPhys 2 permanence 에서 복사가 같거나 조금 낫다 (−5 pt)" | best-C 선택 산물 → "구분 안 됨" | 09-26 | `INTPHYS2_PERMANENCE_AUDIT` 2 차 정정 |
| future-matched 쌍이 공정 (presence 0.9427) | 미래일치 쌍은 누수 흡수 → 문맥일치 2 쌍 표준 | 08-16 | `RESULTS_2026-08-16.md` · CLAUDE.md §1-2 |
| 폐기 pilot v11_occtiming 68.11 % | early/mid 가 문맥을 2k 장 가림 — 인용 금지 | 09-01 | timing README |

금지 표현 (이 장 주제): "context 에서 안 보이면 못 한다" (전부로 — **"마지막 관측에"** 로 좁힌다) · "기억 (memory) 문제다" · "p 가 물체를 마지막 자리에 둔다" · "p 에 정보가 없다" · "가림이 표현을 망가뜨린다" · "AR 이 돌아온다는 것을 안다" (내부 귀속) (`auto_research/paper/PAPER_STORY_2026-10-08.md` §7).

---

## 출처

- `auto_research/paper/PAPER_STORY_2026-10-08.md` (§2-3 ii · §4-3 · §7 · §재현)
- `z_research/IntPhysGenV11/README.md`
- `z_research/IntPhysGenV11/Archive/surprising_score/RESULTS_2026-08-30.md`
- `z_research/IntPhysGenV11/Archive/SYNTHESIS_2026-08-30.md`
- `z_research/IntPhysGenV11/Archive/surprising_score/VOTE_GRID_2026-08-31.md`
- `z_research/IntPhysGenV11/Archive/TALK_2026-08-30.md`
- `z_research/IntPhysGenV11/Archive/STATUS_AND_CRITIQUE_2026-09-03.md`
- `z_research/IntPhysGenV11/Archive/PAPER_STORY_2026-09-06.md`
- `z_research/IntPhysGenV11/Archive/PAPER_STORY_2026-09-22.md`
- `z_research/IntPhysGenV11_occlusion_timing_ablation/README.md`
- `configs/protocols/datasets.md` (`## v13_black`)
- `z_research/IntPhysGenV10/Archive/V10_2026-08-28.md`
- `z_world_model_analysis/RESULTS_2026-08-16.md`
- `z_world_model_analysis/FRAME_ABLATION_2026-08-19.md`
- `z_world_model_analysis/OCCLUSION_LENGTH_2026-08-20.md`
- `z_world_model_analysis/MECHANISM_2026-08-20.md`
- `z_research/v11_realistic/README.md`
- `z_research/v11_realistic/exp_results/tables.md`
- `z_research/v11_realistic/figures/token_surprise/README.md`
- `z_research/v11_realistic/figures/token_surprise/background_checks.md`
- `auto_research/Archive/I_SCENE_CAUSAL_BOTTLENECK_2026-09-25.md`
- `auto_research/Archive/VERIFY_ISCENE2_2026-09-26.md`
- `auto_research/Archive/INTPHYS2_PERMANENCE_AUDIT_2026-09-26.md`
- `z_research/OcclusionStats/README.md`
- `z_research/OcclusionStats/Archive/INTPHYS1_OCCLUSION_2026-09-26.md`
- `z_research/Benchmarks/Archive/INTPHYS1_EVENT_POSITION_2026-09-25.md`
- `z_research/RollOutV2/figures/v5/summary/POSITION_READOUT_2026-09-12.md`
- `z_research/RollOutV2/figures/v5/v11_vanish/timing/attn_diag.md`
- `z_research/RollOutV2/figures/v5/v11_vanish/timing/token_test.md`
- `z_research/RollOutV2/figures/v5/v11_vanish/emergence/emergence.md`
- `z_research/v11_roll_out/README.md`
- `z_research/RollOutV3/README.md`
- `z_research/RollOutV3/Archive/IDENTITY_R8_2026-09-26.md`
- `z_research/RollOutV3/Archive/_superseded/RECALL_TUBELET_LIMIT_2026-09-24.md`
- `z_research/RollOutV3/_superseded/README_2026-09-25.md`
- `z_research/RollOutV3/audit/v11_presence/AUDIT.md`
- `z_research/RollOutV3/audit/v11_identity/AUDIT.md`
- `z_research/RollOutV3/audit/training_v8/AUDIT.md`
- `z_research/context_encoder_analysis/exp_results/ctxenc_boundary_identity_v2/RESULTS.md`
- `z_research/context_encoder_analysis/Archive/STATE_EVOLUTION_CAPABILITIES_2026-09-19.md`
- `auto_research/Archive/FUTURE_FROM_CONTEXT_2026-10-02.md`
- `auto_research/Archive/M2_BOUNDARY_PATCH_2026-09-25.md`
- `auto_research/Archive/COPY_HARD_SUBSET_2026-09-29.md`
- `auto_research/Archive/VERIFY_R3_2026-09-29.md`
- `auto_research/Archive/VERIFY_PRESENCE_2026-09-29.md`
- `auto_research/Archive/AR_PREDICTOR_2026-09-27.md`
- `auto_research/Archive/AR40_READOUT_2026-09-30.md`
- `auto_research/Archive/VERIFY_AR40RD_2026-09-30.md`
- `auto_research/Archive/COPY_VS_POSITION_2026-10-07.md`
- `z_research/TrainingEffects/v11_presence/PRESENCE.md`
- `z_training/ARIEL_CHECKPOINTS.md`
- `z_training/RESULTS_2026-09-19.md`
- `z_research/predictor_training/predictor_IntPhysGenV11_PFT/Archive/SCORING_2026-09-16.md`
- `z_research/predictor_training/predictor_IntPhysGenV11_PFT/Archive/RESULTS_2026-09-16.md`
