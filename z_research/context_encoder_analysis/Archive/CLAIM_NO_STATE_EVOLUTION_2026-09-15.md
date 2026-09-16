# 주장: 릴리즈 V-JEPA 2 predictor 는 마지막 관측 상태를 토큰별로 짧게 외삽할 뿐, 상태를 이어가지 않는다 — 근거 종합

> **하나의 주장**을 독립 측정으로 세우는 문서. 가설 탐색은 `STATE_EVOLUTION_HYPOTHESES_2026-09-15.md`, 시작점은 `../README.md`.
> 수치는 산출물에서 다시 읽은 값이고 칸마다 출처·n 을 적는다. 대안 설명과 남은 구멍을 결론과 같은 비중으로 둔다.
> **판: v0.2 (2026-09-16).** v0 를 독립 검토 3 개로 반박시켜 v0.1 로 좁혔고, v0.2 에서 2 차 캐시 분석 4 개 (정렬 후 채점 재계산 · 가속 교차 쌍 · E2 v2 · 운동 부분공간) 를 반영했다. 바뀐 것은 §7.

---

## 0. 주장 (v0.1, 좁힌 판)

> 릴리즈 V-JEPA 2 (ViT-H, `mask_index 0`, 문맥 16 → 미래 16, 합성 렌더 16 fps·stride 3) 의 predictor 는
> **미래 토큰마다 문맥에서 마지막 관측 상태를 외삽한다 — 재귀 없이, 짧은 지평으로.**
> 물체 특이적 정렬이 살아 있는 지평은 운동 종류에 따라 **등속 평면 3–4 튜블릿, 가속·수직·선반 ≤ 1–2 튜블릿**이다.
> **경계에서 가려진 물체 (k ≥ 2) 는 어느 미래 슬롯에도 놓이지 않는다.**
> 가림 채점 실패는 **encoder 를 바꾸지 않고 predictor 만 이어 학습해 회복 가능**하다 — 다만 그 회복이 상태 진화인지 도메인 prior 인지는 미결이다.

v0 에서 물러선 것 (근거는 §2·§7): "약 4 튜블릿" 은 등속 평면 값 · "물체가 없어진다" 는 **진실 자리**에서만 확인 (옛 자리에 남는지 미결) · "새 동역학을 안 붙인다" 는 ledge 로는 지평과 못 가르고 wall 은 혼합 · "재료는 encoder 에 있었다" 는 추론.

---

## 1. 조작적 정의

물체 상태 s = (정체성, 위치, 속도[, 가속]).

| 축 | 정의 | "한다" 로 판정될 관측 | 실제 관측 (v0.1) |
|---|---|---|---|
| **R. 재귀** | 미래 토큰이 서로를 참조해 상태를 넘긴다 | `mask→mask` 차단 시 채점·예측 붕괴 | 채점 −1.4 (예측은 40 % 움직임) — 채점은 안 의존 (§2-1) |
| **A. 지평** | 슬롯 i 의 물체가 시점 8+i 의 자리에 있다 (물체 특이 gate 통과) | 슬롯 7 까지 | 등속 평면 3–4, 그 밖 ≤1–2 (§2-2) |
| **B. 가림** | 경계에서 가려진 물체가 미래에 놓인다 | 미래 슬롯에 물체 토큰, 가려진 시점 너머 위치 | k=4: 슬롯 0 부터 약함→없음; k=1: 슬롯 0–1 약하게, 그 뒤 없음 (§2-4) |
| **C. 동역학 전환** | 문맥에 없던 힘·제약이 미래에 적용된다 | ledge 낙하 / wall 정지 | ◐ ledge 는 지평과 분리 불가, wall 은 혼합 (§2-3) |

셋 다 라벨·물리 지식 없이 캐시에서 잰다. **토큰별 외삽 (재귀 없는 1–2 차)** 은 이 정의에서 "상태 진화" 로 치지 않는다 — 비평자가 그렇게 부를 수 있음을 인정하고, 주장은 "재귀·지평·가림·전환" 네 축으로 한정한다.

---

## 2. 근거 사슬

### 2-1. 채점은 `mask→ctx` 경로에만 의존한다 (재귀 R)

- **측정**: attention 간선 knockout (frozen). v11_full 21,504 matched pair.
- **수치**: `mask→ctx` 전 층 **38.3 (−37.5)** · `mask→mask` **74.4 (−1.4)** · `ctx→ctx` 76.4 (+0.5); 기준선 75.86 (`clean` 75.85, 차이 0.01 pp; 토큰 max|Δ| 0.015).
- **단서 (v0.1 추가)**: `mask→mask` 를 끊으면 **예측 자체는 움직인다** — `pred_drift` rel_l1 **0.42**, cos 0.93 (`ctx→ctx` 0.13/0.99, `mask→ctx` 1.90/0.14). 미래 토큰끼리 내용을 교환하긴 한다; **채점이 그 교환에 의존하지 않을 뿐**이다. 채점은 슬롯 0 에 지배되므로 (§2-2 iii) 슬롯 4–7 에서 `mask→mask` 가 하는 일은 이 측정으로 안 보인다.
- 층 프로파일: 폭 7 창이 **층 8 을 포함하는 순간** 꺾인다 (창 2–8: −20.6; 5–11: −37.4); 단일 층 knockout 은 L8 **−14.4**, 나머지 ≥ −8.
- **출처**: `IntPhysGenV11_occlusion_timing_ablation/exp_results/knockout__v11_full_vith/results.json` (`specs[].metrics.pred_drift`), `knockout_w1__v11_full_vith/`.

### 2-2. 물체 특이 시간 정렬은 짧고 운동 종류에 따라 다르다 (지평 A)

- **측정**: `D[i][j] = mean_S |p_i − LN(h_j)|`, j = 0..15. v0 (궤적 합집합 S, pooled gate) 와 **v0.1 gated** (Δc = c_S − c_BG, 슬롯별 null; 시점별 S_j). v2 시나리오당 60 (fall 56), v11 k=0 224/조건. 대조군 z–h: 기울기 0.998–1.001, 정수 argmin 100 %.
- **시각 서명 (교란)**: 배경 토큰만으로도 argmin 이 ±1 안에 **슬롯 5 까지** 든다 (flat_v 1.00/1.00/0.98/1.00/1.00/1.00; v11 flat …/0.85/0.92). **argmin 대각선은 물체 증거가 아니다.** 물체 몫은 S−배경 대비뿐.
- **gated 결과 (물체 특이, null A 통과 슬롯 / null B)**: v2 flat_v [0] / [0–4] · flat_a [0,1] / [0–2] · ramp_a [0] / [0,1] · arc [0] / [0,1] · fall [] / [] · ledge [0] / [0]; **v11 flat [0,1,4] / [0–3]** · **v11 ramp [0–6] (j* 7.9/9.0/9.8/9.9/10.0/10.2/10.2) / [0,1]**. null A 는 v2 ramp_a·arc·ledge 와 v11 에서 퇴화 (같은 궤적 기하; z 자체가 A gate 를 못 넘음) → 그 셀은 null B 로 읽는다.
- **읽는 법**: 물체 특이 정렬은 **등속 평면 3–4 슬롯 (v11 flat, flat_v null B)**, **가속·수직·선반 ≤ 1–2 슬롯**. v0 의 "슬롯 0–3 제 시각" 은 시각 서명을 포함한 값이었다.
- **v11 ramp 의 j* ≈ 10 (슬롯 3–6)** — 옛 자리 (t≈10, 슬롯 2 위치) 에 물체가 **남아 있다**는 서명일 수 있다. 다른 시나리오의 평균 D 행도 슬롯 6–7 최솟값이 j=9–11 이다. v0 의 "물체가 없어진다" 는 진실 자리 측정 (자 질량, 토큰 검사) 에서만 성립한다. **"없어짐 vs 옛 자리 유지" 는 미결** (§6-1).
- **독립 측정**: (i) 자 attention 물체 3×3 질량 — **flat k=0** 0.74/0.65/0.73/0.68/0.53/0.34/0.22/0.06, **ramp k=0** 0.74/0.65/0.56/0.20/0.05/0.13/0.03/0.02 (균등 0.035; n 미기재); (ii) 토큰 검사 r>1 비율 (n=56) flat 1.00/1.00/1.00/0.93/0.79/0.05/0.04/0.16, ramp 1.00/1.00/0.98/0.25/0.23/0/0/0; (iii) 튜블릿 단독 채점 (가림 없음, 5,376 쌍, `token_time_profile.npz`) t0..t7 = **90.3 / 72.0 / 76.1 / 72.3 / 75.3 / 62.5 / 66.3 / 61.4** — 큰 계단은 t0→t1 (−18.3), 1–4 평평, t4→t5 −12.8. 세 측정이 같은 슬롯을 가리키지는 않는다 (flat 은 5–6, ramp 는 3).
- **출처**: `context_encoder_analysis/exp_results/time_alignment/`, `.../ctxenc_time_alignment_gated/RESULTS.md` (Verdict 절), `RollOutV2/figures/v5/v11_vanish/timing/{attn_diag,token_test}.md`, `IntPhysGenV11_occlusion_timing_ablation/exp_results/token_time_profile.npz`.

### 2-3. 새 동역학 적용 여부 — ledge 는 못 가르고, wall 은 혼합 (전환 C, ◐)

- **ledge**: 낙하/부유 궤적의 y 차이는 슬롯 0–4 에서 0/2/7/14/24 px (칸 18 px). 슬롯 0–2 는 같은 토큰 창 (질량 pos = imp 0.67/0.59/0.62) 이라 **판정 불가**; 슬롯 3–4 에서 부유 창 질량 0.43/0.39 vs 낙하 0.17/0.19 인데 그 분리 (14/24 px) 는 자의 held-out 오차 (0.87 칸 = 15.7 px) 근처이고, p 의 x 가 진실보다 뒤처져 (20/25 vs 30/40 px) **"낙하 곡선 위에서 지연" 과 구분되지 않는다.** 또 슬롯 3 은 ledge 의 시간 정렬이 이미 깨진 자리 (§2-2). → A 와 C 를 ledge 로는 분리 못 한다. ⚠️ v0 의 "낙하 시작은 슬롯 0–2 안" 은 **철회**.
- **wall**: 슬롯 1–2 통과 창 질량 0.61/0.48, **정지 창도 0.33/0.21** (균등 0.035) — 혼합이다. "자가 벽 몸통에 끌린다" 는 대안 (학습셋에 벽 없음) 미검. probe P(imp) 0.70–0.90 은 "pos 가 아님" 까지만.
- **결론**: C 는 ◐. 결정하려면 (a) 정지 자리·벽 몸통 토큰에서 파라미터 없는 r 검사, (b) 분기가 슬롯 0–2 에 오는 시나리오 (선반 끝을 경계에 두기) 가 필요하다.
- **출처**: `RollOutV2/figures/v5/summary/{POSITION_READOUT_2026-09-12.md §4, two_futures_attn.md}`, `exp_results/probe_pos_imp/`.

### 2-4. 경계에서 가려진 물체는 미래 슬롯에 놓이지 않는다 (가림 B)

- **수치**: 토큰 검사 r>1 비율 (n=56) late **k=4: 0.54/0.39/0.09/0.05/…**, **k=1: 0.80/0.79/0.38/0.21** — k=1 (경계 양쪽 한 샘플만 가림) 은 슬롯 0–1 에 약한 물체 (r≈1.05), 슬롯 2 부터 없음; k=4 는 슬롯 0 부터 약함. 자 질량 k=4 슬롯 0 0.03. ⚠️ v0 의 "슬롯 0 부터 없음" 은 k=4 에 대해서도 과했다 (0.54).
- **채점은 이 k 차이를 안 따른다**: late 59.4/59.1/59.7/60.4 (k=1..4) — 물체 유무가 k 에 따라 다른데 점수는 평평. 채점에는 채점층 요소 (CLAUDE.md "무너지는 건 배치다") 가 따로 있다. 채점 수치는 이 절의 기전 근거가 아니라 **결과** 로만 둔다: visible 87.2 / early 79.0 / mid 78.2 / late 59.5 (각 5,376 쌍).
- **knockout_hidden** (n_pair 1,792/조건): static late `mask→hidden` 차단 **+15.4** (대조군 −0.6) — 평소 가림막 토큰을 읽어 빈 장면을 앞으로 복사. moving late −2.0/−2.6 은 **해석 불가** (대조군 9 셀 범위 −6.8…+3.6; 분포 밖 개입). v0 의 "옛 자리에서 외삽하지 않는다" 는 **철회** — 직접 검정은 시간 범위 knockout (§6-4).
- **대안**: (a) encoder 에 정보 없음 — late 문맥 전체 readout shape moving 94–98 / static 80–90; p self shape 94–99 (색 static late 81–91). (b) 가림막이 코드를 옮김 — 국소 코드 이식 76/94 (self 88/99). (c) 기억 길이 (Bordes 2025) — **◐**: k=1 은 반 튜블릿만 가려도 무너지므로 "길이 무관" 과 "기억 ≈ 0" 이 구분되지 않는다. early/mid 회복은 물체가 경계 전에 다시 보이는 경우라 기억 요구가 없다.
- **출처**: `RollOutV2/.../timing/{token_test,attn_diag}.md`, ablation README 전수 1·4, `knockout_hidden__v11_full_vith/results.json`, `ctxenc_boundary_identity_occlusion/RESULTS.md`.

### 2-5. encoder 를 바꾸지 않고 회복 가능하다 — 무엇을 배웠는지는 미결

- **재료 (경계 튜블릿)**: 정체성 (visible self ≈100, p 통과 후 shape 94–99) · 위치 (자 ≤0.6 칸) · **속도** — flat_v 시나리오 내 z16 t7 물체 창 R² **0.988**, **배경 창 0.871**, 위치 지름길 −0.047 (n=300/시나리오; pooled 0.991 / 0.898 / 지름길 0.37–0.66). 속도는 **경계 튜블릿 전체에** 있다 (물체 창에 국소적이지 않다). 가려진 물체의 상태는 경계 창에 보이던 코드로 없다 (§2-4 a).
- **post-FT** (`v11_postft` e10, temporal_prefix, held-out block): late **57.4/59.4/59.6 → 87.1/84.0/87.4** (static/flat/ramp, 896 쌍씩). **단, visible 셀도 오른다**: static_visible 92.3→96.9, flat_visible 91.1→95.0, ramp_visible 79.8→92.7, early/mid +2…+19. 가림 특이 이득은 대략 late 이득 − visible 이득 ≈ +15…+25.
- **교차 대조**: IntPhys1 로 학습한 predictor → v11 static late +12.5 (57.4→69.9), moving late −2.7 / +0.9; v11 로 학습한 predictor → IntPhys1 88.89 → **77.22**. 도메인 prior 와 부합하는 패턴이다.
- **읽는 법**: "encoder 를 바꾸지 않고 v11 채점을 세우기에 충분한 정보가 encoder 에 있다" 까지. "이어갈 상태가 encoder 에 있었다" 는 post-FT p 의 시간 정렬 (§6-5) 전까지 추론이다.
- **출처**: `ctxenc_boundary_kinematics/RESULTS.md` (n=300), `z_training/runs/{v11_postft,intphys1_postft}/eval/*/summary.json`, `release_vith/eval/v11_split_test_from_existing_runs.json`.

### 2-6. 출력 기하 (서술)

- p participation ratio **16–19** (z/h 62–66), top-16 분산 72–78 % (h 41 %), h top-16 과 겹침 0.62–0.67 (split-half 0.93–0.99), 가림 무관. (v0 의 토큰 std 비교는 h 가 LN 돼 있어 무의미 — 삭제.)
- "없어진다 vs 옛 자리" 가 미결이므로 (§2-2) 이 기하를 "평균 쪽" 으로 읽는 문장도 보류.
- **출처**: `ctxenc_subspace_overlap/RESULTS.md`.

### 2-7. 전역 선형 정렬 p→h 를 걸고 채점을 다시 하면 — 오르지만 작다 (v0.2 추가)

- **측정**: 가능 변이·visible 3 조건 (960 clip) 으로 W 를 맞추고 (Procrustes / ridge / norm-matched / 가족별), held-out block 의 matched pair 에 W(p) 로 채점. 조건당 288 block (576 쌍), block bootstrap 1000.
- **수치 (ridge, 쌍 차이 Δacc [CI])**: visible 88.9 → 91.9 (+3.0 [2.1, 4.0]) · **late 61.2 → 64.4 (+3.2 [1.8, 4.8]; 가족별 ridge 66.7, +5.6)** · early 81.1 → 87.8 (+6.7 [5.4, 8.1]). margin 은 약 2 배. 대조군 (다른 clip 의 h 로 맞춘 ridge, z→h ridge) 은 late 에서 0 (+0.9 / +1.4, CI 가 0 을 포함), visible 에서는 크게 떨어진다 (−13 / −9).
- **읽는 법**: v0.1 의 "토큰 L1 ≤ 9 % 만 닫힌다" 가 채점에서는 **실제 이득**으로 나타난다 — margin 이 base 의 1 % 라 작은 이동도 순위를 바꾼다. 그러나 late 의 이득 (+3~6) 은 post-FT (+25~30) 의 1/5 이다. **CLAUDE.md §10 4 번 (정렬 사상) 은 기각이 아니라 "부분 효과"** 로 둔다.
- **단서**: 이 재계산은 probing 캐시 (bf16) 를 쓰므로 fp32 채점기와 근접 tie 가 뒤집혀 raw 정확도가 참조와 다르다 (static late 63.4 vs 참조 58.2, 같은 block; pairing 은 검증됨). Δ 는 같은 토큰 위의 쌍 차이라 그보다 믿을 만하지만, 정식 수치는 fp32 재추출 후에 쓴다.
- **출처**: `context_encoder_analysis/exp_results/ctxenc_subspace_v2/RESULTS.md` §0–2.

### 2-8. 가속 — 지금 데이터로는 못 가른다 (v0.2 확정)

- flat_v (a=0, v 7 수준) × flat_a (v0=0, a 7 수준) 교차 쌍 (`ctxenc_accel_crosspair`, 각 784): 위치·속도를 맞춘 뒤 "가속 있음" 이진 readout 은 F1 1.00 인데 **배경 창도 0.997, 그리고 둘 다 정지 (v=0, a=0) 인 대조 쌍에서도 갈린다 (AUC 0.20)** — 두 시나리오는 운동과 무관한 렌더·장면 단서로 구분된다. 즉 has_a 1.00 은 시나리오 지름길이다. 연속 ax7 잔차 R² (F1 0.43 / F3 0.48 / 배경 0.20) 도 같은 이유로 못 믿는다.
- **결론**: 경계 토큰의 가속은 **한 시나리오 설정 안에서 (v0 × a) 를 교차한 새 세트**로만 잴 수 있다 (사용자 결정: 만든다). §3 "합성 렌더 단서" 행과 같은 뿌리.
- **부분공간**: 경계 토큰 안에서 위치 ~ 속도 부분공간 겹침 0.019 (랜덤 0.002, split-half 천장 0.39), 속도 ~ 시각 서명 0.004, 시나리오 ~ 시각 0.001 → **위치·속도·시각·장면은 거의 직교**. 위치 ~ 시나리오 0.09–0.15 만 약하게 얽힘. **물체 창의 속도 부분공간과 배경 창의 "속도" 부분공간 겹침 0.043 (천장 0.29)** → 배경에서 읽히는 속도는 다른 코드 (장면·카메라 단서) 다. v0.1 의 "속도가 물체 창에 국소적이지 않다" 는 **"배경에도 속도를 맞히는 다른 코드가 있다" 로 정정**. 출처 `ctxenc_state_subspaces_kinematics/RESULTS.md` (b)(c).

---

## 3. 대안 설명 총정리 (v0.1)

| 대안 | 상태 | 처리·근거 |
|---|---|---|
| encoder 에 정보가 없다 | ❌ | §2-5 재료; 가려진 것도 문맥 앞쪽에 (§2-4 a) |
| 자(readout) artifact | ◐ | held-out 0.87 칸, encoder ≤0.6 칸, 토큰 검사 일치 — 단 ledge 는 분리 14–24 px 로 오차 근처, wall 은 벽 attractor 미검 |
| 해상도 | ❌ | z–h 대조군 대각선 (0.5 칸) |
| 토큰 시각 서명 | ✅ 처리 | gated 판 (§2-2) 로 물체 몫만 읽음 |
| **옛 자리에 남는다 (stale)** | ⬜ 미결 | v11 ramp gated j*≈10, D 행 최솟값 j=9–11 — 옛 자리 질량·r 검사 필요 (§6-1) |
| **낙하 곡선 위 지연 (ledge)** | ⬜ 미결 | §2-3 |
| 채점 프로토콜 | ❌ | 위치·토큰 측정은 채점과 무관 |
| 가림막이 encoder 코드를 옮김 | ◐ | 국소 안정 / 토큰 평균 붕괴 (readout 의존) |
| 기억 길이 | ◐ | k=1 실패 = 기억 ≈ 0 과 구분 불가 |
| in-domain prior (post-FT) | ⬜ | visible 셀도 상승, 교차 대조 (§2-5) |
| 합성 렌더의 시나리오 단서 (정지 대조도 갈림) | ✅ 확인 | §2-8 — 시나리오 간 비교는 전부 이 단서를 의심할 것 |
| 회귀-평균 (hedging) | ⬜ | p 슬롯별 정체성 probe 로 "없음 vs 뭉개짐" 분리 가능 (§6-3) |
| mask token 선택 | ⬜ | 전부 `mask_index 0`; EK100 릴리즈 코드는 1 (영벡터) |
| 단일 체크포인트·fp16·단일 seed | ⬜ | ViT-H 만; ViT-L 은 자체 데이터에서 이기는 축 있음 (CLAUDE.md §1-1) |
| 합성 렌더만 | ⬜ | 실영상은 IntPhys1 채점뿐 |
| 샘플링 주기 (튜블릿 = 0.375 s 는 우리 프로토콜) | ⬜ | 지평이 튜블릿 단위인지 초 단위인지 미검 |
| 표본 | — | 채점 5,376–21,504 쌍; ledge/wall 392; E1 300/시나리오; 시간 정렬 224/조건; 토큰 검사 **56**; E2 셀당 165–184 |

---

## 4. 이 주장이 말하지 않는 것

- "경계를 복사한다" 는 아니다 — 등속 평면에서 3–4 슬롯은 물체 특이 정렬이 있다 (정지 복사 철회는 CLAUDE.md §6, POSITION_READOUT §4-3).
- "predictor 가 X 를 못 만든다" 류 표현은 쓰지 않는다 — 정체성은 p 를 통과한다. 못 하는 것은 시간축으로 나르는 것이고, 그것도 "재귀·지평·가림·전환" 네 축으로 한정한다.
- 원인 (사전학습 마스크 공간 튜브 × 시간 전체, EMA 표적) 은 인용·서술로만 (Bardes 2024, `configs/train/vith16/pretrain-256px-16f.yaml`).
- 정렬 개입 (Procrustes/ridge) 은 **효과가 있으나 작다** (§2-7): late 채점 +3~6 pt. 회복의 주력은 아니다.
- IntPhys 2 는 이 레포에서 실측한 적이 없다 — 그 실패 기전에 대해 말하지 않는다.

---

## 5. use case — 단위를 튜블릿으로

- 우리 지평은 **튜블릿 단위**로 재졌다 (튜블릿 = 샘플 2 장 = raw 6 장 @16 fps = 0.375 s; 초 단위 환산은 프로토콜 의존).
- **EK100 anticipation** (Assran 2025, 릴리즈 코드): 문맥 뒤 **4 튜블릿째** 단일 튜블릿을 예측 (1 s @ 8 fps; 슬롯 0–3 mask 토큰 없음, `mask_index 1` = 영벡터). 그 자리는 우리 측정에서 물체 특이 대비가 gate 를 벗어나는 슬롯이고, 토폴로지·샘플링이 다르다. **"그 안에서는 된다" 는 v0 의 문장은 철회.** predictor 가 거기서 기여하는지는 논문 Table 20 (+0.6) 이 전부이고 우리 encoder-only arm 은 없다.
- V-JEPA 2-AC 는 pretrained predictor 를 쓰지 않고 새로 학습했다 (`load_predictor: false`) — 방향이 같다.
- 가림은 testbed; 지평·재귀·출력 기하는 가림 없이 성립.

---

## 6. 남은 구멍 (닫히는 순서)

| # | 무엇 | 왜 | 비용 · 상태 |
|---|---|---|---|
| 1 | **옛 자리 검사**: 슬롯 4–7 에서 t≈10 위치 3×3 의 자 질량·r 검사, null B gate | "없어짐 vs 옛 자리 유지" (§2-2) | 캐시, ⬜ |
| 2 | ledge 지연 보정 (p 의 x 에서 낙하 곡선 y 평가) · wall 벽 몸통 r 검사 | C 를 ◐ 에서 결정 | 캐시, ⬜ |
| 3 | p 슬롯별 정체성 probe | "없음 vs 뭉개짐" | 캐시, ⬜ |
| 4 | **시간 범위 knockout** `mask→ctx@t0-5` vs `@t6-7` (릴리즈 + post-FT) | 앞 프레임을 읽는가 · post-FT 가 배운 것 | GPU forward, 부분집합 10 분대 — 사용자 결정 |
| 5 | post-FT p 시간 정렬·자 질량 | 회복이 상태 진화인지 | GPU 추출 |
| 6 | 정렬 사상 걸고 채점 재계산 | 채점층 개입 가부 | ✅ §2-7 (fp32 재추출로 확정 필요) |
| 7 | 가속: 교차 쌍 ✅ §2-8 (시나리오 지름길로 무효) → **(v0 × a) 세트 필요** (렌더, 사용자 결정: 만든다) | 경계 토큰의 가속 | 설계 대기 |
| 8 | E2 v2 ✅ (`ctxenc_boundary_identity_v2`): concat 9 토큰도 visible→late chance (shape 20/16–20, ramp 28–34 약간 위); **late→early 역이식 static 36 / flat 41 / ramp 73** → 가려진 자리 코드는 부분적으로 읽히는 정체성 코드; C_noocc 는 late→visible 을 올리지만 (74→82, 79→97) visible→early 는 거의 안 올림 (41→47) → 가림막 존재의 전역 이동은 구성 효과만이 아님. attentive 판은 미실행 | E2·E3 ◐ 유지 | |
| 9 | 상태 부분공간: 운동 편 ✅ §2-8 · v11 (정체성·가림막·시각) 편 ⬜ 미실행 (`ctxenc_state_subspaces_v11.py` 존재, 결과 없음) | 재료 얽힘 | 캐시 |
| 10 | `mask_index 1` 재채점 · ViT-L · stride 2 재추출 | §3 의 ⬜ 셋 | GPU |

---

## 7. 변경 이력 · 철회

| 판 | 무엇 |
|---|---|
| v0 (09-15 밤) | 첫 판. §0 "약 4 튜블릿 1차 이어 붙이기 · 그 너머·가려진 물체·새 동역학은 만들지 않는다 · encoder 에 재료가 없어서가 아니다" |
| **v0.2** (09-16) | §2-7 정렬 후 채점 (+3~6, 부분 효과 — §10-4 기각 철회), §2-8 가속 교차 쌍 무효 (시나리오 지름길) + 운동 부분공간 (위치·속도·시각 직교, 배경 속도는 다른 코드 — "국소적이지 않다" 정정), §6 8·9 갱신. 정렬 개입 관련 §4 문장 수정 |
| **v0.1** | 반박 검토 3 개 반영. **철회**: "낙하 시작은 슬롯 0–2 안" (같은 토큰 창) · "옛 자리에서 외삽하지 않는다" (해석 불가 개입) · "슬롯 4 이후 물체는 없다" (진실 자리에서만; 옛 자리 미결) · "EK100 1 s 는 지평 안" (4 튜블릿째 = 경계, 토폴로지·mask 다름) · "이어갈 재료는 encoder 에 있었다" → 추론으로 하향 · 토큰 std 비교 삭제 · clean/null 차이 0.14 → 0.01 pp · 층 5 → 층 8 진입. **추가**: gated 시간 정렬 (운동 종류별 지평), `mask→mask` 예측 drift, post-FT visible 셀 상승과 교차 대조, k=1 vs k=4, 대안 표 6 행, n (토큰 검사 56) |

---

## 재현

- 2-1: `IntPhysGenV11_occlusion_timing_ablation/README.md` §재현 (`analysis/attention/knockout/run_sharded.sh`); `knockout_w1`, `knockout_hidden` 같은 폴더
- 2-2: `PY=/data/hyuntak/anaconda3/envs/vjepa2/bin/python; $PY z_research/scripts/analysis/ctxenc_time_alignment.py --per-scen 60 --v11-n 224; $PY z_research/scripts/analysis/ctxenc_time_alignment_gated.py` (인자는 docstring); 자·토큰 검사는 `RollOutV2/figures/v5/summary/POSITION_READOUT_2026-09-12.md` §재현; 튜블릿 단독 채점은 `PAPER_STORY_2026-09-06.md` §재현의 npz 코드
- 2-3, 2-4: 같은 POSITION_READOUT §재현; E2 는 `$PY z_research/scripts/analysis/ctxenc_boundary_identity_occlusion.py --blocks 100`
- 2-5: `$PY z_research/scripts/analysis/ctxenc_boundary_kinematics.py 300`; post-FT 는 `z_training/README.md`
- 2-6: `$PY z_research/scripts/analysis/ctxenc_subspace_overlap.py`
