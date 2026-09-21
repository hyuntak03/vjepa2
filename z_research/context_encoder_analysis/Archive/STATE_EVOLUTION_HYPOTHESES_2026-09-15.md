# 관측 없는 상태 진화 (state evolution without observation) — 가설 문서

> **2026-09-19 — 정의는 [`STATE_EVOLUTION_CAPABILITIES_2026-09-19.md`](STATE_EVOLUTION_CAPABILITIES_2026-09-19.md) 가 정본이다** (네 능력: 읽기 · 지속 · 전이 · 영속; 물리 장면은 측정 도구). 이 문서의 축은 거기 §4 대응표로 흡수됐다.
>
> **살아 있는 문서다.** 가설을 세우고, 재고, 고쳐 쓴다. 고친 것은 지우지 않고 §9 변경 이력과 본문의 `⚠️ 정정` 으로 남긴다.
> 레포 규칙은 루트 `CLAUDE.md`. 이 문서의 수치는 전부 `exp_results/` 산출물 또는 다른 세트의 `summary.json`/`results.json` 에서 다시 읽은 값이고, 출처를 칸마다 적는다.
> 시작점: `../README.md`. 판: **v1 (2026-09-15 밤)** — E1–E4 1차 결과와 선행 연구 반영. v0 와의 차이는 §9.
>
> ⚠️ **v1 검토 결과 (2026-09-15 밤, 독립 검토 3 개) — 아래 상태는 이 blockquote 가 본문보다 우선한다. 판정의 정본은 `CLAIM_NO_STATE_EVOLUTION_2026-09-15.md` v0.1.**
> - **E1**: §3-2 는 n=30 값이다. n=300 (`exp_results/ctxenc_boundary_kinematics/RESULTS.md`): flat_v 시나리오 내 F1 vx 0.988, **배경 창 0.871** (n=30 의 0.39 는 표본 artifact), 파라미터 없는 판 배경 창 +0.09 vs −0.04 (CI 안 겹침). → **속도는 경계 튜블릿 전체에 있고 물체 창에 국소적이지 않다.** ramp_a 가속 0.83 은 경사각 = 장면 지름길 (POS67 0.99). "해상도 아래" 는 틀림. E1 상태: 속도 ✅ (국소성 ✗), 가속 ⬜ (교란).
> - **E2**: ✅ → **◐**. ridge 이식 실패 (chance) 이지 "경계에 정체성 없음" 이 아니다 — static 은 가려진 자리에 다른 국소 코드 (B7−BG +35~42). "원리적으로 못 푼다" 삭제. attentive/concat 재검정 중.
> - **E3 (전역)**: "채점기가 보는 것" 과 등치 불가 (채점기는 토큰별 |p−h| 평균, 특징 평균이 아님). B7 76/94 → B7all 54/64 → C 41/47 로 풀링 폭에 단조 → 가림막 토큰 혼입 (구성 효과) 가 더 정확. 설계 요구 3 의 "토큰 국소 채점" 은 CLAUDE.md §6 기각 항목 — 삭제.
> - **E4**: "전역 Procrustes 사실상 기각" → **◐**. 토큰 L1 감소율만 쟀고 채점 재계산은 안 했다 (margin 0.08–3.7 % 라 3–9 % 이동으로 순위가 바뀔 수 있음). z→h 대조군 ridge 31 % 도 같이 읽을 것. "p ∉ span(h)" → "h top-256 안 (0.90) 의 저차원 부분집합". 채점 재계산 진행 중.
> - **P2**: ✅ → ◐. 배경 토큰만으로 슬롯 5 까지 대각선 (시각 서명). gated 판 (`ctxenc_time_alignment_gated/`): 물체 특이 정렬은 등속 평면 3–4 슬롯, 가속·수직 ≤1–2. "물체가 없어진다" 는 진실 자리에서만; v11 ramp 슬롯 3–6 이 j*≈10 → 옛 자리 유지 가능성 미결.
> - **P4**: ✅ → ◐ (관찰: 회복; 원인·내용 미확정). visible 셀도 +4~13 상승, IntPhys1-학습 predictor 는 v11 static late 만 +12.5 (moving −2.7/+0.9). "가장 직접적인 증거" → "encoder 를 바꾸지 않고 회복 가능".
> - §2-3 표 "intphys1_postft … late 그대로" 는 틀림: static late 57.4→69.9 (+12.5). §2-1 knockout "moving 은 이롭다" 삭제 (대조군 범위 −6.8…+3.6). §3-3 "C 96–98" → moving k=2..4 94–97 / static 80–90. §7-5 "전부 frozen" → 분석은 frozen, §2-3 은 기존 재학습 run 재읽기.
> - **09-16 추가 (2 차 분석)**: E4 정렬 사상은 채점을 실제로 올린다 (late +3~6, visible +3, early +7; 대조군 0) — "기각" 이 아니라 부분 효과. E1 가속은 flat_v×flat_a 교차로도 못 가른다 (정지 대조 쌍도 갈림 = 시나리오 지름길) → (v0×a) 세트. 운동 부분공간: 위치·속도·시각 서명·장면이 경계 토큰 안에서 거의 직교 (겹침 ≤0.02, 천장 0.3–0.4); 물체 창 속도 ≠ 배경 창 속도 코드 (0.043). E2: concat 도 chance, late→early 역이식은 36/41/73 (부분 정체성 코드). 정본 판정은 CLAIM v0.2.
> - 본문은 다음 판 (v1.1) 에서 고쳐 쓴다; 그때까지 위 항목이 본문을 덮는다.

---

## 0. 질문

우리가 만들고 싶은 것은 **관측이 끊긴 구간에서 latent state 를 스스로 이어가는 predictor** 다 (video prediction 관점: 문맥 z_{0..7} 만 보고 미래 s_{8..15} 를 만든다. 가림·화면 밖·미발생 전부 같은 문제다).
V-JEPA 2 의 predictor 가 그 자리에 있는데, 지금까지의 증거는 "이어가지 않는다" 다. 그래서 질문은 셋으로 갈린다:

| | 질문 | 답이 이쪽이면 |
|---|---|---|
| **Q1** | 상태를 이어가는 데 필요한 **재료가 encoder 출력에 있는가**, 그것도 **쓸 수 있는 형태**로 | 있다 → Q2 / 없다 → Q3 |
| **Q2** | 있다면 **predictor 의 어느 병목**이 그걸 못 쓰게 하나 | 병목을 고친 predictor 를 학습한다 (beat 6·7) |
| **Q3** | 없다면 encoder 수준에서 상태가 **가림막·타이밍·장면과 얽혀** 있는가 (spurious correlation) | encoder 도 손대거나, 얽힘을 푸는 readout/학습 신호가 필요하다 |

**2026-09-15 밤 시점의 답 (근거는 §3, 판정 트리는 §5):**

> 재료는 있다 — 정체성·위치·장면·**속도**까지 encoder 에, 그것도 predictor 가 실제로 읽는 **경계 토큰**에 있다.
> 없는 것은 두 가지다. (1) **가려진 물체의 상태는 경계 토큰에 "보이는 물체의 코드" 로 남지 않는다** (정보는 문맥 다른 곳에 있다).
> (2) **가속**은 지금 데이터로 못 잰다 (설계상 속도와 분리 안 됨).
> 그러니 병목은 대체로 predictor 다: 경계만 조회하고 (P1), 4 슬롯 뒤 물체를 잃고 (P2), 출력이 표적과 다른 저차원 부분공간에 있으며 그 차이는 전역 선형 정렬로 닫히지 않는다 (P3·E4).
> 얽힘은 **readout 의 종류에 따라** 보인다: 국소 토큰 코드는 가림막 존재에 안정하고, **토큰 평균(채점기가 보는 것)** 은 가림막이 장면에 들어오는 순간 옮겨진다 (E3).

---

## 1. 정의

**상태 s_t** 의 구성 요소 (이 문서에서 "재료"):

| 재료 | 무엇 | 어디서 재나 | 2026-09-15 상태 |
|---|---|---|---|
| 정체성 | shape · color | attentive/ridge probe | ✅ 있음 (§2-1, §3-3) |
| 위치 | 화면 위 물체 자리 | 위치 자, 시간 정렬 행렬, 자 없는 토큰 검사 | ✅ 있음 (§2-1, §3-1) |
| 운동 — 속도 | 방향·크기 | **§3-2 E1** | ✅ 경계 토큰 하나에 있음 |
| 운동 — 가속 | 규칙 (등가속·중력·경사) | **§3-2 E1** | ⬜ 미결 — v2 설계가 v 와 a 를 못 가른다 |
| 존재 | 가려져 있어도 "있다" | **§3-3 E2**, knockout_hidden | ❌ 경계 토큰의 "보이는 물체 코드" 로는 없음; 문맥 전체엔 있음 |
| 장면 | 지면 · 경사 · 가림막 · 배경 | env probe (100) | ✅ 있음 |

**"쓸 수 있는 형태"** — 세 가지를 요구한다:
1. **읽힌다** — 선형/attentive readout 으로 chance 를 크게 넘긴다
2. **자리가 맞다** — predictor 가 실제로 읽는 자리(문맥 **경계 토큰**, §2-2)에 있다. 문맥 전체에 퍼져 있으면 경계만 조회하는 predictor 는 못 쓴다
3. **분리돼 있다** — 가림막 유무·타이밍·장면 같은 nuisance 와 얽히지 않아, 비가림에서 배운 readout 이 가림에서도 통한다

**"진화"** — s_{t+1} = f(s_t) 를 관측 없이 반복할 수 있는 것. 조작적으로는 (a) 미래 슬롯의 물체가 **제 시각의 자리**에 있는가 (§3-1), (b) 미래 토큰끼리 **서로를 참조**하는가 (knockout `mask→mask`).

**병목이 앉을 수 있는 자리**:

```
문맥 프레임 ─► context encoder ─► z ─► predictor ─► p  ◄── |p − LN(h)| ──  h ◄─ target encoder ◄─ 전체 프레임
              [E] 입력 재료·얽힘      [P] 내부 경로·학습        [X] 출력 좌표·표적 기하
```

---

## 2. 이 문서 이전에 확립된 것 (출처 포함)

### 2-1. encoder 쪽

| 사실 | 값 | 출처 |
|---|---|---|
| 정체성은 문맥 z 에 있다 (조건별 self probe) | color 97–100 · env 100 · shape ≈100 (self) | `IntPhysGenV11_occlusion_timing_ablation/README.md` 전수 6 |
| **shape 은 z 에서도 조건을 가로지르면 흔들린다** | 비가림→가림 이식 z: static 90.3 / flat **51.8** / ramp 66.3 (late); color 93–100 | 같은 문서 전수 7 |
| 위치는 encoder 어디서나 읽힌다 | 자 ≤ 0.6 칸 (v2 7 시나리오), v11 가림 타이밍 무관 추적 | `RollOutV2/figures/v5/summary/POSITION_READOUT_2026-09-12.md` §2-2, §5-4 |
| 공간평균 readout 은 가림막 존재만으로 무너진다 (h 도) | static early: h self 19 → visible readout 115 cm | `v11_roll_out/README.md` §3 |
| 가려진 문맥 튜블릿은 predictor 에 **쓰인다** — static 은 해롭고 moving 은 이롭다 | `mask→hidden` 차단: static late **+15.40** / flat late −1.95 / ramp late −2.57 (대조군 ±1) | `knockout_hidden__v11_full_vith/results.json` (`breakdown.condition`, 2026-09-15 재계산) |

### 2-2. predictor 쪽

| 사실 | 값 | 출처 |
|---|---|---|
| 채점을 나르는 경로는 `mask→ctx` 뿐 | `mask→ctx` −37.5 · `mask→mask` −1.4 · `ctx→ctx` +0.5 | ablation README §knockout |
| 층 0 에서 미래 토큰이 문맥 **마지막 프레임**에 몰린다 | 문맥 질량 0.885, 부호 Δt −12.4 프레임 | `IntPhysGenV11/figures/attention/README.md` |
| 정체성은 p 를 통과한다 | p self shape 98.5 / color 99.5 / env 100 | ablation README 전수 6 |
| p 는 h 의 좌표계에 없다 — 가림 없이도 | h→p 이식 static 94.9 / flat 79.0 / ramp **57.3** (color, vis) | ablation README 전수 8 |
| 채점 margin 은 base 의 0.08–3.7 % | 효과크기가 곧 정확도 (ρ 0.986) | ablation README 전수 10 |
| horizon: 신호는 경계 직후에만 | 튜블릿 단독 채점 t0 90.3 → t7 61.4 (가림 없음) | `PAPER_STORY_2026-09-06.md` §3 |
| p 안의 물체다운 토큰이 남는 길이 | k=0 슬롯 4–5 < early/mid 3 < late 0 | POSITION_READOUT §5-3, 5-4 |

### 2-3. 학습 쪽 — predictor 만 바꿨을 때 (`z_training/`, 2026-09-10 실행)

| run | 학습 데이터 · 마스크 | v11_split_test (10,752 쌍) | IntPhys1 dev sliding `skip2_w32/avg` (180 쌍) |
|---|---|---|---|
| 릴리즈 ViT-H | — | **75.83** | **88.89** |
| `v11_postft` e10 | v11 train 절반 (가능만), `temporal_prefix` C16, 10 ep, lr 5e-5 | **91.35** (moving_occ 87.4 · static_occ 87.1 · flat_occ 84.0) | **77.22** (−11.7) |
| `intphys1_postft` e40 | IntPhys1 train 3,750, `temporal_prefix`, 40 ep | 78.53 (late 가림 56.9 / 60.3 / 69.9 — 그대로) | **93.89** (+5.0) |

출처: `z_training/runs/{v11_postft,intphys1_postft}/eval/*/summary.json`, `z_training/runs/release_vith/eval/v11_split_test_from_existing_runs.json`, `runs/*/config.yaml`.

- **encoder 를 그대로 두고 predictor 만 이어 학습해도 가림 채점이 60 → 87 로 오른다.** Q1 = 예 의 가장 직접적인 증거.
- **배운 것이 무엇인지는 모른다.** v11 로 배운 predictor 는 IntPhys1 을 11.7pt 잃고, IntPhys1 로 배운 predictor 는 v11 late 를 못 고친다. 도메인 prior 와 상태 진화를 **지금 수치로는 못 가른다** → §4 P4, §6.
- ⚠️ 문헌과의 긴장: Garrido et al. 2025 는 **사전학습부터** causal(뒤 25 % 시간 마스크) 로 하면 block 마스크보다 나빴다고 보고한다 (§8). 우리 것은 frozen encoder 위 predictor 만, in-domain 후학습이라 같은 실험이 아니다. 화해는 §6 의 P4 검정에서.

---

## 3. 2026-09-15 에 새로 잰 것

전부 캐시만, GPU 없음, 파라미터 0 또는 닫힌 형식 ridge. 스크립트는 `z_research/scripts/analysis/ctxenc_*.py`, 산출물은 `exp_results/<이름>/`.

### 3-1. 시간 정렬 행렬 — p 슬롯 i 는 실제 어느 시점 j 와 닮았나 (`ctxenc_time_alignment.py`)

```
D_ph[i][j] = mean_{s∈S} |p_i − LN(h_j)|      i = 0..7 (p 슬롯 = 진짜 시점 8+i),  j = 0..15 (target encoder 의 16 튜블릿 전부)
S          = 진실 궤적 16 튜블릿의 3×3 토큰 합집합.  D_zh = z(32 장 다 본 context encoder) 로 같은 것 = 대조군 (대각선이어야 함)
null       = 같은 시나리오 다른 clip 의 h·S → 대비 c=(중앙값−최솟값)/중앙값 의 95 백분위 = gate
```

| | 이동 (칸/튜블릿) | z–h 대조군 기울기 | p 일치 비율 s0..s7 | p j* (일치한 것만) s0..s3 | 기울기 (clip 단위) |
|---|---:|---:|---|---|---|
| v2 flat_v | 0.50 | 1.000 | .98 1.00 .97 1.00 · **0 0 .12 0** | 8.06 8.94 10.08 10.91 | 0.91 ± 0.17 |
| v2 flat_a | 0.60 | 1.000 | .78 1.00 1.00 .50 · 0 0 0 0 | 8.01 8.89 10.01 10.80 | 0.97 ± 0.05 |
| v2 ramp_a | 0.57 | 0.999 | .17 .77 .75 .43 · 0 0 0 0 | 8.09 8.99 10.11 10.89 | 0.95 ± 0.03 |
| v2 arc | 0.84 | 1.001 | .85 .75 .62 .02 · 0 0 0 0 | 8.02 8.94 10.07 10.76 | 0.93 |
| v2 ledge | 0.76 | 0.999 | .72 .75 .90 .10 · 0 0 0 0 | 7.15 8.93 9.16 **5.44** | (이봉, 추세 아님) |
| **v11 flat k=0** | 1.01 | 1.000 | .96 1.00 .98 .74 · 0 0 0 0 | 8.02 8.94 10.06 10.88 | 0.98 ± 0.03 |
| **v11 ramp k=0** | 1.28 | 1.000 | 1.00 1.00 1.00 .83 · 0 0 0 0 | 8.04 8.88 10.06 10.87 | 0.97 |

(v2 시나리오당 60 clip, v11 조건당 224; fall 은 미래가 정점 근처라 0.38 칸, 슬롯 0 만 읽힘; wall 은 미래가 정지.)
출처 `exp_results/time_alignment/time_alignment.{json,md}` (행렬 전수).

읽는 법:
1. **도구는 산다.** z–h 가 0.5 칸/튜블릿에서도 정확히 대각선 (정수 argmin 100 %). 해상도는 문제가 아니다.
2. **p 는 경계를 복사하지 않는다.** 슬롯 0–3 의 j* 는 8.0 / 8.9 / 10.1 / 10.9 — 자기 시각. "정지 복사" 기각.
3. **느린 시계도 아니다.** 시계는 정확하고 **슬롯 4 에서 절벽**: 슬롯 4–7 은 어느 시점과도 안 맞는다 (대비 null 수준, 행 최솟값 0.58–0.60 vs 0.53–0.54). 물체가 **4 슬롯 (= 1.5 s) 뒤에 없어진다.**
4. 자 결과 (물체다운 토큰이 k=0 에서 슬롯 4–5 까지) 와 맞는다. ⚠️ **정정** — 자의 gain 0.4–0.5 (ramp/arc) 를 "시간이 절반쯤 흐른다" 로 읽던 것은 슬롯 4–7 의 기본값이 섞인 값이었다. 시계는 슬롯 0–3 에서 거의 1 이다.

단서:
- **토큰에는 시각 코드가 있다.** 궤적 밖 배경 토큰만으로도 슬롯 0–1 은 전 시나리오에서 대각선 (±1 안 100 %), 슬롯 2–5 도 자주 (대비 0.07–0.15). h_j 는 물체와 무관하게 튜블릿 j 의 서명을 갖고 p 가 그걸 맞춘다. 물체 창 S 의 대비는 그 위에 +0.05~0.10. **"p 가 물체를 제 시각 자리에 둔다" 는 이 차이분으로만 말할 수 있고** 이 판은 그걸 따로 gate 하지 않았다 (§6).
- S 는 궤적 전체 합집합 (28–54 토큰). 시점별 3×3 로 재면 더 날카롭다.
- 캐시 `target.npy`·`isolated_ctx`·`ctx_masked` 는 추출 때 이미 LN 된 값이다 (`evals/analysis_vlm/occlusion_identity/forward.py` `_ln`, 행 std 1.0000 실측). v0 의 "캐시는 LN 전" 짐작은 틀렸고 결과엔 무관 (재적용 차이 9e-5).

### 3-2. E1 — 경계 토큰에 운동 상태가 있는가 (`ctxenc_boundary_kinematics.py`)

RollOut_v2 가능 clip, **시나리오당 30 (n=180; 300 판은 §9 참조)**. 표현 z16 = `rollout_v2_z16_vith/ctx_masked` (문맥 16 장만 본 context encoder = **predictor 의 실제 입력**). F1 = 튜블릿 7 의 진실 물체 3×3 평균 (1280-d). 라벨 = 위치 차분 (칸/튜블릿). ridge, block 5-fold, λ 내부 검증. 출처 `exp_results/ctxenc_boundary_kinematics/n30/RESULTS.md`.

**지름길 먼저.** 시나리오 안에서 위치만으로 속도가 맞혀지는 곳은 못 쓴다 (v11_roll_out 의 교훈):

| 시나리오 | POS7→v R² (시나리오 내) | 판정 |
|---|---:|---|
| flat_v · flat_a | −0.16 · −0.19 | 위치 고정 (anchor raw 45) — **속도 검정 가능** |
| fall (vy) | 0.05 | **가능** |
| ramp_a · arc · ledge | 0.998 · 1.000 · 1.000 | 위치가 속도를 결정 (anchor·edge 고정) — **못 씀** |

**속도 — 경계 토큰 하나에 있다** (시나리오 내 fit, OOF R²; 셔플 ≤ 0):

| | F1 (z16 t7 3×3) | F1_bg (물체에서 ≥4 칸 떨어진 3×3) | F1h (h, 32 장 봄) | **P0 (p 슬롯 0)** |
|---|---:|---:|---:|---:|
| flat_v vx | **0.931** | 0.39 | 0.927 | **0.898** |
| flat_a vx | **0.729** | −0.16 | 0.761 | **0.744** |
| fall vy | **0.967** | 0.64 | 0.971 | **0.970** |

파라미터 없는 판 (같은 위치 ±0.3 칸 clip 쌍, 시나리오 평균 뺀 코사인): 속도 비슷 vs 다름 = flat_v **+0.42 vs −0.22**, flat_a **+0.54 vs −0.17**, fall **+0.41 vs −0.18** (95 % CI 안 겹침); 배경 창은 +0.03 vs −0.06 (차이 없음). → 튜블릿이 샘플 2 장을 보므로 물리적으로 가능한 일이고, 실제로 그렇다. **P0 도 같은 수준** — 속도는 p 로 넘어간다.

**가속 — 이 데이터로는 미결.**
- flat_a: v0 = 0 고정이라 a ∝ v7 → 분리 불가 (위치·속도 맞춘 뒤 a 다른 쌍 0 개)
- ramp_a: a 2 수준, 라벨 std **0.011 칸/튜블릿²** (0.2 px) — F1 0.01 / F2 0.09 / F3 0.14 / **h 0.21**. 아무 표현도 못 읽는다 → encoder 의 성질이 아니라 **효과 크기가 토큰 해상도 아래**
- arc: a 변이 없음. fall: ay 변이는 바운스 위상 (F1 0.69, F2 0.84) — 가속이라기보다 궤적 이력
- ⇒ **"경계 토큰에 가속이 없다" 는 아직 말할 수 없다.** 검정하려면 **(v0 × a) 교차 설계**가 필요하다 — 이것이 RollOut 을 **추가로** 만들 유일한 실질 이유다 (§3-5, §6).

### 3-3. E2·E3 — 경계 토큰의 정체성 × 가림 (`ctxenc_boundary_identity_occlusion.py`)

v11_full obj clip, (조건, k) 셀당 100 block (165–184 clip). 표현 z = `v11_full_vith/ctx_masked` (predictor 입력), h = target. B7 = 튜블릿 7 의 진실 물체 위치 3×3 평균 (late 에서는 가림막이 있는 자리), B7all = 튜블릿 7 전체 256 토큰 평균, C = 문맥 2048 토큰 평균, BG = 물체에서 먼 고정 3×3. ridge 분류, block held-out, chance shape 14.3 / color 12.5. 출처 `exp_results/ctxenc_boundary_identity_occlusion/RESULTS.md` (576 fit 전수).

**(1) 가려진 채로 자기 조건 안에서 읽으면 (self, late k=2..4, 경계 두 프레임 모두 가림):**

| | z:B7 (경계 물체 창) | z:BG (배경 창) | z:C (문맥 전체) | h:B7 |
|---|---:|---:|---:|---:|
| static shape | 71.6–76.4 | 29.6–40.2 | 79.9–89.7 | 52.7–77.0 |
| moving_flat shape | 49.4–72.0 | 46.9–53.5 | 94.3–97.1 | 42.4–66.3 |
| static color | 45.0–49.7 | 11–15 | 62–71 | 30–48 |
| moving_flat color | 21.2–45.7 | 17–28 | 83–94 | 30–46 |

- static: 가림막 자리 창에 **국소 코드가 남는다** (B7 − BG +35~42 shape). moving k=3,4: 창 ≈ 배경 — 물체가 그 자리에 있어 본 적이 없으니 국소적으로 남을 게 없다. 정보는 **문맥 전체**에 있다 (C 96–98 shape).
- **h 도 그 창에서 더 높지 않다** (h:B7 ≤ z:B7). 미래에 물체가 다시 나오는 걸 본 encoder 도 가려진 자리에 정체성을 **되돌려 놓지 않는다** — amodal 완성은 이 토큰에 없다.
- ⚠️ BG 가 chance 가 아니다 (shape: visible 43.8 static / 74.6 moving). ViT 가 정체성을 attention 으로 전역에 퍼뜨린다. 국소성 주장은 전부 **B7 − BG 차이**로만.

**(2) 이식 — 보이는 물체에서 배운 readout 을 걸면:**

| 학습 → 평가 | z:B7 shape (static / moving) | z:C shape | z:B7 color | z:C color |
|---|---|---|---|---|
| visible → **late** (가려짐) | **20.2 / 16.0** (≈chance) | 20.9 / 23.7 | 12.5 / 14.0 | 32 / 39 |
| visible → **early** k=4 (가림막만 있고 물체는 경계에서 보임) | **76.1 / 94.5** (self 87.6 / 98.8) | **41.5 / 47.5** (self 91 / 99) | 52.8 / 85.1 | 33.5 / 45.3 |
| early → late | 19.0 / 16.0 | 41.2 / 56.5 | 14.3 / 13.0 | 53.6 / 53.2 |

읽는 법 — **E2 확정, E3 는 readout 종류에 따라 갈린다**:
- **E2**: 가려지면 경계 창의 "보이는 물체 코드" 는 사라진다 (이식 chance). 자기 조건 안에서 읽히는 (1) 의 값은 **다른 코드**다. 즉 predictor 가 경계를 조회해서 얻는 건 가려진 물체의 상태가 아니다.
- **E3 (국소)**: 가림막이 장면에 있어도 물체가 보이면 국소 코드는 그대로다 (76 / 94). **가림막 존재는 국소 코드를 옮기지 않는다.**
- **E3 (전역 평균)**: C 는 가림막이 들어오는 순간 옮겨진다 (91 → 41, 99 → 47). **채점기가 보는 토큰 평균에서는 가림막 존재가 spurious correlation 이다** — v11_roll_out §3 의 공간평균 readout 붕괴와 같은 현상이고, 채점의 `visible → early` 색 하락 (−14.1, `v13_black` 에서도 −13.5) 과 같은 자리다.

### 3-4. E4 — z / p / h 부분공간 (`ctxenc_subspace_overlap.py`) — CLAUDE.md §10 1번

v11_full, 조건 6 (static·moving_flat × visible / late k1–4 / early k4), 조건당 120 block × 256 토큰 = 30,720 토큰. z = ctx_masked, p = predictor, h = target 미래 절반, p·h 는 같은 clip·같은 토큰. 출처 `exp_results/ctxenc_subspace_overlap/RESULTS.md`.

| 측정 | 값 | 기준선 |
|---|---|---|
| **p vs h top-16 주각 겹침** (조건 내) | **0.62–0.67** (k=64: 0.57–0.60), 가림 유무 무관 | split-half 0.93–0.99 · 랜덤 0.013 |
| z vs h | 0.83–0.88 | |
| p 의 차원 | participation ratio **16–19** (z/h 63–66); top-16 이 분산 72–78 % (h 41 %); 토큰 std 0.60–0.64 (h 1.0) | |
| p16 ⊂ h 의 top-256 | 0.90 | split 1.000 |
| z 부분공간의 가림 회전 (vis–late) | static 0.890 / moving 0.852 | 시나리오 차이 (static_vis–moving_vis) 0.837 |
| z vis–early | static 0.892 (= late) / moving 0.923 (> late) | |
| **Procrustes (visible 에서 fit) 가 없애는 \|p−h\|₁** | held-out visible **6.7 %** (ridge 9.0) · late 이식 3.5 (2.6) · early 5.1 | late 에서 직접 fit 한 천장 5.9–6.1 (ridge 8.0–8.4); h→h 100 |
| CKA (같은 clip·토큰) | 0.48–0.55 | 다른 clip·같은 토큰 인덱스 0.42–0.47 (85 %) — **위치가 지배** |

읽는 법:
- **p↔h 좌표 불일치는 가림과 무관한 상시 성질**이고, 회전이라기보다 **저차원 부분집합**이다 (p 는 h 의 주 방향 안에 90 % 들어가지만 분산이 훨씬 작다).
- **가림이 z 를 돌리는 정도는 지면·운동 차이만큼**이다. 조건 간 같은 표현의 겹침(≥0.76)이 조건 안 p–h 겹침(≤0.67)보다 항상 크다.
- **전역 선형 정렬로 닫히는 몫은 격차의 10 % 미만**이고 late 는 그보다 작다. ⚠️ 이건 CLAUDE.md §10 4·5번 (정렬 사상 W: p→h, 채점층 개입) 의 **전역 Procrustes 판을 사실상 기각**한다 — 개입이 있다면 토큰·슬롯 국소적이거나, "어떤 토큰을 세는가" 쪽이어야 한다. 단, 이건 토큰 L1 기준이고 pos/imp 순위(채점)를 직접 잰 건 아니다.

### 3-5. RollOut_v2 재생성 판정 (2026-09-15)

- **해상도 때문이라면 불필요.** §3-1 대조군이 0.5 칸에서 정확히 대각선. 기하 상한도 직선 0.89 칸/튜블릿 (대각 1.01; 예측 16 장만 화면 안이면 1.85) 이라 v11 moving (1.0–1.3) 이 이미 상한 근처다. v2 의 속도 격자는 설계 때 프레임 안에서 스윕한 값이다 (`gen/rollout2.json` note, `ROLLOUT_V2_DESIGN.md` §3).
- **갈아엎지 않는다.** v2 테스트 프레임에 묶인 캐시 172.6 GB, `/data2` 여유 353 G. 같은 video_ids 로 재생성하면 `TokenCache.matches()` 가 렌더를 서명에 안 넣어 옛 캐시를 조용히 재사용한다 (09-10 에 겪음). POSITION_READOUT 의 v2 절이 재현 불가가 된다.
- **추가로 만든다면 이유는 둘이고, 둘 다 새 이름의 세트다**: (a) **(v0 × a) 교차 설계** — 경계 토큰의 가속 검정 (§3-2), (b) **반사실 미래 렌더** — 같은 문맥에 두 진짜 미래 (ledge/wall 형, 원운동 원호/접선 = `PAPER_STORY_2026-09-06` §6 L1). `gen` config 의 `name`/`dataset_name` 을 새로 두면 Shards·plan·캐시 태그가 분리된다.

---

## 4. 가설 (상태: ✅ 확립 · 🔄 측정 중 · ⬜ 미측정 · ❌ 기각 · ◐ 부분)

### [E] encoder 쪽 — 재료와 얽힘 (Q1 · Q3)

| # | 가설 | 상태 · 근거 (§3) | 남는 것 |
|---|---|---|---|
| **E1** | 경계 토큰에 속도는 있고 가속은 없다 | **속도 ✅** (F1 R² 0.73–0.97, 지름길 배제, 파라미터 없는 판 일치, P0 도 같음). **가속 ⬜** — v2 가 v/a 를 못 가름 | (v0 × a) 세트. 가속이 없으면 "등속 0.85 vs 가속 0.4–0.5" 의 설명이 서고, 있으면 P 쪽 |
| **E2** | 가려진 물체의 정체성은 경계 토큰에 남지 않는다 | **✅** visible→late 이식 chance; h 도 못 되돌림. 정보는 문맥 전체 (C 96–98) | 경계 조회 predictor 는 late 를 원리적으로 못 푼다 → 설계 요구 2 |
| **E3** | 가림막 존재가 정체성 코드를 옮긴다 | **◐ readout 의존** — 국소 창 ❌ (76/94 유지), 토큰 평균 ✅ (91→41) | 채점기 = 토큰 평균 → 채점이 보는 공간에서는 spurious. **X1 의 구성 요소** |
| **E4** | 가림이 z 부분공간을 회전시킨다 / p 는 h 부분공간에 없다 | 회전 ◐ (시나리오 차이만큼) / **p ∉ span(h) ✅ 상시**, 전역 선형 정렬로 ≤ 9 % | §10 4·5 재검토 (§3-4) |
| **E5** | 토큰에 시각 코드가 있고 표적의 상당 부분이 그걸로 채워진다 | ◐ 관측만 (§3-1 단서) | 정량화 — 서술로만 (목적함수 논리 금지) |

### [P] predictor 쪽 — 병목 (Q2)

| # | 가설 | 상태 · 근거 |
|---|---|---|
| **P1** | 미래 토큰은 문맥을 **조회**할 뿐 서로를 참조하지 않는다 | ✅ knockout |
| **P2** | 조회로 만든 미래는 **4 슬롯까지 제 시각에 정확**, 그 뒤 **물체가 사라진다** (절벽) | ✅ §3-1, 자와 일치. ⚠️ 시각 코드 단서 |
| **P3** | p 는 h 의 좌표계에 없다 — 가림 없이도, **저차원**으로 | ✅ §2-2 + §3-4 |
| **P4** | 학습 신호가 문제 — `temporal_prefix` 로 predictor 만 이어 학습하면 회복 | ✅ 회복 / ❌ 원인·내용 미확정 (IntPhys1 −11.7). **검정**: post-FT p 의 시간 정렬 → 절벽이 밀려나면 진화, 대비만 커지면 외형 prior |
| **P5** | 마스크 토폴로지 불일치 | 인용·관찰로만 (사용자 결정 09-15). Bardes 2024 §8 |
| **P6** | 절벽은 입력 재료 (E1) 인가 경로 (P1) 인가 | ◐ — 속도는 있으니 등속에서의 절벽은 **경로**. 가속은 미결 |
| **P7** (신규, 문헌) | 예측 불가능성을 담을 자리(잠재변수·mixture·행동 조건)가 없어 긴 지평에서 조건부 평균으로 간다 (Vondrick 2016, MoP-JEPA 2026) | ⬜ 서술. "물체가 사라진다" 와 부합하나 측정은 없음. α 분해는 인용 금지 |

### [X] 출력·표적 쪽

| # | 가설 | 상태 |
|---|---|---|
| **X1** | 채점 margin 이 작은 것은 거리의 대부분이 좌표 불일치 (P3) + 시각 코드·배경 (E5) + 가림막 존재의 전역 이동 (E3) 이기 때문 | ◐ §3-3, §3-4 (CKA 는 위치가 85 %). 물체 토큰만의 분해는 §6 |
| **X2** | 표적 h_t 는 튜블릿 t 의 스냅샷에 가깝다 | ⬜ (GPU forward). E2 의 "h 도 가려진 자리에 정체성을 안 되돌린다" 가 간접 근거 |

---

## 5. 판정 트리 — 지금 위치

```
E1 속도?  ─O─► 가속?  ──⬜ (v0×a 세트 필요)──► 없으면: 경계 조회 predictor 는 가속 운동을 원리적으로 못 나름 → 새 predictor 는 문맥 전체를 시간 통합 (설계 요구 1)
                                                있으면: encoder 는 통합한다, predictor 가 안 쓴다 → P1/P4
E2 가려진 상태가 경계에?  ─X (확정)─► 이어가려면 경계 앞 토큰의 상태를 시간축으로 날라야 (설계 요구 2: 시간 인과·다단계). 지금 predictor 는 가림막 토큰을 읽는다 (static +15.4 가 그 증거)
E3 가림막 존재가 코드를 옮김?  ─국소 X / 전역 O─► 얽힘은 채점기가 보는 공간(토큰 평균)에 있다 → readout·학습 신호에서 nuisance 를 명시적으로 분리하거나 토큰 국소 채점 (설계 요구 3)
E4 전역 정렬로 late 고침?  ─X─► 채점층 전역 개입은 닫히지 않는다 → 개입은 predictor 학습 쪽 (설계 요구 4: 표적 좌표계에 맞춘 출력)
P4 post-FT p 시간 정렬  ─⬜─► 절벽이 밀려나면 "진화를 배웠다" / 대비만 커지면 "외형 prior"
```

**새 predictor 의 설계 요구 (지금까지의 증거로 정해진 것):**
1. 경계만이 아니라 **문맥 전체를 시간 통합**한다 (가속·이력이 필요한 운동; E1 미결이지만 안전한 쪽)
2. **자기 예측을 되먹여 다단계**로 간다 — 미래 토큰끼리 서로를 참조하고 (P1 반대), 가려진 상태를 경계 앞 토큰에서 날라 온다 (E2)
3. **가림막 존재 같은 장면 nuisance 와 물체 상태를 분리**한 채 낸다 (E3 전역)
4. **표적 좌표계 안에서, 표적과 같은 분산으로** 낸다 (E4·P3) — 전역 선형 후처리로는 안 된다
5. 예측 불가능성을 담을 자리 (P7, 문헌) — 이 문서의 범위 밖, 서술만

---

## 6. 실험 큐

| 순 | 실험 | 가설 | 비용 | 상태 |
|---|---|---|---|---|
| 1 | E1 을 시나리오당 300 으로 | E1 | 캐시만 | 🔄 (백그라운드, `run_n300.log`) |
| 2 | **post-FT predictor 의 p 추출 → 시간 정렬 행렬** (v11 late 포함) | **P4** | GPU 추출 — `features:` 추출이 `model.predictor_checkpoint` 를 타는지 확인 필요 | ⬜ |
| 3 | 시간 정렬 v2: 시점별 3×3 S, **배경 대비 차이 gate**, 슬롯별 gate | P2 · E5 | 캐시만 | ⬜ |
| 4 | **amnesic 검정** (Elazar 2020 / LEACE Belrose 2023): 캐시 p·h 에서 "가림막 존재" 방향을 닫힌 형식으로 지우고 채점 재계산 (norm 맞춘 대조 필수) — "정보가 있다" 를 "채점이 쓴다" 로 잇는 다리 | E3 · X1 | 캐시만 | ⬜ |
| 5 | 물체 토큰만의 |p−h| 분해 (시각 코드 / 배경 / 물체) + copy-last latent baseline 채점 (Luc 2017 표준) | X1 · E5 | 캐시만 (`is_p_just_context.py` 확장) | ⬜ |
| 6 | **(v0 × a) 교차 RollOut 세트** (새 이름) → E1 가속 | E1 · P6 | 렌더 + 캐시 | ⬜ 설계 |
| 7 | 반사실 미래 세트 (원호/접선) | beat 4 일반화 | 렌더 | ⬜ (`PAPER_STORY_2026-09-06` §6) |
| 8 | 토큰 norm 히스토그램·토큰별 시각/위치 선형 probe (Darcet 2023 형) | E5 | 캐시만 | ⬜ |
| 9 | target encoder 시간 수용장 (프레임 셔플 Δh_t) | X2 | GPU forward | ⬜ 서술용 |

하지 않기로 한 것: P5 마스크 토폴로지 개입 실험 (인용으로 대체). RollOut_v2 갈아엎기 (§3-5).

---

## 7. 하네스 판정 (CLAUDE.md §12)

1. **beat**: `PAPER_STORY_2026-09-06` beat 4 ((a) 를 §3-1 이 닫음) · beat 5 · **beat 7 의 전제** (encoder 동결 근거 = Q1: E1·E2·E3 이 그 답). E4 는 beat 5 의 "겹침" 을 실측했고 §10 4·5 의 개입 설계를 바꾼다.
2. **양쪽 결과 모두 beat 가 서는가**: 예 (§5). 실제로 E2 는 "없다", E3 는 "readout 의존", E4 는 "중간" 으로 나왔고 각각 설계 요구로 옮겨졌다.
3. **이미 답이 있는가**: 정체성·위치·장면은 있었음. 운동 상태의 자리·가려진 상태·부분공간은 없었고 오늘 쟀다. 문헌에도 없다 (§8: predictor 내부 knockout·horizon·좌표 불일치·가림 하 kinematics 는 미발표).
4. **교란**: 가림막 존재 vs 가려짐 — early 조건이 갈랐다. 지름길 — E1 에서 시나리오별로 배제. 시각 코드 — §3-1 단서로 남김. BG 가 chance 가 아님 — B7−BG 로만 주장.
5. **재학습**: 전부 frozen. P4 검정만 기존 run 추출.
6. **표본**: E1 n=30/시나리오는 얇다 (300 판 대기). E2 셀당 165–184, E4 토큰 3 만.
7. **corner case / use case**: 4 슬롯 절벽·저차원 좌표 불일치·시각 코드는 **비가림에서 성립**. 가림은 testbed. use case: anticipation (EK100 은 predictor 를 suffix 마스크로 그대로 씀), planning energy (V-JEPA 2-AC 가 predictor 를 새로 학습한 이유와 같은 자리).

금지 표현 점검: "predictor 가 X 를 못 만든다" — 안 씀 (p self 98). "가림이 표현을 망가뜨린다" — 안 씀 (E3 국소 ❌). "표현이 회전했다" — E4 실측 후 "저차원 부분집합" 으로만. 목적함수·마스크 토폴로지 — 서술·인용. α — 인용 없음 (P7 은 외부 문헌의 일반 진술).

---

## 8. 선행 연구 (2026-09-15 웹 검증 — 제목·저자·id·초록 기준. 본문 정독 전. UNVERIFIED 표시가 없는 것은 arXiv 로 확인)

### 8-1. 미래 표현 예측과 open-loop 상태 진화 — "이어가는 predictor" 의 재료 목록

| 논문 | 무엇 | 우리와의 관계 |
|---|---|---|
| Vondrick et al. 2016, *Anticipating Visual Representations* (1504.08023) | 프레임 하나 → 미래 fc7 회귀; L2 는 "mean of the modes … off the manifold" 이라 K-mixture | 단발 회귀의 원형. 조건부 평균 문제 (P7) 의 첫 진술 |
| Luc et al. 2017 (1703.07684) · 2018 (1803.11496) | 미래 segmentation / Mask R-CNN feature 예측; batch vs autoregressive vs **BPTT through own predictions**; copy-last baseline; horizon 붕괴 (AP50 39.9→19.4) | "자기 예측 되먹임" 학습의 표준. copy-last 가 우리 latent 채점의 자연스러운 null (§6-5) |
| Karypidis et al. 2024, *DINO-Foresight* (2412.11673) | frozen DINOv2 feature 를 masked-feature transformer 로 미래 프레임 예측, 자기회귀 rollout; mIoU 71.8→59.8 | V-JEPA 2 predictor 의 가장 가까운 형제. 차이: 마스크가 **시간 suffix**, 출력이 **입력과 같은 공간**이라 되먹임 가능 |
| Bardes et al. 2024, *V-JEPA* (2404.08471) | 손실 ‖P(E(x)) − sg(Ē(y))‖₁; 마스크 = 공간 블록 × **시간 전체** ("limits information leakage") | 출발점 문서. prefix→suffix 를 학습한 적 없음 (P5); 입력은 online, 표적은 EMA (P3 의 구조적 원인 후보) |
| Assran et al. 2025, *V-JEPA 2* (2506.09985) | EK100 anticipation 은 pretrained predictor 를 1 s suffix 로 그대로; **V-JEPA 2-AC 는 새 predictor** 를 teacher-forcing + rollout 손실(T=2) 로 학습, "error accumulation" 보고 | 우리 post-FT (§2-3) 는 이 레시피의 action-free 절반 |
| Garrido et al. 2025 (2502.11831) | sliding surprise 프로토콜; IntPhys 98 %; **causal (뒤 25 %) 사전학습은 block 보다 나빴다** | 채점 프로토콜의 원형. 긴장: 사전학습 causal ≠ frozen 위 predictor 후학습 (§2-3) |
| Hafner et al. — PlaNet 2019 (1811.04551) · Dreamer 2020 (1912.01603) · DreamerV3 2023 (2301.04104) | RSSM: 결정론 상태 h_t = f(h_{t−1}, z, a) + 확률 상태; latent overshooting; 상상 지평 15–16; Dreamer 는 one-step 손실로도 충분 (구조가 재귀라서) | "open-loop" 의 조작적 정의. 구조가 되먹임을 강제하면 다단계 손실이 필수는 아니라는 대조 |
| Zhou et al. 2024, *DINO-WM* (2411.04983) | frozen DINOv2 patch 위 19M predictor, one-step teacher forcing, 25–300 step 계획 | **"frozen encoder 에 재료가 있다" 의 가장 강한 외부 증거** — 공간 patch 토큰 (풀링 아님) 이 관건 |
| Baldassarre et al. 2025, *DINO-world* (2507.19468) | frozen DINOv2 위 block-causal predictor, IntPhys 91.3 / GRASP 76.0 / InfLevel 63.7 (그들의 V-JEPA-H 재실행 89.4 / 73.0 / 59.9), copy-last 대조군 chance | frozen 이미지 encoder + **시간 목적으로 새로 학습한 predictor** 가 직관물리에 닿는다 |
| Bar et al. 2024, *Navigation World Models* (2412.03572) | 시간 이동 k 조건, 자기회귀, ~8 s 뒤 붕괴 | "얼마나 앞" 을 명시 조건으로 주는 설계 |
| Wu et al. 2023, *SlotFormer* (2210.05861) | frozen slot 위 자기회귀 transformer, **예측된 slot 을 입력으로** 학습 | 분리된 상태 위 dynamics 의 대조 사례 |
| Chen et al. 2024, *Diffusion Forcing* (2407.01392) · Bengio et al. 2015 *Scheduled Sampling* (1506.03099) | teacher-forcing / open-loop 불일치와 그 치료 | 용어 고정: V-JEPA 2 predictor 는 teacher-forced 도 open-loop 학습도 아니다 |
| Sobal et al. 2025, *PLDM* (2502.14819) | JEPA 계열, predictor 를 H 단계 **자기 예측으로 unroll** 해 손실 | V-JEPA 2 와 정반대 레시피가 같은 계보 안에 있다 |
| LeCun 2022, *A Path Towards AMI* (OpenReview; PDF 는 봇 차단, 2차 출처로 확인) | world model = s_{t+1} = Pred(s_t, a_t, z_t), 잠재 z 가 예측 불가능성 담당 | 릴리즈 predictor 에는 재귀·z·행동 셋 다 없다 |

**종합 (L1)**: open-loop 상태 진화를 보고한 모든 모델이 공유하는 재료 — ① 자기 출력에 적용되는 전이 함수, ② 시간에 관한 학습 신호 (BPTT / rollout 손실 / overshooting), ③ **입력과 같은 좌표계의 출력**, ④ 예측 불가능성의 자리 (z / mixture / action), ⑤ 예측 가능하게 다듬어진 표현. V-JEPA 2 릴리즈 predictor 는 ①–④ 전부에서 벗어나고, ③ 은 우리 §3-4 가 실측했다. 지평 붕괴는 보편적이지만 (mIoU·AP·LPIPS 전부) 우리 **절벽**은 다르다 — 되먹임이 없으니 오차 누적이 아니라 "조회기가 쓸 수 있는 것의 경계" 다.

### 8-2. V-JEPA · 직관물리 분석 — 겹침과 빈자리

| 논문 | 본 곳 | 결과 · 우리와의 관계 |
|---|---|---|
| Bordes et al. 2025, *IntPhys 2* (2506.09849) | 전 체인 zero-shot | V-JEPA 2-h Main 57.5 / Held-Out 87.2. 설명은 "짧은 기억" 가설뿐, 기전 없음 |
| Joseph et al. 2026, *Interpreting Physics in Video World Models* (2602.07050) | **encoder 만** (부록 C.1.4 에서 층별 표현 위에 predictor 새로 학습) | 속도·가속·방향이 **중간 깊이** 에서 선형 해독, 마지막 층으로 갈수록 저하; 국소 attention head 제거 시 붕괴; **새 predictor 는 중간 층 입력이 최적**. → 우리 E1 은 **마지막 층** 경계 토큰에서 속도 0.93 을 읽었다 — 그쪽 "저하" 는 속도에 관한 한 우리 데이터에선 안 보인다 (가속은 미결) |
| Punzo et al. 2026 (2606.09646) | encoder 층별 probe | 선형 probe chance, temporal-attentive ≈65 % — **readout 용량이 "있다" 를 좌우** (우리 E2 의 B7 − BG, ridge 하한 단서와 같은 방향) |
| Musa et al. 2026 (2609.01551) | encoder 층별 probe | 카메라 운동 >90 AUC, IntPhys 2 물리 ≈ chance; 시간 특징이 저차원 매끈한 궤적 |
| Alrasheed et al. 2026 (2605.15618) | encoder 강건성 | 입력 가림에서 class 구조 유지 |
| Mur-Labadia et al. 2026, *V-JEPA 2.1* (2603.14482) | 목적함수 변경 | 문맥 토큰에도 손실 ("no incentive to encode local information within the context tokens"), mask 토큰에 시공간 위치. 직관물리 평가 없음 |
| Su et al. 2025 (2512.06232) | 전 체인 | 데이터 규모·분포만으론 IntPhys2 개선 없음 (0.50–0.54) |
| **HERA** 2026 (2608.05523, 저자 표기 불규칙 — id 로 인용) | **frozen predictor 에 어댑터** | V-JEPA 2-G predictor 에 3M 기억 라우팅 → IntPhys2 52.6→54.4 (continuity 46→58, immutability 46→63). 우리 §2-3 의 가장 가까운 사촌; 진단은 주장만 |
| Song et al. 2026, *MoP-JEPA / Branch-JEPA* (2607.05238) | JEPA 일반 | 분기점에서 회귀 최적 predictor 는 **조건부 평균** = 어떤 상태도 아님 — P7 의 형식 진술 |
| Tokmakov 2021 (2103.14258) · Shamsian 2020 (2003.10469) · Van Hoorick 2023 TCOW (2305.03052) · Traub 2023 Loci-Looped (2310.10372) | 추적·object permanence | 영속성은 **명시적 재귀 기억·what/where slot·상상 루프** 로만 얻어졌고 TCOW 는 지도 transformer 도 "considerable gap". masked-video SSL 이 가려진 상태를 유지한다는 발표는 없음 — 우리 E2 와 부합 |

**빈자리 (L2)**: predictor 의 attention knockout · 튜블릿별 horizon · p↔h 좌표 · 가림 하 정체성+위치+속도 동시 probe — **전부 미발표**. 우리 §2-2·§3 이 그 자리다. 화해 필요: Garrido 의 causal 사전학습 실패 vs 우리 post-FT 성공 (from-scratch joint vs frozen predictor-only, in-domain).

### 8-3. 얽힘 · spurious correlation · object-centric — 측정법과 치료

| 논문 | 무엇 | 우리와의 관계 |
|---|---|---|
| Naseer et al. 2021 (2105.10497) | ViT 는 80 % patch drop 에도 60 % 유지, 위치 순열에 둔감 | 정체성은 내용으로 읽힌다 — z 가 가림에서 정체성을 유지하는 것과 부합 |
| Xiao et al. 2020 (2006.09994) · Beery et al. 2018 (1807.04975) · Geirhos et al. 2020 (2004.07780) | 배경·장소 shortcut; 교란을 **바꿔 넣는** 통제 실험이 측정법 | E3 의 early 조건 = "가림막은 있는데 물체는 보임" 이 정확히 이 형식. in-domain post-FT 가 새 shortcut 을 심을 수 있다는 경고 (§2-3) |
| Darcet et al. 2023 *Registers* (2309.16588) · Jiang et al. 2025 (2506.08010) · Lappe & Giese 2025 (2505.05892) · Yang et al. 2024 *Denoising ViTs* (2401.02957) | 토큰에 절대 위치가 해독됨; 고norm 토큰이 전역 정보를 흡수; 위치 임베딩 격자 잡음 | **E5 시각 코드**와 같은 계열 — 단, 전부 절대 PE 이미지 ViT. V-JEPA 2 는 3D RoPE 라 우리 캐시에서 재야 함 (§6-8) |
| Raghu et al. 2021 (2108.08810) · Kornblith et al. 2019 CKA (1905.00414) | 토큰이 공간 슬롯을 끝까지 유지; CKA | E4 의 CKA 가 위치에 지배된 이유 |
| Elazar et al. 2020 *Amnesic Probing* (2006.00995) · Belrose et al. 2023 *LEACE* (2306.03819) | "읽힌다 ≠ 쓰인다"; 개념을 닫힌 형식으로 지우고 하류 변화 측정 | **우리 긴장 (probe 98 / 채점 4) 의 방법론**. §6-4. L1 은 norm 에 민감하니 norm 맞춘 대조 필수 |
| Locatello 2020 Slot Attention · Kipf 2021 SAVi · Wu 2022 SlotFormer · Jiang 2024 SlotSSM (2406.12272) · Daniel 2026 LPWM (2603.04553) · Zadaianchuk 2023 VideoSAUR (2306.04829) | 물체 단위 상태 분리; VideoSAUR 는 frozen feature 위 slot | 설계 요구 3 의 참조 설계. 우리 캐시 위에 slot 모듈은 미발표 |
| Ozguroglu 2024 pix2gestalt (2401.14398) | amodal 완성은 생성 prior 로 | E2 의 "h 도 가려진 자리에 정체성을 안 되돌린다" 와 부합 |

**빈자리 (L3)**: frozen **video** encoder 에서 정체성 × 가림막 존재 얽힘을 개입으로 잰 논문 없음 — E3 의 visible→early/late 이식은 새로 보인다. 위치·시각 코드 결과는 전부 절대 PE 이미지 ViT 에 대한 것.

---

## 9. 변경 이력

| 날짜 | 무엇 |
|---|---|
| 2026-09-15 낮 | v0. §3-1 시간 정렬 행렬 (파일럿 → `ctxenc_time_alignment.py`), §2-3 post-FT 수치, RollOut_v2 재생성 판정, E1–E4 착수, P5 를 인용으로 대체 |
| 2026-09-15 밤 | **v1.** E1 (n=30) · E2·E3 · E4 결과 반영 (§3-2~3-4), §0 답 갱신, §4 상태, §5 설계 요구 4 개, §6 큐 재정렬 (amnesic 검정·(v0×a) 세트 추가), §8 선행 연구 57 편 웹 검증. **정정**: 자 gain 0.4 = "시간이 절반" 읽기 철회 (§3-1-4); "캐시는 LN 전" 짐작 철회 (§3-1 단서); E3 가설을 "readout 의존" 으로 갈라 씀. E1 은 n=300 재실행 대기 |

---

## 재현

```bash
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
# §3-1 시간 정렬 행렬 (캐시만, ~5 분)
python z_research/scripts/analysis/ctxenc_time_alignment.py --per-scen 60 --v11-n 224
# §3-2 E1 (시나리오당 N; 30 → n30/, 300 → 본 폴더)
python z_research/scripts/analysis/ctxenc_boundary_kinematics.py 300
# §3-3 E2·E3 (셀당 100 block, ~5 분)
python z_research/scripts/analysis/ctxenc_boundary_identity_occlusion.py --blocks 100
# §3-4 E4 (~2 분)
python z_research/scripts/analysis/ctxenc_subspace_overlap.py
# §2-3 post-FT 수치
python -c "import json;[print(f, json.load(open(f))['surprise']['overall']['block_pairwise']) for f in ['z_training/runs/v11_postft/eval/surprise_c16t32__v11_split_test_e10/summary.json','z_training/runs/intphys1_postft/eval/surprise_c16t32__v11_split_test_e40/summary.json']]"
python -c "import json;[print(f, json.load(open(f))['surprise']['skip2_w32/avg']['overall']['block_pairwise']) for f in ['z_training/runs/v11_postft/eval/intphys1_sliding__intphys1_dev_e10/summary.json','z_training/runs/intphys1_postft/eval/intphys1_sliding__intphys1_dev_e40/summary.json']]"
# §2-1 knockout_hidden 조건별 Δ
python -c "import json;r=json.load(open('z_research/IntPhysGenV11_occlusion_timing_ablation/exp_results/knockout_hidden__v11_full_vith/results.json'));[print(c, round(v['acc']['mask->hidden@all']-v['acc']['clean_null'],2), round(v['acc']['mask->hidden_ctrl@all']-v['acc']['clean_null'],2)) for c,v in r['breakdown']['condition'].items()]"
```

산출물: `exp_results/{time_alignment,ctxenc_boundary_kinematics,ctxenc_boundary_identity_occlusion,ctxenc_subspace_overlap}/`. 각 폴더의 md 는 스크립트가 json 에서 생성한다 — 손으로 고치지 않는다.
