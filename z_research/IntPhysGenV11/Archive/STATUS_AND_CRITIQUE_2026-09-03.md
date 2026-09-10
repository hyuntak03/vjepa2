# 현행 상황과 비판 — "가림은 corner case 아닌가" (2026-09-03)

> 외부 피드백(교수님·종서님): **"event transition 에 occlusion 이 걸리는 것은 너무 corner case 다."**
> 사용자 본인도 동의. "안 되는 건 알았다. 그래서 왜 해야 되는데?" 가 안 없어진다.
> 이 문서는 (1) 지금까지 확립된 것을 한 곳에 모으고, (2) 그중 무엇이 corner case 이고 무엇이
> 일반 주장인지 가르고, (3) 스토리 후보와 확장 축을 적는다.
> `PAPER_STORY_2026-08-31.md` 는 **재검토 중**이다 — 폐기가 아니라 축을 옮기는 검토.

---

## 1. 확립된 것 (전부 `exp_results/` 에서 재검증된 수치)

### 1-1. 채점이 무너지는 조건은 좁다

v11_full 12조건 21,504 pair, `surprise_c16t32`, chance 50 (`IntPhysGenV11_occlusion_timing_ablation/README.md`):

| | visible | early | mid | **late** |
|---|---:|---:|---:|---:|
| vanish | 100 | 97.8 | 95.8 | **57.5** |
| shape | 84.3 | 78.1 | 76.0 | **55.4** |
| colour | 81.5 | 67.4 | 67.6 | 65.3 |

- 물체가 가려졌다가 **사건 전에 다시 보이면** (early/mid) vanish·shape 는 거의 회복된다
- 무너지는 것은 **가림이 문맥/미래 경계에 걸릴 때**(late) 뿐이고, k(가려진 장수)는 무관하다
- 같은 기록에 **sliding window 로 43.7pt 가 돌아오는 사례**가 있다 (v6aug static+occluded shape,
  fixed 46.88 → sliding max 90.62, CLAUDE.md §1-3). 즉 실패의 상당 부분이 **평가 프로토콜의 고정
  경계**와 얽혀 있다. 실제 사용에서 아무도 그 프레임에서 문맥을 끊지 않는다 — 이것이 "왜 해야
  되는데" 가 안 닫히는 근본 이유다
- colour 만 early 부터 떨어진다. 가림막 색·질감 가설은 v13 에서 기각됐고 원인은 미해결

### 1-2. 정보는 남아 있고, 자리는 late 에서만 바뀐다

- `p` self probe: 12조건 전부 ≥ 95 (shape/colour/env)
- 비가림 head 를 가림에 이식: `z`·`h` 93~100 유지, **`p` 만 late 에서 15~21** (early/mid 98~100)
- `h→p` 이식은 **가림이 없어도** static 95 / flat 79 / ramp 57 — `p` 와 `h` 는 애초에 같은 좌표계가 아니다

### 1-3. 채점 점수의 대부분은 물리와 무관하다 — corner case 가 **아닌** 결과

- base `|p − h|` ≈ 0.58 인데 margin(가능 vs 불가능 차)은 그 **0.08 ~ 3.7%** 다. 효과크기가 곧 정확도 (ρ 0.986)
- 방향 비대칭은 **고정된 물체 순서** 하나로 설명된다 (42방향 중 40, Hodge R² 0.82–0.91) 그리고
  그 순서는 **비가림 등가속에서 이미 R² 0.816** — 가림이 만든 게 아니다
- precision 폭 24 vs recall 폭 86 → 능력 차가 아니라 **과생산/과소생산** (외형 편향)
- vanish 50% 는 100(빈→물체) 과 0(물체→빈) 의 평균 — 점수는 "장면이 비어 있는 쪽" 을 늘 선호한다
- 토큰 시간 프로파일: 신호는 미래 첫 튜블릿(raw 48–51)에 몰리고 곧 사라진다. 채점은 사실상
  **경계 직후 두 프레임**으로 결정된다

### 1-4. 기전은 조건 특이적이지 않다 (knockout, 2026-09-03)

- 채점을 나르는 경로는 `mask→ctx` 하나. `mask→mask` −1.4, `ctx→ctx` +0.5
- 층 8 이 병목 (단일 층 −18, ±3 창은 층 8 포함 여부로 계단), 층 0–7 전부 끊어도 −10
- **곡선 모양이 타이밍·k·위반·운동에 무관** — 가림에서 다른 경로가 열리는 게 아니다
- 문맥을 못 읽은 예측은 chance 아래(38, visible/vanish 13) — 고정 예측이 "빈 장면" 쪽에 가깝다
- 대기 중: ±1 창 전수 (`ko_w1`), 가려진 문맥 프레임만 못 읽게 하기 (`ko_hidden`, 대조군 포함)

---

## 2. 무엇이 corner case 이고 무엇이 아닌가

| 주장 | 범위 | 판정 |
|---|---|---|
| "가림이 예측 경계에 닿으면 무너진다" | 물체 × 사건 순간 × 고정 경계 — 셋이 동시에 맞아야 한다 | **corner case.** 프로토콜과 얽혀 있고 sliding 으로 상당 부분 회복 |
| "`p` 의 자리가 바뀐다" | late 에서만 | 위와 같은 범위 |
| "정보는 안 없어진다" | 12조건 전부 | 일반 — 그러나 이것만으로는 논문이 안 된다 (기대값에 가깝다) |
| **"latent surprise 는 물리 검출기가 아니라 외형 일치 검출기다"** | 비가림 포함 전 조건 | **일반.** margin/base, 물체 순서, precision 편향, vanish 방향, 토큰 프로파일이 전부 여기에 모인다 |
| **"`p` 와 `h` 는 같은 좌표계가 아니고 그 불일치가 점수를 지배한다"** | 비가림 포함 | **일반.** 개입(정렬)의 근거 |
| "채점 경로는 `mask→ctx` 층 8 하나" | 전 조건 | 일반 — 단독으로는 관찰이지 주장이 아니다 |

**결론: 가림을 주인공에서 내려야 한다.** 가림 타이밍은 "정보 / 자리" 를 깨끗하게 분리해 보여주는
testbed 로는 여전히 좋다 (early/mid 가 완벽한 대조군이다). 주인공은 **점수가 무엇을 재는가**다.

---

## 3. "왜 해야 되는데" — use case 후보

1. **planning.** V-JEPA 2-AC 는 같은 latent L1 을 planning 의 energy 로 쓴다. surprise 가 외형 편향과
   좌표 불일치를 담으면 planning cost 도 그것을 물려받는다 → 평가지표의 문제가 **행동**의 문제가 된다.
   실측 가능: 물체 모양만 바꾼 두 goal 에 대한 energy 차이가 물리적 차이보다 큰가
2. **벤치마크 갭 설명.** IntPhys 1 은 되는데 IntPhys 2 는 무너진다. IntPhys 2 가 물체·장면 다양성이
   크다면 "외형 편향이 커지는 방향" 이고 우리 분석이 그 갭을 설명한다. **IntPhys 2 의 항목별 실측치를
   먼저 봐야 한다** (§10 의 3번, 아직 안 함)
3. **평가 방법론.** latent surprise 로 world model 을 평가하는 모든 논문에 해당 — "이 점수는 물리를
   재지 않는다. 정렬 층 하나로 이만큼 바뀐다" 는 재학습 없는 진단·보정 도구
4. **anomaly / plausibility 검출** 응용은 (1)–(3) 의 따름이지 독립 use case 는 아니다

use case 가 (1) 이나 (2) 에서 실증되면 "왜" 가 닫힌다. 안 되면 (3) 만 남고 그건 workshop 급이다.

---

## 4. 확장 축 — corner case 를 벗어나는 방법

"정보는 있는데 채점이 못 잡는다" 가 가림 밖에서도 재현되는지를 본다. **재현되면 일반화되고, 안 되면
"가림이 특별하다" 자체가 결과**다 — 어느 쪽이어도 beat 가 선다.

| 축 | 무엇을 바꾸나 | 이미 있는 것 | 비용 |
|---|---|---|---:|
| **외형 다양성** | 색·질감·크기·물체 수를 축으로. 순서 편향이 다양성과 함께 커지는가 | v11 7모양 × 8색, 순서 R² | 렌더 1세트 |
| **사건 종류** | 접촉·지지(support)·연속성(continuity)·궤적 — IntPhys 2 / Physion 계열 사건 | v11 은 vanish/shape/colour 만 | 렌더 + 채점 |
| **경계 위치** | 사건과 문맥 경계의 거리를 축으로 (late 를 연속화). sliding 과의 관계를 정면으로 | early/mid/late 3점 | 재채점만 |
| **실사** | IntPhys 2, Physion 에서 probe·margin 분해 재현 | IntPhys1 dev 180쌍 | 추출 |
| **planning energy** | V-JEPA 2-AC energy 에서 외형 편향 재현 | 없음 | 구현 |

⚠️ **"경계 위치" 축이 가장 싸고 가장 위험하다** — 결과가 "sliding 이면 다 된다" 이면 가림 스토리는
그 자리에서 끝난다. 그래도 먼저 봐야 한다. 모르고 넘어가면 리뷰어가 찾는다.

---

## 5. 스토리 후보

| | 축 | 강점 | 리스크 |
|---|---|---|---|
| A (현행) | 가림 → 정보/자리 → frozen 개입 | 실험이 다 있다 | **corner case.** 프로토콜 교란 |
| **B (제안)** | **latent surprise 는 외형 일치 검출기다** → 정보/자리 분리(가림은 testbed) → 정렬 개입 → IntPhys 2 갭 | 비가림 결과가 전부 여기 모인다. use case (1)(2) 로 닫힌다 | IntPhys 2 실측·항목별 분석이 없다. 개입이 아직 없다 |
| C | 평가 방법론 비판 (fixed vs sliding, matched pairing, appearance bias) | 싸다 | 기여가 "지적" 에 그친다 |

**B 를 추천한다.** A 의 실험은 B 안에서 전부 재사용된다 (가림 = 정보/자리를 가르는 도구).
B 가 서려면 §3 의 (2) 와 §4 의 "경계 위치" 가 먼저다.

---

## 6. 선행 연구 — 찾아봐야 할 것 (⚠️ 전부 **확인 필요**, 기억으로 적었다)

| 갈래 | 후보 | 왜 |
|---|---|---|
| latent surprise 로 물리 평가 | Garrido et al. 2025 (V-JEPA, IntPhys/GRASP/InfLevel) · Bordes et al. 2025 (IntPhys 2) · V-JEPA 2 (Assran et al. 2025) | 우리 채점의 원형. IntPhys 2 의 항목별 수치 |
| 가림·object permanence 벤치마크 | Riochet et al. IntPhys 2019 · Piloto et al. 2022 (PLATO) · ADEPT (Smith et al. 2019) · InfLevel · GRASP | "가림은 corner case" 에 대한 반론 재료 — 발달심리의 표준 패러다임이 가림이다 |
| 사건 종류 확장 | Physion / Physion++ · CLEVRER · CRAFT | 접촉·지지·연속성 사건의 정의 |
| 생성 모델의 물리 평가 | Physics-IQ (Motamed et al.) · VideoPhy · "How far is video generation from world model" | latent 이 아닌 픽셀 쪽 평가와의 대비 |
| predictor 해석 | "Interpreting Physics in Video World Models" (attention distance) | 우리 knockout·attention 과 직접 비교 |
| planning energy | V-JEPA 2-AC | use case (1) |

이 표는 **검색으로 채워야 한다.** 제목·연도·주장을 확인하기 전에는 인용하지 않는다.

### 6-1. 확인한 것 (2026-09-03 웹 검색·초록 기준 — 본문은 아직 안 읽음)

| 논문 | 확인된 주장 | 우리와의 관계 |
|---|---|---|
| **V-JEPA 2** (arXiv 2506.09985) | 초록의 selling point 는 understanding(SSv2 77.3) · prediction(EK-100 R@5 39.7) · **planning**(V-JEPA 2-AC, Franka zero-shot pick-and-place). 초록에 intuitive physics 언급 없음 | 공격 지점은 "물리" 가 아니라 **latent 거리 = planning energy** 다 |
| **IntPhys 2** (Bordes et al., arXiv 2506.09849) | V-JEPA 2 main set **57.51%** (IntPhys 1 은 98.30). 4원리: permanence · immutability · continuity · solidity. 대부분 모델 chance. 저자 설명: "많은 샘플이 ~200 프레임 기억을 요구", moving camera 에서 더 낫다 | **우리가 반박할 수 있는 설명이다** — 우리 데이터의 실패는 기억과 무관 (k 무관, early/mid 회복). 단 200 프레임 기억 실패가 *따로* 있을 수 있다 |
| **V-JEPA 2.1** (2026-03, ViT-G 2B, 384 res) | dense predictive loss(전 토큰) · deep self-supervision · 모달리티별 tokenizer · 163M 데이터. grasping +20%, planning 128→8 steps. "denser features". IntPhys 언급 없음 | **frozen 으로 검사 가능한 예측**: dense loss 가 좌표 불일치·외형 편향을 줄이는가 |
| **"Do Video Foundation Models Understand Intuitive Physics? A Layerwise Probing Analysis"** (arXiv 2606.09646, 2026-06) | V-JEPA · VideoMAE · LTX-Video 를 IntPhys 2 · MVP 에서 층별 probing. "physics-relevant information is weakest in early layers and becomes most accessible at intermediate-to-late depth". V-JEPA 최강 | ⚠️ **겹침 위험.** "정보는 있다" beat 는 이 논문이 먼저 했다. 우리 차별점은 **같은 클립에서 정보(probe)와 채점(surprise)이 갈린다** 는 것과 그 갈림의 기전·개입 |
| **"Interpreting Physics in Video World Models"** (arXiv 2602.07050) | encoder 의 layerwise probing · subspace geometry · patch decoding · attention ablation. "Physics Emergence Zone" (중간 깊이 전이), 방향은 고차원 원형 population code | ⚠️ 겹침: attention ablation·subspace. 그쪽은 **encoder**, 우리는 **predictor + 채점 층** |

**함의**: "정보가 있다" 와 "층별 어디에 있다" 는 이미 두 편이 냈다. 남는 자리는
(a) **정보와 채점의 괴리** 그 자체 (probe 100 / 채점 3 을 같은 클립에서), (b) 그 괴리의 원인이 latent 거리의
구성(좌표 불일치 + 외형 편향)이라는 것, (c) 그것이 **planning energy 로 이어진다**는 것, (d) frozen 개입.

---

## 7. 다음 단계 (제안 순서)

1. **IntPhys 2 실측** — V-JEPA 2 ViT-H 의 항목별 정확도. 문헌이면 문헌, 없으면 돌린다 (§10-3)
2. **경계 위치 축** — v11_full 캐시로 sliding 채점 재현, 사건-경계 거리별 정확도 (재채점만)
3. **외형 편향의 비가림 정량화** — 순서 편향·precision 폭을 조건별 표 하나로 (있는 산출물)
4. 선행 연구 표 채우기
5. 그 다음에 정렬 개입 (Procrustes) — B 에서는 이것이 "왜" 를 닫는 실험이다

## 8. 하네스에 추가하는 질문 (2026-09-03)

7. **이 조건이 실제 사용·벤치마크에서 얼마나 흔한가 — corner case 인가.** 그리고 결과가 **어느 use case 에
   닿는가** (planning / 벤치마크 갭 / 평가 방법론). 못 대면 그 실험은 스토리를 못 세운다.

## 재현

수치는 새로 만든 것이 없다. 출처:
- 채점·probing·기전: `z_research/IntPhysGenV11_occlusion_timing_ablation/README.md` §재현
- 순서·precision/recall: `z_research/IntPhysGenV11/Archive/surprising_score/RESULTS_2026-08-30.md`
- knockout: `analysis/attention/knockout/README.md` §재현, `IntPhysGenV11_occlusion_timing_ablation/exp_results/knockout__v11_full_vith/`
- sliding vs fixed: `CLAUDE.md` §1-3 (v6aug, `z_exp/.../summary.json`)
