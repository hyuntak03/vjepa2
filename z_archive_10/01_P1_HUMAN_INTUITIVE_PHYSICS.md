# 01 — P1: 사람의 intuitive physics · 관측 학습 · 적응 (문헌)

> **결론** — P1 은 우리 측정 없이 **문헌으로만** 서는 문단이다. 그런데 레포의 문헌 조사 문서 셋에는 **영유아 · 동물 · 적응 문헌이 한 편도 없다**. 지금 있는 것은 사용자가 쓴 인용 목록 (9 편, 원문 확인 기록 없음) 과, 원문을 확인한 계산 모델 · 배경 인용 몇 편 (Craik 1943 · LeCun 2022 · Wolpert & Kawato 1998 등) 뿐이다.
> **근거** — (1) 정본 §2-1: P1 의 근거 상태는 "**문헌** — 인용 확인 필요", 그리고 "'동물' 과 '새 환경 · task 적응' 은 아직 인용이 없다 … 실물 확인 전 인용 금지" (`auto_research/paper/PAPER_STORY_2026-10-08.md` §2 · §2-1).
> (2) 영유아 인용 5 편 (사용자 인용 9 편 중) 은 전부 `auto_research/paper/NARRATIVE_v2_USER_2026-09-29.md` §A 의 사용자 원문에서만 나온다 (9 편 중 문헌 문서에도 나오는 것은 Piloto 2022 = PLATO 하나, URL 만). 문헌 문서 셋 (`auto_research/Archive/LITERATURE_*`) 의 묶음 판정은 "P1 — 세 문서 모두 공백" 이다 (G04a 묶음 요약 (b)).
> (3) 원문을 확인한 것: LeCun 2022 "(1) to estimate missing information … (2) to predict plausible future states" (✅ Meta 재게재, 절/쪽 ❓) · Craik 1943 (✅ 2 차 전사) (`auto_research/Archive/LITERATURE_PREDICTOR_ROLE_2026-09-29.md` §8).
> **미결** — 영유아 5 편 · 동물 · 적응 · "가장 먼저" 를 원문으로 확인해야 한다 (정본 §5 #10, CLAUDE.md §10 #10). 담당은 Claude 이고, 아직 하지 않았다.
> 이 장이 받치는 논문 문장: **P1** "사람은 intuitive physics (사라지지 않음 · 중력) 덕분에 plausible future 를 쉽게 상상한다. 그것은 영유아 · 동물이 관측만으로 가장 먼저 갖추고, 새 환경 · task 적응의 바탕이다."  ·  원재료: G04a (문헌 셋) · G05 (논문 스토리 판들) 추출문 → 원 문서는 각 절 끝 `출처`.

---

## 1. P1 원문과 이 문단이 세우려는 세 주장

**사용자 원문 (그대로, 정본 §0)**

> **P1 : 사람은 intuitive physics 덕분에 plausible future를 쉽게 상상, intuitive physics는 observation 갖춰짐**
> Physical world에서, 사람은 앞으로 어떤 일이 일어날지, 물리적으로 가능한 미래들을 쉽게 상상할 수 있다.
> 이것은, (물체가 갑자기 사라지지 않는다, .. 중력이 있다) 와 같은 intuitive physics를 알고 있기 때문이다.
> 이런 intuitive physics는 영유아, 동물들에게 가장 초기에 observation만으로 갖춰지는 능력이며,
> 이 능력 덕분에 intelligent system은 새로운 환경이나 task에 대해 빨리 적응하고 이해할 수 있다.

**세 주장으로 나눈다** (아래 각 주장의 인용 상태는 §2 표)

| # | 주장 | 문장의 핵심어 | 인용 상태 (요약) | 이 주장이 이어지는 곳 |
|---|---|---|---|---|
| (a) | 사람은 intuitive physics 로 plausible future 를 상상한다 | "물리적으로 가능한 미래들을 쉽게 상상" · 예시 = 사라지지 않음 · 중력 | 배경 문헌 있음 (Craik ✅ 2 차 · Battaglia 2013 등급 문서에 없음 · Clark ✅ 초록). **intuitive physics 를 직접 다룬 인지과학 인용은 Battaglia 2013 하나** | P2 의 gap ("사람은 쉽게 상상하는데 world model 은 못 한다") · P3 의 판정 대상 |
| (b) | 영유아 · 동물이 **관측만으로 · 가장 먼저** 갖춘다 | "영유아, 동물" · "가장 초기에" · "observation만으로" | 영유아 = 사용자 목록 5 편만 (원문 확인 기록 없음). **동물 = 인용 없음.** **"가장 먼저" = 인용 없음** | 목표 ("action 없이 관측만으로 배우는 predictor") 의 동기 |
| (c) | 그것이 새 환경 · task 적응의 바탕이다 | "새로운 환경이나 task 에 대해 빨리 적응" | **인용 없음.** 후보 Lake et al. 2017 하나 (미확인) | 목표 · AC subsection (`09b`, 초안) 의 순서 논리 |

⚠️ P1 의 두 예시는 우리 조작적 정의와 1:1 로 맞춘다 (정본 §4-9 "P1 의 두 예시가 영속 · 중력과 1:1"):
"물체가 갑자기 사라지지 않는다" = **4 영속** · "중력이 있다" = **3a 추세 (가속 · 꼭대기에서 되돌아옴) + 3b 사건 (받침 끝 낙하)** (`z_research/context_encoder_analysis/Archive/STATE_EVOLUTION_CAPABILITIES_2026-09-19.md` §1; 3a/3b 분할은 2026-10-08, 정본 §3).

출처: `auto_research/paper/PAPER_STORY_2026-10-08.md` §0 · §2 · §3 · §4-9 · `z_research/context_encoder_analysis/Archive/STATE_EVOLUTION_CAPABILITIES_2026-09-19.md` §1

---

## 2. 주장별 인용 후보 표

**확인 상태 표기** (G04a 가 원 문서 표기를 그대로 옮긴 것):
✅ = 레포 문서가 원문 확인 (verbatim · pdftotext · 본문) · ✅ 초록 / ✅ 2 차 전사 = 그 수준까지만 · ◐ = 요약만 (인용 전 재확인) · ❓ = 미확인 ·
**"기록 없음"** = 레포 어느 문서도 원문 확인 기록이 없다 → **실물 확인 전 인용 금지**.
등급은 전부 **문헌** (우리가 안 잼). 서지는 원 문서에 적힌 그대로만 옮겼다 — 원 문서에 제목이 없으면 "제목 원 문서에 없음" 이라 적었다.

### 2-1. (a) 사람은 intuitive physics 로 plausible future 를 상상한다

| 제목 | 저자/연도 | 한 줄 요지 (원 문서 표현) | 우리 문장과의 관계 | 확인 상태 | 출처 |
|---|---|---|---|---|---|
| The Nature of Explanation (p. 61) | Craik 1943 | 생물이 외부 현실의 'small-scale model' 을 지니면 "react to future situations before they arise" | 지지 — 서론 첫 문장 후보 (V-JEPA 2 도 Craik 1967 인용) | ✅ 2 차 전사 (jhamrick.github.io) | `LITERATURE_PREDICTOR_ROLE_2026-09-29.md` §8 |
| Simulation as an engine of physical scene understanding | Battaglia et al. PNAS 2013 | 직관물리 = 정신 시뮬레이션 (인지과학) | 지지 — **이 레포 문헌 문서에서 유일한 인지과학 직관물리 인용** | 등급 문서에 없음 (배경 목록에만) | `LITERATURE_POSITIONING_2026-09-25.md` §2 · §6-4 |
| (제목 원 문서에 없음) BBS 36(3) | Clark 2013 | 뇌 = prediction machine | 지지 (배경) | ✅ 초록 (Cambridge Core) | `LITERATURE_PREDICTOR_ROLE_2026-09-29.md` §8 |
| (제목 원 문서에 없음) Nat. Neurosci. 2:79 | Rao & Ballard 1999 | 피드백 = 예측, 피드포워드 = 잔차 | 지지 (배경) | ◐ 검색 요약 문장 — 인용 전 원문 확인 | 〃 |
| (제목 원 문서에 없음) Neural Networks 11 | Wolpert & Kawato 1998 | forward model 이 현재 상태 + 운동 명령으로 다음 상태 예측 | 지지 (predictor 정의) | ✅ pdftotext | 〃 |
| (제목 원 문서에 없음) Psychol. Rev. 55(4) | Tolman 1948 | 동물의 "tentative, cognitive-like map" | 지지 (배경 · **동물** 쪽으로 쓸 수 있는 유일한 확인 인용 — 단 intuitive physics 가 아니라 인지 지도) | ✅ 전사 (CSULB) | 〃 |
| (제목 원 문서에 없음) | Köster et al. 2020 · Ellis et al. 2021 | (사용자 문단 2) predictive processing — "예측과 관찰을 연결하는 학습이 세계 지식 형성의 한 원리" | 지지 (사용자가 붙인 자리) | **기록 없음 → 인용 금지** | `NARRATIVE_v2_USER_2026-09-29.md` §A 문단 2 |

### 2-2. (b) 영유아 · 동물이 관측만으로 가장 먼저 갖춘다

| 제목 | 저자/연도 | 한 줄 요지 (원 문서 표현) | 우리 문장과의 관계 | 확인 상태 | 출처 |
|---|---|---|---|---|---|
| (제목 원 문서에 없음) | Johnson et al. 2003 | (사용자 문단 1) 영아기부터 관찰로 앞일의 지식을 쌓는다 · "조작 없이 관찰 경험만으로 예측 능력이 발달" | 지지 ("관측만으로") | **기록 없음 → 인용 금지** | `NARRATIVE_v2_USER_2026-09-29.md` §A 문단 1 · 정본 §2-1 |
| (제목 원 문서에 없음) | Kochukhova & Gredebäck 2007 | 〃 (사용자 문단 1) | 지지 | **기록 없음 → 인용 금지** | 〃 |
| (제목 원 문서에 없음) | Baillargeon & DeJong 2017 | 〃 (사용자 문단 1) | 지지 | **기록 없음 → 인용 금지** | 〃 |
| (제목 원 문서에 없음) | Kaufman et al. 2005 | (사용자 문단 3) "가려져도 존재는 유지" | 지지 (사라지지 않음 = 영속) | **기록 없음 → 인용 금지** | `NARRATIVE_v2_USER_2026-09-29.md` §A 문단 3 · 정본 §2-1 |
| (제목 원 문서에 없음) | Rosander & von Hofsten 2004 | (사용자 문단 3) "움직임은 가림 동안에도 진행" | 지지 (가려진 운동 지속 = 영속 + 지속) | **기록 없음 → 인용 금지** | 〃 |
| (제목 원 문서에 없음) — "갓 부화한 병아리의 가려진 물체 추론" | Vallortigara 그룹 (연도 원 문서에 없음) | 동물 · 경험 거의 없이 가려진 물체 추론 | **후보** ("동물" 의 유일한 후보) | **기록 없음 → 인용 금지** | 정본 §2-1 |
| Benchmarking Progress to Infant-Level Physical Reasoning in AI (InfLevel) | Weihs et al. TMLR 2022 | continuity · solidity · gravity VoE, 당시 video 모델 chance 근처 | 다리 — 영아 수준을 표제로 쓴 ML 벤치 (P2 에서 평가 판으로 쓴다) | ◐ (P-29 표기) | `LITERATURE_POSITIONING_2026-09-25.md` 논문 목록 · `LITERATURE_PREDICTOR_ROLE_2026-09-29.md` |
| 영아 VoE 계보: Baillargeon 1985 · 1987 · Spelke 1990 (continuity) · Kahneman, Treisman & Gibbs 1992 (object files) | 원 문서 표기 그대로 | 네 능력 정의의 인지과학 계보 — "계보를 인용해야 자의성 비판을 피한다" | 지지 (정의의 계보) | **기록 없음** (리뷰 문서가 이름만 적음) | `REVIEW_CVPR_ORAL_2026-09-25.md` §2 N 행 · §6 |

### 2-3. (c) 새 환경 · task 적응의 바탕이다

| 제목 | 저자/연도 | 한 줄 요지 | 우리 문장과의 관계 | 확인 상태 | 출처 |
|---|---|---|---|---|---|
| (제목 원 문서에 없음) | Lake et al. 2017 | "intuitive physics 를 학습의 start-up software 로" | **후보** (적응의 유일한 후보) | **기록 없음 → 인용 금지** | 정본 §2-1 |
| General agents need world models (2506.01622) | ICML 2025 | 여러 걸음 과제를 일반화하는 에이전트는 예측 모형을 담는다 | 간접 — world model 의 중요성 (P2a 용). **사람 · 동물의 적응이 아니라 에이전트 이론** | 등급 문서에 없음 | `LITERATURE_POSITIONING_2026-09-25.md` §6-4 (use case 이론) · 영역 R1 논문 목록 |

⚠️ (c) 에는 사람 · 동물 쪽 근거가 하나도 없다. 위 ICML 2025 는 ML 에이전트 이론이라 "사람 · 동물이 빨리 적응한다" 를 받치지 못한다.

출처: G04a 문서별 표 (`auto_research/Archive/LITERATURE_PREDICTOR_ROLE_2026-09-29.md` §8 · `LITERATURE_STATE_WORLD_MODELS_2026-09-26.md` · `LITERATURE_POSITIONING_2026-09-25.md` §2 · §6-4 · 논문 목록) · `auto_research/paper/NARRATIVE_v2_USER_2026-09-29.md` §A · `auto_research/paper/PAPER_STORY_2026-10-08.md` §2-1 · `auto_research/Archive/REVIEW_CVPR_ORAL_2026-09-25.md` §2 · §6

---

## 3. 빈칸 — P1 에서 인용이 없는 것 (크게)

> ⚠️ **레포의 문헌 조사 문서 셋 (P-25 1,525 줄 · S-26 270 줄 · P-29 397 줄) 에는 영유아 발달 · 동물 문헌이 한 편도 없다.**
> 세 문서는 world model · VoE 벤치마크 · predictor 역할을 조사했고, P1 이 정본이 된 날 (10-08) 보다 먼저 (09-25 ~ 09-29) 쓰였다. P1 이 인간 비유를 intro 첫 문단 전체로 키운 뒤 P1 을 따로 조사한 적은 없다.

| 빈칸 | 지금 있는 것 | 없는 것 | 필요한 확인 | 출처 |
|---|---|---|---|---|
| **영유아 (b)** | 사용자 목록 5 편 (Johnson 2003 · Kochukhova & Gredebäck 2007 · Baillargeon & DeJong 2017 · Kaufman 2005 · Rosander & von Hofsten 2004) | 제목 · 원문 · 각 논문이 실제로 "관측만으로" 를 보였는지 | 5 편 실물 대조, 각 편이 받칠 수 있는 문장 한 줄씩 | 정본 §2-1 · NARRATIVE_v2 §A |
| **동물 (b)** | 후보 이름 하나 (Vallortigara 그룹, 병아리) · Tolman 1948 (✅, 단 인지 지도) | 동물의 intuitive physics 문헌 0 편 | 후보 실물 확인. 못 찾으면 P1 에서 "동물" 을 빼거나 줄인다 | 정본 §2-1 |
| **"가장 먼저 · 가장 초기에" (b)** | 없음 | 다른 능력보다 **먼저** 라는 비교를 받치는 인용 0 편 | 별도 확인. 못 찾으면 "초기에" 로 약하게 쓴다 | 정본 §0 원문 · 이 장 §1 |
| **"관측만으로" (b)** | 사용자 문단 1 ("조작 없이 관찰 경험만으로") | 원문 확인 · 반론 검토 | §3-1 의 긴장 (SAYCam) 과 같이 읽는다 | NARRATIVE_v2 §A 문단 1 |
| **적응 (c)** | 후보 Lake et al. 2017 하나 | 사람 · 동물 쪽 인용 0 편 | 실물 확인 | 정본 §2-1 |
| **영아 VoE 해석 논쟁** | 리뷰 문서가 이름만 적음 (§3-1) | 원문 확인 | 리뷰어가 칠 수 있는 자리 — 확인하고 P1 문장 강도를 정한다 | `REVIEW_CVPR_ORAL_2026-09-25.md` §4 |

정본의 할 일 표가 이 빈칸을 가리킨다 — §5 #10 "문헌 확인 — P1 (동물 · 적응)", 담당 Claude, 비용 "문헌" (`PAPER_STORY_2026-10-08.md` §5). CLAUDE.md §10 #10 도 같은 줄이다.

### 3-1. P1 과 긴장하는 문헌 (반례 · 약화)

| 제목 | 저자/연도 | 원 문서 요지 | P1 의 어느 주장과 긴장하나 | 확인 상태 | 출처 |
|---|---|---|---|---|---|
| Opinion: Learning Intuitive Physics May Require More than Visual Data (2512.06232) | Su, Legris, Gureckis, Ren 2025 | SAYCam (아동 시점) 데이터로 V-JEPA 를 사전학습해도 IntPhys 2 가 오르지 않는다. 시각 데이터의 양 · 분포만으로는 부족하다 | (b) "관측만으로" — 기계에서 아동 시점 관측 데이터만 바꿔서는 안 된다 | ✅ | `LITERATURE_STATE_WORLD_MODELS_2026-09-26.md` §4-1 · `LITERATURE_POSITIONING_2026-09-25.md` 논문 목록 |
| (제목 원 문서에 없음) 2310.19943 | Schulze Buschoff, Schulz, Binz ICML 2023 | 생성 모델이 물리 과정을 잘 예측해도 학습 궤적이 아동 발달과 다르다 | (b) 의 "사람처럼 배운다" 비유 | ✅ | `LITERATURE_STATE_WORLD_MODELS_2026-09-26.md` §4-1 |
| 영아 VoE 논쟁: Bogartz, Shinskey & Speaker 1997 · Haith 1998 · Schöner & Thelen 2006 · NHB 2024 메타분석 (Perceptual and conceptual novelty independently guide infant looking behaviour) · Margoni, Surian & Baillargeon 2024 (Psychological Review, VoE overview) | 원 문서 표기 그대로 | "더 오래 보는 것은 개념적 위반이 아니라 지각적 새로움 때문일 수 있다". 2024 메타분석은 지각적 새로움과 개념적 새로움이 **따로** 시선을 이끈다고 정리 | (b) 영아 VoE 를 "intuitive physics 를 갖췄다" 의 증거로 읽는 것 | **기록 없음** (리뷰 문서의 요약 · 링크만) | `REVIEW_CVPR_ORAL_2026-09-25.md` §4 "문헌 계보" · §6 · 논문 목록 |

⚠️ Su et al. 2025 의 읽는 법 — 원 문서는 "'데이터를 바꿔도 안 된다 → 구조 (predictor) 를 바꿔야 한다' 는 논지의 간접 지지" 로 읽었다 (`LITERATURE_POSITIONING` 논문 목록). 정본 금지 표현에 따라 **목적함수 원인으로는 쓰지 않는다** — discussion 한 문단까지 (`LITERATURE_STATE_WORLD_MODELS` 단서 · 정본 §7).
⚠️ 영아 VoE 논쟁은 P2 쪽에서 쓸모가 있다: 리뷰 문서는 "기계 VoE (Garrido 2025) 는 그 논쟁을 그대로 물려받았다. 복사 기준선은 그 **기계판 친숙도 (familiarity) 대조**다" 라고 적었다 (`REVIEW_CVPR_ORAL_2026-09-25.md` §4). 같은 논쟁이 P1 에는 약점, P2 에는 근거가 된다 — 쓰려면 원문 확인이 먼저다.
⚠️ (작성자 메모, 원 문서에 없음) 영아의 물체 개념이 타고난 것인가 관측으로 배운 것인가는 발달심리의 오랜 쟁점이다. "관측만으로 갖춰진다" 는 그중 한쪽에 서는 문장인데, 레포 문서 중 이 쟁점을 다룬 것은 없다. §3 의 영유아 5 편을 확인할 때 같이 본다.

출처: `auto_research/Archive/LITERATURE_STATE_WORLD_MODELS_2026-09-26.md` · `auto_research/Archive/LITERATURE_POSITIONING_2026-09-25.md` · `auto_research/Archive/REVIEW_CVPR_ORAL_2026-09-25.md` §4 · §6 · `auto_research/paper/PAPER_STORY_2026-10-08.md` §5 · §7

---

## 4. 계산 모델 · 관측 학습 — P3 의 "안 보이는 것 / 없는 것" 과의 대응

### 4-1. 인용 후보

| 제목 | 저자/연도 | 원 문서 요지 | 우리와의 관계 | 확인 상태 | 출처 |
|---|---|---|---|---|---|
| A Path Towards Autonomous Machine Intelligence (v0.9.2, OpenReview BZ5a1r-kVsf, §3 모듈 설명) | LeCun 2022 | "The world model module … Its role is twofold: (1) to estimate missing information about the state of the world not provided by perception, and (2) to predict plausible future states of the world." / world model = "a kind of simulator of the part of the world relevant to the task at hand." | **지지 — 기능 분리의 정본 인용.** 인지과학 문단과 ML 을 한 문장으로 잇는다 (§4-2) | ✅ Meta AI blog 재게재 verbatim · **절/쪽 ❓** (OpenReview 인증 벽) | `LITERATURE_PREDICTOR_ROLE_2026-09-29.md` §8 · 단서 |
| Intuitive physics learning in a deep-learning model inspired by developmental psychology (PLATO) | Piloto et al. Nature Human Behaviour 2022 | 객체 수준 표현과 추적으로 VoE 를 학습, 발달심리에서 영감. "객체 표현이 결정적이다" | 지지 (P1 의 ML 쪽 다리, 관측 학습) — ⚠️ **object-centric slot.** 우리는 slot 없이 장면 단위라 그 차이를 방법 절에서 적는다. 반론 근거 ("patch 로는 부족") 도 된다 | 등급 문서에 없음 (URL 만) | 정본 §2-1 · `NARRATIVE_v2` §B-5 · `LITERATURE_POSITIONING` 논문 목록 · `REVIEW_CVPR_ORAL` §6 ("object-centric 이 VoE 에 필요하다는 반대 입장") |
| Modeling Expectation Violation in Intuitive Physics with Coarse Probabilistic Object Representations (ADEPT) | Smith et al. NeurIPS 2019 | 확률적 물리 시뮬레이션 + particle filter 로 '가림 너머로 이어지는 기대' 를 만든다. surprise 의 원 정의 = 진화한 belief 대비 관측의 불일치 | 지지 — "문맥 대비 변화 감지 (copy) 는 VoE 가 의도한 측정이 아니다" (P2 복사 기준선의 근거로 더 쓰인다) | 등급 문서에 없음 (URL 만) | `LITERATURE_POSITIONING_2026-09-25.md` §1 · §2-7 · 영역 R1 논문 목록 · 저자명은 `REVIEW_CVPR_ORAL_2026-09-25.md` §6 ("ADEPT (Smith 2019)") |
| World Models | Ha & Schmidhuber 2018 | "the role of the M model is to predict the future" — V = 압축 · M = 미래 예측 | 지지 (encoder / predictor 분업의 원형) | ✅ | `LITERATURE_PREDICTOR_ROLE_2026-09-29.md` §8 |
| PredNet (1605.08104) | Lotter, Kreiman, Cox ICLR 2017 | 미래 프레임 예측을 "unsupervised learning rule for learning about the structure of the visual world" 로 | 배경 (관측만으로 배운다의 ML 판) | ✅ 초록 | 〃 |
| (제목 원 문서에 없음) | Baek et al. 2025 | (사용자 문단 2) 계산 모델 | 사용자가 붙인 자리 | **기록 없음 → 인용 금지** | `NARRATIVE_v2_USER_2026-09-29.md` §A 문단 2 |
| (제목 원 문서에 없음) | Lake et al. 2017 | "intuitive physics 를 학습의 start-up software 로" | 후보 (§2-3) | **기록 없음 → 인용 금지** | 정본 §2-1 |

### 4-2. LeCun 2022 의 두 역할 ↔ P3 의 세 갈래

정본은 LeCun 2022 를 "P3 의 '안 보이는 것 / 없는 것' 과 그대로 겹친다" 고 적었다 (정본 §2-1). 옛 판 (09-29) 은 같은 문장을 permanence · reach 에 붙였다. 두 대응을 나란히 둔다.

| LeCun 2022 역할 | 정본 (10-08) 대응 | 옛 판 (09-29) 대응 | P3 측정 (요지만 — 수치 · 조건 · 등급은 해당 장) | 네 능력 |
|---|---|---|---|---|
| (1) "estimate missing information about the state of the world not provided by perception" | **안 보이는 것** — 마지막 관측에 가려진 물체 | permanence | v11 문맥 끝 가림 vanish **물체→빈 0 %** (빈→물체 100, 평균 50) — CLAUDE.md §5-1 · 05 장 | 4 영속 |
| (2) "predict plausible future states of the world" | **없는 것** — 아직 일어나지 않은 가속의 누적 · 낙하 · 되돌아옴 | reach | flat_a p 28 ≈ 등속 32 vs 진실 54 · `v11_realistic_ledge` **0 / 120** — CLAUDE.md §0 · §5-6 · 06 장 | 3a 추세 · 3b 사건 |
| (2) 의 일부 | **보이는 것** — 1 차 운동으로 짧게 외삽 (2–3 칸) | reach | 마지막 관측 자리에서 약 2–3 칸 (40–60 px) 안에서만 유지 — CLAUDE.md §5-5 · 04 장 | 2 지속 |
| (perception 역할: "estimates the current state of the world") | **encoder 에는 정보가 있다** | encoder 충분 | z → 문맥 끝 속도 R² 0.996 / 0.989 — CLAUDE.md §5-5 · 07 장 | 1 읽기 |

⚠️ 이 대응은 **해석**이다 (등급: 문헌 문장 자체는 ✅, 대응은 우리 문장). LeCun 의 (1) 은 "perception 이 주지 않은 정보" 라서, 가림 (안 보이는 것) 만이 아니라 아직 관측된 적 없는 가속도 · 사건 (없는 것) 까지 덮는다고 읽을 수도 있다. 정본은 (1)↔안 보이는 것 · (2)↔없는 것으로 나누었지만, 원 문서들은 이 경계를 따로 논증하지 않았다.
⚠️ 옛 판의 대응 (permanence · reach · closure) 은 09-29 스토리 소속이다. 문헌 표는 유효하지만, 그 틀의 "state 를 진화시키지 않는다 · 기억" 문장은 **현재 금지** (memory/state 프레임, 09-30) (G04a 단서).
⚠️ 위 표의 P3 수치는 이 장이 재검증하지 않았다. 조건 · n · 등급 · 원 문서는 04–07 장에서 확인한다. 여기 둔 이유는 P1 인용이 P3 의 어느 칸과 맞물리는지 보이기 위해서다.

출처: `auto_research/Archive/LITERATURE_PREDICTOR_ROLE_2026-09-29.md` §0 · §8 · 단서 · `auto_research/paper/NARRATIVE_v2_USER_2026-09-29.md` §B-5 · `auto_research/paper/PAPER_STORY_2026-10-08.md` §2-1 · §3 · CLAUDE.md §0 · §5-1 · §5-5 · §5-6

---

## 5. 인간 비유를 어디까지 쓰나

| 시점 | 규칙 | 출처 |
|---|---|---|
| 09-25 (대체된 판) | "인간 · 유아 비유는 첫 문장과 discussion 에만. 본문 기준은 reach · permanence · closure." 당시 목표 문장은 "더 인간 같은 latent world model" · "인간 (유아) 은 action 없이도 근미래를 anticipate 한다" | `auto_research/paper/DIRECTION_2026-09-25.md` §1 · §5 |
| 09-29 (대체된 판) | 사용자 서사 7 문단 중 1–3 이 인지과학. 검토 판정: "1–2 는 intro 첫 문단 하나로 압축 (CVPR 독자)" · 3 의 Read · persistence · transition = 우리 네 능력 · "LeCun 2022 를 3 에 인용하면 인지과학과 ML 이 한 문장으로 붙는다" | `auto_research/paper/NARRATIVE_v2_USER_2026-09-29.md` §A · §B-5 |
| **10-08 (정본, 현행)** | **인간 비유 = P1 전체 (intro 첫 문단).** 옛 "첫 문장과 discussion 에만" 을 바꾼다. **본문 판정은 여전히 측정 가능한 정의로** — 네 능력 (1 읽기 · 2 지속 · 3a 추세 · 3b 사건 · 4 영속) 을 intuitive physics 의 조작적 정의로 쓴다. 판정은 행동 ("낙하하는 미래를 예측하는가") 이고 내부 귀속은 금지 | `auto_research/paper/PAPER_STORY_2026-10-08.md` §2-1 · §3 |

**쓰는 범위 (정본 기준)**

| 자리 | 인간 비유 | 근거 |
|---|---|---|
| intro 첫 문단 (P1) | ✅ 전부 — 사람 · 영유아 · 동물 · 적응 | 정본 §2-1 · §3 |
| P2 hook (Figure 1) | ✅ "사람은 쉽게 직관적으로 이런 예시에서 가능한 미래를 상상할 수 있다 ← gap" (사용자 원문) | 정본 §0 P2 |
| P3 · 결과 · 방법 | ❌ 비유로 판정하지 않는다 — 행동 정의 + 복사 기준선 | 정본 §3 · §6 |
| discussion | 정본에 따로 규칙 없음 (옛 판은 허용) | DIRECTION §5 |

**금지 (정본 §7 에서 P1 과 닿는 것)**

| 쓰지 말 것 | 대신 |
|---|---|
| "world model 은 물리를 이해하지 못한다 · intuitive physics 가 없다" (조건 없이) | "마지막 관측에 없는 X 를 따르는 미래를 예측하지 못한다" |
| "모델이 중력을 안다 / 모른다 / 물리 법칙을 적용한다" | "낙하하는 미래를 예측하지 않는다" |
| "기억 (memory) 문제다" · "state 를 잃는다" | 정보는 남아 있다 (07 장) |
| "action-free 학습으로 해결된다" | Ariel · post-FT 반례 (08 장) |

⚠️ 우리는 사람을 재지 않았다. P1 의 사람 · 영유아 · 동물 문장은 전부 **문헌 등급**이고, P2 의 "사람은 쉽게 상상한다 ← gap" 도 우리 측정이 아니다. 사람 대비 점수를 쓰려면 벤치마크 논문이 보고한 사람 점수를 인용해야 하는데, 그 확인은 02 장 몫이다.
⚠️ P1 이 intro 전체가 되면서 비유가 본문 주장으로 새어 들 위험이 커졌다. 판정 문장은 정본 §7 금지 표 · §6 하네스 질문 4 ("보이는 것 / 안 보이는 것 / 없는 것 중 무엇을 재나") 로 거른다.

출처: `auto_research/paper/PAPER_STORY_2026-10-08.md` §0 · §2-1 · §3 · §6 · §7 · `auto_research/paper/DIRECTION_2026-09-25.md` §1 · §5 · `auto_research/paper/NARRATIVE_v2_USER_2026-09-29.md` §B-5

---

## 6. 선점 위험 — P1 과 닿는 것만 (전체 표는 02 · 09 장)

| 위험 | 논문 | P1 과 닿는 지점 | 남는 delta (원 문서 기준) | 출처 |
|---|---|---|---|---|
| 중 | PLATO (Piloto et al. NHB 2022) | "발달심리에서 영감받아 관측으로 intuitive physics 를 배운다" 를 먼저 했다 | object-centric slot vs 우리는 장면 단위 (방법 절에서 구분) · 반론 "patch 로는 부족" 에 답해야 한다 | 정본 §2-1 · `REVIEW_CVPR_ORAL` §6 · P-25 §2-7 · 영역 R3 논문 목록 |
| 중 | Intuitive physics understanding emerges from SSL (Garrido et al. 2025, 2502.11831) | "자기지도 (관측) 학습으로 intuitive physics 가 창발한다" — P1 의 기계판 주장 | 반박 대상은 창발 자체가 아니라 그 근거 점수가 예측을 요구하지 않는다는 것 → **02 장** | 정본 §4-1 · G04a P-29 / S-26 |
| 중 | Learning Latent Action World Models In The Wild (Garrido et al. 2026, 2601.05230) | frozen V-JEPA 2 + 실영상 action-free 새 predictor — "관측만으로 배우는 predictor" 구성을 먼저 냈다 | 복사 귀무 · 안 보이는 것 / 없는 것 판독 · encoder 정보 국소화 → **09 장** | G04a P-29 · (c) 선점 위험표 |
| 낮음 | Opinion (Su et al. 2025) · Schulze Buschoff 2023 | 선점이 아니라 긴장 (§3-1) | — | §3-1 |

전체 선점 위험표 (StEvo-Bench · WRBench · Human-JEPA · HERA · GEOPHYS · Physion · Nayebi · Joseph …) 는 **02 장 (P2)** 과 **09 장 (목표)** 이 쓴다.

출처: G04a 묶음 요약 (c) · `auto_research/Archive/LITERATURE_PREDICTOR_ROLE_2026-09-29.md` · `auto_research/Archive/LITERATURE_POSITIONING_2026-09-25.md` · `auto_research/Archive/REVIEW_CVPR_ORAL_2026-09-25.md` §6

---

## 7. 이 장의 상태 한 줄씩

| 무엇 | 상태 |
|---|---|
| P1 (a) 상상 · 예측 | 배경 인용 있음 (Craik ✅ 2 차 · Clark ✅ 초록 · Wolpert & Kawato ✅) · 직관물리 인지과학은 Battaglia 2013 하나 (확인 기록 없음) |
| P1 (b) 영유아 | 사용자 목록 5 편, **원문 확인 기록 없음 → 인용 금지** |
| P1 (b) 동물 | **인용 없음** (후보 Vallortigara 그룹, 미확인) |
| P1 (b) 가장 먼저 | **인용 없음** |
| P1 (c) 적응 | **인용 없음** (후보 Lake 2017, 미확인) |
| 계산 모델 다리 | LeCun 2022 ✅ (절/쪽 ❓) · PLATO · ADEPT (URL 만) |
| 긴장 문헌 | Su 2025 ✅ · Schulze Buschoff 2023 ✅ · 영아 VoE 논쟁 (기록 없음) |
| 할 일 | 정본 §5 #10 — 문헌 확인 (Claude), 미착수 |

---

## 출처

- `auto_research/paper/PAPER_STORY_2026-10-08.md` — §0 원문 · §2 문단별 뼈대 · §2-1 P1 근거 상태 · §3 바뀌는 것 · §4-1 · §4-9 · §5 #10 · §6 · §7 (정본)
- `auto_research/paper/NARRATIVE_v2_USER_2026-09-29.md` — §A 사용자 인용 원문 · §B-5 · §B-7 (대체됨 → 정본, 인용 목록은 정본 §2-1 이 가리킨다)
- `auto_research/paper/DIRECTION_2026-09-25.md` — §1 · §5 옛 인간 비유 규칙 (대체됨)
- `auto_research/Archive/LITERATURE_PREDICTOR_ROLE_2026-09-29.md` — §0 · §8 (LeCun 2022 · Craik · Tolman · Wolpert & Kawato · Clark · Rao & Ballard · Ha & Schmidhuber · PredNet verbatim 표) · 단서
- `auto_research/Archive/LITERATURE_STATE_WORLD_MODELS_2026-09-26.md` — §4-1 (Schulze Buschoff 2023 · Su et al. 2025)
- `auto_research/Archive/LITERATURE_POSITIONING_2026-09-25.md` — §2 정의의 계보 · §6-4 배경 (Battaglia 2013) · 논문 목록 (PLATO · ADEPT · InfLevel · Opinion)
- `auto_research/Archive/REVIEW_CVPR_ORAL_2026-09-25.md` — (절 번호는 문서 안 "관련 연구 · 새로움 검토" 리뷰의 것) §2 N 행 (네 능력의 인지과학 계보) · §4 "문헌 계보" (영아 VoE 논쟁) · §6 · 논문 목록
- `z_research/context_encoder_analysis/Archive/STATE_EVOLUTION_CAPABILITIES_2026-09-19.md` — §1 네 능력 (3a/3b 분할은 정본 §3, 10-08)
- `CLAUDE.md` — §0 · §5-1 · §5-5 · §5-6 · §10 #10 (P3 수치의 요약 위치; 원 문서는 04–07 장)
- 원재료: `scratchpad/archive10/extracts/G04a_literature.md` · `G05_paper.md`
