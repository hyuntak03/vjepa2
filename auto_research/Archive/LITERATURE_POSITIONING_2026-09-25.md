# 문헌 · 리뷰어 행태 조사 — 위치 잡기 (2026-09-25)
> 웹 조사 워크플로 (영역 4 + 종합 1, 에이전트 5) 산출물을 그대로 옮겼다. 인용 전 원문 재확인 필요 항목은 본문에 표시돼 있다.> ⚠️ WebSearch 한도 소진 · OpenReview 챌린지로 일부 누락 가능. 2026 preprint 는 동료 심사 전.
## 종합
# 선행연구 종합: latent world model 의 state evolution 과 V-JEPA 2 predictor 병목 (2026-09-25)

> **입력.** 웹 조사 보고 4편을 합쳤다.
> - (R1) 정의와 JEPA 선행 검증
> - (R2) probing 과 인과 해석
> - (R3) 관측 없이 상태를 진화시키는 predictor 설계
> - (R4) 분석 논문에 대한 실제 리뷰어 행태
>
> **인용 원칙.** 네 보고가 존재를 확인한 인용만 썼다. 보고끼리 venue 가 어긋나면 더 직접적인 근거(OpenReview API 확인)를 따랐고, 그런 항목은 §6-5 에 모았다.
>
> **레포에서 읽기 전용으로 확인한 사실 둘.**
> - 사전학습 mask 의 두 블록이 모두 `temporal_scale: [1.0, 1.0]` 이다 (`configs/train/vith16/pretrain-256px-16f.yaml`, `cooldown-256px-64f.yaml`).
> - V-JEPA 2 Table 20 의 전사값이 있다 (`z_research/anticipation/EK100/README.md`). action recall 은 encoder 만 39.1, 둘 다 39.7, predictor 만 20.2 다. 원 논문과 다시 대조해야 한다.
>
> **우리 수치의 출처.** `auto_research/README.md` 정정판이다. M2 와 P2 (2026-09-25) 는 1차 결과이고 아직 적대 검증을 거치지 않았다.
>
> **한계.**
> - 2026 preprint 는 대부분 동료 심사 전이다. 초록과 HTML 요약으로만 확인했다.
> - R2·R4 는 조사 도중 WebSearch 한도를 다 썼다.
> - CVPR 리뷰는 비공개다. CVPR 리뷰어 행태는 ICLR·ICML·NeurIPS 공개 리뷰에서 추정했다.

---

## 0. 결론 먼저

**결론.**
- **개념은 이미 선점됐다.** "관측이 끊겨도 상태가 진화해야 한다" 는 생성형 video world model 쪽이 먼저 말했다: StEvo-Bench (ECCV 2026), WRBench (arXiv 2606.20545).
- **encoder 쪽 물리 probing 도 선점됐다**: Joseph et al. (ICML 2026), Bouwens (ICML 2026 WS).
- **우리 몫은 셋으로 좁혀야 선다.**
  1. 벤치마크가 world model 로 쓰는 **사전학습 V-JEPA 2 predictor** 를 잰다. 학습 없는 readout 에 복사·등속 외삽·위치 귀무 기준선을 붙인다.
  2. 병목이 encoder 가 아니라 predictor 에 있음을 **component 수준에서 국소화**한다.
  3. **평가 감사**: VoE surprise 점수는 상태 진화를 인증하지 못한다.
- **new predictor 는 그 자체로는 새롭지 않다.** "frozen encoder + 새 predictor" 는 V-JEPA 2-AC, DINO-WM, DINO-world, JEPA-WMs 가 이미 표준으로 만들었다. 우리 진단에서 나온 부품이 **진단 시그니처를 없앨 때만** 기여가 된다.

**근거 3줄.**
1. IntPhys 2 (NeurIPS 2025 D&B) 는 borderline accept 4개를 받고도 reject 됐다. 그 논문이 답하지 못한 리뷰어 질문이 곧 우리 질문이다: "Is it due to memory? Or representation and perception?"
2. 가장 가까운 선례 둘이 우리 설계를 지지한다.
   - Nayebi et al. (NeurIPS 2023): frozen encoder + CTRNN 이 46–49% 이고, 복사('No Dynamics')는 19–25% 다.
   - Physion (NeurIPS 2021): 시뮬레이션한 readout 이 관측 readout 보다 낫지 않다 (p=0.53).
3. "왜 사전학습 predictor 인가" 의 가장 강한 답은 config 에 있다. 사전학습 mask 가 시간 전체에 걸친 tube 라서, 이 predictor 는 future-only 질의를 학습한 적이 없다. 그런데 Garrido, IntPhys 2, EK100, WMReward 가 바로 그 용도로 쓴다.

**미결 1줄.** "encoder is enough" 의 근거는 아직 decodability (속도 R² 0.99) 뿐이다. closure 와 개입 증거가 없으면 리뷰를 통과하지 못한다 (§3-4).

**사용자가 정해야 할 것 4가지.**
1. **동기 문장의 사실 오류.** planning (V-JEPA 2-AC) 은 사전학습 predictor 를 쓰지 않는다. 원문은 "freeze the video encoder and learn a new action-conditioned predictor" 다. 레포 `auto_research/Archive/PLAN_ENCODER_PREDICTOR_2026-09-25.md` 의 L0 도 "AC planning" 을 사용처로 적었다. 선택지는 둘이다: 동기에서 planning 을 빼거나, 2-AC 를 직접 잰다.
2. **new predictor 학습 범위.** 현재 범위(2026-09-24: 새 학습 없음)와 CLAUDE.md 의 "frozen 위에서 닫힌다" 가 새 방향(new predictor 까지)과 충돌한다.
3. **학회.**
   - CVPR 은 rebuttal 에서 큰 실험을 요구할 수 없다. 다중 모델과 실영상 결과가 제출본에 있어야 한다.
   - 평가 감사 기여는 NeurIPS 2026 Evaluations & Datasets 의 auditing 기준과 가장 잘 맞는다.
4. **한 문장 메시지** (§1-6 의 후보 셋 중 하나).

---

## 1. 개념 정의

### 1-1. 채택할 정의

> **A latent world model maintains a belief state that is sufficient for predicting the latent future, and advances that state with a transition that stays predictive when no new observation arrives.**
>
> latent world model 은 잠재 미래를 예측하기에 충분한 belief state 를 유지하고, 새 관측이 들어오지 않아도 예측력을 잃지 않는 전이로 그 상태를 전진시킨다.

| 계보 | 출처 | 정의에서 가져오는 것 |
|---|---|---|
| 제어·RL | [Planning and acting in partially observable stochastic domains](https://www.sciencedirect.com/science/article/pii/S000437029800023X) (AI 1998); [Predictive Representations of State](https://proceedings.neurips.cc/paper/2001/hash/1e4d36177d71bbb3558e43af9577d70e-Abstract.html) (NIPS 2001) | 상태는 과거의 충분통계량이고, 미래 예측에 충분한 것이다. 관측이 없으면 전이 모형만으로 갱신한다 |
| latent WM | [Learning Latent Dynamics for Planning from Pixels (PlaNet)](https://arxiv.org/abs/1811.04551) (ICML 2019) | posterior 는 관측으로 고치고, prior 는 관측 없이 굴린다. 한 걸음 학습의 gradient 는 "never traverses a chain of multiple p" |
| JEPA | [A Path Towards Autonomous Machine Intelligence](https://openreview.net/pdf?id=BZ5a1r-kVsf) (OpenReview 2022) | world model 역할 (1) "estimate missing information about the state of the world not provided by perception", (2) 미래 상태 예측. 원문 PDF 는 2차 인용으로만 확인했다 |
| JEPA (각주로만) | LeCun 2024 게시글 `s(t+1)=Pred(h(t), s(t), z(t), a(t))` | predictor 가 이전 상태 s(t) 를 입력으로 받는다. 릴리즈 V-JEPA 2 predictor 는 `Pred(h(t−C:t))` 라 s(t) 가 없다. SNS 출처라 본문 근거로 쓰지 않는다 |
| 생성형 WM | [Out of Sight, Out of Mind? Evaluating State Evolution in Video World Models (StEvo-Bench)](https://arxiv.org/abs/2603.13215) (ECCV 2026); [Current World Models Lack a Persistent State Core (WRBench)](https://arxiv.org/abs/2606.20545) (arXiv 2026) | "states should correctly evolve, even if not observed" / "an internal world state that keeps evolving over time, decoupled from observation" |
| 직관물리 | [IntPhys 2](https://arxiv.org/abs/2506.09849) | Permanence = "Objects persist in space and time, even when out of sight" |

**모듈 분업.** encoder E 는 belief 를 **읽는** 쪽(충분성)을, predictor P 는 **전진시키는** 쪽을 맡는다. "encoder 는 관측한 것만 잘 넘기면 되고, 상태를 진전시키는 것은 predictor 의 일" 이라는 사용자 프레임과 같다.

### 1-2. 네 능력의 조작화 (측정량과 귀무)

**기호.**
- 문맥 상태: s_t = E(o_{t−C+1:t})
- 예측: ŝ_{t+k} = P_k(s_t)
- R: E(o_{t+k}) 위에서 검증한, 학습 없는 readout

| 요건 | 형식 | 능력 | 귀무·기준선 | 근거 문헌 |
|---|---|---|---|---|
| 읽기 (충분성) | R(s_t) 가 정체성·위치·속도·방향을 복원한다. **predictor 가 받는 최종층에서** 잰다 | 1 | 무작위 초기화 encoder, 픽셀·광류, 셔플 라벨 (selectivity) | PSR, Guo 2018, Gregor 2019, Joseph 2026, Hewitt & Liang 2019 |
| closure | z_{t+k} 가 z_{≤t} 로 예측된다 (정보가 입력에 있다) | 1 의 2단 | z 에서 읽은 상태의 등속 외삽 | Semigroup-JEPA 의 closure 항 |
| 지속 | R(P_k(s_t)) 가 진실 자리에 물체를 두고, 그 수명이 충분히 길다 | 2 | 위치 귀무 (같은 법칙, 다른 궤적의 진실 칸) | RSSM prior rollout, Dreamer imagination |
| 전이 | 문맥이 다르면 미래도 다르고, 복사와 등속 외삽을 넘는다 | 3 | 복사 (s_t 유지), 등속 외삽 | Physion++, Kang 2025 |
| 영속 | 물체가 t 에서 가려져도 위 줄들이 같은 수준으로 성립한다 | 4 | 비가림 짝 (matched) | IntPhys 2, OPNet, ADEPT, Loci-Looped |
| 합성 (기제) | P(P(s)) ≈ P_2(s) | 되먹임 | 관측 되먹임 (양성 대조) | PlaNet latent overshooting, V-JEPA 2-AC rollout loss |

**정의 단계에서 박아 둘 세 가지.**
- **지속은 복사가 아니다.** [TSA](https://arxiv.org/abs/2606.13714) 는 가려진 slot 을 이전 상태에 anchor 한다. 영속은 "진화한 상태의 유지" 이고, 복사형 해법은 대조군으로 둔다.
- **kinematics 로 한정한다.** 우리가 재는 전이와 가속은 위치 변화다. 힘(dynamics)이 아니다. ICLR 2026 리뷰 [fJH5](https://openreview.net/forum?id=DCbQUijwtf&noteId=Q472XAZ24e) 가 이 구분을 요구했다.
- **상태 표현은 하나로 정해지지 않는다.** Vafa ICML 2025 리뷰 [Genp](https://openreview.net/forum?id=i9npQatSev&noteId=tBfatOvBUz) 의 지적처럼, 우리가 정한 위치 표현과 안 맞는다고 상태가 없다고 할 수는 없다. 그래서 학습된 자(readout)가 아니라 파라미터 없는 교차검증을 주 근거로 둔다.

### 1-3. 반드시 갈라서 써야 할 두 장면

| 장면 | 예 | RSSM 대응 | 우리 결과 | corner case 인가 |
|---|---|---|---|---|
| (a) 관측 지평 너머의 진화 | P32 먼 슬롯, 자기 되먹임 | prior rollout | H4·H23: 시간 감쇠. H5: 두 걸음 뒤 멈춤. H23: 등속 외삽보다 뒤처짐 | 아니다. 가림 없이 성립한다 |
| (b) 문맥 안의 관측 결손 | v11 late 가림, 경계 가림 | 관측 누락 중 filtering | M2 (1차): 미래에 물체가 있을지는 문맥 마지막 1–2 튜블릿이 정한다 | 그렇다 (CLAUDE.md 2026-09-03). testbed 로 둔다 |

현재 README 는 두 장면을 한 문장으로 묶어 쓰고 있다. 논문에서는 반드시 나눈다.

### 1-4. 용어 충돌은 서론에서 흡수한다

| 다른 용법 | 출처 | 처리 |
|---|---|---|
| world model 은 encoder 표현의 성질 (선형 식별가능성) | [When Does LeJEPA Learn a World Model?](https://arxiv.org/abs/2605.26379) (Klindt·LeCun·Balestriero, arXiv 2026) | "읽기" 로 흡수한다 |
| world model 은 견고한 표현 (frozen encoder 만 평가) | [Latent Video Prediction Learns Better World Models](https://arxiv.org/abs/2605.15618) (arXiv 2026) | "읽기" 로 흡수하고, predictor 는 따로 잰다 |
| world model 은 행동 조건부여야 한다 | [Genie](https://arxiv.org/abs/2402.15391) (ICML 2024), Dreamer, LeCun 의 a(t), [Genie 3 blog](https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/) | "passive / action-free world model" 로 명시한다. 근거는 LeCun 의 역할 (1), a 를 뺀 RSSM, [2606.28455](https://arxiv.org/abs/2606.28455) 의 passive object-state world model |
| JEPA 는 애초에 world model 이 아니다 | [Critiques of World Models](https://arxiv.org/abs/2507.05169) (arXiv 2025) | 우리 결과가 이 비판의 증거로 오용되지 않게 주장 범위를 한정한다 |

### 1-5. 정의에서 곧바로 나오는 평가 요건

- VoE 의 surprise 는 원래 "진화한 belief 대비 관측의 불일치" 다. 출처는 [ADEPT](https://proceedings.neurips.cc/paper/2019/hash/e88f243bf341ded9b4ced444795c3f17-Abstract.html) (NeurIPS 2019) 와 [PLATO](https://www.nature.com/articles/s41562-022-01394-8) (Nat. Hum. Behav. 2022) 다.
- 따라서 surprise 가 상태 진화의 증거가 되려면 **복사(지속 없는 예측) 기준선을 넘어야 한다.**
- 이 기준선은 특징 공간 예측에서는 이미 표준이다: [DINO-Foresight](https://arxiv.org/abs/2412.11673) 의 Copy-Last (NeurIPS 2025), Nayebi 의 'No Dynamics'.
- Garrido et al. 의 VoE 평가에는 이 기준선이 없었다. H6 은 이 요건에서 바로 나오는 검정으로 쓴다.

### 1-6. 한 문장 메시지 후보 3개

**M-A. 병목 이동 (Physion 훅)**
> "The bottleneck has moved from perception to evolution: V-JEPA 2's encoder already exposes the observed state, but the pretrained predictor that intuitive-physics evaluations credit with world knowledge advances that state only while fresh observations arrive."
- **강점.** Physion (2021) 은 "extracting physical representations of scenes is the main bottleneck" 이라고 결론 내렸다. 웹 규모 SSL 이후 병목이 지각에서 진화로 옮겨 갔다는 서사로 문헌 사이의 긴장을 해소한다. IntPhys 2 리뷰어 질문에도 곧바로 답한다.
- **위험.**
  - encoder 충분이 closure 와 개입으로 증명돼야 한다.
  - 가속은 아직 미결이다 (H1).
  - Physion 류의 충돌·접촉 과제는 재지 않았으므로 범위를 명시해야 한다.

**M-B. 평가 감사**
> "Surprise accuracy does not certify state evolution: on IntPhys, the pretrained V-JEPA 2 predictor's advantage over copying the last observed latent vanishes in the best-scoring protocol cell, and part of its signal comes from future events that a bidirectional target encoder leaks into pre-event tokens."
- **강점.** NeurIPS E&D 의 auditing 기준, Vafa 2024 (spotlight) 와 Mirage (oral) 가 쓴 패턴에 맞다.
- **위험.**
  - 결론이 칸마다 뒤집힌다. skip2_w16 에서는 p 가 +12.2 로 유의하게 앞선다.
  - 헤드라인으로 쓰면 적대적 리뷰어가 w16 칸을 들이민다.
  - 리뷰어가 Meta 쪽일 가능성이 높으므로 "반박" 이 아니라 "분해" 하는 톤으로 쓴다.

**M-C. 상위 원칙**
> "A world model must keep evolving its state when it cannot observe; the predictor used as V-JEPA 2's world model takes one step from each new observation but stalls on its own predictions — the latent counterpart of the 'tracking-shot' failure reported for generative world models."
- **강점.** 생성형과 latent 두 패러다임에서 같은 실패가 나온다는 일반성을 준다. new predictor 로 가장 자연스럽게 이어진다.
- **위험.**
  - 개념을 처음 주장하는 것처럼 읽히면 리젝 사유가 된다. StEvo 와 WRBench 인용이 필수다.
  - "stalls" 는 H5 의 확정 범위(C16, RollOut v3 6 시나리오, 1칸 해상도)로 한정해야 한다.

**권고.**
- 본문의 논지는 M-A 로 한다.
- M-C 는 서론 마지막 문단의 상위 문장으로 둔다.
- M-B 는 독립 기여 절로 두고 헤드라인으로 쓰지 않는다.
- 초록은 가장 약한 증거 수준에 맞춘다: "여러 걸음 진화에서 실패한다. 한 걸음 전이는 있다." 과대 문장은 리뷰에서 그대로 인용돼 돌아온다 ([9tpf](https://openreview.net/forum?id=ZyLkNVHBZF&noteId=KG2exrhEOy), [rGrg](https://openreview.net/forum?id=XdLgOm5giq&noteId=7tuuGAguoe)).

**표현 주의** (CLAUDE.md 와 README 정정에 맞춤).
- "predictor 가 물체를 마지막 관측 자리에 둔다" 는 **철회된 문장**이다. "observation-anchored retrieval" 도 같은 뜻으로 읽히므로, 조작적 정의 없이는 쓰지 않는다. M2 가 검증되면 "문맥 경계 조회" 가 대체 후보다.
- "predictor 가 X 를 못 만든다 / 정보를 잃는다" 는 쓰지 않는다. p probe 가 shape 을 98.46 으로 읽는다. 대신 "정보는 있고 자리가 없다" 로 쓴다.
- "목적함수 때문이다" 는 discussion 한 문단으로만 쓴다.
- "encoder 는 관측만 담는다" 는 **문맥 encoder 에만** 쓴다. target encoder 는 양방향이라 미래를 사건 전 토큰에 번지게 한다 (H6).

### 1-7. "왜 굳이 future prediction 을 학습하지 않은 사전학습 predictor 를 보나" — 서론 첫 문단 재료

1. **실제로 world model 로 쓰인다.**
   - Garrido et al.: "we can use the encoder and predictor networks, without any additional adaptation, to probe the model's understanding of the world"
   - IntPhys 2 의 V-JEPA 2 평가
   - V-JEPA 2 본문의 EK100: predictor 가 1초 뒤를 예측하고 그 출력을 encoder 출력과 이어 붙인다
   - [TAP-JEPA](https://arxiv.org/abs/2606.00662) (CVPR 2026 EgoVis WS)
   - [WMReward](https://arxiv.org/abs/2601.10553) (CVPR 2026, surprise 를 생성 보상으로 써서 PhysicsIQ 챌린지 1위)
   - [Off-Manifold Refinement](https://arxiv.org/abs/2608.29904) (BMVC 2026)
   - [HERA](https://arxiv.org/abs/2608.05523) (frozen V-JEPA 2 predictor 위에 adapter)
2. **그 용도를 학습한 적이 없다.**
   - 사전학습 mask 는 공간 블록을 시간 전체에 반복하는 tube 다. 레포 config 의 `temporal_scale [1.0, 1.0]` 로 확인했다.
   - V-JEPA 1 은 causal multi-block 을 ablation 한 뒤 버렸다 (K400 71.9 vs 72.9).
   - Garrido 스스로 "V-JEPA is never trained using a causal prediction task" 라고 적었다.
   - 이 사실은 약점으로 쓰지 않는다. **평가 타당성 문제**로 쓴다: 평가들이 학습하지 않은 용도로 이 predictor 를 쓰면서 물리 이해를 귀속시킨다.
3. **Meta 자신도 planning 에는 재사용하지 않았다.** V-JEPA 2-AC 는 새 predictor 에 rollout loss 를 쓰고, "error accumulation" 을 한계로 적었다.

**주의.** EK100 Table 20 에서 predictor 의 기여는 +0.6 (39.1 → 39.7) 이다. 저자도 "EK100 task mostly requires strong semantic understanding, as opposed to forecasting capabilities" 라고 썼다. 따라서 이 결과는 "downstream 이 predictor 에 기댄다" 가 아니라 "encoder 충분" 쪽 증거로 쓴다.

### 1-8. 중요한 문제인가, 풀어야만 하는 문제인가

| 방향 | 근거 |
|---|---|
| **중요하다** | V-JEPA 2 한계 절이 "autoregressive prediction suffers from error accumulation" 이라고 적었다. [Hierarchical Planning with Latent World Models](https://arxiv.org/abs/2604.03208) (Meta 공저) 는 오차 누적을 동기로 삼았다 (초록 기준 성공률 0% → 70%). WRBench 에서 23개 모델이 실패했고 Wan 을 1.3B 에서 14B 로 키우자 0.66 → 0.62 로 오히려 떨어졌다. IntPhys 2 에서는 chance 근처다. [Opinion: Learning Intuitive Physics May Require More than Visual Data](https://arxiv.org/abs/2512.06232) 는 데이터만 바꿔서는 오르지 않는다고 보고했다. [General agents need world models](https://arxiv.org/abs/2506.01622) (ICML 2025) 는 여러 걸음 과제를 일반화하는 에이전트가 예측 모형을 담는다고 보였다 |
| **약하다** | 2-AC planning 은 짧은 지평 closed-loop 라 매 스텝 다시 관측하고, 사전학습 predictor 도 쓰지 않는다. EK100 은 predictor 에 거의 기대지 않는다. 경계 가림은 corner case 다 |

**판정 (corner case / use case).**
- 무게는 가림 없이 성립하는 장면 (a) 의 결과에 둔다: 시간 감쇠, 등속 외삽보다 뒤처짐, 되먹임 멈춤.
- 닿는 use case 는 셋이다.
  - 벤치마크 점수 해석 (IntPhys, surprise)
  - surprise 를 보상·guidance 로 쓰는 후속 연구 (WMReward, Off-Manifold)
  - 예측 롤아웃 (계층 계획 논문이 동기로 삼은 오차 누적)
- planning 은 2-AC 를 직접 재기 전까지 use case 로 주장하지 않는다.

---

## 2. 선행연구 지도

### 2-1. 정의와 belief 계보

| 논문 | 학회·연도 | 그들이 보인 것 | 우리와의 차이·새로움 |
|---|---|---|---|
| [Recurrent World Models Facilitate Policy Evolution](https://arxiv.org/abs/1809.01999) | NeurIPS 2018 | VAE(V) + MDN-RNN(M) 구조. 에이전트를 world model 의 꿈 안에서 학습시킨다 | 압축(encoder)과 진화(dynamics)를 나눈 원형이다 |
| [PlaNet](https://arxiv.org/abs/1811.04551) | ICML 2019 | RSSM 의 prior/posterior, latent overshooting, deterministic path | 정의의 형식 틀이다. H5 해석(한 걸음 학습은 여러 걸음을 학습하지 않는다)과 설계 원칙의 출처다 |
| [Dreamer](https://arxiv.org/abs/1912.01603) / [DreamerV3](https://www.nature.com/articles/s41586-025-08744-2) | ICLR 2020 / Nature 2025 | imagination rollout 으로 행동을 학습한다 | 기능적 정의: 관측 없이 굴린 궤적이 쓸모 있어야 한다 |
| [TD-VAE](https://arxiv.org/abs/1806.03107) | ICLR 2019 | belief state 와 jumpy 한 여러 걸음 예측 | one-shot 먼 질의(P32)와 걸음별 전개를 구분하는 계보 |
| [Shaping Belief States with Generative Environment Models for RL](https://arxiv.org/abs/1906.09237) | NeurIPS 2019 | 시야 밖 대상을 belief 에서 probe 로 읽는다. overshooting 이 결정적이다 | 관측 밖 probe 의 직접 선례다 (RL 3D 미로, JEPA 아님) |
| [Neural Predictive Belief Representations](https://arxiv.org/abs/1811.06407) | arXiv 2018 | belief 여부를 하부 상태 probe 로 판정한다 | "읽기" 판정 방법의 계보 |
| [Transformers represent belief state geometry in their residual stream](https://arxiv.org/abs/2405.15943) | NeurIPS 2024 | belief 기하가 residual stream 에 선형으로 나타난다 | representation geometry 로 encoder 충분을 보일 때의 방법 선례 (언어) |

### 2-2. 생성형 video world model 의 state evolution — 개념 선점

| 논문 | 학회·연도 | 그들이 보인 것 | 우리와의 차이·새로움 |
|---|---|---|---|
| [StEvo-Bench](https://arxiv.org/abs/2603.13215) | ECCV 2026 | 가림막 삽입, 소등, lookaway 로 관측을 끊는다. Veo 3, Sora 2, Genie 3 모두 성공률 10% 미만이고, 진화가 멈추거나 장면이 freeze 된다. V-JEPA2·DINO-WM 은 관련연구에서 언급만 하고 평가하지 않았다 | **스쿠프 위험이 가장 크다.** 우리는 latent predictor 로 옮기고, 모듈을 국소화하고, surprise 채점을 감사한다 |
| [WRBench](https://arxiv.org/abs/2606.20545) | arXiv 2026 | 23개 모델이 'tracking shot' 처럼 동작한다. 돌아온 대상을 떠날 때의 상태 그대로 재개하고, 규모를 키우면 오히려 나빠진다 | 우리 결과의 생성형 쌍둥이다. 일반성의 근거로 쓰고, 개념 최초 주장은 하지 않는다 |
| [A Mechanistic View on Video Generation as World Models: State and Dynamics](https://arxiv.org/abs/2601.17067) | arXiv 2026 | video 모델은 'stateless' 라고 진단하고 상태 구성과 동역학을 분리해 본다 | encoder/predictor 틀의 동시대 선언문 (survey 성격) |
| [Objects in Generated Videos Are Slower Than They Appear](https://arxiv.org/abs/2512.02016) | CVPR 2026 Findings (검색 요약 기준, 미확인) | 유효 중력이 0.38–2.27 m/s² 이고 frame-rate 재척도로도 교정되지 않는다 | "등속 외삽보다 뒤처짐" 의 생성형 대응이다. frame-rate 교락 검정의 선례 |
| [Sora report](https://openai.com/index/video-generation-models-as-world-simulators/) / [Genie 3 blog](https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/) | 2024 / 2025 (동료심사 없음) | object permanence 를 world model 요건으로 내세운다 | 업계 주장 인용용 |

### 2-3. 사전학습 V-JEPA(2) predictor 를 쓰거나 다룬 연구 — 우리의 검증 대상

| 논문 | 학회·연도 | 그들이 보인 것 | 우리와의 차이·새로움 |
|---|---|---|---|
| [Revisiting Feature Prediction for Learning Visual Representations from Video (V-JEPA)](https://arxiv.org/abs/2404.08471) | TMLR 2024 (R1 기준; R3 는 미확인) | tube mask 를 쓴다. causal multi-block 은 ablation 에서 더 나빠 채택하지 않았다 | 사전학습 predictor 는 future-only 를 배우지 않았다. V-JEPA 2 config 도 같다 (레포 확인) |
| [Intuitive physics understanding emerges from self-supervised pretraining on natural videos](https://arxiv.org/abs/2502.11831) | arXiv 2025 | frozen encoder+predictor surprise 로 창발을 보고한다. 대조군은 무작위 초기화, VideoMAEv2, MLLM 이고 복사 기준선은 없다. 3–4 s 기억 한계를 스스로 적었다 | 반박이 아니라 "그 점수가 상태 진화를 요구하는가" 로 쓴다 |
| [IntPhys 2](https://arxiv.org/abs/2506.09849) ([리뷰](https://openreview.net/forum?id=Xpf5x3mLvn)) | NeurIPS 2025 D&B, Reject | V-JEPA 계열이 chance 근처다. 원인으로 "stricter requirements on short term memory" 를 들었다 | 원인을 memory 로 지목만 했다. 우리는 모듈과 기제까지 좁힌다 |
| [V-JEPA 2](https://arxiv.org/abs/2506.09985) | arXiv 2025 | EK100: predictor 가 1초 뒤를 예측한다 (Table 20 기여 +0.6). 2-AC 는 새 predictor (약 300M, 24층, block-causal, teacher forcing + rollout T=2) 다. error accumulation 을 한계로 적었다 | 사용처 근거인 동시에 "planning 은 이 predictor 를 안 쓴다" 의 근거 |
| [V-JEPA 2.1](https://arxiv.org/abs/2603.14482) | arXiv 2026 | dense predictive loss, 중간층 deep self-supervision | "2.1 에서도 그런가" 는 반드시 나올 질문이다 |
| [HERA](https://arxiv.org/abs/2608.05523) | arXiv 2026 | frozen V-JEPA 2 predictor 에 기억 라우팅 adapter 를 붙인다. IntPhys2 main 52.57 → 54.35 | 동기가 거의 같다. 우리 차별점은 학습 없는 측정, 복사 대조, 병목 국소화, 여러 걸음 진화 측정 |
| [WMReward (Inference-time Physics Alignment…)](https://arxiv.org/abs/2601.10553) | CVPR 2026 | V-JEPA 2 surprise 를 생성 보상으로 쓴다. PhysicsIQ 1위 (62.64%) | 사용처다. 복사 기준선 결과가 이 관행에 직접 닿는다 |
| [Off-Manifold Refinement](https://arxiv.org/abs/2608.29904) | BMVC 2026 | V-JEPA 2.1 surprise 의 기울기로 생성을 유도한다 (+5.0pt) | 사용처 |
| [TAP-JEPA](https://arxiv.org/abs/2606.00662) | CVPR 2026 EgoVis WS | 2.1 predictor 미래 토큰으로 EK100 2위. predictor ablation 은 없다 | 검증 없이 쓰는 관행의 예 |
| [Asleep at the Wheel: JEPA's Limitations in Evaluating Novel Driving Data](https://arxiv.org/abs/2608.01336) | arXiv 2026 | surprise 가 novelty 를 잡는 것은 도메인 이동 덕분이고, 단일 데이터셋 안에서는 chance 다. 같은 embedding 에 가벼운 probe 를 얹으면 AP 가 약 2배다 | "표현은 충분하고 예측 점수가 병목" 의 독립 선례다. 동시에 "합성에서의 실패도 도메인 이동" 이라는 역공의 근거가 된다 |
| [SR-JEPA](https://arxiv.org/abs/2608.05774) | arXiv 2026 | 3D 에서 지운 개체를 predictor 가 공간적으로 채운다 | 공간 채움은 된다. 우리 주장은 시간 진화로 한정한다 |

### 2-4. encoder 쪽 물리 probing (선점됨) 과 "encoder 충분" 지지

| 논문 | 학회·연도 | 그들이 보인 것 | 우리와의 차이·새로움 |
|---|---|---|---|
| [Interpreting Physics in Video World Models](https://arxiv.org/abs/2602.07050) ([OpenReview](https://openreview.net/forum?id=aijGVmEG9Y)) | ICML 2026 | V-JEPA 2 L/H/g **encoder 만** 층별로 분석했다. 속도·가속 크기는 초기층부터, 방향은 약 1/3 깊이 PEZ 에서 읽히고, 출력층으로 가며 약해진다 | encoder 쪽은 새로움이 없다. predictor 가 받는 최종층에서 재야 한다. 가속 결과가 우리 H1 (C8 ≈ 0) 과 **충돌**하므로 조정이 필요하다 |
| [Causal State Variables in V-JEPA 2 Latents](https://openreview.net/forum?id=CC5xKstwmf) | ICML 2026 Mech Interp WS | ViT-L 방향 probe 가 layer 7 에서 100% 이고, 인과 절제 효과가 무작위의 43배 (ViT-H 54배) 다 | predictor 가 그 부분공간을 쓰는지는 재지 않았다. 우리 interchange 실험의 선례 |
| [How Do Video Foundation Models Encode Intuitive Physics?](https://arxiv.org/abs/2606.09646) | arXiv 2026 | IntPhys 2·MVP 에서 층별 probing. V-JEPA 가 가장 강하다 | surprise 가 chance 인 곳에서도 encoder 에는 정보가 있다 |
| [V-JEPA 2 Video Transcoders Find Motion-Based Features](https://openreview.net/forum?id=Q3osfMTUbE) | IJCAI SPAI WS 2026 | SSv2 운동 primitive feature 를 찾았다 | SSv2 encoder 쪽 분석은 이미 있다 |
| [GEOPHYS](https://arxiv.org/abs/2606.20707) | arXiv 2026 | predictor 없이 이미지 encoder 기하만으로 IntPhys 2 93.3% | "encoder 충분" 의 가장 강한 증거이자 "그럼 predictor 는 왜 필요한가" 라는 반문의 근거 |
| [Physion](https://arxiv.org/abs/2106.08261) | NeurIPS 2021 D&B | observed+simulated readout 이 observed 보다 낫지 않다 (p=0.53). 결론은 "지각이 병목" | 개념상 가장 가까운 선례이고 서사 훅이다. 동시에 반대 선례이기도 하다 |
| [Neural Foundations of Mental Simulation](https://arxiv.org/abs/2305.11772) | NeurIPS 2023 | frozen VC-1/R3M + CTRNN 이 46.34–48.83%, No Dynamics 가 19.03/24.79% 다. VIP 와 정지 이미지 encoder 는 실패한다 | encoder 충분 + dynamics 필요의 직접 선례이고, 복사 기준선 설계가 같다. encoder 선택도 중요하다는 반대 증거가 같은 논문에 있다 |

### 2-5. "encoder 충분" 반례와 한정

| 논문 | 학회·연도 | 그들이 보인 것 | 우리와의 차이·새로움 |
|---|---|---|---|
| [Semigroup-JEPA](https://arxiv.org/abs/2609.10464) | arXiv 2026 | rollout loss 를 encoder 까지 역전파하면 encoder 가 "features that the predictor can carry forward" 를 남긴다. 새 predictor 에서도 약 12% 이득. 오차를 closure 와 operator mismatch 로 분해한다 | 가장 직접적인 반례다 (ViT-Tiny, 장난감 물리). 그 분해식은 우리 국소화의 틀로 빌린다 |
| [Cloning Deterministic Worlds (GRWM)](https://arxiv.org/abs/2510.26782) | arXiv 2025 | 장기 충실도의 병목은 latent 기하다 | 정면 반대다. encoder 교체와 다층 입력 ablation 이 필요하다 |
| [What Drives Success in Physical Planning with JEPA World Models? (JEPA-WMs)](https://arxiv.org/abs/2512.24497) | arXiv 2025 (venue 미확인) | 계획에서 DINOv2/v3 가 V-JEPA encoder 를 이긴다. rollout 은 sim 에서 2-step, DROID 에서 6-step 이 최적이다. W^p ≤ W | 반대 증거이자 설계 수치 |
| [What Can Latent World Models Know?](https://arxiv.org/abs/2607.27017) | arXiv 2026 (저평판 신규) | recoverable → decodable → functionally used 세 단계. drag 는 0.89 → 0.13 | 틀만 빌린다 |
| [The Observer Effect in World Models](https://arxiv.org/abs/2602.12218) | arXiv 2026 | 고용량 probe 와 fine-tuning 이 잠재 물리 구조를 무너뜨린다 (ρ≈0.05). 저용량 probe 는 회복한다 (ρ>0.90) | 우리 49.3M attentive probe 가 공격받는 근거다. 저용량·파라미터 없는 측정을 주 근거로 둔다 |

### 2-6. frozen encoder + 새 predictor — new predictor 의 경쟁 영역

| 논문 | 학회·연도 | 그들이 보인 것 | 우리와의 차이·새로움 |
|---|---|---|---|
| V-JEPA 2-AC ([V-JEPA 2](https://arxiv.org/abs/2506.09985)) | arXiv 2025 | 프레임별 인코딩, 같은 공간에서 되먹임, block-causal, rollout T=2 | 우리 병목 중 셋(표적 번짐, 공간 불일치, 되먹임)을 이미 부분적으로 고쳤다. **기준선이자 가장 큰 novelty 위험** |
| [DINO-WM](https://arxiv.org/abs/2411.04983) | ICML 2025 | frozen DINOv2 + ViT predictor 로 zero-shot planning | encoder 충분 + 새 predictor 의 대표 사례 |
| [Back to the Features: DINO as a Foundation for Video World Models (DINO-world)](https://arxiv.org/abs/2507.19468) | arXiv 2025 | frozen DINOv2 + causal predictor, 절대 timestamp RoPE, Δt 균등 표본. IntPhys 91.3 vs V-JEPA ViT-H 89.4. "all predictions become inaccurate as the forecasting interval approaches 1 second" | 미래 예측으로 학습해도 1 s 에서 무너진다. new predictor 가 자명한 해결이 아니라는 경고다 |
| [PLDM](https://arxiv.org/abs/2502.14819) | arXiv 2025 | H 스텝을 완전히 unroll 하며 학습한다 | 되먹임 학습의 극단형 (대조 설계) |
| [Hierarchical Planning with Latent World Models](https://arxiv.org/abs/2604.03208) | arXiv 2026 | 다중 시간척도. VJEPA2-AC, PLDM, DINO-WM 에 적용한다 | use case 근거 |
| [Causal-JEPA](https://arxiv.org/abs/2602.11389) | ICML 2026 | 물체 단위 latent masking, 반사실 VQA 약 +20pt | 가림 강제 학습의 가장 가까운 경쟁 설계 |
| [UWM-JEPA](https://arxiv.org/abs/2605.25313) | arXiv 2026 | blind rollout 에서 probe R² 손실이 10pt 미만이다 (벡터 latent 는 41/68pt). "분리는 predictor 에서 생긴다" | 같은 방향의 최신 선행 (소규모, encoder 공동 학습) |
| [DINO-Foresight](https://arxiv.org/abs/2412.11673) | NeurIPS 2025 | frozen 다층 특징으로 미래를 예측하고 Copy-Last 와 비교한다 | 복사 기준선이 표준이라는 근거이자 다층 입력의 선례 |

### 2-7. 가림 지속과 object permanence

| 논문 | 학회·연도 | 그들이 보인 것 | 우리와의 차이·새로움 |
|---|---|---|---|
| [Learning Object Permanence from Videos via Latent Imaginations (Loci-Looped)](https://arxiv.org/abs/2310.10372) | arXiv 2023 (ICANN 2024 는 미확인) | slot 별 percept gate 에 L0 손실을 건다. 가림 중 99.2% 를 상상으로 처리하고, SAVi·G-SWM 은 단순한 가림에서도 무너진다 | 관측과 상상을 전환하는 장치의 직접 처방 (소규모 합성, slot 기반) |
| [PLATO](https://www.nature.com/articles/s41562-022-01394-8) / [ADEPT](https://proceedings.neurips.cc/paper/2019/hash/e88f243bf341ded9b4ced444795c3f17-Abstract.html) | Nat. Hum. Behav. 2022 / NeurIPS 2019 | 객체 표현, 추적, particle filter belief | VoE 계보. "patch 로는 부족하다" 는 반론의 근거 |
| [OPNet](https://arxiv.org/abs/2003.10469) · [PermaTrack](https://openaccess.thecvf.com/content/ICCV2021/html/Tokmakov_Learning_To_Track_With_Object_Permanence_ICCV_2021_paper.html) · [RAM](https://arxiv.org/abs/2204.01784) · [TCOW](https://arxiv.org/abs/2305.03052) | ECCV 2020 · ICCV 2021 · ICML 2022 · CVPR 2023 | 영속을 과제로 분해한다. 재귀 메모리와 가림 감독, 자기지도 메모리를 쓴다. transformer 의 permanence 격차 | 영속에는 명시적 메모리와 목적이 필요하다는 추적 쪽 증거 (지도학습) |
| [TSA](https://arxiv.org/abs/2606.13714) | arXiv 2026 | 가려진 slot 을 이전 상태에 anchor 한다 | "지속 = 복사" 의 대조 사례 |
| [SAMURAI](https://arxiv.org/abs/2411.11922) | arXiv 2024 | SAM 2 에 Kalman 운동 사전만 얹어 재학습 없이 LaSOT_ext AUC +7.1 | 강한 frozen 표현에 운동 상태만 더해도 오른다 |

### 2-8. 되먹임 drift 와 먼 지평 — 설계 처방

| 논문 | 학회·연도 | 그들이 보인 것 | 우리와의 차이·새로움 |
|---|---|---|---|
| [GNS](https://arxiv.org/abs/2002.09405) · [MP-PDE (pushforward)](https://arxiv.org/abs/2202.03376) | ICML 2020 · ICLR 2022 | 입력 잡음이 장기 성능을 좌우한다. 2 스텝을 펼치고 마지막만 역전파한다 | H5 되먹임 멈춤에 대한 가장 싼 처방 |
| [Diffusion Forcing](https://arxiv.org/abs/2407.01392) · [Self Forcing](https://arxiv.org/abs/2506.08009) · [GameNGen](https://arxiv.org/abs/2408.14837) | NeurIPS 2024 · arXiv 2025 · arXiv 2024 | 토큰별 잡음 수준, 완전 자기 롤아웃 학습, 문맥 잡음 augmentation | 관측 blackout 과 되먹임 안정화를 한 틀로 묶는다 (Self Forcing 은 상한 대조) |
| [DPWM](https://arxiv.org/abs/2608.07420) · [Clockwork VAE](https://arxiv.org/abs/2102.09532) | arXiv 2026 · NeurIPS 2021 | 끝점 목적 (구조보다 목적이 중요), 계층적 느린 latent | 먼 지평 감쇠의 처방 |
| [S4WM](https://arxiv.org/abs/2307.02064) · [R2I](https://arxiv.org/abs/2403.04253) · [SlotSSM](https://proceedings.neurips.cc/paper_files/paper/2024/hash/158ac5698e36a01ee5ca9e6732685b34-Abstract-Conference.html) · [Long-Context SSM Video WM](https://arxiv.org/abs/2505.20171) · [Echo-Memory](https://arxiv.org/abs/2606.09803) | NeurIPS 2023 · ICLR 2024 oral · NeurIPS 2024 · arXiv 2025 · arXiv 2026 | 재귀·SSM 상태가 장기 상상과 기억을 푼다. Echo-Memory 는 raw context 가 강한 기준선이라고 보고했다 | 운반되는 상태 설계의 근거 |
| [Belief State Transformer](https://arxiv.org/abs/2410.23506) · [NextLat](https://arxiv.org/abs/2511.05963) | ICLR 2025 · arXiv 2025 | belief 를 강제하는 보조 목적 | 가벼운 보조 손실 후보 (언어, 이식성 미검증) |
| [Deep multi-scale video prediction beyond MSE](https://arxiv.org/abs/1511.05440) | ICLR 2016 | L1 은 조건부 중앙값을 낸다. 먼 미래일수록 흐려진다 | hedging 대안 설명의 출처 |
| [Branch-JEPA](https://arxiv.org/abs/2607.05238) · [VJEPA (Variational JEPA)](https://arxiv.org/abs/2601.14354) | arXiv 2026 | 다중 가설 또는 확률적 JEPA | 확률화만으로는 새롭지 않다 |

### 2-9. 평가 방법론: 점수는 상태 모형이 아니다

| 논문 | 학회·연도 | 그들이 보인 것 | 우리와의 차이·새로움 |
|---|---|---|---|
| [Evaluating the World Model Implicit in a Generative Model](https://arxiv.org/abs/2406.03689) | NeurIPS 2024 spotlight | 다음 토큰 정확도와 probe 는 좋은데 상태 일관성 지표로 보면 불일관하다 | H6 과 같은 논리 구조 (기존 진단의 과대평가) |
| [What Has a Foundation Model Found?](https://arxiv.org/abs/2507.06952) | ICML 2025 | 궤도 데이터로 학습한 모델이 Newton 역학을 쓰지 않는다 | 예측 성능과 법칙 표현은 다르다 |
| [How Far is Video Generation from World Model](https://arxiv.org/abs/2411.02385) | ICML 2025 (ICLR 2025 reject) | case-based 일반화, 속성 우선순위 color > size > velocity > shape | "retrieval" 의 행동 수준 선례. 다만 그들의 retrieval 은 학습 사례 검색이고 우리는 문맥 조회다 |
| [Physics-IQ](https://arxiv.org/abs/2501.09038) · [Are Emergent Abilities of LLMs a Mirage?](https://openreview.net/forum?id=ITw9edRDlD) | arXiv 2025 · NeurIPS 2023 oral | 시각 사실성과 물리 이해가 무관하다. 평가 지표 인공물이 서사를 만든다 | 감사형 기여가 받아들여지는 조건 |

---

## 3. 우리 주장 분류 — 이미 알려진 것 / 새 것 / 사소하다고 볼 위험

### 3-1. 주장별 판정

| 우리 주장 (근거) | 판정 | 이미 한 쪽 / 공백 | 쓰는 법 |
|---|---|---|---|
| 관측이 끊기면 상태가 진화하지 않는다 (일반 명제) | **알려짐** (생성형) | StEvo, WRBench, Mechanistic View | latent 판으로, 일반성의 근거로 쓴다. "we define / introduce state evolution" 은 금지 |
| V-JEPA predictor 는 기억이 부족하다 | **알려짐** (지목 수준) | IntPhys 2 "short term memory", Garrido 3–4 s, HERA 의 동기 | 모듈과 기제까지 좁혀야 새로워진다 |
| 자기회귀 오차가 누적된다 | **알려짐** | V-JEPA 2 한계 절, DINO-world 1 s, JEPA-WMs | 새로운 부분은 아래 H5 행 |
| 사전학습 predictor 는 future-only 를 학습하지 않았다 | **부분적으로 알려짐** | V-JEPA 1 ablation, Garrido 가 스스로 인정 | 평가 타당성 문제로 쓴다. 기제 가설(tube mask 때문에 같은 자리의 과거를 조회한다)은 미검증 |
| H5: 관측을 되먹이면 한 걸음 전이가 된다 (새 칸 착지 0.78–0.96). 자기 예측을 되먹이면 두 걸음 뒤 멈추고, 멈춘 자리는 걸음 수에 묶인다. target 공간의 참 미래도 j≥4 에서 복사 쪽으로 간다 | **새 것** (3팔 비교; 검색 범위 안) | 2-AC 와 UWM-JEPA 는 자기 모델을 학습한다 | 가장 non-obvious 한 결과다. ar_z 를 양성 대조로 앞세우고, 되먹임 실험은 stress test 로 자리매김한다 |
| H4·H23·H8: 먼 슬롯 물체 신호가 튜블릿당 −4.3~−5.8pt 줄고, 렌즈나 knockout 으로 거의 돌아오지 않는다 | 현상은 알려짐, **측정은 새 것** | DINO-world, Garrido | P2 (1차): 한계의 단위는 튜블릿 수가 아니라 "마지막 관측 자리에서 약 2.6칸 + 약 2걸음" 이다. 초와 거리 단위로 보고한다 |
| H23: 물체 자리가 등속 외삽보다 뒤처진다 (진행률 중앙값 0.5–0.67) | 생성형 대응은 있음, latent 에서는 **새 것** | Objects Slower, Kang | **사소 위험**: 조건부 중앙값 수축으로 설명될 수 있다 (§3-3) |
| H6: IntPhys 1 에서 복사 85.00 vs p 88.89 (+3.9 n.s.). 양방향 표적 번짐: 이동+보임 90.4 → 인과 표적 50.0 | **새 것** (VoE 에 복사 기준선을 붙인 것, 번짐 분리 모두 선례를 못 찾음) | 특징 예측에서는 Copy-Last 가 표준이고, Nayebi 의 No Dynamics 가 있다 | **양날이다.** skip2_w16 에서는 +12.2 로 유의하다. 전 격자 표를 내고 "프로토콜 민감도" 로만 쓴다 |
| encoder 가 속도·방향·정체성을 담는다 (속도 R² 0.99) | **알려짐** | Joseph, Bouwens, Punzo, V-JEPA 2 SSv2 77.3 | 기여로 내세우지 않는다. 최종 LN(z) 인터페이스에서 확인하고 대조군을 붙인다 |
| encoder 는 충분하고 병목은 predictor 다 (component 국소화) | **새 것이 될 수 있다 (아직 미성립)** | Physion (개념), Nayebi (다른 encoder·과제), Asleep at the Wheel | closure 와 개입이 없으면 상관 증거에 그친다 (§3-4) |
| H4: RoPE 가 순수 상대적이지 않다 (위치만 밀어도 p 가 19–26% 변한다) | 기술적 발견 | DINO-world 의 절대시간 설계 | 설계 원칙의 근거로만 쓰고 부록에 둔다 |
| M2 (1차): 미래에 물체가 있을지는 문맥 마지막 1–2 튜블릿이 정한다 | **새 것** (기제 증거) | 공백 | 적대 검증 뒤에 쓴다. "경계 조회" 의 조작적 정의 후보 |
| v11 경계 가림에서 미래에 물체가 없다 | **corner case** | TCOW, OPNet (지도 추적) | testbed 로 둔다 |

### 3-2. 방법의 표준성 (R2)

| 우리 방법 | 선례 | 판정 | 리뷰어가 요구할 것 |
|---|---|---|---|
| ridge 층별 probe (H1, H7) | Alain & Bengio 2016, Joseph 2026 | 표준 | 무작위 초기화 ViT-H, 픽셀·광류, 셔플 selectivity ([Hewitt & Liang](https://aclanthology.org/D19-1275/)), 궤적이 적으므로 MDL ([Voita & Titov](https://aclanthology.org/2020.emnlp-main.14/)) |
| attentive probe + 조건 간 이식 | 일반적 관행 | 변형 | 포화 (train_acc 1.000) 때문에 이식 줄만 읽는다. 좌표계 문제 ([Nanda et al. 2023](https://aclanthology.org/2023.blackboxnlp-1.2/)) |
| 파라미터 없는 템플릿 + 위치 귀무 | RSA, 템플릿 디코딩 | **강점** | 모든 문서에 일괄 적용한다 |
| 층별 렌즈 p^(l) | [logit lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens), [Vilas et al.](https://arxiv.org/abs/2310.18969) | 표준 | 초기층 cos 0.32 는 logit lens 편향이다. [tuned lens](https://arxiv.org/abs/2303.08112) 판을 붙인다 |
| attention knockout (H8, H9) | [Geva et al.](https://aclanthology.org/2023.emnlp-main.751/), [Neo et al.](https://arxiv.org/abs/2410.07149) | 표준 | zero 차단은 분포 밖 개입이다 (L7–11 에서 L1 +64%). resample/mean 대체, [path patching](https://arxiv.org/abs/2211.00593) |
| VoE 복사 기준선 (H6) | Copy-Last, Physion observed readout | **적용이 새롭다** | — |
| 인과 표적으로 번짐 분리 (H6) | 선례 못 찾음 | **새롭다** | 인과 표적이 정지 장면 점수를 올리는 분포 효과를 단서로 적는다 |
| 관측/자기/표적 되먹임 3팔 (H5) | 2-AC, UWM-JEPA, [Lambert et al.](https://arxiv.org/abs/2203.09637) | 변형·새로움 | "rollout 학습을 받지 않은 predictor" 라는 strawman 비판에 대비한다 |
| **빠진 것: 개입** | [Amnesic probing](https://aclanthology.org/2021.tacl-1.10/), [LEACE](https://arxiv.org/abs/2306.03819), [DAS](https://proceedings.mlr.press/v236/geiger24a.html), [Othello-GPT](https://arxiv.org/abs/2210.13382) | **필수** | §3-4 의 3단 |

### 3-3. 사소하다고 볼 위험과 반박 논리

1. **"마스크 복원 head 를 롤아웃하면 실패하는 게 당연하다" (strawman).**
   - 사용처를 인용한다 (§1-7).
   - 주 주장은 실제 사용 방식인 one-shot 채점 프로토콜에 건다.
   - "관측을 되먹이면 한 걸음 전이는 된다" 가 non-obvious 하다는 점을 앞세운다.
2. **결정론적 L1 predictor 의 조건부 중앙값 수축.** 근거 문헌은 Mathieu 2016, LeCun 의 z, Branch-JEPA 다.
   - 행동 수준에서 판별한다. "상태가 없다" 면 p 의 이동이 문맥 끝 속도 v_T 와 무관해야 한다. "상태는 읽되 수축한다" 면 v_T 에 비례하되 이득이 1 보다 작아야 한다.
   - H5 에서 멈춘 자리가 속도 분위별 진실 j2 자리와 같다는 결과는 "속도를 읽는다" 쪽 증거다.
   - hedging(궤적 위로 퍼짐)과 anchoring(관측 자리에 묶임)을 가르는 믿음 단면 검정을 넣는다.
   - 원인은 discussion 한 문단으로만 쓴다.
3. **"합성 등속 장면에서 속도 R² 0.99 는 픽셀 차분으로도 나온다".**
   - 무작위 초기화 ViT-H (Garrido 가 20 seed 를 쓴 관례), 픽셀·광류 기준선, 셔플 selectivity, MDL 을 붙인다.
4. **"IntPhys 1 은 원래 진화를 요구하지 않는 벤치마크다. 모델이 아니라 벤치마크 문제다".**
   - 평가 방법론 결과로 따로 세운다.
   - DINO-world (비디오 사전학습 없이 91.3) 와 GEOPHYS 가 같은 방향의 방증이다. 단 이것은 해석이고 검증하지 않았다.
5. **"SSv2 좌/우 encoder probe 는 이미 있다".**
   - 추가로, SSv2 는 V-JEPA 2 사전학습 데이터에 포함돼 있고, 좌/우는 1-bit 과제다.
   - 새로워지려면 z 로 학습한 방향 probe 를 **p 의 미래 토큰에 그대로 걸어** horizon 별 유지율을 잰다. h 를 천장으로, 복사를 기준선으로 둔다.
   - 대조군: 좌우 반전, 시간 역재생, 단일 프레임, 프레임 셔플, motion energy 기준선.
   - `/local_datasets/vlm_direction/ssv2vp` 경로는 이번 조사 노드에서 확인되지 않았다.
6. **"몇 튜블릿이 몇 초냐".**
   - 초와 거리 단위로 보고하고 stride 를 통제한다. P2 의 s=4 (7.5 fps) 는 사전학습 fps 4 에 더 가깝다.
   - Garrido 의 3–4 s, DINO-world 의 약 1 s 와 같은 축에 놓는다.

### 3-4. "encoder is enough" 를 세우는 순서

문헌이 요구하는 수준을 단계로 나눴다.

1. **decodable — 인터페이스에서.**
   - predictor 가 받는 최종 LN(z) 에서 정체성·위치·속도·방향을 읽는다.
   - 대조군, 궤적 단위 split, MDL, 층별 곡선(Joseph 와 대조)을 붙인다.
   - 가속은 조건부로만 쓰거나 미결로 둔다.
2. **충분 (closure).**
   - (a) 소량 학습 판: z(문맥 끝) → t+k 물체 위치 probe 를 만들고, 같은 자로 읽은 p 슬롯 k 와 비교한다 ([Future Lens](https://arxiv.org/abs/2311.04897), Physion 식). train 과 test 의 운동 법칙을 다르게 둔다.
   - (b) 0 파라미터 판: 문맥 encoder 에서 읽은 상태를 등속 외삽하거나, h 에서 물체 부호를 운반한다 (레포 PLAN 의 L2: M3, P2 loc_apph).
   - 이기면 "입력에 선형으로 있는 정보가 p 의 출력 자리에 실현되지 않는다" 가 선다.
   - 단서: probe 는 지도 학습이다. 그래서 '가용성' 주장으로만 쓴다.
3. **사용 여부 (개입).**
   - LEACE 로 z 의 속도·방향 부분공간을 지우거나, DAS interchange 로 두 clip 사이에서 교환한다. 그리고 p 의 물체 이동이 따라가는지 본다.
   - 대조: 같은 차원의 무작위 부분공간, patch 뒤 분포 적합성 ([Makelov et al.](https://arxiv.org/abs/2311.17030)), completeness 와 selectivity ([Canby et al.](https://aclanthology.org/2025.ijcnlp-long.47/)).
   - 결과가 어느 쪽이어도 beat 가 선다. 안 따라가면 "입력 속도를 쓰지 않는다". 따라가되 뒤처지면 "쓰지만 시간으로 적분하지 못한다" 이고, H5 와 연결된다.
4. **기하 (사용자 제안: geometry / cos sim).**
   - RSA 로 z 거리와 물리 상태 거리를 비교하고, 중심화 cos (h̄ 차감) 를 병기한다.
   - CKA 단독은 쓰지 않는다 ([Davari et al.](https://arxiv.org/abs/2210.16156), ICLR 2023).

---

## 4. 새 predictor 설계 원칙 후보

전제는 둘이다.
- 층 위치에서 설계 원리를 바로 끌어내지 않는다. localization 과 editing 은 다르다 ([Hase et al.](https://arxiv.org/abs/2301.04213), NeurIPS 2023).
- 원칙은 기능 수준 증거에 걸고, 검증은 **점수가 아니라 진단 시그니처가 사라지는지** 로 한다. Registers 도 이 점을 지적받았다 ([mTMB](https://openreview.net/forum?id=2dnO3LLiJ1&noteId=ojGA6WMjKm)).

| 원칙 | 겨냥하는 우리 진단 | 선행 증거 | 없애야 할 시그니처 | novelty 주의 |
|---|---|---|---|---|
| **P1 운반되는 재귀 상태** (state/register/slot 토큰, 스텝마다 전이) | 한 번 질의 시 먼 슬롯 감쇠 (H4·H23). 경계 1–2 튜블릿만 조회 (M2 1차). s(t) 부재 | LeCun 정의, RSSM deterministic path ("difficult to remember information over multiple time steps"), SlotSSM, S4WM, R2I, Long-Context SSM | M2 쌍둥이 패칭 결과가 경계 토큰만으로 결정되지 않는다 (late 에서 역사로부터 물체를 가져온다). 먼 슬롯 감쇠 기울기가 0 으로 간다 | Echo-Memory 에서 raw context 가 강한 기준선이었다. state 토큰 없는 2-AC 식과 반드시 비교한다 |
| **P2 입출력 공간 일치** (받는 공간에 예측하고 그대로 되먹임) | H5 ar_h: 참 미래를 target 공간으로 넣어도 j≥4 에서 되먹인 칸으로 간다 | 2-AC, DINO-world, DINO-WM | ar_h 와 ar_z 사이의 차이가 사라진다 | 이미 표준이라 원칙이라기보다 전제다 |
| **P3 여러 걸음 open-loop 학습 + 되먹임 입력 잡음** | H5: 자기 되먹임이 두 걸음 뒤 멈춘다 (C16 j3–7 기울기 −0.009, j7 적중 0.10 < 귀무 0.28) | PlaNet overshooting, Gregor 2019, 2-AC T=2, JEPA-WMs (2/6-step 최적, 그 이상은 악화), PLDM, GNS, MP-PDE, GameNGen, Self Forcing (상한) | j3–7 기울기가 진실에 가까워지고 j7 적중이 귀무를 넘는다 | 2-AC 가 이미 T=2 를 쓴다. 새로움은 시그니처 소거에서만 나온다 |
| **P4 관측 결손 게이팅 + blackout 학습** (입력에서는 빼고 표적은 참값) | v11 late 가림, 문맥 경계 가림 | Loci-Looped (gate + L0), PermaTrack, RAM, Causal-JEPA, Diffusion Forcing | late 와 early 의 차이가 사라진다. 영속이 복사가 아니라 진화다 (TSA 식 anchor 를 대조군으로) | Causal-JEPA·Loci-Looped 와 비교가 필수다. slot 병목 여부를 ablation 한다 (PLATO) |
| **P5 future-only·가변 Δt 학습 + 순수 상대 또는 절대 시간 위치 + 끝점 목적** | tube mask (future-only 가 분포 밖), RoPE 비상대성 (H4), 시간·거리 감쇠 (P2 1차) | DINO-world (절대 timestamp, Δt 균등), JEPA-WMs W^p ≤ W, DPWM, Clockwork VAE, 계층 계획 | 위치 인덱스를 밀어도 불변이다. 초 단위 감쇠가 선행 한계 (1 s, 3–4 s) 를 넘는다 | DINO-world 는 이것을 갖추고도 1 s 에서 무너졌다. 감쇠는 먼저 "학습된 적 없는 오프셋" 으로 설명해 보고, 그다음 구조 탓으로 간다 |
| **P6 인과 표적** | H6 양방향 표적 번짐 | 2-AC "encode each frame independently" | 인과 표적에서도 점수와 시그니처가 유지된다 | 프레임별 인코딩은 튜블릿 안의 운동 정보를 잃는다. 인과 슬라이딩 창이 절충안이다 |
| **P7 다중 가설 / 잠재변수 z** | hedging 과 anchoring 을 못 가른다 (뒤처짐의 대안 설명) | LeCun z, Branch-JEPA, VJEPA, UWM-JEPA | 믿음 단면이 궤적 위로 퍼지는가, 관측 자리에 묶이는가 | 이미 여러 편 있다. 단독 기여가 아니라 진단 도구로 쓴다 |
| P8 (선택) 명시적 운동 사전 / 합성 일관성 | 등속 외삽보다 뒤처짐, 가속 효과가 거의 없음 (H1 flat_a −0.04) | SAMURAI, Semigroup-JEPA | 진행률이 1 로 간다 | "물리 엔진을 가졌는가는 묻지 않는다" 는 원칙과 충돌하지 않도록 최소판만 |

**검증 규칙.**
- **기준선 넷**: 복사, 등속 외삽, 릴리즈 predictor, **2-AC 식 무액션 predictor** (state 토큰·blackout·Δt 없음). 부품별로 ablation 한다. 리뷰어가 "2-AC 에서 action 만 뺀 것" 이라고 공격할 것이기 때문이다.
- **시그니처로 판정한다**:
  - H5 기울기와 착지
  - H4/H23 공통 문턱 물체다움과 진행률
  - M2 경계 결정성
  - H6 복사 대비 마진
  - 외부 검증: IntPhys 1/2 (HERA 52.57 → 54.35 와 비교), SSv2 에 가림·끊김 조건을 만든 것
- **반대 가설 배제**: encoder 교체 (DINOv2/3) 와 다층 입력 (DINO-Foresight 식). 실패하면 결론이 GRWM·Semigroup 쪽(표현이 병목)으로 바뀐다. 두 결론을 실험 전에 적어 둔다.
- **레포 post-FT 결과를 정면으로 다룬다.** v11 held-out 은 75.83 → 91.35 로 올랐지만 IntPhys1 은 88.89 → 77.22 로 떨어졌다. 학습 도메인 밖 일반화가 판정 기준이다.
- **범위.** 새 predictor 학습은 사용자 결정 전까지 설계와 최소 검증안에 머문다 (§0 결정 2).

---

## 5. 예상 리뷰어 공격 체크리스트와 선제 대응

"실제 사례" 열은 R4 와 R2 가 OpenReview 원문으로 확인한 링크다. "(도출)" 은 공개 리뷰가 아니라 선행 연구에서 추론한 공격이다.

| # | 공격 | 실제 사례 | 우리 현재 취약점 | 선제 대응 (실험) | 선제 대응 (서술) |
|---|---|---|---|---|---|
| 1 | 동기가 허수아비다 ("그런 믿음이 정말 퍼졌나") | [How Far ICLR'25, 저자 답글에 인용된 요구](https://openreview.net/forum?id=ZyLkNVHBZF&noteId=G0RzqbtB9e) | planning 을 사용처로 잘못 적었다. 되먹임 결과가 전면에 있다 | — | 1쪽에 Garrido 원문 두 문장, EK100, WMReward·Off-Manifold·TAP-JEPA·HERA 를 둔다. planning 은 뺀다. 주 주장은 one-shot 채점에, 되먹임은 stress test 로 |
| 2 | 단일 모델, 주장 범위가 실험보다 넓다 | [Binding AC](https://openreview.net/forum?id=5BS6gBb4yP&noteId=328mSaifL0), [Binding 리뷰](https://openreview.net/forum?id=5BS6gBb4yP&noteId=1sug92evNW), [Interpreting Physics AC](https://openreview.net/forum?id=aijGVmEG9Y&noteId=wdr5fSceeM), [XtX3](https://openreview.net/forum?id=DCbQUijwtf&noteId=sZia8lROVj) | ViT-H 하나뿐이다 | ViT-L/g 와 V-JEPA 2.1 에서 세 시그니처를 재현한다 (기준은 ViT-H, 나머지는 robustness). 대조군에서는 결과가 달라지는 것까지 보인다 | 제목을 "the pretrained V-JEPA 2 predictor" 수준으로 좁힌다 |
| 3 | 읽힌다고 쓰이는 것은 아니다 | [VLM-interaction Tj6t](https://openreview.net/forum?id=XdLgOm5giq&noteId=3ry1VY4IN5) | 근거가 속도 R² 0.99 뿐이고 대조군이 없다 | §3-4 의 1–3단 | 읽기 충분의 범위를 정체성·위치·속도·방향으로 한정한다 |
| 4 | 원인이 측정 도구다 (readout, 문턱, 추적 오차) | [Vafa RjHx](https://openreview.net/forum?id=aVK4JFpegy&noteId=5mks29Rzpl), [Vafa S5LR](https://openreview.net/forum?id=aVK4JFpegy&noteId=cVj9P19oLT), [Morpheus ICLR](https://openreview.net/forum?id=1E6pburMKc&noteId=zarqBE5G9D) → [ICML AC](https://openreview.net/forum?id=f4xAzcMug6&noteId=JDzjrQGHeI), [rtwz](https://openreview.net/forum?id=DCbQUijwtf&noteId=dtNDRLky4Y) | 자기 기준 문턱 때문에 헤드라인 3개를 철회한 이력이 있다 | 공통 고정 문턱, 위치 귀무, 자 오차 예산 (held-out, encoder 대조), 섭동 민감도 표. 파라미터 없는 교차검증을 주 근거로 | 주장마다 어느 표적, 어느 세트인지 적는다 |
| 5 | descriptive 하고 해결책이 없다 | [IntPhys 2 q2be](https://openreview.net/forum?id=Xpf5x3mLvn&noteId=jVZdtEOm3Z), [fK2C](https://openreview.net/forum?id=aijGVmEG9Y&noteId=bzUBm5zw5w), XtX3 (저자가 AI 생성 리뷰라고 지적, 논문 철회), [Registers mTMB](https://openreview.net/forum?id=2dnO3LLiJ1&noteId=ojGA6WMjKm) | 범위가 '새 학습 없음' 이다 | 사용자 결정: (a) frozen 최소 개입 + 시그니처 소거 (H9 는 IntPhys1 에서 안 오르고 정지 조건 −5.4 라 약함), (b) 측정 도구 자체를 기여로, (c) 새 predictor | 진단 → 기제 → 최소 개입 → 다중 모델 (Registers 패턴) |
| 6 | 이미 했다 | CVPR/ICCV 가이드라인이 "'has been done before' MUST be backed up with specific references" 를 요구하므로, 리뷰어는 StEvo, WRBench, HERA, Joseph, DINO-world, Nayebi, UWM-JEPA, 2-AC 를 댈 것이다 | 차별화 문단이 없다 | — | 한 문단으로 차별점을 쓴다: 대상, 통제된 물리 + 기준선 셋, component 국소화, 감사, 처방 연결. Meta 리뷰어를 전제로 "분해" 하는 톤 |
| 7 | 뻔하다 | [r3mF](https://openreview.net/forum?id=ZyLkNVHBZF&noteId=nJEETfVJZQ), [hEH6](https://openreview.net/forum?id=DLlVjZQ7vD&noteId=VKl2zK39ow), [duNp](https://openreview.net/forum?id=1OkVexYLct&noteId=zIvyYWT8Lt) | "masked predictor 가 롤아웃 못 하는 건 당연하다" | — | 1쪽에 non-obvious 한 점 셋: 관측 되먹임은 한 걸음 전이가 된다, 멈춤은 걸음 수에 묶인다, one-shot 에서도 등속 외삽보다 뒤처진다 |
| 8 | hedging 이다 (조건부 중앙값) | (도출) Mathieu 2016 | 뒤처짐·감쇠가 수축과 같은 모양이다 | v_T 비례성 검정, 믿음 단면 | 원인은 discussion 한 문단 |
| 9 | 합성 데이터만, 도메인 이동 | [hTWK](https://openreview.net/forum?id=6UJf6B8RZ8&noteId=g95hP3rGHM) / 성공한 반박 [LikePhys 저자](https://openreview.net/forum?id=6UJf6B8RZ8&noteId=uZKCzmHEiv), [Othello testbed](https://openreview.net/forum?id=DeG07_TcZvT&noteId=Pqz0ck3NrY) | 전부 합성이다. Asleep at the Wheel 식 역공 | encoder 읽기가 정상임을 보여 지각 OOD 를 배제한다. SSv2 에 가림·끊김 조건. 실영상 롤아웃 1건 (추적기 GT) | matched pair (context 비트 동일), 하한 논리, testbed framing |
| 10 | 점수가 친숙도를 잰다 | [y5Dy](https://openreview.net/forum?id=6UJf6B8RZ8&noteId=mYaB0rsKA3) | surprise 채점 전체가 해당 | 복사 기준선과 인과 표적을 병기한다 | — |
| 11 | 정의가 없다, kinematics 와 dynamics 혼동, 상태 표현은 하나가 아니다 | [ezgd](https://openreview.net/forum?id=1OkVexYLct&noteId=aVVDtJTtJE), [iWcy](https://openreview.net/forum?id=aijGVmEG9Y&noteId=EMFJTZe6Sj), [fJH5](https://openreview.net/forum?id=DCbQUijwtf&noteId=Q472XAZ24e), [Genp](https://openreview.net/forum?id=i9npQatSev&noteId=tBfatOvBUz) | 'retrieval' 의 조작적 정의가 없다 | — | §1-2 표를 본문에 둔다. kinematics 로 한정한다. 'retrieval' 은 "문맥 경계 조회" 로 정의한다 |
| 12 | 대안 가설 (context 길이 등) | [PW3t](https://openreview.net/forum?id=Xpf5x3mLvn&noteId=hiw7bPrz0U), [9GPH](https://openreview.net/forum?id=Xpf5x3mLvn&noteId=07jQGHXxI9), [Mirage ccWP](https://openreview.net/forum?id=ITw9edRDlD&noteId=IHydEpdJy5) | 대응이 문서마다 흩어져 있다 | 이미 있는 것: C=4/8/12/16 무차이 (CLAUDE.md §6), H1 의 C8/C16/C32 | 대안 가설 표: context 길이, 튜블릿 해상도, RoPE, 표적 번짐, mask token, 도메인, 목적함수 (discussion) |
| 13 | 통계, 규모, 누수, forking paths | [86Gp](https://openreview.net/forum?id=i9npQatSev&noteId=OguTYHoOcK), [khH4](https://openreview.net/forum?id=Q6a9W6kzv5&noteId=1P5ZXUpkV0), [Méloux et al.](https://arxiv.org/abs/2510.00845) | 궤적 84개. H23·H5·H8 은 clip 단위 CI. 가설 9개와 여러 차례 철회 | 궤적 단위 bootstrap, 주 지표·주 셀 사전 고정, hold-out 1회, 다중 비교 보정 | — |
| 14 | 과대 문장 | [9tpf](https://openreview.net/forum?id=ZyLkNVHBZF&noteId=KG2exrhEOy), [rGrg](https://openreview.net/forum?id=XdLgOm5giq&noteId=7tuuGAguoe) | "does not evolve latent state" 가 전칭이다 | — | 초록을 가장 약한 증거 수준에 맞춘다 |
| 15 | IntPhys 복사 결과 역공 | (도출) | w32 는 +3.9 n.s. 인데 w16 은 +12.2 유의. 88.89 는 격자 최대값이라 descriptive 다 | 전 격자 표. 복사 기준선에도 같은 선택 규칙 적용 | "어느 방향으로도 약하다" 로만 쓴다 |
| 16 | 시간 척도 | (도출) | 튜블릿 단위로 보고한다 | P2 stride sweep (적대 검증 뒤) | 초와 거리 단위로 쓴다 |
| 17 | knockout·렌즈 방법이 취약하다 | [Zhang & Nanda](https://arxiv.org/abs/2309.16042), [Optimal ablation](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c55e6792923cc16fd6ed5c3f672420a5-Abstract-Conference.html), [Parodi et al.](https://arxiv.org/abs/2604.14433) | L1 +64% 로 분포 밖. 초기층 렌즈 cos 0.32 | resample/mean ablation, path patching, tuned lens | 보강하지 못하면 부록으로 내린다 |
| 18 | 내부 모순으로 보인다 | (도출) | p probe 는 shape 98.46 인데 "미래에 물체가 없다" 고 한다 | — | "정보는 있고 자리가 없다" 로 통일한다 |
| 19 | new predictor 는 2-AC 에서 action 만 뺀 것이다 | (도출) | — | 2-AC 식 무액션 기준선을 직접 학습하고 부품별 ablation 한다 | — |

**리뷰 과정에 대한 메모.**
- **리뷰어 풀.** Garrido·Bordes·Joseph (Meta FAIR) 계열이 리뷰어일 가능성이 높다. 프로토콜 세부(AvgSurprise, A.8 격자, Filtered)가 하나만 어긋나도 신뢰를 잃는다.
- **리뷰 잡음이 크다.** NeurIPS 2021 일관성 실험에서 한 위원회가 채택한 논문의 50.6% 를 다른 위원회가 떨어뜨렸다. 중간값 리뷰어가 1쪽에서 한 문장 주장을 이해하도록 쓰는 것이 가장 싼 보험이다.
- **rebuttal 규칙이 다르다.** CVPR 2026 은 "reviewers should not request substantial additional experiments for the rebuttal" 이다. ICLR·ICML·NeurIPS 에서는 rebuttal 에 실험을 더해 살아난 사례가 많다 (Binding, Interpreting Physics, Morpheus).
- **가이드라인은 분석 논문을 인정한다.**
  - CVPR/ICCV: "does not exceed the state-of-the-art ... is not grounds for rejection by itself"
  - ICLR: "does not necessarily require state-of-the-art results"
  - NeurIPS: "a work that provides novel insights by evaluating existing methods ... is also equally valuable"
- **가이드라인과 실제 리뷰의 간극.** 이것을 메운 논문은 둘 중 하나를 했다. (a) 기제 + 최소 개입 (Registers, PhysBench). (b) 기존 진단을 뒤집는 새 측정 도구 (Vafa 2024, Mirage, LikePhys).

---

## 6. 인용해야 할 필수 목록

### 6-1. 반드시 (빠지면 리젝 사유)

**검증 대상과 사용처**
- Garrido et al. (arXiv 2502.11831)
- IntPhys 2 (arXiv 2506.09849, NeurIPS 2025 D&B reject)
- V-JEPA 2 (arXiv 2506.09985)
- V-JEPA (arXiv 2404.08471)
- IntPhys (TPAMI 2022, arXiv 1803.07616)
- WMReward (CVPR 2026, 2601.10553)
- TAP-JEPA (2606.00662)

**개념 선점**
- StEvo-Bench (ECCV 2026, 2603.13215)
- WRBench (2606.20545)
- Mechanistic View (2601.17067)

**직접 경쟁**
- HERA (2608.05523)
- Interpreting Physics in Video World Models (ICML 2026, 2602.07050)
- Causal State Variables in V-JEPA 2 Latents (ICML 2026 WS)
- DINO-world (2507.19468)
- DINO-WM (ICML 2025)
- JEPA-WMs (2512.24497)
- Nayebi et al. (NeurIPS 2023)
- UWM-JEPA (2605.25313)
- Causal-JEPA (ICML 2026)

**반례**
- Semigroup-JEPA (2609.10464)
- GRWM (2510.26782)
- Physion (NeurIPS 2021 D&B)

**정의**
- PlaNet (ICML 2019)
- LeCun 2022 (OpenReview)
- Kaelbling et al. 1998
- PSR 2001
- When Does LeJEPA Learn a World Model? (2605.26379)

### 6-2. 방법 정당화

- **probing**: Alain & Bengio 2016, Hewitt & Liang (EMNLP 2019), Voita & Titov (EMNLP 2020), Ravichander et al. (EACL 2021), Belinkov (CL 2022), Observer Effect (2602.12218)
- **개입**: Elazar et al. (TACL 2021), LEACE (NeurIPS 2023), DAS (CLeaR 2024), Causal Abstraction (JMLR 2025), Makelov et al. (ICLR 2024), Canby et al. (IJCNLP-AACL 2025), Huang & Chang (2510.09794)
- **patching·ablation**: Zhang & Nanda (ICLR 2024), Heimersheim & Nanda (2404.15255), Li & Janson (NeurIPS 2024), Parodi et al. (2604.14433), Hase et al. (NeurIPS 2023)
- **렌즈**: logit lens (2020), tuned lens (2303.08112), Future Lens (CoNLL 2023)
- **통계·기하**: Méloux et al. (2510.00845), Davari et al. (ICLR 2023)
- **world model 평가**: Vafa (NeurIPS 2024; ICML 2025), Othello-GPT (ICLR 2023), Kang et al. (ICML 2025)
- **복사 기준선**: DINO-Foresight (NeurIPS 2025), Nayebi (No Dynamics)
- **VoE surprise 의 원래 정의**: ADEPT (NeurIPS 2019), PLATO (NHB 2022)
- **hedging**: Mathieu et al. (ICLR 2016)

### 6-3. 설계

- **상태·기억**: RSSM/PlaNet, Dreamer, DreamerV3, TD-VAE, Gregor et al. (NeurIPS 2019), S4WM, R2I, SlotSSM, SlotFormer, Long-Context SSM (2505.20171), Echo-Memory (2606.09803)
- **가림 지속**: Loci-Looped (2310.10372), PermaTrack, RAM, TSA (대조), SAMURAI
- **되먹임 안정화**: GNS, MP-PDE, Diffusion Forcing, Self Forcing, GameNGen, PLDM, Dreamer 4 (2509.24527)
- **먼 지평**: DPWM (2608.07420), Clockwork VAE, Hierarchical Planning (2604.03208)
- **belief 목적**: Belief State Transformer, NextLat
- **확률적 predictor**: Branch-JEPA (2607.05238), VJEPA (2601.14354)

### 6-4. 배경

- **world model 계보**: World Models (NeurIPS 2018), Genie (ICML 2024), Genie 3 blog, Sora report
- **직관물리 인지과학**: Battaglia et al. (PNAS 2013)
- **VoE 벤치마크·데이터**: InfLevel (TMLR 2022), OPNet, TCOW, Physion++ (NeurIPS 2023 D&B), Opinion (2512.06232)
- **use case 이론**: General agents need world models (ICML 2025)
- **외부 비판**: Critiques of World Models (2507.05169)
- **유사 현상**: Objects in Generated Videos Are Slower (2512.02016), Physics-IQ, Asleep at the Wheel (2608.01336), GEOPHYS (2606.20707), Punzo et al. (2606.09646), V-JEPA 2.1 (2603.14482), SR-JEPA (2608.05774)
- **글쓰기 체크리스트**: Troubling Trends in ML Scholarship (1807.03341)

### 6-5. 인용 주의와 금지

**인용 금지**
- 2607.16274 *The JEPA Predictor: A Transferable Operator for Occluded Feature Completion*: 2026-07-21 에 "major mistakes" 로 **철회**됐다.
- α 분해 수치 (CLAUDE.md).
- 영상 예측 문헌의 'copy-last-frame 이 Human3.6M 에서 모든 모델을 이긴다': 출처를 확정하지 못했다.

**수치를 인용하기 전에 원문 확인**
- IntPhys 2 의 V-JEPA 2 수치: fetch 요약은 main 57.5 인데 HERA 기준선은 52.57 이다. 프로토콜이 다르다.
- Joseph et al. 의 수치 (R² 0.97→0.14, IntPhys 78.3→61.7): WebFetch 요약값이다. 가속 실험의 split 방식도 확인하지 못했다.
- V-JEPA 2 Table 20: 레포 README 의 전사값이다 (encoder 만 61.3/57.0/39.1, predictor 만 48.7/34.7/20.2, 둘 다 63.6/57.1/39.7).
- Hierarchical Planning 의 0% → 70%: 초록 기준이다.
- 2026 preprint 다수는 미심사다: WRBench, UWM-JEPA, Semigroup-JEPA, DPWM, TSA, Echo-Memory, HERA, Asleep at the Wheel, Observer Effect, GEOPHYS, Punzo.

**출처의 성격이 약한 것**
- LeCun 2024 게시글: SNS 이므로 각주로만 쓴다.
- LeCun 2022 원문 PDF: OpenReview 챌린지 때문에 2차 인용으로만 확인했다.
- What Can Latent WMs Know? (2607.27017): 저평판 신규 preprint 라 틀만 참고한다.

**서지·venue 불확실**
- 제목이 바뀐 논문: 2606.28455 ('Permanence Fields…' → 'Event-Conditioned Diagnostics…').
- venue 미확인: Loci-Looped 의 ICANN 2024, Self Forcing, GameNGen, JEPA-WMs (OpenReview 등재만 확인), Objects Slower 의 CVPR 2026 Findings, Garrido 의 동료심사 판.
- 보고서 사이 불일치:
  - V-JEPA 의 TMLR 2024: R1 은 TMLR 로, R3 는 미확인으로 적었다.
  - Interpreting Physics: ICML 2026 로 적는다 (R2·R4 OpenReview 확인; R1·R3 는 arXiv 로만 적음).
  - IntPhys 2: NeurIPS 2025 D&B Reject 로 적는다 (R2·R4 OpenReview API; R1 은 미확인).
- Morpheus: ICLR'26 판과 ICML'26 판이 같은 작업이라는 것은 요약 기준 추정이다.
- Geva 2023·Neo 2025 의 'attention knockout' 방법명: 원문을 기억에 의존해 적었다.

**리뷰 근거의 가중치**
- XtX3 리뷰는 저자가 AI 생성이라고 지적한 리뷰다. 리뷰어 행태의 근거로는 가중치를 낮춘다.

**문장 금지**
- "최초" 문장은 추가 검색 전까지 쓰지 않는다. R2·R4 가 WebSearch 한도를 소진해 검색이 망라적이지 않다.

---

## 7. 다음 단계 (비용 순, frozen 우선)

메모리 규칙("공식·계획 먼저")에 따라, 새 지표는 기호·공식·판정 규칙을 먼저 고정하고 확인받은 뒤 돌린다.

1. 서술만 고친다 (비용 0): 동기 문장에서 planning 제외, 차별화 문단, §1-2 정의 표, 표현 정리.
2. M2 와 P2 의 적대 검증.
3. probe 대조군: 무작위 초기화 ViT-H, 픽셀·광류, 셔플 selectivity, MDL.
4. closure: z → t+k 위치 vs p 슬롯 k, 그리고 0 파라미터 운반 외삽 (캐시만 필요).
5. 개입: LEACE / DAS 로 z 의 속도·방향을 건드리고 p 의 이동을 본다 (predictor 재실행).
6. IntPhys 전 격자 복사 표 (선택 규칙 동일). v_T 비례성과 믿음 단면 (hedging 대안).
7. 궤적 단위 CI 를 일괄 적용하고 주 셀을 사전 고정한다.
8. V-JEPA 2.1 과 ViT-L/g 에서 세 시그니처를 재현한다.
9. tuned lens, resample knockout (못 하면 부록으로).
10. (사용자 결정 뒤) 2-AC 식 무액션 기준선과 부품별 ablation.

---

## 이 문서를 만든 방법

- 네 보고(R1–R4)의 papers, insights, risks, report_md 를 합쳤다. 보고에 없는 인용은 넣지 않았다.
- 레포는 읽기만 했고 파일은 고치지 않았다. 확인한 파일:
  - `configs/train/vith16/pretrain-256px-16f.yaml`
  - `configs/train/vith16/cooldown-256px-64f.yaml`
  - `z_research/anticipation/EK100/README.md`
  - `auto_research/README.md`
  - `auto_research/Archive/PLAN_ENCODER_PREDICTOR_2026-09-25.md`, `P2_STRIDE_SWEEP_HORIZON_UNIT_2026-09-25.md`, `M2_BOUNDARY_PATCH_2026-09-25.md` (앞부분만)

---

# 영역별 보고


---

## 영역: (latent) world model 과 state evolution 의 정의, 그리고 JEPA 계열 predictor 에서 이를 누가 이미 주장하거나 검증했는가

# World model 과 state evolution 의 정의: 누가 이미 주장·검증했나 (2026-09-25)

> 범위: 정의 계보(고전→현대), JEPA predictor 선행 검증, 스쿠프 위험, 정의에서 나오는 리뷰어 공격과 new predictor 설계 원칙.
> 레포 파일은 고치지 않았다. 사실 확인은 두 가지뿐이다.
> - V-JEPA 2 사전학습 mask config 를 읽기 전용으로 확인했다(`configs/train/vith16/pretrain-256px-16f.yaml`, `cooldown-256px-64f.yaml`).
> - 이미 레포에 있는 문헌 목록(`auto_research/Archive/REVIEW_CVPR_ORAL_2026-09-25.md`)과 겹치지 않게 보완했다.
>
> **한계**: OpenReview 는 브라우저 챌린지에 막혀 실제 리뷰 원문을 읽지 못했다.

## 0. 결론 먼저

1. **채택할 정의.** 영문 한 줄:
   *"A latent world model maintains a belief state that is sufficient for predicting the latent future, and advances that state with a transition that stays predictive when no new observation arrives."*
   - 계보: POMDP belief / PSR → RSSM 의 prior 와 posterior → LeCun 의 `s(t+1)=Pred(h(t), s(t), z(t), a(t))`.
   - 네 능력(읽기·지속·전이·영속)은 이 문장을 조작화한 것이다.
2. **개념 자체는 새롭지 않다.** 생성형 video world model 쪽에 이미 있다.
   - **StEvo-Bench** (ECCV 2026): "states should correctly evolve, even if not observed".
   - **WRBench** (2606.20545): 'tracking shot' 실패.
   - 둘 다 **JEPA·latent predictor 는 재지 않았다.**
   - 우리 몫: 이 정의를 latent 예측 모델로 옮기고, 병목을 encoder 가 아닌 predictor 에 국소화하고, surprise 채점이 그 실패를 가린다는 것을 보이는 것.
3. **'왜 사전학습 predictor 인가' 에 가장 강한 답이 config 에 있다.**
   - V-JEPA 2 사전학습 마스크는 tube 마스크다(temporal_scale [1.0, 1.0]). 공간 블록이 모든 시간에 걸쳐 가려진다.
   - 그래서 릴리즈 predictor 는 "과거에는 보이고 미래만 가려진" 과제를 **학습한 적이 없다.**
   - 그런데 Garrido·IntPhys 2·EK100 은 바로 이 predictor 를 미래 마스크로 써서 직관물리·예측 능력을 귀속시킨다.

---

## 1. 제안 정의와 네 능력의 대응

**기호.**
- 관측 o_t, encoder E, 문맥 상태 s_t = E(o_{t−C+1:t}).
- predictor P, 예측 ŝ_{t+k} = P_k(s_t).
- readout R: 학습 없는 템플릿이나 고정 자. E(o_{t+k}) 에서 검증한 것을 쓴다.

**정의 (passive latent world model).** (E, P) 가 지평 H 에서 **상태를 진화시킨다**는 것은, 모든 k ≤ H 에 대해 다음이 성립한다는 뜻이다.

| 요건 | 형식 | 우리 능력 | 문헌 근거 |
|---|---|---|---|
| **읽기 (충분성)** | R(s_t) 가 진짜 상태(정체성·위치·속도·방향)를 복원한다. **predictor 가 받는 층에서.** | 1 읽기 | PSR (Littman 2001), belief probe (Guo 2018; Gregor 2019), Joseph 2026 (층 문제) |
| **closure** | z_{t+k} = E(o_{t+k}) 가 s_{≤t} 로 예측 가능하다 (정보가 입력에 있다) | 1 읽기의 2단 | Semigroup-JEPA 의 closure 항 |
| **지속** | R(P_k(s_t)) 가 진실 자리에 물체를 둔다. 수명이 동역학 불확실성보다 빨리 줄지 않는다 | 2 지속 | RSSM prior rollout, Dreamer imagination |
| **전이** | R(P_k(s_t)) 가 **copy (s_t 유지)** 와 **등속 외삽**을 넘는다. 문맥이 다르면 미래도 다르다 | 3 전이 | Physion++ (문맥 추론), Kang 2025 |
| **영속** | 물체가 t 에서 가려져 있어도 위 두 줄이 같은 수준으로 성립한다 | 4 영속 | IntPhys 2 의 Permanence 정의, OPNet, ADEPT, Loci-Looped |
| **합성 (기제)** | P(P(s)) ≈ P_2(s) — 되먹여도 진행한다 | (R 재귀) | PlaNet latent overshooting, V-JEPA 2-AC rollout loss |

**꼭 나눠야 할 두 장면.**
- (a) 관측 지평 너머의 진화. 미래는 원래 안 보인다. 해당: P32 먼 슬롯, 자기 되먹임 → RSSM 의 **prior rollout**.
- (b) 문맥 안의 관측 결손. 해당: v11 late 가림 → **관측 누락 중 filtering**.
- 지금 README 는 두 장면을 한 문장으로 묶는다.

**정의가 벤치마크 감사를 낳는다.** ADEPT 이래 VoE 의 surprise 는 "진화한 belief 대비 관측의 불일치" 다. 따라서 surprise 가 상태 진화의 증거가 되려면 copy·지속 기준선을 넘어야 한다. 이 요건을 정의 절에 넣으면 H6 의 copy 85.00 vs p 88.89 가 정의에서 곧바로 나온 검정이 된다.

## 2. 정의의 계보

| 층 | 대표 | 정의 핵심 | 우리에게 |
|---|---|---|---|
| 인지과학 | Battaglia 2013 PNAS; IntPhys 2 ("persist … even when out of sight") | 관측 안 된 것까지 포함한 근사 시뮬레이션 | 배경. "물리 엔진" 은 묻지 않는다 |
| 제어·RL | Kaelbling 1998 (POMDP belief); Littman 2001 (PSR); TD-VAE 2019 | 상태 = 미래 예측의 충분통계량 | "예측적 충분성" 절 |
| latent WM | Ha & Schmidhuber 2018; **PlaNet 2019**; Dreamer 2020; DreamerV3 2025 | posterior(관측) / prior(관측 없음) 분리. 한 걸음 학습은 여러 걸음을 학습하지 않는다 | 정의의 형식 틀. H5 의 해석 |
| JEPA | LeCun 2022 ("estimate missing information … not provided by perception"); LeCun 2024 게시글 `Pred(h(t), s(t), …)` | predictor 가 **이전 상태 s(t)** 를 받는다 | 릴리즈 predictor 에는 s(t) 가 없다 → "관측 앵커 조회" 를 정의 차원에서 받친다 |
| 표현 중심 정의 | Klindt·LeCun·Balestriero 2026 (선형 식별가능성); Alrasheed 2026 (encoder 견고성만 평가) | world model = encoder 성질 | 충돌. 서론에서 "읽기 + 진화" 로 흡수해야 한다 |
| 생성형 WM | Sora 2024, Genie 2024, Genie 3 2025 (permanence 를 주장) | 시뮬레이터 = 일관성과 permanence | StEvo·WRBench 가 반박 |
| 평가 방법론 | Vafa 2024/2025; Physics-IQ; Kang 2025; WorldModelBench; WorldScore; World-in-World | 점수 ≠ 상태 모형 | copy 기준선 논리의 가족 |

## 3. 누가 이미 주장·검증했나

### 3-1. 생성형 video world model (개념 선점)

- **StEvo-Bench** (Ma, Liufu, Gkioxari; ECCV 2026; arXiv 2603.13215)
  - 가림막 삽입, 소등, lookaway 로 관측을 끊는다.
  - Veo 3·Sora 2·Genie 3 등이 **10% 미만**이다. 진화가 멈추거나 장면을 freeze 한다.
  - memory 기반 모델은 정적 장면 편향을 오히려 강화한다.
  - V-JEPA2·DINO-WM 은 관련연구에서 'latent world state' 로 언급만 한다.
- **WRBench** (arXiv 2606.20545)
  - 23개 모델이 "resuming a returning target in the state at which it was abandoned" 한다.
  - scale 이나 geometry prior 를 넣어도 변하지 않는다.
- **Mechanistic View** (2601.17067): video 모델은 "stateless" 이고, 평가를 "physical persistence" 로 옮기자고 한다.
- **판정**: 개념의 최초 주장은 불가능하다. 우리 latent 결과(앵커링, 등속 외삽보다 뒤처짐, 되먹임 멈춤)는 생성형 실패의 latent 대응이다. **"두 패러다임에서 같은 실패"** 를 일반성의 근거로 쓴다.

### 3-2. JEPA predictor 를 직접 다룬 것

| 연구 | 무엇을 했나 | 우리와의 거리 |
|---|---|---|
| Garrido 2025 (2502.11831) | frozen V-JEPA predictor surprise → IntPhys 창발. copy 기준선 없음. 3–4 s memory 한계 자인 | 검증 대상 |
| IntPhys 2 (2506.09849) | V-JEPA 2 가 chance. "stricter … short term memory". 프레임을 더 넣으면 하락 | 원인을 memory 로 이미 지목. 우리는 모듈·기제까지 좁혀야 한다 |
| V-JEPA 2 (2506.09985) | EK100: predictor 가 1초 뒤를 예측해 encoder 출력과 이어 붙인다. planning 에는 **새 predictor** + rollout loss. "error accumulation" 자인 | 사용처 근거 + frozen encoder 설계의 선례 |
| HERA (2608.05523) | frozen V-JEPA 2 에 기억 라우팅 어댑터 (IntPhys2 52.57→54.35) | 동기 거의 동일. 차별점은 학습 없는 측정·기준선 |
| DINO-world (2507.19468) | frozen DINOv2 + causal 미래 predictor. IntPhys 91.3 (V-JEPA-H 89.4). **약 1 s 에서 모든 예측 부정확** | new predictor 가 자명한 해결이 아니라는 경고 |
| Joseph 2026 (2602.07050) | V-JEPA 2 encoder 층별 physics. 방향은 약 1/3 깊이. **층마다 새 predictor** 를 학습하면 중간층이 낫다 | "encoder 충분" 의 층 한정 |
| Semigroup-JEPA (2609.10464) | rollout loss 가 encoder 를 "carry-forward 가능한 feature" 로 만든다 | "encoder 충분" 반례 (장난감 규모) |
| SR-JEPA (2608.05774) | 3D 에서 지운 개체를 predictor 가 공간적으로 채운다 | 공간 채움은 된다. 우리 주장은 시간 진화로 한정 |

**공백**: "사전학습 V-JEPA 2 predictor 가 관측이 끊긴 뒤 latent 상태를 진화시키는가" 를 **학습 없는 readout + copy·CV·위치 귀무 기준선**으로 잰 연구는 찾지 못했다. 검색은 망라적이지 않다.

## 4. "encoder is enough" 를 어떻게 써야 하나

**지지 문헌.**
- V-JEPA 2 SSv2 77.3 (frozen encoder probe).
- Joseph: speed·accel 은 초기층부터 있다.
- DINO-WM, DINO-world, V-JEPA 2-AC, JEPA-WMs 가 모두 frozen encoder 를 전제한다.
- Physion (2021) 의 "지각이 병목" 결론과 대비하는 서사가 가능하다: **"The bottleneck has moved."**

**반례와 한정.**
- Semigroup-JEPA: closure 는 encoder 가 정한다.
- Joseph: 방향은 출력층 쪽으로 약해진다. predictor 입력은 최종층이다.
- probing 비판 (Hewitt & Liang 2019; Ravichander 2021).
- "What Can Latent World Models Know?" (2607.27017): 목적함수가 무엇이 남는지를 정한다.

**권장 문장 (상대형).** "predictor 가 받는 최종층 토큰에 실패 조건의 상태 정보가 이미 있고, 실패는 그 정보를 진화시키는 쪽에서 난다."

**필요한 증거 두 층.**
1. **decodability**: 최종층에서 위치·속도·방향·정체성을 읽는다. 궤적 단위 split, control task, 층별 곡선을 붙인다.
2. **closure**:
   - 학습 없는 판: z 에서 읽은 상태를 등속 외삽해 z_{t+k} 의 readout 과 비교한다.
   - 학습하는 판: 작은 predictor.

**SSv2 좌/우 방향 대조.** 데이터 경로 `/local_datasets/vlm_direction/ssv2vp` 는 이 노드에서 확인되지 않았다(다른 노드일 가능성).
- 좌우 반전 영상에서 출력이 뒤집히는가.
- 시간 역재생에서 뒤집히는가.
- 단일 프레임·프레임 섞기 기준선.
- **최종층 수치가 천장인가**가 Joseph 결과와의 판정점이다.

## 5. 예상 리뷰어 공격 (정의 관련)과 답

| 공격 | 출처 | 답 |
|---|---|---|
| "사전학습 predictor 는 pretraining head 다. world model 이라 주장한 적 없다 — straw man" | tube 마스크 사실, V-JEPA 2-AC 재학습 | 사용처(Garrido, IntPhys 2, EK100, HERA)에 묶는다. "as used in these evaluations" 로 범위를 한정한다. tube 마스크 사실은 약점이 아니라 **평가 타당성 문제**로 쓴다 |
| "world model 은 행동 조건부다" | Genie, Dreamer, LeCun 의 a(t) | "passive / action-free world model" 로 명시한다. 근거는 LeCun 의 역할 (1) 과 RSSM 에서 a 를 뺀 형태 |
| "StEvo·WRBench 가 이미 보였다" | 3-1 | latent 로 확장, 모듈 국소화, surprise 채점 감사가 새롭다. 같은 실패의 일반성으로 쓴다 |
| "encoder 도 부족하다" | Semigroup-JEPA, Joseph, Physion | 최종층 decodability + closure. 주장은 상대형으로 |
| "결정론 L1 이니 hedging 은 당연하다" | LeCun z, Branch-JEPA | 앵커링 대 hedging 을 가르는 믿음 단면 검정 |
| "V-JEPA 2 는 정의상 이미 world model (선형 식별가능)" | Klindt·LeCun·Balestriero 2026 | 서론에서 world model = 읽기(encoder) + 진화(predictor) 로 두 용법을 흡수한다 |
| "몇 튜블릿은 모델 시간으로 몇 초냐" | fps 4 대 30 fps | 초 단위로 보고하고 stride 통제. Garrido 3–4 s, DINO-world 약 1 s 와 같은 축에 놓는다 |

## 6. new predictor 설계 원칙 (문헌 근거)

1. **명시적 재귀 상태 s(t)**
   - 근거: LeCun 정의. RSSM 의 deterministic path (PlaNet: 순수 확률 전이는 기억을 못 함).
2. **관측 없는 여러 걸음 open-loop 학습**
   - 근거: PlaNet latent overshooting, Gregor 2019, V-JEPA 2-AC rollout loss, Semigroup-JEPA.
3. **관측 결손 게이팅**
   - 관측이 없으면 prior 를 쓰고, 있으면 관측으로 고친다.
   - 근거: Loci-Looped, RAM/PermaTrack.
4. **인과 표적**: 양방향 target 누수를 제거한다 (레포 H6).
5. **순수 상대 위치 또는 절대 시간**
   - 근거: DINO-world timestamps, 레포 H4 의 RoPE 비상대성.
6. **다중 가설 / 잠재변수 z**
   - 근거: LeCun, Branch-JEPA.
   - 목적: hedging 과 앵커링을 가를 수 있게 한다.

**경고**
- DINO-world 는 2·5 를 일부 갖추고도 약 1 s 에서 무너졌다.
- 따라서 new predictor 는 **StEvo 식 관측 개입 + copy·CV 기준선**을 넘을 때만 기여가 된다.
- auto_research 범위(새 학습 없음)와 충돌한다. 결정이 필요하다.

## 7. 인용 주의

- IntPhys 2 의 V-JEPA 2 수치는 요약상 main 57.51 인데 HERA 기준선은 52.57 이다. 프로토콜이 다르다. 원 표를 확인해야 한다.
- 2606.28455 는 제목이 개정됐다.
- 2607.16274 는 **철회**됐다.
- "Objects … Slower" 의 CVPR 2026 Findings 표기는 검색 요약 기준이다.
- LeCun 2024 정의는 SNS 게시다. 각주로만 쓴다.
- LeCun 2022 원문 PDF 는 OpenReview 챌린지로 직접 확인하지 못했다. 2차 인용으로 확인했다.

## Sources

- [World Models / Recurrent World Models (1809.01999)](https://arxiv.org/abs/1809.01999) · [PlaNet (1811.04551)](https://arxiv.org/abs/1811.04551) · [Dreamer (1912.01603)](https://arxiv.org/abs/1912.01603) · [DreamerV3, Nature 2025](https://www.nature.com/articles/s41586-025-08744-2)
- [LeCun 2022, OpenReview](https://openreview.net/pdf?id=BZ5a1r-kVsf) · [LeCun world model 정의 게시글](https://www.linkedin.com/posts/yann-lecun_lots-of-confusion-about-what-a-world-model-activity-7165738293223931904-vdgR)
- [Kaelbling et al. 1998](https://www.sciencedirect.com/science/article/pii/S000437029800023X) · [PSR 2001](https://proceedings.neurips.cc/paper/2001/hash/1e4d36177d71bbb3558e43af9577d70e-Abstract.html) · [TD-VAE](https://arxiv.org/abs/1806.03107) · [Gregor 2019](https://arxiv.org/abs/1906.09237) · [Guo 2018](https://arxiv.org/abs/1811.06407) · [Shai 2024](https://arxiv.org/abs/2405.15943)
- [V-JEPA](https://arxiv.org/abs/2404.08471) · [V-JEPA 2](https://arxiv.org/abs/2506.09985) · [Garrido 2025](https://arxiv.org/abs/2502.11831) · [IntPhys 2](https://arxiv.org/abs/2506.09849) · [IntPhys](https://arxiv.org/abs/1803.07616)
- [DINO-WM](https://arxiv.org/abs/2411.04983) · [DINO-world](https://arxiv.org/abs/2507.19468) · [JEPA-WMs](https://arxiv.org/abs/2512.24497) · [Genie](https://arxiv.org/abs/2402.15391) · [Genie 3 blog](https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/) · [Sora report](https://openai.com/index/video-generation-models-as-world-simulators/)
- [StEvo-Bench](https://arxiv.org/abs/2603.13215) · [ECCV 2026 poster](https://eccv.ecva.net/virtual/2026/poster/4982) · [WRBench](https://arxiv.org/abs/2606.20545) · [Mechanistic View](https://arxiv.org/abs/2601.17067)
- [HERA](https://arxiv.org/abs/2608.05523) · [Joseph 2026](https://arxiv.org/abs/2602.07050) · [Semigroup-JEPA](https://arxiv.org/abs/2609.10464) · [LeJEPA world model](https://arxiv.org/abs/2605.26379) · [Alrasheed 2026](https://arxiv.org/abs/2605.15618) · [SR-JEPA](https://arxiv.org/abs/2608.05774) · [JEPA predictor occluded completion (철회)](https://arxiv.org/abs/2607.16274) · [Branch-JEPA](https://arxiv.org/abs/2607.05238) · [What Can Latent WMs Know](https://arxiv.org/abs/2607.27017)
- [Kang 2025](https://arxiv.org/abs/2411.02385) · [Physics-IQ](https://arxiv.org/abs/2501.09038) · [Physion](https://arxiv.org/abs/2106.08261) · [Physion++](https://arxiv.org/abs/2306.15668) · [WorldModelBench](https://arxiv.org/abs/2502.20694) · [WorldScore](https://arxiv.org/abs/2504.00983) · [World-in-World](https://arxiv.org/abs/2510.18135) · [Objects … Slower](https://arxiv.org/abs/2512.02016)
- [OPNet](https://arxiv.org/abs/2003.10469) · [PermaTrack](https://openaccess.thecvf.com/content/ICCV2021/html/Tokmakov_Learning_To_Track_With_Object_Permanence_ICCV_2021_paper.html) · [RAM](https://arxiv.org/abs/2204.01784) · [TCOW](https://arxiv.org/abs/2305.03052) · [Loci-Looped](https://arxiv.org/abs/2310.10372) · [PLATO](https://www.nature.com/articles/s41562-022-01394-8) · [ADEPT](https://proceedings.neurips.cc/paper/2019/hash/e88f243bf341ded9b4ced444795c3f17-Abstract.html) · [InfLevel](https://mlanthology.org/tmlr/2022/weihs2022tmlr-benchmarking/) · [Battaglia 2013](https://www.pnas.org/doi/10.1073/pnas.1306572110)
- [Vafa 2024](https://arxiv.org/abs/2406.03689) · [Vafa 2025](https://arxiv.org/abs/2507.06952) · [Richens 2025](https://arxiv.org/abs/2506.01622) · [Critiques of World Models](https://arxiv.org/abs/2507.05169) · [Hewitt & Liang](https://aclanthology.org/D19-1275/) · [Ravichander 2021](https://aclanthology.org/2021.eacl-main.295/) · [R2I](https://arxiv.org/abs/2403.04253) · [2606.28455](https://arxiv.org/abs/2606.28455) · [Su 2025 opinion](https://arxiv.org/abs/2512.06232) · [MBench](https://arxiv.org/abs/2606.00793) · [OpenWorldLib](https://arxiv.org/abs/2604.04707)


### 논문 목록

| 제목 | 학회·연도 | URL | 요점 | 우리와의 관계 |
|---|---|---|---|---|
| World Models (NeurIPS 판: Recurrent World Models Facilitate Policy Evolution) | arXiv 2018 (1803.10122) / NeurIPS 2018 (1809.01999) | https://arxiv.org/abs/1809.01999 | VAE(V) + MDN-RNN(M) 구조. M 이 압축 latent 의 시간 전개를 맡고, 에이전트를 자기 world model 이 만든 '꿈' 안에서 학습시킨다. | '관측을 압축하는 모듈(encoder)'과 '상태를 진화시키는 모듈(predictor)'을 나눈 고전 계보다. 'encoder 는 관측만, 진화는 predictor' 라는 우리 분업 틀의 원형이다. |
| Learning Latent Dynamics for Planning from Pixels (PlaNet) | ICML 2019 | https://arxiv.org/abs/1811.04551 | RSSM 의 filtering posterior q(s_t/o≤t,a<t) 와 관측 없는 prior p(s_t/s_{t-1},a_{t-1}) 로 구성된다. 원문 요지는 두 가지다. 한 걸음 학습에서는 gradient 가 'never traverses a chain of multiple p' 하므로 latent overshooting 으로 여러 걸음 prior 를 posterior 쪽으로 학습시킨다. 또 순수 확률 전이는 'difficult to remember information over multiple time steps' 이라 deterministic path 를 둔다. | '관측 없이 상태를 굴리는 prior' 와 '관측으로 고치는 posterior' 의 구분이 우리 정의(관측이 끊겨도 예측력 유지)의 형식적 근거다. H5 의 되먹임 멈춤은 한 걸음만 학습한 전이의 예상된 한계로 읽힐 수 있다. new predictor 설계 원칙 1순위(여러 걸음 open-loop 학습)의 출처이기도 하다. |
| Dream to Control: Learning Behaviors by Latent Imagination (Dreamer) | ICLR 2020 | https://arxiv.org/abs/1912.01603 | 학습된 RSSM 의 latent 안에서 관측 없이 상상한 궤적(imagination rollout)으로 행동을 학습한다. | '관측 없이 굴린 latent 궤적이 쓸모 있어야 world model' 이라는 기능적 정의의 대표 사례다. |
| Mastering diverse control tasks through world models (DreamerV3) | Nature 640, 2025 | https://www.nature.com/articles/s41586-025-08744-2 | 단일 설정으로 150여 과제에서 imagination 기반 학습이 동작한다. | RSSM 식 world model 이 현재 표준 정의로 쓰인다는 인용용이다. |
| A Path Towards Autonomous Machine Intelligence (v0.9.2) | OpenReview position paper 2022 | https://openreview.net/pdf?id=BZ5a1r-kVsf | world model 의 역할은 두 가지다. (1) estimate missing information about the state of the world not provided by perception, (2) predict plausible future states. 불확실성은 잠재변수 z 로 표현한다. (원문 PDF 는 OpenReview 챌린지에 막혀 2차 인용으로 확인했다.) | (1) 은 곧 '못 볼 때도 상태를 채운다' 로, 우리 정의와 같다. V-JEPA 2 predictor 에 z 가 없다는 점은 hedging 공격의 근거가 된다. |
| LeCun 의 world model 정의 게시글 (X/LinkedIn) | SNS 게시 2024-02 (논문 아님) | https://www.linkedin.com/posts/yann-lecun_lots-of-confusion-about-what-a-world-model-activity-7165738293223931904-vdgR | h(t)=Enc(x(t)); s(t+1)=Pred(h(t), s(t), z(t), a(t)). predictor 가 이전 상태 추정 s(t) 를 입력으로 받는다. | JEPA 제안자 본인의 정의에 재귀 상태 s(t) 가 들어 있다. 릴리즈 V-JEPA 2 predictor 는 Pred(h(t−C:t)) 로, s(t) 가 없는 한 번짜리 사상이다. '관측 앵커 조회' 라는 우리 주장을 정의 차원에서 받쳐 준다. SNS 출처라 본문 각주로만 쓴다. |
| Planning and acting in partially observable stochastic domains | Artificial Intelligence 101, 1998 | https://www.sciencedirect.com/science/article/pii/S000437029800023X | POMDP 의 belief state b(s)=P(s/history) 는 과거의 충분통계량이고, 관측이 없으면 전이 모형만으로 갱신한다. | 'belief state' 용어를 쓸 때 인용하는 고전 정의다. |
| Predictive Representations of State | NIPS 2001 | https://proceedings.neurips.cc/paper/2001/hash/1e4d36177d71bbb3558e43af9577d70e-Abstract.html | 상태를 '미래 관측에 대한 여러 걸음 예측들의 집합' 으로 정의한다(PSR). | '상태 = 미래를 예측하기에 충분한 것' 이라는 행동적 정의의 근거다. 우리 정의의 '예측적 충분성' 절에 쓴다. |
| Temporal Difference Variational Auto-Encoder (TD-VAE) | ICLR 2019 | https://arxiv.org/abs/1806.03107 | 미래 예측에 필요한 모든 정보를 담는 belief state 를 명시적으로 배우고, 여러 걸음 앞을 jumpy 하게 직접 굴린다. | '한 번에 먼 미래 질의(P32)' 와 '걸음별 전개' 를 구분하는 설계 계보다. |
| Shaping Belief States with Generative Environment Models for RL | NeurIPS 2019 | https://arxiv.org/abs/1906.09237 | belief state 를 probe 해 시야 밖으로 나간 건물도 지도에 남아 있음을 보였다. 여러 걸음 예측(overshooting)이 안정된 belief 형성에 결정적이다. | '관측 밖 대상을 belief 에서 probe 로 읽는다' 는 우리 측정의 직접 선례다(단 RL 3D 미로이고 JEPA 는 아니다). overshooting 의 필요성은 new predictor 설계 근거다. |
| Neural Predictive Belief Representations | arXiv 2018 (1811.06407) | https://arxiv.org/abs/1811.06407 | belief = 지금까지 관측의 충분통계량. 표현에서 에이전트 위치 같은 하부 상태를 probe 로 읽어 belief 여부를 평가하고, 여러 걸음 CPC 가 중요하다고 보고한다. | '읽기(1)' 를 probe 로 판정하는 방법론의 계보다. |
| Transformers represent belief state geometry in their residual stream | NeurIPS 2024 | https://arxiv.org/abs/2405.15943 | 최적 예측 이론이 정한 belief state 기하가 residual stream 에 선형으로 나타난다. | representation geometry 로 'encoder 가 상태를 담는다' 를 보이려 할 때 쓸 방법론적 선례다(언어 쪽). |
| Revisiting Feature Prediction for Learning Visual Representations from Video (V-JEPA) | TMLR 2024 | https://arxiv.org/abs/2404.08471 | 마스크는 공간 블록을 'repeat … across the entire temporal dimension' 하는 tube 마스크다(short 8블록×15%, long 2블록×70%). 과거만 보고 미래를 채우는 causal multi-block 은 ablation 에서 더 나빠 채택되지 않았다(K400 71.9 vs 72.9). | 핵심 사실: 사전학습 predictor 는 '과거만 보고 미래 채우기' 과제를 학습한 적이 없다. 레포 config `configs/train/vith16/*.yaml` 의 temporal_scale [1.0, 1.0] 으로 V-JEPA 2 도 같음을 확인했다. Garrido·IntPhys 2·EK100 은 이 predictor 를 학습 분포 밖(미래 마스크)에서 쓴다. |
| V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning | arXiv 2025 (2506.09985) | https://arxiv.org/abs/2506.09985 | EK100 anticipation 에서 predictor 가 '1초 뒤' mask token 을 예측하고 그 출력을 encoder 출력과 이어 attentive probe 에 넣는다. planning 용 V-JEPA 2-AC 는 'freeze the video encoder and learn a new action-conditioned predictor' 이고, teacher forcing 과 2-step rollout loss 를 쓴다. 논문 스스로 'autoregressive prediction suffers from error accumulation' 이라고 적었다. | (a) 사전학습 predictor 가 실제로 '예측' 용도로 쓰인다는 증거다(EK100). (b) Meta 자신이 planning 에는 frozen encoder + 새 predictor 를 택했다. 'encoder 는 충분, predictor 는 새로' 라는 우리 설계를 지지하지만, 동시에 그 설계 자체가 새롭지 않다는 뜻이다. |
| Intuitive physics understanding emerges from self-supervised pretraining on natural videos | arXiv 2025 (2502.11831) | https://arxiv.org/abs/2502.11831 | frozen V-JEPA encoder + predictor 의 미래 예측 오차를 surprise 로 쓴다. IntPhys 에서 object permanence 등이 창발한다고 보고한다(permanence M≈85.7). 기준선은 학습 안 된 네트워크, VideoMAEv2, MLLM 뿐이고 copy 기준선은 없다. 한계로 'limited memory … typically 3–4 seconds' 를 적었다. world model 을 정식으로 정의하지 않는다. | 우리가 검증하려는 주장의 정본이다. 반박이 아니라 '그 점수가 상태 진화를 요구하는가' 로 쓴다. |
| IntPhys 2: Benchmarking Intuitive Physics Understanding In Complex Synthetic Environments | arXiv 2025 (2506.09849), NeurIPS 2025 D&B 제출 (채택 여부 미확인) | https://arxiv.org/abs/2506.09849 | Permanence 를 'Objects persist in space and time, even when out of sight' 로 정의한다. V-JEPA 계열은 main 에서 chance 근처다. 원인으로 'stricter requirements on short term memory' 를 들고, 'decline in performance when additional frames are introduced' 라고 적었다. | motivation 의 두 번째 다리다(IntPhys 1 은 되는데 2 는 무너진다). 저자들이 memory 를 원인으로 이미 지목했으므로 우리는 '어느 모듈의 무엇' 까지 좁혀야 새롭다. (V-JEPA 2 의 정확한 수치는 fetch 요약에서 main 57.5 로 읽혔지만 표를 직접 다시 확인해야 한다.) |
| IntPhys: A Framework and Benchmark for Visual Intuitive Physics Reasoning (IntPhys 2019) | TPAMI 2022 (arXiv 1803.07616) | https://arxiv.org/abs/1803.07616 | 픽셀 매칭 4중항 VoE 설계. 가능/불가능 사건을 open vs occluded 로 나눠 분석한다. | 4중항은 단일 프레임 감지기는 막지만 '문맥 대비 변화 감지기(copy)' 는 막지 못한다. 우리 copy 기준선의 논리적 출발점이다. |
| DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning | ICML 2025 | https://arxiv.org/abs/2411.04983 | frozen DINOv2 patch feature 위에 ViT predictor 만 학습해 zero-shot planning 을 한다. | 'encoder is enough + 새 predictor' 의 대표 선례다. new predictor 제안의 새로움은 여기서 차별화해야 한다. |
| Back to the Features: DINO as a Foundation for Video World Models (DINO-world) | arXiv 2025 (2507.19468) | https://arxiv.org/abs/2507.19468 | frozen DINOv2 + 절대 timestamp 를 쓰는 causal next-frame predictor(teacher forcing)를 대규모 비디오로 학습했다. IntPhys 91.3 vs V-JEPA ViT-H 89.4 (저자는 'sanity check' 로 취급)이다. 'all predictions become inaccurate as the forecasting interval approaches 1 second' 라고 적었다. | 세 가지로 중요하다. (1) 비디오 사전학습 없는 image encoder + 미래 예측 predictor 가 IntPhys 에서 V-JEPA 와 비슷하다. IntPhys 가 video-encoder 의 동역학을 요구하지 않을 수 있다는 방증이다(copy 기준선과 같은 방향). (2) 미래 예측으로 학습한 predictor 도 1초 부근에서 무너진다. 'new predictor = 해결' 이 자명하지 않다는 경고다. (3) 절대 시간 조건화는 설계 참고가 된다. |
| What Drives Success in Physical Planning with Joint-Embedding Predictive World Models? (JEPA-WMs) | arXiv 2025 (2512.24497) | https://arxiv.org/abs/2512.24497 | frozen encoder + 학습 predictor 로 구성한 JEPA-WM 의 설계 선택(encoder, predictor, planner)을 체계적으로 조사했고, DINO-WM 과 V-JEPA 2-AC 를 넘는다. | Meta 내부에서 이미 'JEPA world model = frozen encoder + 학습된 predictor' 가 표준이 됐다는 증거다. 우리 new predictor 절이 겹치는 영역이다. |
| Genie: Generative Interactive Environments | ICML 2024 | https://arxiv.org/abs/2402.15391 | video tokenizer + 자기회귀 dynamics + latent action 으로 구성한 'foundation world model'. | action-conditioned 생성형 world model 정의의 대표다. 'world model 은 행동 조건이 필요하다' 는 공격의 출처가 될 수 있다. |
| Genie 3: A new frontier for world models (blog) | Google DeepMind blog 2025 | https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/ | world model 을 '환경이 어떻게 진화하고 행동이 어떤 영향을 주는지 예측하는 시스템' 으로 규정한다. 약 1분의 visual memory 와 'emergent object permanence' 를 주장한다. | 업계가 object permanence 를 world model 의 요건으로 공개적으로 내세운다는 인용용이다(동료심사 없음). |
| Video generation models as world simulators (Sora 기술 보고) | OpenAI 2024 (블로그/보고서) | https://openai.com/index/video-generation-models-as-world-simulators/ | 가려지거나 화면 밖으로 나간 대상을 유지하는 object permanence 가 scale 로 창발한다고 주장한다. | 'world simulator' 주장이 permanence 를 핵심으로 삼는다는 인용이다. StEvo-Bench 와 WRBench 가 이 주장을 반박한다. |
| Out of Sight, Out of Mind? Evaluating State Evolution in Video World Models (StEvo-Bench) | ECCV 2026 (arXiv 2603.13215) | https://arxiv.org/abs/2603.13215 | 'states should correctly evolve, even if not observed' 가 전제다. 가림막 삽입, 소등, lookaway 로 관측을 끊고 state progress·plausibility·coherence 를 잰다. Veo 3·Sora 2·Genie 3 등이 모두 성공률 10% 미만이고, 관측을 끊으면 진화가 멈추거나 장면을 'freeze' 한다. V-JEPA2·DINO-WM 은 'latent world state' 를 쓴다고 관련연구에서만 언급하고 평가하지 않았다. | 스쿠프 위험 최상이다. 'state evolution' 이라는 용어와 '관측과 분리된 진화' 라는 정의가 이미 ECCV 2026 에 있다. 우리 몫은 이 정의를 latent JEPA predictor 로 옮기고, 병목을 encoder 가 아닌 predictor 에 국소화하고, surprise 채점이 이 실패를 가린다는 점을 보이는 것이다. 반드시 인용한다. |
| Current World Models Lack a Persistent State Core (WRBench) | arXiv 2026 (2606.20545) | https://arxiv.org/abs/2606.20545 | persistent state core = 'an internal world state that keeps evolving over time, decoupled from observation' 이다. 23개 생성형 모델이 'tracking shot' 처럼 동작한다. 돌아온 대상을 버려졌던 상태 그대로 재개할 뿐 안 보이는 동안 사건을 진행시키지 않는다. JEPA 는 평가하지 않았다. | 우리 '관측 앵커 조회(observation-anchored retrieval)' 의 생성형 쌍둥이다. 두 패러다임에서 같은 실패가 나온다는 점은 강점(일반성)으로 쓸 수 있다. 다만 개념의 최초 주장은 할 수 없다. |
| A Mechanistic View on Video Generation as World Models: State and Dynamics | arXiv 2026 (2601.17067) | https://arxiv.org/abs/2601.17067 | 현대 video 모델은 'stateless' 라고 진단하고 State Construction(암묵/명시)과 Dynamics Modeling 을 두 축으로 정리한다. 평가를 'physical persistence' 를 재는 기능 벤치마크로 옮기자고 주장한다. | '상태 구성(encoder)' 과 '동역학(predictor)' 을 나눠 보자는 우리 틀의 동시대 선언문이다. survey 성격이다. |
| HERA: Historical Evidence Routing Adapter for Physical Prediction in Latent World Models | arXiv 2026 (2608.05523) | https://arxiv.org/abs/2608.05523 | frozen V-JEPA 2 predictor 에 기억 라우팅 어댑터를 붙인다. 문제의식은 '나중 예측이 현재 시야에 없는 과거 물체 증거에 의존한다' 이다. IntPhys2 main 52.57 → 54.35. | 'V-JEPA 2 predictor 가 안 보이는 상태를 못 들고 간다' 를 동기로 이미 썼다. 차별점은 우리가 학습 없는 측정·기준선·병목 국소화를 한다는 것이다. 이득 +1.8pt 가 진짜 예측인지 copy 대조로 가를 수 있다. |
| Interpreting Physics in Video World Models | arXiv 2026 (2602.07050) | https://arxiv.org/abs/2602.07050 | V-JEPA 2 L/H/g encoder 의 층별 분석이다. speed·acceleration 은 초기층부터 있고, 운동 방향은 약 1/3 깊이의 'Physics Emergence Zone' 에서 원형 기하로 나타난 뒤 출력층 쪽으로 약해진다. V-JEPA 2-L predictor 를 층마다 새로 학습하면 VoE 에서 중간층 입력이 최종층보다 낫다. | 'encoder is enough' 를 층 단위로 좁혀야 한다는 경고다. predictor 가 받는 최종층에서 방향·속도가 천장 수준으로 읽히는지를 따로 보여야 한다. 'frozen encoder 위 새 predictor' 실험을 Meta 가 이미 층 선택 용도로 했다는 뜻이기도 하다. |
| Semigroup-JEPA: Latent Dynamics Consistency for Zero-Shot Physics Generalization | arXiv 2026 (2609.10464) | https://arxiv.org/abs/2609.10464 | 여러 걸음 rollout loss 를 encoder 로 역전파하면 encoder 가 'features that the predictor can carry forward' 를 남긴다. frozen encoder 교차 실험에서 rollout 학습 encoder 가 새 predictor 로도 rollout 오차를 약 12% 낮춘다. 오차를 closure 항(encoder 가 버린 정보)과 operator mismatch 로 분해한다. | 'encoder is enough' 의 가장 직접적인 반례다(단 ViT-Tiny from scratch, 장난감 물리). 우리는 '충분' 을 closure 로 정의하고, V-JEPA 2 최종층 토큰에서 미래 토큰이 예측 가능함을 보여야 한다. 그 분해식은 우리 병목 국소화의 형식 틀로 빌릴 수 있다. |
| When Does LeJEPA Learn a World Model? | arXiv 2026 (2605.26379) | https://arxiv.org/abs/2605.26379 | (Klindt·LeCun·Balestriero) 'learns a world model' 을 world latent 의 선형 식별가능성(encoder 성질)으로 정의하고, 정상·가법잡음 전이에서 성립 조건을 증명한다. | JEPA 진영 안에서도 world model 이 'encoder 표현 성질' 로 정의되기도 한다는 증거다. 우리 정의는 이를 '읽기(1)' 로 흡수하고, 진화(2~4)를 predictor 쪽 별도 요건으로 둔다. 용어 충돌을 서론에서 정리해야 한다. |
| Latent Video Prediction Learns Better World Models | arXiv 2026 (2605.15618) | https://arxiv.org/abs/2605.15618 | 제목과 달리 frozen encoder 만 평가한다(가림 입력에서 표현 유사도·일관성). world model 을 '견고한 표현' 으로 조작화한다. | 'world model = encoder 품질' 로 쓰는 흔한 관행의 사례다. predictor 를 따로 재야 하는 이유를 보여 준다. 그들의 occlusion 은 입력 가림이고 상태 진화는 아니다. |
| How Far is Video Generation from World Model: A Physical Law Perspective | ICML 2025 | https://arxiv.org/abs/2411.02385 | 2D 운동·충돌 testbed 에서 diffusion video 모델을 scale 한 결과, in-distribution 은 완벽하고 OOD 는 실패한다. 법칙 추상이 아니라 case-based 로 일반화한다. | '합성 testbed 로 world model 의 법칙 일반화를 판정한다' 는 방법론 선례다(생성형). 우리 RollOut/v11 의 정당화에 쓴다. |
| Do generative video models understand physical principles? (Physics-IQ) | arXiv 2025 (2501.09038) | https://arxiv.org/abs/2501.09038 | 실촬영 물리 벤치마크다. 시각 사실성과 물리 이해가 무관하다. | '평가 점수와 능력의 괴리' 라는 같은 계열의 문제 제기다. |
| Physion: Evaluating physical prediction from vision in humans and machines | NeurIPS 2021 Datasets & Benchmarks | https://arxiv.org/abs/2106.08261 | 물리 상태에 직접 접근하는 GNN 이 훨씬 낫다. 결론: 'extracting physical representations of scenes is the main bottleneck'. | 서사 훅이 된다. 2021 년에는 병목이 지각(encoder)이었는데, 웹 규모 SSL encoder 이후 병목이 predictor 로 옮겨 갔는가를 묻는다. 동시에 'encoder 가 병목' 이라는 반대 선례이므로 우리 데이터로 뒤집어야 한다. |
| Physion++: Evaluating Physical Scene Understanding that Requires Online Inference of Different Physical Properties | NeurIPS 2023 Datasets & Benchmarks | https://arxiv.org/abs/2306.15668 | 질량·마찰 같은 잠재 물성을 문맥에서 온라인 추론해야 예측할 수 있는 과제다. 표준 모델은 잠재 물성 추론을 저절로 배우지 않는다. | 사용자 틀의 '맥락 파악은 predictor 역할' 에 해당하는 과제 선례다(전이 능력 3). |
| Learning Object Permanence from Video (OPNet) | ECCV 2020 | https://arxiv.org/abs/2003.10469 | 보임·가림·담김·운반 네 상황에서 안 보이는 물체 위치를 추론한다. | 영속(4)의 과제 분해 선례다. 우리 v11 의 가림은 그중 'occluded' 칸이다. |
| Learning to Track with Object Permanence / Object Permanence Emerges in a Random Walk along Memory | ICCV 2021 / ICML 2022 | https://arxiv.org/abs/2204.01784 | 가림 중 물체 위치를 memory 로 유지·외삽하도록 학습한다. RAM 은 자기지도 memory random walk 에서 permanence 가 창발한다고 보인다. | 영속은 명시적 memory 와 목적함수가 있어야 생긴다는 증거다. new predictor 의 memory/상태 설계 근거가 된다. |
| Tracking through Containers and Occluders in the Wild (TCOW) | CVPR 2023 | https://arxiv.org/abs/2305.03052 | transformer video 모델이 일부 조건에서만 가림 추적에 성공하고 'true notion of object permanence' 에는 격차가 남는다. | CV 커뮤니티에서 permanence 부족이 이미 보고된 영역이라는 맥락이다(tracking, 지도학습). |
| Learning Object Permanence from Videos via Latent Imaginations (Loci-Looped) | ICANN 2024 (arXiv 2310.10372) | https://arxiv.org/abs/2310.10372 | latent imagination 과 관측을 적응적으로 섞는 루프로 가림 중 물체를 추적하고 재등장을 예상한다. 이상 사건에는 surprise 를 보인다. | '관측이 없으면 자기 상상(prior)을 쓰고 있으면 관측을 쓴다' 는 게이팅이 new predictor 설계 원칙의 직접 근거다(RSSM prior/posterior 의 객체 중심판). |
| Intuitive physics learning in a deep-learning model inspired by developmental psychology (PLATO) | Nature Human Behaviour 2022 | https://www.nature.com/articles/s41562-022-01394-8 | VoE 데이터셋과, 객체 수준 표현과 추적을 쓰는 PLATO. 객체 표현이 결정적이다. | VoE 계보 인용이다. '추적되는 객체 상태' 가 있어야 permanence 가 선다는 발달심리식 정의의 ML 구현이다. |
| Modeling Expectation Violation in Intuitive Physics with Coarse Probabilistic Object Representations (ADEPT) | NeurIPS 2019 | https://proceedings.neurips.cc/paper/2019/hash/e88f243bf341ded9b4ced444795c3f17-Abstract.html | 확률적 물리 시뮬레이션 + particle filter 로 '가림 너머로 이어지는 기대' 를 만든다. | 'surprise = 진화한 belief 대비 관측의 불일치' 라는 본래 정의다. 이 정의에 비추면 문맥 대비 변화 감지(copy)는 VoE 가 의도한 측정이 아니다. |
| Benchmarking Progress to Infant-Level Physical Reasoning in AI (InfLevel) | TMLR 2022 | https://mlanthology.org/tmlr/2022/weihs2022tmlr-benchmarking/ | continuity·solidity·gravity VoE. 당시 video 모델은 모두 chance 근처였다. | Garrido 격자의 한 축으로, 외부 벤치마크 계보 인용이다. |
| Simulation as an engine of physical scene understanding | PNAS 2013 | https://www.pnas.org/doi/10.1073/pnas.1306572110 | 인간 직관물리는 관측 안 된 정보를 포함한 근사 확률 시뮬레이션('intuitive physics engine')이다. | '이해 = 상태를 앞으로 굴리는 시뮬레이션' 이라는 인지과학 정의다. 단 우리는 '물리 엔진 여부' 를 묻지 않으므로 배경 인용에만 쓴다. |
| Evaluating the World Model Implicit in a Generative Model / What Has a Foundation Model Found? | NeurIPS 2024 / ICML 2025 | https://arxiv.org/abs/2507.06952 | 기존 진단(다음 토큰 정확도·probe)은 좋은데 Myhill-Nerode 식 상태 일관성이나 inductive-bias probe 로 보면 world model 이 불일관하다. 궤도 데이터로 학습한 모델이 Newton 역학을 쓰지 않는다. | '벤치마크 점수가 좋아도 상태 모형이 없을 수 있다' 는 평가 방법론 계보다. copy 기준선 논리와 같은 가족이다. |
| General agents need world models | ICML 2025 | https://arxiv.org/abs/2506.01622 | 여러 걸음 목표 과제를 일반화하는 에이전트는 반드시 환경의 예측 모형을 담고 있다. | '왜 상태 진화가 풀어야만 하는 문제인가' 의 이론적 근거다(planning use case). |
| Critiques of World Models | arXiv 2025 (2507.05169) | https://arxiv.org/abs/2507.05169 | world model 의 목표를 'simulating all actionable possibilities' 로 보고, 감각 데이터만 쓰는 JEPA 식 encoder-encoder·latent 목적 학파를 비판한다. | 리뷰어가 가져올 수 있는 'JEPA 는 애초에 world model 이 아니다' 라는 외부 비판의 출처다. 우리 결과를 그 비판의 증거로 오용하지 않도록 주장 범위를 한정한다. |
| Objects in Generated Videos Are Slower Than They Appear | CVPR 2026 Findings (arXiv 2512.02016) — 검색 요약 기준 | https://arxiv.org/abs/2512.02016 | video 생성기의 유효 중력이 0.38–2.27 m/s² 로, 물체가 실제보다 느리게 떨어진다. frame-rate 재척도로도 교정되지 않는다. | '등속 외삽보다 뒤처진다(H23)' 의 생성형 대응이다. 사전분포 쪽 수축이 패러다임을 가로지른다는 근거이자, frame-rate 교락 검정의 선례다. |
| Designing and Interpreting Probes with Control Tasks / Probing the Probing Paradigm | EMNLP 2019 / EACL 2021 | https://aclanthology.org/2021.eacl-main.295/ | probe 정확도가 높아도 표현이 과제에 쓰인다는 뜻은 아니다. control task 로 probe 표현력을 통제해야 한다. | 'encoder is enough' 를 probe 만으로 주장하면 반드시 받는 공격의 출처다. 정보 존재(decodable)와 predictor 가 쓸 수 있음(closure)을 분리해 보여야 한다. |
| SR-JEPA: Learning Predictive Latent State in 3D Scenes | arXiv 2026 (2608.05774) | https://arxiv.org/abs/2608.05774 | 물체 하나를 통째로 지운 3D 장면에서 predictor 에 centroid 질의를 던지면 그 개체의 내용을 문맥 의존적으로 채운다(semantic identity 43%). | JEPA predictor 가 '공간적으로' 부재 대상을 채울 수 있다는 반대 방향 증거다. 우리 주장은 '시간적 진화' 에 한정해야 한다. |
| The JEPA Predictor: A Transferable Operator for Occluded Feature Completion | arXiv 2026 (2607.16274) — 철회됨 | https://arxiv.org/abs/2607.16274 | I-JEPA·V-JEPA 2 predictor 를 '가려진 공간 feature 를 채우는 연산자' 로 규정한다. 2026-07-21 'major mistakes' 로 철회됐다. | '사전학습 predictor = 공간 보간 연산자' 라는 시각의 동시대 사례다. 철회됐으므로 인용하지 않는다(기록용). |
| Branch-JEPA: Finite-Support Predictive Distributions for JEPA World Models | arXiv 2026 (2607.05238) | https://arxiv.org/abs/2607.05238 | 대부분의 JEPA world model 은 부분관측·확률 동역학에서도 후속 latent 하나만 낸다. 유한 개 가지로 대체한다. | 결정론 predictor 의 hedging 공격과 그 해법 계보다. new predictor 설계 원칙(다중 가설) 근거다. |
| Mastering Memory Tasks with World Models (R2I) | ICLR 2024 (oral) | https://arxiv.org/abs/2403.04253 | SSM 을 world model 에 넣어 먼 과거 관측 회상과 장기 credit assignment 를 푼다. | '지속' 에 필요한 memory 구조 설계 참고다. |
| Event-Conditioned Diagnostics of Kinematic, Contact, and Object-Permanence Structure in Passive Object-State World Models (구제목: Permanence Fields in Passive Object-State World Models) | arXiv 2026 (2606.28455) | https://arxiv.org/abs/2606.28455 | GRU / Transformer-lite / RSSM-lite 수동(action-free) world model 에서 hard occlusion 중 permanence 구조를 진단한다. 'control specificity remains mixed'. | 'passive world model' 용어의 선례다(행동 조건 없는 world model 도 정의상 성립). 규모가 작아 스쿠프 위험은 낮다. |
| Opinion: Learning Intuitive Physics May Require More than Visual Data | arXiv 2025 (2512.06232) | https://arxiv.org/abs/2512.06232 | SAYCam 으로 학습한 V-JEPA 가 IntPhys2 에서 개선되지 않는다. 데이터 양·분포만으로는 부족할 수 있다. | '데이터가 아니라 구조·목적(= predictor) 문제' 라는 해석과 맞물린다. 단 우리는 목적함수를 원인으로 걸지 않는다(CLAUDE.md). |


---

## 영역: Probing and causal interpretability of video/self-supervised models: V-JEPA/V-JEPA 2 physics probing, probing critiques (control tasks, amnesic probing, causal abstraction), activation patching and attention knockout, logit/tuned lens, regression to the mean of deterministic predictors, copy baseline

# 조사 보고 — probing · causal interpretability 영역 (2026-09-25)

> 범위: V-JEPA / V-JEPA 2 의 물리 변수 probing, probing 비판 (control task, amnesic, causal abstraction), activation patching 과 attention knockout, logit/tuned lens, 결정론적 predictor 의 평균 회귀, 복사 기준선.
> 목적: 우리 방법 중 무엇이 표준이고 무엇이 새로운지, 그리고 리뷰어가 기대하는 rigor 가 무엇인지 정한다.
> 방법: 논문마다 arXiv, ACL Anthology, PMLR, NeurIPS 페이지에서 제목, 연도, 학회를 직접 확인했다. 실제 리뷰어 행태는 OpenReview API 로 원문을 확인했다.
> ⚠️ 세션 전체 WebSearch 한도(200회)가 조사 도중 소진됐다. 2026 preprint 는 누락됐을 수 있다. 확인하지 못한 항목에는 따로 표시했다.

## 0. 결론 먼저

**결론.** probing, 렌즈, knockout, bootstrap 같은 개별 방법은 전부 표준이다. 우리 논문의 새로움은 방법이 아니라 두 곳에 있다.
- **표적**: frozen 사전학습 *predictor* 를 분석한다.
- **대조군**: 복사 기준선, 인과 표적, 관측 대 자기 되먹임 3팔을 둔다.

"encoder is enough" 를 세우려면 probe 결과만으로는 안 된다. 개입 증거와 충분성 증거가 있어야 한다.

**근거 3줄.**
1. encoder 쪽 층별 물리 probing 과 인과 부분공간 분석은 2026 년에 이미 나왔다. Joseph et al. 은 ICML 2026 (V-JEPA 2 L/H/g) 이고, Bouwens 는 ICML 2026 Mech Interp WS 다. 두 편 모두 **predictor 는 보지 않았다.**
2. "predictor 는 관측 이상을 보태지 않는다" 에는 외부 선례가 넷 있다.
   - Physion (NeurIPS 2021 D&B): observed+simulated readout 이 observed 보다 낫지 않다, p=0.53.
   - V-JEPA 2 Table 20: EK100 에서 predictor 기여 +0.6 (로컬 README 전사값, 원문 재대조 필요).
   - GEOPHYS (2026): predictor 없이 IntPhys 2 93.3%.
   - Punzo et al. (2026): IntPhys 2 에서도 encoder probe 는 읽힌다.
3. 실제 리뷰어가 이 질문을 요구했다. IntPhys 2 (NeurIPS 2025 D&B, 4× borderline accept 인데 reject) 리뷰 원문은 "Is it due to memory? Or representation and perception? … probing in the intermediate reasoning process" 다.

**미결 1줄.** 가장 큰 대안 설명이 아직 막히지 않았다. 결정론적 L1 predictor 의 조건부 중앙값 수축 (Mathieu et al. 2016) 이다. "뒤처짐과 시간 감쇠는 실패가 아니라 기대된 최적 행동" 이라는 반박이다.

---

## 1. 가장 가까운 선행 연구 — encoder 쪽은 선점됐다

| 논문 | 무엇을 했나 | 우리와의 관계 |
|---|---|---|
| **Interpreting Physics in Video World Models** (Joseph et al., ICML 2026; arXiv 2602.07050) | V-JEPA 2 L/H/g · VideoMAE-v2 **encoder 만** 분석. 층 깊이 약 1/3 에 "Physics Emergence Zone". 정보가 거기서 정점을 찍고 **출력층으로 가며 떨어진다**. 속도·가속은 초기층부터, 방향은 PEZ 에서 원형 코드. PEZ 국소 attention 절제로 방향 R² 0.97→0.14, IntPhys 78.3→61.7 (WebFetch 요약값) | encoder 쪽 새로움 0. predictor 는 최종 z 만 받으므로 **최종 LN(z) 인터페이스에서** 충분성을 보여야 한다. 가속 결과가 우리 H1 (C8/C16 ≈0) 과 **충돌** 한다 → 조정 필요 |
| **Causal State Variables in V-JEPA 2 Latents** (Bouwens, ICML 2026 MI WS) | ViT-L 방향 probe 가 layer 4 에서 96%, layer 7 에서 100%. 인과 절제 효과가 무작위의 43배 (ViT-H 54배) | encoder 부분공간이 "probe 에 대해" 인과적임을 보였다. **predictor 가 그것을 쓰는지는 안 쟀다** → 우리 interchange 실험의 선례 |
| **How Do Video Foundation Models Encode Intuitive Physics?** (Punzo et al., arXiv 2606.09646) | IntPhys 2 와 MVP 에서 frozen 층별 probing. V-JEPA 가 최강 | surprise 가 chance 인 곳에서도 encoder 에는 정보가 있다 → 우리 서사를 받친다 |
| **GEOPHYS** (Internò et al., arXiv 2606.20707) | predictor 없이 이미지 encoder 임베딩 기하로 IntPhys 2 93.3% | "encoder 충분" 의 강한 증거인 동시에 "predictor 는 왜 필요한가" 반문의 근거 |
| **V-JEPA 2 Video Transcoders** (Ghassemlou & Joseph, IJCAI SPAI WS) | SSv2 운동 템플릿 feature | SSv2 encoder 쪽 분석은 이미 있다 |
| **Intuitive physics … emerges** (Garrido et al., arXiv 2502.11831) | V-JEPA(v1) ViT-H IntPhys dev 98%. 대조군은 무작위 초기화 20 seed. **encoder 와 predictor 를 분리하지 않았고 복사 기준선도 없다** | 우리 복사 기준선과 인과 표적이 이 빈틈을 채운다 |

**검색 범위 안에서 frozen 사전학습 predictor 를 표적으로 한 기제 분석은 찾지 못했다.**
근처에 있는 것은 셋이다.
- UWM-JEPA: 자기 모델의 blind rollout.
- 철회된 "The JEPA Predictor: A Transferable Operator…".
- TAP-JEPA 와 WMReward: predictor 를 사용만 한다.

"최초" 문장은 추가 검색으로 확인하기 전에는 쓰지 않는다.

## 2. 우리 방법별 — 표준인가, 새로운가, 리뷰어가 요구할 것

| 우리 방법 (문서) | 선례 | 판정 | 리뷰어가 요구할 것 / 고칠 것 |
|---|---|---|---|
| encoder ridge 층별 probe (H1, H7) | Alain&Bengio 2016, Joseph 2026 | **표준** | **대조군**: 무작위 초기화 ViT-H (Garrido 식), 픽셀·광류 기준선, 셔플 라벨 → selectivity (Hewitt&Liang 2019). 궤적 수가 적으니 MDL 병기 (Voita&Titov 2020). 궤적 단위 split 은 이미 했다 ✅ |
| attentive probe + 조건 간 이식 (z/h/p, 가시→가림) | probe transfer 일반 | 변형 | 49.3M head 는 포화한다 → control task 에 걸린다. 그래서 "이식 줄만 읽는다" 규칙을 본문에 명시한다. 좌표계 문제 (Nanda 2023) 로 "정보 없음" 해석을 금지하는 근거를 인용한다 |
| 파라미터 없는 템플릿 argmax + 위치 귀무 (H23) | RSA·템플릿 디코딩 관행 | **강점** | 리뷰어가 선호한다. 귀무를 모든 문서에 일괄 적용한다 (교차 검토 A4) |
| 층별 렌즈 p^(l)=proj(norm(q^(l))) (H23, H7) | logit lens (nostalgebraist 2020), Vilas 2023 | **표준** | 초기층 수렴 cos 0.32 는 Belrose et al. 이 말한 logit-lens 편향 그대로다 → **tuned lens** 판을 붙인다 (문맥 토큰으로 층별 affine 을 맞추고 미래 슬롯에 적용, frozen) |
| attention knockout mm/mc (H8, H9) | Geva et al. 2023 (attention knockout), Neo et al. ICLR 2025 | **표준** | zero 식 차단은 분포 밖이다 (L7–11 에서 L1 +64%). Zhang&Nanda 2024, Li&Janson 2024, **Parodi 2026 (ViT zero-ablation 과대평가)** 참조. resample/mean 대체와 path patching (Wang 2023) 이 필요하다 |
| surprise 복사 기준선 (H6) | 특징 예측의 Copy-Last (DINO-Foresight, NeurIPS 2025), Physion observed readout | **적용이 새로움** | JEPA VoE 평가에는 처음 붙인다 (검색 범위 안). 표준과 인과 표적을 병기한다 ✅ |
| 인과 표적으로 양방향 누수 분리 (H6) | 선례를 못 찾음 | **새로움** | 인과 표적이 정지 장면 점수를 오히려 올리는 분포 효과를 단서로 둔다 (H6 정정 12) |
| 관측/자기/표적 되먹임 3팔 rollout (H5) | V-JEPA 2-AC rollout loss, UWM-JEPA blind rollout, 누적 오차 (Lambert 2022) | 변형·새로움 | "rollout 학습을 안 받은 predictor 에 대한 strawman" 비판을 받는다 → ar_z 를 양성 대조로 앞세우고, 주장을 '진화 능력 탐침' 으로 한정한다 |
| RoPE 위치 밀기 (H4) | 불변성 sanity check | 표준 | — |
| **빠진 것: 개입** | Elazar 2021 (amnesic), LEACE 2023, DAS 2024, Othello (Li 2023) | **필수** | 아래 §3 |

## 3. "encoder is enough" 를 세우는 순서 — 문헌이 요구하는 세 단

Tan et al. (2026, arXiv 2607.27017) 의 recoverable → decodable → functionally used 틀을 그대로 빌린다.
(저평판 신규 preprint 라 틀만 참고하고 인용 비중은 낮춘다.)

1. **decodable, 인터페이스에서.** 최종 LN(z) 에서 속도, 방향, 위치, 정체성을 읽는다.
   - 대조군: 무작위 초기화, 픽셀, 셔플.
   - 역사 단서 기준선 (H1 이 이미 함): 창 시작 위치, 입장 시각.
   - 중간층이 아니라 **predictor 가 받는 층**에서 잰다 (Joseph 의 출력층 감쇠 때문).
2. **충분 (Physion / Future Lens 식).**
   - z(문맥 끝) → t+k 물체 위치를 디코딩하는 probe 를 만든다.
   - 같은 자로 읽은 p 슬롯 k 위치와 비교한다.
   - 학습과 test 의 운동 법칙을 다르게 둔다.
   - z-probe > p 이면 "예측에 필요한 정보가 입력에 선형으로 있는데 predictor 가 실현하지 않는다" 가 선다.
   - 단서: probe 는 지도 학습이고 predictor 는 자연 영상 자기지도라 공정한 경쟁이 아니다. '가용성' 주장으로만 쓴다.
3. **사용 여부 (개입).**
   - z 에서 속도·방향 부분공간을 **LEACE 로 지우거나**, 두 clip 사이에서 **DAS/interchange 로 교환**한다.
   - 그런 다음 p 의 물체 이동이 따라 바뀌는지 본다.
   - 대조: 같은 차원의 무작위 부분공간, patch 뒤 z 의 분포 적합성 (Makelov 2024 illusion 대비).
   - completeness 와 selectivity 를 함께 보고한다 (Canby 2025).
   - 결과가 어느 쪽이어도 beat 가 선다. 안 바뀌면 "predictor 가 입력 속도를 안 쓴다". 바뀌되 뒤처지면 "쓰지만 시간으로 적분하지 못한다" 이고, H5 의 '두 걸음 뒤 멈춤' 과 연결된다.
4. **기하 (사용자 제안: cos sim / geometry).**
   - z 거리와 물리 상태 거리의 RSA (Spearman) 로 본다.
   - CKA 단독은 Davari et al. (ICLR 2023) 의 취약성 비판에 걸린다 → 중심화 cos (h̄ 차감, H23 이 이미 씀) 와 병기한다.
5. **SSv2 (사용자 제안).**
   - encoder 방향 probing 만으로는 선행 연구와 겹친다 (Bouwens 8방향, Ghassemlou&Joseph).
   - 이렇게 바꾼다: **z(문맥 끝) 로 학습한 좌/우 probe 를 p 미래 토큰에 그대로 걸어** horizon 별 유지율을 잰다. h 를 천장으로, 복사를 기준선으로 둔다.
   - 그러면 자연 영상에서 '지속·전이' 증거가 된다.
   - `/local_datasets/vlm_direction/ssv2vp` 는 이 노드에서 경로가 확인되지 않았다 (다른 노드일 수 있음).

## 4. 실제 리뷰어 행태 (OpenReview 원문)

- **IntPhys 2, NeurIPS 2025 D&B → Reject** (4× borderline accept, PC 순위로 탈락).
  - "only stay at performance comparisons, failing to deeply analyze the underlying causes of model failures (e.g., whether continuity errors stem from motion prediction flaws)"
  - "Is it due to memory? Or representation and perception? Is it possible to probing in the intermediate reasoning process…"
  - "The performance bottleneck may not lie in the physics understanding capability but in the insufficient context … only 16 frames"
  - → 성능표만 있는 논문은 떨어졌다. 원인 분리가 받아들여지는 기여다. 문맥 길이 반론은 본문에서 미리 막는다 (CLAUDE.md §6 C=4~16 무차이, H1 C8/C16/C32).
- **mech-interp 쪽 공통 요구 (문헌에서 도출).**
  - probe 는 가설 생성용이고 확인은 개입으로 한다 (Belinkov 2022, Ravichander 2021, Elazar 2021, Huang&Chang 2025).
  - 주장마다 Pearl 사다리 등급을 표기한다 (Joshi 2026).
  - 분포 안 개입 (Zhang&Nanda 2024, Li&Janson 2024).
  - 분산과 안정성 보고 (Méloux 2025).
  - localization 에서 설계 원리로 바로 뛰지 않는다 (Hase 2023).

## 5. 가장 큰 대안 설명 — 평균·중앙값 회귀

- **내용.** L2 는 조건부 평균, L1 은 조건부 중앙값을 낸다. 다봉 미래에서는 흐려지고 horizon 이 길수록 심해진다 (Mathieu et al., ICLR 2016).
- **우리 결과와의 관계.** 우리 '등속 외삽보다 뒤처짐' 과 '먼 슬롯 물체 신호의 시간 감쇠' 는, 자연 영상 분포로 학습한 결정론적 L1 predictor 가 미래 위치 분포가 넓어질 때 보일 수축과 모양이 같다.
- **대응 (CLAUDE.md 규칙: 목적함수를 원인으로 걸지 않는다).**
  - 원인 주장은 discussion 한 문단으로만 쓴다.
  - 본문에는 행동 수준 판별 증거를 둔다.
    - "상태 없음" 이면 p 의 이동이 v_T 와 무관해야 한다.
    - "상태는 읽되 수축" 이면 v_T 에 비례하되 이득이 1 보다 작다.
  - H5 의 "멈춘 자리 = 속도 분위별 진실 j2 자리" 는 속도를 읽는다는 쪽의 증거다.
  - 이 구분 자체가 새 predictor 원리로 이어진다: 문제는 읽기가 아니라 잇기다.
- **새로움 경계.** 확률적/다중 가설 JEPA 는 이미 여럿 있다 (VJEPA 2601.14354, Branch-JEPA, feature-space flow matching). 가림 강제 학습에는 Causal-JEPA (ICML 2026) 가 있다. 새 predictor 가 "확률적으로" 나 "가림을 넣어" 로만 가면 새롭지 않다.

## 6. 동기 절의 사실 점검

- **planning 은 사전학습 predictor 를 쓰지 않는다.** V-JEPA 2-AC 는 "freeze the video encoder and learn a new action-conditioned predictor" 다 (약 300M, 24층, block-causal, DROID 62h).
- 사전학습 predictor 를 쓰는 곳은 다음이다.
  - surprise 채점: IntPhys (Garrido), **WMReward** (CVPR 2026, 생성 보상, PhysicsIQ 1위), **Off-Manifold Refinement** (BMVC 2026, V-JEPA 2.1 surprise 기울기).
  - EK100 anticipation: V-JEPA 2 본문, TAP-JEPA.
- **고친 문장 제안.** "사전학습 predictor 는 surprise 채점, 생성 보상, anticipation 에 world model 로 쓰이지만, planning 을 위해서는 같은 encoder 위에 predictor 를 새로 학습한다. 우리는 전자의 전제를 검증한다."

## 7. 추천 추가 실험 (frozen, 비용 순)

| 순 | 실험 | 막는 비판 | 비용 |
|---|---|---|---|
| 1 | probe 대조군: 무작위 초기화 ViT-H, 픽셀·광류, 셔플 → selectivity | "속도 R² 0.99 는 당연" | 추출 1회 + CPU |
| 2 | z→t+k 위치 디코딩 대 p 슬롯 k 위치 (다른 법칙에서 test) | encoder 충분성 (Physion 식) | 캐시만 |
| 3 | LEACE 삭제 / DAS 교환 (z 속도·방향) → p 이동 변화. 무작위 부분공간 대조, completeness/selectivity | "있음 ≠ 쓰임" | predictor 재실행 |
| 4 | tuned lens (층별 affine, 문맥 토큰으로 맞춤) | logit-lens 편향 | 캐시 + CPU |
| 5 | knockout 을 resample/mean 대체로 다시 하고 표적 공간 L1 을 병기 | 분포 밖 개입 | predictor 재실행 |
| 6 | Joseph 합성 공 프로토콜을 ViT-H 로 재현 (가속) | 외부 결과와의 모순 | 소량 추출 |
| 7 | 복사 기준선과 자기 되먹임을 V-JEPA 2.1 에서 재현 | "2.1 이면 해결?" | 추출 |
| 8 | 전 문서 궤적 단위 CI, 주 셀 사전 지정, 다중 비교 보정 | 분산·선택 편향 | CPU |

## 8. 주장 문장 가이드 (금지 표현과 정합)

- ❌ "predictor 가 정보를 잃는다 / 못 만든다" — p probe 가 98% 로 읽는다 (CLAUDE.md 금지 표현).
- ✅ "관측이 끊긴 뒤 p 는 물체 신호를 진실 궤적 위에 이어 놓지 못한다. 입력 z 에는 그 궤적을 선형으로 예측할 정보가 있다 (§3-2). 속도 부분공간을 교환하면 p 가 [따라간다/안 따라간다] (§3-3)."
- ✅ 병목은 **component 수준**으로 쓴다 (encoder 충분, predictor 출력 결핍). 층·간선 수준은 "아직 특정 못함" 을 유지한다 (README 와 일치).

## 출처

- Garrido et al. 2025 https://arxiv.org/abs/2502.11831
- Joseph et al. 2026 https://arxiv.org/abs/2602.07050 (OpenReview aijGVmEG9Y, 'ICML 2026 regular')
- Punzo et al. 2026 https://arxiv.org/abs/2606.09646
- Bouwens 2026 https://openreview.net/forum?id=CC5xKstwmf
- Ghassemlou & Joseph 2026 https://openreview.net/forum?id=Q3osfMTUbE
- GEOPHYS https://arxiv.org/abs/2606.20707
- IntPhys 2 https://arxiv.org/abs/2506.09849 · 리뷰 https://openreview.net/forum?id=Xpf5x3mLvn
- V-JEPA 2 https://arxiv.org/abs/2506.09985
- V-JEPA 2.1 https://arxiv.org/abs/2603.14482
- WMReward https://arxiv.org/abs/2601.10553
- Off-Manifold Refinement https://arxiv.org/abs/2608.29904
- TAP-JEPA https://arxiv.org/abs/2606.00662
- Physion https://arxiv.org/abs/2106.08261
- Nayebi et al. 2023 https://arxiv.org/abs/2305.11772
- Alain & Bengio https://arxiv.org/abs/1610.01644
- Hewitt & Liang https://aclanthology.org/D19-1275/
- Voita & Titov https://aclanthology.org/2020.emnlp-main.14/
- Ravichander et al. https://aclanthology.org/2021.eacl-main.295/
- Elazar et al. https://aclanthology.org/2021.tacl-1.10/
- Belinkov https://aclanthology.org/2022.cl-1.7/
- LEACE https://arxiv.org/abs/2306.03819
- Canby et al. https://aclanthology.org/2025.ijcnlp-long.47/
- Huang & Chang https://arxiv.org/abs/2510.09794
- Joshi et al. https://arxiv.org/abs/2602.16698
- ROME https://arxiv.org/abs/2202.05262
- IOI https://arxiv.org/abs/2211.00593
- Geva et al. https://aclanthology.org/2023.emnlp-main.751/
- Neo et al. https://arxiv.org/abs/2410.07149
- Zhang & Nanda https://arxiv.org/abs/2309.16042
- Heimersheim & Nanda https://arxiv.org/abs/2404.15255
- Hase et al. https://arxiv.org/abs/2301.04213
- Makelov et al. https://arxiv.org/abs/2311.17030
- DAS https://proceedings.mlr.press/v236/geiger24a.html
- Causal Abstraction https://www.jmlr.org/papers/v26/23-0058.html
- Optimal ablation https://proceedings.neurips.cc/paper_files/paper/2024/hash/c55e6792923cc16fd6ed5c3f672420a5-Abstract-Conference.html
- Parodi et al. https://arxiv.org/abs/2604.14433
- Méloux et al. https://arxiv.org/abs/2510.00845
- logit lens https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens
- tuned lens https://arxiv.org/abs/2303.08112
- Future Lens https://arxiv.org/abs/2311.04897
- Vilas et al. https://arxiv.org/abs/2310.18969
- Othello https://arxiv.org/abs/2210.13382
- Nanda et al. https://aclanthology.org/2023.blackboxnlp-1.2/
- Vafa 2024 https://arxiv.org/abs/2406.03689
- Vafa 2025 https://arxiv.org/abs/2507.06952
- Kang et al. https://arxiv.org/abs/2411.02385
- Li, Cao, Cheung https://openreview.net/forum?id=lzfzjYuWgY
- Tan et al. https://arxiv.org/abs/2607.27017
- UWM-JEPA https://arxiv.org/abs/2605.25313
- Mathieu et al. https://arxiv.org/abs/1511.05440
- VJEPA https://arxiv.org/abs/2601.14354
- DINO-Foresight https://arxiv.org/abs/2412.11673
- Causal-JEPA https://arxiv.org/abs/2602.11389
- Davari et al. https://arxiv.org/abs/2210.16156
- Lambert et al. https://arxiv.org/abs/2203.09637

## 미확인 표시

- Joseph et al. 의 수치는 WebFetch 요약 기준이고 원문 표와 대조하지 않았다.
- V-JEPA 2 Table 20 은 로컬 `z_research/anticipation/EK100/README.md` 전사값이다.
- Geva et al. 과 Neo et al. 의 'attention knockout' 방법명은 원문 기억이고, 이번 세션에서 abstract 로 확인하지 못했다.
- 영상 예측 문헌의 'copy-last-frame 이 Human3.6M 에서 모든 모델을 이긴다' 는 출처 논문을 확정하지 못해 인용하지 않았다.


### 논문 목록

| 제목 | 학회·연도 | URL | 요점 | 우리와의 관계 |
|---|---|---|---|---|
| Intuitive physics understanding emerges from self-supervised pretraining on natural videos (Garrido et al.) | arXiv 2025 (2502.11831) | https://arxiv.org/abs/2502.11831 | surprise = ‖p(f(V_ctx)) − g(V)‖1 로 encoder+predictor 를 함께 쓴다. V-JEPA(v1) ViT-H IntPhys dev 98% [95,99]. 대조군은 무작위 초기화 망 20개(chance 근처), VideoMAEv2(픽셀 예측), MLLM. encoder 와 predictor 를 분리하지 않았고 복사(예측 없음) 기준선도 없다. 3~4초 기억 한계와 충돌·상호작용 약점은 스스로 인정한다. | 우리 출발점이다. 우리의 복사 기준선(85.00 vs 88.89, n.s.)과 양방향 표적 누수는 이 논문에 없는 대조군이다. 여기서 쓴 무작위 초기화 대조군을 우리 probing 에도 붙여야 리뷰어 기준을 맞춘다. |
| Interpreting Physics in Video World Models (Joseph, Garrido, Balestriero, Kowal, Fel, Bakhtiari, Richards, Rabbat) | ICML 2026 (OpenReview venue 표기 'ICML 2026 regular'; arXiv 2602.07050) | https://arxiv.org/abs/2602.07050 | frozen encoder 만 분석한다 (V-JEPA 2 L/H/g, VideoMAE-v2 g). 층 깊이 약 1/3 지점에서 물리 변수가 갑자기 읽히기 시작한다('Physics Emergence Zone'). 거기서 정점을 찍고 출력 쪽으로 가며 떨어진다. 속도·가속 크기는 초기층부터 읽히고, 방향은 PEZ 에서 원형 population code 로 나타난다. PEZ 의 국소 attention 을 절제하면 방향 R² 0.97→0.14, IntPhys 78.3→61.7 인데 ImageNet 은 거의 그대로다. 수치는 WebFetch 요약 기준이고 원문 표와 대조하지 않았다. | encoder 쪽 layerwise probing 은 이미 선점됐다 (Meta/Mila, Garrido 공저). 우리가 새로울 수 있는 자리는 predictor 뿐이다. 이 논문은 '정보가 출력층으로 가며 줄어든다' 고 보고하므로, 'encoder is enough' 는 predictor 가 실제로 받는 최종 z 에서 증명해야 한다. 또 가속이 초기층부터 읽힌다는 보고가 우리 H1 (C8/C16 에서 국소 가속 ≈ 0) 과 충돌하므로 조정이 필요하다. |
| How Do Video Foundation Models Encode Intuitive Physics? Probing Across Pretraining Paradigms (Punzo et al.) | arXiv 2026 (2606.09646) | https://arxiv.org/abs/2606.09646 | V-JEPA, VideoMAE, LTX-Video 의 frozen 특징을 IntPhys 2 와 MVP 에서 층별로 probing 했다. V-JEPA 가 가장 강하고, 물리 정보는 중간~후반 깊이에서 가장 잘 읽힌다. 프레임 순서를 섞는 대조군을 썼다. | surprise 가 chance 인 IntPhys 2 에서도 encoder 특징으로는 probe 가 읽힌다. '정보는 encoder 에 있고 채점·예측 경로가 병목' 이라는 서사를 외부에서 받쳐 준다. |
| Causal State Variables in V-JEPA 2 Latents: Discovery, Intervention, and Portability (Bouwens) | ICML 2026 Mech Interp Workshop (virtual poster; 단독 저자) | https://openreview.net/forum?id=CC5xKstwmf | V-JEPA 2 ViT-L encoder 에서 운동 방향 probe 가 layer 4 에서 96%, layer 7 에서 100% 다. 차원의 57% 를 쓰는 분산 부분공간이다. layer 7 인과 절제의 효과는 무작위 방향의 43배, ViT-H 에서는 54배다. 관찰에 그치지 않고 개입과 대조군 4종으로 인과성을 보였다. | encoder 부분공간이 '인과적으로 기능한다' 는 증거지만 그 기능은 probe readout 에 대한 것이다. predictor 가 그 부분공간을 쓰는지는 안 쟀다. 우리가 할 interchange / erasure 실험의 직접 선례다. |
| V-JEPA 2 Video Transcoders Find Motion-Based Features (Ghassemlou, Joseph) | IJCAI SPAI Workshop 2026 | https://openreview.net/forum?id=Q3osfMTUbE | V-JEPA 2 특징에 sparse transcoder 를 붙였다. SSv2 운동 템플릿에서 물체와 무관한 운동 primitive feature 와 낙하·충돌에 선택적인 feature 를 찾았다. | SSv2 운동이 encoder 에 있다는 것은 이미 나와 있다. 사용자의 SSv2 계획은 encoder 쪽만으로는 새롭지 않다. predictor 미래 토큰으로 방향이 이어지는지를 재야 새로워진다. |
| GEOPHYS: The Geometry of Physical Plausibility (Internò et al.) | arXiv 2026 (2606.20707) | https://arxiv.org/abs/2606.20707 | predictor 없이 frozen 이미지 encoder 의 프레임별 임베딩 기하 성질 5개만으로 LikePhys 98.3%, IntPhys 2 93.3% 를 낸다. 같은 조건에서 V-JEPA 2 등은 chance 근처다. | 'encoder 만으로 충분하다' 를 가장 강하게 받치는 외부 증거다. 동시에 리뷰어가 '그럼 predictor 가 왜 필요하냐' 를 물을 근거도 된다. predictor 의 고유 역할(관측이 없을 때의 진화)을 분명히 해야 한다. |
| IntPhys 2: Benchmarking Intuitive Physics Understanding In Complex Synthetic Environments (Bordes et al.) | arXiv 2025 (2506.09849); NeurIPS 2025 D&B 제출, 최종 Reject (4× borderline accept) | https://openreview.net/forum?id=Xpf5x3mLvn | V-JEPA 등 예측 모델이 chance 근처다 (≈50~58%, 사람 ≈96%). 실제 리뷰 원문: 'failing to deeply analyze the underlying causes of model failures (e.g., whether continuity errors stem from motion prediction flaws)', 'Is it due to memory? Or representation and perception? ... probing in the intermediate reasoning process', 'bottleneck may ... lie ... in the insufficient context (16 frames)'. | 리뷰어들이 원인 분리(표현 vs 기억 vs 예측)를 명시적으로 요구했다. 우리 논문이 바로 그 질문에 답하는 형태다. 문맥 길이 반론(16 프레임)에 대한 방어도 미리 넣어야 한다 (CLAUDE.md §6 의 C=4~16 무차이, H1 의 C8/C16/C32). |
| V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning (Assran et al.) | arXiv 2025 (2506.09985) | https://arxiv.org/abs/2506.09985 | EK100 anticipation 은 사전학습 predictor 출력을 encoder 출력과 이어 붙여 probe 에 넣는다. 반면 로봇 planning 인 V-JEPA 2-AC 는 'freeze the video encoder and learn a new action-conditioned predictor' 다 (약 300M, 24층, block-causal, DROID 62시간). Table 20 (ViT-g384, 로컬 EK100 README 에 옮겨 적은 값, 원문 재대조 필요): action recall 은 encoder 만 39.1, 둘 다 39.7, predictor 만 20.2. | 동기 문장을 고쳐야 한다. planning 은 사전학습 predictor 를 쓰지 않는다. EK100 에서 predictor 의 기여는 +0.6 이다. 이 수치는 'encoder is enough' 를 받치지만 'downstream 이 predictor 에 기댄다' 는 약하게 만든다. |
| V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning (Mur-Labadia et al.) | arXiv 2026 (2603.14482) | https://arxiv.org/abs/2603.14482 | dense predictive loss (보이는 토큰도 손실에 포함)와 여러 중간층 deep self-supervision 을 도입했다. predictor 목적함수와 표적 구성이 V-JEPA 2 와 다르다. | 리뷰어가 'V-JEPA 2.1 predictor 에서도 그런가' 를 물을 것이다. 적어도 복사 기준선과 자기 되먹임 두 실험은 2.1 에서 재현해 둘 가치가 있다. CLAUDE.md 의 ViT-H 원칙과는 별개 축이다. |
| Inference-time Physics Alignment of Video Generative Models with Latent World Models (WMReward; Yuan et al.) | CVPR 2026 (arXiv 2601.10553) | https://arxiv.org/abs/2601.10553 | V-JEPA 2 surprise 점수를 물리 타당성 보상으로 써서 영상 생성의 denoising 궤적을 탐색·조향한다. ICCV 2025 PhysicsIQ 챌린지 1위 (62.64%). | 사전학습 predictor 의 surprise 가 실제 downstream 신호로 쓰이고 있다는 강한 동기 근거다. 우리 복사 기준선 결과는 그 신호의 상당 부분이 예측 없이도 나온다는 뜻이라 이 사용 방식에 직접 닿는다. |
| Off-Manifold Refinement: Guiding Video Generators with a Frozen World Model (Nguyen-Truong et al.) | BMVC 2026 (arXiv 2608.29904) | https://arxiv.org/abs/2608.29904 | frozen V-JEPA 2.1 encoder+predictor 의 surprise energy 기울기로 생성 ODE 를 유도한다 (+5.0pt). | frozen predictor 를 world model 로 가정한 downstream 사례가 하나 더 있다. 동기 절에서 쓸 수 있다. |
| TAP-JEPA: Frozen Future-Latent Probing and Two-Stage Score Fusion for EPIC-KITCHENS-100 Action Anticipation | CVPR 2026 EgoVis Workshop (arXiv 2606.00662) | https://arxiv.org/abs/2606.00662 | frozen V-JEPA 2.1 predictor 의 미래 토큰을 encoder 토큰과 섞어 EK100 2위를 했다. predictor 기여를 떼어 본 ablation 은 없다. | '사전학습 predictor 를 미래 예측기로 쓴다' 는 관행이 있다는 증거다. 그 기여를 검증하지 않는 관행이라는 점도 함께 보여 준다. |
| Physion: Evaluating Physical Prediction from Vision in Humans and Machines (Bear et al.) | NeurIPS 2021 Datasets & Benchmarks (arXiv 2106.08261) | https://arxiv.org/abs/2106.08261 | readout 프로토콜을 observed / observed+simulated / full 세 가지로 나눴다. 시각 dynamics 모델의 observed+simulated readout 은 observed 보다 낫지 않았다 (p=0.53). 즉 모델이 시뮬레이션한 미래 latent 가 관측 latent 이상의 정보를 주지 못했다. | 'encoder(관측)로 충분하고 predictor(시뮬레이션)는 보태지 않는다' 의 가장 가까운 개념적 선례다. 우리 설계를 이 프로토콜 언어로 정식화하면 리뷰어가 곧바로 알아본다. |
| Neural Foundations of Mental Simulation: Future Prediction of Latent Representations on Dynamic Scenes (Nayebi et al.) | NeurIPS 2023 (arXiv 2305.11772) | https://arxiv.org/abs/2305.11772 | frozen foundation encoder 위에 미래 latent 예측 모듈을 올린 모델이 신경 반응과 행동 오류 패턴을 가장 잘 맞춘다. | 'frozen encoder + 별도 dynamics' 분업 구도의 선례다. 새 predictor 설계 절의 근거로 쓸 수 있다. |
| Understanding intermediate layers using linear classifier probes (Alain, Bengio) | arXiv 2016 (1610.01644) | https://arxiv.org/abs/1610.01644 | 층마다 독립 linear probe 를 붙여 정보 접근성을 잰다. | 우리 layerwise ridge probing 의 표준 출처다. |
| Designing and Interpreting Probes with Control Tasks (Hewitt, Liang) | EMNLP 2019 | https://aclanthology.org/D19-1275/ | 무작위 라벨 control task 와 selectivity (과제 정확도 − 대조 정확도) 를 제안했다. 표현력이 큰 probe 는 표현과 무관하게 라벨을 외운다. | attentive probe (49.3M, train_acc 1.000 포화) 결과에 직접 적용된다. 연속 변수(속도)에는 셔플 라벨·무작위 초기화 encoder·픽셀 기준선으로 selectivity 를 보고해야 한다. |
| Information-Theoretic Probing with Minimum Description Length (Voita, Titov) | EMNLP 2020 | https://aclanthology.org/2020.emnlp-main.14/ | 정확도 대신 MDL (online coding) 로 '정보를 얼마나 쉽게 꺼낼 수 있나' 를 잰다. 더 안정적이다. | 궤적이 84개뿐이라 probe 가 학습 궤적 수에 묶인다 (H1: 7궤적이면 R²≈0). 이런 적은 표본 영역에서 R² 대신 MDL 곡선을 병기하면 '학습 궤적 수에 묶인 읽기' 를 정량화할 수 있다. |
| Probing the Probing Paradigm: Does Probing Accuracy Entail Task Relevance? (Ravichander, Belinkov, Hovy) | EACL 2021 | https://aclanthology.org/2021.eacl-main.295/ | probe 로 읽힌다고 모델이 그 정보를 과제에 쓰는 것은 아니다. | 'encoder 에 속도가 있다' 에서 'predictor 가 쓸 수 있다/쓴다' 로 넘어가는 순간 이 비판을 받는다. 개입 실험으로 막아야 한다. |
| Amnesic Probing: Behavioral Explanation with Amnesic Counterfactuals (Elazar et al.) | TACL 2021 | https://aclanthology.org/2021.tacl-1.10/ | 속성을 표현에서 지우는 개입(INLP)을 한 뒤 행동 변화로 '사용 여부' 를 잰다. probe 성능과 과제 중요도는 상관이 없었다. | z 에서 속도 부분공간을 지운 뒤 predictor 출력의 물체 이동이 바뀌는지 보는 실험 설계 그대로다. frozen 에서 가능하다. |
| Probing Classifiers: Promises, Shortcomings, and Advances (Belinkov) | Computational Linguistics 48(1), 2022 | https://aclanthology.org/2022.cl-1.7/ | probing 의 한계를 정리한 리뷰다 (대조군, probe 표현력, 상관 대 인과). | related work 에서 '우리는 probe 를 가설 생성에만 쓰고 개입으로 확인한다' 고 쓸 때 인용하는 표준 출처다. |
| LEACE: Perfect linear concept erasure in closed form (Belrose et al.) | NeurIPS 2023 | https://arxiv.org/abs/2306.03819 | 닫힌 형태의 최소 변경 선형 개념 삭제다. 모든 선형 분류기가 그 개념을 못 읽게 만든다. | z 의 속도·방향 정보를 지우는 amnesic 개입 도구로 가장 싸고 표준이다. 학습 없이 닫힌 해로 구한다. |
| How Reliable are Causal Probing Interventions? (Canby et al.) | IJCNLP-AACL 2025 (arXiv 2408.15510) | https://aclanthology.org/2025.ijcnlp-long.47/ | 인과 probing 개입에는 completeness (표적을 얼마나 지웠나) 와 selectivity (다른 속성을 얼마나 덜 건드렸나) 의 상충이 있다. 비선형 개입이 대개 더 신뢰할 만하다. | erasure 나 interchange 결과를 낼 때 두 지표를 함께 보고해야 리뷰어가 받아들인다. |
| Causality ≠ Decodability, and Vice Versa: Lessons from Interpreting Counting ViTs (Huang, Chang) | arXiv 2025 (2510.09794) | https://arxiv.org/abs/2510.09794 | ViT 에서 중간층 물체 토큰은 디코딩은 약한데 인과 효과는 크다. 최종층 물체 토큰은 디코딩은 잘 되는데 기능적으로 비활성이다. 디코딩 가능성과 인과성은 서로 다른 축이다. | 비전 트랜스포머에서 '있음 ≠ 쓰임' 을 보인 직접 선례다. 우리 'p 는 probe 로 98% 읽히는데 채점은 실패한다' 와 같은 형태다. |
| Causality is Key for Interpretability Claims to Generalise (Joshi et al.) | arXiv 2026 (2602.16698) | https://arxiv.org/abs/2602.16698 | Pearl 의 사다리(관찰·개입·반사실)로 해석 주장을 증거 수준에 맞추라고 요구한다. | 우리 주장마다 '관찰(probe) / 개입(knockout·erasure) / 반사실(렌즈)' 등급을 표기하면 이 기준을 충족한다. README 의 확정/시사 등급과 결합할 수 있다. |
| Locating and Editing Factual Associations in GPT (ROME; Meng et al.) | NeurIPS 2022 | https://arxiv.org/abs/2202.05262 | causal tracing (오염 뒤 복원)으로 사실 회상의 결정적 층·위치를 찾는다. | 우리는 병목 층을 아직 특정하지 못했다. 다음 단계의 표준 방법은 clean/corrupt 쌍(예: 속도가 다른 두 clip)의 predictor 층별 활성 복원 실험이다. |
| Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small (Wang et al.) | ICLR 2023 | https://arxiv.org/abs/2211.00593 | path patching 으로 특정 경로의 인과 효과만 분리한다. | H8 이 밝힌 대로 mm knockout 은 '문맥→미래' 2단 경로를 열어 둔다. 경로 단위로 분리하려면 path patching 이 표준 해법이다. |
| Dissecting Recall of Factual Associations in Auto-Regressive Language Models (Geva et al.) | EMNLP 2023 | https://aclanthology.org/2023.emnlp-main.751/ | 위치 사이의 attention 간선을 막는 'attention knockout' 으로 정보 흐름을 추적한다 (방법명은 원문 기억, 이번 세션에서 abstract 로 확인하지는 못함). | H8 attention knockout 의 직접 선례다. 우리 방법은 표준이고, 새로움은 대상(video predictor)에 있다. |
| Towards Interpreting Visual Information Processing in Vision-Language Models (Neo et al.) | ICLR 2025 | https://arxiv.org/abs/2410.07149 | LLaVA 의 시각 토큰 흐름을 ablation 으로 분석했다. 물체 토큰을 지우면 식별 정확도가 70% 넘게 떨어진다. knockout 과 logit lens 계열 도구를 시각 토큰에 적용했다 (세부 방법은 원문 미확인). | 시각 토큰에 LLM 식 해석 도구를 옮긴 정착된 선례다. 리뷰어는 이런 식의 대조와 보고를 기대한다. |
| Towards Best Practices of Activation Patching in Language Models: Metrics and Methods (Zhang, Nanda) | ICLR 2024 | https://arxiv.org/abs/2309.16042 | 오염 방식(Gaussian noise 대 대칭 치환)과 지표 선택에 따라 localization 결과가 달라진다. 분포 밖 오염은 결론을 왜곡한다. | H8 에서 L7–11 미래→문맥 차단은 출력이 표적 공간 밖으로 나갔다 (L1 +64%). 이것이 바로 분포 밖 개입 문제다. 이 결과는 증거로 쓰기 어렵고, 분포 안 개입으로 다시 해야 한다. |
| How to use and interpret activation patching (Heimersheim, Nanda) | arXiv 2024 (2404.15255) | https://arxiv.org/abs/2404.15255 | patching 의 변형, 해석, 지표 함정을 정리한 실무 지침이다. | knockout·patching 절의 방법 정당화에 쓰는 참고문헌이다. |
| Does Localization Inform Editing? (Hase, Bansal, Kim, Ghandeharioun) | NeurIPS 2023 (spotlight) | https://arxiv.org/abs/2301.04213 | causal tracing 이 찾은 위치와 편집이 가장 잘 되는 위치가 다르다. | '병목 층을 찾으면 그 층을 고치면 된다' 는 설계 원리 도출은 이 비판을 받는다. 새 predictor 원리는 층 위치보다 기능 수준 증거로 세우는 편이 안전하다. |
| Is This the Subspace You Are Looking for? An Interpretability Illusion for Subspace Activation Patching (Makelov, Lange, Geiger, Nanda) | ICLR 2024 | https://arxiv.org/abs/2311.17030 | 부분공간 patching 은 출력과 인과적으로 끊긴 성분을 통해 잠든 병렬 경로를 켜서 원하는 효과를 가짜로 만들 수 있다. | z 속도 부분공간 interchange 실험을 할 때 같은 차원의 무작위 부분공간 대조, 그리고 patch 뒤 z 의 분포 적합성(norm·LN 통계) 확인이 필수다. |
| Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations (DAS; Geiger et al.) | CLeaR 2024 (PMLR 236) | https://proceedings.mlr.press/v236/geiger24a.html | 기울기로 분산 부분공간 정렬을 찾고 interchange intervention 으로 고수준 인과 변수와의 대응을 검증한다. | 'z 의 속도 변수 → predictor 의 미래 위치' 라는 고수준 모형을 검증하는 표준 틀이다. frozen 모델 위에서 회전 행렬만 학습하면 된다. |
| Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability (Geiger et al.) | JMLR 26, 2025 (arXiv 2301.04709) | https://www.jmlr.org/papers/v26/23-0058.html | patching, 인과 매개, causal scrubbing, SAE 를 인과 추상화 하나로 통합한다. | 'encoder=관측 상태, predictor=전이' 라는 우리 분업 가설 자체를 인과 추상화 문장으로 정식화할 수 있다. |
| Optimal ablation for interpretability (Li, Janson) | NeurIPS 2024 (spotlight) | https://proceedings.neurips.cc/paper_files/paper/2024/hash/c55e6792923cc16fd6ed5c3f672420a5-Abstract-Conference.html | zero, mean, resample ablation 은 구성요소 중요도를 과대평가한다. median 구성요소에서 optimal ablation 손실 차는 zero 의 11.1%, mean 의 33.0% 다. | knockout 의 '되돌아오지 않는다' (음성 결과)와 '출력이 무너진다' (양성 결과) 둘 다 ablation 방식에 민감하다. 최소한 mean/resample 판을 병기해야 한다. |
| Zero-Ablation Overstates Register Content Dependence in DINO Vision Transformers (Parodi, Matelsky, Segado) | CVPR 2026 HOW Vision Interpretability Workshop (arXiv 2604.14433) | https://arxiv.org/abs/2604.14433 | ViT 에서 zero-ablation 은 급락을 만들지만 mean, noise, 다른 이미지 셔플로 대체하면 성능이 약 1pp 안에 머문다. | 비전 트랜스포머에서 zero 방식 개입의 과대평가를 직접 보인 최신 선례다. 우리 knockout 은 attention 을 0 으로 막는 방식이라 같은 비판을 받는다. |
| Mechanistic Interpretability as Statistical Estimation: A Variance Analysis (Méloux, Portet, Peyrard) | arXiv 2025 (2510.00845) | https://arxiv.org/abs/2510.00845 | 단일 입력 인과 매개 점수는 분산이 크다. 회로 발견은 데이터와 하이퍼파라미터에 민감하다. bootstrap 과 안정성 지표 보고를 요구한다. | README 의 '궤적 단위 bootstrap' 규칙이 이 요구와 맞는다. 다만 H23·H5·H8 은 아직 clip 단위 CI 라 수정해야 한다. |
| interpreting GPT: the logit lens (nostalgebraist) | LessWrong 2020 (블로그) | https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens | 중간층 hidden 에 최종 unembedding 을 바로 걸어 '그 층에서 멈췄다면의 예측' 을 본다. | 우리 '렌즈' p^(l) = proj(norm(q^(l))) 는 정확히 logit lens 의 latent 판이다. 표준 방법이다. |
| Eliciting Latent Predictions from Transformers with the Tuned Lens (Belrose et al.) | arXiv 2023 (2303.08112) | https://arxiv.org/abs/2303.08112 | logit lens 는 초기층에서 편향되고 부서지기 쉽다. 층마다 affine translator 를 학습하면 더 신뢰할 수 있다. | 우리 렌즈 수렴 cos 은 L0 0.32, L6 0.70 으로 초기층이 약하다. 리뷰어는 tuned lens 판을 요구할 것이다. 문맥 토큰으로 affine 을 맞추고 미래 슬롯에 적용하는 방식으로 frozen 에서 할 수 있다. |
| Future Lens: Anticipating Subsequent Tokens from a Single Hidden State (Pal et al.) | CoNLL 2023 | https://arxiv.org/abs/2311.04897 | 한 위치의 hidden 에서 몇 단계 뒤 토큰을 선형 근사나 개입으로 얼마나 예측할 수 있는지 잰다. | 'z(문맥 끝)에서 t+k 미래 위치를 직접 디코딩' 대 'p 슬롯 k 의 위치' 비교의 선례다. encoder 충분성을 보이는 가장 싼 실험이다. |
| Analyzing Vision Transformers for Image Classification in Class Embedding Space (Vilas et al.) | NeurIPS 2023 | https://arxiv.org/abs/2310.18969 | ViT 중간 표현을 class embedding 공간에 사영해 층별 범주 형성을 추적한다 (ViT 판 logit lens). | 비전 쪽 렌즈 방법의 선례다. |
| Emergent World Representations: Exploring a Sequence Model Trained on a Synthetic Task (Li et al.) | ICLR 2023 (oral) | https://arxiv.org/abs/2210.13382 | Othello-GPT 에서 보드 상태를 probe 로 찾았다. 그 표현에 개입하면 다음 수 예측이 바뀌는 것으로 인과성을 확인했다. | world model 주장은 'probe 로 읽힌다 + 개입하면 출력이 따라 바뀐다' 두 단계로 세운다는 규범의 원형이다. 우리 predictor 쪽 주장에도 두 번째 단계가 필요하다. |
| Emergent Linear Representations in World Models of Self-Supervised Sequence Models (Nanda, Lee, Wattenberg) | BlackboxNLP 2023 | https://aclanthology.org/2023.blackboxnlp-1.2/ | 적절한 기저('내 것/남의 것')를 고르면 보드 상태가 선형으로 읽히고 선형 개입이 된다. | probe 가 안 읽힌다고 없는 게 아니라 좌표계 선택 문제일 수 있다. 우리 이식 실패(p 17.3)를 '정보 없음' 으로 읽지 않는 근거로 쓸 수 있다. |
| Evaluating the World Model Implicit in a Generative Model (Vafa et al.) | NeurIPS 2024 | https://arxiv.org/abs/2406.03689 | 다음 토큰 정확도나 probe 는 좋은데 Myhill-Nerode 식 상태 일관성 지표로 보면 world model 이 훨씬 비일관적이다. | '벤치마크 점수는 좋아도 상태 진화는 없다' 는 우리 서사의 LLM 쪽 선례다. 동기 절에서 인용할 수 있다. |
| What Has a Foundation Model Found? Using Inductive Bias to Probe for World Models (Vafa et al.) | ICML 2025 (PMLR 267) | https://arxiv.org/abs/2507.06952 | 궤도 궤적으로 학습한 foundation model 은 예측은 잘하지만 Newton 역학을 새 과제에 적용하지 못한다. 과제 특화 휴리스틱으로 행동한다. | '예측 성능 ≠ 상태/법칙 표현' 의 대표 선례다. 우리 '관측에 닻을 내린 검색(retrieval)' 가설과 같은 계열이다. |
| How Far Is Video Generation from World Model: A Physical Law Perspective (Kang et al.) | ICML 2025 | https://arxiv.org/abs/2411.02385 | 영상 생성 모델은 가장 가까운 학습 예시를 흉내 내는 'case-based' 일반화를 한다. 참조 우선순위는 color > size > velocity > shape 다. | 'retrieval' 서술의 행동 수준 선례다. 우리 v11 외형 편향(물체 순서)과도 닿는다. 다만 이들은 생성 모델, 픽셀, 학습 데이터 검색이고 우리는 문맥 관측 복사라 층위가 다르다. |
| Do LLMs Build World Representations? Probing Through the Lens of State Abstraction (Li, Cao, Cheung) | NeurIPS 2024 | https://openreview.net/forum?id=lzfzjYuWgY | RL 상태 추상화 이론으로 '세계 상태' 추상화와 '목표 지향' 추상화를 나눠 probing 한다. 모델은 목표 지향 추상화를 우선한다. | '읽기·지속·전이·영속' 능력 분해를 상태 추상화 이론 언어로 정당화할 수 있는 틀이다. |
| What Can Latent World Models Know? Physical Parameter Identifiability in Multimodal Predictive Representations (Tan et al.) | arXiv 2026 (2607.27017) | https://arxiv.org/abs/2607.27017 | 세 단계 검증 틀: raw 관측에서 recoverable → latent 에서 decodable → 예측·제어에서 functionally used. 예: drag 는 recoverable 0.89 인데 예측 목적에서는 0.13 에 머문다. | 'encoder is enough' 를 세우는 절차의 거의 그대로인 틀이다 (단 저평판 신규 preprint). 우리 H1 의 '역사 단서 기준선' 은 이 틀의 recoverable 단계에 해당한다. |
| UWM-JEPA: Predictive World Models That Imagine in Belief Space (Radha, Goktas) | arXiv 2026 (2605.25313) | https://arxiv.org/abs/2605.25313 | 'blind rollout' (중간 관측 없이 여러 걸음)에서 target 표현의 probe R² 유지율로 predictor 를 비교한다. 벡터 latent 기준선은 41~68pt 잃는다. | '관측이 없을 때 상태를 유지·진화하는가' 를 probe R² 로 재는 지표의 선례다. 새 predictor 평가 지표 후보다. |
| Deep multi-scale video prediction beyond mean square error (Mathieu, Couprie, LeCun) | ICLR 2016 (arXiv 1511.05440) | https://arxiv.org/abs/1511.05440 | L2 는 조건부 평균, L1 은 조건부 중앙값을 낸다. 다봉 미래에서는 흐린 예측이 나오고, 먼 미래일수록 심해진다. | '먼 슬롯 물체 신호 감쇠', '등속 외삽보다 뒤처짐' 에 대한 리뷰어의 1순위 대안 설명이다. 결정론적 L1 predictor 가 자연 영상 불확실성 아래 조건부 중앙값으로 수축한다는 설명이다. |
| VJEPA: Variational Joint Embedding Predictive Architectures as Probabilistic World Models (Huang) | arXiv 2026 (2601.14354; 단독 저자) | https://arxiv.org/abs/2601.14354 | 결정론적 JEPA 회귀는 확률 의미를 가린다는 문제의식에서 미래 latent 분포를 변분 목적으로 학습한다. | 새 predictor 가 '확률적 JEPA' 로만 가면 이미 여러 편(VJEPA, Branch-JEPA, feature-space flow matching)이 있다. 차별점이 필요하다. |
| DINO-Foresight: Looking into the Future with DINO (Karypidis et al.) | NeurIPS 2025 (arXiv 2412.11673) | https://arxiv.org/abs/2412.11673 | frozen VFM 특징 공간에서 미래 특징을 예측하고 Copy-Last 기준선과 비교한다. | 특징 공간 예측에서 '마지막 관측 복사' 기준선은 표준이다. 그런데 JEPA 의 VoE surprise 평가에는 그 기준선이 없었다. 우리가 채우는 빈틈을 이렇게 서술할 수 있다. |
| Causal-JEPA: Learning World Models through Object-Level Latent Masking (Nam, Le Lidec, Maes, LeCun, Balestriero) | ICML 2026 (arXiv 2602.11389) | https://arxiv.org/abs/2602.11389 | 물체 단위 latent masking 으로 부분 관측을 강제한다. 반사실 VQA 에서 약 20pt 오른다. | '가려져도 상태를 추론하게 만드는 predictor 학습' 의 가장 가까운 경쟁 설계다. 새 predictor 절에서 반드시 비교해야 한다. |
| Reliability of CKA as a Similarity Measure in Deep Learning (Davari et al.) | ICLR 2023 | https://arxiv.org/abs/2210.16156 | CKA 는 이상치와 선형 분리성을 보존하는 단순 변환에 민감해서 직관에 반하는 결과를 낸다. | 사용자가 말한 'cos sim / representation geometry' 로 encoder 충분성을 보일 때 CKA 하나에 기대면 공격받는다. 중심화 cos (h̄ 차감)나 RSA 와 병기해야 한다. |


---

## 영역: 관측 없이 latent 상태를 유지·진화시키는 predictor/dynamics 설계 (frozen JEPA encoder 위 새 predictor): RSSM·재귀 롤아웃·객체 중심 동역학·메모리·block-causal JEPA predictor·rollout 학습·belief state·가림 지속 모델

# 관측 없이 상태를 진화시키는 predictor — 설계 문헌 조사 (2026-09-25)

> 범위: frozen JEPA encoder 위에 새 predictor 를 만들 때 **(a) 가림 아래 지속 · (b) 자기 되먹임 drift · (c) 먼 지평 감쇠** 를 무엇이 고치는지에 대한 선행 증거.
> 우리 발견은 `auto_research/README.md` (적대 검증 뒤 정정판) 기준이다. 모든 인용은 arXiv 초록, HTML, PDF 에서 존재를 확인했다. 확인하지 못한 venue 는 §9 에 모았다.

## 0. 결론 먼저

**결론.** 문헌은 우리 진단을 **이미 알려진 세 실패 유형**으로 번역해 준다. 각 유형의 처방 부품도 흩어져 있다.
- (a) **관측→상상 전환 장치의 부재**: percept gate, blackout 학습, 운반되는 state.
- (b) **exposure bias + 입출력 공간 불일치**: 같은 공간에서 예측·되먹임, 짧은 rollout loss, 입력 잡음.
- (c) **학습 안 된 질의 오프셋**: 가변 Δt, 끝점 목적, 계층.

이 부품들을 한 predictor 에 모은 사례는 V-JEPA 2-AC 가 가장 가깝다. 다만 (a)(c) 부품은 없다.

**근거 3줄.**
1. Nayebi et al. (NeurIPS 2023): frozen 비디오 encoder + 단순 CTRNN 이 가려진 공을 시뮬레이션한다 (46–49%). 복사('No Dynamics')는 19–25% 다. **encoder 충분 + dynamics 필요**의 직접 선례이고, 우리 복사 기준선과 같은 설계다.
2. V-JEPA 2-AC: 프레임별 인코딩, 같은 공간 되먹임, block-causal, rollout T=2 를 쓴다. 우리 병목 중 셋을 이미 고쳤는데도 '롤아웃이 길수록 오차 누적'을 한계로 적었다.
3. Loci-Looped: 게이트로 관측을 드물게 쓰게 하면 모델이 가림을 흉내 내며 학습하고, 가림 중 99.2% 상상으로 추적한다. 반면 SAVi·G-SWM 은 단순 가림에서도 무너진다.

**미결 1줄.** 'encoder is enough'는 probing 으로는 닫히지 않는다. frozen 위 새 predictor 가 (a)(b)(c)를 회복하는지로만 닫힌다. 실패하면 결론은 GRWM·Semigroup-JEPA 쪽(표현 기하가 병목)으로 뒤집힌다.

## 1. 우리 발견 → 문헌상 이름 → 고치는 부품

| 우리 발견 (정정판) | 문헌상 이름 | 처방 부품 | 선행 증거 |
|---|---|---|---|
| 한 번에 질의(P32)하면 먼 슬롯의 물체 신호가 문맥 끝부터의 시간을 따라 준다. 어떤 간선 차단으로도 대부분 되돌아오지 않는다 | stateless 병렬 mask 질의. pretrain 마스크가 시간 전체에 걸친 공간 tube 라서 future-only 질의는 분포 밖이다 | block-causal 롤아웃 + 운반 state (slot/register/SSM) + future-only·가변 Δt 학습 | V-JEPA 2-AC, DINO-world, Dreamer 4, SlotSSM, S4WM, V-JEPA 마스킹 |
| 자기 예측을 되먹이면 두 걸음 뒤 멈춘다. 참 미래를 target 공간으로 넣어도(ar_h) j≥4 에서 되먹인 칸으로 돌아간다 | exposure bias + 문맥(online) 공간과 표적(EMA+LN) 공간의 불일치 | 같은 공간에서 입출력, rollout loss (2–6 step), 입력 잡음, pushforward | 2-AC, PLDM, JEPA-WMs, GNS, MP-PDE, GameNGen, Self Forcing, Diffusion Forcing |
| 문맥 경계에서 가리면 미래에 물체가 없다 | 관측→상상 전환 부재, 'tracking shot' | percept gate + 관측 희소화(blackout) 학습, 객체 state, 운동 사전 (Kalman), belief | Loci-Looped, PLATO, ADEPT, PermaTrack, SAMURAI, RSSM, UWM-JEPA, WRBench |
| 물체 자리가 등속 외삽보다 뒤처진다. 가속 효과가 거의 없다 | case-based 일반화, 짧은 전이 | 명시적 속도 state, 끝점 손실, semigroup 일관성 | Kang et al., DPWM, Semigroup-JEPA, SAMURAI |
| IntPhys1 에서 복사 85.00 vs p 88.89 (n.s.) | 'No Dynamics' 대조 | 모든 채점에 표준 기준선으로 병기 | Nayebi et al. |
| 양방향 target encoder 가 사건 전 튜블릿으로 미래를 번지게 한다 | 표적 누수 | 프레임별 또는 인과 창 표적 | 2-AC ("encode each frame independently") |
| 릴리즈 RoPE 는 순수 상대적이지 않다 | 학습되지 않은 위치 오프셋의 외삽 | 절대시간 RoPE + Δt 균등 표본, W^p ≤ W | DINO-world, JEPA-WMs |

## 2. (a) 가림 아래 지속

- **가장 직접적인 처방은 Loci-Looped** (arXiv 2310.10372). slot 마다 percept gate 가 관측과 내부 상상을 Kalman 식으로 보간한다. 게이트 열림에 **L0 손실**을 걸면 보일 때도 91.1% 를 상상으로 처리하고, 결과적으로 "가림을 스스로 흉내 내며" 학습한다. 게이트가 닫히면 오차는 transition 으로만 역전파된다. ADEPT 가림 장면에서 SAVi, G-SWM, Loci-v1 을 큰 차로 이긴다. → 우리 판으로 옮기면: **운반되는 state 토큰 + 관측 토큰을 gated cross-attention 으로 주입 + 학습 때 문맥 끝이나 미래 프레임을 무작위 blackout**.
- **지속 ≠ 복사.** TSA (2026) 는 가려진 slot 을 이전 상태에 anchor 한다. 이것은 우리가 비판하는 '멈춘 채 기다림'과 같다. 영속의 정의(능력 4)에 '진화'를 넣고 복사형 해법을 대조군으로 둘 것.
- 객체 수준 표현이 결정적이라는 증거: PLATO (Nature Hum. Behav. 2022), ADEPT (particle filter belief, NeurIPS 2019). Causal-JEPA (ICML 2026) 는 slot 마스킹으로 '구조화된 부분관측'을 학습시킨다. → **patch 토큰만으로 되는지 slot/state 병목이 필요한지**는 ablation 축이다.
- 추적 쪽: PermaTrack (ConvGRU 메모리 + 합성 가림 감독, ICCV 2021), RAM (메모리 random walk 일관성, ICML 2022), SAMURAI (SAM 2 에 Kalman 운동 사전만 얹어 재학습 없이 LaSOT_ext AUC +7.1). → **강한 frozen 표현에 명시적 운동 state 만 더해도 가림·빠른 운동이 오른다.**
- belief 형 predictor: RSSM (결정적 + 확률적 상태, PlaNet), UWM-JEPA (2026, 가린 5-step 77% vs LSTM-JEPA 53%, blind rollout R² 손실 <10pt vs 41/68pt, 소규모·비관례 구조).

## 3. (b) 자기 되먹임 drift

1. **공간 일치.** 2-AC, DINO-world, DINO-WM 은 모두 같은 frozen encoder 공간에서 예측하고 되먹인다. 우리 릴리즈 predictor 는 문맥(online encoder) 공간에서 받고 표적(EMA + LN) 공간으로 낸다. H5 ar_h 결과는 이 전환 비용이 실재함을 시사한다. **새 predictor 는 자기가 받는 공간에 예측해야 한다.**
2. **rollout loss.** 2-AC 는 T=2 에 한 스텝만 역전파한다 (MP-PDE 의 pushforward 계보). JEPA-WMs 에서는 sim 에서 2-step, DROID 에서 6-step 이 최적이고, 그 이상은 "예측 과제에 덜 특화된다". PLDM 은 첫 인코딩에서 H 스텝을 완전히 unroll 한다. Self Forcing (2025) 은 KV cache 로 완전 자기 롤아웃을 학습한다 (상한 대조).
3. **입력 잡음.** GNS (ICML 2020) 는 "장기 성능의 주 결정 요인이 잡음 주입"이라고 했고, GameNGen 은 문맥 latent 잡음으로 수 분 롤아웃을 안정화했다. Diffusion Forcing 은 토큰별 잡음 수준으로 관측 blackout 과 되먹임 안정화를 한 틀로 묶는다.
4. 목적 쪽: NextLat (2025) 은 next-latent 보조 목적으로 transformer latent 가 belief 로 수렴하게 한다. Belief State Transformer (ICLR 2025) 도 같은 방향이다.

## 4. (c) 먼 지평 감쇠

- **학습 분포를 질의에 맞춘다.** V-JEPA 의 multiblock 마스크는 공간 블록을 시간 전체에 반복한다 (temporal ratio 100%). 그래서 pretrain predictor 는 'future-only' 를 학습한 적이 없다 (V-JEPA 2 가 같은 마스킹을 계승했는지는 재확인 필요). DINO-world 는 Δt 를 균등 표본하고 RoPE 에 절대 timestamp 를 넣었다. JEPA-WMs 는 계획 문맥이 학습 문맥을 넘으면 급격히 나빠진다고 보고했다. → 우리 RoPE 비상대성과 시간 감쇠는 **먼저 '학습된 적 없는 오프셋'으로 설명해 보고**, 그다음 구조 탓으로 갈 것.
- **끝점 목적.** DPWM (2026) 에서는 재귀 기준선도 같은 long-horizon 끝점 목적으로 재학습하면 비슷하게 좋아진다. 구조보다 목적이 중요하다는 뜻이다. Semigroup-JEPA (2026) 는 합성 일관성을 쓴다. PlaNet 의 latent overshooting 은 모든 다단계 prior 를 학습한다.
- **계층.** Clockwork VAE (NeurIPS 2021) 와 Hierarchical Planning with Latent World Models (2026, 초록 기준 단일 수준 0% → 계층 70%).
- 한계를 인정한 선례: DINO-world 는 "1초에 가까우면 모든 예측이 부정확"하다고 썼고, 짧은 Δt 에서는 direct 가, 긴 지평에서는 AR 이 낫다고 했다.

## 5. 'encoder is enough' — 지지 / 반대 / 닫는 법

**지지.**
- Nayebi: VC-1/R3M latent 만 쓰면 19–25%, 여기에 CTRNN 을 얹으면 46–49% 다.
- Joseph et al. 2026 (V-JEPA 2 encoder): 속도·가속 크기는 초기 층부터, 방향은 중간 층부터 읽힌다.
- V-JEPA 2 frozen attentive probe SSv2 77.3.
- DINO-WM, DINO-world, 2-AC 모두 frozen encoder 위에서 작동한다.

**반대.**
- JEPA-WMs: 계획에서 DINOv2/v3 가 V-JEPA encoder 를 이긴다.
- GRWM (2025): 장기 병목은 표현 기하다.
- Semigroup-JEPA: rollout 손실이 encoder 를 운반 가능한 형태로 바꾼다.
- Joseph: 물리 probe 가 출력층으로 가며 떨어진다. predictor 가 받는 층이 최종층이다.
- Nayebi 안에서도 VIP·정지 이미지 encoder 는 dynamics 를 얹어도 실패한다.
- 우리 H1: 가속은 C8 ≈ 0.

**닫는 법.** 구성적 증명이다 (Hewitt & Liang 의 decodable ≠ usable 을 피하려면). frozen V-JEPA 2 위 작은 predictor 가 (a)(b)(c)를 회복하면 'encoder 충분'이 확정된다. 주장 범위는 '읽기(정체성·위치·속도·방향)'로 한정하고 가속은 미결로 둔다. ablation 에 **다층 입력** (DINO-Foresight 식) 과 **encoder 교체** (DINOv2/3) 를 넣어 반대 가설을 직접 배제한다.

## 6. 새 predictor 최소 설계안 (제안, 미검증)

- **입력·표적**: frozen V-JEPA 2 ViT-H.
  - 표적은 **인과 창** (과거만 본 encoder 호출) 이나 2-AC 식 프레임별 인코딩을 쓴다.
  - ⚠️ 트레이드오프: 프레임별로 인코딩하면 표적 누수는 없지만 encoder 의 튜블릿 내 운동 정보를 잃는다. 인과 슬라이딩 창이 절충안이다.
  - 예측·되먹임은 **입력과 같은 공간**에서 한다.
- **구조**: 2-AC 모양의 block-causal transformer (무액션) + **운반되는 state 토큰 K 개** (register/slot 형, 스텝마다 전이). 관측은 **gated cross-attention** 으로 들어오고, 게이트에 희소 정규화를 건다 (Loci-Looped).
- **학습**:
  - teacher forcing next-step L1
  - rollout k-step (2→6 커리큘럼) + 되먹임 입력 잡음
  - **관측 blackout**: 입력에서는 프레임·영역을 빼고 표적은 참값을 유지한다
  - **가변 Δt 끝점 손실**
- **기준선**:
  - 복사 (No Dynamics)
  - 등속 외삽 (읽기 공간)
  - 릴리즈 predictor
  - 2-AC 식 무액션 predictor (state 토큰·blackout·Δt 없음)
- **부품 ↔ 측정 대응**:
  - blackout + gate → (a) v11 late 가림, 문맥 경계 가림
  - 공간 일치 + rollout + 잡음 → (b) H5 프로토콜 (ar_p 기울기, 착지, 위치 귀무)
  - Δt + 끝점 → (c) H4/H23 프로토콜 (공통 문턱 물체다움, 등속 대비 진행률)
  - 외부 검증: IntPhys 1/2, SSv2 방향 (가림·끊김 조건 추가)

## 7. 이 문제가 중요한가 / 풀어야 하는가 (비판적 판정)

- **중요하다는 근거.**
  - V-JEPA 2 스스로 롤아웃 오차 누적을 한계로 적었다.
  - 계층 계획 논문 (Meta 공저) 이 같은 문제를 동기로 삼았다.
  - WRBench 는 23 개 비디오 world model 이 영속에 실패하고 규모를 키워도 나빠진다고 보고했다.
  - IntPhys 2 는 chance 근처다.
  - 데이터만 바꿔도 안 오른다 (Opinion 2025).
- **약하다는 근거.**
  - 현재 2-AC 계획은 짧은 지평의 closed-loop 로 매 스텝 재관측한다.
  - 경계 가림은 corner case 다 (CLAUDE.md 09-03).
- **판정.** 주장의 중심은 **비가림에서도 성립하는 (b) 되먹임 멈춤과 (c) 먼 지평 감쇠**에 두고, 가림은 testbed 로 쓴다. use case 는 '계획 롤아웃' (계층 계획), 'anticipation' (EK100 은 pretrained predictor 를 예측기로 쓴다), '평가 방법론' (복사 기준선) 세 곳이다.

## 8. 예상 리뷰어 공격과 대응 (선행 연구에서 추론; OpenReview 는 403 으로 직접 수집 실패)

1. "2-AC 에서 action 만 뺀 것 아닌가" → 2-AC 식 기준선을 직접 학습하고, 추가 부품별 이득을 분리한다.
2. "DINO-world, Nayebi, WRBench, UWM-JEPA 가 이미 말했다" → 대상 (벤치마크가 쓰는 JEPA predictor), 통제 물리, 병목 국소화, 처방 연결이 차별점이다. 한 문단으로 명시한다.
3. "probing ≠ use" → 구성적 실험으로 답한다.
4. "encoder 가 병목이다 (JEPA-WMs, GRWM, Joseph)" → encoder 교체와 다층 입력 ablation 으로 답한다.
5. "복사 결과는 벤치마크 결함이다" → 평가 방법론 기여로 따로 세운다.
6. "합성뿐이다" → SSv2 가림 조건과 IntPhys 2 로 보완한다.
7. "slot 이면 다 풀린다 (Loci-Looped, PLATO)" → patch 토큰 + state 토큰 대 slot 병목을 ablation 한다. 어느 쪽이든 인용한다.

## 9. 미확인·주의

- 2026 arXiv 프리프린트 (WRBench, UWM-JEPA, Semigroup-JEPA, DPWM, TSA, Echo-Memory, Hierarchical Planning, Joseph et al.) 는 초록·HTML 요약 기준으로만 확인했다. 수치는 인용 전에 원문 표로 재확인할 것.
- venue 미확인: Loci-Looped 의 ICANN 2024 (Springer 목록 기준), Self Forcing, GameNGen, JEPA-WMs (OpenReview 등재만 확인), V-JEPA 의 TMLR 2024, WorldMem 의 NeurIPS 2025 (GitHub 표기).
- V-JEPA 2 가 V-JEPA 의 multiblock (temporal 100%) 마스킹을 그대로 쓰는지는 원문 확인이 필요하다. §4 의 핵심 논거이므로 먼저 확인할 것.
- 'The JEPA Predictor: A Transferable Operator for Occluded Feature Completion' (2607.16274) 은 **철회된 논문**이다. 인용 금지.
- 검색 요약기가 PLDM 을 'Probabilistic Latent Diffusion Model' 로 오기한 사례가 있었다. PLDM 은 Planning with Latent Dynamics Models 다.

## 출처

- [V-JEPA 2 (2506.09985)](https://arxiv.org/abs/2506.09985) · [V-JEPA (2404.08471)](https://arxiv.org/abs/2404.08471) · [V-JEPA 2.1 (2603.14482)](https://arxiv.org/abs/2603.14482)
- [DINO-world (2507.19468)](https://arxiv.org/abs/2507.19468) · [DINO-WM (2411.04983)](https://arxiv.org/abs/2411.04983) · [DINO-Foresight (2412.11673)](https://arxiv.org/abs/2412.11673)
- [JEPA-WMs (2512.24497)](https://arxiv.org/abs/2512.24497) · [PLDM (2502.14819)](https://arxiv.org/abs/2502.14819) · [Hierarchical Planning (2604.03208)](https://www.alphaxiv.org/abs/2604.03208) · [Causal-JEPA (2602.11389)](https://arxiv.org/abs/2602.11389) · [LeWorldModel (2603.19312)](https://arxiv.org/abs/2603.19312)
- [Nayebi et al. (2305.11772)](https://arxiv.org/abs/2305.11772) · [Loci-Looped (2310.10372)](https://arxiv.org/abs/2310.10372) · [PLATO](https://www.nature.com/articles/s41562-022-01394-8) · [ADEPT](https://proceedings.neurips.cc/paper/2019/hash/e88f243bf341ded9b4ced444795c3f17-Abstract.html)
- [SlotFormer (2210.05861)](https://arxiv.org/abs/2210.05861) · [SlotSSM](https://proceedings.neurips.cc//paper_files/paper/2024/hash/158ac5698e36a01ee5ca9e6732685b34-Abstract-Conference.html) · [TSA (2606.13714)](https://arxiv.org/abs/2606.13714) · [OCVT (2107.09240)](https://arxiv.org/abs/2107.09240) · [SAVi++ (2206.07764)](https://arxiv.org/abs/2206.07764)
- [PermaTrack](https://openaccess.thecvf.com/content/ICCV2021/html/Tokmakov_Learning_To_Track_With_Object_Permanence_ICCV_2021_paper.html) · [RAM (2204.01784)](https://arxiv.org/abs/2204.01784) · [SAMURAI (2411.11922)](https://arxiv.org/abs/2411.11922)
- [PlaNet (1811.04551)](https://arxiv.org/abs/1811.04551) · [S4WM (2307.02064)](https://arxiv.org/abs/2307.02064) · [R2I (2403.04253)](https://arxiv.org/abs/2403.04253) · [Dreamer 4 (2509.24527)](https://arxiv.org/abs/2509.24527) · [Clockwork VAE (2102.09532)](https://arxiv.org/abs/2102.09532)
- [Belief State Transformer (2410.23506)](https://arxiv.org/abs/2410.23506) · [NextLat (2511.05963)](https://arxiv.org/abs/2511.05963) · [UWM-JEPA (2605.25313)](https://arxiv.org/abs/2605.25313)
- [GNS (2002.09405)](https://arxiv.org/abs/2002.09405) · [MP-PDE (2202.03376)](https://arxiv.org/abs/2202.03376) · [Diffusion Forcing (2407.01392)](https://arxiv.org/abs/2407.01392) · [Self Forcing (2506.08009)](https://arxiv.org/abs/2506.08009) · [GameNGen (2408.14837)](https://arxiv.org/abs/2408.14837)
- [WRBench (2606.20545)](https://arxiv.org/abs/2606.20545) · [Mechanistic View (2601.17067)](https://arxiv.org/abs/2601.17067) · [Interpreting Physics (2602.07050)](https://arxiv.org/abs/2602.07050) · [GRWM (2510.26782)](https://arxiv.org/abs/2510.26782) · [Semigroup-JEPA (2609.10464)](https://arxiv.org/abs/2609.10464) · [DPWM (2608.07420)](https://arxiv.org/abs/2608.07420)
- [Kang et al. (2411.02385)](https://arxiv.org/abs/2411.02385) · [IntPhys 2 (2506.09849)](https://arxiv.org/abs/2506.09849) · [Garrido et al. (2502.11831)](https://arxiv.org/abs/2502.11831) · [Opinion (2512.06232)](https://arxiv.org/abs/2512.06232)
- [Long-Context SSM Video WM (2505.20171)](https://arxiv.org/abs/2505.20171) · [Echo-Memory (2606.09803)](https://arxiv.org/abs/2606.09803) · [WorldMem (2504.12369)](https://arxiv.org/abs/2504.12369) · [TECO (2210.02396)](https://arxiv.org/abs/2210.02396)
- [Navigation World Models (2412.03572)](https://arxiv.org/abs/2412.03572) · [Control Tasks probes (1909.03368)](https://arxiv.org/abs/1909.03368)


### 논문 목록

| 제목 | 학회·연도 | URL | 요점 | 우리와의 관계 |
|---|---|---|---|---|
| V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning | arXiv 2025 (2506.09985) | https://arxiv.org/abs/2506.09985 | V-JEPA 2-AC: frozen encoder 위에 새로 학습한 약 300M(24층) block-causal predictor. 각 프레임을 이미지처럼 독립 인코딩하고, teacher forcing(15 위치)과 rollout loss(T=2, 한 스텝만 역전파)를 같이 쓴다. 한계 절에 '자기회귀 롤아웃이 길어질수록 정확도가 떨어져 긴 지평 계획이 어렵다'고 명시되어 있다. EK100 anticipation 은 pretrained predictor 에 1초 뒤 mask token 을 넣어 미래 표현을 예측하는 방식이다. | (1) '왜 pretrained predictor 를 보나'의 직접 근거다. EK100 은 그 predictor 를 예측기로 쓴다. (2) 2-AC 는 우리가 찾은 병목 중 표적 번짐(→프레임별 인코딩), 입출력 공간 불일치(→같은 encoder 공간), 미래 토큰 사이 상태 전달 부재(→block-causal), 되먹임(→rollout loss)을 이미 부분적으로 고친 설계다. 그래서 새 predictor 의 기준선이자 가장 큰 novelty 위험이다. |
| Revisiting Feature Prediction for Learning Visual Representations from Video (V-JEPA) | arXiv 2024 (2404.08471); TMLR 2024 로 알려져 있으나 미확인 | https://arxiv.org/abs/2404.08471 | multiblock 마스크는 공간 블록을 클립 전체 시간에 걸쳐 반복한다 (temporal masking ratio 100%, 시공간 tube). | pretraining predictor 가 배운 과제는 '과거→미래'가 아니라 시간 전체에 걸친 공간 구멍 메우기다. 그래서 IntPhys/EK100 식 future-only 질의는 학습 분포 밖이다. 먼 지평 감쇠와 RoPE 비상대성이 문제되는 이유를 설명하는 후보다. V-JEPA 2 가 같은 마스킹을 물려받았는지는 원문으로 재확인이 필요하다. |
| Back to the Features: DINO as a Foundation for Video World Models (DINO-world) | arXiv 2025 (2507.19468) | https://arxiv.org/abs/2507.19468 | frozen DINOv2 위에 cross-attention predictor 를 둔다. block-triangular(인과) 마스크를 쓰고, RoPE 에 절대 timestamp(초)를 넣고, Δt 를 균등 표본해 가변 간격을 학습한다. 학습은 next-frame teacher forcing. IntPhys 91.3 / GRASP 76.0 / InfLevel 63.7 로, 같은 표의 V-JEPA ViT-H 는 89.4 / 73.0 / 59.9 다. 저자들은 '1초에 가까워지면 모든 예측이 부정확해진다'고 스스로 적었다. 짧은 Δt 에서는 direct 가, 긴 지평에서는 AR 이 낫다. | 'frozen encoder + 새 predictor + intuitive physics 채점' 조합의 직접 선례다 (novelty 위험). 설계 면에서는 가변 Δt 와 절대시간 RoPE 가 우리 RoPE·먼 지평 문제의 후보 해법이다. 다만 DINO-world 도 먼 지평 문제는 풀지 못했다. |
| What Drives Success in Physical Planning with Joint-Embedding Predictive World Models? (JEPA-WMs) | arXiv 2025 (2512.24497); OpenReview 등재, 채택 여부 미확인 | https://arxiv.org/abs/2512.24497 | rollout loss 는 teacher forcing 대비 2-step 까지는 좋아지지만 시뮬레이션 환경에서는 그 뒤로 나빠지고, DROID 에서는 6-step 이 최적이다. 계획 시 문맥 길이는 학습 문맥 이하여야 한다 (W^p ≤ W). 계획 성공률에서는 DINOv2/v3 encoder 가 V-JEPA encoder 보다 뚜렷이 낫다 (세밀한 객체 분할 때문이라고 해석). proprioception 은 도움이 된다. | (b) 되먹임의 처방 폭(rollout 스텝 수)을 보여 주는 실측이다. 동시에 'V-JEPA 2 encoder is enough'에 대한 반대 증거다. 학습 분포 밖 질의가 급격히 나빠진다는 결과는 우리 먼 지평 감쇠와 같은 방향이다. |
| Learning from Reward-Free Offline Data: A Case for Planning with Latent Dynamics Models (PLDM) | arXiv 2025 (2502.14819) | https://arxiv.org/abs/2502.14819 | z0 = Enc(s0) 에서 시작해 ẑ_t = f(ẑ_{t−1}, a_{t−1}) 로 H 스텝을 자기 예측으로 펼친 채 학습한다 (완전 재귀 unroll). VICReg, 역동역학, 시간 평활 정규화를 쓰며 encoder 와 predictor 를 함께 학습한다. | '자기 출력으로 이어 가며 학습'하는 JEPA 계열의 극단형이다. 우리 H5 의 되먹임 멈춤이 학습 방식(teacher forcing 만 씀) 문제인지 가르는 대조 설계로 쓸 수 있다. |
| Neural Foundations of Mental Simulation: Future Prediction of Latent Representations on Dynamic Scenes | NeurIPS 2023 (2305.11772) | https://arxiv.org/abs/2305.11772 | frozen 비디오 foundation encoder(VC-1, R3M) 위에 단순 LSTM/CTRNN dynamics 를 학습하고 Mental-Pong 가림 구간을 unroll 한다. DMFC 신경 예측도는 46.34–48.83% 인데, 같은 encoder latent 만 쓰면('No Dynamics' = 마지막 문맥 latent 복사) 19.03 / 24.79% 다. 비교 모델들은 end-to-end 픽셀 13.69–25.35%(SVG/FitVid)와 C-SWM 슬롯 13.69–23.05%, 정지 이미지 encoder 는 최대 29.67%, VIP 는 7.37–18.12% 다. CTRNN 만으로도 LSTM 과 같거나 낫다. | 'encoder 는 충분하고 상태 진화는 dynamics 모듈의 몫'이라는 우리 논지의 가장 가까운 선례이자 지지 증거다. 'No Dynamics' 대조는 우리 IntPhys1 복사 기준선(85.00 vs 88.89)과 같은 설계라 표준 기준선으로 인용할 수 있다. 반면 encoder 목적(VIP)이나 정지 이미지 사전학습에 따라 dynamics 를 얹어도 실패한다. 즉 encoder 선택도 중요하다는 반대 증거가 같은 논문 안에 있다. |
| Learning Object Permanence from Videos via Latent Imaginations (Loci-Looped) | arXiv 2023 (2310.10372); Springer ICANN 2024 수록으로 보임 (venue 는 Springer 목록 기준, 원문 미확인) | https://arxiv.org/abs/2310.10372 | 슬롯마다 percept gate 를 두어 관측과 내부 상상의 섞는 비율을 학습한다 (Kalman 유사 보간). 게이트 열림에 L0 손실을 걸어 관측을 드물게 쓰게 하면, 모델이 '가림을 스스로 흉내 내며' 학습한다. 가림 중 inner-loop 통합은 99.2%, 보일 때도 91.1% 다. SAVi, G-SWM, Loci-v1 은 단순한 가림에서도 무너진다. 게이트가 닫히면 오차가 transition 모듈로만 역전파된다. | (a) 가림 지속의 핵심 부품인 '관측 기반 갱신 → 상상 기반 갱신 전환'과 (b) '자기 상상으로 학습'을 한 장치로 묶었다. 우리 발견('경계에서 가리면 미래에 물체가 없다')에 가장 직접적인 처방 후보다. 단 규모가 작은 합성 데이터이고 객체 슬롯 기반이다. |
| Intuitive physics learning in a deep-learning model inspired by developmental psychology (PLATO) | Nature Human Behaviour 2022 | https://www.nature.com/articles/s41562-022-01394-8 | 객체 중심 표현과 추적, 동역학 predictor 로 VoE 효과를 학습한다. 물리 개념 학습이 객체 수준 표현에 결정적으로 의존한다. | 'patch 토큰 predictor 로는 부족하고 객체 상태가 필요하다'는 리뷰어 반론의 근거가 될 수 있다. 새 predictor 에 slot/state 토큰을 둘지 결정하는 ablation 의 근거다. |
| Modeling Expectation Violation in Intuitive Physics with Coarse Probabilistic Object Representations (ADEPT) | NeurIPS 2019 | https://proceedings.neurips.cc/paper/2019/hash/e88f243bf341ded9b4ced444795c3f17-Abstract.html | 거친 객체 표현, 확률적 물리 시뮬레이션, particle filter 로 가림을 넘어 기대를 형성한다. 명시적 belief 를 유지하는 방식이다. | '가림 중 belief 를 유지·전파'하는 고전적 형태다. 우리 'observation-anchored retrieval'과 대비되는 기준점이다. |
| SlotFormer: Unsupervised Visual Dynamics Simulation with Object-Centric Models | ICLR 2023 (2210.05861) | https://arxiv.org/abs/2210.05861 | 고정된 slot 표현 위의 transformer 자기회귀 동역학이다. 장기 객체 동역학 합성에서 기존보다 좋고, VQA와 계획에 쓰인다. | 'frozen 표현 위 자기회귀 transformer dynamics' 설계의 객체 중심 판이다. 우리 predictor 에 slot 병목을 넣을 때의 비교 대상이다. |
| Slot State Space Models (SlotSSM) | NeurIPS 2024 | https://proceedings.neurips.cc/paper_files/paper/2024/hash/158ac5698e36a01ee5ca9e6732685b34-Abstract-Conference.html | 상태를 여러 slot 벡터로 두고 slot 별 SSM 전이를 독립적으로 하며, slot 사이 상호작용은 self-attention 병목으로만 둔다. 긴 문맥 비디오 이해에 강하다. | '운반되는 상태(state carried across steps)'를 객체 단위로 둔 확장 가능한 형태다. (a)(c)의 후보 부품이다. |
| TSA: Temporal Slot Activation for Persistent Object-Centric Video Representation | arXiv 2026 (2606.13714) | https://arxiv.org/abs/2606.13714 | 비활성(가려진) slot 을 이전 상태에 anchor 해서 보존한다. 진화시키지 않고 복사하는 방식이다. | '지속 = 복사'와 '지속 = 진화'(Loci-Looped)를 가르는 대조 사례다. 우리가 요구하는 영속은 복사가 아니라 진화라는 점을 정의 수준에서 구분하는 데 쓴다. |
| Learning to Track with Object Permanence (PermaTrack) | ICCV 2021 | https://openaccess.thecvf.com/content/ICCV2021/html/Tokmakov_Learning_To_Track_With_Object_Permanence_ICCV_2021_paper.html | ConvGRU 시공간 재귀 메모리로 완전 가림 객체를 추론한다. 합성 데이터의 가림 감독과 실제 데이터를 함께 학습한다. | 재귀 메모리와 '가림 상황을 명시적으로 학습에 넣기'가 영속에 필요하다는 추적 쪽 선례다. |
| Object Permanence Emerges in a Random Walk along Memory | ICML 2022 (2204.01784) | https://arxiv.org/abs/2204.01784 | 메모리 시공간 그래프 위 random walk 의 시간 일관성을 자기지도 목적으로 쓰면, 가려진 객체를 저장하고 움직임을 예측하는 메모리가 생긴다. | 라벨 없이 영속을 학습하는 자기지도 목적의 선례다. frozen encoder 위에서도 쓸 수 있는 보조 손실 후보다. |
| SAMURAI: Adapting Segment Anything Model for Zero-Shot Visual Tracking with Motion-Aware Memory | arXiv 2024 (2411.11922) | https://arxiv.org/abs/2411.11922 | SAM 2 의 메모리 선택에 Kalman filter 운동 모델을 얹는다. 재학습 없이 LaSOT_ext AUC +7.1%, GOT-10k AO +3.5% 다. | 강한 frozen 표현 위에 '명시적 운동 사전'만 더해도 가림·빠른 운동에서 오른다. 우리 '등속 외삽보다 뒤처짐'에 대한 최소 처방(등속 상태 변수)의 근거다. |
| Learning Latent Dynamics for Planning from Pixels (PlaNet / RSSM) | ICML 2019 (1811.04551) | https://arxiv.org/abs/1811.04551 | RSSM 은 결정적 재귀 상태와 확률적 상태를 결합한다. latent overshooting 으로 모든 다단계 prior 를 학습해 다단계 예측을 직접 최적화한다. | '관측 없이 prior 로만 전진'하는 상태를 학습 목적으로 명시한 원형이다. (b)(c)의 multi-step 목적과 (a)의 belief 상태 설계 근거다. |
| Facing Off World Model Backbones: RNNs, Transformers, and S4 | NeurIPS 2023 (2307.02064) | https://arxiv.org/abs/2307.02064 | S4WM 이 장기 상상, 문맥 의존 회상, 메모리 추론에서 RNN·Transformer 보다 낫다. RNN 은 상상 처리량이 가장 높다. | predictor 백본(attention 만 쓰는가, 재귀 상태를 두는가)의 선택이 장기 상상에 영향을 준다는 통제 비교다. |
| Mastering Memory Tasks with World Models (R2I) | ICLR 2024 oral (2403.04253) | https://arxiv.org/abs/2403.04253 | DreamerV3 world model 에 S4 계열 SSM 을 넣어 Memory Maze 등 메모리 과제에서 SOTA 를 낸다. | 재귀·SSM 상태가 부분관측 기억을 푼다는 RL 쪽 증거다. |
| The Belief State Transformer | ICLR 2025 (2410.23506) | https://arxiv.org/abs/2410.23506 | prefix 의 다음 토큰과 suffix 의 이전 토큰을 함께 예측하면 compact belief state 가 학습되어 forward-only transformer 가 못 푸는 과제를 푼다. | 'belief state 를 강제하는 목적'의 transformer 판이다. predictor 가 문맥을 상태로 압축하게 만드는 보조 목적 후보다 (언어 영역이라 이식성은 미검증). |
| Next-Latent Prediction Transformers Learn Compact World Models (NextLat) | arXiv 2025 (2511.05963) | https://arxiv.org/abs/2511.05963 | '다음 토큰이 주어질 때 다음 latent 를 예측'하는 보조 목적을 두면 transformer 에 재귀 구조가 들어간다. 그 latent 가 belief state 로 수렴함을 이론으로 보인다. | transformer predictor 에 재귀적 상태 일관성을 넣는 가벼운 목적이다. 우리 되먹임 멈춤을 줄이는 보조 손실 후보다. |
| UWM-JEPA: Predictive World Models That Imagine in Belief Space | arXiv 2026 (2605.25313) | https://arxiv.org/abs/2605.25313 | density-matrix latent 과 unitary predictor 로 롤아웃 중 불확실성 소산을 막는다. 가려진 목표의 5-step 전방 시뮬레이션(hidden-velocity)에서 77% 로 LSTM-JEPA 의 53% 를 넘는다. blind rollout 에서 probe R² 손실이 10pt 미만인데 벡터 latent 는 41 / 68pt 를 잃는다. 분리가 encoder 가 아니라 predictor 에서 생긴다고 주장한다. | '병목은 predictor, blind rollout 에서 belief 가 필요'라는 우리 논지와 같은 방향의 최신 선행이다 (novelty 위험). 다만 소규모이고 비관례적 구조이며 encoder 를 함께 학습한다. |
| Learning to Simulate Complex Physics with Graph Networks (GNS) | ICML 2020 (2002.09405) | https://arxiv.org/abs/2002.09405 | one-step 참값으로만 학습하면 롤아웃 입력이 분포 밖으로 나가 오차가 누적된다. 입력에 random-walk 잡음을 넣어 학습 분포를 롤아웃 분포에 가깝게 하는 것이 장기 성능의 주 결정 요인이다. | (b) 되먹임 drift 의 가장 싼 처방(잡음 주입)이다. 우리 ar_p 가 분포 밖 입력이라는 H5 단서와 같은 기제다. |
| Message Passing Neural PDE Solvers (pushforward trick) | ICLR 2022 (2202.03376) | https://arxiv.org/abs/2202.03376 | 모델을 2 스텝 펼치되 마지막 스텝만 역전파해, 자기 예측 오차가 섞인 입력에서 학습한다. | V-JEPA 2-AC 의 rollout loss(T=2, 한 스텝 역전파)와 같은 계보다. (b) 처방의 이론적 동기다. |
| Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion | NeurIPS 2024 (2407.01392) | https://arxiv.org/abs/2407.01392 | 토큰마다 독립된 잡음 수준으로 학습하는 인과 모델이다. 학습 지평을 넘는 비디오 롤아웃이 안정적이다. | '부분 관측(잡음 섞인 과거)에서 미래를 이어 간다'는 학습 형식이다. 관측 blackout 학습과 되먹임 안정화를 한 틀로 묶을 수 있다. |
| Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion | arXiv 2025 (2506.08009); 학회 채택 미확인 | https://arxiv.org/abs/2506.08009 | 학습 중에 KV cache 로 자기 생성 프레임을 조건 삼아 자기회귀 롤아웃을 수행하고, 비디오 수준 손실로 exposure bias 를 직접 줄인다. | (b)의 가장 강한 형태(완전 자기 롤아웃 학습)다. 비용이 크므로 우리 설계에서는 상한 대조로 쓴다. |
| Diffusion Models Are Real-Time Game Engines (GameNGen) | arXiv 2024 (2408.14837); 학회 채택 미확인 | https://arxiv.org/abs/2408.14837 | 문맥 프레임 latent 에 여러 수준의 Gaussian 잡음을 넣는 augmentation 이 자기회귀 drift 를 막아 수 분 길이의 롤아웃을 안정화한다. | (b) 잡음 주입 처방의 대규모 비디오 사례다. |
| Current World Models Lack a Persistent State Core (WRBench) | arXiv 2026 (2606.20545) | https://arxiv.org/abs/2606.20545 | 23개 비디오 생성 world model 이 모두 'tracking shot'으로 동작한다. 사라졌던 대상을 떠날 때의 상태 그대로 재개하고, 보이지 않는 동안 사건을 진행시키지 않는다. 재관측 상태 일관성은 0.62–0.66 이고, Wan 1.3B→14B 로 키우면 0.66→0.62 로 오히려 떨어진다. 해법으로 'what-memory'(숨은 변화를 적는 상태 writer)를 제안한다. | 우리 상위 메시지('관측할 수 없어도 상태를 진화시켜야 한다')와 거의 같은 주장을 비디오 생성 모델에서 먼저 했다. 가장 큰 novelty 위험이다. 차별점은 JEPA latent predictor, IntPhys/계획에 쓰이는 predictor, 통제된 물리, 병목 국소화이며, 이것을 명시해야 한다. |
| A Mechanistic View on Video Generation as World Models: State and Dynamics | arXiv 2026 (2601.17067) | https://arxiv.org/abs/2601.17067 | 현대 비디오 아키텍처는 'stateless'라서 고전적 상태 중심 world model 과 간극이 있다고 본다. 상태 구성(암묵적 문맥 vs 명시적 압축)과 동역학을 분리한 분류를 제시하고, 지속성 평가로의 전환을 주장한다. | 우리 진단('observation-anchored retrieval')을 받쳐 줄 개념 틀이다. 동시에 같은 주장이 이미 퍼져 있다는 신호다. |
| Interpreting Physics in Video World Models | arXiv 2026 (2602.07050) | https://arxiv.org/abs/2602.07050 | V-JEPA 2 (L/H/g) 와 VideoMAE-v2 encoder 만 분석했다. 속도·가속 크기는 초기 층부터 읽히고, 방향은 약 1/3 깊이의 'Physics Emergence Zone'에서 급격히 읽히며, probe 성능은 중간 층에서 정점을 찍고 출력 쪽으로 떨어진다. predictor 는 분석하지 않았다. | 'encoder is enough'의 지지 근거(물리 변수가 encoder 에 있다)이면서 위험 근거(predictor 가 받는 최종층에서는 약해진다)다. 이 논문이 비워 둔 predictor 분석이 곧 우리 기여 자리다. |
| Cloning Deterministic Worlds: The Critical Role of Latent Geometry in Long-Horizon World Models (GRWM) | arXiv 2025 (2510.26782) | https://arxiv.org/abs/2510.26782 | 장기 충실도의 주 병목은 dynamics 모델이 아니라 latent 표현의 기하 구조라고 주장한다. 시간 대조 정규화로 latent 를 다듬어 해결한다. | 우리 논지('병목은 predictor')에 정면으로 반대되는 선행 주장이다. 새 predictor 실험에 encoder 교체·다층 입력 ablation 을 넣어 이 대안을 배제해야 한다. |
| Semigroup-JEPA: Latent Dynamics Consistency for Zero-Shot Physics Generalization | arXiv 2026 (2609.10464) | https://arxiv.org/abs/2609.10464 | encoder 와 predictor 를 자기회귀 latent 롤아웃으로 함께 학습한다. multi-step rollout 손실을 표현까지 역전파하면 'predictor 가 앞으로 운반할 수 있는 특징'을 encoder 가 남긴다고 분석한다. 중력장을 바꾼 OOD 에서 개루프 오차를 최대 2배 줄인다. | frozen encoder 는 '읽을 수 있는 정보'는 가져도 '운반 가능한 형태'는 아닐 수 있다는 반론이다. 새 predictor 가 frozen 위에서 실패할 때의 대체 설명이 되므로 미리 적어 둔다. |
| Beyond Myopic World Models: Long-Horizon End-to-End Training for Direct Future Prediction (DPWM) | arXiv 2026 (2608.07420) | https://arxiv.org/abs/2608.07420 | few-step 손실은 국소 전이만 최적화하고 재귀 추론이 작은 오차를 키운다. 끝점을 직접 예측하는 long-horizon 목적은 지평이 길수록 이득이 크다. 재귀 기준선도 같은 목적으로 재학습하면 비슷하게 좋아지므로 구조보다 목적이 중요하다. | (c) 먼 지평 감쇠의 처방인 끝점·가변 Δt 손실의 근거다. 우리 H4 (단일 질의 vs joint) 결과와 연결된다. |
| How Far is Video Generation from World Model: A Physical Law Perspective | ICML 2025 (2411.02385) | https://arxiv.org/abs/2411.02385 | 분포 안에서는 완벽하게 일반화하지만 분포 밖에서는 실패한다. 규칙을 추상화하지 않고 가장 가까운 학습 예를 흉내 내는 'case-based' 일반화를 보인다. | 'observation-anchored retrieval' 명명의 선행 근거다. 우리 물체 순서·복사 기준선 결과를 retrieval 행동으로 해석할 때 인용한다. |
| Clockwork Variational Autoencoders | NeurIPS 2021 (2102.09532) | https://arxiv.org/abs/2102.09532 | 느리게 tick 하는 상위 latent 계층으로 1000 프레임 장기 비디오 예측을 개선한다. | (c)의 계층적·시간 추상 처방의 원형이다. |
| Hierarchical Planning with Latent World Models | arXiv 2026 (2604.03208) | https://arxiv.org/abs/2604.03208 | 예측 오차 누적을 동기로 다중 시간척도 latent world model 을 제안하고 VJEPA2-AC, PLDM, DINO-WM 에 적용한다. 초록 기준 pick-&-place 성공률이 단일 수준 0% 에서 70% 로 오른다. | '먼 지평 감쇠가 실제 계획을 막는다'는 use-case 근거다 (Meta 공저). |
| Training Agents Inside of Scalable World Models (Dreamer 4) | arXiv 2025 (2509.24527) | https://arxiv.org/abs/2509.24527 | block-causal transformer world model 과 shortcut forcing(x-prediction) 목적으로 장기 롤아웃과 상상 학습을 한다. 오프라인 데이터만으로 Minecraft 다이아몬드를 얻는다. | block-causal 과 자기 롤아웃 안정화를 대규모로 쓴 최신 사례다. |
| Long-Context State-Space Video World Models | arXiv 2025 (2505.20171) | https://arxiv.org/abs/2505.20171 | block-wise SSM scan 으로 긴 시간 메모리를 두고, dense local attention 으로 인접 프레임 일관성을 유지한다. | 'attention 문맥 + 운반 상태(SSM)' 혼합 설계의 비디오 사례다. |
| Echo-Memory: A Controlled Study of Memory in Action World Models | arXiv 2026 (2606.09803) | https://arxiv.org/abs/2606.09803 | raw context, 압축, 공간 요약, state-space 재귀를 통제 비교했다. raw context 가 강한 기준선이고 block-wise SSM 재귀가 open-domain 복귀에서 가장 낫다. 재생 품질로는 메모리 능력을 예측할 수 없다. | 메모리 부품 선택의 통제 실험이다. 'L1 이 좋아도 지속은 별개'라는 우리 H4 결과와 같은 교훈이다. |
| Causal-JEPA: Learning World Models through Object-Level Latent Masking | arXiv 2026 (2602.11389); arXiv 표기상 ICML 2026 채택 | https://arxiv.org/abs/2602.11389 | slot 수준 마스킹으로 가려진 객체 상태를 다른 객체에서 추론하게 한다. 반사실 VQA +약 20%p, patch 기반 대비 1% latent 로 제어한다. | '객체 수준 마스킹 = 구조화된 부분관측 학습'으로, 관측 blackout 학습의 객체 버전이다. |
| V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning | arXiv 2026 (2603.14482) | https://arxiv.org/abs/2603.14482 | dense predictive loss(보이는 토큰도 학습 신호)와 중간층 deep self-supervision 으로 dense 특징을 개선한다. 실로봇 grasp 에서 V-JEPA 2-AC 대비 +20pt. | Meta 의 다음 단계가 encoder 개선이었다는 점은 'encoder is enough'에 대한 리뷰어 반론 재료가 된다. |
| IntPhys 2: Benchmarking Intuitive Physics Understanding In Complex Synthetic Environments | arXiv 2025 (2506.09849) | https://arxiv.org/abs/2506.09849 | V-JEPA 2 를 포함한 모델이 영속, 불변, 시공간 연속, 고체성 전 부분에서 chance(50%) 근처다. | motivation 의 벤치마크 갭이다. |
| Opinion: Learning Intuitive Physics May Require More than Visual Data | arXiv 2025 (2512.06232) | https://arxiv.org/abs/2512.06232 | 발달 데이터(SAYCam)로 V-JEPA 를 학습해도 IntPhys 2 가 오르지 않는다. 시각 데이터의 양·분포만으로는 부족하다고 주장한다. | '데이터를 바꿔도 안 된다 → 구조(predictor)를 바꿔야 한다'는 논지의 간접 지지다. |
| DINO-Foresight: Looking into the Future with DINO | NeurIPS 2025 (2412.11673) | https://arxiv.org/abs/2412.11673 | frozen VFM 의 다층·고해상도 특징을 masked feature transformer 로 미래 예측한다. | '다층 encoder 특징을 predictor 입력으로' 넣는 설계의 선례다. Joseph 2026 의 최종층 약화에 대한 대안 ablation 이다. |
| Designing and Interpreting Probes with Control Tasks | EMNLP 2019 (1909.03368) — 잘 알려진 문헌, 이번 세션에서 재검색하지 않음 | https://arxiv.org/abs/1909.03368 | probe 정확도가 높아도 probe 의 표현력 때문일 수 있다. control task 로 selectivity 를 재야 한다. | 'encoder is enough 를 probing 으로만 보이면 decodable ≠ usable' 반론의 근거다. 구성적(새 predictor) 증명이 필요한 이유다. |


---

## 영역: 분석·진단 논문에 대한 실제 리뷰어 행태와 채택 기준 (CVPR/ICCV/ECCV/ICLR/NeurIPS/ICML). 사전학습 video·world model 분석, 물리 이해 probing, 직관물리 벤치마크 논문의 OpenReview 공개 리뷰에서 반복되는 공격, 성공한 반박, oral·spotlight 과 탈락을 가른 요인을 뽑아 우리 논문이 선제 대응할 공격 체크리스트로 정리했다.

# 분석·진단 논문의 채택 기준과 실제 리뷰어 행태 — 우리 논문이 먼저 막아야 할 공격

## 0. 결론

**요약.** 분석 논문 자체는 학회 가이드라인이 명시적으로 인정한다. 그러나 실제 리뷰는 같은 공격을 반복한다.
1. 주장 범위가 실험 범위(단일 모델·단일 데이터)보다 넓다.
2. 합성 데이터만 썼다.
3. descriptive 하고 해결책이 없다.
4. 원인이 모델이 아니라 측정 도구(probe·readout·문턱)일 수 있다.
5. 뻔하다, 그리고 정의가 없다.

oral·spotlight 는 공통으로 네 가지를 갖췄다: 여러 방법이 같은 답을 낸다, 기제를 설명한다, 최소 개입이나 새 측정 도구를 낸다, 여러 모델에서 확인한다.

**근거 3줄.**
- IntPhys 2 는 4×borderline accept 로 탈락했다. 리뷰어가 '원인 분석이 없다' 며 요구한 것(병목이 기억인가, 표현인가)이 곧 우리 질문이다.
- How Far 는 'VAE 가 병목이 아님' 을 먼저 보여 칭찬받았다. 우리 'encoder is enough' 가 같은 구조다.
- 단일 모델 공격은 binding spotlight 에서 리뷰어 전원의 초기 우려였고, 다중 모델과 대조군으로 해소됐다.

**미결 1줄.** CVPR 리뷰는 비공개다 (OpenReview CVPR 2026 reply 검색 0건). 그래서 CVPR 리뷰어 행태는 가이드라인과 ICLR·ICML·NeurIPS 공개 리뷰로 추정했다. ECCV 2026 가이드라인 페이지는 받아오지 못했다.

---

## 1. 가이드라인 원문 (직접 받아 확인)

| 학회 | 원문 |
|---|---|
| CVPR 2026 / ICCV 2025 (같은 문구) | "the fact that a proposed method does not exceed the state-of-the-art accuracy on an existing benchmark dataset is not grounds for rejection by itself" · "Claims in a review that the submitted work 'has been done before' MUST be backed up with specific references" · 한계 논의는 "weigh ... POSITIVELY" · rebuttal 은 "small experiments ... reasonably run within the rebuttal phase" 까지만 허용 ([CVPR 2026](https://cvpr.thecvf.com/Conferences/2026/ReviewerGuidelines), [ICCV 2025](https://iccv.thecvf.com/Conferences/2025/ReviewerGuidelines)) |
| ICLR 2026 | 네 질문: "What is the specific question ... Is the approach well motivated ... Does the paper support the claims ... What is the significance". "this does not necessarily require state-of-the-art results. Submissions bring value ... when they convincingly demonstrate new, relevant, impactful knowledge" ([ICLR 2026 Reviewer Guide](https://iclr.cc/Conferences/2026/ReviewerGuide)) |
| NeurIPS 2025/2026 main | "Does the work provide new insights, deepen understanding, or highlight important properties of existing methods?" · "a work that provides novel insights by evaluating existing methods ... is also equally valuable" · "authors should be rewarded rather than punished for being up front about the limitations" ([NeurIPS 2025](https://neurips.cc/Conferences/2025/ReviewerGuidelines)) |
| NeurIPS 2026 Evaluations & Datasets (auditing 트랙) | "Findings must be grounded in rigorous, systematic analysis — not superficial observations. If a negative result: is it deep and carefully controlled?" · "Negative results are valuable when rigorously supported" ([E&D 2026](https://neurips.cc/Conferences/2026/EvaluationsDatasetsReviewerGuidelines)) |
| 리뷰 잡음 | NeurIPS 2021 일관성 실험: 한 위원회가 채택한 논문의 50.6% 를 다른 위원회가 떨어뜨렸다. spotlight 추천의 절반 이상도 다른 위원회에서 탈락했다 ([blog](https://blog.neurips.cc/2021/12/08/the-neurips-2021-consistency-experiment/)) |

가이드라인과 실제 리뷰가 어긋나는 예가 있다. ICLR 2026 리뷰 한 건은 "For a top-tier conference like ICLR, a more substantial contribution would involve proposing a solution" 이라고 썼다 ([XtX3](https://openreview.net/forum?id=DCbQUijwtf&noteId=sZia8lROVj)). 저자들은 이 리뷰를 AI 생성이라고 지적했고, 논문은 결국 철회됐다.

---

## 2. 사례표 — 분석·진단 논문의 운명

| 논문 | 결과 (점수) | 가른 요인 |
|---|---|---|
| IntPhys 2 | NeurIPS'25 D&B **Reject** (4/4/4/4 = borderline accept ×4) | 원인 분석 없음, context 16 frame 교란, SAC 순위 |
| How Far is Video Generation from World Model | ICLR'25 **Reject** (3/6/5/8) → ICML'25 poster | 'known ID/OOD', 과대 문장, 허수아비 동기 → 재구성 뒤 채택. VAE 병목 검증이 칭찬받음 |
| Interpreting Physics in Video World Models | ICML'26 Accept (최종 4/5/4/4) | 단일 모델·descriptive 공격 → rebuttal 에서 모델·frame rate 추가. AC 가 주장 축소 요구 |
| Inferring Dynamic Physical Properties from Video Foundation Models | ICLR'26 **Withdrawn** (4/6/4/2) | 해결책 없음, 클래스당 모델 하나 (V-JEPA-2), kinematics/dynamics 혼동 |
| LikePhys | ICLR'26 Poster (6/4/4/…) | 합성 전용 공격을 matched pair + 하한 논리로 방어 |
| Morpheus | ICLR'26 **Reject** (6/6/4/2) → ICML'26 Accept | 추적 오차 민감도 분석 부재 → 강건성·인간 상관 추가 |
| Does Object Binding Naturally Emerge in Large Pretrained ViTs? | NeurIPS'25 **Spotlight** | 전원 'DINOv2 만 vs 제목' → CLIP·MAE·supervised 추가 + 대조 |
| Evaluating the World Model Implicit in a Generative Model | NeurIPS'24 **Spotlight** (4/3/9/5) | 기존 진단의 과대평가를 새 지표로 보임. readout·문턱 공격을 받음 |
| Emergent World Representations (Othello-GPT) | ICLR'23 top 5% | probe + 개입, testbed framing |
| Revisiting the Othello World Model Hypothesis | ICLR'25 **Reject** (5/3/5) | 정의 부재, 놀랍지 않음 |
| Can VLMs Learn Intuitive Physics from Interaction? | ICLR'26 **Reject** (2/6/2) | 좁은 범위, probe 대조 부족, 결론 과장 |
| Vision Transformers Need Registers | ICLR'24 **Oral** (8/8/8/8) | 분석 → 기제 → 최소 개입 → 다중 모델 |
| Are Emergent Abilities of LLMs a Mirage? | NeurIPS'23 **Oral** (7/8/9/7) | 평가 인공물로 서사를 뒤집음. 대안 가설 논의 요구 |
| PhysBench | ICLR'25 **Oral** (8×4) | 진단 + 처방 (PhysAgent) |
| PhyWorldBench | ICLR'26 **Oral** (초기 4/6/6/6) | 포괄성 + 다수 모델 + 인간 검증 평가기 |
| PhyGenBench | ICLR'25 **Reject** | 규모, 평가기 신뢰성 |
| V-JEPA | ICLR'24 **Reject** | AC: "misses ... unique modeling challenges that video data, such as temporal correlations due to motion" ([r108EL4op6](https://openreview.net/forum?id=WFYbBOEOtv&noteId=r108EL4op6)) |

---

## 3. 반복되는 공격 11종 — 원문과 효과가 있던 반박

### A1. 단일 모델, 주장 범위 > 실험 범위
- binding AC: "claims about 'Large Pretrained Vision Transformers' were not fully supported by an analysis that focused almost exclusively on the DINOv2 pretraining objective" ([328mSaifL0](https://openreview.net/forum?id=5BS6gBb4yP&noteId=328mSaifL0)). 리뷰어 한 명은 "The title and framing of the paper does not match the experimental setup" 이라고 썼다 ([1sug92evNW](https://openreview.net/forum?id=5BS6gBb4yP&noteId=1sug92evNW)).
- "The paper evaluates only one ... Video Self-Supervised Model (V-JEPA-2)" ([XtX3](https://openreview.net/forum?id=DCbQUijwtf&noteId=sZia8lROVj)).
- Interpreting Physics AC: "Narrow the main claims so they are clearly related to the experiment (dataset, motion regime, model families)" ([wdr5fSceeM](https://openreview.net/forum?id=aijGVmEG9Y&noteId=wdr5fSceeM)).
- **효과가 있던 반박:** 모델을 추가하되 **대조군에서 달라지는 결과**까지 보였다 (supervised ViT 는 binding 이 약하다). 이 대조가 공격을 강점으로 바꿨다.

### A2. 합성 데이터만
- "the method's validity depends on a curated set of synthetic simulations" ([hTWK](https://openreview.net/forum?id=6UJf6B8RZ8&noteId=g95hP3rGHM)). "Limited scenario diversity (2D, synthetic data)" ([How Far ICML](https://openreview.net/forum?id=DLlVjZQ7vD&noteId=ttSdttX9Nd)).
- **효과가 있던 반박:** "each valid–invalid video pair differs only by a specific, well-defined physics violation" 그리고 "if a model struggles on these controlled cases, it is unlikely to handle more noisy, unstructured real-world scenes" ([LikePhys 저자](https://openreview.net/forum?id=6UJf6B8RZ8&noteId=uZKCzmHEiv)). Othello-GPT 는 "a constrained task, as a testbed for research on world representations" 라고 반박했다 ([Pqz0ck3NrY](https://openreview.net/forum?id=DeG07_TcZvT&noteId=Pqz0ck3NrY)).

### A3. descriptive, 해결책 없음
- "failing to deeply analyze the underlying causes of model failures ... and also failing to propose targeted improvement directions" ([IntPhys 2 q2be](https://openreview.net/forum?id=Xpf5x3mLvn&noteId=jVZdtEOm3Z)).
- "the contribution remains largely descriptive and does not clearly demonstrate how the identified phenomenon could be utilized in practice" ([fK2C](https://openreview.net/forum?id=aijGVmEG9Y&noteId=bzUBm5zw5w)).
- "The work feels more like an excellent diagnostic study than a paper presenting a new advance" ([XtX3](https://openreview.net/forum?id=DCbQUijwtf&noteId=sZia8lROVj)).
- **효과가 있던 대응:** Registers 는 최소 개입으로 진단을 확인했다. 그래도 "does not have experiments showing that such behavior is eliminated by adding the register tokens" 라는 지적을 받았다 ([mTMB](https://openreview.net/forum?id=2dnO3LLiJ1&noteId=ojGA6WMjKm)). 처방은 **점수가 아니라 진단 시그니처가 사라짐**을 보여야 한다.

### A4. 측정 도구가 원인 (probe ≠ use, readout 오차, 문턱)
- "Poor performance according to the metric could stem from an bad transformation of actions from the world state, rather than a bad implicit world model" ([RjHx](https://openreview.net/forum?id=aVK4JFpegy&noteId=5mks29Rzpl)). 같은 논문에 지표 간 불일치 지적도 있었다 ([S5LR](https://openreview.net/forum?id=aVK4JFpegy&noteId=cVj9P19oLT)).
- "there is no sensitivity analysis showing metric robustness to visual noise or tracking failure" ([Morpheus ICLR](https://openreview.net/forum?id=1E6pburMKc&noteId=zarqBE5G9D)). ICML AC 는 rebuttal 의 "human correlation analysis (Spearman 0.708), robustness analysis under trajectory perturbation (<2.6% score deviation)" 을 채택 근거로 들었다 ([JDzjrQGHeI](https://openreview.net/forum?id=f4xAzcMug6&noteId=JDzjrQGHeI)).
- "Linear probes can decode relevant physical quantities ... but this competence does not translate into zero-shot performance" / "lacks controls such as image only probes or interventions" ([Tj6t](https://openreview.net/forum?id=XdLgOm5giq&noteId=3ry1VY4IN5)).
- "learnable query and additional network may lead to some shortcut" ([rtwz](https://openreview.net/forum?id=DCbQUijwtf&noteId=dtNDRLky4Y)).
- **효과가 있던 대응:** 한 방향으로 모인 증거 — 개입(Othello), ablation(binding), 대체 지표로 같은 결론(Othello 지표 교체), 도구 민감도 분석(Morpheus).

### A5. 점수가 친숙도를 잰다
- "the LikePhys metric may sometimes conflate distribution familiarity with physical plausibility" ([y5Dy](https://openreview.net/forum?id=6UJf6B8RZ8&noteId=mYaB0rsKA3)). surprise 채점에도 그대로 적용되는 공격이다.

### A6. 뻔하다 / 이미 알려졌다
- "The findings are primarily reiterations of known phenomena" / "an attempt to frame familiar findings in a new light" ([r3mF](https://openreview.net/forum?id=ZyLkNVHBZF&noteId=nJEETfVJZQ)). "I personally just think that is too obvious" ([hEH6](https://openreview.net/forum?id=DLlVjZQ7vD&noteId=VKl2zK39ow)). "not surprising" ([duNp](https://openreview.net/forum?id=1OkVexYLct&noteId=zIvyYWT8Lt)).
- **효과가 있던 반박:** 무엇이 새로운지를 좁혀 다시 썼다. How Far 는 combinatorial generalization 과 속성 우선순위로 ICML 에 붙었다.

### A7. 정의 부재
- "I'm missing a definition of world model. I cannot judge with full certainty whether the experiments adequately support the claims" ([ezgd](https://openreview.net/forum?id=1OkVexYLct&noteId=aVVDtJTtJE)).
- "does not clearly define what constitutes a physical or semantic variable in operational terms" ([iWcy](https://openreview.net/forum?id=aijGVmEG9Y&noteId=EMFJTZe6Sj)).
- "dynamics have to do with forces rather than observed changes in positions alone" ([fJH5](https://openreview.net/forum?id=DCbQUijwtf&noteId=Q472XAZ24e)).
- "if the foundation model fails to match a particular expert-defined representation, it does not mean the foundation model lacks a world model altogether" ([Genp](https://openreview.net/forum?id=i9npQatSev&noteId=tBfatOvBUz)).

### A8. 동기 허수아비
- How Far 리뷰어: "I would like some more evidence that the belief that these models are actually good world models ... is widespread in the literature". 저자 답글이 이 요구를 인용하고 UniSim 등으로 답했다 ([G0RzqbtB9e](https://openreview.net/forum?id=ZyLkNVHBZF&noteId=G0RzqbtB9e)).

### A9. 대안 가설과 교란
- "The performance bottleneck may not lie in the physics understanding capability but in the insufficient context" ([PW3t](https://openreview.net/forum?id=Xpf5x3mLvn&noteId=hiw7bPrz0U)).
- "Is it due to memory? Or representation and perception?" ([9GPH](https://openreview.net/forum?id=Xpf5x3mLvn&noteId=07jQGHXxI9)).
- "There was no discussion of the relationship to alternative hypotheses" ([ccWP](https://openreview.net/forum?id=ITw9edRDlD&noteId=IHydEpdJy5)).

### A10. 통계, 규모, 누수
- "The experiment is fairly small, and its not immediately clear how robust the result is" ([86Gp](https://openreview.net/forum?id=i9npQatSev&noteId=OguTYHoOcK)).
- "The test set of PhysBench is derived from the similar set of images as train" ([khH4](https://openreview.net/forum?id=Q6a9W6kzv5&noteId=1P5ZXUpkV0)).
- How Far 는 오염을 이유로 처음부터 학습한 설계를 방어했다 ([WcW9iTwjle](https://openreview.net/forum?id=ZyLkNVHBZF&noteId=WcW9iTwjle)).

### A11. 과대 문장
- "I think that these statements are too strong" ([9tpf](https://openreview.net/forum?id=ZyLkNVHBZF&noteId=KG2exrhEOy)). "I strongly disagree with the statement made in the conclusion" ([rGrg](https://openreview.net/forum?id=XdLgOm5giq&noteId=7tuuGAguoe)). AC: "Include a discussion on the limitation of the concept of PEZ (eg it's not understanding physics)".

---

## 4. 우리 논문 체크리스트 — 공격 → 현재 취약점 → 선제 조치

| # | 예상 공격 | 현재 상태 (auto_research README 기준) | 선제 조치 |
|---|---|---|---|
| 1 | 단일 모델 (A1) | V-JEPA 2 ViT-H 만. V-JEPA 2.1 (arXiv 2603.14482) 이 있다 | 세 시그니처를 ViT-L/g·2.1 에서 재현. 제목은 'the V-JEPA 2 predictor' 로 좁힘. 기준은 ViT-H 로 유지 |
| 2 | 'encoder is enough' 는 decodability 일 뿐 (A4) | 근거는 속도 R² 0.99. 가속은 약하고 교락 (H1). Joseph et al. 은 가속이 초기 층부터 decodable 이라 함 | **충분성 시험**: 같은 frozen z 위 닫힌형·등속 외삽 latent 예측기 vs p (먼 슬롯). 궤적 단위 split + control task + cos/geometry. 가속은 조건부 주장 |
| 3 | 병목이 readout 탓 (A4) | 문턱 인공물로 헤드라인 3개 철회 이력 | 공통 고정 문턱 · 위치 귀무 · 자 오차 예산 · 섭동 민감도 표를 본문에 |
| 4 | 허수아비 동기 (A8) | self-feedback 결과가 전면에 있다 | Garrido 원문 "without any additional adaptation, to probe the model's understanding of the world" 와 "never trained using a causal prediction task", V-JEPA 2 Table 20 을 1쪽에. 되먹임은 stress test 로, 주 주장은 one-shot 채점 프로토콜에 |
| 5 | 해결책 없음 (A3) | 범위가 '새 학습 없음' 이라 사용자 방향(new predictor)과 충돌 | **사용자 결정 필요.** (a) frozen 최소 개입 + 시그니처 소거, (b) 측정 도구를 기여로, (c) 새 predictor — post-FT 도메인 한정 결과 (v11 75.83 → 91.35 / IntPhys1 88.89 → 77.22) 를 정면으로 다룸 |
| 6 | 이미 했다 (HERA, Interpreting Physics) | 인용 안 됨 | 차별화 문단: 진화 측정 + 병목 귀속 + 점수 감사. Meta 리뷰어를 전제로 '반박' 이 아니라 '분해' 톤 |
| 7 | 합성·도메인 이동 (A2) | 전부 합성 | encoder 읽기 정상 → 지각 OOD 배제, 하한 논리. SSv2 는 사전학습에 포함 (V-JEPA 2 원문 확인) + 1-bit 이므로 motion energy 기준선 필수. 실영상 롤아웃 1건 |
| 8 | IntPhys 복사 역공 (A5, A10) | w32 +3.9 n.s. 이지만 **공식 칸 w16 은 +12.2 유의** | 전 격자 표, 복사 기준선에도 같은 선택 규칙 적용, '어느 방향으로도 약하다' 로만 |
| 9 | 정의 부재 (A7) | 네 능력 정의는 있음 | 각 능력 = 측정량 + 귀무 + 문턱을 식으로. 'kinematics' 로 한정. 'retrieval' 은 문맥 앵커이지 학습 사례 검색이 아님 |
| 10 | 내부 모순 | p probe shape 98.46 인데 '미래에 물체 토큰 없음'. 표적 encoder 가 미래를 번지게 함 | '정보는 있고 자리가 없다' 로 통일, 금지 표현 회피. 'encoder 는 관측만' 은 문맥 encoder 로 한정 |
| 11 | 대안 가설 (A9) | 흩어져 있음 | context 길이 · tubelet · RoPE · 표적 번짐 · mask token · 도메인 · 목적함수(discussion 한 문단) 표 |
| 12 | 통계 (A10) | 일부 clip 단위 CI, 가설 9개와 여러 철회 | 궤적 84 단위 bootstrap, 분석 사전 고정 + hold-out 1회, 다중 비교 보정 |
| 13 | 과대 문장 (A11) | "does not evolve latent state" 는 전칭 | 초록은 최약 증거 수준으로: '여러 걸음 진화에서 실패한다. 한 걸음 전이는 있다' |
| 14 | CVPR rebuttal 제약 | — | 다중 모델·실영상은 제출본에. auditing 기여는 NeurIPS E&D 적합도가 높다 |

**corner case / use case.** 가림이 문맥 경계에 걸리는 조건은 corner case 다. 무게는 가림 없이도 나오는 결과에 둔다: 먼 horizon 감쇠, 등속 외삽보다 뒤처짐, 되먹임 멈춤. 이 결과가 닿는 use case 는 세 가지다.
- 벤치마크 점수 해석 (IntPhys, surprise)
- EK100 anticipation
- frozen predictor 를 guidance·novelty 에 쓰는 후속 연구 (HERA, Asleep at the Wheel, Off-Manifold)

planning 은 V-JEPA 2-AC 를 재기 전까지 use case 로 주장하지 않는다.

---

## 5. 방향에 대한 비판적 판단

1. **'encoder is enough' 가 가장 약한 고리다.** 리뷰어는 "enough for what?" 이라고 묻는다. decodability 는 필요조건일 뿐이다. 가장 싸고 강한 증거는 '같은 frozen encoder 위에서, 거의 학습하지 않은 예측기가 먼 horizon 에서 pretrained predictor 를 이긴다' 이다. 이것이 서면 predictor 병목과 new predictor 로 가는 다리가 한 실험으로 이어진다. 안 서면 주장을 '읽기는 된다' 로 낮춰야 한다.
2. **new predictor 로 가려면 범위 결정이 먼저다.** 현재 README 범위(새 학습 없음)와 CLAUDE.md 원칙(frozen 위에서 닫힘)을 유지하면, 처방은 frozen 개입이나 측정 도구 쪽이 된다. 학습을 허용하면 HERA 와의 차별화, 도메인 밖 일반화, 시그니처 소거 검증이 필수 과제가 된다.
3. **IntPhys 복사 결과는 양날이다.** 칸마다 결론이 뒤집히므로 헤드라인이 아니라 '점수의 프로토콜 민감도' 로 쓴다.
4. **리뷰어 풀을 전제한다.** Garrido·Bordes·Joseph (Meta) 계열이 리뷰어일 확률이 높다. 프로토콜 세부(max vs avg surprise, sliding 격자, 인과 예측)에서 한 곳만 어긋나도 신뢰를 잃는다.

## 6. 불확실·미검증

- CVPR 리뷰는 비공개다. ECCV 2026 가이드라인은 미확보다.
- XtX3 리뷰는 저자가 AI 생성이라고 지적한 리뷰다.
- Morpheus ICLR'26 판과 ICML'26 판이 같은 작업이라는 것은 요약문 기준 추정이다.
- HERA·Asleep at the Wheel·Observer Effect 는 arXiv 미심사 논문이다.
- Joseph et al. 가속 실험의 split 방식은 확인하지 못했다.
- 점수는 OpenReview API 에 기록된 값이다 (일부는 rebuttal 뒤 최종값일 수 있다).
- 웹 검색 한도가 소진돼 이후 문헌 확인은 arXiv 페이지 fetch 와 OpenReview API 로 했다.

## Sources
- [IntPhys 2 OpenReview](https://openreview.net/forum?id=Xpf5x3mLvn)
- [How Far ICLR'25](https://openreview.net/forum?id=ZyLkNVHBZF) · [How Far ICML'25](https://openreview.net/forum?id=DLlVjZQ7vD)
- [Interpreting Physics in Video World Models](https://openreview.net/forum?id=aijGVmEG9Y) · [arXiv 2602.07050](https://arxiv.org/abs/2602.07050)
- [Inferring Dynamic Physical Properties](https://openreview.net/forum?id=DCbQUijwtf)
- [LikePhys](https://openreview.net/forum?id=6UJf6B8RZ8)
- [Morpheus ICLR'26](https://openreview.net/forum?id=1E6pburMKc) · [Morpheus ICML'26](https://openreview.net/forum?id=f4xAzcMug6)
- [Object Binding](https://openreview.net/forum?id=5BS6gBb4yP)
- [Vafa NeurIPS'24](https://openreview.net/forum?id=aVK4JFpegy) · [Vafa ICML'25](https://openreview.net/forum?id=i9npQatSev)
- [Othello-GPT](https://openreview.net/forum?id=DeG07_TcZvT) · [Revisiting Othello](https://openreview.net/forum?id=1OkVexYLct)
- [VLM Interaction](https://openreview.net/forum?id=XdLgOm5giq)
- [Registers](https://openreview.net/forum?id=2dnO3LLiJ1) · [Mirage](https://openreview.net/forum?id=ITw9edRDlD)
- [PhysBench](https://openreview.net/forum?id=Q6a9W6kzv5) · [PhyWorldBench](https://openreview.net/forum?id=rlZeILv3fm) · [PhyGenBench](https://openreview.net/forum?id=6rMHcLWxl4)
- [V-JEPA ICLR'24](https://openreview.net/forum?id=WFYbBOEOtv)
- [Garrido et al. 2502.11831](https://arxiv.org/abs/2502.11831) · [V-JEPA 2 2506.09985](https://arxiv.org/abs/2506.09985) · [V-JEPA 2.1 2603.14482](https://arxiv.org/abs/2603.14482)
- [HERA 2608.05523](https://arxiv.org/abs/2608.05523) · [Asleep at the Wheel 2608.01336](https://arxiv.org/abs/2608.01336) · [Observer Effect 2602.12218](https://arxiv.org/abs/2602.12218)
- [Troubling Trends 1807.03341](https://arxiv.org/abs/1807.03341) · [Belinkov 2102.12452](https://arxiv.org/abs/2102.12452)
- [CVPR 2026 guidelines](https://cvpr.thecvf.com/Conferences/2026/ReviewerGuidelines) · [ICCV 2025 guidelines](https://iccv.thecvf.com/Conferences/2025/ReviewerGuidelines) · [ICLR 2026 guide](https://iclr.cc/Conferences/2026/ReviewerGuide) · [NeurIPS 2025 guidelines](https://neurips.cc/Conferences/2025/ReviewerGuidelines) · [NeurIPS 2026 E&D](https://neurips.cc/Conferences/2026/EvaluationsDatasetsReviewerGuidelines) · [NeurIPS 2021 consistency](https://blog.neurips.cc/2021/12/08/the-neurips-2021-consistency-experiment/)


### 논문 목록

| 제목 | 학회·연도 | URL | 요점 | 우리와의 관계 |
|---|---|---|---|---|
| IntPhys 2: Benchmarking Intuitive Physics Understanding In Complex Synthetic Environments | NeurIPS 2025 Datasets & Benchmarks 제출, Reject (리뷰 4명 모두 4 = borderline accept) · arXiv 2506.09849 | https://openreview.net/forum?id=Xpf5x3mLvn | 리뷰 4명 모두 borderline accept 였는데 PC 가 SAC 순위로 탈락시켰다. 리뷰어 지적: '16 frame 문맥 부족이 병목일 수 있다' (PW3t), '원인 분석 없이 성능 비교에 그친다. 개선 방향도 없다' (q2be), '기억 문제인가, 표현·지각 문제인가? 중간 과정을 probing 하라' (9GPH), '직접 학습한 classifier 기준선으로 과제 난이도와 일반화를 분리하라' (vk7Z). | 우리 motivation 의 출발점이다. 리뷰어들이 요구했지만 논문이 답하지 못한 것(병목이 memory·표현·predictor 중 어디인가)이 우리 논문의 질문과 정확히 겹친다. 그만큼 '가치 있는 질문' 이라는 근거가 된다. 반대로 'context 길이 교란' 은 우리에게도 그대로 들어온다. |
| How Far is Video Generation from World Model: A Physical Law Perspective | ICLR 2025 Reject (3/6/5/8) → ICML 2025 poster (3/3/4/4) | https://openreview.net/forum?id=ZyLkNVHBZF | ICLR 에서는 '이미 알려진 ID/OOD 현상의 반복' (r3mF), '결론이 너무 강하다' (9tpf), '이 모델을 world model 로 믿는다는 게 정말 널리 퍼졌나' (동기 허수아비), 'VAE 가 픽셀 차이를 지운다 — VAE 재구성부터 검증하라' (ym7u) 로 떨어졌다. ICML 에서는 'VAE 가 병목이 아님을 검증해 오류가 diffusion 학습에서 온다' 는 점이 칭찬받았다 (https://openreview.net/forum?id=DLlVjZQ7vD&noteId=VKl2zK39ow). '뻔하다' 는 평은 그대로 남았다. | 'encoder 는 병목이 아니다' 를 먼저 보이고 오류를 뒷단(predictor)에 귀속하는 설계가 곧 우리 'encoder is enough' 구조이고, 리뷰어가 이것을 명시적으로 높이 샀다. 같은 논문이 문장을 다듬고 구성 요소 검증을 보강한 뒤 한 번 떨어졌다가 붙었다. |
| Interpreting Physics in Video World Models | ICML 2026 Accept (최종 4/5/4/4) · arXiv 2602.07050 | https://openreview.net/forum?id=aijGVmEG9Y | V-JEPA 2 / VideoMAE-v2 encoder 를 층별 probing·subspace·attention ablation 으로 분석해 1/3 깊이의 'Physics Emergence Zone' 을 찾았다. 공격: 모델 다양성 부족, '대체로 descriptive 하고 실용적 함의가 없다' (fK2C), '선형 probe 한계', '읽어낼 수 있는지만 보고 변수를 조작적으로 정의하지 않았다' (iWcy). AC 는 '주장을 실험 범위로 좁히고 PEZ 가 physics 이해가 아니라는 한계를 쓰라' 고 요구했다. | 가장 가까운 경쟁·지지 논문이다 (Meta, Garrido 공저). encoder 쪽에 물리 변수가 있다는 우리 'encoder is enough' 를 받쳐 준다. 그런데 이 논문은 '가속도가 초기 층부터 decodable' 이라고 하고, 우리 H1 은 처음 보는 궤적에서 가속 정보가 약하다고 본다. 이 충돌을 풀어야 한다. 저자들이 우리 리뷰어일 가능성이 높다. |
| Inferring Dynamic Physical Properties from Video Foundation Models | ICLR 2026 Withdrawn (4/6/4/2) | https://openreview.net/forum?id=DCbQUijwtf | 'top-tier 라면 해결책을 제시해야 한다. 훌륭한 진단 연구일 뿐 새 진전이 아니다' (XtX3 — 저자들이 AI 생성 리뷰라고 지적함). 'V-JEPA-2 하나로 모델 클래스 전체를 말한다', 'kinematics 와 dynamics(힘)를 구분하라' (fJH5), 'learnable query readout 이 shortcut 일 수 있다' (rtwz). | V-JEPA 2 를 frozen readout 으로 분석한 논문이 '해결책 없음 + 단일 모델' 로 철회된 직접 선례다. 우리가 '전이·가속' 이라는 말을 쓸 때 kinematics 로 한정해야 한다는 교훈도 준다. |
| LikePhys: Evaluating Intuitive Physics Understanding in Video Diffusion Models via Likelihood Preference | ICLR 2026 Poster (6/4/4/…) | https://openreview.net/forum?id=6UJf6B8RZ8 | 공격: '합성 Blender 만', 'likelihood 가 물리 타당성과 분포 친숙도를 섞는다' (y5Dy). 성공한 반박: 짝 안에서 외형을 맞춰 차이가 물리 위반 하나뿐이라는 설계, 그리고 '통제된 쉬운 사례에서 실패하면 실제 장면에서도 어렵다' 는 하한 논리. | 우리 surprise 채점도 '친숙도 교란' 공격을 똑같이 받는다 (H6 복사 기준선이 그 증거). 합성 전용 방어는 matched pair(비트 동일 context) + 하한 논리로 쓴다. |
| Benchmarking Physical Reasoning of Video Generative Models with Real Physical Experiments (Morpheus) → Evaluating Newtonian Mechanics in Video Generative Models with Real Physical Systems | ICLR 2026 Reject (6/6/4/2) → ICML 2026 Accept (같은 Morpheus 작업으로 추정) | https://openreview.net/forum?id=f4xAzcMug6 | ICLR 에서는 'SAM-2 추적 오차가 점수로 번지는데 민감도 분석이 없다' (https://openreview.net/forum?id=1E6pburMKc&noteId=zarqBE5G9D) 로 떨어졌다. ICML AC 는 rebuttal 에서 더한 인간 상관 (Spearman 0.708), 궤적 섭동 강건성 (<2.6%), 창 민감도 분석을 수락 근거로 꼽았다. | 측정 도구(자·readout)의 오차 예산이 채택을 가른 사례다. 우리 위치 자·물체다움 문턱이 헤드라인 3개를 만들었다가 철회된 이력이 정확히 이 공격 지점이다. |
| Does Object Binding Naturally Emerge in Large Pretrained Vision Transformers? | NeurIPS 2025 Spotlight | https://openreview.net/forum?id=5BS6gBb4yP | 리뷰어 전원이 같은 초기 우려를 냈다: 'Large Pretrained ViTs' 라는 제목인데 DINOv2 만 봤다. 한 리뷰어는 '제목과 framing 이 실험 설정과 맞지 않는다' 고 썼다. rebuttal 에서 CLIP·MAE·supervised ViT 를 더했고, supervised 는 binding 이 약하다는 대비까지 나오자 전원 상향해 spotlight 가 됐다. probe + ablation(기능적 관련성)이 강점으로 꼽혔다. | 단일 모델(V-JEPA 2 ViT-H) 공격의 표준 사례다. 다른 모델에서 같은 시그니처가 재현되고, 동시에 대조군에서 달라지는 것까지 보이면 공격이 강점으로 바뀐다. |
| Evaluating the World Model Implicit in a Generative Model | NeurIPS 2024 Spotlight (4/3/9/5) | https://openreview.net/forum?id=aVK4JFpegy | 기존 진단(next-token, state probe)이 world model 을 과대평가함을 새 지표로 보였다. 공격: '지표가 나쁜 이유가 world model 이 아니라 상태→행동 변환(readout) 때문일 수 있다 (accountability)', 문턱 의존성, 지표 간 불일치 (S5LR). 옹호: '지금까지 가장 설득력 있는 증거' (7KXV, 9점). | '기존 점수(IntPhys surprise)는 상태 진화를 과대평가한다' 는 우리 H6 의 논리 구조와 같다. 새 측정 도구로 서사를 뒤집는 패턴이 spotlight 를 받았다. 동시에 readout·문턱 공격을 그대로 받는다. |
| What Has a Foundation Model Found? Using Inductive Bias to Probe for World Models | ICML 2025 Poster (3/4/3) | https://openreview.net/forum?id=i9npQatSev | 궤도 데이터로 학습한 모델이 Newton 역학을 쓰지 않는다. 공격: '전문가가 정한 상태 표현과 안 맞는다고 world model 이 없는 것은 아니다' (Genp), '실험이 작아 강건성이 불명확' (86Gp). | 우리 위치 자(position)는 특정 상태 표현을 가정한다. 'predictor 가 다른 부호화를 쓸 뿐' 이라는 반론에 대비해 파라미터 없는 교차검증(r = /p−h_imp///p−h_pos/)을 앞세운다. |
| Emergent World Representations: Exploring a Sequence Model Trained on a Synthetic Task (Othello-GPT) | ICLR 2023 notable top 5% | https://openreview.net/forum?id=DeG07_TcZvT | 합성 과제라는 공격에 'Othello 는 world representation 연구를 위한 통제된 testbed' 라고 반박했다. probing 에 개입(intervention)을 더해 표현이 실제로 쓰임을 보였다. 지표 혼동 지적에는 대체 지표 두 개로 같은 결론을 보였다. | 'testbed' framing 과 probe+개입 조합이 합성 전용 분석 논문의 성공 공식이다. 우리 H8/H9 knockout·차단 실험이 개입 역할을 하지만 결과가 대부분 '되돌아오지 않음' 이라 해석이 조심스럽다. |
| Revisiting the Othello World Model Hypothesis | ICLR 2025 Reject (5/3/5) | https://openreview.net/forum?id=1OkVexYLct | 'world model 의 정의가 없다. 정의가 없으면 실험이 주장을 받치는지 판단할 수 없다' (ezgd), '새 관찰이 놀랍지 않다' (duNp). | 'state evolution' · 'observation-anchored retrieval' · 'world model' 을 조작적으로 정의하지 않으면 같은 공격을 받는다. 네 능력(읽기·지속·전이·영속)을 각각 측정량과 귀무로 정의해야 한다. |
| Can Vision Language Models Learn Intuitive Physics from Interaction? | ICLR 2026 Reject (2/6/2) | https://openreview.net/forum?id=XdLgOm5giq | '선형 probe 는 물리량을 읽는데 그 능력이 행동으로 이어지지 않는다' (Tj6t, 긍정적 요약). 공격: 단일 모델군, 좁은 과제, probe 에 image-only·개입 대조가 없다, 결론 문장이 과하다 (rGrg), 원인 분석이나 해결책이 없다 (aHQU). AC: '증거가 넓은 결론을 받치기에 부족하다'. | '읽히지만 쓰이지 않는다' 는 우리 주장(encoder 가 담은 것을 predictor 가 진화시키지 않는다)과 같은 모양이다. 부정적 결과는 존중받았지만 좁은 범위와 과한 결론 때문에 탈락했다. |
| Vision Transformers Need Registers | ICLR 2024 Oral, Outstanding Paper (8/8/8/8) | https://openreview.net/forum?id=2dnO3LLiJ1 | 분석(고노름 토큰) → 기제 가설(전역 정보 저장) → 최소 개입(register) → 다중 모델 검증. 칭찬 속에서도 '개입 뒤 진단된 행동이 사라졌다는 실험이 없다' (mTMB) 는 지적이 나왔다. | '분석 → new predictor' 를 자연스럽게 잇는 모범이다. new predictor 가 점수만 올리는 게 아니라 진단 시그니처(2걸음 뒤 멈춤, 시간 감쇠, 등속 외삽보다 뒤처짐)를 없애는지 보여야 한다. |
| Are Emergent Abilities of Large Language Models a Mirage? | NeurIPS 2023 Oral, Outstanding Paper (7/8/9/7) | https://openreview.net/forum?id=ITw9edRDlD | 평가 지표 인공물로 널리 퍼진 서사를 뒤집었다. 공격: '대안 가설과의 관계를 논하지 않았다' (ccWP), '증거의 한계와 다른 가능성을 충분히 인정하지 않았다' (Cfbn). | H6 (IntPhys 점수 ≈ 복사) 유형의 기여가 받아들여지는 조건: 대안 가설 표와 증거 한계를 결론과 같은 비중으로 쓴다. |
| PhysBench: Benchmarking and Enhancing Vision-Language Models for Physical World Understanding | ICLR 2025 Oral (8/8/8/8) | https://openreview.net/forum?id=Q6a9W6kzv5 | 벤치마크 + 결함 진단 + 간단한 개선법(PhysAgent). AC 는 '결함을 찾고 개선 가능성까지 보였다' 를 강점으로 꼽았다. 공격: test 가 train 과 비슷한 이미지에서 왔다 (누수, khH4). | 진단과 처방을 한 논문에 담으면 oral 급이 된다는 사례다. 누수 공격은 우리 궤적 누수(84 궤적 × 28 외형) 이력과 겹친다. |
| PhyWorldBench: A Comprehensive Evaluation of Physical Realism in Text-to-Video Models | ICLR 2026 Oral (초기 4/6/6/6) | https://openreview.net/forum?id=rlZeILv3fm | AC 가 꼽은 채택 근거는 포괄성, 폭넓은 모델 평가, 인간 판단과 상관을 보인 자동 평가기다. 초기 점수가 높지 않아도 AC 재량으로 oral 이 됐다. | 평가 방법론 기여가 인간·기준 검증과 다중 모델을 갖추면 oral 까지 간다는 사례다. 리뷰 점수의 잡음이 크다는 점도 보여 준다. |
| Towards World Simulator: Crafting Physical Commonsense-Based Benchmark for Video Generation (PhyGenBench) | ICLR 2025 Reject | https://openreview.net/forum?id=6rMHcLWxl4 | 공격: 규모(160 prompt), 평가기(GPT-4o 다단계) 신뢰성·재현성, 비교가 불공정하다. | 측정 도구 신뢰성 공격의 또 다른 사례다. 우리 자·문턱도 신뢰도 수치를 같이 내야 한다. |
| V-JEPA: Latent Video Prediction for Visual Representation Learning | ICLR 2024 Reject | https://openreview.net/forum?id=WFYbBOEOtv | AC: I-JEPA 를 video 로 옮기면서 '운동에서 오는 시간 상관처럼 video 고유의 문제를 다루지 않았다'. 리뷰어 zTBB 는 attentive probing 으로는 논문 간 비교가 어렵다고 지적했다. | 원 V-JEPA 도 '시간 진화를 다루는가' 로 비판받았다는 역사적 맥락이다 (discussion 한 줄 정도). attentive probe 프로토콜 의존성 공격이 오래된 것임도 보여 준다. |
| Intuitive physics understanding emerges from self-supervised pretraining on natural videos | arXiv 2502.11831 (2025-02, 동료심사 판 확인 못함) | https://arxiv.org/abs/2502.11831 | 원문: 'we can use the encoder and predictor networks, without any additional adaptation, to probe the model's understanding of the world'. 동시에 'V-JEPA is never trained using a causal prediction task' 라고 인정한다. | '왜 굳이 pretrained predictor 를 보나' 라는 허수아비 공격을 막는 1차 인용이다. 커뮤니티가 이 predictor 를 적응 없이 world model 처럼 쓴다는 직접 증거다. |
| V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning | arXiv 2506.09985 (2025) | https://arxiv.org/abs/2506.09985 | Table 20 (EK100): encoder 출력만으로 이미 경쟁력이 있고 predictor 를 더하면 작지만 일관된 향상이 있다. predictor 출력만으로는 encoder 보다 훨씬 낮다 — 'EK100 task mostly requires strong semantic understanding, as opposed to forecasting capabilities'. 사전학습 데이터에 SSv2 가 포함된다 (원문 확인). | 'encoder 가 대부분을 한다' 를 저자들이 직접 인정한 인용이다. 한편 SSv2 로 'encoder is enough' 를 보이면 사전학습 분포 안이라는 공격을 받는다. |
| V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning | arXiv 2603.14482 (2026-03) | https://arxiv.org/abs/2603.14482 | dense predictive loss(보이는 토큰도 손실에 기여), 중간 층 deep self-supervision 을 도입했다. | '왜 최신 V-JEPA 2.1 은 안 봤나' 가 거의 확실히 나온다. 결론이 V-JEPA 2 릴리즈에 특유한지, 계열 전반의 문제인지 가르는 첫 추가 모델 후보다. |
| HERA: Historical Evidence Routing Adapter for Physical Prediction in Latent World Models | arXiv 2608.05523 (2026-08, 미심사) | https://arxiv.org/abs/2608.05523 | frozen V-JEPA 2 predictor 에 기억 adapter(RRPM)로 과거 물체 증거를 라우팅한다. IntPhys2 Main 에서 52.57 → 54.35, 고정 카메라 continuity 46.15 → 57.69. | 'new predictor / frozen 위 보강' 방향의 직접 선행이다 ('has been done' 공격). 차별점: HERA 는 기억 보강만 하고 여러 걸음 진화(전이)는 재지 않는다. 우리는 병목 국소화와 진화 측정을 한다. |
| Asleep at the Wheel: JEPA's Limitations in Evaluating Novel Driving Data | arXiv 2608.01336 (2026-08, 미심사) | https://arxiv.org/abs/2608.01336 | V-JEPA predictor surprise 의 novelty 성공은 도메인 이동의 결과이고 단일 데이터셋 안에서는 chance 로 무너진다. 같은 frozen embedding 에 가벼운 probe 를 얹으면 AP 가 약 2배다. | '표현은 충분한데 예측 점수가 병목' 이라는 우리 논지의 독립 선례다. 동시에 '합성 데이터에서의 실패도 도메인 이동 아닌가' 라는 역공의 근거가 되기도 한다. |
| The Observer Effect in World Models: Invasive Adaptation Corrupts Latent Physics | arXiv 2602.12218 (2026-02, 미심사) | https://arxiv.org/abs/2602.12218 | fine-tuning·고용량 probe 같은 적응 기반 평가가 잠재 물리 구조를 무너뜨린다 (ρ≈0.05). 저용량 probe 는 구조를 회복한다 (ρ>0.90). | 우리 attentive probe head (49.3M, train_acc 1.0 포화)가 공격받을 근거다. 저용량·파라미터 없는 측정을 주 근거로 둔다. |
| Troubling Trends in Machine Learning Scholarship | ICML 2018 The Debates (arXiv 1807.03341) | https://arxiv.org/abs/1807.03341 | 설명과 추측을 구분하지 않음, 향상의 출처를 밝히지 않음, mathiness, 용어 오용 — 네 가지 경향. | 'world model', 'understands physics', 'retrieval' 같은 용어 오용과 원인 추측(목적함수)을 막는 체크리스트로 쓴다. |
| Probing Classifiers: Promises, Shortcomings, and Advances | Computational Linguistics 2022 (arXiv 2102.12452) | https://arxiv.org/abs/2102.12452 | probing 의 방법론적 한계(읽힘 ≠ 쓰임, probe 가 과제를 스스로 배움) 정리. | 'encoder is enough' 를 decodability 만으로 주장하면 이 문헌으로 공격받는다. 충분성·개입 증거가 필요하다. |
