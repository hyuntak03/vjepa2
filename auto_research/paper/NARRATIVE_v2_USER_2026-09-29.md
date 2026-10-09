# 사용자 서사 초안 v2 (2026-09-29) + 검토

> ⚠️ **대체됨 (2026-10-08).** 논문 스토리 정본은 [`PAPER_STORY_2026-10-08.md`](PAPER_STORY_2026-10-08.md) 다. 사용자 서사가 2026-10-08 판 (P1 intuitive physics · P2 world model gap · P3 testbed · 관측 학습 predictor · AC post) 으로 바뀌었다 — 원문은 정본 §0. 이 문서 §B 의 검토 (B-1 복사 beat · B-2 측정된 표현 · B-3 '목표만으로는 안 된다') 는 정본 §2-2 · §2-3 · §2-4 에 반영했다.

> 사용자가 준 7 문단 서사 (인지과학 → 목표 → 분석 → 새 predictor → 검증). 원문은 §A, 검토 · 수정 제안은 §B.
> 정본 [`DIRECTION_2026-09-25.md`](DIRECTION_2026-09-25.md) §0-b, 논리 [`RETHINK_PREDICTOR_ROLE_2026-09-29.md`](RETHINK_PREDICTOR_ROLE_2026-09-29.md).

## A. 원문 (사용자, 그대로)

중심 메시지: "영상에서 세계의 변화 규칙을 배우고, 그 지식으로 미관측 미래를 전개하는 predictor 를 학습한다." 인지과학은 관찰 → 기대 형성 → 경험에 따른 기대 수정.

1. 사람은 세계가 변화하는 모습을 관찰하면서 앞으로 일어날 일에 대한 지식을 쌓는다. 영아기부터 (Johnson et al. 2003; Kochukhova & Gredebäck 2007; Baillargeon & DeJong 2017). → 조작 없이 관찰 경험만으로 예측 능력이 발달한다.
2. 예측과 관찰을 연결하는 학습이 세계 지식 형성의 한 원리다 (predictive processing: Köster et al. 2020; Ellis et al. 2021; 계산 모델: Piloto et al. 2022, Baek et al. 2025). → action-free video prediction 을 world knowledge 학습 objective 로 삼는다.
3. 우리가 만들 world model 은 관찰한 상태를 물리적으로 타당한 미래로 전개해야 한다. Read (문맥에서 상태 구성) · state persistence · state transition. 가려져도 존재는 유지, 움직임은 가림 동안에도 진행 (Kaufman et al. 2005; Rosander & von Hofsten 2004). → 이 기능들이 world knowledge 사용의 확인 기준.
4. 그런데 현재는 표현 학습용으로 훈련한 pretrained predictor 로 이 능력을 평가한다 (V-JEPA 2 EK100; IntPhys 2). encoder 에서 물리 정보를 읽는 것과 predictor 가 그것으로 변화를 예측하는 것은 구분해야 한다 (Joseph et al. 2026). → 이미 쓰이는 모듈의 기능 검증에서 출발.
5. 분석: pretrained predictor 는 world knowledge 를 충분히 학습하지 못했다 — 후반부로 갈수록 객체 정보 소실, 운동 약화, 가림 · 충돌 · 중력 조건에서 부적절한 예측. → frozen encoder 의 정보를 물리적으로 타당한 state evolution 으로 연결하는 것이 문제.
6. frozen encoder 위에 새 predictor 를 만들고 action-free video prediction 으로 world knowledge 를 학습시킨다. [설계 → 확인된 문제를 해결하는 원리]
7. 검증: context 활용 · horizon · 가림 · 상호작용을 통제해 persistence · transition 평가; 같은 frozen encoder 위에서 pretrained vs 새 predictor; IntPhys · IntPhys 2 · GRASP · InfLevel-lab · EK100.

## B. 검토 (2026-09-29)

**판정: 방향은 맞고 흐름도 선다. 그대로 두면 리뷰어가 치는 곳이 넷이고, 넷 다 우리 데이터로 고칠 수 있다.**

### B-1. 빠진 beat 하나 — "벤치마크가 그 능력을 안 잰다" 를 명시적 문단으로

지금 초안은 4→5 로 바로 간다. 그러면 첫 반론이 "inpainting 으로 학습한 predictor 가 예측 못 하는 건 당연" 이다. 우리가 가진 가장 새롭고 센 근거는 **그 predictor 를 인증해 온 점수가 마지막 latent 복사로 맞춰진다** 는 것이다 (PREDICTOR_SHARE: 복사 대비 +2 ~ +7 pt 전 벤치마크; V-JEPA 2 자체 Table 20 도 predictor 몫 +0.6). 4 와 5 사이에 한 문단:
> "이 predictor 가 world model 로 읽혀 온 근거 (VoE · anticipation) 는 predictor 의 예측을 요구하지 않는다 — 복사가 같은 점수를 낸다."
이 문단이 있어야 5 의 분석이 "당연한 것의 확인" 이 아니라 "숨어 있던 공백의 발견" 이 된다.

### B-2. 5 의 표현을 측정된 것으로 좁힌다

| 초안 표현 | 문제 | 바꿀 표현 (근거) |
|---|---|---|
| "후반부로 갈수록 객체 정보가 소실" | 시간이 아니라 **마지막 관측에서의 거리** 가 주 변수 (실영상은 둘 다) — VERIFY 1·2 차 정정 | "마지막으로 본 자리에서 2–5 칸 넘게 움직인 내용은 제자리에 놓지 못한다 (reach)" |
| "가림 · 충돌 · 중력 조건에서 부적절한 예측" | 충돌 · 중력 (ledge · wall) 은 2026-09-19 결정으로 **정의에서 뺀 common-sense 사건** 이고 자 해석에 단서가 많다 (§5-5) | 가림만 본문: "가려진 내용은 이어지지 않는다 (permanence 1/3)". ledge · wall 은 그림 예시 한 줄 |
| "world knowledge 를 충분히 학습하지 못했다" | 정의 없는 말. 리뷰어가 "어떻게 재나" 로 친다 | "관측 밖 미래를 전개하는 세 능력이 없다: reach · permanence · closure (자기 출력을 입력으로 못 씀)" + **정보는 encoder 에 있다** (probe · 속도 · 출처) 를 같이 |
| (없음) | **재학습 대조가 없다** | "미래 전용 · 자기회귀 목표로 실영상에서 다시 학습해도 같다 (Ariel full · prefix · AR ep38)" — 이게 없으면 6 의 objective 가 곧 해법으로 읽힌다 |

### B-3. 6 의 논리 — "objective 만으로는 안 된다" 를 인정하고 설계 원칙으로

우리 데이터에서 action-free video prediction 목표 **자체** 는 답이 아니다 (Ariel 세 팔: reach 도메인 안에서만, EK100 불변, permanence 그대로, AR 은 첫 되먹임부터 붕괴). 그러니 6 은 "objective 로 학습한다" 가 아니라 "objective + **측정된 실패에서 도출한 설계**" 로 써야 하고, 이게 Garrido 2026 (frozen V-JEPA 2 + 새 predictor, 이미 있음) 과의 델타다. 원칙 (RETHINK §4): 무엇/어디 분리 · 상태/렌더 분리 · 역사는 predictor 가 · (1 인칭 카메라 운동은 미해결).

### B-4. 7 의 검증 — 복사 기준선과 "예측이 필요한 칸"

- 모든 downstream 점수에 **복사 대비 Δ** 병기. 없으면 새 predictor 의 +2 도 잡음으로 읽힌다.
- VoE (IntPhys · GRASP · InfLevel) 는 predictor 를 안 요구하므로 (GEOPHYS 류 반론) **"떨어지지 않는다"** 용도. **"오른다"** 는 예측이 실제로 필요한 칸에서: 실영상 reach · permanence · closure (1 차 지표), EK100 2–3 s, IntPhys 2 가림 (2 차).
- IntPhys 2 는 세 predictor 모두 우연 수준 (AR ep38 50.8) — "오른다" 를 약속하기 전에 무엇이 필요한지 (재등장 순간 예측) 를 먼저 적는다.

### B-5. 인지과학 문단 (1–3) 은 좋다 — 길이만

- 1–2 는 intro 첫 문단 하나로 압축 (CVPR 독자). 3 의 Read · persistence · transition 은 우리 네 능력 (읽기 · 지속 · 전이 · 영속) 과 같다 — 영속 (가려진 뒤에도 존재) 을 persistence 안에 넣었으면 그렇게 명시.
- LeCun 2022 의 "(1) estimate missing information (2) predict plausible future states" 가 permanence · reach 와 정확히 대응 — 3 에 인용하면 인지과학과 ML 이 한 문장으로 붙는다.
- PLATO (Piloto 2022) 는 object-centric slot 이라 "과거 객체 표현 통합" 인용은 맞지만, 우리는 slot 없이 scene-level 이라는 점을 6 에서 구분.

### B-6. 문장 금지 목록 (기존 규칙 그대로)

"predictor 가 X 를 못 만든다" (정보는 있다) · "가림이 표현을 망가뜨린다" · "목적함수 때문이다" (discussion 한 문단) · "synthetic 에서 predictor 가 문제다" · "world knowledge 가 없다" (정의 없이).

### B-7. 수정 반영한 뼈대 (7 문단 → 8 문단)

1. 관찰만으로 예측이 발달한다 (영아). 2. 예측-관찰 학습이 그 원리 → action-free video prediction. 3. world model 의 요건 = 읽기 · 지속/영속 · 전이 (LeCun 2022 대응). 4. 현재는 표현 학습용 predictor 를 그 자리에 쓴다 (V-JEPA 2 EK100, IntPhys 1/2). **4′. 그 자리를 인증한 점수는 복사로 맞춰진다 (+2 ~ +7; Table 20 +0.6).** 5. 분석: encoder 는 정보를 담고 (충분성 표), predictor 는 가져와 채운다 (reach · permanence · closure); 미래 예측 · 자기회귀로 다시 학습해도 같다. 6. 새 predictor: frozen encoder + action-free prediction + **실패에서 도출한 설계** (Garrido 2026 과의 델타). 7. 검증: 복사 Δ 병기; VoE 는 유지, 예측 필요 칸에서 향상. 8. 한계: 1 인칭 카메라 운동, IntPhys 2 우연 수준, 판독의 한계.
