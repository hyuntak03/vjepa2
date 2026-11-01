# Object Permanence in V-JEPA2 — 실험 계획

Oct 31, 2026 · @이종서

## 목표와 핵심 주장

관찰 가능한 영상만으로 V-JEPA2 predictor에 "가려져도 존재한다"는 성질을 가르치고, 그것이 운동 타입·가림 타입·도메인에 무관하게 일반화됨을 보인다.

- **있었으면 있게, 없었으면 없게**: 재등장과 부재를 둘 다 맞혀야 한다.
- **관찰만으로**: 학습에 impossible 영상, 정답 라벨, 보조 head를 쓰지 않는다. impossible 영상은 평가 전용.
- **shortcut 없이**: 특정 가림 패턴이 아니라 성질 자체를 배웠음을 실험으로 증명한다.

**한 줄 스토리**: V-JEPA2는 장면을 지각하지만(perception) 관찰 공백 너머로 물체를 유지하지 못한다(memory/dynamics). 관찰 데이터에서 공백을 넘는 예측을 강조하면, 본 적 없는 가림 타입과 도메인에서도 permanence가 생긴다. 그리고 그건 prior가 아니라 기억이다.

**Framing**: object permanence = temporal amodal completion. 부분 가림을 공간적으로 채우는 amodal perception을 완전 가림·시간축으로 확장한 문제로 정의한다.

## 능력 분해

permanence는 여섯 능력으로 나뉘고, 1·3은 perception(encoder), 4\~6은 memory/dynamics(predictor)다. 기존 분석(병목은 predictor)이 맞다면 V-JEPA2는 1·3은 이미 하고 4\~6을 못 한다.

| # | 능력 | 담당 | 가르치나? | 측정 |
| --- | --- | --- | --- | --- |
| 1 | 장면에 무엇이 있는지 (물체 개체화) | encoder | 아니오 (동결) | 물체 수·존재 linear probe |
| 2 | 보이는 동안 같은 물체로 추적 (identity) | encoder + predictor | 간접 | 프레임 간 물체 토큰 매칭 |
| 3 | 앞뒤 관계: 사라진 이유가 가림인지 | encoder | 아니오 (동결) | depth order probe |
| 4 | 안 보이는 동안 위치·속도 유지 + 외삽 | predictor | **예** | 가림 중 위치 probe, 재등장 시간·위치 tuning |
| 5 | 재등장 시 같은 물체 (모양·색) | predictor | **예** | 정체 위반 surprise |
| 6 | 안 들어간 건 안 나옴 (부재) | predictor | **예** | appear 위반 surprise |

제외 범위: 가림막이 막혀 물체가 갇히는 경우(통과 불가 구조)는 이번 연구에서 다루지 않는다.

## 학습 설계

encoder는 동결하고 pretrained predictor만 V-JEPA 기본 future prediction으로 post-training한다. method의 핵심은 loss가 아니라 데이터와 학습 신호의 강조다.

**고정 사항**

- Objective: context 프레임 → 미래 프레임의 target encoder embedding 예측 (L1). 보조 head·mask·impossible 없음.
- Encoder(context/target): 동결. Predictor: pretrained에서 시작, 낮은 LR.
- 클립 길이·해상도·context/예측 구간: IntPhys2 평가 세팅과 동일.

**데이터 구성**

| 소스 | 역할 | 비고 |
| --- | --- | --- |
| WROP OP-1·2·3 (공개 데이터) | 구조화된 가림 시나리오 | 90개 task, 재등장이 예측 구간에 위치 |
| WROP 빈 클립 (개조 생성) | 부재(능력 6) | 같은 장면에서 target만 렌더 제외, 존재:부재 = 50:50 |
| SSv2 + 가림 합성 | real 도메인 | TAOS 방식 붙여넣기, 경계 blur |
| 원본 SSv2 | 일반 능력 유지 | 학습 배치의 일부로 혼합 |

**변형 (메인 결과 이후 추가)**

1. **Memory-critical token weighting**: 동결 encoder feature로 토큰마다 s = max\_sim(먼 과거) − max\_sim(최근)을 계산해, 봤다가 사라졌다 다시 나온 토큰의 loss 가중치를 높인다. 데이터 생성 없이 real video 전체에 적용 가능.
2. **Peekaboo masking**: SAM2로 추적한 물체 토큰을 일정 시간 context에서 제거하고, 같은 비율로 빈 배경도 제거한다. 픽셀 합성 artifact 없음.
3. **Imagine-then-predict (supervised upper bound)**: amodal mask를 predictor 경로 안에 넣는 버전. 관찰 원칙 밖이므로 상한선 비교용.

## 평가 ① 능력별 분리 측정

학습 전후로 각 능력을 따로 재서, 무엇이 바뀌었는지를 정확히 보인다. 기대 결과는 1·3은 학습 전부터 높고 변화 없음, 4\~6만 학습 후 상승이다.

| 측정 | 대상 | 방법 | 기대 (학습 전 → 후) |
| --- | --- | --- | --- |
| 존재·개수 probe | 능력 1 | 동결 encoder feature에 linear probe | 높음 → 높음 |
| 앞뒤 순서 probe | 능력 3 | 겹친 두 물체 중 앞 물체 분류 probe | 높음 → 높음 |
| 가림 중 위치 probe | 능력 4 (기억) | 가림 구간 predictor hidden에서 물체 2D 위치 회귀 | 낮음 → 높음 |
| 재등장 surprise | 능력 4·5·6 (사용) | possible vs impossible surprise 차이 | chance → 유의 |

가림 중 위치 probe와 재등장 surprise를 같이 보는 이유: 이전 regression head 실험에서 정보가 decodable해도 예측에 쓰이지 않을 수 있음을 확인했다. 두 지표가 함께 올라야 "기억하고, 그 기억을 쓴다"가 성립한다.

surprise는 세 가지로 보고한다: IntPhys2 표준 프로토콜, 재등장 시점 정렬, 물체 영역 토큰만으로 계산한 값.

## 평가 ② 재등장 tuning curve

possible/impossible 이진 판정은 "뭔가 나오면 정상"이라는 shortcut으로도 맞힐 수 있다. 그래서 재등장을 축별로 연속적으로 틀어, 정답에서 멀어질수록 surprise가 단조 증가하는지 본다.

| 축 | 변형 | 검증하는 능력 | 기대 |
| --- | --- | --- | --- |
| 시간 | 재등장을 Δt만큼 일찍/늦게 (예: −12 \~ +12 프레임) | 4 (속도 유지) | Δt=0에서 최소, \|Δt\|에 비례 증가 |
| 위치 | 재등장 지점을 Δx만큼 이동 | 4 (궤적 외삽) | Δx=0에서 최소 |
| 정체 | 색·모양을 단계적으로 변경 | 5 (identity) | 변화량에 비례 증가 |
| 부재 | 들어간 물체 없이 무언가 등장 | 6 (부재) | 높은 surprise |
| 소멸 | 들어간 물체가 안 나옴 | 4 | 높은 surprise |

- 생성: WROP generator를 개조해 각 축을 파라미터로 정확히 조절한다 (keyframe 애니메이션이라 시간·위치 이동이 쉽다).
- 보고: 축별 곡선을 학습 전/후/baseline으로 겹쳐 그린다. 학습 후 곡선이 정답 지점에서 V자로 깊어지는 것이 핵심 그림.
- 통제: 같은 장면에서 변형만 다르므로 quadruplet 단위 paired 비교가 가능하다.

## 평가 ③ Context ablation (인과 증거)

가림 이전 프레임에서 물체를 지웠을 때 재등장 예측이 사라져야, 모델이 prior가 아니라 기억을 쓴다고 말할 수 있다.

1. 같은 클립을 두 버전으로 만든다: (A) 원본, (B) 가림 이전 프레임에서 target을 렌더 제외 (처음부터 없던 장면).
2. 두 버전의 미래 예측 feature를 비교한다. 학습된 모델이라면 (A)에서는 재등장 위치에 물체 feature, (B)에서는 배경 feature를 예측해야 한다.
3. 지표: 재등장 위치 토큰에서 예측 feature와 "물체 있음" target / "물체 없음" target 간 거리 차이.

해석:

- (A)와 (B)의 예측이 다름 → 과거 관찰을 쓰는 기억.
- (A)와 (B)의 예측이 같음 → "가림막 뒤엔 뭔가 나온다"는 prior. shortcut 실패.

추가로 가림 이전 관찰 길이(물체가 보인 프레임 수)를 줄여가며, 기억이 형성되는 데 필요한 최소 관찰량을 본다.

## 평가 ④ 일반화

"운동 타입 agnostic"은 학습에서 뺀 타입으로 테스트해야 주장할 수 있다.

**Leave-one-out** (각각 하나씩 학습에서 제외 → 그 타입으로 테스트)

| 제외 축 | 예시 | 비고 |
| --- | --- | --- |
| 운동 타입 | 정지 물체+이동 occluder / 이동 물체+정지 occluder / 컨테이너 이동 | WROP OP-1·2·3 계열 단위 |
| 가림막 타입 | 터널, 스크린, 커튼, 컵 등 | WROP task 단위 |
| 가림 방식 | 화면 밖으로 나갔다 돌아옴 (카메라 이동) | 시각적 가림막 없는 가림, 가장 강한 증거 |

**도메인 사다리** (학습에 안 쓴 평가셋, 아래로 갈수록 학습 분포에서 멀다)

1. WROP held-out task (같은 엔진, 본 적 없는 시나리오)
2. IntPhys2 permanence 조건 (다른 엔진)
3. 자체 driving VoE 벤치마크 (실사풍, 도로 장면)
4. real video 평가셋 (준비된 다양한 가림막·disappear·appear 세트)

학습 데이터에서 driving 장면은 제외해 leakage를 막는다.

## Baseline·통제·부작용 체크

모든 비교는 같은 init, 같은 step 수, 같은 영상 풀에서 한다.

| 모델 | 학습 | 분리하는 효과 |
| --- | --- | --- |
| A | 없음 (pretrained V-JEPA2) | 출발점 |
| B | 원본 SSv2로 post-train (가림 없음) | 추가 학습 자체 |
| C | 가림 데이터로 post-train (존재:부재 균형) | 가림 경험 (메인) |
| C − 부재 | C에서 빈 클립 제거 | 부재 균형의 효과 (shortcut 증명) |
| C + weighting | C + memory-critical token weighting | 학습 신호 강조 |
| Upper bound | imagine-then-predict (amodal mask 사용) | 지도 신호의 상한 |

**부작용 체크**

- 일반 능력 유지: 표준 V-JEPA 평가(예: SSv2 action recognition attentive probe)가 A 대비 떨어지지 않는지.
- 일반 예측 품질: 원본 클립 held-out에서 JEPA loss 비교.

**통계**

- seed 3개 이상, 95% 신뢰구간 보고.
- 같은 장면의 possible/impossible을 quadruplet 단위로 paired 비교 (pairwise normalization).
- 여러 평가셋·축을 비교하므로 다중 비교 보정 명시.

## 데이터 준비

공개 WROP는 그대로 다운로드해 쓰고, 없는 것만 generator를 개조해 생성한다. 렌더 환경은 Blender 4.4.x + GPU(EEVEE)가 필요하다 (GPU 없으면 Cycles로 자동 전환되어 느림).

**공개 WROP에서 확인한 것** ([GitHub](https://github.com/hokindeng/object-permanence), [paper](https://arxiv.org/abs/2609.28654))

- 샘플: 60+60프레임, 1280×720, 24fps, 사건 시작 시점에서 split. 카메라는 클립 안에서 고정.
- 라벨: trajectory.npz에 물체별 프레임별 3D 위치·회전, 역할·모양·색, is\_target 플래그. mask·depth·visibility·카메라 intrinsic 없음.
- 물체마다 역할 태그(pb\_role)가 있어 target과 occluder를 코드로 구분 가능.
- 부재 클립이 거의 없음 (이름상 two\_carts\_reveal\_empty, shell\_game\_fakeout\_reveal 정도).
- 라이선스 CC BY-NC 4.0 (연구용 개조·사용 가능).

**개조 목록** (우선순위순)

| # | 개조 | 용도 | 방법 |
| --- | --- | --- | --- |
| 1 | 빈 클립 | 학습 (능력 6) | target을 전 프레임 렌더 제외 |
| 2 | 평가 변형 | 평가 ②③ | split 이후/이전 target 숨김, 재등장 시간·위치·색 파라미터화, 가림 이전 target 제거 |
| 3 | 120프레임 보존 + 카메라 행렬 export | 윈도우 선택, 2D 투영 | split 후 원본 삭제 로직 수정 |
| 4 | object index pass (visible / amodal mask) | upper bound, 위치 probe 라벨 | occluder 숨김 렌더 1회 추가 (비용 약 2배) |

**real 쪽**: SSv2에 TAOS 방식 가림 합성 (SAM2 추적, 존재:부재 균형). memory-critical weighting은 추가 생성 없이 SSv2·K400 전체에 적용.

## 예상 리뷰어 반론과 대응

| 반론 | 대응 실험 |
| --- | --- |
| "가림막 뒤엔 뭔가 나온다"는 shortcut 아니냐 | 평가 ③ context ablation, 부재(appear) 위반, C − 부재 ablation |
| 그냥 더 학습해서 좋아진 것 아니냐 | baseline B (같은 영상, 가림 없이 post-train) |
| 학습한 가림 타입에만 맞춘 것 아니냐 | 평가 ④ leave-one-out, 특히 화면 밖 가림 |
| 합성 데이터라 real에선 안 되지 않냐 | 도메인 사다리 마지막 단계 (real 평가셋) |
| 이진 판정은 운으로도 맞힌다 | 평가 ② tuning curve (단조 증가) |
| 모델이 망가진 것 아니냐 | 표준 V-JEPA 평가 유지 확인 |
| WROP와 뭐가 다르냐 | 생성 모델·사람 Elo 평가 vs latent 예측·VoE 정량 평가·관찰만으로·real 일반화 |
| decodable하다고 쓰는 건 아니다 | 평가 ①에서 위치 probe와 재등장 surprise를 함께 보고 |

## 진행 순서

첫 실험은 공개 WROP만으로 A·B·C를 돌려 방향을 정하고, 생성은 부족한 게 확인된 뒤에 한다.

- [ ] WROP OP task 다운로드, V-JEPA2 입력 형식(길이·해상도)으로 변환
- [ ] 평가 파이프라인 고정: 기존 real 평가셋 + IntPhys2, 재등장 정렬 surprise, 물체 토큰 surprise
- [ ] 능력 1·3 probe로 "perception은 이미 된다" 확인 (학습 불필요, 빠름)
- [ ] A·B·C 학습 및 평가 (C는 OP 계열 하나를 leave-one-out으로 빼두고 시작)
- [ ] Blender 4.4 + GPU 환경 구축, 빈 클립·평가 변형 generator 개조
- [ ] 평가 ② tuning curve, 평가 ③ context ablation
- [ ] memory-critical token 시각화 → 재등장 물체에 몰리는지 확인 후 weighting 실험
- [ ] real 쪽 (SSv2 가림 합성), upper bound

**C의 결과별 다음 행동**

| 결과 | 해석 | 다음 행동 |
| --- | --- | --- |
| C > B, held-out 타입에서도 우세 | 데이터 설계가 곧 method | 일반화·인과 실험으로 확장 |
| C > B, 학습한 타입에서만 우세 | 패턴 암기 | 다양성 확대, memory-critical weighting |
| C > B인데 appear 쪽 surprise 안 오름 | 부재 불균형 shortcut | 빈 클립 생성 후 재학습 |
| C ≈ B | 데이터만으로는 부족 | predictor 구조 변경 (memory token, imagine-then-predict) |
