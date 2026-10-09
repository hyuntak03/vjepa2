# 다시 생각한 것 — "predictor 의 자리가 비어 있다" (2026-09-29)

> ⚠️ **대체됨 (2026-10-08).** 논문 스토리 정본은 [`PAPER_STORY_2026-10-08.md`](PAPER_STORY_2026-10-08.md) 다. §3 논지 ("예측의 자리가 비어 있다") 는 새 정본의 P3 마지막 문장 (encoder 는 충분, predictor 가 실패) 으로 흡수됐다. **그대로 유효**: §0-c memory 금지, §1-b encoder 충분성 표 (→ 정본 §2-3 iv), §4 설계 원칙 (→ 방법 후보, 미검증), §5 · §5-b 복사 Δ · copy-hard (→ 정본 하네스 5), §6 리뷰 대응.

> 사용자 지시 (09-29): downstream (EK100 · intuitive physics) 은 사전학습 predictor 를 **기능상 예측기**로 쓰는데, 우리 분석은 그것이
> world knowledge 도 없고 예측도 못 한다는 것을 보였다. 기능상 world knowledge 는 predictor 에 있어야 맞다. 이 둘을 놓고 다시 생각하라.
> 정본은 [`DIRECTION_2026-09-25.md`](DIRECTION_2026-09-25.md) §0-b. 이 문서는 그 밑의 논리 전개다.

## 0-c. 프레임 정정 (2026-09-30 사용자: "memory 문제로 쓰고 싶지 않다")

**"기억 (memory · carried state) 문제" 로 쓰지 않는다.** 데이터가 그 프레임을 지지하지 않는다: (1) 잃어버린 정보가 없다 (encoder 에 속도 · 정체 · 출처가 그대로); (2) 기억이 필요 없는 상황 (가림 없는 v3, 문맥 전부 보임, 분포 안 t0–7) 에서도 가속 α < 0 — 보이는 속도 변화를 미래에 적용하지 않는다; reach 도 보이는 내용을 옮기는 문제다; (3) 가림 결과도 "가려진 것은 계속 존재 · 이동한다" 는 규칙을 적용하지 않은 것이지 잊은 것이 아니다 (encoder 창 안에 가려지기 전 프레임이 있어도 절반).
→ 세 실패 (reach · permanence · 가속) 는 하나다: **predictor 가 관측된 상태에 변화 규칙 (world knowledge) 을 적용하지 않고 마지막 관측을 옮겨 적는다.** 본문 용어: "규칙 적용" · "지속 · 전이 · 영속 능력"; permanence = "가림 아래에서의 지속 규칙", closure = "자기 예측에도 같은 규칙이 적용되는가". memory · state 는 쓰지 않는다. §4 의 "상태/렌더 분리" 는 구조 선택지로만 남기고 논지로 세우지 않는다 — 논지는 "예측 목표가 규칙을 요구하도록 만들면 predictor 가 규칙을 배운다" (DESIGN_CHOICES §3-d 의 C · D · H 류 손실 + 먼 지평 · copy-hard).

## 1. 지금 시스템에서 실제로 일어나는 일 (측정된 것)

| 역할 | 기대 | 실제 (우리 측정) |
|---|---|---|
| encoder | 관찰을 표현 | **기대대로.** 속도 (z→v R² 0.996) · 정체 (probe 98–100) · 출처 (미래 encoder 판독 ≈ 천장) 를 담는다 |
| predictor | 관찰 밖 미래를 만든다 | **아니다.** 마지막 관측에서 가져와 채운다: reach 2–5 칸, 가려진 내용 1/3, 자기 출력은 입력이 못 된다 |
| downstream 점수 | predictor 의 예측력 | **predictor 몫이 작다.** 복사 대비 IntPhys 1 +4 · IntPhys 2 ≈ 0 · EK100 +2 (predictor 를 빼도 −1~−3) |

정확히 쓰면: predictor 는 "예측을 못 한다" 가 아니라 **가까운 미래 (마지막 관측 근처 · 한 걸음) 는 맞히고 그 밖은 못 한다.** 벤치마크가 요구하는 것이 그 안쪽뿐이라 점수가 난다.
그래서 지금 시스템의 world knowledge 는 (있다면) **encoder 의 표현 안에** 있고, predictor 는 그것을 자리만 옮겨 적는다. 기능상 자리가 뒤바뀌어 있다.

### 1-b. "encoder 는 필요한 정보를 담고 있다" 의 근거 표 (사용자 09-29: 이게 encoder 쪽에서 보여야 할 전부)

| 정보 | 증거 | 강도 |
|---|---|---|
| 정체 (모양 · 색 · 환경) | p probe 98 / 99 / 100, z 이식 97 (v11) | 확정 |
| 운동 (문맥 끝 속도) | z → v ridge R² 0.996 / 0.989 (RollOutV2 09-19) | 확정 |
| 출처 (미래 내용이 마지막 관측의 어디에 있었나) | 진짜 미래 프레임의 encoder 판독 = 천장 (합의 토큰 정의, I-scene) | 확정 |
| 가려지기 전 내용 | hc[L−m] 템플릿으로 permanence 판독 성립 — 창 안이면 encoder 가 든다 | 확정 (창 안 한정) |
| frozen z 만으로 먼 전이가 가능한가 | z 위 새 predictor: reach 팬 4.8 → 7.0, SSv2 2.7 → 4.2; 출처 힌트 하나로 +.2 ~ .3; AR teacher-forced SSv2 ∞ | 확정 (팬 · SSv2 · 합성) |
| 〃 **EK100** | 어떤 학습도 2.2 를 안 넘고 AR teacher-forced 도 4.0. **09-29 판독**: encoder 토큰 평균 → ego-motion ridge R² .69 / .63 (10-fold · 영상 split 재현; flow 자기 일관성 clip-CV .53 / .36 보다 높음); EK100 미래 변위의 67–75 % 가 카메라 운동 ([`../Archive/ENCODER_MOTION_READOUT_2026-09-29.md`](../Archive/ENCODER_MOTION_READOUT_2026-09-29.md)) | **확정 (1 차)** — encoder 는 카메라 운동을 담는다; 병목은 그것을 미래 자리로 옮겨 쓰는 predictor |

→ "frozen encoder" 는 팬 · SSv2 · 합성 · EK100 (ego-motion) 네 근거로 선다. 남은 단서: 토큰별 입력에서 predictor 가 쓰기 좋은 형태인지 (선형 판독은 토큰 평균), EK100 물체 운동 (잔차 R² .04–.09) 은 encoder 한계인지 표적 잡음인지 미결.

## 2. 왜 그렇게 됐나 (해석 — 원인 검정 아님, discussion 한 문단)

- 사전학습 목표가 **같은 클립 안 가려진 튜블릿 채우기** 다. 그 손실의 최저점은 근처에서 비슷한 것을 가져오는 것이고, 멀리 옮기거나 없는 것을 이어 가는 것은 보상이 작다. Sobal 2022 의 "느린 것이 손실의 최저점" 이 encoder 에서 predictor 로 옮겨 온 형태.
- 미래 전용 목표로 다시 학습해도 (Ariel full · prefix) **도메인 안 reach 만 늘고** EK100 은 그대로, permanence 는 그대로다. 되먹임 목표 (AR) 는 한 걸음은 완벽한데 자기 출력을 되먹이면 첫 걸음부터 무너진다. 즉 **목표를 바꾸는 것만으로는 predictor 에 세계 지식이 들어오지 않는다.**

## 3. 논문의 논지 (다시 세운 것)

> **JEPA 계열 world model 에서 "예측" 의 자리가 비어 있다.** encoder 는 관찰을 잘 표현하지만, predictor 는 마지막 관측을 옮겨 적을 뿐이고
> 흔한 평가는 그것으로 충분하다. 우리는 (i) 그 자리가 비었음을 재는 자 (reach · permanence · closure, 실영상) 를 만들고,
> (ii) 미래 예측 목표만으로는 채워지지 않음을 보이고, (iii) 세계 지식을 predictor 에 넣는 설계 원칙을 실패 지점에서 도출해 새 predictor 를 학습하고,
> (iv) 예측이 실제로 필요한 downstream (EK100 장거리 · IntPhys 2 가림 · Ego4D) 에서 그 효과를 보인다.

주인공은 새 predictor 다. 분석은 "무엇을 채워야 하는가" 의 명세다.

## 4. 새 predictor 가 채워야 할 것 — 실패 지점에서 도출한 설계 원칙 (사용자 학습 항목)

| 측정된 실패 | 원칙 | 근거가 되는 개입 · 대조 |
|---|---|---|
| reach: 멀리 있는 내용을 못 옮긴다. 정보는 있고 자리만 틀린다 | **"무엇" 과 "어디" 를 분리**: 내용은 가져오되 자리는 운동으로 정한다 (변위 · 속도 예측 → 조회 위치) | 출처 방향 bias 하나로 먼 적중 +.2~.3 (상한), 등속 외삽 추정은 무효 → 자리 추정이 학습돼야 함 |
| closure: 자기 출력을 입력으로 못 먹는다 (AR 도) | **굴리는 상태와 렌더 출력을 분리**: 되먹이는 것은 상태, 출력은 encoder 공간으로의 렌더 | AR tf ∞ vs roll 2–3 칸; 채널 affine 어댑터로는 안 닫힘 |
| permanence: 마지막 관측에 없는 내용은 안 이어 간다. AR (블록별 인코딩) 은 더 나쁨 | **역사는 predictor 가 들고 있는다** (encoder 창이 아니라 상태에 축적) | k=2 AR 최저 (.59) — encoder 창의 역사 경로에 기대면 창 밖은 사라짐 |
| EK100: 어떤 학습도 reach 를 안 늘린다 (2.2 천장) | 1 인칭 카메라 운동을 상태가 다뤄야 한다 (ego-motion 이 장면 전체를 옮김) | tf 조차 4.0 — 한 걸음 전이도 짧다. **미해결, 측정만** |

⚠️ 원칙 넷은 분석에서 **나온** 것이지 검증된 것이 아니다. 각 원칙은 학습 한 팔로 검정된다 (DESIGN_CHOICES §3 에 판정 기준을 둔다).

## 5. 평가도 같이 바뀐다

- 모든 downstream 점수에 **복사 기준선 대비 Δ** 를 병기한다 (predictor 의 몫). 지금 값: IntPhys 1 +4 / +7 (AR 공간), IntPhys 2 ≈ 0, EK100 +2.
- 새 predictor 의 1 차 지표는 **실영상 reach · permanence · closure** 다 (합성은 보정용). 2 차가 downstream Δ.
- 흔한 평가에서 "떨어지지 않는다" 를 먼저 보이고, 예측이 필요한 칸 (EK100 2–3 s · IntPhys 2 가림 · Ego4D) 에서 "오른다" 를 보인다.

### 5-b. copy-hard 부분집합 프로토콜 (2026-09-29 사용자: "late 가림은 복사로 안 나온다 — 그게 오히려 좋다")

벤치마크마다 **복사 기준선이 우연 수준인 쌍 (칸)** 만 골라 그 위에서 predictor 점수와 복사 대비 Δ 를 보고한다. 전체 점수는 "떨어지지 않는다", copy-hard 는 "오른다" 용도.
- v11: late 가림 (문맥 경계에 가림) 3 운동 × 위반 — 복사 39–63, 릴리즈 50–70, post-FT · Ariel 도 물체→빈 25–36 (아무도 못 채운 칸 = 새 predictor 의 자리). 복사가 천장인 칸 (vanish visible/early/mid 99–100, shape 76–91) 은 점수가 predictor 를 안 잰다는 증거로 따로 보고.
- IntPhys 1: 쌍별 복사 결과에서 복사가 틀리는 쌍. IntPhys 2: 셋 다 우연이라 재등장 순간 쌍만. EK100: clip 별 예측 저장 규약이 필요.
- 산출 (1 차, 쌍 단위 재계산): `Archive/COPY_HARD_SUBSET_2026-09-29.md`. **복사가 틀리는 쌍에서 릴리즈는 우연이다** — v11 48.4 [46.6, 50.2] (n 2,877) · IntPhys 1 44.4 (n 27) · IntPhys 2 21.2 (n 240, 우연 아래). 복사가 맞는 쌍에서는 86 / 97 / 84. 릴리즈 점수 = 복사 + 나머지에서 동전 던지기. Ariel prefix · full · post-FT ip2 도 같다 (49–51) — 그들의 +3~4 는 전부 copy-correct 위에서 난다. 복사 천장 칸 (v11 20 칸) 은 모든 predictor 98–100 → predictor 차이가 사라진다.
- **예외 = AR** (ep18/38): v11 copy-hard 73.8, late 65.8 (릴리즈 · prefix 38 = 우연 아래), late·vanish→빈 72–99 (복사 10–68, 릴리즈 0–65). 도메인 안 post-FT v11_e10 은 82.5. ⚠️ AR 은 표적 공간이 블록별이라 표준 복사와 덜 상관될 뿐일 수 있다 (IntPhys 1 copy_blk-hard 에서는 AR 57–63 vs 릴리즈 77 로 대칭; v11 copy_blk 부분 표본 n 45 에서는 AR 82 유지). 판정 보류 — v11 copy_blk 전체가 필요. **검증 3 차 정정**: 무작위 겹침 귀무에서 copy-hard 정확도 = 전체 정확도이고 모든 predictor 가 그 아래 (release −27 · prefix −28 · AR −10 · v11_e10 −9) → AR 의 73.8 은 "복사가 못 푸는 걸 푼다" 가 아니라 **"오류가 복사의 오류와 덜 묶여 있다"** 로 쓴다.
- **모순 기록 (지우지 않는다)**: AR 은 장면 판독에서 가려진 내용을 절반만 복원 (permanence 비 .46, 다른 팔과 같음) 하고 reach 최단인데, v11 late 가림 채점에서는 혼자 높다. 두 측정의 정의가 다르다 — 판독은 **어디에** 놓았나 (Chebyshev ≤ 1), 채점은 전 토큰 L1 로 **무엇이 있나** 에 가깝다. 즉 "AR 은 돌아올 것을 알지만 (presence) 어디로 오는지는 모른다 (placement)" 가 두 결과를 잇는 가설이고, v11 late 클립 presence 검정 (09-29 밤, `AR_PREDICTOR` §2-6 결과): **presence-without-placement 확인 (1 차)** — AR 이 맞히는 움직이는 물체 쌍에서 margin 이 재등장 칸에서는 음수 (−.04), 나머지 칸에서 양수 (+.01), placement share 균등 수준. "돌아온다" 는 알고 "어디로" 는 모른다. 두 측정이 양립한다. 적대 검증: 재등장 칸 음수는 모든 팔 공통, AR 만의 것은 먼 칸의 전역 양수 오프셋 (복사 지도 + 오프셋, r .59); 문맥 편향 성분 있음; "presence" = 물체 있는 미래 쪽 전역 편향.



## 6. 리뷰어가 칠 곳과 답 (미리)

| 공격 | 답 |
|---|---|
| "inpainter 가 예측 못 하는 건 당연" | 벤치마크가 그것을 world model 로 부르고 있고 (감사표), 미래 전용 · 되먹임 목표로 다시 학습해도 같다 (Ariel 셋), 정보는 있다 (개입 상한) |
| "world knowledge 정의가 없다" | 본문에서는 쓰지 않는다. 네 능력 · reach · permanence · closure 로만 |
| "판독이 자기 것" | 합성 GT 보정 (≤ 1 칸 .87–.95) · encoder 천장 · 복사 기준선 · 파라미터 0 · 적대 검증 두 차례 |
| "frozen encoder 가 천장" | encoder 는 필요한 정보를 담고 있다 (§1). 천장이면 새 predictor 도 못 넘는다 — 그 자체가 결과 |
| "AR 이 IntPhys 1 최고인데?" | AR 공간 복사가 83.3, 여유 +7 vs 릴리즈 +4, 잡음 안. reach 는 최저 |

### 6-b. 문헌에서 온 공격 (2026-09-29, [`LITERATURE_PREDICTOR_ROLE_2026-09-29.md`](LITERATURE_PREDICTOR_ROLE_2026-09-29.md))

| 공격 | 출처 | 답 |
|---|---|---|
| "predictor 가 애초에 필요 없다 — frozen image encoder 기하만으로 IntPhys 2 93.3 %" | GEOPHYS 2606.20707 (◐ 프로토콜 미확인) | 그게 바로 우리 논지의 절반이다 (VoE 는 predictor 를 안 요구한다). 우리는 여기서 멈추지 않고 predictor 가 **해야 할** 일 (reach · permanence · closure) 을 재고 채운다. 수치 인용 전 본문 확인 |
| "frozen V-JEPA 2 + 새 predictor 는 FAIR 가 이미 했다" | Garrido et al. 2601.05230 (latent action WM) | 복사 귀무 · reach · permanence 없음; 그들 스스로 "표현 공간이 예측용으로 설계되지 않았다" 를 한계로 둠. 우리 델타 = 무엇을 채워야 하는지의 명세 + 설계 원칙 |
| "copy-last 기준선은 Human-JEPA 가 먼저" | 2608.21160 (자기 predictor · K700 · latent cosine) | 릴리즈 predictor 를 벤치마크 점수에서 복사와 잰 건 우리가 처음. 프로토콜이 달라 직접 비교 불가 |
| "미래 특징 회귀가 anticipation 에 안 든다는 건 2019 에 알려짐" | RULSTM 2005.02190 (EGTEA 60.2 → 50.2, 카메라 운동 탓) · AVT-b · ImagineRNN | 인용한다. 우리는 그 원인을 latent 수준에서 측정 (EK100 reach 천장 2.2 · ego-motion 판) 하고, V-JEPA 2 Table 20 (+0.6) 이 2025 에도 같음을 잇는다 |
| "VoE 가 입력의 물리를 요구하지 않는다는 건 저자들이 이미 씀" | Joseph et al. 2602.07050 App. C.1.4 (layer 0 위 predictor 도 VoE 점수) | 지지 근거로 인용 |
| 기능 분리 문장의 출처 | LeCun 2022 "(1) estimate missing information … (2) predict plausible future states" | 그대로 인용 — (1) = permanence, (2) = reach. ⚠️ Garrido 2025 에는 "predictor acts as a world model" 문장이 **없다** (귀속 금지) |

## 7. 분석 쪽에서 아직 해야 하는 것 (내 몫)

1. **AR ep38 재판독** (진행 중) → AR 절 확정.
2. **복사 대비 Δ 표** 하나로 정리 (IntPhys 1 · 2 · v11 · EK100 1/2/3 s) — "predictor 의 몫" 그림.
3. **"무엇/어디 분리" 원칙의 분석 근거 보강**: 출처 방향 bias 상한을 실영상 2–3 s EK100 · v11 가림에 걸어 "자리를 알면 얼마나 오르나" 를 downstream 점수로 (분석→방법 다리).
4. 스토리 그림에 AR 패널 · 복사 Δ 패널.
5. 초록 · README 를 §3 논지로 다시 쓴다.

## 재현
이 문서는 논리 문서다. 수치 출처: `Archive/I_SCENE_CAUSAL_BOTTLENECK_2026-09-25.md`, `Archive/AR_PREDICTOR_2026-09-27.md`, `Archive/INTPHYS2_PERMANENCE_AUDIT_2026-09-26.md`, `Archive/SC3_*`, `Archive/SC5_*`, `z_training/ARIEL_CHECKPOINTS.md` §5-2.
