# oral 급 방향 — 분석 측면 제안과 sanity check (2026-09-25, SC1–SC3 반영 개정)

> ⚠️ **대체됨 (2026-10-08).** 논문 스토리 정본은 [`PAPER_STORY_2026-10-08.md`](PAPER_STORY_2026-10-08.md) 다. DESIGN_CHOICES 로 대체된 뒤 다시 대체됐다 — 현재 스토리 정본은 아래 경로.

> ⚠️ **대체됨 (09-25 저녁).** 뼈대 정본은 [`DESIGN_CHOICES_2026-09-25.md`](DESIGN_CHOICES_2026-09-25.md) 다. 이 문서의 A′ 는 거기 §2 로 흡수됐고, "reach 법칙" 은 결론이 아니라 DC1 의 근거다.

> 사용자 요청 (2026-09-25): "oral 급이 되려면 어떻게 해야 되는지 방향성 및 일부는 sanity check. 분석적인 측면에서. 주제도 새로 제안해도 괜찮다.
> anticipation 엮으면서 real world 도 볼 거다. 큰 틀에서 synthetic 은 분석용 testbed 이자 state evolving 판단용이고, 하고 싶은 건 world video prediction 모델."
> 근거 문서: 리뷰 패널 [`REVIEW_v1_2026-09-25.md`](REVIEW_v1_2026-09-25.md) (현재 weak reject 3/6, 보완하면 poster, oral 아님) · 검증 2 차 [`../Archive/VERIFY_0925_R2_2026-09-25.md`](../Archive/VERIFY_0925_R2_2026-09-25.md).
> sanity check 문서: [SC1](../Archive/SC1_REACH_NOT_HORIZON_2026-09-25.md) · [SC2v2](../Archive/SC2V2_FLOW_REACH_SSV2_2026-09-25.md) · [SC3](../Archive/SC3_EK100_PREDICTOR_SUBSTITUTION_2026-09-25.md). 셋 다 **1 차 결과, 적대 검증 전**이다.

> ⚠️ **정정 (같은 날, SC1 결과)** — 첫 판의 안 A "Horizon, not perception — 학습 질의 지평까지만 외삽한다" 는 **철회한다.**
> - 사전 가설은 "학습된 predictor 는 자기 학습 지평 (8 슬롯) 에서 멈춘다" 였다.
>   - SC1 에서 슬롯 상수 모형은 절벽 분산의 **0 %** 를 설명한다.
>   - Ariel 은 stride 1 에서 13–17 슬롯까지 간다.
> - 대신 멈춤은 **마지막 관측에서의 이동 거리** 로 모인다. 거리만 모형이 R² 0.68–0.84 다.
> - 새 안은 A′ (§1) 이다. 첫 판 본문은 §7 에 기록으로 남긴다.

## 0. 결론 먼저

**지금 판이 oral 이 아닌 이유는 셋이다** (리뷰 패널).
1. 주인공 (릴리즈 predictor) 이 "미래 예측용으로 학습 안 한 모듈" 이라 "당연하다" 공격을 받는다.
2. 기전 주장이 판독에 따라 바뀌었다.
3. 해결책이 없다.

**제안: 주인공을 "특정 predictor 의 실패" 에서 "latent video predictor 의 도달 거리 (reach)" 로 바꾼다.**

> **"Latent video predictors carry a scene only a bounded distance from where they last saw it."**
> 관측이 끊기면 predictor 는 물체를 **마지막 관측 자리에서 일정 거리까지만** 옮긴다. ~~몇 걸음 (슬롯) · 몇 fps (stride) · 화면 어디서든 그 거리는 같다.~~ → 멈춤은 걸음 수가 아니라 거리 쪽으로 모이고 (고정 걸음 수 R² 0), 화면 위치와 무관하다. release 는 거리 + 약 1.6 걸음이 섞인 모양이고, 학습된 predictor 는 순수 거리에 가깝다.
>
> ⚠️ 정정 (같은 날 오후) — "걸음 수 · fps 와 무관한 고정 거리" 는 **release 에 대해 과장**이었다. release 는 한 stride 안에서 멈춘 자리가 속도에 비례하고 (빠름/느림 비 1.96–2.07 vs 속도비 1.85, `REVIEW_CVPR_ORAL` K4 확정), stride 사이에서는 plateau 가 2.3 → 3.2 → 4.9 칸으로 δ 에 따라 오른다 (약 2 칸 + 1.6 걸음). 순수 거리에 가까운 것은 학습된 predictor (Ariel plateau 5–6.7 칸) 다. 기각된 것은 "고정 걸음 수" 모형이다 (R² 0).
> 미래 예측으로 학습하면 그 거리가 늘지만 (합성 약 1.5–2 배, 실영상 약 1.35 배) 한계는 남는다.
> 흔한 평가 (VoE surprise, 1 s anticipation) 는 이 거리 **안쪽** 만 재서 차이를 보지 못한다.

- **사용자의 상위 메시지와 같은 문장이다.** "a true latent world model must evolve state even when it can't see" 를 측정 가능한 형태로 바꾸면 이렇다:
  - "관측 없이 운반할 수 있는 거리 (reach)" 를 잰다.
  - 진짜 world model 의 reach 는 관측과 무관해야 한다.
- **world video prediction 모델의 설계 목표** 는 "더 긴 학습 지평" 이 아니라 **reach 가 변위에 무관한 predictor** 다. SC1 에서 학습 지평은 원인이 아니었다.
- **analysis 는 그 목표를 측정 가능하게 만든다.**
  - 합성에서는 GT 를 쓴다.
  - 실영상에서는 flow 로 유사 정답을 만들어 reach 곡선을 잰다. 이 판독은 합성에서 GT 와 같은 곡선을 내는 것으로 보정했다.

## 1. 주제 안

| | 한 문장 | 분석 기여 | 모델 기여 (world video prediction) | oral 가능성 |
|---|---|---|---|---|
| **A′ (추천)**<br>"Reach, not horizon" | latent predictor 는 마지막 관측에서 일정 **거리** 까지만 상태를 운반한다. 멈춤은 걸음 수가 아니라 거리로 모인다 (release 는 걸음 몫이 섞임, 학습된 predictor 는 순수 거리에 가까움) | reach 법칙 (합성 GT, predictor 넷) · 실영상 reach 곡선 (flow 유사 정답) · 앵커 (관측 vs 자기 예측) · 평가 감사 (VoE · EK100) | reach 가 변위에 무관한 predictor. 후보: 운반형 (token transport / latent warp), 자기 예측을 앵커로 쓰는 닫힘 학습, 큰 변위 커리큘럼 | **높음.** 한 문장 법칙, 사전 예측을 뒤집은 반증 (학습 지평 가설 기각), 실영상 재현, 처방 방향이 다 있다 |
| B<br>"Surprise is not simulation" | VoE · anticipation 점수는 reach 를 재지 못한다 | 복사 기준선 · EK100 치환 (SC3) | 없음 | 중간. A′ 의 한 절로 흡수한다 |
| C<br>"Label-free reach curves" | 합성에서 보정한 flow · encoder 판독으로 실영상 reach 를 잰다 | 측정 도구 | 없음 | A′ 의 방법 절로 흡수한다 |

## 2. A′ 논문 구조 (분석 측면)

1. **측정 (합성 = 판정기, 실영상 = 보정된 유사 정답).**
   - 합성: RollOut_v3 의 GT 위치를 쓴다. 판독은 진실 칸 템플릿 · 외형 템플릿 plateau 이고, 귀무 · 복사 · 등속 오라클을 같이 둔다.
   - 실영상: flow 추적을 유사 정답으로 쓴다. v3 에서 GT 1 칸 안 0.87–0.95 이고, flow 로 잰 곡선과 GT 로 잰 곡선이 같다.
   - encoder 판독과 합의한 쌍만 쓴다.
2. **reach 법칙 (SC1).**
   - 설계: stride 1 / 2 / 4, 분할점 raw 16 / 32, 문맥 4 / 8, predictor 넷.
   - 절벽 위치는 슬롯이 아니라 거리로 모이고, 분할점을 옮겨도 같은 거리다 (절대 위치 아님).
   - 학습 지평 가설은 사전 예측과 함께 기각을 싣는다. 이 **반증이 논문의 핵심 비자명성** 이다.
3. **실영상 (SC2v2).** SSv2 에서 predictor 넷 모두 적중이 변위로 떨어진다 (칸당 −0.07 ~ −0.08). Ariel 은 곡선 전체가 위다 (4–6 칸 +0.24).
4. **앵커 (기존 P3 · P3b 재해석).** reach 는 **마지막 관측** 에서 잰다.
   - 진짜 관측을 한 튜블릿씩 넣으면 (`ar_z`) 매 걸음 앵커가 새로 생겨 계속 전진한다 (0.361, 진실 0.440).
   - 릴리즈의 자기 예측은 앵커가 되지 못하고 두 걸음 뒤 멈춘다 (−0.012).
   - 학습된 predictor 는 자기 토큰 하나는 앵커로 받는다 (tf_last ≈ 0). 사슬이 되면 진실의 1/3–1/2 로 준다.
   - 연결 고리: 실제 시스템 (V-JEPA 2-AC planning) 이 horizon 1 closed-loop 로 **매 걸음 다시 관측하는** 것은 이 한계를 피해 가는 설계로 읽힌다. 이는 해석이고 검정하지 않았다.
5. **왜 (기전, 가를 수 있는 예측).**
   - (a) attention 조회 반경: 미래 질의가 문맥 물체 key 를 가져와야 한다. 필요성은 M4 에서 봤다 (시사).
     - 예측: reach 가 긴 predictor 는 거리 감쇠가 완만하다. **pv1 에서 맞는 방향이다** (1 차, SC1 §3-5): log attention 칸당 release −0.42 vs pv1 −0.30. 슬롯 몫은 release −0.096, pv1 ≈ 0 으로, 행동의 걸음 몫 (K 1.6 vs ≈ 0) 과 같은 방향이다.
   - (b) 학습 데이터의 변위 분포.
     - pv1 학습셋의 8 슬롯 최대 변위: 중앙값 3.76 칸, p75 5.67 칸.
     - pv1 의 reach 3.9 칸 (거리만 적합) 과 5.7 칸 (문맥 8 절벽) 이 이 범위다.
   - (a) 와 (b) 는 서로를 배제하지 않는다. (b) 를 가르려면 학습이 필요하다.
6. **평가 감사.**
   - VoE: 복사 기준선 85.00 vs 릴리즈 88.89 (n.s.).
   - EK100 (SC3): predictor 미래를 복사 · 평균 토큰으로 바꿔도 action −1.3 ~ −2.6 점이다 (상한).
   - 결론: "reach 곡선으로 평가하라".
7. **처방 → world video prediction 모델** (학습 필요, 사용자 결정 뒤).
   - 성공 기준: (i) 합성 · 실영상 reach 곡선이 변위에 평평하다, (ii) 자기 예측 사슬의 기울기가 진실과 같다, (iii) 장거리 anticipation · 가림 뒤 재등장에서 오른다.

## 3. sanity check 결과

| # | 질문 | 결과 | 판정 |
|---|---|---|---|
| **SC1** | 멈춤이 학습 지평 (슬롯) 에서 오나 | **아니다.** 슬롯만 R² 0, 거리만 R² 0.71 / 0.84 / 0.68 (release / pv1 / Ariel). Ariel 은 stride 1 에서 13–17 슬롯, stride 4 에서 2–3 슬롯에서 멈추고, 멈춘 거리는 6–7 칸으로 같다. 분할점 16 → 32 에서도 같은 거리다 (release 4.27 / 4.31, Ariel 6.50 / 6.14). 예측 변위 plateau: release 2.3–3.5 칸 (δ 에 따라 조금 오름), Ariel 5–6.7 칸 | 안 A **기각** → A′ |
| **SC2v2** | 실영상에서도 나오나 | **나온다.** SSv2 합의 쌍 2,031 에서 P(hit_p) 는 변위 0–1 → 4–6 칸 사이에 release 0.73 → 0.13, Ariel 0.85 → 0.37 (짝 +0.24 [0.18, 0.30]). 슬롯 효과도 있다 (슬롯당 −0.04 ~ −0.06). 합성 학습 pv1 · v11_postft 는 release 와 거의 같다 | A′ 실영상 일반성 **확보** (단서: Ariel 은 SSv2 로 학습) |
| **SC3** | anticipation 이 predictor 미래를 쓰나 | **거의 안 쓴다.** 같은 probe 에서 predictor 출력을 복사 / 평균으로 바꿔도 action R@5 는 released 35.34 → 32.80 / 34.01, paper 19.07 → 16.46 / 17.59. 재현 검사는 35.34 로 같다 | B 절 (평가 감사) **확보** |
| **B1 (묶음 G)** | 같은 학습 run 안에서 reach 가 학습 따라 느나 | **는다 (Ariel).** stride 1 plateau (s1_C16): ep5 / 19 / 30 / 43 = 3.52 / 4.25 / 4.69 / 5.08 칸 (release 2.30). pv1 · v11_postft (release 에서 이어 학습) 는 1 epoch 만에 도달하고 평평하다 | 학습 요인 교락 없는 증거 (1 차, CI 없음; SC1 §3-6) |

SC2 1 판 (encoder 템플릿 궤적 유사 정답) 은 **무효** 였다. 궤적이 슬롯 사이 평균 5.5 칸씩 뛰었다. 그래서 flow 판으로 바꿨다 (SC2v2 §1).

## 4. 다음 (우선순위)

| 순 | 무엇 | 왜 | 비용 · 학습 |
|---|---|---|---|
| 1 | **SC1 · SC2v2 · SC3 적대 검증** | 세 결과가 새 뼈대다. 판독 · 귀무 · 단서를 검증자가 다시 계산해야 한다 | CPU, 워크플로 |
| 2 | 기전 (a): pv1 에서는 예측한 방향이다 (SC1 §3-5). 남은 것: Ariel 판 (hook 을 prefix 마스크와 합친다), 그리고 인과 검정 (attention 온도 · RoPE 거리 척도를 바꾸면 reach 가 따라 움직이나) | 상관을 인과로 올린다 | GPU 약 6 분 / predictor + hook 수정 |
| 3 | 거리 vs raw 시간 분리: 같은 stride 에서 속도 폭이 큰 clip (v3 법칙 안 속도 분위를 넓히거나 flat_v 고속 부분집합) | SC1 단서 1 | CPU 재분석 먼저, 모자라면 데이터 |
| 4 | EK100 라벨 없는 reach 곡선 (SC4): `e_flowtrack.py` 를 EK100 clip 에 건다 | egocentric (손 · 카메라 운동) 에서도 같은 모양인지 본다 | GPU 약 10 분 + loader |
| 5 | ~~묶음 G (학습 run 안 epoch 곡선)~~ → 끝났다 (§3 B1 행). 남은 것: epoch 별 적중 판독 · CI | 교락 없는 증거를 확정으로 | CPU |
| 6 | **world video prediction 모델** (A′ 의 처방) | oral 의 "해결책" 칸 | **학습 필요 → 사용자 결정** |

⚠️ 6 은 새 학습이다. 사용자 결정 ("새로 학습은 말고") 에 따라 아직 손대지 않는다. 결정할 때 쓸 후보 세 가지:
- (i) **운반형 predictor.** 문맥 토큰을 예측된 변위로 옮겨 놓는다 (latent warp). 학습 없는 등속 오라클이 실영상 2–4 칸에서 release 를 이미 이긴다 (1–2 칸은 같다, SC2v2 §3-1).
- (ii) **닫힘 학습.** 자기 예측을 앵커로 되먹이며 학습한다.
- (iii) **큰 변위 커리큘럼.** 기전 (b) 가 맞다면 이것만으로 reach 가 는다.

## 5. 위험과 선제 대응

| 위험 | 대응 |
|---|---|
| "attention 이 가까운 것을 본다는 건 당연하다" | 당연하지 않은 것은 셋이다. (1) 학습 지평 가설이 **기각** 되고 멈춤이 걸음 수가 아니라 **거리** 쪽으로 모인다 (release 는 걸음 몫이 섞임). (2) 미래 예측 학습이 거리를 늘리지만 없애지 못한다. (3) 표준 평가가 그걸 못 본다. 기전은 (a) · (b) 둘 다 반증 가능한 예측으로 낸다 |
| 판독 인공물 | 판독 넷 (진실 칸 템플릿 · 외형 템플릿 plateau · flow 유사 정답 · encoder 천장) 이 같은 방향이다. flow 판은 합성에서 GT 와 같은 곡선을 낸다 |
| 모델 하나 (ViT-H) | encoder 는 고정 조건이 곧 실험 설계다 (predictor 넷을 바꿈). encoder 규모 재현은 보류 결정을 유지하고 한계 절에 적는다 |
| Ariel 실영상 우위는 도메인 안 | release 도 SSv2 를 사전학습에서 봤다. 합성 (도메인 밖) 에서 Ariel 우위가 더 크다 (+0.34 / +0.64) |
| 거리와 raw 시간이 섞였다 | §4-3. predictor 는 raw 시간을 직접 보지 않는다는 기전 논거는 있지만 데이터로는 아직 못 갈랐다 |

## 6. 재현 · 진행

- 묶음 H (`scripts/chain_0925h.sh`): SC1 추출 (`v3_p2s16_<tag>`). 분석은 `sc1_horizon_units.py` · `p2_analyze.py`.
- 묶음 I (`scripts/chain_0925i.sh`): SC2v2 flow 추적 (v3 보정 + SSv2). 분석은 `e_flowtrack_analyze.py`.
- SC3: EK100 `pred_replace` (코드 한 분기), val 만 5 회 (vll3).
- 묶음 J (`scripts/chain_0925j.sh`, 끝남): M4 pv1 → SC1 §3-5.
- 묶음 G: 끝남 → §3 B1 행 · SC1 §3-6.

## 7. 철회된 첫 판 (기록)

> 아래는 SC1 전의 안 A 요지다. **인용하지 않는다.**
> - 제안 메시지: "Latent video predictors extrapolate only as far as they were trained to look."
> - 근거로 든 것: 관측 한 걸음 전이는 넷이 같다 (0.36–0.39). open-loop 지평은 학습 질의 지평 근처에서 멈춘다 (릴리즈 ≈ 3, 학습 ≈ 8 튜블릿).
> - SC1 이 이것을 기각했다. 학습된 predictor 는 stride 1 에서 학습 지평의 2 배까지 가고, stride 4 에서는 2–3 슬롯에서 멈춘다. 같은 모양은 거리 축에서만 나온다.
> - 첫 판의 SC2 설계 (encoder 템플릿 궤적) 는 무효였다. SC3 · SC4 · SC5 계획 중 SC3 만 돌렸다.
