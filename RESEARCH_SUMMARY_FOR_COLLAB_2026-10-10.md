# 연구 여정 요약 — Rabbat 코웍 제안용 1차 자료 (2026-10-10)

> **용도**: IntPhys 2 저자 Michael Rabbat 에게 보낼 2장짜리 PDF 의 **원재료**. PI 가 읽고 "무엇을 묻고, 무엇을 해 왔고, 무엇을 하고 싶고, 같이 하면 무엇이 되나" 를 알 수 있게 정리했다. 세부 수치·방법·철회 기록은 뺐다.
> **근거**: 논문 스토리 정본 `auto_research/paper/PAPER_STORY_2026-10-08.md`, 결과 아카이브 `z_archive_10/` (레포 결과 문서 264개를 정리한 것), 루트 `CLAUDE.md`. **새로 계산한 수치는 없다.** 수치는 정본의 값을 옮겼고, 아직 적대 검증 전인 값은 *(1차)* 로 표시했다.
> **PDF 로 줄일 때 쓰는 우선순위**: §0 → §2 → §4 → §6 → §7. §3 (여정) 과 §8 (주의) 은 메일 본문이나 내부 참고용.

---

## 0. 한 문단 요약

사람은 관측만으로 "물체는 갑자기 사라지지 않는다", "받침이 없으면 떨어진다" 같은 intuitive physics 를 일찍 갖추고, 그걸로 그럴듯한 미래를 상상한다.
우리는 대표적인 latent world model 인 **V-JEPA 2** 가 이런 미래를 예측하는지, 못 한다면 **어디서, 왜** 못 하는지를 반년 동안 통제된 합성 testbed 와 외부 벤치마크로 쟀다.
결론은 세 가지다. **(1)** intuitive physics 가 "창발했다" 는 근거인 IntPhys 1 점수는 *예측하지 않아도* 대부분 나온다. 마지막으로 본 프레임의 표현을 미래 자리에 그대로 복사하는 기준선이 85%, 모델이 89% 다. 예측 없이 풀리는 정도(leakage)가 작은 IntPhys 2 에서는 모델이 복사 기준선과 구분되지 않는다. **(2)** 모델은 마지막 관측에 **보이는 것**만 짧게 외삽한다. 마지막에 **가려진 것**과 아직 **일어나지 않은 것**(가속, 낙하, 꼭대기에서 되돌아옴)은 미래에 넣지 않는다. **(3)** 그 미래를 만드는 데 필요한 정보는 encoder 표현에 이미 있다. 실패는 predictor 에 있다. predictor 만 다시 학습해도(미래 전용 목표, 자기회귀 포함) 학습 도메인 밖에서는 달라지지 않았다.
그래서 다음 목표는 **action 없이 관측만으로 이런 미래를 예측하는 predictor** 이고, action conditioning 은 그 위의 post-training 으로 둔다.

### English pitch draft (메일·PDF 첫 단락 후보)

> Humans acquire intuitive physics — objects persist, unsupported things fall — from observation alone, and use it to imagine plausible futures. We asked whether a leading latent video world model, V-JEPA 2, actually *predicts* such futures, and where it fails. Three findings: (1) the IntPhys 1 score often cited as evidence of emergent intuitive physics is largely reachable *without prediction* — copying the last observed latent into every future slot gets 85% vs. the model's 89% — whereas on IntPhys 2, where this leakage is small, the model is indistinguishable from the copy baseline. (2) On a controlled synthetic testbed, the model extrapolates only what is *visible at the last observation*, over a short range. It does not carry forward what is *hidden at that moment* (occluded objects), and it does not add what is *not yet present* (acceleration, falling off a ledge, turning back at the apex). (3) The information needed for these futures is linearly present in the encoder; the failure sits in the predictor, and retraining the predictor (future-only and autoregressive objectives) did not help outside its training domain. Our goal is an observation-only predictor that passes these tests, with action conditioning as post-training.

---

## 1. 연구 질문

| # | 질문 | 한 줄 답 (지금까지) |
|---|---|---|
| RQ1 | 지금의 벤치마크 점수는 **미래를 예측해야만** 얻을 수 있는가? | IntPhys 1 은 아니다 (복사 85 vs 모델 89). IntPhys 2 는 복사와 모델이 둘 다 우연 근처다 |
| RQ2 | world model 은 **어떤 종류의** 미래를 예측하고, 어떤 미래를 못 하는가? | 마지막 관측에 *보이는 것*은 2–3 물체폭 거리까지 외삽한다. *가려진 것*, *아직 없는 것*은 못 한다 |
| RQ3 | 실패는 **표현(encoder)** 쪽인가, **예측(predictor)** 쪽인가? | 정보는 encoder 에 있다. predictor 가 그것을 맞는 미래로 옮기지 않는다 |
| RQ4 | predictor 를 **다시 학습하면** 해결되는가? | 지금까지 시도한 학습(post-FT 4개, 공동작업자의 미래 전용·자기회귀 predictor 3개)은 전부 학습 도메인 안에서만 올랐다 |
| RQ5 (목표) | **action 없이 관측만으로** intuitive physics 를 따르는 미래를 예측하는 predictor 를 만들 수 있는가? AC 는 그 위에서 더 잘 되는가? | 진행 예정. 합격선(넘어야 할 칸)은 정해 두었다 |

**intuitive physics 의 판정은 행동으로 한다.** "모델이 중력을 안다" 같은 내부 귀속은 하지 않는다. 판정 단위는 "마지막 관측 기준으로 **보이는 것 / 안 보이는 것 / 없는 것**의 미래를 예측하는가" 이다.

---

## 2. 무엇을 만들었나 (방법론 기여)

1. **복사 기준선 (copy baseline) = leakage 의 측정자.** 문맥의 마지막 latent 를 미래 모든 자리에 그대로 둔 채, 같은 surprise 규칙으로 채점한다. 이 점수가 높으면 그 벤치는 *예측 없이* 풀린다. 우리가 확인한 범위에서 IntPhys 1 · IntPhys 2 · GRASP 논문은 이런 기준선을 보고하지 않았다. 이 기준선에서 파생한 **copy-hard 부분집합**(복사가 틀리는 쌍만 모은 것)도 만들었다.
2. **통제된 합성 testbed — IntPhysGen v11 (Unreal Engine)**: 21,504 clip, 10,752 쌍. 축은 가림 유무 · **가림이 문맥의 어디에 걸리는가**(가운데 / 끝) · 운동(정지 / 등속 / 등가속) · 위반 종류(사라짐 / 모양 / 색) 이다. 가능·불가능 쌍은 **문맥 픽셀이 완전히 같게** 렌더했다. 그래서 predictor 출력이 비트 단위로 같고, 채점은 "그 예측이 어느 미래에 더 가까운가" 라는 기하 문제가 된다.
   - 실물 메시로 다시 렌더한 판(`v11_realistic`, 선반 낙하 `ledge`)과, **미래 다섯 개 중 고르기**(`gravity_realistic`: 포물선 · 부유 · 멈춤 등) 판을 추가했다. 이로써 VoE 쌍 하나로는 가를 수 없는 "덜 움직이는 미래가 싸 보이는 것" 교란을 피한다.
3. **운동 법칙 세트 (RollOut v2/v3)** + **위치 readout(자)**: predictor 가 미래의 *어디에* 물체를 두는지 읽는다. readout 이 대상이 없어도 기본값을 내는 함정이 있어서, held-out 정밀도 · encoder 대조 · 파라미터 없는 교차검증을 함께 내는 규칙을 세웠다.
4. **재현 가능한 평가 하네스**: V-JEPA 2 의 IntPhys 1 / GRASP / InfLevel(Garrido 프로토콜), IntPhys 2(Bordes 프로토콜), EK100 action anticipation 을 하나의 런처로 재현했다. 예를 들어 IntPhys 1 은 ViT-H 88.89 로 공개 수치를 재현했다.

---

## 3. 여정 — 질문이 어떻게 바뀌어 왔나 (기록상 2026-07 ~ 10)

| 시기 | 무엇을 했나 | 무엇을 배웠고 무엇을 버렸나 |
|---|---|---|
| **7월–8월 중순** | Blender toy 데이터로 V-JEPA 2 의 물리 해석을 재현하려 함. IntPhys 1 Garrido 프로토콜 재현(ViT-H 88.89 / ViT-L 64.44) | 가림이 있으면 surprise 채점이 무너지는 것을 처음 봄. **문맥 픽셀이 같은 쌍**으로만 공정하게 채점된다는 규약을 확정 |
| **8월 말** | IntPhysGen v1 → v8 → v10 → **v11** 로 testbed 를 키움 | "가림이 무너뜨린다", "등가속은 가림 없이도 낮다", "가려진 *시간*은 무관하다(k=1–4 평평)" 를 확인. 반대로 *표현에는 정보가 남아 있다*(probe 98–100%). context 길이 · 배경 · 카메라 · 신호 크기 같은 원인 후보는 소거. "목적함수가 원인" 이라는 설명은 재학습 없이 검증할 수 없어 축에서 내림 |
| **9월 초** | 외부 피드백: "가림이 사건 순간에 걸리는 건 corner case". 위치 readout 도입, predictor knockout | 가림은 주인공이 아니라 **testbed 의 한 축**으로 내림. "predictor 가 물체를 마지막 자리에 둔다" 는 readout 기본값으로 판명해 철회 |
| **9월 중순** | 현상을 정의로 묶음 → **네 능력**(읽기 · 지속 · 전이 · 영속). predictor post-FT 시작 | 외삽 한계는 시간(슬롯)이 아니라 **거리**(마지막 관측에서 2–3 물체폭). post-FT 는 도메인 안에서만 오름 |
| **9월 말** | **복사 기준선** 도입 → IntPhys 1 85.0 vs 88.89. IntPhys 2 감사. 공동작업자(Ariel)의 미래 전용 · 자기회귀 predictor 평가. 실영상(SSv2 · EK100 · K400) 판독 | "벤치마크가 예측을 요구하지 않는다" 가 P2 의 중심이 됨. 재학습도 같은 실패를 보임. "memory 문제" 라는 프레임은 쓰지 않기로 함 |
| **10월 초** | 실물 메시 재렌더, 선반 낙하, 포물선 다중 미래, DINO-Foresight(다른 world model) 평가 | 다른 모델에서도 같은 실패. **사건(낙하 · 벽)과 꼭대기에서의 반전도 논문 범위에 넣기로** 결정(10-08) |
| **10-08 (현재)** | 스토리 확정: 사람 → world model 의 gap → testbed 진단 → 관측만으로 배우는 predictor | 목표 = 공동작업자 predictor 를 *넘는* 관측 학습 predictor. AC 는 작은 subsection(아직 최종 아님) |

**여정의 특징 (PI 에게 보여 줄 만한 점)**: 이야기의 틀은 열 번 넘게 바뀌었지만, 측정값과 철회 기록은 전부 남겼다. 틀을 바꾼 계기는 대부분 **결과**였다(knockout, readout 기본값 판명, 거리 한계, 복사 기준선). 철회한 가설은 370여 건이고 따로 표로 남겼다.

---

## 4. 핵심 결과 6가지 (PI 용, 수치는 최소)

### R1. IntPhys 1 은 예측 없이 대부분 풀린다. IntPhys 2 는 그렇지 않고, 거기서는 모델도 무너진다

| 벤치 · 모델 | 모델 | 복사 기준선 (예측 없음) | 읽기 |
|---|---:|---:|---|
| IntPhys 1 dev · V-JEPA 2 ViT-H | **88.9** | **85.0** | 차이 +3.9, 95% CI 가 0 을 포함 |
| IntPhys 1 · DINO-Foresight *(1차)* | 84.4 | 88.9–92.2 | 복사가 모델보다 높다 |
| **IntPhys 2 Main** (506 쌍) · V-JEPA 2 ViT-H | **54.3** | **55.5** | 같은 조건에서 +1.8 [−1.8, +5.5]. 구분되지 않음 |
| v11 문맥 끝 가림 · V-JEPA 2 / DINO-F | 우연 근처 | — | 예측이 필요한 칸에서 무너짐 |

- IntPhys 1 이 새는 이유로 두 후보를 측정했다. 위반이 전부 *외형* 사건이고(궤적 위반 0건), 움직이는 물체는 약 0.2초(모델 입력으로 1–3장)만 가려진다.
- 주장은 "창발이 없다" 가 아니다. **"창발의 근거로 쓰인 점수가 예측을 요구하지 않는다"** 이다.

### R2. 마지막 관측에 *보이는 것*만, 짧게 외삽한다
- 방향과 속도는 이어 간다. 하지만 물체다운 예측은 **마지막으로 본 자리에서 2–3 물체폭 안**에서만 유지된다. 이 한계는 시간보다 거리로 설명된다.
- 실영상에서도 움직이는 내용의 약 절반(SSv2 46%, EK100 51%)이 이 범위 밖이다 *(1차)*. 즉 corner case 가 아니다.

### R3. 마지막 관측에 *가려진 것*은 예측하지 못한다
- 가림이 문맥 **끝**에 걸리면 "물체가 사라진 미래" 를 정상으로 본다. 등속·등가속 팔에서 "물체→빈" 방향 정답률 0%, "빈→물체" 100% 이고, 실물 메시에서도 0–10% / 100% 로 같다.
- 같은 가림이라도 문맥 **가운데**에 걸려 끝에서 다시 보이면 처리한다(97.8 / 95.8 vs 끝 57.5). 그래서 문제는 "기억" 이 아니라 **마지막 관측** 기준의 예측이다.
- 가려진 시간(1–4 스텝)은 결과를 바꾸지 않는다.

### R4. 마지막 관측에 *없는 것*(가속 · 사건)은 미래에 넣지 않는다
- **가속**: 등가속 운동에서 predictor 의 진행은 등속 외삽 수준이다(28 vs 등속 32 vs 진실 54). v11 등가속은 가림이 없어도 모양 위반 감지가 94 → 69 로 떨어진다.
- **낙하**: 선반 끝을 지나는 공(실물 메시)에서 "떨어지는 미래" 보다 "같은 높이로 떠 가는 미래" 를 **120쌍 중 120쌍** 덜 놀랍게 본다.
- **꼭대기에서 반전**: 위로 던진 공에서 오르는 속도는 잇지만, 꼭대기 문맥에서는 98% block 에서 공을 높이 유지 쪽에 둔다.
- ⚠️ 떠 가는 미래는 "복사" 와 닮아서 VoE 쌍 하나로는 가를 수 없다. 그래서 **다중 미래(낙하 · 부유 · 상승 · 멈춤)** 설계로 다시 낸다(진행 중).

### R5. 정보는 encoder 에 있다. 실패는 predictor 에 있다
- encoder 표현에서 문맥 끝 속도를 선형으로 거의 완벽하게 읽는다(R² 0.996).
- 과거 16장만 본 encoder 표현 위의 작은 readout 이 낙하 · 벽 정지 · 가속 · **가려진 뒤 미래 위치**를 5–10 px 로 맞힌다. 같은 readout 을 predictor 출력에 걸면 12–30 px 다 *(1차)*.
- predictor 출력에서도 물체의 정체(모양·색)는 98–100% 로 읽힌다. 그런데 채점 민감도는 4.5% 다. **정보는 있는데 맞는 미래로 놓지 않는다.**
- IntPhys 2 에서도 encoder 평균 벡터 위의 선형 probe 는 쌍 정확도 73.7% 인데, 같은 영상의 VoE 는 54.3% 다(지도 학습 probe 라 같은 능력을 재는 것은 아니다).

### R6. predictor 를 다시 학습해도 학습 도메인 밖에서는 같다
- v11 로 post-FT 하면 v11 은 75.8 → 91.4 로 오르지만 IntPhys 1 은 88.9 → 77.2 로 떨어진다. 반대로 IntPhys 1 로 post-FT 하면 IntPhys 1 은 93.9 이지만 v11 은 78.5 다.
- frozen encoder 위에서 action 없이 실영상(SSv2 + K400)으로 미래를 예측하도록 학습한 predictor 3종(full attention · block-causal · 자기회귀; 공동작업자 Ariel 학습)도 마찬가지다. EK100 외삽 범위 2.1 → 2.2, 가림 복원 .53 → .46, IntPhys 2 51–54 로 전부 그대로다.
- **문맥에 없던 운동 변화(가속 · 중력 · 사건)를 미래에 넣게 된 predictor 는 아직 하나도 없다.**

---

## 5. IntPhys 2 에 대해 우리가 본 것 (Rabbat 이 가장 관심 가질 부분)

| 관찰 | 내용 | 의미 |
|---|---|---|
| **IntPhys 2 는 leakage 가 작다** | 복사 기준선이 우연 근처(55.5)다. IntPhys 1(85.0)과 대조적 | IntPhys 2 가 IntPhys 1 보다 **예측 능력을 더 정직하게 재는 벤치**라는 독립 근거. 협업 제안의 긍정적 출발점 |
| permanence 조건 | 우리 모델·채점에서 release 의 permanence 는 CI 가 50 을 포함한다. 반면 복사 기준선은 permanence-Fixed 에서 63.5 (CI 하한 > 50) | 논문의 "permanence 가 가장 쉽다" 는 신호의 일부가 *예측 없이* 나올 수 있다. ⚠️ 표본이 작다(52–68쌍) |
| 위반 시점 | 자동 검출(90% 커버, 오류율 10–15% 추정)로 보면 위반의 32%(solidity 는 55%)가 영상 앞 4초 안에 일어난다 → 고정 창에서는 예측 구간 밖 | 창 설계와 위반 시점의 관계를 같이 맞추면 벤치가 더 날카로워질 여지가 있다 |
| 쌍의 문맥 | 가능/불가능 쌍이 **위반 전에도** 픽셀 단위로 같지 않다 | matched-context 채점(문맥 동일 → 순수 기하 비교)이 애초에 성립하지 않는다. 다음 판의 설계 포인트 |
| encoder 는 읽는다 | 선형 probe 73.7 vs VoE 54.3 | gap 이 predictor 쪽에 있다는 우리 진단과 일관된다 |
| 재학습해도 우연 근처 | 공동작업자 predictor 3종 50.8–53.8 | 벤치가 "학습만 더 하면 풀리는" 쪽이 아니다 |

⚠️ IntPhys 2 공식 config 가 우리 환경에서 그대로 돌지 않아 코드를 따라 다시 구현했다. 정밀도 설정 수정은 아직 반영하지 않았다. 협업 시 **저자 쪽 하네스와 수치 대조**가 첫 작업으로 좋다.

---

## 6. 앞으로 하고 싶은 것

**주 목표 — 관측만으로 intuitive physics 를 따르는 미래를 예측하는 predictor**
- 조건: frozen encoder(정보는 이미 있다) + 새 predictor, action 없이 영상만으로 학습한다.
- **넘어야 할 합격선**은 공동작업자 predictor 를 같은 판에서 넘는 것이다. 구체적으로: (a) EK100 외삽 범위 > 2.2, (b) v11 문맥 끝 가림 "물체→빈" 정답률 > 50% (v11 로 학습하지 않고), (c) 가속을 등속보다 진실 쪽으로 넣기, (d) 꼭대기 문맥에서 되돌아오기 > 50%, (e) 선반에서 낙하 선택 > 우연, (f) IntPhys 1 이 복사 기준선 아래로 떨어지지 않기.
- 왜 기존 학습이 실패했는지에서 나온 설계 후보: 손실의 최저점이 "복사" 이므로 **copy-hard 가중 · 먼 지평**, 마지막 관측 dropout, 움직임 정렬 조회, 상태/렌더 분리, 시간 척도 합성. **사건(낙하 · 충돌)을 겨누는 후보는 아직 비어 있다.** 이것이 가장 열린 문제다.
- 판정 원칙: 학습 도메인 밖에서 재고(궤적 외우기와 구분), 모든 점수에 복사 대비 Δ 를 같이 낸다.

**작은 subsection — action conditioning 은 post-training 이어야 하는가** (가설, 아직 최종 아님)
- 같은 AC 데이터에서 AC scratch 와 "관측 학습 predictor 위 AC post-training" 을 비교하고, AC 데이터를 10 / 30 / 100% 로 바꾼다.

**분석 쪽 남은 일**: Figure 1(가림 + 선반), GRASP · InfLevel 복사 기준선(벤치 전체 leakage 지도), ledge · wall 다중 미래 세트, 다른 world model 하나 더(생성형 포함 검토), 벤치마크별 가림 통계.

---

## 7. 협업 포텐셜 — 무엇을 같이 할 수 있나

**우리가 가진 것**
- 벤치 점수를 "예측의 몫" 과 "예측 없이 풀리는 몫" 으로 가르는 **복사 기준선 · copy-hard 진단**과, 이를 IntPhys 1/2 · v11 · DINO-F · 여러 학습 predictor 에 적용한 결과
- *보이는 것 / 가려진 것 / 아직 없는 것* 을 분리해 재는 **통제 testbed**(문맥 픽셀 동일 쌍, 가림 위치 축, 실물 메시, 다중 미래)와 검증된 위치 readout
- "encoder 는 정보를 가졌고 predictor 가 그것을 미래로 옮기지 않는다" 는 **구체적인 실패 지도**와, 그로부터 나온 predictor 설계 합격선

**저쪽(Meta FAIR / IntPhys 2 팀)이 가진 것**
- IntPhys 2 를 설계한 관점과 메타데이터(위반 시점 등. 우리는 추정으로 대신했다), 사람 기준선
- V-JEPA 2 / 2-AC 를 학습 규모에서 바꿔 볼 수 있는 역량 (우리는 frozen encoder 위 predictor 학습, GPU 8장 규모)

**구체적인 협업안 (우선순위 순)**
1. **Leakage-aware evaluation**: 복사 기준선과 copy-hard 를 IntPhys 2 · GRASP · InfLevel 에 표준 진단으로 붙여 "벤치 × 모델 leakage 지도" 를 만든다. 짧고 분명한 공동 결과다.
2. **다음 판 벤치 설계 (IntPhys 2 후속)**: 불가능 미래가 *복사와 다른 방향*으로 움직이도록 한 다중 미래 설계, 문맥 동일 쌍, 위반 시점을 예측 구간 안에 두는 설계, "마지막 관측에서 가려진 / 아직 없는" 축의 분리. 우리 testbed 경험을 IntPhys 2 의 실사 장면 다양성과 합친다.
3. **관측만으로 학습하는 predictor**: 우리의 실패 지도와 합격선을 기준으로, 저쪽의 학습 규모에서 목표 · 구조 후보(copy-hard 가중, 먼 지평 등)를 검증한다. 사건(낙하·충돌)을 배우게 하는 문제는 양쪽 모두에게 열려 있다.
4. **AC post-training 가설**: V-JEPA 2-AC 인프라에서 "관측 학습 predictor 위 AC" vs "AC scratch" 를 데이터 양별로 비교한다.

---

## 8. 보내기 전 주의 (내부용)

- **저자 관계 확인**: Rabbat 은 V-JEPA 2, Garrido et al. 2025("intuitive physics emerges…"), IntPhys 2 의 공저자로 알고 있다. 메일 전에 저자 목록을 확인할 것. R1 은 그 Garrido 논문의 근거 점수를 겨누므로, **"반박" 이 아니라 "점수의 분해"** 톤으로 쓴다. 금지: "창발하지 않았다", "IntPhys 1 은 틀린 벤치다".
- **IntPhys 2 를 칭찬하는 방향으로**: "IntPhys 2 는 leakage 가 작아 진짜 gap 을 드러내는 벤치" 가 우리 결과가 실제로 말하는 것이다. "permanence 가 가장 쉽다" 와의 차이는 우리 모델·채점 기준이고 표본이 작다는 단서를 같이 붙인다.
- **언급하지 말 것**: IntPhys 2 의 심사 이력(venue 결과) 같은 내부 메모 내용.
- **경쟁 선행**: HERA(frozen V-JEPA 2 predictor + memory adapter, IntPhys 2 52.6 → 54.4), GEOPHYS(predictor 없이 IntPhys 2 를 푼다고 주장, split 미확인). 둘 다 동기가 겹친다. 우리 차별점은 학습 없는 측정 · 복사 대조 · 위치 readout · 실패의 종류 분리다.
- **공동작업자 결과**: R6 의 predictor 3종은 Ariel 이 학습한 것이다. 외부 공유 전에 동의와 크레딧 표기를 확인할 것.
- **쓰지 않는 표현** (CLAUDE.md §12 · 정본 §7): "world model 은 물리를 이해하지 못한다"(조건 없이) / "중력을 안다·모른다" / "memory 문제" / "encoder 가 미래를 표현한다" / "action-free 학습으로 해결된다" / "AC 는 post 여야 효과가 있다"(실험 전) / "predictor 에 정보가 없다". 대신 "**마지막 관측에 없는 X 를 따르는 미래를 예측하지 못한다**" 로 쓴다.
- **아직 약한 곳**: P1 문헌(영유아·동물 인용) 원문 미확인 / 픽셀 생성형 world model 은 안 쟀다 / 거리 한계 회귀는 귀무 보정 전 / ledge 결과는 다중 미래 세트로 다시 내야 한다 / 실영상 수치는 *(1차)*.

---

## 9. 출처 지도

| 내용 | 문서 |
|---|---|
| 스토리 · 합격선 · 리뷰 대응 · 금지 표현 | `auto_research/paper/PAPER_STORY_2026-10-08.md` |
| 결과 전체 (논문 흐름 순) | `z_archive_10/README.md` → `02`(leakage) · `04–06`(보이는/안 보이는/없는 것) · `07`(encoder vs predictor) · `08`(재학습) · `09`(목표) |
| 스토리 변천 · 결정 기록 | `z_archive_10/00_STORY_AND_TIMELINE.md` |
| IntPhys 2 감사 | `auto_research/Archive/INTPHYS2_PERMANENCE_AUDIT_2026-09-26.md` · `INTPHYS2_ONSET_2026-09-30.md` · `INTPHYS2_ENCODER_LINEAR_PROBE_2026-09-30.md` |
| IntPhys 1 복사 기준선 | `z_research/previous/TrainingEffects/ip1_copy/IP1_COPY.md` · `auto_research/Archive/H6_*` |
| testbed | `z_research/previous/IntPhysGenV11/README.md` · `z_research/previous/v11_realistic/README.md` |
| 철회 · 기각 목록 | `z_archive_10/10_REJECTED_AND_RETRACTED.md` · `CLAUDE.md` §6 |
