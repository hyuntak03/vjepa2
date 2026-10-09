# 논문 스토리 정본 (2026-10-08, 사용자) — world model 은 intuitive physics 를 따르는 미래를 예측하지 못한다 → 관측만으로 그것을 배우는 predictor

> **정본. 실험을 제안하거나 결과를 문장으로 옮기기 전에 이 문서를 먼저 본다.** 레포 운영 규칙은 루트 `CLAUDE.md`.
> 이 문서가 대체하는 것 (각 문서 머리에 "대체됨" 을 달았다):
> `auto_research/paper/` 의 `DIRECTION_2026-09-25` (§0-b 논지 · §1 목표 · §2 스토리) · `RETHINK_PREDICTOR_ROLE_2026-09-29` (§3 논지) · `NARRATIVE_v2_USER_2026-09-29` ·
> `ABSTRACT_v0/v1` · `STORY_EVIDENCE_2026-09-26` · `DESIGN_CHOICES_2026-09-25` 의 스토리 부분 · `DRAFT_v0–v2` / `REVIEW_v1–v2` 가 심사한 뼈대,
> `z_research/IntPhysGenV11/Archive/PAPER_STORY_2026-09-22` (및 그 앞 09-06 · 08-31 · 08-29 판).
> **옛 문서의 측정 수치 · 정정 · 기각표는 그대로 유효하다.** 바뀐 것은 이야기의 틀과 목표다 (§3).
> 등급: **확정** (적대 검증 통과) · **1 차** (검증 전) · **문헌** (우리가 안 잼) · **★** (미실행). ✓ = 2026-10-08 에 산출물에서 다시 읽어 확인 (§재현).

---

## 0. 원문 (사용자, 그대로)

**P1 : 사람은 intuitive physics 덕분에 plausible future를 쉽게 상상, intuitive physics는 observation 갖춰짐**
Physical world에서, 사람은 앞으로 어떤 일이 일어날지, 물리적으로 가능한 미래들을 쉽게 상상할 수 있다.
이것은, (물체가 갑자기 사라지지 않는다, .. 중력이 있다) 와 같은 intuitive physics를 알고 있기 때문이다.
이런 intuitive physics는 영유아, 동물들에게 가장 초기에 observation만으로 갖춰지는 능력이며,
이 능력 덕분에 intelligent system은 새로운 환경이나 task에 대해 빨리 적응하고 이해할 수 있다.

**P2 : world model은 발전 빠르게, 근데 intuitive physics 못한다.**
World model은 physical world의 dynamics를 배우고자 하고, action planning, prediction, action anticipation, ,. Etc 등에서 빠르게 발전하고 있다.
하지만 physical world의 dynamics를 이해한 것의 가장 기초적이고 근본적인 intuitive physics 못한다.
Figure 1 예시 hook
사람은 쉽게 직관적으로 이런 예시에서 가능한 미래를 상상할 수 있다. ← world model과 Human 간의 gap 이 있다.

**P3 : To understand where world models fail, we build a controlled synthetic testbed.**
정확히 어디서 무너지는지 파악하기 위해 testbed 제작
봤더니, 지금은 context에서 보이는것만 외삽하지, context에서 안보이거나 없는 정보는 못함 ⇒ physical world의 intuitive physics 배우고 이해했다고 보기 어렵
encoder는 정보 충분한데, 단순히 predictor가 intuitive physics에 따르는 plausible future prediction 하는 것에서 실패

**목표 (사용자: "어떻게 될 지는 모르겠지만 아래와 같은 것을 하고 싶음")**
Action conditioned 없이, 순수 observation 만으로 intuitive physics를 배우고 ⇒ 이에 따른 plausible future prediction 을 잘하는 predictor를 만들고 싶음
(1)이 선행되어야 action conditioned 학습도 효과가 있다. {scratch 부터 action 학습하는게 아니라, action-conditioned은 post 가 되어야 한다.}

**후속 결정 (사용자, 2026-10-08 같은 날, 그대로)**
- Ariel 반례에 대해: "이건 어케든 맞춰야지." → §2-4 합격선
- IntPhys 1 88.89 와 P2 의 충돌에 대해: "이거 leakage있는 벤치잖아." → §2-2
- ledge · wall · 꼭대기: "이것도 포함해서 논문 쓸거야." → 09-19 의 "범위 밖" 결정 번복, §2-3 (iii)
- AC: "이건 이제 자그많게 subsection 느끼믕로 실험 하나 넣을거라 괜찬ㅇ하." → §2-5

---

## 1. 한 문장

> **사람은 관측만으로 intuitive physics 를 갖추고 그것으로 plausible future 를 상상한다.
> 지금의 world model 은 마지막 관측에 보이는 것만 외삽하고, 거기 안 보이는 것 (가려진 물체) 과 거기 없는 것 (아직 일어나지 않은 낙하 · 되돌아옴 · 가속의 누적) 은 예측하지 못한다.
> 필요한 정보는 encoder 에 있으므로 실패는 predictor 의 미래 예측에 있다.
> 우리는 action 없이 관측만으로 intuitive physics 를 따르는 미래를 예측하는 predictor 를 만들고, action conditioning 은 그 위의 post-training 으로 둔다.**

"context 에서 보이는 것" 은 **마지막 관측 (문맥 끝)** 으로 좁혀 쓴다 — 문맥 가운데에서 가려졌다 끝에 다시 보이는 물체는 처리한다 (v11 vanish 채점 early 97.8 / mid 95.8 / late 57.5, CLAUDE.md §5-5). 이 좁힘이 P3 를 측정과 맞춘다.

---

## 2. 문단별 뼈대 — 무엇을 말하고, 무엇으로 서나

| # | 문단 | 세우는 문장 | 근거 상태 | 근거 위치 |
|---|---|---|---|---|
| **P1** | 사람 · 영유아 · 동물 | intuitive physics (사라지지 않음 · 중력) 는 관측만으로 가장 먼저 갖춰지고, 새 환경 · task 적응의 바탕이다 | **문헌** — 인용 확인 필요 (§2-1) | NARRATIVE_v2 §A 인용 목록 |
| **P2** | world model 의 gap | world model 은 빠르게 발전하지만, intuitive physics 를 따르는 미래는 예측하지 못한다. Figure 1 hook | 문헌 + 우리 측정 (IntPhys 1/2 · 복사 기준선 · DINO-F) — **1 차 ~ 확정 혼재** (§2-2) | `PREDICTOR_SHARE` · `INTPHYS2_PERMANENCE_AUDIT` · `H6` · `DINOF_*` · `COPY_VS_POSITION` |
| **P3** | testbed 진단 | 마지막 관측에 보이는 것만 외삽한다 · 안 보이는 것 / 없는 것은 못 한다 · encoder 에는 정보가 있고 predictor 가 실패한다 | **확정 (합성) · 1 차 (실영상 일부)** (§2-3) | CLAUDE.md §5 · `RollOutV2` · `v11_realistic` · `FUTURE_FROM_CONTEXT` · `I_SCENE` |
| **목표** | 방법 | action 없이 관측만으로 intuitive physics 를 배워 plausible future 를 예측하는 predictor — **Ariel 을 넘어야 한다** | **★ 미실행 (사용자 학습 항목)**. 넘을 합격선은 §2-4 | `DESIGN_CHOICES` §3-b–d · `TrainingEffects` · `AR_PREDICTOR` |
| **AC** | 작은 subsection · 실험 하나 | (1) 이 먼저여야 action-conditioned 학습이 효과가 있다 — AC 는 post-training | **★ 실험 하나로 세운다** (§2-5) | `z_AC_training/README.md` |

### 2-1. P1 — 문헌만으로 선다. 인용을 확인할 것

- 영아의 사라지지 않음 · 가려진 운동 지속: Baillargeon & DeJong 2017; Johnson et al. 2003; Kochukhova & Gredebäck 2007; Kaufman et al. 2005; Rosander & von Hofsten 2004 (NARRATIVE_v2 §A 에서 사용자가 쓴 목록).
- 계산 모델 · 관측 학습: Piloto et al. 2022 (PLATO — object-centric slot, 우리는 장면 단위라 그 차이를 방법 절에서 적는다); LeCun 2022 의 "(1) estimate missing information (2) predict plausible future states" 는 P3 의 "안 보이는 것 / 없는 것" 과 그대로 겹친다.
- ⚠️ **"동물" 과 "새 환경 · task 적응" 은 아직 인용이 없다.** 후보: 갓 부화한 병아리의 가려진 물체 추론 (Vallortigara 그룹), Lake et al. 2017 (intuitive physics 를 학습의 start-up software 로). **실물 확인 전 인용 금지.**
- 인간 비유가 이제 intro 첫 문단 전체다 (옛 규칙 "첫 문장과 discussion 에만" 을 바꾼다). **본문 판정은 여전히 측정 가능한 정의로 한다** (§3).

### 2-2. P2 — "world model 은 intuitive physics 를 못 한다" 를 무엇으로 세우나

V-JEPA 2 의 IntPhys 1 **88.89** (Garrido et al. 의 "창발") 는 P2 의 반례가 아니다 — **IntPhys 1 은 leakage 가 있는 벤치다** (사용자 10-08).
여기서 **leakage = 미래를 예측하지 않고 관측만으로 (마지막 관측과 실제 미래의 변화를 비교해서) 정답이 나오는 정도** 이고, **복사 기준선** (마지막 문맥 latent 를 미래 자리에 그대로 둔 채점) 으로 잰다.
⚠️ CLAUDE.md §1-2 의 "미래일치 쌍의 누수" 와 다른 뜻이다 — 논문에서는 용어를 한 번 정의하고 쓴다.

**(a) IntPhys 1 은 leakage 가 크다 — 높은 점수가 예측을 요구하지 않는다**
- 복사만으로 **85.00** — V-JEPA 2 ViT-H 88.89 와 Δ +3.9 [−0.6, +8.9], CI 가 0 을 포함 ✓. 다른 world model (DINO-Foresight) 은 복사 88.9–92.2 가 predictor 84.4 보다 높다.
- 왜 새는가 (기제 후보, 둘 다 측정됨): 위반이 전부 **외형 사건** (사라짐 · 모양 · 연속성) 이고 궤적 · 속도 위반은 dev 에 0 건 (CLAUDE.md §6) — 토큰 평균 L1 은 외형 변화를 잘 보고 위치는 못 본다 (`COPY_VS_POSITION` §1). 그리고 **움직이는 물체는 0.2 초 (모델에게 1–3 장) 만 가려진다** (`OcclusionStats`) — 위반이 사실상 보이는 채로 일어난다.

**(b) leakage 가 작은 곳에서는 무너진다**

| 벤치마크 · 칸 · 모델 | predictor | 복사 (= leakage) | 판정 | 등급 |
|---|---:|---:|---|---|
| IntPhys 1 dev · V-JEPA 2 ViT-H (`skip2_w32`) ✓ | 88.89 | 85.00 | leakage 가 큼 — predictor 몫 n.s. | 확정 |
| IntPhys 1 dev · DINO-Foresight | 84.4 | 88.9–92.2 | leakage 가 predictor 보다 큼 | 1 차 |
| IntPhys 2 Main (506 쌍) · V-JEPA 2 ViT-H ✓ | 54.3 | 55.5 | leakage 작음 → predictor 도 그 수준. 같은 C 에서 +1.8 [−1.8, +5.5] | 확정 |
| IntPhysGen v11 · 문맥 끝 가림 (late) 칸 · V-JEPA 2 ViT-H | 50–70 (ramp vanish 50 = A 0 / B 100) | 39–63 | 예측이 필요한 칸에서 우연 근처 | 확정 (`RETHINK` §5-b) |
| IntPhysGen v11 · DINO-Foresight ✓ | 77.1 | 76.4 | 가림 세 조건 chance, vanish A/B 0/100 — ViT-H 와 같은 실패 | 1 차 |
| 위치만 다른 쌍 (RollOut_v2 ledge 낙하 vs 부유 / wall 정지 vs 통과, 392 쌍씩) · ViT-H ✓json | 0.0 / 17.6 | 7.9 / 2.3 | predictor 는 불가능 쪽을 고른다 | 1 차 |

→ 쓸 문장: **"intuitive physics 창발의 근거인 IntPhys 1 은 예측 없이 85 % 가 풀리는 (leakage 가 큰) 벤치이고, leakage 가 작은 곳 (IntPhys 2 · 문맥 끝 가림 · 위치 위반) 에서 world model 은 우연 근처다."**
→ **P2 그림 후보: leakage (복사 정확도, x) vs 모델 정확도 (y)** — 벤치마크 · 칸마다 한 점. 대각선 위에 붙으면 "모델이 leakage 이상을 못 한다". GRASP · InfLevel 의 복사 기준선이 아직 없다 (§5).
⚠️ 마지막 줄의 VoE metric (토큰 평균 L1) 은 위치를 거의 못 본다 (`COPY_VS_POSITION` §2) — 위치 위반의 0 % 는 "metric 한계 + predictor 의 틀린 미래" 가 섞인 값이다. 위치 주장은 자 판독 (RollOutV2 §5-5) 과 같이 쓴다.
⚠️ IntPhys 2 의 낮은 점수에는 장면 복잡도 등 다른 이유도 섞일 수 있다 — "leakage 가 작아서 무너진다" 는 상관 서술이고, 인과로 쓰려면 같은 장면에서 leakage 만 바꾼 쌍 (v11 의 가림 위치 축이 그 역할) 을 같이 보인다.

**(c) 문헌**: IntPhys 2 (Bordes et al. 2025) 는 평가한 모든 모델이 사람 (96.4) 에 크게 못 미친다고 보고한다 (V-JEPA 2 계열 57.51 — 논문 수치). 생성형 world model 쪽 대응 문헌 (Physics-IQ 류) 은 **실물 확인 후** 인용.
⚠️ 일반성: 우리가 잰 것은 latent 예측형 (V-JEPA 2 계열 · Ariel · post-FT · DINO-Foresight) 뿐이다 — 생성형까지 "world model" 로 쓰려면 하나를 더 재거나 문헌으로 받친다 (§4-2).

**Figure 1 hook 후보** (사람에게는 뻔한데 모델은 틀리는 장면, P1 의 두 예시와 짝):

| 후보 | P1 예시 | 모델이 내는 것 | 근거 | 단서 |
|---|---|---|---|---|
| 가림막 뒤로 굴러가는 공 (문맥 끝 가림) | 사라지지 않는다 | p 의 미래에 처음부터 물체다운 토큰이 없다; '물체→빈' 미래를 덜 놀랍게 본다 (v11 0 %, 실물 메시 0–10 %) | CLAUDE.md §5-1 · §5-5, `v11_realistic` §1 | 경계 가림은 corner case 비판 (§4-3) |
| 선반 끝을 지나는 공 | 중력 | 떨어지는 미래보다 같은 높이로 떠 가는 미래가 덜 놀랍다 (실물 메시 120 / 120 쌍); 위치 자는 선반 높이 유지 | `v11_realistic` §2 · RollOutV2 §5-5 | L1 이 덜 움직이는 미래를 싸게 볼 수 있다 (§4-4) |
| 위로 던진 공 (문맥이 오르는 중 · 꼭대기에서 끝남) | 중력 | 오르는 속도는 잇되 꼭대기에서 돌아 내려오지 않는다; 꼭대기 문맥에서 공 자리는 98 % block 이 높이 유지 쪽 | `v11_realistic` §4 (2026-10-08 정정판) | 튜블릿별 판정은 흔들린다 — 줄의 흐름으로만 |

추천: **앞 두 개를 한 그림의 두 패널** (P1 의 "사라지지 않음 · 중력" 과 1:1). 모델의 미래는 latent 라 그대로 못 그린다 → **후보 미래 중 무엇에 가까운가** (파라미터 0, `fig_gravity_choice` 형식) 를 주 표현으로, 검증된 위치 자 궤적을 보조로.

### 2-3. P3 — testbed 가 보여 준 것

testbed: IntPhysGen v11 (가림 × 운동 × 위반, 문맥 픽셀 동일 쌍) · v11_realistic (실물 메시) · gravity_realistic (포물선의 미래 다섯) · RollOut_v2/v3 (운동 법칙 · 위치 자). 문단 순서대로:

**(i) 마지막 관측에 보이는 것은 외삽한다 — 1 차 운동까지, 짧게**
- 방향 · 속도는 잇는다: 등속 flat_v 위치 자 0.71 칸, gain 0.85 (RollOutV2 §5-5). 포물선의 처음 1–4 튜블릿은 진짜 궤적을 따른다 (gravity_realistic §4).
- 짧다: 물체다운 토큰은 **마지막 관측 자리에서 약 2–3 칸 (40–60 px) 안** 에서만 유지된다 (거리 100 px 당 −0.47; CLAUDE.md §5-5). 실영상에서도 움직이는 내용의 약 절반 (SSv2 46 %, EK100 51 %) 이 그 범위 밖이다 (`I_SCENE` §4-3, 1 차).

**(ii) 마지막 관측에 안 보이는 것 (가려진 물체) 은 못 한다**
- v11: 가림이 문맥 끝에 걸리면 vanish 채점이 무너지고 (운동 팔 50 % = '물체→빈' 0 / '빈→물체' 100), k = 1–4 는 평평하다 (CLAUDE.md §5-1). 실물 메시에서도 같다 (`v11_realistic` §1, A 0–10 / B 100).
- 위치: 문맥 끝 가림이면 p 의 미래에 **처음부터** 물체다운 토큰이 없다 (late 0 슬롯 < early/mid 3 < 가림 없음 4–5; §5-5).
- 실영상: K400 팬 + 마지막 k 프레임 가림 띠 — 가려졌던 내용은 보이는 내용의 절반만 제자리에 복원되고 (k=4 근거리 .21 vs .42), predictor 는 원래 내용보다 가림막을 그린다 (`I_SCENE` §4-4, 확정).

**(iii) 마지막 관측에 없는 것 (아직 일어나지 않은 사건 · 누적되는 변화) 은 못 한다**
- 가속을 미래에 넣지 않는다: flat_a p 28 ≈ 등속 외삽 32 (진실 54) (STATE_EVOLUTION_CAPABILITIES §2); v11 등가속은 가림 없이도 shape 69.0 (정지 94.0 · 등속 89.9); 학습한 AR predictor 도 α < 0 (arc −0.42 · ledge −0.46, `AR40_READOUT`, 1 차).
- 낙하가 문맥에 없으면 떨어뜨리지 않는다: ledge 실물 메시 0 / 120 쌍 ✓ (`v11_realistic` §2); 위치 자 (등속 학습셋) 는 선반 높이 유지 (§5-5); 벽에서는 2 슬롯 통과 뒤 물체를 잃는다.
- 꼭대기에서 돌아오지 않는다 (미래 다섯 중 고른 비율, chance 20 %; 오르는 문맥 float 83.3 · 꼭대기 arc 62.5 / float 33.3 · 내려가는 문맥 arc 85.4 ✓): 오르는 문맥 → 앞 4 튜블릿은 arc, 꼭대기 뒤로 내려오지 않음; 마지막 튜블릿은 세 조건 모두 '그 자리 멈춤' 미래에 가장 가깝다 (`v11_realistic` §4, 2026-10-08 정정판).
- **ledge · wall · 꼭대기는 논문에 넣는다** (사용자 10-08 — 09-19 의 "common sense 라 정의 밖" 결정 번복). 네 능력 안에서는 **3 전이** 를 둘로 나눠 읽는다: **3a 추세** (가속 · 감속 · 꼭대기에서 되돌아옴 — 문맥이 드러낸 운동 추세의 연장) / **3b 사건** (받침이 끝나면 낙하, 벽에 닿으면 정지 — 장면 요소가 촉발하는 변화).
- ⚠️ **넣는 조건 (옛 검증이 남긴 단서, 같은 무게로)**:
  1. **떠 가는 미래는 복사와 닮았다** — 물체가 선반 높이에 머물러 "진화 없는 복사" 도 같은 쪽을 고른다 (H23 §3-7: 복사 LN(z) 도 59–99 % 가 부유 쪽, RollOut_v2 VoE 복사 7.9 %). 그래서 VoE 쌍 하나로는 "불가능한 연속을 예측한다" 와 "그냥 머문다" 를 못 가른다. **P3 의 주장 ("보이는 것만 외삽하고 낙하를 만들지 않는다") 에는 둘 다 맞는 방향**이지만, 논문 그림은 **불가능 미래가 복사와 다른 방향으로 움직이는 설계** (gravity_realistic 처럼 미래 여럿: 낙하 · 부유 · 상승 · 멈춤 / 정지 · 통과 · 튕김) 로 낸다 → §5.
  2. **갈림 시점 누수** — RollOut_v2 ledge 쌍은 프레임 35 에서 갈려 슬롯 0 (프레임 32–33) 의 h 차이가 target encoder 가 뒤를 본 몫이다 (H23 §3-7). 새 세트는 갈림을 예측 구간 안쪽으로 두고 슬롯 0 을 따로 보고한다.
  3. **자 둘이 다른 답을 낸다** — 등속 학습셋 자 (RollOutV2) 는 p 가 선반 높이를 유지한다고, ledge 라벨로 학습한 자 B (`FUTURE_FROM_CONTEXT` §1) 는 가림 없는 v2 의 p 에서 낙하를 5.4 px 로 읽는다. 자 B 는 장면 종류를 읽고 지도 라벨의 곡선을 붙이므로 "p 가 낙하를 담는다" 의 근거가 못 된다 — 두 결과를 조건 표로 같이 낸다.
  4. L1 교란 — 같은 높이 직진은 문맥 끝 속도 그대로라 L1 이 작을 수 있다 (`v11_realistic` §2). gravity 에서 가장 덜 움직이는 '멈춤' 은 1/144 만 고르므로 "가장 덜 변하는 미래" 만으로는 설명되지 않는다.

**(iv) encoder 에는 정보가 있다 — 실패는 predictor 의 미래 예측이다**
- 문맥 encoder z → 문맥 끝 속도 ridge R² 0.996 / 0.989 (확정).
- 과거 16 장만 본 z 위의 작은 자 (12.8k) 가 ledge 낙하 · wall 정지 · ramp 가속 · **v11 late 가림 k=1–4 의 미래 위치** 를 5–10 px 로 맞힌다 (미래를 본 h 와 같은 수준). 같은 자를 p 에 걸면 v11 late 가림에서 12–30 px (`FUTURE_FROM_CONTEXT`, 1 차).
- p 에도 정체는 남는다: 등가속 + 가림에서 p probe shape 98.46 / color 99.49 / env 100 인데 채점 민감도는 4.46 (CLAUDE.md §5-2) — 정보는 있는데 맞는 미래로 놓지 않는다.
- 실영상: encoder 토큰 평균 → ego-motion R² .69 / .63, EK100 미래 변위의 67–75 % 가 카메라 운동 (`ENCODER_MOTION_READOUT`, 1 차).
- ⚠️ **"encoder 가 미래를 표현한다" 는 못 쓴다** — 이 데이터에서 미래는 현재 상태의 결정적 함수이고 자는 지도 라벨이 가르친 곡선을 붙인다 (`FUTURE_FROM_CONTEXT` §3). 쓸 문장: "미래를 만드는 데 필요한 정보는 입력에 선형으로 있고, predictor 는 그것을 미래로 옮기지 않는다."
- ⚠️ shape 은 encoder 도 흔들린다 (z 3 팔 평균 64.8) — "encoder 가 충분하다" 는 color · env · 위치 · 속도로 쓴다.

**(v) 다시 학습해도 같다 — "단순히" 를 받치는 대조이자 목표의 제약**
- 미래 전용 · 되먹임 목표로 실영상에서 학습한 predictor (Ariel full · prefix · AR): reach 는 도메인 안에서만 늘고 EK100 은 2.2 그대로, permanence 그대로 (hid/vis .46–.53), AR 은 자기 되먹임 첫 걸음부터 무너진다 (`AR_PREDICTOR`, 확정).
- post-FT 넷: 학습 도메인 안에서만 오른다 (v11 post-FT v11 75.8 → 91.4 인데 IntPhys 1 88.89 → 77.22; IntPhys 1 post-FT 93.89 인데 v11 78.5). **문맥에 없던 운동 변화 (가속 · 감속 · 중력) 를 미래에 넣게 됐다고 확인되는 predictor 는 없다** (`TrainingEffects` §0).

### 2-4. 목표 — 관측만으로 intuitive physics 를 배우는 predictor (★, 사용자 학습 항목) — **Ariel 을 넘어야 한다**

분업은 그대로다 (DIRECTION §0): **학습은 사용자, 분석 · 판정 기준 · 설계 근거는 Claude.**
Ariel 세 팔 (full · prefix · AR) 은 정확히 "frozen encoder + 새 predictor + action 없는 실영상 미래 예측" 이고 실패했다. **사용자 결정: 반드시 넘는다** ("어케든 맞춰야지"). 그러려면 무엇을 넘었다고 할지가 먼저 고정돼야 한다.

**합격선 — 새 predictor 가 같은 판에서 Ariel 을 넘어야 하는 칸** (값은 문서 값, ✓ 는 10-08 재확인. ★ = Ariel 값 미측정 → §5-3)

| P3 축 | 지표 (판) | 릴리즈 | Ariel (가장 나은 팔) | 합격선 | 출처 |
|---|---|---:|---:|---|---|
| 보이는 것 | 실영상 reach R50 (칸) — SSv2 · **EK100** | 2.7 · 2.1 | 4.1 · 2.2 (full/prefix) | EK100 > 2.2 (도메인 밖; 어떤 학습도 못 넘은 천장) | `I_SCENE` · `AR_PREDICTOR` |
| 안 보이는 것 | v11 문맥 끝 가림 vanish '물체→빈' (static · flat · ramp) | 65 · 3 · 0 | 49 · 37 · 13 (block-causal ep43) | 셋 다 > 50, **v11 로 학습하지 않고** (v11 post-FT 87 · 100 · 100 은 도메인 안이라 기준이 아니다) | `TrainingEffects/scores/SCORES.md` |
| 안 보이는 것 | 실영상 permanence (가려졌던 내용 / 보이던 내용 복원 비) | .53 | .46–.47 | > .53 | `I_SCENE` §4-4 · `AR_PREDICTOR` |
| 없는 것 (3a) | 가속 반영 α (RollOut_v3 arc · ledge, t4–7; 0 = 등속 외삽, 1 = 진실) | +0.08 · −0.25 | AR −0.42 · −0.46 (prefix · full ★) | 둘 다 분명히 > 0 (encoder 천장 h +1.43 · +1.28) | `AR40_READOUT` §1 (1 차) |
| 없는 것 (3a) | gravity 꼭대기 문맥 — 공 자리가 arc 쪽인 block | 2 % | ★ | > 50 % | `v11_realistic` §4 |
| 없는 것 (3b) | ledge 실물 메시 VoE · 다중 미래에서 낙하 선택 | 0 / 120 ✓ | ★ | 복사 이상 + 낙하 선택 > 우연 | `v11_realistic` §2 |
| 닫힘 | 자기 예측 되먹임 − one-shot (장면 적중) | −0.03 ~ +0.01 | −0.12 ~ −0.22 | ≥ 0 | `I_SCENE` §4-5 |
| 떨어지지 않음 | IntPhys 1 (leakage 큼 — 올라도 증거 아님) · 복사 대비 Δ | 88.89 (복사 85.0) | 69.44 | 복사 85.0 아래로 안 떨어짐 | `SCORES.md` |

**Ariel 이 왜 실패했나 — 분석이 아는 것 → 설계 후보** (`DESIGN_CHOICES` §3-b · §3-c · §3-d, 전부 미검증)

| Ariel 의 실패 (측정) | 설계 후보 | 겨누는 칸 |
|---|---|---|
| 손실의 최저점이 복사 — 복사가 틀리는 쌍에서 prefix · full 은 우연 (49–51) | copy-hard 가중 · 먼 지평 (§3-b 1, §3-d B) | 보이는 것 · 안 보이는 것 |
| 먼 출처 조회가 약하다 — reach 가 도메인 안에서만 는다 | where 경로 (변위 → 조회 위치, §3-c ②) | 보이는 것 |
| 마지막 관측에 없는 내용은 안 이어 간다 — permanence 그대로 | 마지막 관측 dropout (§3-d D) · 역사 축적 (§3-c ④) | 안 보이는 것 |
| 가속을 안 넣는다 (α < 0) | 시간 척도 합성 (§3-d C) · 합성 규칙 (§3-d H) | 없는 것 3a |
| 자기 출력을 되먹이면 첫 걸음부터 무너진다 (AR) | 상태/렌더 분리 (§3-c ①) | 닫힘 |
| **사건 (낙하 · 충돌) 을 안 만든다** | **후보 없음 — 빈칸.** 관측만으로 사건을 배우려면 사건이 있는 창의 비중 (데이터) 이나 그 창을 고르는 압력 (가림 사건 채굴 §3-d E 의 사건판) 이 필요하다 | 없는 것 3b |

**방법 절이 지켜야 하는 것**
1. **첫 줄은 "Ariel 과 무엇이 다른가"** — 위 표의 어느 후보를 왜 골랐는지.
2. **궤적 외우기와 구분** — 학습 도메인 밖 (held-out 장면 · 운동 · 데이터셋) 에서 잰다 (`PAPER_STORY_2026-09-22` §4-3, v11 post-FT 가 두 궤적의 재생이었던 교훈). 합성 물리 데이터로 학습하면 testbed 판은 도메인 안이 된다 — 그러면 합격선은 실영상 · 다른 testbed 에서 본다.
3. **판정은 예측이 필요한 칸에서** — 모든 점수에 복사 대비 Δ, copy-hard 부분집합, 위치만 다른 쌍. VoE 전체 점수는 "떨어지지 않는다" 용도만.
4. **intuitive physics 의 판정 = 행동** — §3 대응표 (보이는 것 · 안 보이는 것 · 없는 것 3a/3b).

### 2-5. AC — 작은 subsection, 실험 하나 (사용자 10-08)

- 주장: "관측만으로 intuitive physics 를 배운 predictor 가 먼저여야 action-conditioned 학습이 효과가 있다 — AC 는 scratch 가 아니라 post-training." 지금 근거는 0 이고, **실험 하나로 세운다.**
- **추천 설계 — 한 실험, 두 팔 × 데이터 양 축**: 같은 AC 데이터 · 같은 step 예산에서 (a) AC scratch (V-JEPA 2-AC 레시피, `load_predictor: false`) vs (b) **우리 관측 학습 predictor 위 AC post-training**, AC 데이터를 10 / 30 / 100 % 로. post-training 의 이득은 데이터가 적을 때 먼저 보이므로 데이터 양 축이 있어야 어느 쪽 결과든 읽힌다 (하네스 2). (c) 릴리즈 predictor 위 post-training 은 여유가 있으면 — "intuitive physics 가 먼저" 와 "아무 사전학습이나 먼저" 를 가르는 팔이다.
- 판정: held-out 행동 조건부 rollout 오차 (k 걸음 latent L1) + 같은 intuitive physics 판 (§2-4 합격선 일부) 이 AC 뒤에도 남는가. planning 성공률은 환경이 있으면 (`z_AC_training/Archive/PLANNING_AND_ROBOCASA_2026-09-20.md` 의 RoboCasa).
- 정할 것: AC 데이터 (DROID 는 로컬에 없다) 와 encoder (우리 predictor 가 ViT-H 위라 AC 도 ViT-H — 릴리즈 AC 는 ViT-g). 실행은 사용자, 설계 · 판정 기준은 Claude (`z_AC_training/README.md`).
- 어느 쪽이어도: (b) > (a) 면 subsection 의 문장이 선다. (b) ≈ (a) 면 그대로 보고하고 본 논지 (P1–P3 · 목표) 는 영향받지 않는다 — 그래서 subsection 크기가 맞다.

---

## 3. 옛 스토리에서 바뀌는 것 · 남는 것

| 항목 | 옛 (출처) | 새 (2026-10-08) |
|---|---|---|
| 표제어 | "state 를 이어가지 않는다" (09-22) → "예측의 자리가 비어 있다 · world knowledge 는 predictor 에" (09-29) | **"intuitive physics 를 따르는 plausible future prediction"** — 옛 역할 분리 논지는 P3 마지막 문장 ("encoder 는 충분, predictor 가 실패") 로 흡수 |
| 인간 · 영유아 비유 | 첫 문장과 discussion 에만 (DIRECTION §5) | **P1 전체 (intro 첫 문단).** 본문 판정은 측정 가능한 정의로 |
| 물리 장면의 지위 | "측정 도구 — 물리 엔진을 가졌는가 · 중력을 아는가를 묻지 않는다" (09-19) | intuitive physics 가 표제가 된다. **그래도 판정은 행동** ("낙하하는 미래를 예측하는가") 이고 내부 귀속 ("중력을 안다 / 법칙을 적용한다") 은 계속 금지 |
| 네 능력 (읽기 · 지속 · 전이 · 영속) | state evolution 의 정의 (09-19) | **intuitive physics 의 조작적 정의로 유지.** 대응: 보이는 것 외삽 = 지속 · reach / 안 보이는 것 = 영속 / 없는 것 = 전이 3a 추세 (가속 · 꼭대기) + 3b 사건 (낙하 · 충돌) / encoder 정보 = 읽기 |
| ledge · wall · 꼭대기 (사건별 변화) | common sense 라 정의 밖 (09-19) | ✅ **포함 (사용자 10-08, 09-19 결정 번복).** 3 전이를 3a 추세 (가속 · 꼭대기) / 3b 사건 (낙하 · 충돌) 로 나눠 읽는다. 넣는 조건 넷 (복사와 닮은 미래 · 갈림 시점 누수 · 자 둘의 다른 답 · L1 교란) 은 §2-3 (iii) |
| memory · state 프레임 | 금지 (09-30, RETHINK §0-c) | **유지.** "안 보이는 것을 못 한다" 를 기억 문제로 쓰지 않는다 — 가려지기 전 프레임이 encoder 창 안에 있어도 절반, 가림 없이도 가속을 안 넣는다 |
| 벤치마크 감사 (복사 기준선) | 독립 beat · 기여 (DIRECTION §2 · RETHINK §5) | **P2 의 근거 — "IntPhys 1 은 leakage 가 있는 벤치"** (복사 = leakage 의 측정) + 방법의 판정 규칙 |
| 주 평가 | EK100 · IntPhys 2 · Ego4D · EgoExo4D (DIRECTION §1) | ⚠️ **결정 대기 2** — 기본값: **intuitive physics 판 (IntPhys 1/2 · GRASP · InfLevel + testbed 의 copy-hard · 위치 쌍) 이 1 차**, EK100 등 world model downstream 은 "떨어지지 않는다 / 오른다" 2 차 |
| action conditioning | "그다음" (DIRECTION §1) | **작은 subsection + 실험 하나** (사용자 10-08) — "AC 는 post-training" 을 scratch 대조로 (§2-5) |
| "frozen 위에서 닫힌다" | 08-31 원칙 → 09-06 폐기 | 폐기 유지 (목표가 predictor 학습) |
| 목적함수를 원인으로 | discussion 한 문단 | 유지 |
| 분업 | 분석 = Claude, 학습 = 사용자 (DIRECTION §0) | 유지 |

**결정 (사용자 10-08)** — 1. ledge · wall · 꼭대기 **포함** ✅ · 3. AC 는 **작은 subsection + 실험 하나** ✅.
**결정 대기** — 2. 주 평가를 intuitive physics 판으로 옮기는가 (EK100 · Ego4D 의 자리). 문서는 기본값 (intuitive physics 판 1 차) 으로 썼다.

---

## 4. 리뷰어가 칠 곳 (단서는 결론과 같은 무게)

| # | 공격 | 답 · 남는 단서 |
|---|---|---|
| 4-1 | "Garrido 는 창발했다고 했다 (IntPhys 1 88.89)" | **IntPhys 1 은 leakage 가 있는 벤치다** — 복사만으로 85.0 (Δ CI 0 포함), DINO-F 는 복사가 더 높다, 위반이 외형 사건뿐이고 이동 물체 가림은 0.2 초. leakage 가 작은 IntPhys 2 에서는 복사 수준 (§2-2). ⚠️ 반박 대상은 "창발" 자체가 아니라 **그 근거로 쓰인 점수가 예측을 요구하지 않는다** 는 것. leakage 를 복사로 잰다는 정의를 본문에 먼저 |
| 4-2 | "V-JEPA 2 하나 아닌가 — world model 일반이라 할 수 있나" | 우리가 잰 것: V-JEPA 2 ViT-H/L 릴리즈 · Ariel 세 팔 · post-FT 넷 · 사전학습부터 다시 한 8 개 (`TrainingEffects` 그룹 B) · DINO-Foresight (다른 encoder + frame predictor, v11 가림에서 같은 실패). ⚠️ **픽셀 생성형 world model 은 안 쟀다** → P2 의 "world model" 은 latent 예측형으로 좁히거나 하나를 더 잰다 (§5-2) |
| 4-3 | "문맥 끝 가림은 corner case" (09-03 외부 피드백) | 실영상: 움직이는 내용의 약 절반이 reach 밖 (SSv2 46 · EK100 51 %), 실영상 가림 띠에서도 절반 복원. 벤치마크는 이 상황을 거의 안 낸다 (IntPhys 1 이동 물체 가림 0.2 초, `OcclusionStats`). ⚠️ "context 에서 안 보이면" 은 **마지막 관측** 으로 좁혀야 측정과 맞는다 (§1) |
| 4-4 | "ledge 0 % 는 L1 이 덜 움직이는 미래를 싸게 보는 것" · "복사도 같은 쪽이다 (H23 §3-7)" | 복사도 같은 쪽인 것은 P3 ("보이는 것만 외삽") 와 맞는 방향이다 — 다만 그림은 불가능 미래가 복사와 다른 방향으로 움직이는 다중 미래 설계로 낸다 (§2-3 iii 조건 1).  위치 자 (RollOutV2) 가 독립적으로 선반 높이 유지; gravity 에서 가장 덜 움직이는 '멈춤' 미래를 고르는 block 은 0.7 % (1/144, 고른 비율이지 쌍 정확도가 아니다). ⚠️ 그래도 ledge VoE 단독으로는 못 가른다 (`v11_realistic` §2) |
| 4-5 | "encoder 충분성은 지도 자의 산물" | z 자 ≈ h 자, 파라미터 없는 교차검증, 안 본 속도 · 가림 문맥에서도. ⚠️ "encoder 가 미래를 표현한다" 는 안 쓴다 (§2-3 iv) |
| 4-6 | "inpainter 가 예측 못 하는 건 당연" | 벤치마크가 그것을 world model 로 인증했고 (P2), 미래 전용 · 되먹임 목표로 다시 학습해도 같다 (§2-3 v) |
| 4-7 | "관측만으로 학습하는 건 너희 Ariel 이 이미 했고 실패했다" | **반드시 넘는다 (사용자).** 합격선 표와 "Ariel 이 왜 실패했나 → 설계 후보" 표 (§2-4). 방법 절 첫 줄이 그 차이다. 사건 (3b) 을 겨누는 후보는 아직 없다 |
| 4-8 | "AC 가 post 여야 한다는 근거?" | subsection 실험 하나 (scratch vs post × 데이터 양, §2-5). 결과가 같으면 그대로 보고 |
| 4-9 | "intuitive physics 의 정의?" | 행동 정의 (§3 대응표). P1 의 두 예시가 영속 · 중력과 1:1 |

---

## 5. 다음 할 것 (문단 순)

| 순 | 무엇 | 세우는 곳 | 비용 | 담당 |
|---|---|---|---|---|
| 1 | **Figure 1** — 가림 + 선반 두 패널, "후보 미래 중 어디에 가까운가" 형식 (§2-2) | P2 hook | CPU, 기존 산출물 | Claude |
| 2 | **leakage 그림** — GRASP level2 · InfLevel 세 판에 복사 기준선 (Garrido 격자, 같은 창) → 벤치마크 · 칸마다 (복사, 모델) 한 점 | P2 | GPU 소량 (작은 세트) | Claude |
| 3 | **Ariel 세 팔 (prefix ep45 · full ep40 · AR ep40) 을 v11_realistic · ledge · gravity_realistic 에** — 합격선 표의 ★ 를 채운다 (`z_training/eval.sh`, AR 은 `ar_scoring`) | 목표 합격선 | GPU 소량 | Claude |
| 4 | **ledge · wall 다중 미래 세트** — gravity_realistic 형식: ledge 미래 = 낙하 (가능) · 부유 · 상승 · 멈춤, wall 미래 = 정지 (가능) · 통과 · 튕김 · 멈춤-전. 갈림은 예측 구간 안쪽 (§2-3 iii 조건 1 · 2) | P3 3b · Figure 1 | 렌더 (UnrealEngine) + CPU | Claude (설계) · 렌더 |
| 5 | **"없는 정보" 축 한 표** — 가속 · 꼭대기 (3a) 와 낙하 · 벽 (3b) 를 같은 판 (위치 자 + 후보 미래 + 파라미터 0) 으로 | P3 (iii) | CPU 대부분 | Claude |
| 6 | **P2 일반성** — latent 예측형 하나 더 (V-JEPA 2.1 · DINO-world 류) 또는 생성형 하나를 v11 late 가림 · ledge · gravity 에 복사 기준선과 같이 | P2 · 4-2 | GPU, 모델 로더 | Claude (모델 선택은 사용자) |
| 7 | **corner case 방어** — `OcclusionStats` 를 IntPhys 2 · GRASP · InfLevel 로 (가림 시간 × 가려진 동안 이동) | 4-3 | CPU + 추적기 | Claude |
| 8 | **관측 학습 predictor** — §2-4 합격선을 넘는 것, Ariel 과의 차이가 첫 줄 | 목표 | 학습 | **사용자** |
| 9 | **AC subsection 실험** — scratch vs post × 데이터 10/30/100 % (§2-5). 데이터 · encoder 결정 | AC | 설계 → 학습 | Claude (설계) · 사용자 (실행) |
| 10 | 문헌 확인 — P1 (동물 · 적응), P2 (world model 발전 · IntPhys 2 · 생성형 물리 벤치마크 · VoE leakage 선행) | P1 · P2 | 문헌 | Claude |

---

## 6. 하네스 — 실험을 제안 · 수락할 때 (CLAUDE.md §12 가 이 목록을 가리킨다)

1. **어느 문단 (P1 · P2 · P3 · 목표 · AC) 의 어느 문장을 세우거나 무너뜨리나.** 못 대면 그렇게 말하고 돌리지 않는다.
2. **결과가 어느 쪽이어도 문장이 서나.** 한쪽만이면 확증편향이다.
3. **이미 답이 있나** — CLAUDE.md §6 기각표, `auto_research/README.md` §2 철회표, 이 문서 §3 · §4.
4. **"보이는 것 / 안 보이는 것 / 없는 것" 중 무엇을 재나.** 셋을 섞은 평균으로 결론 내지 않는다 (마지막 관측 기준).
5. **복사 기준선을 같이 내나.** VoE 점수는 예측 없이 풀릴 수 있다 — 모든 점수에 복사 대비 Δ (가능하면 copy-hard · 위치 쌍). metric 이 위치를 못 보는 교란 (`COPY_VS_POSITION`) 이 남으면 못 하는 주장을 미리 적는다.
6. **학습이면**: action 없이 관측만인가 · 학습 도메인 밖에서 재나 (궤적 외우기 구분) · 이미 실패한 Ariel · post-FT 와 무엇이 다른가.
7. **표본이 결론을 낼 크기인가** (v11: 조건 × 위반 672 쌍 / k 168 / 방향당 16; v11_realistic: k 칸 15 쌍).
8. **학습된 자를 쓰면** held-out 정밀도 · encoder 대조 · 파라미터 없는 교차검증을 같이 낸다 (자는 대상이 없어도 기본값을 낸다).
9. **합성에서만 서는가.** corner case 인지, 실영상 · 벤치마크에서 얼마나 흔한지, 어느 use case (planning · 벤치마크 갭 · 평가 방법론) 에 닿는지를 같이 말한다.

> **동의만 하고 돌리는 것이 가장 큰 실패다.** 반대로 사용자가 이유를 대면 그건 결정이다 — 한 번 말하고 진행한다.

---

## 7. 금지 표현 (갱신)

| 쓰지 말 것 | 왜 · 대신 |
|---|---|
| "world model 은 물리를 이해하지 못한다" · "intuitive physics 가 없다" (조건 없이) | 행동만 쟀다. → "**마지막 관측에 없는 X 를 따르는 미래를 예측하지 못한다**" (조건 명시) / "배웠다고 보기 어렵다" (근거와 함께) |
| "직관물리는 창발하지 않았다" | 반박 대상은 그 측정이다. → "창발 근거인 IntPhys 1 은 leakage 가 크다 (예측 없이 85 %)" |
| "IntPhys 1 은 쓸모없다 / 틀린 벤치다" | 잰 것은 leakage 의 크기다. → "예측 없이 85 % 가 풀린다" (복사 기준선으로 정의하고) |
| "IntPhys 2 에서 chance" | V-JEPA 2-h 57.51 (논문). 우리 측정은 → "복사와 구분되지 않는다" |
| "모델이 중력을 안다 / 모른다 / 물리 법칙을 적용한다" | 내부 귀속. → "낙하하는 미래를 예측하지 않는다" |
| "predictor 에 정보가 없다" / "X 를 못 만든다" | p probe 98–100. → "맞는 미래로 놓지 않는다" |
| "encoder 가 미래를 표현한다" | 미래가 현재의 결정적 함수인 데이터다 (`FUTURE_FROM_CONTEXT` §3) |
| "기억 (memory) 문제다" · "state 를 잃는다" | 09-30 사용자. 정보는 남아 있고, 기억이 필요 없는 곳에서도 가속을 안 넣는다 |
| "context 에서 안 보이면 못 한다" (전부로) | 문맥 가운데 가림은 처리한다 (early 97.8 / mid 95.8). → "**마지막 관측에** 안 보이면" |
| "action-free 학습으로 해결된다" | Ariel · post-FT 반례 (§2-3 v). 방법은 차이를 대야 한다 |
| "AC 는 post 여야 효과가 있다" (사실로) | subsection 실험 (§2-5) 결과가 나오기 전까지 가설 |
| "ledge 에서 불가능한 연속을 예측한다" | 떠 가는 미래는 복사와 닮았다 (H23 §3-7). → "낙하하는 미래를 만들지 않는다" (다중 미래 설계로) |
| "post-FT 가 intuitive physics 를 배웠다" | 도메인 안 궤적 prior 다 |
| "가림이 표현을 망가뜨린다" | env 72 칸 100 |
| "목적함수 때문이다" | discussion 한 문단까지 |
| "p 가 물체를 마지막 자리에 둔다" · "슬롯 N 이후 멈춘다" | 철회 — 자의 기본값 |
| α · 보정 · 헤지 계수 | 인용 금지 (사용자 지시 전까지) |

---

## 8. 문서 지도

| 무엇 | 어디 | 지위 |
|---|---|---|
| **이 문서** | `auto_research/paper/PAPER_STORY_2026-10-08.md` | **정본** |
| 결과 로그 · 철회표 | `auto_research/README.md` | 현행 (스토리 머리말만 이 문서로) |
| 네 능력 정의 | `z_research/context_encoder_analysis/Archive/STATE_EVOLUTION_CAPABILITIES_2026-09-19.md` | 현행 (intuitive physics 의 조작적 정의로 읽는다; 3 전이 = 3a 추세 + 3b 사건, 10-08 번복) |
| 방법 설계 후보 | `auto_research/paper/DESIGN_CHOICES_2026-09-25.md` §3-b–d | 현행 (설계안, 미검증). 스토리 부분은 대체됨 |
| 옛 방향 · 논지 | `DIRECTION_2026-09-25` · `RETHINK_PREDICTOR_ROLE_2026-09-29` · `NARRATIVE_v2_USER_2026-09-29` · `ABSTRACT_v0/v1` · `STORY_EVIDENCE_2026-09-26` | 대체됨 — 근거 표 · 리뷰 대응 · 분업은 유효 |
| 옛 스토리 (z_research) | `IntPhysGenV11/Archive/PAPER_STORY_2026-09-22` · `-09-06` · `-08-31` · `IntPhysGenV10/Archive/PAPER_STORY_2026-08-29` | 대체됨 — 수치 · 금지 표현은 유효 |
| 학습 하네스 | `z_training/README.md` (predictor) · `z_AC_training/README.md` (AC) | 현행 |
| v11 세트 | `z_research/IntPhysGenV11/README.md` | 현행 (testbed) |

---

## 재현

이 문서는 논리 문서다. 새로 계산한 수치는 없고 전부 아래 출처에서 옮겼다. ✓ 는 2026-10-08 에 산출물에서 다시 읽거나 다시 센 것이다 (아래 두 번째 표).

| 수치 | 출처 (그 문서의 `## 재현`) |
|---|---|
| v11 73.37 · vanish A/B 0/100 · shape 94.0/89.9/69.0 | `z_research/IntPhysGenV11/exp_results/report.json` (CLAUDE.md §5-1) |
| p probe 98.46 / 99.49 / 100, 이식 17.3 (세 운동 팔 평균 — ramp 단독 16.4) | `z_research/IntPhysGenV11/exp_results/attn_probe__v11_vith/summary.json` (`per_group` 에서 읽는다. `overall` 은 6 조건 평균이라 인용 금지) |
| v11_realistic 79.6, ledge 0 / 120, gravity 표 | `z_research/v11_realistic/exp_results/tables.json` · `gravity_retrieval.json` |
| IntPhys 1 88.89 vs 복사 85.00 | `z_research/TrainingEffects/ip1_copy/IP1_COPY.md` · `auto_research/Archive/H6_*` |
| IntPhys 2 54.3 vs 55.5 | `auto_research/Archive/INTPHYS2_PERMANENCE_AUDIT_2026-09-26.md` |
| DINO-F 84.4 / 88.9–92.2 · 77.1 / 76.4 | `auto_research/Archive/DINOF_INTPHYS1_2026-10-02.md` · `DINOF_V11_2026-10-07.md` |
| 위치 쌍 0.0 / 17.6 vs 7.9 / 2.3 | `auto_research/exp_results/scene/rollout2_copy_voe.json` (`COPY_VS_POSITION`) |
| reach 2–3 칸 · late 0 슬롯 · flat_a 28/32/54 | `z_research/RollOutV2/Archive/PREDICTOR_DISTANCE_LIMIT_2026-09-19.md` · `RollOutV2/README.md` §4b |
| 실영상 46 / 51 %, 가림 띠 .21 / .42 | `auto_research/Archive/I_SCENE_CAUSAL_BOTTLENECK_2026-09-25.md` §4-3 · §4-4 |
| z 자 5–10 px · p 자 12–30 px | `auto_research/Archive/FUTURE_FROM_CONTEXT_2026-10-02.md` §2 |
| post-FT 91.4 / 77.22 · 93.89 / 78.5 | `z_research/TrainingEffects/scores/SCORES.md` |
| AR α −0.42 / −0.46 | `auto_research/Archive/AR40_READOUT_2026-09-30.md` |

### 2026-10-08 산출물 재확인 (vll3, 읽기 전용)

| 수치 | 확인 | 방법 · 단서 |
|---|---|---|
| v11 73.37 (10,752 쌍, tie 0) · ramp 가림 vanish 50.0 = 물체→빈 0.0 / 빈→물체 100.0 · shape 69.05 vs 94.05 | ✓ | `surprise_c16t32__v11_vith/per_block.json` 의 `per_video_surprise` + `data_csv/intphysgen_v11/index.csv` 로 matched 쌍을 다시 셌다 |
| v11_realistic 79.58 (720) · ledge 0 / 120 (tie 0) | ✓ | `v11_realistic/exp_results/tables.json` + 두 `summary.json` |
| gravity 표 (n 48 / 조건, stop 1/144) | ✓ | `gravity_realistic` `per_block.json` 에서 미래 다섯의 argmin 을 다시 셌다. stop 0.7 은 고른 비율 (arc vs stop 쌍 정확도는 98.6) |
| IntPhys 1 88.89 vs 복사 85.0 [79.4, 90.0] | ✓ (복사는 파생 json 두 개 일치) | release `Benchmarks/exp_results/intphys1_sliding__intphys1_dev_vith_w32/summary.json`; 복사 `TrainingEffects/ip1_copy/ip1_copy.json` · `auto_research/exp_results/h6/h6g_intphys1.json`. 원 배열 (`/data2/.../ip1score_final/arrays_v360.npz`) 은 vll3 에 없다. 인과 표적 · LN 없음이면 복사 77.8 |
| IntPhys 2 54.35 (C 24) vs 복사 55.53 (C 42) | ✓ | `Benchmarks/exp_results/intphys2/bench_intphys2_main_vith{,_copy_w48}/per_video.csv` 에서 (scene, pair) 로 다시 셌다 |
| 위치 쌍 0.0 / 17.6 vs 7.9 / 2.3 | ✓ json 만 | `auto_research/exp_results/scene/rollout2_copy_voe.json`. 원 캐시는 vll5 `/local_datasets` |
| DINO-F v11 77.10 vs 복사 76.44 | ✓ | `IntPhysGenV11/exp_results/surprise_c16t32__v11_dinof_highres{,_copy}/per_block.json` 에서 다시 셌다 |
| p probe 98.46 / 99.49 / 100 · 이식 z 97.41 / h 99.15 / p 17.32 | ✓ | `per_group['moving_occlusion']` (self). 이식은 세 팔 평균 (p 15.0 / 20.6 / 16.4) |
| post-FT v11 91.35 (릴리즈 75.83) · IntPhys 1 77.22 · IntPhys 1 post-FT 93.89 / v11 78.53 | ✓ | `z_training/runs/{v11,intphys1}_postft/eval/*/summary.json` · `Benchmarks/exp_results/intphys1_sliding__intphys1_dev_vith_pft_*_w32/summary.json`. 릴리즈 75.83 은 `index_test.csv` 로 다시 셌다 |

확인하지 않은 것 (문서 값): reach 2–3 칸 · late 0 슬롯 · flat_a 28/32/54 · 실영상 46 / 51 % · 가림 띠 .21 / .42 · z/p 자 px · AR α · ego-motion R² · IntPhys 2 논문 57.51.
