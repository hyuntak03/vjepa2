# 00 — 지금의 이야기 · 스토리 변천 · 사용자 결정 · 데이터 세대 연표

> **결론** — 지금의 이야기는 2026-10-08 사용자 정본 하나다: 사람은 관측만으로 intuitive physics 를 갖춘다 (P1) → world model 은 그것을 따르는 미래를 예측하지 못하고, 그 반례처럼 보이는 IntPhys 1 은 leakage 가 큰 벤치다 (P2) → 통제 testbed 에서 V-JEPA 2 는 마지막 관측에 **보이는 것만** 짧게 외삽하고 **안 보이는 것 · 없는 것** 은 예측하지 못하며, 정보는 encoder 에 있고 실패는 predictor 에 있다 (P3) → action 없이 관측만으로 배우는 predictor 가 목표이고 Ariel 을 넘어야 한다. **AC 는 그 predictor 위의 post-training 이고 논문의 작은 subsection 이다 — 스토리는 쓰되 아직 최종이 아니다 (사용자 2026-10-10, `09b_AC_STORY.md`).**
> **근거** — (1) 08-13 ~ 10-08 사이 이야기의 틀은 열 번 넘게 바뀌었지만 (§2), 바뀐 것은 틀과 목표이고 **측정 수치 · 정정 · 기각표는 그대로 유효하다** (`auto_research/paper/PAPER_STORY_2026-10-08.md` 머리).
> (2) 틀을 바꾼 것은 대부분 결과였다 — knockout `mask→mask` −1.4 (09-06), 위치 자 기본값 철회 (09-12), 거리 한계 2–3 칸 (09-19), SC1 "학습 지평" 기각 (09-25), IntPhys 1 복사 85.00 vs 88.89 (09-24 H6).
> (3) 나머지는 사용자 결정이다 — 네 능력 (09-19), memory · state 프레임 금지 (09-30), 10-08 후속 결정 넷, AC 는 "아직 최종 아님 — 스토리는 쓴다" (10-10) (§3).
> **미결** — 주 평가를 intuitive physics 판으로 옮기는가 (정본 §3 "결정 대기 2") 는 아직 사용자 결정 전이다. 10-10 AC 지시 ("아직 최종 아님") 는 레포 문서에 아직 없다 (§1-4).
> 이 장이 받치는 논문 문장: 전부 (P1 · P2 · P3 · 목표) — 이 장은 지도다  ·  원재료: G05 · G06 (c) · G07 · G12 (c) · G13 · G03 (`auto_research/README.md`) · G02a (`README_UNTIL`) 추출문 + 정본 직접 확인 → 원 문서는 각 절 끝 `출처`.

---

## 1. 지금의 이야기 한 장 (정본 2026-10-08 + 2026-10-10 AC 지시)

### 1-1. 사용자 원문 (그대로, 정본 §0)

> **P1 : 사람은 intuitive physics 덕분에 plausible future를 쉽게 상상, intuitive physics는 observation 갖춰짐**
> Physical world에서, 사람은 앞으로 어떤 일이 일어날지, 물리적으로 가능한 미래들을 쉽게 상상할 수 있다.
> 이것은, (물체가 갑자기 사라지지 않는다, .. 중력이 있다) 와 같은 intuitive physics를 알고 있기 때문이다.
> 이런 intuitive physics는 영유아, 동물들에게 가장 초기에 observation만으로 갖춰지는 능력이며,
> 이 능력 덕분에 intelligent system은 새로운 환경이나 task에 대해 빨리 적응하고 이해할 수 있다.
>
> **P2 : world model은 발전 빠르게, 근데 intuitive physics 못한다.**
> World model은 physical world의 dynamics를 배우고자 하고, action planning, prediction, action anticipation, ,. Etc 등에서 빠르게 발전하고 있다.
> 하지만 physical world의 dynamics를 이해한 것의 가장 기초적이고 근본적인 intuitive physics 못한다.
> Figure 1 예시 hook
> 사람은 쉽게 직관적으로 이런 예시에서 가능한 미래를 상상할 수 있다. ← world model과 Human 간의 gap 이 있다.
>
> **P3 : To understand where world models fail, we build a controlled synthetic testbed.**
> 정확히 어디서 무너지는지 파악하기 위해 testbed 제작
> 봤더니, 지금은 context에서 보이는것만 외삽하지, context에서 안보이거나 없는 정보는 못함 ⇒ physical world의 intuitive physics 배우고 이해했다고 보기 어렵
> encoder는 정보 충분한데, 단순히 predictor가 intuitive physics에 따르는 plausible future prediction 하는 것에서 실패
>
> **목표 (사용자: "어떻게 될 지는 모르겠지만 아래와 같은 것을 하고 싶음")**
> Action conditioned 없이, 순수 observation 만으로 intuitive physics를 배우고 ⇒ 이에 따른 plausible future prediction 을 잘하는 predictor를 만들고 싶음
> (1)이 선행되어야 action conditioned 학습도 효과가 있다. {scratch 부터 action 학습하는게 아니라, action-conditioned은 post 가 되어야 한다.}

⚠️ 원문의 목표 둘째 줄 ("(1)이 선행되어야 …") 은 **아직 최종이 아니다** (사용자 2026-10-10: "아직 AC가 우리가 도달할 이상은 아니야" → 같은 날 풀이 "이게 아직 최종이 아니라는거라서 AC 스토리도 일단 작성을 해줘야"). AC 스토리 초안은 `09b_AC_STORY.md` 에 있고, 주장은 실험 전까지 가설이다.

### 1-2. 한 문장 (정본 §1)

> **사람은 관측만으로 intuitive physics 를 갖추고 그것으로 plausible future 를 상상한다.
> 지금의 world model 은 마지막 관측에 보이는 것만 외삽하고, 거기 안 보이는 것 (가려진 물체) 과 거기 없는 것 (아직 일어나지 않은 낙하 · 되돌아옴 · 가속의 누적) 은 예측하지 못한다.
> 필요한 정보는 encoder 에 있으므로 실패는 predictor 의 미래 예측에 있다.
> 우리는 action 없이 관측만으로 intuitive physics 를 따르는 미래를 예측하는 predictor 를 만들고, action conditioning 은 그 위의 post-training 으로 둔다.**

- 마지막 문장의 뒤 절 ("action conditioning 은 그 위의 post-training") 은 **아직 최종이 아니다** (10-10) — 작은 subsection 의 스토리 초안으로 `09b_AC_STORY.md` 에 둔다.
- 정본의 좁힘: "context 에서 보이는 것" 은 **마지막 관측 (문맥 끝)** 으로 좁혀 쓴다 — 문맥 가운데에서 가려졌다 끝에 다시 보이는 물체는 처리한다 (v11_full vanish 채점 early 97.8 / mid 95.8 / late 57.5, ViT-H `surprise_c16t32`, 확정) (`auto_research/paper/PAPER_STORY_2026-10-08.md` §1 · `z_research/IntPhysGenV11/Archive/STATUS_AND_CRITIQUE_2026-09-03.md` §1-1).

### 1-3. 문단별 근거 상태 → 이 아카이브의 장 (정본 §2 표 + 장 배정)

| # | 문단 | 세우는 문장 | 근거 상태 (정본 등급) | 정본이 가리키는 근거 | 이 아카이브 |
|---|---|---|---|---|---|
| P1 | 사람 · 영유아 · 동물 | intuitive physics (사라지지 않음 · 중력) 는 관측만으로 가장 먼저 갖춰지고, 새 환경 · task 적응의 바탕이다 | **문헌** — 인용 확인 필요. "동물" · "적응" 은 인용 없음 | `NARRATIVE_v2` §A 인용 목록 | `01_P1_HUMAN_INTUITIVE_PHYSICS.md` |
| P2 | world model 의 gap | world model 은 빠르게 발전하지만 intuitive physics 를 따르는 미래는 예측하지 못한다. IntPhys 1 은 leakage 가 있는 벤치 (복사 기준선으로 잰다). Figure 1 hook | 문헌 + 측정 — **1 차 ~ 확정 혼재** | `PREDICTOR_SHARE` · `INTPHYS2_PERMANENCE_AUDIT` · `H6` · `DINOF_*` · `COPY_VS_POSITION` | `02_P2_WORLD_MODELS_AND_LEAKAGE.md` |
| P3 testbed | 무엇으로 쟀나 | v11 · v11_realistic · gravity_realistic · RollOut_v2/v3 · 표준 채점 · 자 · 네 능력 | 설계 · 규약 | CLAUDE.md §1 · §5 | `03_TESTBED_AND_MEASUREMENT.md` |
| P3 (i) | 보이는 것 | 1 차 운동까지, 짧게 (마지막 관측에서 2–3 칸) 외삽한다 | **확정 (합성) · 1 차 (실영상)** | RollOutV2 §5-5 · `I_SCENE` §4-3 | `04_P3_VISIBLE_EXTRAPOLATION.md` |
| P3 (ii) | 안 보이는 것 | 문맥 끝 가림이면 vanish '물체→빈' 0 %, k=1–4 평평, p 미래에 처음부터 물체다운 토큰이 없다 | **확정 (합성)** · 실영상 가림 띠 확정 | CLAUDE.md §5-1 · §5-5 · `v11_realistic` §1 | `05_P3_NOT_VISIBLE_OCCLUSION.md` |
| P3 (iii) | 없는 것 | 3a 추세 (가속 · 꼭대기) · 3b 사건 (낙하 · 벽) 을 미래에 넣지 않는다 | 확정 · 1 차 혼재. 넣는 조건 넷 | `STATE_EVOLUTION_CAPABILITIES` §2 · `v11_realistic` §2 · §4 · `AR40_READOUT` | `06_P3_ABSENT_ACCEL_AND_EVENTS.md` |
| P3 (iv) | encoder 정보 | 정보는 encoder 에 있고 predictor 가 그것을 미래로 옮기지 않는다 | 확정 (속도 R²) · 1 차 (`FUTURE_FROM_CONTEXT`) | `FUTURE_FROM_CONTEXT` · CLAUDE.md §5-2 · `ENCODER_MOTION_READOUT` | `07_P3_ENCODER_HAS_INFO_PREDICTOR_FAILS.md` |
| P3 (v) | 다시 학습해도 | Ariel 세 팔 · post-FT 넷 모두 도메인 안에서만 오른다 | 확정 (`AR_PREDICTOR`) · 문서 값 (`TrainingEffects`) | `TrainingEffects` §0 · `AR_PREDICTOR` | `08_RETRAINING_DOES_NOT_FIX.md` |
| 목표 | 방법 | action 없이 관측만으로 intuitive physics 를 배우는 predictor — **Ariel 을 넘어야 한다** | **★ 미실행 (사용자 학습 항목)** · 합격선은 정본 §2-4 | `DESIGN_CHOICES` §3-b–d · `TrainingEffects` · `AR_PREDICTOR` | `09_GOAL_OBSERVATION_ONLY_PREDICTOR.md` |
| AC | 작은 subsection | 정본 §2-5 의 "작은 subsection, 실험 하나" — **스토리 초안, 아직 최종 아님 (10-10)** | ★ 근거 0, 실험 전까지 가설 | `z_AC_training/README.md` | `09b_AC_STORY.md` |
| — | 기각 · 철회 | 다시 쓰지 말 문장 | — | CLAUDE.md §6 · `auto_research/README.md` §2 | `10_REJECTED_AND_RETRACTED.md` |
| — | 미결 · 다음 | 결정 대기 · 리뷰어 공격 · 다음 할 것 | — | 정본 §4 · §5 | `11_OPEN_ISSUES_AND_NEXT.md` |

⚠️ 표의 "근거 상태" 는 정본의 등급이다. 정본은 등급을 **확정** (적대 검증 통과) · **1 차** (검증 전) · **문헌** · **★** (미실행) 넷으로 쓴다 (정본 머리).
⚠️ 정본 §2 표의 AC 행은 "★ 실험 하나로 세운다" 였다. 이 아카이브는 그 행을 "스토리 초안 · 최종 아님" 으로 읽는다 (§1-4).

### 1-4. AC 의 자리 — 10-08 안과 10-10 "아직 최종 아님"

| 날짜 | 무엇이 적혔나 | 출처 |
|---|---|---|
| 10-08 (사용자) | "이건 이제 자그많게 subsection 느끼믕로 실험 하나 넣을거라 괜찬ㅇ하." → 정본 §2-5: (a) AC scratch (V-JEPA 2-AC 레시피, `load_predictor: false`) vs (b) 관측 학습 predictor 위 AC post-training, AC 데이터 10 / 30 / 100 %; 판정 = held-out 행동 조건부 rollout 오차 + intuitive physics 판 유지 | `auto_research/paper/PAPER_STORY_2026-10-08.md` §0 · §2-5 · `z_AC_training/README.md` 머리 |
| 10-10 (사용자) | "아직 AC가 우리가 도달할 이상은 아니야 그걸 명심해." → 풀이 (같은 날): "AC 보류는 뭔 말이냐면 이게 아직 최종이 아니라는거라서 AC 스토리도 일단 작성을 해줘야". → AC 스토리는 쓴다 (`09b_AC_STORY.md`), 설계 · 결정은 바뀔 수 있다. 주 목표 (관측 학습 predictor) 보다 앞에 두지 않는다 | 이 아카이브 작업 중 사용자 지시 (레포 문서 미반영 — 정본 §2-5 · §5 #9 · CLAUDE.md §0 · §10 #9 는 10-08 판 그대로) |

⚠️ **레포 문서와 이 아카이브 사이의 불일치를 기록한다**: 정본 §5 표 9 행 · CLAUDE.md §10 9 행은 아직 "AC subsection 실험 — scratch vs post × 10/30/100 %" 를 "다음 할 것" 으로 둔다. 이 아카이브는 그 행을 그대로 두되 "초안 · 최종 아님" 을 단다 (11 장 §5). 정본을 고치는 것은 이 아카이브의 범위 밖이다.

### 1-5. 결정 · 결정 대기 (정본 §3, 10-08 기준)

| # | 항목 | 상태 |
|---|---|---|
| 1 | ledge · wall · 꼭대기 사건을 논문에 넣는가 | ✅ **포함** (사용자 10-08, 09-19 "정의 밖" 번복) — 3 전이 = 3a 추세 + 3b 사건 |
| 2 | 주 평가를 intuitive physics 판 (IntPhys 1/2 · GRASP · InfLevel + testbed copy-hard · 위치 쌍) 으로 옮기는가 | ⚠️ **결정 대기** — 정본은 이것을 기본값으로 썼다. EK100 등 downstream 은 "떨어지지 않는다 / 오른다" 2 차 |
| 3 | AC | 10-08 ✅ "작은 subsection + 실험 하나" → **10-10 스토리 초안 · 아직 최종 아님** (§1-4, `09b_AC_STORY.md`) |

### 출처 (§1)
`auto_research/paper/PAPER_STORY_2026-10-08.md` §0 · §1 · §2 · §2-5 · §3 · §5 · `z_AC_training/README.md` 머리 blockquote · `z_research/IntPhysGenV11/Archive/STATUS_AND_CRITIQUE_2026-09-03.md` §1-1 · `CLAUDE.md` §0 · §10

---

## 2. 스토리 변천 (2026-07-03 → 2026-10-10)

### 2-1. 판별 변천 표 (가장 중요)

표의 "한 문장 틀" 은 그 판이 세운 표제 문장이다. 지금 쓰면 안 되는 표현은 `(원 문서 표현 — 현재 금지)` 를 달았다. "지금 남은 것" 은 정본 · CLAUDE.md 가 유효하다고 명시한 것만 적는다.

| 날짜 | 판 (문서) | 한 문장 틀 | 무엇을 버렸나 | 왜 (결과 · 결정) | 지금 남은 것 |
|---|---|---|---|---|---|
| 07-03 (전사) | `evals/analysis_vlm/modes/REPRODUCTION_PLAN.md` | V-JEPA 2-L encoder 의 물리 해석 (attention distance · ablation · orthogonal probe · steering) 재현 blueprint, Blender toy 데이터 | 세트 전체 — 결과 없이 끝남 | 2026-09-21 config 정리에서 `analysis_vlm` · `blender_toy_dataset` 를 `configs/_archive/` 로 | 없음 (기록) |
| 08-13 | `z_world_model_analysis/intphys1_dev_2026-08-13.md` · `occlusion_v2_2026-08-13_PREVIOUS-DATA.md` · `probe_set_…` | blender occlusion 에서 surprise 가 무너진다 — f (미래 내용) / g (예측 오차) 교란 발견, **미래일치 쌍이 공정** 이라고 권고. IntPhys 1 자체 재구성 0.6750 | 미래일치 쌍 선호 · 자체 IntPhys 재구성 · 재렌더 전 수치 | 08-15 영상 재렌더 (superseded) · 08-16 Garrido sliding 으로 교체 | f/g 교란이라는 관찰 → 문맥일치 쌍 규약의 동기 |
| 08-16 | `z_world_model_analysis/RESULTS_2026-08-16.md` | IntPhys 1 dev ViT-H **88.89** / ViT-L 64.44 (Garrido `skip2_w32`, 180 쌍); blender occlusion_v2 문맥일치 52.82 (presence = 100 / 0) — **문맥일치 2 쌍을 표준으로 확정** | 미래일치 쌍 · "모델 규모가 결정적" (blender v3 C=16 에서는 L→H +2.5 — `FRAME_ABLATION_2026-08-19.md`, 08-16 문서의 값 아님) | 문맥 픽셀 동일 → `p` 비트 동일 (max abs diff 0.00e+00) 이어야 채점이 기하 문제로 환원 | **88.89 는 지금도 정본 수치** · matched pairing (CLAUDE.md §1-2) · ViT-H 기본 (08-19) |
| 08-19 ~ 08-20 | `FRAME_ABLATION_2026-08-19` · `TOKEN_SUBSET_2026-08-19` · `MECHANISM_2026-08-20` · `WHY_INTPHYS_WORKS_2026-08-20` · `VERIFICATION_2026-08-20` · `OCCLUSION_LENGTH_2026-08-20` (`z_world_model_analysis/`) | 원인 후보 소거 + "정보는 p 에 있다" (IntPhysGen v1 층별 probe 68 → 86, proj 후 88.7); IntPhys 1 은 **내용량 누수** 로 풀린다 (corr 0.486 vs 우리 0.922); "가림 길이가 레버" | "p 에 정보가 없다" · "엉뚱한 자리에 그린다" · "IntPhys 물체 2 배" (각 철회) · context 길이 · 희석 · 배경 · 카메라 가설 (기각) · 가림 길이 레버 (v10 · v11 이 뒤집음) | 각 문서 안 정정 · v10 4→1 프레임 무관 (08-28) · v11 k=1–4 평평 (08-30) | CLAUDE.md §6 기각표의 첫 행들 · **IntPhys 1 위반이 외형 사건뿐 · 궤적 위반 0 건 · 180/180 magic 즉시 흔적** (→ 정본 §2-2 (a) leakage 기제) · α 분해 (**인용 금지**) |
| 08-26 | `z_world_model_analysis/PREDICTOR_HEDGING_2026-08-26.md` (v8) | α · 보정 · 헤지 계수 | — | 사용자 지시로 인용 금지 | 존재만 기록 |
| 08-28 ~ 08-29 | `z_research/IntPhysGenV10/Archive/V10_2026-08-28.md` · `GEOMETRY_2026-08-29.md` · **`PAPER_STORY_2026-08-29.md`** (12 beat "How Far is V-JEPA2 from a World Model?") | "V-JEPA2 가 가진 건 perception model 이지 world model 이 아니다" · "근본 원인은 목적함수다 — L1 의 최적해는 조건부 중앙값" · "물체 영속성도 없다" (원 문서 표현 — 현재 금지) | **목적함수 원인 축** → 08-31 discussion 한 문단 · 내부 귀속 (perception / world model) | 재학습 없이 검증되지 않는 원인을 축으로 두면 스스로 못 갚는 빚 (CLAUDE.md §0) | v10 수치: 가림 길이 4→1 무관 · ramp shape 67.58 vs flat 89.84 · `p` probe 98.96 · 균등 증폭 천장 51–65 (기각표 "세기를 키우면 된다") |
| 08-30 | `z_research/IntPhysGenV11/Archive/SYNTHESIS_2026-08-30.md` · `TALK_2026-08-30.md` · `surprising_score/RESULTS_2026-08-30.md` | v11 결과 사슬 ①–⑤; 발표 "표현은 있다. 굴리지 않는다. 그리고 그 격차는 목적함수에서 나온다" (원 문서 표현 — 현재 금지) | 목적함수 결론 (08-31 강등) · "동역학이 아니라 외형을 외삽한다" (내부 귀속) | 같은 이유 | **v11 73.37 표 · vanish A/B 0/100 · 방향 비대칭 순서 · probing p 98.46 / 99.49 / 100 · 이식 p 17.3** (CLAUDE.md §5-1 ~ §5-3, 전부 유효) |
| 08-31 | **`z_research/IntPhysGenV11/Archive/PAPER_STORY_2026-08-31.md`** (8 beat) | "정보는 남고 배치가 무너진다, 그 자리는 **frozen 상태에서 되돌릴 수 있다**" · "논문 전체가 frozen model 위에서 닫힌다" | (그 판이 버린 것) 08-29 의 목적함수 축 → discussion · "총점이 병목을 가린다" → IntPhys 1 vs 2 문헌 긴장 | — | beat 1 IntPhys 1 운동 × 가림 분해 (정지 100 / 100, 이동 91.67 / **75.0**) · beat 3 · 4 수치 · 금지 표현 표. ⚠️ beat 4 Luce 수치 ("cone 문맥에서 cube 77 %") 는 `_superseded/` |
| 09-03 | `z_research/IntPhysGenV11/Archive/STATUS_AND_CRITIQUE_2026-09-03.md` | 외부 비판 "event transition 에 occlusion 이 걸리는 것은 너무 corner case" 에 답 — 가림을 주인공에서 내리고 testbed 로; 후보 B "latent surprise 는 물리 검출기가 아니라 외형 일치 검출기" | 가림을 주인공으로 두는 판 | 무너지는 조건이 물체 × 사건 순간 × 고정 경계로 좁고 sliding 으로 상당 부분 회복 (v6aug 46.88 → 90.62) | **corner case / use case 판정 절차** (하네스 9 → 정본 §4-3) · v11_full timing 표 |
| 09-06 | **`z_research/IntPhysGenV11/Archive/PAPER_STORY_2026-09-06.md`** | "predictor 는 상태를 이어가지 않는다. 문맥을 조회할 뿐이다" (원 문서 표현 — state 프레임, 현재 금지) | **"frozen 위에서 닫힌다" 폐기** (beat 7 = predictor 재학습) · beat 5 부분공간 / Procrustes · beat 6 · 7 frozen 개입 | knockout (v11_full 21,504 쌍, 기준 75.86): `mask→ctx` −37.5 · `mask→mask` −1.4 · `ctx→ctx` +0.5; AC 가 `load_predictor: false` (당시 "암묵적 시인" 으로 읽음) | knockout 수치 · v11_full timing 87.2 / 79.0 / 78.2 / 59.5 · horizon t0 90.3 → t7 61.4 · 금지 추가 ("predictor 에 정보가 없다" · "직관물리는 창발하지 않았다" · "IntPhys 2 chance 근처") |
| 09-12 | `z_research/RollOutV2/figures/v5/summary/POSITION_READOUT_2026-09-12.md` | 위치 자 (3,842 파라미터, held-out 0.87 칸) 로 p 의 미래를 읽는다 | "predictor 가 물체를 마지막 관측 자리에 둔다" **철회** · "wall 에서 한 칸 반 파고들어 멈춘다" 정정 | 자의 기본값 (attention 이 퍼지면 토큰 평균 ≈ 마지막 관측 위치) 이었다 | **자 검증 규칙** (held-out · encoder 대조 · 파라미터 없는 교차검증, 사용자 09-12) · "가림이 경계에 걸리면 p 의 미래에 처음부터 물체다운 토큰이 없다" |
| 09-15 ~ 09-16 | `z_research/context_encoder_analysis/Archive/CLAIM_NO_STATE_EVOLUTION_2026-09-15.md` (v0 → v0.1 → v0.2) · `STATE_EVOLUTION_HYPOTHESES_2026-09-15.md` | "predictor 는 미래 토큰마다 문맥에서 마지막 관측 상태를 외삽한다 — 재귀 없이, 짧은 지평으로 (등속 평면 3–4 튜블릿, 가속 · 수직 · 선반 ≤ 1–2)" (state 프레임 — 현재 금지; 측정 사실은 유효); 축 R 재귀 · A 지평 · B 가림 · C 동역학 전환 | 구조 + 행동이 섞인 R/A/B/C 축 → 09-19 행동 네 능력 · v0.1 철회 다섯 ("슬롯 4 이후 물체 없음" 등) · 정렬 사상 "기각" → "부분 효과 (+3~6)" | 독립 검토 3 개 · n=300 재실행 · 사용자 09-19 결정 | 수치 (속도 OOF R² 0.988, 정렬 사상 +3.0 / +3.2 / +6.7) · 철회 기록 |
| 09-19 | **`z_research/context_encoder_analysis/Archive/STATE_EVOLUTION_CAPABILITIES_2026-09-19.md`** · `z_research/RollOutV2/Archive/PREDICTOR_DISTANCE_LIMIT_2026-09-19.md` | state evolution = **네 능력 (1 읽기 · 2 지속 · 3 전이 · 4 영속)** — 물리 장면은 측정 도구, "중력을 아는가" 는 묻지 않는다; ledge · wall 은 common sense 라 정의 밖 | R 재귀 → 기제 진단, 정합 → 측정 조건 (능력 표에서 뺌) · ledge · wall (정의 밖으로) | 사용자 결정 ("물리적 engine 을 가지고 있느냐를 볼려고 하는게 아니야, 그냥 수단으로써 간접적으로 보는거지" — 레포 밖 메모 인용) · 거리 한계: 슬롯이 아니라 거리 (100 px 당 −0.47) | **정의는 그대로 유효** — 10-08 에 intuitive physics 의 조작적 정의로 재해석 · ledge · wall 은 10-08 번복 (3b 사건) |
| 09-22 | **`z_research/IntPhysGenV11/Archive/PAPER_STORY_2026-09-22.md`** · `z_research/OVERVIEW_2026-09-22.md` · `HANDOFF_2026-09-22.md` | "predictor 는 상태를 이어가지 않는다. 마지막으로 본 자리 주변을 다시 읽어 **두세 칸** 을 그릴 뿐이다 … 학습시켜야 한다 — 다만 궤적을 외우는 것과 구분되게" (앞 절은 state 프레임 — 현재 금지) | 09-06 beat 4(a) 이분법 (진전 / 정지) · "위치가 진전하지 않는다" · "표현이 회전했다" (측정했고 아니었다) · "post-FT 가 상태 진화를 배웠다" (09-17 사용자 지적 철회) · 09-03 "corner case" 판정 (flat_v 에서도 같은 성질) | RollOutV2 09-12 · 09-19, ctxenc 실험 1–6, z_training 결과 | **네 능력 표 · 재학습 통과 조건 4 (→ 정본 §2-4 방법 절 2) · EK100 규약 (released 35.34 vs paper 19.07) · frozen 개입 작음 (+3)** · 금지 표현 |
| 09-24 | **`auto_research/Archive/PLAN_2026-09-24.md`** (auto_research 개설) + H1 · H23 · H4 · H5 · H6 · H7 · H8 · H9 · `CROSS_REVIEW_2026-09-24.md` | 가설 트리 — 출력 p 에서 "못 한다" 는 확인됐고 빠진 것은 (i) 입력 재료의 공정한 기준 (ii) predictor 안 어디서 끊기나 (iii) 시간 · 되먹임으로도 안 되나 (iv) 외부 점수가 같은 짧은 조회로 설명되나; 병목 지도 [E]→[I]→[P]→[O]→[S] | 데이터 RollOut_v2 → **RollOut_v3** (사용자 "데이터는 RollOut_v3 쓰면 되잖아") · 창 하나 → 16 창 (사용자 "다양하게 봐") · 첫 판 헤드라인 셋 (층별 "뒤층이 지운다" 등) | 적대 검증 — 자기 기준 문턱 인공물 · 궤적 누수 · clip CI 가 너무 좁음 | **H6: IntPhys 1 복사 85.00 vs 88.89 (Δ +3.9, CI 0 포함, 확정)** → 정본 P2 의 중심 · H23 §3-7 (떠 가는 미래는 복사와 닮았다) · H1 정정판 |
| 09-25 오전 | `auto_research/Archive/README_UNTIL_2026-09-25.md` · `paper/DRAFT_v0/v1/v2_2026-09-25.md` · `paper/REVIEW_v1_2026-09-25.md` | "입력 토큰에는 가시 운동이 읽히고, 릴리즈 predictor 는 몇 걸음 외삽한 뒤 멈춘다. 같은 frozen encoder 위에 학습된 predictor 는 약 두 배 간 뒤 멈춘다"; 초안 제목 "Seeing Is Not Evolving … Does Not Carry the World State" (원 문서 표현 — state 프레임, 현재 금지) | **"encoder 는 충분하고 predictor 가 병목이다" 철회** · 기전 귀속 셋 (마지막 토큰이 정한다 · 역사 토큰이 뒤로 끈다 · 속도를 한 걸음만 쓴다) 철회 · P2 반경 공식 "2.64 칸 + 1.89 걸음" 철회 | 충분성 대조 (M13 z_cv) 가 진실 위치를 썼다 · 적대 검증 1 · 2 차 (`VERIFY_0925*`) · REVIEW_v1 weak reject 3/6 | 정정 1 · 2 표 · 읽기 규칙 1–10 · 한 걸음 전이 탐침 0.361 (진실 0.440) |
| 09-25 낮 | **`auto_research/paper/ORAL_DIRECTION_2026-09-25.md`** · `Archive/SC1_REACH_NOT_HORIZON_2026-09-25.md` | 첫 안 A "Horizon, not perception — 학습 질의 지평까지만 외삽" → 안 **A′ "Reach, not horizon"**: "Latent video predictors carry a scene only a bounded distance from where they last saw it." | 안 A (학습 지평 8 슬롯) · "걸음 수 · fps 와 무관한 고정 거리" (release 에 과장) | SC1: 슬롯 상수 R² 0, 거리 R² 0.68–0.84, Ariel stride 1 에서 13–17 슬롯 | reach 법칙 (DC1 근거로 흡수) · 실영상 reach 밖 비율 (SSv2 46 · EK100 51 %, I_SCENE) |
| 09-25 저녁 | **`auto_research/paper/DESIGN_CHOICES_2026-09-25.md`** | 주인공 = **새 video predictor**, 분석은 design choice 근거: DC1 움직임 정렬 조회 · DC2 미래 전용 목표 · DC3 자기 앵커 닫힘 · DC4 있음 · DC5 reach 평가; 학습 단계 M → N → O → P → S → Q | "synthetic 에서 predictor 가 문제다" (사용자: "이건 별로야") | 사용자 프레임 결정 (09-25) | **§1 DC 표 · §3-b–d 설계 후보는 현행 (미검증 설계안)** — 정본 §2-4 "Ariel 이 왜 실패했나 → 설계 후보" 의 출처 |
| 09-25 밤 | **`auto_research/paper/DIRECTION_2026-09-25.md`** · `paper/ABSTRACT_v0_2026-09-25.md` | 목표 = "video prediction 사전학습으로 관측이 불완전해도 state 를 진화시켜 근미래를 anticipate 하는 — 더 인간 같은 — latent world model"; 초록 "what is missing is not representation but a carried state" (둘 다 state 프레임 — 현재 금지) | — (09-29 · 10-08 에 대체) | 사용자 목표 재정의 | §0 분업 (분석 Claude · 학습 사용자, 09-26 확정) · §4 근거 수치 · §4-2 문헌 |
| 09-26 | `auto_research/paper/STORY_EVIDENCE_2026-09-26.md` · `paper/REVIEW_v2_2026-09-26.md` · `Archive/INTPHYS2_PERMANENCE_AUDIT_2026-09-26.md` | 옛 스토리 문장별 근거 지도 (그림 셋 + 표 하나); 리뷰 2 차 [A] weak reject 경계 · [D] poster | IntPhys 2 "복사 ≥ 릴리즈" → "구분 안 됨" (best-C 산물) · 매개 "95 / 99 %" → 44–67 % · qshift · 팬 epoch 포화 → 시사 | 적대 검증 2 차 (`VERIFY_ISCENE2`) · REVIEW_v2 가장 약한 고리 = "미래 예측으로 학습해도 같다" 가 Ariel 한 run 에 걸림 | 행별 수치 · 등급 · 빈틈 목록 · STORY_EVIDENCE §2-4 "다시 쓰면 안 되는 문장" |
| 09-29 | **`auto_research/paper/RETHINK_PREDICTOR_ROLE_2026-09-29.md`** · `NARRATIVE_v2_USER_2026-09-29.md` · `ABSTRACT_v1_2026-09-29.md` | "JEPA 계열 world model 에서 **'예측' 의 자리가 비어 있다** — encoder 는 관찰을 잘 표현하지만 predictor 는 마지막 관측을 옮겨 적을 뿐이고 흔한 평가는 그것으로 충분하다"; 초록 "World knowledge belongs in the predictor" (내부 귀속 — 현재 금지); 사용자 7 문단 서사 | — (10-08 에 대체) | 사용자 논지 이름 정리 ("encoder 는 표현, predictor 는 예측 → world knowledge 는 predictor 에", 방향 변경 아님) | **§1-b encoder 충분성 표 · §5-b copy-hard (릴리즈 v11 48.4 · IntPhys 1 44.4 · IntPhys 2 21.2) · predictor 몫 Δ IntPhys 1 +4 (AR 공간 +7) · IntPhys 2 ≈ 0 · EK100 +2** (`RETHINK` §5) · NARRATIVE_v2 §A 인용 목록 (→ 01 장) |
| 09-30 | `RETHINK_PREDICTOR_ROLE_2026-09-29.md` §0-c · `DESIGN_CHOICES` §3-c · §3-d | **memory · carried state 프레임 금지** — 세 실패 (reach · permanence · 가속) 는 "predictor 가 관측된 상태에 변화 규칙을 적용하지 않고 마지막 관측을 옮겨 적는다" 로 하나 (원 문서 표현 — "규칙 적용" 은 09-30 의 권장 용어였지만 10-08 정본 §7 은 "물리 법칙을 적용한다" 류 내부 귀속을 금지한다; 본문은 행동으로) | "기억 문제" · "state 모듈" 을 논지로 쓰기 · "상태/렌더 분리" 는 구조 선택지로만 | 사용자 ("memory 문제로 쓰고 싶지 않다") · 데이터: 정보는 encoder 에 남고, 가림 없는 v3 에서도 가속 반영 계수 α_acc < 0 (인용 금지인 α 분해 · 헤지 계수와 다른 양) | **금지 유지** (정본 §3) · 설계 후보 표 §3-c · §3-d |
| 10-08 | **`auto_research/paper/PAPER_STORY_2026-10-08.md` (현 정본)** | §1-2 의 한 문장 | 표제어 "예측의 자리가 비어 있다" → "intuitive physics 를 따르는 plausible future prediction" · 인간 비유 "첫 문장과 discussion 에만" → P1 전체 · 벤치마크 감사 = 독립 beat → P2 근거 · 주 평가 EK100 · Ego4D → 결정 대기 · ledge · wall "정의 밖" → 포함 | 사용자 스토리 + 같은 날 후속 결정 넷 (§3) | 전부 — 정본 §3 "남는 것" (§5 이 장) |
| 10-10 | (레포 문서 없음 — 이 아카이브 작업 중 사용자 지시) | "아직 AC가 우리가 도달할 이상은 아니야 그걸 명심해." | AC 를 최종 판으로 쓰기 | 사용자 결정 (같은 날 풀이: "아직 최종이 아니라는 거라서 AC 스토리도 일단 작성") | 주 목표는 관측 전용 predictor · AC 는 작은 subsection 초안 (`09b_AC_STORY.md`) |

⚠️ **문서 간 모순 (지우지 않는다)**: `auto_research/README.md` §2 철회표 1 행은 "encoder 는 충분하고 predictor 가 병목이다" 를 **철회** 했고 (09-25, 충분성 대조가 진실 위치를 썼다), `STORY_EVIDENCE_2026-09-26.md` §2-4 는 그 문장을 금지했다. 10-08 정본 P3 는 "encoder 는 **정보** 충분, predictor 가 실패" 를 다시 세운다. 재근거는 `FUTURE_FROM_CONTEXT_2026-10-02.md` (과거만 본 z 위의 자 ≈ h 자, 1 차) · `ENCODER_MOTION_READOUT_2026-09-29.md` (EK100 R² .69 / .63, 1 차) · `INTPHYS2_ENCODER_LINEAR_PROBE_2026-09-30.md` (쌍 73.7 vs VoE 54.3) 이고, 세 문서 모두 "encoder 가 미래를 표현한다" 는 못 쓴다고 단서를 단다. 이 문장의 근거 상태는 07 장이 다룬다.
⚠️ `DRAFT_v2_2026-09-25.md` 의 "stall near their own training horizon" 은 같은 날 SC1 이 기각했다 (STORY_EVIDENCE §2-4 금지 문장 "학습 지평 (8 슬롯) 에서 멈춘다"). 두 문서는 같은 날짜다.
⚠️ 정본 머리의 대체 목록은 "`PAPER_STORY_2026-09-22` (및 그 앞 09-06 · 08-31 · 08-29 판)" 이다 — 08-29 판의 원 경로는 `z_research/IntPhysGenV10/Archive/PAPER_STORY_2026-08-29.md` 다 (V11 폴더가 아님).

### 출처 (§2-1)
`evals/analysis_vlm/modes/REPRODUCTION_PLAN.md` · `z_world_model_analysis/{intphys1_dev_2026-08-13, occlusion_v2_2026-08-13_PREVIOUS-DATA, probe_set_2026-08-13_PREVIOUS-DATA, RESULTS_2026-08-16, FRAME_ABLATION_2026-08-19, TOKEN_SUBSET_2026-08-19, MECHANISM_2026-08-20, WHY_INTPHYS_WORKS_2026-08-20, VERIFICATION_2026-08-20, OCCLUSION_LENGTH_2026-08-20, PREDICTOR_HEDGING_2026-08-26}.md` · `z_research/IntPhysGenV10/Archive/{V10_2026-08-28, GEOMETRY_2026-08-29, PAPER_STORY_2026-08-29}.md` · `z_research/IntPhysGenV11/Archive/{SYNTHESIS_2026-08-30, TALK_2026-08-30, PAPER_STORY_2026-08-31, STATUS_AND_CRITIQUE_2026-09-03, PAPER_STORY_2026-09-06, PAPER_STORY_2026-09-22}.md` · `z_research/IntPhysGenV11/Archive/surprising_score/RESULTS_2026-08-30.md` · `z_research/RollOutV2/figures/v5/summary/POSITION_READOUT_2026-09-12.md` · `z_research/RollOutV2/Archive/PREDICTOR_DISTANCE_LIMIT_2026-09-19.md` · `z_research/context_encoder_analysis/Archive/{CLAIM_NO_STATE_EVOLUTION_2026-09-15, STATE_EVOLUTION_HYPOTHESES_2026-09-15, STATE_EVOLUTION_CAPABILITIES_2026-09-19}.md` · `z_research/{OVERVIEW,HANDOFF}_2026-09-22.md` · `auto_research/Archive/{PLAN_2026-09-24, CROSS_REVIEW_2026-09-24, H6_INTPHYS1_WHAT_THE_SCORE_MEASURES_2026-09-24, README_UNTIL_2026-09-25, SC1_REACH_NOT_HORIZON_2026-09-25, INTPHYS2_PERMANENCE_AUDIT_2026-09-26}.md` · `auto_research/paper/{DRAFT_v0_2026-09-25, DRAFT_v1_2026-09-25, DRAFT_v2_2026-09-25, REVIEW_v1_2026-09-25, ORAL_DIRECTION_2026-09-25, DESIGN_CHOICES_2026-09-25, DIRECTION_2026-09-25, ABSTRACT_v0_2026-09-25, STORY_EVIDENCE_2026-09-26, REVIEW_v2_2026-09-26, RETHINK_PREDICTOR_ROLE_2026-09-29, NARRATIVE_v2_USER_2026-09-29, ABSTRACT_v1_2026-09-29, PAPER_STORY_2026-10-08}.md` · `auto_research/README.md` §0 머리말 · §2

### 2-2. 주제별 변천 — 같은 질문이 판마다 어떻게 답해졌나

판별 표 (§2-1) 를 주제로 다시 자른 것이다. 화살표는 시간 순서다.

| 주제 | 변천 | 지금 (10-08 · 10-10) | 출처 |
|---|---|---|---|
| **원인 귀속 — 목적함수** | 08-29 "근본 원인은 목적함수다" (beat 11) · 08-30 TALK "격차는 목적함수에서 나온다" → 08-31 discussion 한 문단으로 강등 → 09-22 "원인 축으로 걸지 않는다" | **discussion 한 문단** (정본 §3 유지, CLAUDE.md §0) | `IntPhysGenV10/Archive/PAPER_STORY_2026-08-29.md` · `IntPhysGenV11/Archive/TALK_2026-08-30.md` · `PAPER_STORY_2026-08-31.md` · `-09-22.md` |
| **frozen 개입으로 닫기** | 08-31 원칙 ("논문 전체가 frozen model 위에서 닫힌다") → 09-06 **폐기** (beat 7 = predictor 재학습) → 09-16 · 09-22 개입을 재 보고 닫음: 정렬 사상 visible +3.0 / late +3.2 / early +6.7 (재학습 +25~30 의 1/5) · 균등 증폭 천장 51–65 → 09-24 H9 frozen 개입 채점 | **폐기 유지** — 목표가 predictor 학습 | `PAPER_STORY_2026-08-31` · `-09-06` · `-09-22` §4-1 · `CLAIM_NO_STATE_EVOLUTION` §2-7 · `auto_research/Archive/H9_FROZEN_INTERVENTION_SCORING_2026-09-24.md` |
| **가림의 지위** | 08-13 blender occlusion 이 주인공 → 09-03 "corner case" 비판, 주인공에서 내림 → 09-22 "가림 없는 flat_v 에서도 같은 성질 → corner case 아님" → 09-25 실영상 reach 밖 비율 (SSv2 46 · EK100 51 %) 로 방어 → 10-08 "안 보이는 것" 한 축 + **마지막 관측** 으로 좁힘 | P3 (ii). corner case 판정 절차는 하네스 9 · 정본 §4-3 | `STATUS_AND_CRITIQUE_2026-09-03` · `PAPER_STORY_2026-09-22` §3 · `auto_research/Archive/I_SCENE_CAUSAL_BOTTLENECK_2026-09-25.md` §4-3 · 정본 §1 · §4-3 |
| **"정보는 있다" (encoder · p)** | 08-20 "p 에 정보가 없다" 철회 (층별 probe 68 → 86) → 08-30 v11 p probe 98.46 → 09-25 "encoder 충분 · predictor 병목" **철회** → 09-29 RETHINK §1-b 충분성 표 → 10-08 "encoder 는 **정보** 충분, predictor 가 실패" (재근거 `FUTURE_FROM_CONTEXT` 등) | P3 (iv) — 단 "encoder 가 미래를 표현한다" 금지 · shape 은 encoder 도 흔들림 (z 3 팔 64.8) | `MECHANISM_2026-08-20` · `RESULTS_2026-08-30` · `README_UNTIL` 정정 1 · `RETHINK` §1-b · 정본 §2-3 (iv) |
| **외삽의 단위 (슬롯 · 지평 · 거리)** | 09-06 horizon t0 90.3 → t7 61.4 (튜블릿 축) → 09-15 "등속 3–4 슬롯, 가속 ≤ 1–2" → 09-19 **거리 한계** (100 px 당 −0.47 vs 슬롯 −0.28) → 09-25 SC1 "학습 지평 (8 슬롯)" 기각, reach (안 A′) → 09-25 정정: release 는 약 2 칸 + 1.6 걸음, "고정 거리" 는 과장 | P3 (i) "2–3 칸 (40–60 px)" | `PAPER_STORY_2026-09-06` §3 · `CLAIM` §2-2 · `PREDICTOR_DISTANCE_LIMIT_2026-09-19` · `SC1` · `auto_research/README.md` §0 정정 |
| **사건 (ledge · wall · 꼭대기)** | 09-10 RollOut_v2 재생성 (ledge · wall 포함) → 09-12 위치 자: ledge 선반 높이 유지 · wall 2 슬롯 통과 뒤 소실 → 09-19 "common sense, 정의 밖" (사용자) → 09-22 근거로만 → 09-29 NARRATIVE_v2 B-2 "그림 예시 한 줄" → 10-06 / 07 실물 메시 ledge 0 / 120 · gravity 꼭대기 → **10-08 포함 (3b 사건)** | P3 (iii). 넣는 조건 넷 (복사와 닮은 미래 · 갈림 시점 누수 · 자 둘의 다른 답 · L1 교란) | `POSITION_READOUT_2026-09-12` · `STATE_EVOLUTION_CAPABILITIES` 머리 · `NARRATIVE_v2` §B-2 · `z_research/v11_realistic/README.md` · 정본 §2-3 (iii) |
| **벤치마크 감사 → leakage** | 08-20 WHY_INTPHYS_WORKS "IntPhys 1 은 내용량 누수로 풀린다" (corr 0.486) → 09-03 "latent surprise = 외형 일치 검출기" → 09-24 H6 복사 85.00 vs 88.89 → 09-25 · 09-29 독립 beat · 기여 (DIRECTION §2 · RETHINK §5) → 10-07 위치만 다른 쌍 (`COPY_VS_POSITION`) · DINO-F → **10-08 P2 의 근거 "IntPhys 1 은 leakage 가 있는 벤치"** (사용자 "이거 leakage있는 벤치잖아.") | P2. leakage = 복사 기준선으로 잰 "예측 없이 풀리는 정도" — CLAUDE.md §1-2 의 미래일치 쌍 누수와 다른 뜻 | `WHY_INTPHYS_WORKS_2026-08-20` · `STATUS_AND_CRITIQUE` §2 · `H6` · `auto_research/Archive/COPY_VS_POSITION_2026-10-07.md` · 정본 §2-2 |
| **주 평가** | 08-31 beat 8 IntPhys 1 · 2 격차 → 09-06 beat 8 "frozen encoder + 우리 predictor 로 anticipation + intuitive physics" (미실행) → 09-20 EK100 `released` 규약 기준 → 09-25 밤 EK100 · IntPhys 2 · Ego4D · EgoExo4D → 10-08 **결정 대기** (기본값 = intuitive physics 판 1 차, EK100 등 2 차) | 결정 대기 2 | `PAPER_STORY_2026-08-31` · `-09-06` · `z_research/anticipation/EK100/README.md` 머리 · `DIRECTION` §1 · 정본 §3 |
| **주인공 · 목표** | 08-29 ~ 08-31 frozen 모델의 진단 → 09-06 "predictor 를 새로 학습해야" → 09-22 "궤적 외우기와 구분되게" (통과 조건 4) → 09-25 "새 video predictor 의 design choice" → 09-29 "world knowledge 를 가진 predictor" → 10-08 "action 없이 관측만으로 intuitive physics 를 배우는 predictor — **Ariel 을 반드시 넘는다**" (사용자 "이건 어케든 맞춰야지.") | 목표 ★ (사용자 학습 항목). 합격선 = 정본 §2-4 | `PAPER_STORY_2026-09-06` · `-09-22` §4-3 · `DESIGN_CHOICES` · `DIRECTION` §0-b · 정본 §2-4 |
| **인간 · 영유아 비유** | 09-25 DIRECTION §5 "첫 문장과 discussion 에만" → 09-29 NARRATIVE_v2 사용자 7 문단 서사 (영아 관찰 학습으로 시작) → 10-08 **P1 전체 (intro 첫 문단)**. 본문 판정은 계속 측정 가능한 정의로 | P1 (문헌, 인용 확인 필요) | `DIRECTION` §5 · `NARRATIVE_v2` §A · 정본 §3 |
| **물리 장면의 지위 · 정의** | 09-19 "측정 도구 — 물리 엔진 · 중력을 아는가를 묻지 않는다" → 10-08 intuitive physics 가 표제. 판정은 계속 **행동**, 내부 귀속 ("중력을 안다 / 법칙을 적용한다") 금지 | 네 능력 = intuitive physics 의 조작적 정의 (§5-1) | `STATE_EVOLUTION_CAPABILITIES` §0 · 정본 §3 |
| **memory · state 프레임** | 09-06 ~ 09-29 표제 대부분이 이 프레임 ("상태를 이어가지 않는다" · "carried state" · "state 모듈") → **09-30 금지** (사용자) → 10-08 유지 | 금지. 용어는 "지속 · 전이 · 영속" (09-30 의 "규칙 적용" 은 행동 서술로만 — 내부 귀속 금지, 정본 §7) | `RETHINK` §0-c · 정본 §3 · §7 |
| **AC 의 역할** | 09-06 beat 7 근거 (AC 가 `load_predictor: false` — 당시 "암묵적 시인") → 09-19 `z_AC_training/` 개설 (코드 분석만, 실행 없음) → 09-20 "AC 는 가림 측정 대상이 못 된다" (horizon 1 · agent 가 일으킨 변화만) · Jongseo RoboCasa "AC 가 action 안 읽음" 인용 불가 → 09-25 DESIGN_CHOICES 단계 Q (AC fine-tune) → 10-08 작은 subsection + 실험 하나 → **10-10 "아직 최종 아님" — 스토리 초안은 쓴다** | `09b_AC_STORY.md` (초안) | `PAPER_STORY_2026-09-06` · `z_AC_training/README.md` §5-1 · `z_AC_training/Archive/PLANNING_AND_ROBOCASA_2026-09-20.md` · `DESIGN_CHOICES` §3 · 정본 §2-5 |

### 출처 (§2-2)
§2-1 출처와 같음 + `z_research/v11_realistic/README.md` · `auto_research/Archive/{I_SCENE_CAUSAL_BOTTLENECK_2026-09-25, COPY_VS_POSITION_2026-10-07, H9_FROZEN_INTERVENTION_SCORING_2026-09-24}.md` · `z_research/anticipation/EK100/README.md` · `z_AC_training/README.md` · `z_AC_training/Archive/PLANNING_AND_ROBOCASA_2026-09-20.md`

---

## 3. 사용자 결정 기록

원문 인용이 레포 문서에 있으면 그대로 옮겼다. 레포 문서에 원문이 없고 사용자 auto-memory (레포 밖) 에만 있는 것은 "레포 밖 메모" 라고 적었다 — 그런 줄은 원 문서로 확인되지 않은 인용이다.

| 날짜 | 결정 (원문 인용이 있으면 그대로) | 무엇을 바꿨나 | 출처 |
|---|---|---|---|
| 08-16 | (규약 — 사용자 결정인지 문서에 없음) 채점 표준 = **문맥일치 2 쌍** | 미래일치 쌍 권고 (08-13) 를 뒤집음. `scoring.pairing: matched` | `z_world_model_analysis/RESULTS_2026-08-16.md` §1.4 · CLAUDE.md §1-2 |
| 08-19 | 기준 모델 = **ViT-H** (IntPhys 1 88.89 vs ViT-L 64.44, 외부 벤치마크에서 먼저 정함). 자체 데이터로 사후 모델 선택을 하지 않는다 | 이후 모든 채점 · probing 이 ViT-H. "ViT-H 가 항상 낫다" 는 거짓 (v8 shape ViT-L 83.98 vs 73.83 등) | CLAUDE.md §1-1 · `configs/protocols/models.md` 머리 |
| 날짜 문서에 없음 | **α 분해 · 헤지 · 보정 계수 인용 금지** (사용자 지시가 있을 때까지) | 08-20 MECHANISM §7 · 08-26 PREDICTOR_HEDGING · 08-29 GEOMETRY §1 의 α 값이 전부 인용 불가. 이 아카이브도 값을 옮기지 않는다 | CLAUDE.md §0 · §5-4 |
| 09-03 | 외부 피드백 "event transition 에 occlusion 이 걸리는 것은 너무 corner case" 에 사용자 동의 — 피드백에 corner case / use case 판정을 같이 준다 | 가림을 주인공에서 testbed 로. 하네스 9 (→ 정본 §6 9 · §4-3) | `z_research/IntPhysGenV11/Archive/STATUS_AND_CRITIQUE_2026-09-03.md` · CLAUDE.md §12 9 |
| 09-06 | (문서 결정 — 사용자 지시 여부 문서에 없음) **"frozen 위에서 닫힌다" 를 의도적으로 폐기** | beat 7 = predictor 재학습. CLAUDE.md §10 · §12 에 한동안 옛 문장이 남았다 (HANDOFF §1 지적) | `PAPER_STORY_2026-08-31.md` 머리 blockquote · `PAPER_STORY_2026-09-06.md` · `z_research/HANDOFF_2026-09-22.md` §1 |
| 09-10 | **릴리즈 ViT-H predictor 에서 post-FT, 데이터 = IntPhysGen v11 의 block 단위 train 절반 (가능만)** | `z_training/` 개설 · `v11_split_train` / `v11_split_test` · 출발선 75.83 | `z_training/README.md` 머리 · CLAUDE.md §2-7 |
| 09-12 | "decoder 문제 없다를 보여야 predictor 가 만드는 세상에 대해 주장할 수 있지" | 학습된 자를 쓰면 held-out · encoder 대조 · 파라미터 없는 교차검증을 같이 낸다 (하네스 8). "p 가 물체를 마지막 자리에 둔다" 철회의 계기 | `z_research/IntPhysGenV11/Archive/PAPER_STORY_2026-09-22.md` · `z_research/RollOutV2/figures/v5/summary/POSITION_READOUT_2026-09-12.md` · CLAUDE.md §12 |
| 09-15 | (레포 밖 메모) "정리가 너무 안 되어 있는 것 같다" — 진입점 하나, 결론 → 근거 3 줄 → 미결 1 줄 | 이 아카이브의 장 머리 형식의 원형 | 레포 밖 메모 (`feedback-one-entry-point-less-parallel`) |
| 09-16 | context_encoder_analysis 의 방향 = encoder 토큰 (T×N, D) 안의 video prediction 용 상태 변수 분석 | ctxenc 실험 1–6 의 축 | 레포 밖 메모 (`project-ctxenc-direction`) — 레포 문서에 결정 원문 없음 (세트 README 제목 "관측 없는 상태 진화를 위한 재료는 encoder 에 있는가" · `z_research/context_encoder_analysis/Archive/EXPERIMENTS_2026-09-16.md`) |
| 09-17 | (사용자 지적) post-FT 의 "학습 도메인 안에서는 진짜 상태 진화처럼 보인다" 철회 — v11 속도 1 종이라 held-out 도 궤적 같음 | 09-22 beat 7 통과 조건 4 (도메인 밖 β → 1 등). 정본 §2-4 방법 절 2 "궤적 외우기와 구분" | `PAPER_STORY_2026-09-22.md` §4-3 정정 · CLAUDE.md §6 |
| 09-19 | **state evolution = 네 능력 (읽기 · 지속 · 전이 · 영속)**; 전이는 운동 추세에 한정, **ledge · wall 은 common sense 라 정의 밖**; 원칙 "물리 엔진 · 중력을 아는가를 묻지 않는다" | 이후 문서가 네 칸을 채우는 순서로. 6 개 안을 사용자가 4 개로 줄였다 (레포 밖 메모) | `z_research/context_encoder_analysis/Archive/STATE_EVOLUTION_CAPABILITIES_2026-09-19.md` §0 · §1 |
| 09-20 | **EK100 비교 기준은 공개 코드 규약 (`released`) 하나** | released 35.34 vs paper 19.07 중 released 를 기준선으로 | `z_research/anticipation/EK100/README.md` 머리 · `OVERVIEW_2026-09-22.md` §5-11 |
| 09-20 | (사용자 지적 · 동의) **AC 는 "가림에서 상태를 이어가는가" 의 측정 대상이 못 된다** (horizon 1 · agent 가 일으킨 변화만 · action 0 = 아무 일 없음) | AC 를 진단 대상에서 뺌 · Jongseo RoboCasa 결론 인용 불가 판정과 같은 날 | `z_AC_training/README.md` §5-1 · `z_AC_training/Archive/PLANNING_AND_ROBOCASA_2026-09-20.md` |
| 09-24 | auto_research 범위: **frozen 릴리즈 ViT-H 하나 · 새 학습 없음 · 합성 (v11 · RollOut) + IntPhys 1 · vll6 최대 8 GPU** · 평균 쪽 수축은 새 지표로만 (옛 α · 헤지 인용 금지) | `auto_research/` 개설, H1–H9 | `auto_research/Archive/PLAN_2026-09-24.md` 머리 · `auto_research/Archive/README_UNTIL_2026-09-25.md` 범위 |
| 09-24 | "데이터는 RollOut_v3 쓰면 되잖아" · "다양하게 봐" | H1–H5 주 데이터 RollOut_v2 → **RollOut_v3**; 창을 16 개 (C · P ∈ {4,8,16,32}) 로 | `auto_research/Archive/PLAN_2026-09-24.md` 변경 1 · 변경 2 |
| 09-24 | (레포 밖 메모) "흠 뭔가 공식이 있으면 좋을 거 같은데" · "아니 뭐 어떻게 할건지 먼저 알려줘야지" — 공식 · 계획 먼저 | 새 지표는 기호 · 공식 · 판정 규칙을 먼저. 09-25 에 auto_research 안에서는 "문서에 적고 바로 돌린다" 로 완화 | 레포 밖 메모 (`feedback-plan-before-running` · `feedback-autonomous-oral-push`) |
| 09-25 | 이미 학습된 z_training predictor 비교는 허용 (예외) · **"새로 학습은 말고"** | 학습된 predictor 넷 비교 (`TRAINED_PREDICTORS_SIGNATURES`) · world video prediction 모델 학습은 결정 전 안 함 | `auto_research/Archive/README_UNTIL_2026-09-25.md` 범위 · `auto_research/README.md` §3 |
| 09-25 | SSv2 방향 세트 (`ssv2_VP`) 를 실영상 encoder 판독에 쓴다 | 범위 확장 (합성 + IntPhys 1 + SSv2) | 레포 밖 메모 (`project-auto-research-scope`) · `auto_research/Archive/E_SSV2_ENCODER_READS_PREDICTOR_STALLS_2026-09-25.md` |
| 09-25 저녁 | 논문 프레임: 주인공은 새 video predictor, 분석은 design choice 근거. "synthetic 에서 predictor 가 문제다" 는 쓰지 않는다 (원문 "…synthetic data 내에서 predictor 가 문제다 이건 별로야…" 는 레포 밖 메모) | DESIGN_CHOICES DC1–5 | `auto_research/README.md` 머리말 09-25 저녁 · `auto_research/paper/DESIGN_CHOICES_2026-09-25.md` §4 |
| 09-25 | slot attention · 물체 중심 구조는 안 한다 (장면 단위 유지) | 설계 후보에서 제외. 정본 §2-1 의 PLATO 구분 | `auto_research/paper/DESIGN_CHOICES_2026-09-25.md` §3-c |
| 09-25 | RollOutV3 자를 위치 + 정체 (모양 7 × 색 8 = 56 + 없음, 57 분류) 로 교체 · 가림 음성은 자에 넣지 않는다 | presence 자 → identity 자 → identity_r8 (09-26) | `z_research/RollOutV3/Archive/_superseded/NEW_ARCHIVE_README_2026-09-25_r1.md` · `z_research/RollOutV3/README.md` |
| 09-25 밤 | 목표 재정의: "video prediction 사전학습으로 더 인간 같은 latent world model"; 주 평가 EK100 · IntPhys 2 · Ego4D · EgoExo4D; 합성은 분석용 testbed (state 프레임 — 현재 금지) | DIRECTION · ABSTRACT_v0 | `auto_research/paper/DIRECTION_2026-09-25.md` §1 |
| 09-26 | **분업: 분석은 Claude, 학습은 사용자** | 학습 설계는 DESIGN_CHOICES 에 적어 두기만. 정본 §2-4 까지 유지 | `auto_research/paper/DIRECTION_2026-09-25.md` §0 · 정본 §2-4 |
| 09-29 | 논지 이름: encoder 는 표현, predictor 는 예측 → world knowledge 는 predictor 에; frozen encoder + 새 predictor (사용자 확인: 방향 변경 아님) | DIRECTION §0-b · RETHINK · ABSTRACT_v1 | `auto_research/paper/DIRECTION_2026-09-25.md` §0-b |
| 09-30 | "memory 문제로 쓰고 싶지 않다" — memory · carried state 프레임 금지 | 세 실패를 "규칙 미적용" 하나로. "상태/렌더 분리" 는 구조 선택지로만 | `auto_research/paper/RETHINK_PREDICTOR_ROLE_2026-09-29.md` §0-c |
| 10-08 | 스토리 원문 (P1 · P2 · P3 · 목표, §1-1) | 현 정본 | `auto_research/paper/PAPER_STORY_2026-10-08.md` §0 |
| 10-08 | Ariel 반례: "이건 어케든 맞춰야지." | 목표 = Ariel 을 반드시 넘는다 → 합격선 표 (정본 §2-4) | 정본 §0 후속 결정 |
| 10-08 | IntPhys 1 88.89 와 P2 의 충돌: "이거 leakage있는 벤치잖아." | 벤치마크 감사가 독립 beat 에서 P2 의 근거로 | 정본 §0 · §2-2 |
| 10-08 | ledge · wall · 꼭대기: "이것도 포함해서 논문 쓸거야." | 09-19 "범위 밖" 번복 → 3 전이 = 3a 추세 + 3b 사건 | 정본 §0 · `STATE_EVOLUTION_CAPABILITIES_2026-09-19.md` 머리 |
| 10-08 | AC: "이건 이제 자그많게 subsection 느끼믕로 실험 하나 넣을거라 괜찬ㅇ하." | 정본 §2-5 (→ 10-10 "아직 최종 아님", `09b`) | 정본 §0 · §2-5 |
| 10-09 | 레포를 `cvpr/` 코드 스페이스로 구조화 (엔진 제자리 + resolver · 런처 · 레지스트리) | 새 진입점. 옛 resolver 와 동일성 27 / 27 (건너뜀 1) | `cvpr/README.md` §1 · CLAUDE.md §2-0-b |
| 10-10 | **"아직 AC가 우리가 도달할 이상은 아니야 그걸 명심해."** → 풀이: "이게 아직 최종이 아니라는거라서 AC 스토리도 일단 작성을 해줘야" | AC 스토리는 쓴다, 최종 판은 아니다 (`09b_AC_STORY.md`). 주 목표보다 앞에 두지 않는다 | 이 아카이브 작업 중 사용자 지시 (레포 문서 미반영) |

⚠️ 결정 대기 (10-08 기준): 주 평가를 intuitive physics 판으로 옮기는가 (정본 §3 2). 09-25 의 "world video prediction 모델을 학습할지 · 어느 후보로" 는 10-08 에 "사용자 학습 항목 (★)" 으로 바뀌었다 (정본 §5 8 행).
⚠️ 커밋 결정도 대기다 — `auto_research/` 는 아직 git 추적 밖 (CLAUDE.md §8-1).

### 출처 (§3)
`z_world_model_analysis/RESULTS_2026-08-16.md` · `configs/protocols/models.md` · `CLAUDE.md` §0 · §1 · §2-7 · §6 · §8-1 · §12 · `z_research/IntPhysGenV11/Archive/{STATUS_AND_CRITIQUE_2026-09-03, PAPER_STORY_2026-08-31, PAPER_STORY_2026-09-06, PAPER_STORY_2026-09-22}.md` · `z_research/HANDOFF_2026-09-22.md` · `z_training/README.md` · `z_research/RollOutV2/figures/v5/summary/POSITION_READOUT_2026-09-12.md` · `z_research/context_encoder_analysis/README.md` · `z_research/context_encoder_analysis/Archive/STATE_EVOLUTION_CAPABILITIES_2026-09-19.md` · `z_research/anticipation/EK100/README.md` · `z_research/OVERVIEW_2026-09-22.md` · `z_AC_training/README.md` · `z_AC_training/Archive/PLANNING_AND_ROBOCASA_2026-09-20.md` · `auto_research/Archive/{PLAN_2026-09-24, README_UNTIL_2026-09-25, E_SSV2_ENCODER_READS_PREDICTOR_STALLS_2026-09-25}.md` · `auto_research/README.md` · `auto_research/paper/{DESIGN_CHOICES_2026-09-25, DIRECTION_2026-09-25, RETHINK_PREDICTOR_ROLE_2026-09-29, PAPER_STORY_2026-10-08}.md` · `z_research/RollOutV3/README.md` · `z_research/RollOutV3/Archive/_superseded/NEW_ARCHIVE_README_2026-09-25_r1.md` · `cvpr/README.md`

---
## 4. 데이터 · 실험 세대 연표 (짧게 — 설계 · 규모 · 감사는 `03_TESTBED_AND_MEASUREMENT.md`)

### 4-1. 합성 VoE testbed — blender → IntPhysGen v1 … v11 → realistic

| 세대 | 날짜 | 규모 · 무엇 | 대표 수치 (프로토콜) | 지금 상태 | 출처 |
|---|---|---|---|---|---|
| blender toy (`analysis_vlm`) | 07-03 (전사) | V-JEPA 2-L 물리 해석 재현 blueprint | 결과 없음 | config `configs/_archive/` (09-21) | `evals/analysis_vlm/modes/REPRODUCTION_PLAN.md` |
| blender occlusion_v2 · probe_set | 08-13 (재렌더 전) → 08-15 재렌더 | 1,700 video = 425 block, 40 프레임 @ 8 fps, ViT-L | 문맥일치 52.82; presence 100 / 0 (재렌더본, ViT-L) | 재렌더 전 수치 superseded | `z_world_model_analysis/RESULTS_2026-08-16.md` §3 |
| blender occlusion_v3 · v4 | 08-19 ~ 08-20 | v3 1,760 video (ViT-H) · v4 (가림 길이 3~7 / 10~14 프레임) | v3 C=4/8/12/16 문맥일치 50.9~54.3 · v4 64 % → 53.9 % | v4 "가림 길이 레버" 결론은 뒤집힘 | `FRAME_ABLATION_2026-08-19.md` · `OCCLUSION_LENGTH_2026-08-20.md` |
| IntPhysGen v1 (UnrealEngine) | 08-20 | 1,024 video / 256 block, ViT-H | 층별 probe 68 → 86, p 88.7; 토큰평균 L1 52.9 → 88.2 | v1 데이터셋 자체가 없다 (`z_world_model_analysis/` 일부 스크립트가 깨짐, CLAUDE.md §2-6) | `MECHANISM_2026-08-20.md` |
| IntPhysGen v2 ~ v7 · v9 | 날짜 문서에 없음 | v4 는 가림 길이 통제 스펙 제안 (`OCCLUSION_LENGTH`), v6 / v6aug · v7 은 CLAUDE.md §1-3 표와 `z_exp/` 에만 | v6aug fixed / sliding avg / max 75.49 / 76.76 / 79.49 · static+occluded shape 46.88 vs 90.62 | 레지스트리에서 뺌 · 프레임 삭제 | CLAUDE.md §1-3 · §4-3 · `z_research/HANDOFF_2026-09-22.md` §3 |
| IntPhysGen v8 (+ `v8_halfsize`) | 08-26 이전 (08-26 문서가 캐시를 씀; 생성일 문서에 없음) | 2,048 clip / 512 block, 가림 4+4 | ViT-H 79.10 · ViT-L 73.44 (fixed, 1,024 쌍) · halfsize 74.71 | 아카이브. `IntPhysGenV8/exp_results/` 는 비어 있음 — 출처는 `z_exp/world_model_analysis/results/` | CLAUDE.md §5-4 · `OVERVIEW_2026-09-22.md` §9-2 · halfsize 74.71 은 `configs/protocols/datasets.md` `## v8_halfsize` |
| 2D transit (v8) | — | 프레임 195 M | ViT-H / ViT-L 71.48 / 75.39 | **08-28 프레임 삭제 · 생성 코드도 없음** · 캐시 41 G 만 | CLAUDE.md §4-3 |
| IntPhysGen v10 · `v10_flat` · `v10_occ_low` | 08-28 | 8,192 / 4,096 / 1,024 clip | 74.32 / 75.20 / 73.44; 가림 4→1 무관; ramp shape 67.58 vs flat 89.84 | 프레임 삭제 (결과만) | `z_research/IntPhysGenV10/Archive/V10_2026-08-28.md` |
| **IntPhysGen v11** | 08-30 | 21,504 clip / 5,376 block / **10,752 쌍**; 6 조건 × 위반 3 × `sym_k` 0–4 | **73.37** (ViT-H `surprise_c16t32` matched, 확정) | 본 testbed | `z_research/IntPhysGenV11/README.md` · `RESULTS_2026-08-30.md` |
| `v11_earlymid` → `v11_full` | 09-01 → 09-02 | +21,504 clip (early · mid) → 43,008 clip / 21,504 쌍, 12 조건 | 76.00; visible / early / mid / late 87.2 / 79.0 / 78.2 / 59.5 | 현역 (캐시 840 G 는 vll5 에 없음). `v11_timing` 은 대체됨 · `v11_occtiming` pilot 은 사용 불가 | `z_research/IntPhysGenV11_occlusion_timing_ablation/README.md` · `configs/protocols/datasets.md` |
| `v13_black` | 09-02 ~ 09-03 | 4,608 clip, 가림막 검정 (v11 은 나뭇결) | colour visible→early −14.1 → −13.5 (가설 기각) · shape −6.2 → 0.0 | 프레임 삭제, 세트 폴더 없음 — 결과만 | `IntPhysGenV11_occlusion_timing_ablation/README.md` · CLAUDE.md §6 |
| `v11_split_train` / `_test` | 09-10 | v11_full block 절반씩 | 릴리즈 75.83 → post-FT 91.35 | 현역 (학습 held-out) | `z_training/README.md` · CLAUDE.md §2-7 |
| **`v11_realistic` · `v11_realistic_ledge`** | 10-06 | 실물 메시 1,440 clip / 720 쌍 (vanish) · ledge 240 clip / 120 쌍 | vanish 79.58 · ledge **0 / 120** (✓ 10-08 재확인 · 등급 문서에 없음) | 현역 | `z_research/v11_realistic/README.md` |
| **`gravity_realistic`** | 10-07 (재렌더, 옛 미래 넷 판 대체) | 720 clip / 144 block = 과거 1 + 미래 5 | 오르는 문맥 float 83.3 · 꼭대기 arc 62.5 / float 33.3 · 내려가는 문맥 arc 85.4 · stop 1 / 144 (✓ 10-08, 10-08 정정판) | 현역 (`SET="scoring.pairing=cross"`) · metadata 정정 10-07 · 실제 g 120 (생성 로그 `gravreal_219837.log`; g 170 은 대체된 미래 넷 판, `dataset.json` 의 "g 101" 문구는 틀림) | `z_research/v11_realistic/README.md` §4 · `configs/protocols/datasets.md` |

### 4-2. 운동 법칙 · 위치 판독 세트 — RollOut v1 / v2 / v3 · 자 학습셋

| 세대 | 날짜 | 규모 · 무엇 | 지금 상태 | 출처 |
|---|---|---|---|---|
| RollOut_v1 | ~ 09-06 | 등속 직선 7,392 clip, 공간평균 선형 위치 readout | **09-10 ~ 11 프레임 · 캐시 · 결과 삭제**, 레지스트리에서 뺌 (방법만 `v11_roll_out` 에 이어짐) | `z_research/scripts/_superseded/README.md` |
| RollOut_v2 | 09-09 판 → **09-10 재생성** | 7 시나리오 (`flat_v flat_a ramp_a arc fall ledge wall`) 5,488 clip; 09-09 판은 wall 불가능이 정지 궤적 복사였다 | 현역 (vll5). 09-10 이전 pipeline (h 로 배운 자를 p 에 이식) 폐기 | `configs/protocols/datasets.md` `rollout_v2` · `_superseded/README.md` |
| `rollout_v2_decel` | 09-19 | `flat_d` · `ramp_d` 1,568 clip → 9 시나리오 7,056 clip | 현역 | `datasets.md` |
| 자 학습셋 v1 ~ v4 → **v5** | 09-11 삭제 → 09-12 | v5 = 무중력 등속 8,064 clip (위치 544 셀 고루) — v3 (kink · 고속) 은 재시도 금지 | v5 현역 (위치 자 3,842 파라미터, held-out 0.87 칸) | `RollOutV2/figures/v5/summary/POSITION_READOUT_2026-09-12.md` §6 · §8 · CLAUDE.md §6 |
| 자 학습셋 v6 · v8 | v6 09-22 · v8 09-24 | v6 = v5 + 빈 장면 288 · v8 = 14,360 clip (presence 자) | v6 원천 재생성 불가 · 특징 캐시 09-26 삭제 | `datasets.md` |
| **RollOut_v3** | 09-22 렌더 | 3,136 clip = 8 법칙 × 392, 64 프레임, 사건 f32 고정, 16 창 | 현역 (auto_research H1–H9 · SC1 · RollOutV3) | `auto_research/Archive/PLAN_2026-09-24.md` 변경 1 · G09 타임라인 |
| `training_r8` (= 새 렌더 `RollOut_v2_training_v8`) | 09-25 ~ 09-26 | 8,640 clip; 감사 1 · 2 차 결함 → 3 차 통과 | **현 자 `identity_r8`** (위치 + 56 조합 + 없음; v11 판 빈 장면 오탐 0 %) | `z_research/RollOutV3/README.md` · `Archive/IDENTITY_R8_2026-09-26.md` |
| `Predictor_v1_training` | 날짜 문서에 없음 | 14,250 clip (무중력 등속 · 가림 k=4 · 경사 가속) — post-FT `vith_pft_predv1_e15` 학습셋 | 현역 (197 G, tar 없음) | `configs/protocols/models.md` · `HANDOFF_2026-09-22.md` §3 |

### 4-3. predictor · 모델 세대 (대조군)

| 무엇 | 날짜 | 비고 | 출처 |
|---|---|---|---|
| post-FT 넷 (v11 · IntPhys 1 · IntPhys 2 · Predictor_v1) | 09-10 ~ 09-19 | 전부 학습 도메인 안에서만 오른다 | `z_training/RESULTS_2026-09-19.md` |
| Ariel block-causal ep5 ~ ep43 (옛) | 09-23 수령 | **09-27 폴더 삭제** — REVIEW_v2 의 "가장 약한 고리" 실험 (epoch 포화) 은 그 체크포인트로 더 못 돌린다 | `configs/protocols/models.md` |
| Ariel 세 팔 (prefix ep45 · full ep40 · AR ep18 → ep38 / 40) · ctx_ar ep18 | 09-27 ~ 09-29 수령 | 레지스트리 `vith_ariel_ar_ep18` · `_ep36` 은 같은 경로 `ar_future_1e_5/latest.pt` 를 가리키고 그 파일이 ep36 (09-28 17:22) → ep38 (09-28 20:19) → ep40 (09-29 19:00) 으로 덮였다 — epoch 가 고정된 것은 사본 `ariel_eval/ckpt/ar_ep38.pt` · `ar_ep40.pt` (`models.md` `## vith_ariel_ar_ep36` · `ARIEL_CHECKPOINTS.md` §1-b · §5-1c) | `z_training/ARIEL_CHECKPOINTS.md` · `models.md` |
| jongseo 사전학습 8 개 | 09-25 등록 | ViT-Tiny · ViT-B, K400 / 합성 물리; 16 f @ 4 fps (시간 해상도 다름) | `models.md` · `z_research/TrainingEffects/README.md` |
| V-JEPA 2.1-g · VideoMAEv2-g | 09-20 ~ 09-21 | Garrido 격자 외부 벤치마크 비교 (IntPhys 1 51.11 / 54.44) | `z_research/Benchmarks/README.md` · `OVERVIEW` §5-8 |
| DINO-Foresight | 10-02 | IntPhys 1 84.4 vs 복사 88.9–92.2 · v11 77.10 vs 76.44 | `auto_research/Archive/DINOF_INTPHYS1_2026-10-02.md` · `DINOF_V11_2026-10-07.md` |

### 4-4. 외부 벤치마크 재측정 — **2026-09-21 이전 Benchmarks 수치는 전부 무효**

- 09-21 에 Garrido 프로토콜을 다시 읽고 전면 재작성했다: A.8 격자를 **모델마다 탐색** 해 최고를 보고 (Table S3 한 칸 `skip2_w16` = 83.33 만 돌린 것은 틀림) · IntPhys 은 C 스윕 대신 **Filtered** · 지표는 쌍 비교 AvgSurprise 하나 (**Max 철회**) · 창마다 실행 분리 · InfLevel 인덱스 · cross · priming 폐기 (공식 로더와 3 곳 달랐다). 무효 산출물은 `exp_results/_invalid_*` 에 보존 (8 가지 사유) (`z_research/Benchmarks/PROTOCOLS.md` · `z_research/OVERVIEW_2026-09-22.md` §7-2).
- 그 결과 V-JEPA 2 ViT-H IntPhys 1 = **88.89** (`skip2_w32`, Filtered) — 08-16 값과 같은 칸. 공식 config 칸 `skip2_w16` 은 83.33, 노트북 규칙은 90.56 (참고) (CLAUDE.md §2-2 · §5-4).
- IntPhys 2 는 별도 하네스이고 Benchmarks 의 범위 밖이다 (`analysis/intphys2/README.md`). 09-06 의 53.26 (ViT-L, IntPhys 1 68.89 파이프라인) 은 재측정 대상으로 남았고, 현재 값은 감사의 54.35 (C 24) / 복사 55.53 (C 42) 다 (`auto_research/Archive/INTPHYS2_PERMANENCE_AUDIT_2026-09-26.md`).
- 10-09 `cvpr/` 새 resolver 는 옛 resolver 와 같은 config 를 낸다 (28 행: 동일 27 · 차이 0 · 건너뜀 1) — 수치 세대를 바꾸지 않았다 (`cvpr/README.md` §1).

### 4-5. 삭제된 것 (되살릴 수 없다)

| 무엇 | 언제 | 남은 것 | 출처 |
|---|---|---|---|
| `IntPhys-Like_data/` 리포트 · figure | 08-28 | 없음 (git 추적 0) | CLAUDE.md §8-1 |
| 2D transit 프레임 (195 M) + 생성 코드 | 08-28 | 토큰 캐시 41 G (사후 분석만) | CLAUDE.md §4-3 |
| `configs/world_model_analysis/` 43 개 | 08-28 | 3 개 → 09-21 `configs/_archive/` | CLAUDE.md §2-6 |
| v6 ~ v10 · `v10_occlusion_low` · `v13_black` · 2D · RollOut_v1 프레임 | ~ 09-22 | 결과 · 일부 캐시 (`datasets.md` 의 `v10*` · `v13_black` 은 resolve 에서 죽음) | `z_research/HANDOFF_2026-09-22.md` §3 |
| RollOut_v1 프레임 · 캐시 · 결과 · 자 학습셋 v1 ~ v4 | 09-10 ~ 11 | 스크립트 기록 (`_superseded/`) | `z_research/scripts/_superseded/README.md` |
| `z_scripts/` (구버전 run 스크립트 139 개) | 09-21 | `build_intphys1_pairs.py` 만 공식 로직으로 복원 · v11 계열 index 생성기 (`build_probe_index.py`) 는 없음 → `data_csv/` 는 복사만 | CLAUDE.md §2-0 · `HANDOFF_2026-09-22.md` §2 |
| Ariel ep5 ~ ep43 체크포인트 · 옛 `z_training/ariel/README.md` | 09-27 | 원자료 (epoch 포화 결과) 만 | CLAUDE.md §3-1 · `auto_research/README.md` §0 (09-27) |
| 자 학습셋 v6 · v8 특징 캐시 (121 G · 210 G) | 09-26 | 인덱스 · 결과 | `configs/protocols/datasets.md` |

⚠️ `.gitignore` 가 `*csv · *.json · *.png · *.pt · z_exp/ · data_gen/ · */exp_results/` 를 막는다 — 인덱스 · 산출물 · 그림은 untracked 가 정상이고, `auto_research/` 는 통째로 git 추적 밖이다 (CLAUDE.md §8-1 · §9).

### 출처 (§4)
`evals/analysis_vlm/modes/REPRODUCTION_PLAN.md` · `z_world_model_analysis/{RESULTS_2026-08-16, FRAME_ABLATION_2026-08-19, OCCLUSION_LENGTH_2026-08-20, MECHANISM_2026-08-20}.md` · `z_research/IntPhysGenV10/Archive/V10_2026-08-28.md` · `z_research/IntPhysGenV11/README.md` · `z_research/IntPhysGenV11/Archive/surprising_score/RESULTS_2026-08-30.md` · `z_research/IntPhysGenV11_occlusion_timing_ablation/README.md` · `z_research/v11_realistic/README.md` · `configs/protocols/{datasets,models}.md` · `z_research/scripts/_superseded/README.md` · `z_research/RollOutV2/figures/v5/summary/POSITION_READOUT_2026-09-12.md` · `z_research/RollOutV3/README.md` · `z_research/RollOutV3/Archive/IDENTITY_R8_2026-09-26.md` · `auto_research/Archive/PLAN_2026-09-24.md` · `z_training/{README,RESULTS_2026-09-19,ARIEL_CHECKPOINTS}.md` · `z_research/TrainingEffects/README.md` · `z_research/Benchmarks/{README,PROTOCOLS}.md` · `analysis/intphys2/README.md` · `auto_research/Archive/{INTPHYS2_PERMANENCE_AUDIT_2026-09-26, DINOF_INTPHYS1_2026-10-02, DINOF_V11_2026-10-07}.md` · `z_research/{OVERVIEW,HANDOFF}_2026-09-22.md` · `cvpr/README.md` · `CLAUDE.md` §1-3 · §2 · §4 · §5 · §8-1 · §9

---

## 5. 옛 스토리에서 바뀐 것 · 지금도 유효한 것

### 5-1. 바뀐 것 (정본 §3 표, 10-08)

| 항목 | 옛 (출처) | 새 (2026-10-08) |
|---|---|---|
| 표제어 | "state 를 이어가지 않는다" (09-22) → "예측의 자리가 비어 있다 · world knowledge 는 predictor 에" (09-29) (둘 다 원 문서 표현 — 현재 금지 계열) | **"intuitive physics 를 따르는 plausible future prediction"** — 역할 분리 논지는 P3 마지막 문장 ("encoder 는 충분, predictor 가 실패") 으로 흡수 |
| 인간 · 영유아 비유 | 첫 문장과 discussion 에만 (DIRECTION §5) | **P1 전체 (intro 첫 문단).** 본문 판정은 측정 가능한 정의로 |
| 물리 장면의 지위 | "측정 도구 — 물리 엔진을 가졌는가 · 중력을 아는가를 묻지 않는다" (09-19) | intuitive physics 가 표제. **판정은 행동** ("낙하하는 미래를 예측하는가"), 내부 귀속은 계속 금지 |
| 네 능력 | state evolution 의 정의 (09-19) | **intuitive physics 의 조작적 정의로 유지** — 보이는 것 외삽 = 2 지속 · reach / 안 보이는 것 = 4 영속 / 없는 것 = 3a 추세 + 3b 사건 / encoder 정보 = 1 읽기 |
| ledge · wall · 꼭대기 | common sense 라 정의 밖 (09-19) | ✅ **포함** (사용자 10-08). 넣는 조건 넷 (정본 §2-3 iii) |
| memory · state 프레임 | 금지 (09-30, RETHINK §0-c) | **유지** |
| 벤치마크 감사 (복사 기준선) | 독립 beat · 기여 (DIRECTION §2 · RETHINK §5) | **P2 의 근거 — "IntPhys 1 은 leakage 가 있는 벤치"** + 방법의 판정 규칙 |
| 주 평가 | EK100 · IntPhys 2 · Ego4D · EgoExo4D (DIRECTION §1) | ⚠️ **결정 대기 2** — 기본값 intuitive physics 판 1 차, downstream 2 차 |
| action conditioning | "그다음" (DIRECTION §1) | 10-08: 작은 subsection + 실험 하나 → **10-10: 스토리 초안 · 아직 최종 아님** (`09b`) |
| 목표 | "state 모듈 / world knowledge 를 가진 predictor" | **Ariel 을 반드시 넘는다** (합격선 표, 정본 §2-4) |
| "frozen 위에서 닫힌다" | 08-31 원칙 → 09-06 폐기 | 폐기 유지 |
| 목적함수를 원인으로 | discussion 한 문단 | 유지 |
| 분업 | 분석 = Claude, 학습 = 사용자 | 유지 |

### 5-2. 지금도 유효한 것 — 판이 대체돼도 남는 다섯 가지

정본 머리: "옛 문서의 측정 수치 · 정정 · 기각표는 그대로 유효하다. 바뀐 것은 이야기의 틀과 목표다."

| 종류 | 무엇 | 어디에 있나 (원 문서) | 이 아카이브 |
|---|---|---|---|
| **수치** | v11 73.37 표 · vanish A/B 0/100 · k=1–4 평평 · probing p 98.46 / 99.49 / 100 · 이식 p 17.3 · 방향 비대칭 순서 · knockout −37.5 / −1.4 · timing 87.2 / 79.0 / 78.2 / 59.5 · 위치 자 0.87 칸 · 거리 한계 2–3 칸 · IntPhys 1 88.89 / 복사 85.00 · IntPhys 2 54.35 / 55.53 · post-FT 75.83 → 91.35 / 88.89 → 77.22 · EK100 35.34 / 19.07 | CLAUDE.md §5 · `RESULTS_2026-08-30.md` · `PAPER_STORY_2026-09-22.md` §2 · §4 · §5 · `OVERVIEW_2026-09-22.md` §5 · `auto_research/README.md` §0 | 02 · 04 – 08 장 |
| **정정** | 문서 안 `⚠️ 정정` 블록 전부 (예: 09-12 자 기본값 · 09-17 post-FT · 09-25 정정 1 · 2 · 09-26 2 차 · 10-01 EK100 AR 키 누락 · 10-08 gravity 정정판) | 각 원 문서 머리 · `README_UNTIL_2026-09-25.md` 정정 표 · `auto_research/README.md` 머리말 | 각 장 ⚠️ 줄 · 10 장 |
| **기각 · 철회표** | CLAUDE.md §6 기각 가설 (context 길이 · 희석 · 배경 · "p 에 정보가 없다" · "세기를 키우면 된다" · Luce · block 밖 retrieval · "마지막 관측 자리에 둔 채 머문다" 등) · `auto_research/README.md` §2 철회표 6 행 · `OVERVIEW` §7-2 철회 9 건 · `STORY_EVIDENCE` §2-4 다시 쓰면 안 되는 문장 | 같은 곳 | `10_REJECTED_AND_RETRACTED.md` |
| **금지 표현** | 정본 §7 표 (아래 5-3) + 옛 금지표 ("표현이 회전했다" · "z 와 h 가 같다" · "팔을 가로지르는 이식 실패가 증거" · "synthetic 에서 predictor 가 문제다" 등) | 정본 §7 · `PAPER_STORY_2026-08-31` · `-09-06` · `-09-22` 금지표 · `DESIGN_CHOICES` §4 | 모든 장의 문체 규칙 |
| **하네스** | 실험 제안 · 수락 질문 9 (정본 §6) · 자 검증 규칙 (09-12) · corner case 판정 (09-03) · 복사 기준선 동반 (09-24 ~) · 읽기 규칙 1–13 (`README_UNTIL` 1–10 · `auto_research/README.md` §6 11–13) · 고정 프로토콜 스펙 (CLAUDE.md §1-7) | 정본 §6 · CLAUDE.md §1 · §7 · §12 · `auto_research/README.md` §6 | 03 장 |

⚠️ 남는 수치에도 프로토콜이 붙어 있다 — IntPhys 1 88.89 (Garrido sliding `skip2_w32` Filtered) 와 66.67 (우리 fixed C16) 은 같은 데이터에서 22.2 pt 다르다. 프로토콜이 다른 숫자를 한 표에서 비교하지 않는다 (CLAUDE.md §1-3).
⚠️ 대체된 판의 **그림 수치** 일부는 이미 `_superseded/` 로 내려갔다 (예: 08-31 beat 4 Luce "cone 문맥에서 cube 77 %") — 판이 유효하다고 그 판의 모든 수치가 유효한 것은 아니다 (`z_research/IntPhysGenV11/figures/surprise/README.md`).

### 5-3. 금지 표현 (정본 §7, 10-08 갱신 — 요약)

| 쓰지 말 것 | 대신 |
|---|---|
| "world model 은 물리를 이해하지 못한다" · "intuitive physics 가 없다" (조건 없이) | "**마지막 관측에 없는 X 를 따르는 미래를 예측하지 못한다**" |
| "직관물리는 창발하지 않았다" · "IntPhys 1 은 틀린 벤치다" | "창발 근거인 IntPhys 1 은 leakage 가 크다 (예측 없이 85 %)" |
| "IntPhys 2 에서 chance" | "복사와 구분되지 않는다" (논문 수치 V-JEPA 2-h 57.51) |
| "모델이 중력을 안다 / 모른다 / 물리 법칙을 적용한다" | "낙하하는 미래를 예측하지 않는다" |
| "predictor 에 정보가 없다" · "X 를 못 만든다" | "맞는 미래로 놓지 않는다" (p probe 98–100) |
| "encoder 가 미래를 표현한다" | "미래를 만드는 데 필요한 정보는 입력에 선형으로 있다" |
| "기억 (memory) 문제다" · "state 를 잃는다" | 09-30 사용자 — 규칙 적용 · 지속 · 전이 · 영속 |
| "context 에서 안 보이면 못 한다" (전부로) | "**마지막 관측에** 안 보이면" (early 97.8 / mid 95.8) |
| "action-free 학습으로 해결된다" · "post-FT 가 intuitive physics 를 배웠다" | Ariel · post-FT 반례 — 차이를 대야 한다 / 도메인 안 궤적 prior |
| "AC 는 post 여야 효과가 있다" (사실로) | 가설 — subsection 실험 전까지 (`09b` §6) |
| "ledge 에서 불가능한 연속을 예측한다" | "낙하하는 미래를 만들지 않는다" (다중 미래 설계로) |
| "가림이 표현을 망가뜨린다" · "목적함수 때문이다" · "p 가 물체를 마지막 자리에 둔다" · "슬롯 N 이후 멈춘다" | env 72 칸 100 / discussion 한 문단 / 철회 — 자의 기본값 |
| α · 보정 · 헤지 계수 | 인용 금지 (사용자 지시 전까지) |

### 출처 (§5)
`auto_research/paper/PAPER_STORY_2026-10-08.md` 머리 · §3 · §6 · §7 · `CLAUDE.md` §1-3 · §1-7 · §5 · §6 · §12 · `z_research/IntPhysGenV11/Archive/{PAPER_STORY_2026-08-31, PAPER_STORY_2026-09-06, PAPER_STORY_2026-09-22}.md` · `z_research/OVERVIEW_2026-09-22.md` §5 · §7 · `auto_research/README.md` §0 · §2 · §6 · `auto_research/Archive/README_UNTIL_2026-09-25.md` · `auto_research/paper/{STORY_EVIDENCE_2026-09-26, DESIGN_CHOICES_2026-09-25, DIRECTION_2026-09-25}.md` · `z_research/IntPhysGenV11/figures/surprise/README.md`

---

## 6. 이 장의 단서

- 날짜는 원 문서 파일명 · 본문 날짜를 따랐다. IntPhysGen v2 ~ v7 · v9 · `Predictor_v1_training` 의 생성 날짜는 레포 문서에 없다 ("날짜 문서에 없음").
- 사용자 원문 인용 중 레포 문서에 없는 것 (09-15 · 09-19 "물리적 engine …" · 09-24 공식 · 09-25 "synthetic data 내에서 …" · 10-10) 은 사용자 auto-memory (레포 밖) 에서 왔다. 원 문서로 확인되지 않은 인용이다. 10-10 지시는 그중에서도 레포 문서에 아직 반영이 없다.
- 판별 표의 "왜" 열은 원 문서가 스스로 적은 이유다. 원 문서가 이유를 적지 않은 경우 (09-06 frozen 폐기의 결정 주체 등) 는 "문서에 없음" 으로 적었다.
- 이 장은 새 계산을 하지 않았다. 수치는 전부 원 문서 · 정본에서 옮겼고, ✓ 표시는 정본 §재현 (2026-10-08 재확인) 의 표시를 따른다.

---

## 출처

정본 · 운영 규칙
- `auto_research/paper/PAPER_STORY_2026-10-08.md` — §0 원문 · §1 · §2 · §2-2 · §2-3 · §2-4 · §2-5 · §3 · §4-3 · §5 · §6 · §7 · §8 · §재현 (정본, 직접 확인)
- `CLAUDE.md` — §0 · §1 · §2 · §4 · §5 · §6 · §8 · §9 · §10 · §12

옛 스토리 판 (대체됨)
- `z_research/IntPhysGenV10/Archive/PAPER_STORY_2026-08-29.md`
- `z_research/IntPhysGenV11/Archive/{SYNTHESIS_2026-08-30, TALK_2026-08-30, PAPER_STORY_2026-08-31, STATUS_AND_CRITIQUE_2026-09-03, PAPER_STORY_2026-09-06, PAPER_STORY_2026-09-22}.md`
- `z_research/context_encoder_analysis/Archive/{CLAIM_NO_STATE_EVOLUTION_2026-09-15, STATE_EVOLUTION_HYPOTHESES_2026-09-15}.md`
- `z_research/{OVERVIEW,HANDOFF}_2026-09-22.md`
- `auto_research/Archive/{PLAN_2026-09-24, CROSS_REVIEW_2026-09-24, README_UNTIL_2026-09-25}.md`
- `auto_research/paper/{DRAFT_v0_2026-09-25, DRAFT_v1_2026-09-25, DRAFT_v2_2026-09-25, REVIEW_v1_2026-09-25, ORAL_DIRECTION_2026-09-25, DESIGN_CHOICES_2026-09-25, DIRECTION_2026-09-25, ABSTRACT_v0_2026-09-25, STORY_EVIDENCE_2026-09-26, REVIEW_v2_2026-09-26, RETHINK_PREDICTOR_ROLE_2026-09-29, NARRATIVE_v2_USER_2026-09-29, ABSTRACT_v1_2026-09-29}.md`
- `auto_research/README.md` — 머리말 (09-25 → 09-29 → 10-08) · §0 · §2 철회표 · §3 · §5 · §6

현행 정의 · 하네스
- `z_research/context_encoder_analysis/Archive/STATE_EVOLUTION_CAPABILITIES_2026-09-19.md` (네 능력, 10-08 머리 번복)
- `z_research/context_encoder_analysis/README.md`
- `z_training/README.md` · `z_training/RESULTS_2026-09-19.md` · `z_training/ARIEL_CHECKPOINTS.md`
- `z_AC_training/README.md` · `z_AC_training/Archive/PLANNING_AND_ROBOCASA_2026-09-20.md`
- `z_research/anticipation/EK100/README.md`
- `configs/protocols/datasets.md` · `configs/protocols/models.md`
- `cvpr/README.md`

결과 · 근거 문서 (이 장이 표 안에서 가리킨 것)
- `evals/analysis_vlm/modes/REPRODUCTION_PLAN.md`
- `z_world_model_analysis/{intphys1_dev_2026-08-13, occlusion_v2_2026-08-13_PREVIOUS-DATA, probe_set_2026-08-13_PREVIOUS-DATA, RESULTS_2026-08-16, FRAME_ABLATION_2026-08-19, TOKEN_SUBSET_2026-08-19, MECHANISM_2026-08-20, WHY_INTPHYS_WORKS_2026-08-20, VERIFICATION_2026-08-20, OCCLUSION_LENGTH_2026-08-20, PREDICTOR_HEDGING_2026-08-26}.md`
- `z_research/IntPhysGenV10/Archive/{V10_2026-08-28, GEOMETRY_2026-08-29}.md`
- `z_research/IntPhysGenV11/README.md` · `z_research/IntPhysGenV11/Archive/surprising_score/RESULTS_2026-08-30.md` · `z_research/IntPhysGenV11/figures/surprise/README.md`
- `z_research/IntPhysGenV11_occlusion_timing_ablation/README.md`
- `z_research/RollOutV2/figures/v5/summary/POSITION_READOUT_2026-09-12.md` · `z_research/RollOutV2/Archive/PREDICTOR_DISTANCE_LIMIT_2026-09-19.md`
- `z_research/RollOutV3/README.md` · `z_research/RollOutV3/Archive/IDENTITY_R8_2026-09-26.md` · `z_research/RollOutV3/Archive/_superseded/NEW_ARCHIVE_README_2026-09-25_r1.md`
- `z_research/scripts/_superseded/README.md`
- `z_research/v11_realistic/README.md` · `z_research/TrainingEffects/README.md`
- `z_research/Benchmarks/{README,PROTOCOLS}.md` · `analysis/intphys2/README.md`
- `auto_research/Archive/{H6_INTPHYS1_WHAT_THE_SCORE_MEASURES_2026-09-24, H9_FROZEN_INTERVENTION_SCORING_2026-09-24, SC1_REACH_NOT_HORIZON_2026-09-25, E_SSV2_ENCODER_READS_PREDICTOR_STALLS_2026-09-25, I_SCENE_CAUSAL_BOTTLENECK_2026-09-25, INTPHYS2_PERMANENCE_AUDIT_2026-09-26, DINOF_INTPHYS1_2026-10-02, COPY_VS_POSITION_2026-10-07, DINOF_V11_2026-10-07}.md`

원재료 (추출문): `scratchpad/archive10/extracts/G05_paper.md` · `G06_v11_core.md` (그룹 요약 (c)) · `G07_v11timing_v10_wma.md` · `G12_ctxenc.md` (그룹 요약 (c)) · `G13_ac_overview_misc.md` · `G03_auto_late.md` (`auto_research/README.md` 항목) · `G02a_auto_scene_M.md` (`README_UNTIL_2026-09-25.md` 항목) · `G09_rollout_superseded_audit.md` (타임라인) · `G10_training.md`
