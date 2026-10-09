# auto_research — world model 은 intuitive physics 를 따르는 미래를 예측하지 못한다: predictor 분석 결과 로그

> **🆕 2026-10-08 논문 스토리 교체 (사용자). 정본 → [`paper/PAPER_STORY_2026-10-08.md`](paper/PAPER_STORY_2026-10-08.md).**
> P1 사람은 관측만으로 intuitive physics (사라지지 않음 · 중력) 를 갖추고 plausible future 를 상상한다 →
> P2 world model 은 빠르게 발전하지만 intuitive physics 를 따르는 미래는 예측 못 한다 (되는 것처럼 보이는 VoE 점수는 복사로 풀린다; Figure 1 hook) →
> P3 통제 testbed: 마지막 관측에 **보이는 것만** 짧게 외삽하고 **안 보이는 것** (가림) · **없는 것** (가속 · 낙하 · 꼭대기) 은 못 한다; encoder 에는 정보가 있고 predictor 가 실패한다 →
> 목표 (★ 사용자 학습): action 없이 관측만으로 그것을 배우는 predictor, AC 는 post-training (★ 가설).
> 이 README 는 **결과 로그 · 철회표 · 문서 지도** 로 남는다. 아래 09-25 ~ 10-07 머리말과 §0–§4 의 "논지 · 프레임 · 결정 대기 · 다음" 은 옛 스토리 기준이다 —
> 수치 · 정정 · 철회는 유효하고, 틀 · 다음 할 일 · 하네스는 정본 §3 · §5 · §6 을 따른다. 10-08 사용자 결정: IntPhys 1 은 leakage 벤치 (복사로 잰다) · ledge · wall · 꼭대기 포함 (3b 사건) · Ariel 은 반드시 넘는다 (합격선 정본 §2-4) · AC 는 작은 subsection + 실험 하나. 남은 결정 대기: 주 평가 (정본 §3).
> 옛 제목: "새 latent video predictor 의 design choice: pretrained predictor 는 마지막 관측에서 먼 내용을 attention 으로 가져오지 못해 장면을 멈춘다 → 움직임 정렬 조회 · 미래 전용 목표 · 자기 앵커 닫힘 학습"

> **2026-09-29 논지 재정리: [`paper/RETHINK_PREDICTOR_ROLE_2026-09-29.md`](paper/RETHINK_PREDICTOR_ROLE_2026-09-29.md) — "JEPA world model 에서 예측의 자리가 비어 있다": encoder 는 표현을 다 하고 predictor 는 마지막 관측을 옮겨 적으며, downstream 은 그것으로 충분하다 (복사 대비 Δ ≤ +4). world knowledge 는 predictor 에 있어야 하므로 frozen encoder + 새 predictor (실영상 video prediction) 가 주인공. DIRECTION §0-b. 초록 v1 [`paper/ABSTRACT_v1_2026-09-29.md`](paper/ABSTRACT_v1_2026-09-29.md). 문헌 [`Archive/LITERATURE_PREDICTOR_ROLE_2026-09-29.md`](Archive/LITERATURE_PREDICTOR_ROLE_2026-09-29.md) (같은 대상 · 같은 복사 귀무로 잰 선행 없음; 부분 선점 Human-JEPA · GEOPHYS · Garrido 2026; V-JEPA 2 Table 20 predictor 몫 +0.6). **encoder 운동 판독** [`Archive/ENCODER_MOTION_READOUT_2026-09-29.md`](Archive/ENCODER_MOTION_READOUT_2026-09-29.md): EK100 ego-motion R² .69/.63 (검증 3 차: 10-fold · 영상 split 에서도 .61–.71; flow 자기 일관성 clip-CV .53/.36 보다 높음), 미래 변위의 67–75 % 가 카메라 운동 → EK100 병목도 encoder 아님 (확정 1 차). **copy-hard 부분집합** [`Archive/COPY_HARD_SUBSET_2026-09-29.md`](Archive/COPY_HARD_SUBSET_2026-09-29.md): 복사가 틀리는 쌍에서 릴리즈 · prefix · full 은 우연 (v11 48 · IntPhys 1 44 · IntPhys 2 21), AR 은 v11 74 · late vanish→빈 72–99. 검증 3 차 정정: 무작위 겹침 귀무에서 copy-hard = 전체 정확도이고 모두 그 아래 (release −27, AR −10) → AR 의 값은 "복사가 못 푸는 걸 푼다" 가 아니라 "오류가 복사와 덜 묶여 있다" 로 읽는다 presence 검정 (09-29 밤): AR 의 late 정답은 **presence-without-placement** (적대 검증 완료: 재등장 칸 음수는 릴리즈 · prefix 도 같고, AR 만의 것은 먼 칸 ≥ 3 의 전역 양수 오프셋; margin 지도가 복사 지도와 r .59; 문맥 편향 성분도 있음); ctx_ar ep16 은 빈 문맥에서도 물체 미래를 고름 (empty 쌍 21 %, 편향). `AR_PREDICTOR` §2-6. predictor 의 몫 표 [`Archive/PREDICTOR_SHARE_2026-09-29.md`](Archive/PREDICTOR_SHARE_2026-09-29.md): 복사 대비 Δ 는 모든 벤치마크에서 +2 ~ +7 pt (IntPhys 1 +3.9 [−0.6, +8.3] · AR 공간 +7.2 [+2.2, +12.8] · IntPhys 2 구분 안 됨 · v11 +2.5 · EK100 +2.1~2.5, CI 없음).**
> **갱신 2026-09-25 밤. 이 README 가 유일한 진입점이다. 연구 방향의 정본은 [`paper/DIRECTION_2026-09-25.md`](paper/DIRECTION_2026-09-25.md) (사용자 결정: 목표 = 인간 같은 latent world model, 주 평가 EK100 · IntPhys 2 · Ego4D · EgoExo4D, 합성은 testbed). 실험은 그 §3 여섯 질문을 통과해야 한다.**
> **🆕 프레임 (09-25 저녁, 사용자): 주인공은 새 video predictor 다.** 분석은 design choice 의 근거다. 정본 → [`paper/DESIGN_CHOICES_2026-09-25.md`](paper/DESIGN_CHOICES_2026-09-25.md) (DC1 움직임 정렬 조회 · DC2 미래 전용 목표 · DC3 자기 앵커 · DC4 있음 · DC5 reach 평가; 학습 계획 M → N → O → P → AC fine-tune Q).
> "synthetic 에서 predictor 가 문제다" 는 쓰지 않는다.
> 09-24 ~ 09-25 오전의 옛 본문은 [`Archive/README_UNTIL_2026-09-25.md`](Archive/README_UNTIL_2026-09-25.md) 로 옮겼다. 거기에 있는 것: 정정 1 · 2 표, 적대 검증 2 차 뒤 논리 사슬, 주장 등급, 읽기 규칙 10 개, 실험 현황 전체, 원자료 위치, 재현 명령.
> 등급: **확정** · 시사 · **판정 보류** · **철회**. **1 차** = 적대 검증 전 (09-25 오후 결과는 전부 1 차다).

## 0. 한 장 요약

**결론.** frozen V-JEPA 2 ViT-H encoder 위의 predictor 는 관측이 끊기면 물체를 **마지막 관측 자리에서 일정 거리까지만** 옮기고 멈춘다.
- ~~그 거리 (reach) 는 걸음 수 · fps · 화면 위치와 무관하다.~~ → 멈춤은 **걸음 수가 아니라 거리 쪽으로** 모인다 (고정 걸음 수 모형 R² 0). 화면 위치와는 무관하다. 다만 release 는 순수 거리가 아니다: 약 2 칸 + 1.6 걸음이 섞여 있고, 한 stride 안에서 멈춘 자리가 속도에 비례한다. 순수 거리에 가까운 것은 학습된 predictor (Ariel) 다.
  ⚠️ 정정 (같은 날 오후) — "걸음 수 · fps 와 무관한 고정 거리" 는 **release 에 대해 과장**이었다. release 는 한 stride 안에서 멈춘 자리가 속도에 비례하고 (빠름/느림 비 1.96–2.07 vs 속도비 1.85, `REVIEW_CVPR_ORAL` K4 확정), stride 사이에서는 plateau 가 2.3 → 3.2 → 4.9 칸으로 δ 에 따라 오른다 (약 2 칸 + 1.6 걸음). 순수 거리에 가까운 것은 학습된 predictor (Ariel plateau 5–6.7 칸) 다. 기각된 것은 "고정 걸음 수" 모형이다 (R² 0).
- 미래 예측으로 학습한 predictor 는 reach 가 더 길지만 (학습량에 따라 는다), 여전히 한계가 있다.
- 흔한 평가 (IntPhys VoE, EK100 1 s anticipation) 는 이 한계 안쪽만 재서 차이를 보지 못한다.

사용자 상위 메시지 "a true latent world model must evolve state even when it can't see" 를 잴 수 있게 쓰면 이렇다: **"관측 없이 운반하는 거리가 변위와 무관해야 한다."** world video prediction 모델의 설계 목표도 이것이다.

**근거 (핵심 다섯 줄).**

| # | 무엇 | 수치 | 등급 | 문서 |
|---|---|---|---|---|
| 1 | 멈춤의 단위는 걸음이 아니라 **거리** (합성 v3, stride 1/2/4 × 분할 16/32 × predictor 넷) | 절벽 위치를 "슬롯 상수" 로 설명하면 R² 0, "거리 상수" 로 설명하면 0.68–0.84. Ariel 은 stride 1 에서 13–17 슬롯 (학습 지평 8 의 2 배), stride 4 에서 2–3 슬롯에 멈추는데 거리는 6–7 칸으로 같다. 분할점을 옮겨도 거리가 같다 | 1 차 | [SC1](Archive/SC1_REACH_NOT_HORIZON_2026-09-25.md) §3-1–3-4 |
| 2 | 학습하면 reach 가 는다, 같은 run 안에서도 | 예측 변위 plateau (stride 1, s1_C16): release 2.3 칸 → Ariel ep5 / 19 / 30 / 43 = 3.5 / 4.3 / 4.7 / 5.1 칸 | 1 차 | SC1 §3-6 |
| 3 | 실영상 (SSv2) 에서도 변위로 떨어진다 | flow 유사 정답과 encoder 판독이 합의한 2,031 쌍에서, 적중이 변위 0–1 → 4–6 칸 사이에 release 0.73 → 0.13, Ariel 0.85 → 0.37 (같은 쌍 +0.24 [0.18, 0.30]). flow 판독은 v3 에서 GT 로 보정했다 | 1 차 | [SC2v2](Archive/SC2V2_FLOW_REACH_SSV2_2026-09-25.md) |
| 4 | 기전 후보: attention 의 거리 감쇠 | 진실 칸 미래 질의 → 문맥 물체 key 의 log attention: 칸당 release −0.42, pv1 −0.30 (CI 안 겹침). 슬롯 몫은 release −0.096, pv1 ≈ 0 으로, 행동의 걸음 몫과 같은 방향 | 1 차 · 상관 | SC1 §3-5 · [M4](Archive/M4_RETRIEVAL_RADIUS_2026-09-25.md) |
| 5 | 흔한 평가는 못 본다 | IntPhys 1 최고 칸: 마지막 문맥 복사 85.00 vs 릴리즈 88.89 (+3.9 [−0.6, +8.4], n.s.). **IntPhys 2 Main (506 쌍): 복사 55.5 vs 릴리즈 54.3, permanence 고정 카메라 63.5 vs 59.6 (각자 최고 C) — 같은 C 에서는 +1.8 [−1.8, +5.5] 로 **구분 안 됨**, 릴리즈 permanence 는 전 조건 CI 가 50 포함** ([`INTPHYS2_PERMANENCE_AUDIT`](Archive/INTPHYS2_PERMANENCE_AUDIT_2026-09-26.md), 1 차). EK100: 같은 probe 에서 predictor 미래를 복사 · 평균 토큰으로 바꿔도 action R@5 −1.3 ~ −2.6 (35.34 → 32.80 / 34.01; paper 규약 19.07 → 16.46 / 17.59) | 확정 (IntPhys) · 1 차 (EK100) | [H6](Archive/H6_INTPHYS1_WHAT_THE_SCORE_MEASURES_2026-09-24.md) · [SC3](Archive/SC3_EK100_PREDICTOR_SUBSTITUTION_2026-09-25.md) |

**🆕 병목의 인과 규명 (09-25 저녁, 1 차 · 적대 검증 전) → [`Archive/I_SCENE_CAUSAL_BOTTLENECK_2026-09-25.md`](Archive/I_SCENE_CAUSAL_BOTTLENECK_2026-09-25.md)**
- 검정 판: 물체 하나가 아니라 **장면 전체**. K400 val 1,600 clip 위에 인공 카메라 팬을 걸어 모든 토큰의 출처가 정확한 판이다. predictor 셋에 frozen 개입을 걸었다.
- **병목은 predictor 의 미래 질의가 먼 문맥 출처를 가져오는 attention 강도다.**
  - 방향은 안다 (진짜 출처를 같은 거리 대조보다 2–22 배 본다). 그러나 강도가 거리로 약 15 배 준다.
  - 그 강도만 올리면 reach 가 세기에 비례해 는다 (release 4–6 칸 +0.22, Ariel 6–9 칸 +0.43). 반대 방향으로 올리면 준다.
  - Ariel 이 release 보다 나은 몫의 **약 절반 (거리 층화 44–67 %)** 이 이 강도로 설명된다 (매개, 시사; 적대 검증 [`VERIFY_ISCENE`](Archive/VERIFY_ISCENE_2026-09-25.md) 로 95 % 에서 낮춤).
- **양방향 attention 규칙은 병목이 아니다** (release 에 prefix 규칙: +0.02).
- **encoder 도 아니다.** 진짜 출처의 encoder 내용을 끌어오면 먼 자리에 맞는 내용이 나온다.
- 자연 영상에서는 일반적인 날카롭히기 · 복사 억제로 안 고쳐진다 (≤ +0.03). 실영상에서는 출처 방향 자체도 약하다 (토큰 중앙값 선택성 < 1) → 새 predictor 는 출처 추정과 조회 강도를 둘 다 학습해야 한다 (DC1).
- 심층 (I-scene §4-3, SC1 §3-7, 1 차): (B) reach 밖 출력은 **절반은 제자리 복사, 절반은 흐려진 토큰**. (C) 실영상에서 움직이는 내용의 **약 절반 (SSv2 46 %, EK100 51 %)** 이 release 의 reach 밖 — corner case 가 아니다. (D) 방향을 아는 토큰은 첫 슬롯 근처 · 국소적으로 일관되게 움직이는 토큰이고, 조회는 중간 층에서 가장 선택적이다. (E) 거리를 고정하면 시간 효과가 없고 시간을 고정하면 거리 효과가 뚜렷하다 (부분 R² 7–130 배) → 단위는 거리. (F) Ariel 우위의 75–95 % 는 "출처를 더 봄" 칸에서 나오지만 토큰 상관은 약하다 → 매개 약 절반.
- 학습 검정 (I-scene §5-1): 인공 팬 증강 post-FT (Ariel 규칙) 는 팬 평가에서만 reach 를 늘리고 (6.5 → 8.7 칸) SSv2 · EK100 에서는 변화 없음 → 데이터 커리큘럼은 도메인 안 한정.
- 단서: 필요성 개입은 약하다 (중복 경로). RoPE 축척 개입은 해석 불가. 연속 위치 판독 (cos 지도, 묶음 P) 은 대기 중. M2 (release post-FT) · O (mret) 는 vll6 대기 중.

**🆕 09-26 오후 (1 차, vll3 묶음 Q · R · T · SC5 · IntPhys 2 감사):**
- **permanence 실영상 장면 판** (K400 팬 + 마지막 k 프레임 가림 띠): 가려졌던 내용은 같은 clip 의 보이는 내용의 **절반** 만 제자리에 복원되고 (k=4 근거리 release .21 vs .42), predictor 는 원래 내용보다 **가림막을 그린다** (cos .39 vs .57). release · Ariel · pv1 · 팬 post-FT 모두 같다.
- **closure 실영상 장면 판**: 자기 예측 되먹임은 one-shot 보다 낫지 않고 (release −0.03 ~ +0.01), 미래 예측 학습 predictor 는 오히려 −0.12 ~ −0.22. 진짜 관측 되먹임은 +0.1 ~ +0.2.
- **Ariel epoch × reach 포화**: 팬 R50 4.3 (ep5) → 5.7 → 6.3 → 6.5 → 6.5 (ep43), SSv2 3.5 → 3.9 → 4.1 → 4.0, EK100 2.2 불변 → "학습 부족" 반론 해소는 **EK100 한정** (⚠️ 2 차 검증: cosine lr 끝과 팬 포화가 구분 안 됨).
- **EK100 지평 스윕**: predictor 고유 몫 Δaction 1 s +2.5 · 2 s +2.1 · 3 s +2.1 (paper 2 s +2.8) — 지평과 무관하게 약 2 점.
- **IntPhys 2**: 복사와 release 를 **구분할 수 없다** (⚠️ 2 차 정정: "복사 ≥" 는 best-C 산물, 같은 C 에서 +1.8 [−1.8, +5.5]; release permanence 전 조건 CI 가 50 포함; w16/32/48 모두).
- **질의 위치 평행이동 (묶음 P)**: 미래 질의의 RoPE 위치를 출처 자리로 옮기면 실영상 먼 거리 적중이 거리와 무관해진다 (EK100 release 4–6 칸 .03 → .47; SSv2 .06 → .35). ⚠️ 2 차 검증으로 **시사로 강등**: 먼 적중은 base 근거리의 53–72 % (정확 칸 .08), 팬 근거리 −0.2 비용, 출력이 출처를 지나치는 토큰 32–58 % (v3 는 같은 기전). 인과 추정 (등속 외삽) 판은 무효.
- **연속 위치 판독 (cos 지도)**: 허용폭 없이 재도 거리에 따라 진행률이 준다. **실영상에서는 경과 시간 효과가 거리 효과와 같은 크기** (SSv2 슬롯당 −0.023 vs 칸당 −0.027) → "거리가 주 변수" 는 Chebyshev 판독 한정으로 좁힌다 (⚠️ 2 차: τ 의존 큼, 귀무 ρ = 0.5, 팬도 둘을 못 가른다).
- **🆕 09-27 저녁 — 자기회귀 (AR) predictor 판독 (1 차)**: rollout 목표로 학습한 Ariel AR (ep18) 의 reach 가 **가장 짧다** (R50 팬 3.2 · SSv2 2.2 · EK100 1.8 · v3 4.2 vs 릴리즈 4.8 · 2.7 · 2.1 · 4.7 vs full/prefix 7.0 · 4.1 · 2.2 · 7.5). 정지 내용은 AR 이 최고 (v3 0 칸 +.26), 결손은 움직이는 내용에만. IntPhys 1 AR 공간 복사 기준선 83.33 도착 (AR 90.56 = 복사 + 7, 릴리즈 = 복사 + 4, 차이는 잡음 안). 진짜 블록을 한 걸음씩 주면 9.2 · ∞ 로 잘 옮기지만 자기 예측을 되먹이면 3 걸음 안에 뭉개진다 (출처 < 복사 유사도). full ≈ prefix (규칙 무관, 학습 판 확인). 같은 AR 이 IntPhys 1 에서 90.56 으로 최고 → "벤치마크가 이 능력을 안 잰다" 의 가장 날카로운 쌍 (AR 복사 기준선 대기). permanence 도 절반 (hid/vis 비 .47 = full/prefix, 릴리즈 .53 보다 낮음). **ep40 최종 ≡ ep38; ctx_ar ep16 (문맥을 창 단위 z 로 되돌린 팔) 도 같은 reach · 같은 붕괴 · 같은 permanence, 다만 AR 의 정지 우위는 사라짐 (prefix 보다도 낮음, 검증 3 차 확정); **벤치마크도 낮다 — IntPhys 1 64.4 · v11 61.0 (AR ep18 90.6 · 83.2)**. "문맥 인코딩은 원인 아님" 은 시사 (ep16) — 안전한 문장: 창 단위 z 문맥으로 되돌려도 되먹임 reach · 붕괴 · permanence 는 회복되지 않는다 (`AR_PREDICTOR` §2-7).** **ep38 (학습 완료) 재판독으로 결론 유지** (팬 3.9 · SSv2 2.2, 여전히 릴리즈 아래; perm 비 .46). 적대 검증 (`Archive/VERIFY_AR_2026-09-27.md`): reach 최단 **확정** (교집합 · 정확 칸 · 느린 팬 모두), 붕괴는 자기 되먹임 첫 걸음부터, "뭉개짐" 은 릴리즈와 같은 정도 (시사). `Archive/AR_PREDICTOR_2026-09-27.md`. **10-01 EK100 probe (1/3 train · 20 ep · 같은 val 8,761): AR ep40 action 22.03 vs 릴리즈 21.80 (재현 잡음 0.06–0.4 안) — predictor 를 rollout 학습판으로 바꿔도 EK100 probe 는 안 움직인다** (§2-9; 09-30 의 'AR 21.74' 는 키 누락으로 릴리즈가 돈 것, 인용 금지).
- **🆕 09-30 AR ep40 의 p 를 새 정체 자로 읽음** (`Archive/AR40_READOUT_2026-09-30.md`, vll1 스테이징 실행): 가속 반영 α 가 arc −0.42 · ledge −0.46 (분포 안 t0–7, 모든 정의 변형에서 음수; 릴리즈와 같거나 나쁨; h ≈ 1) → **중력 · 가속을 미래에 넣지 않는다** (검증: flat_a n.s., ramp/감속은 분포 밖 값이라 제외, α 절대값은 자 의존); late 가림 뒤 움직이는 물체 토큰 0–10 % (릴리즈 40–90 %, 자리 틀림) → late 채점 정답은 물체 토큰이 아님 (presence 검정과 정합). 자 검증 test 91.1 (릴리즈 자 84.0). 1 차.
- **🆕 09-27 Ariel 체크포인트 교체** (다른 세션 통지): 옛 ep5–43 삭제 → `block_causal_future_only_1e_5` (ep45) · full-attention 대조 `full_attn_future_pred_1e_5` (ep40) · **자기회귀 `ar_future_1e_5` (ep18/40 학습 중)**. §5-3 epoch 포화는 재실행 불가 (원자료만). `reach_k400_{prefix,mret}.yaml` · `e_scene_reach.py` 는 ep45 로 옮겼다 (DRYRUN 통과). AR 이 끝나면 reach · permanence · closure 를 AR 에 걸어야 한다 (closure 의 결정 실험).
- **release 팬 증강 post-FT (M)**: 팬 6.3 → 10.2, SSv2 +0.2, EK100 +0.0 (N 과 같음). 미래 전용 post-FT 자체가 release 를 팬 · EK100 에서 Ariel 수준까지 올리고 (SSv2 는 간격의 52 %) 그 위 한계는 같다. 2 차 검증 전문: [`VERIFY_ISCENE2`](Archive/VERIFY_ISCENE2_2026-09-26.md).

**미결 (결론과 같은 무게).**
- 거리와 경과 시간을 떼어 내지 못했다. 이 설계에서 둘이 같이 움직인다.
- 실영상에서는 걸음 수에 따른 감쇠도 있다 (슬롯당 −0.04 ~ −0.06).
- 기전은 상관까지다. Ariel attention 은 아직 못 쟀다 (hook 이 prefix 마스크를 덮어쓴다).
- 1 차 결과 전부가 적대 검증 전이다.

## 1. 지금 서 있는 이야기 (논문 뼈대 A′) — 단계별

| 단계 | 문장 | 등급 | 어디 |
|---|---|---|---|
| 읽기 | predictor 입력 z 에서 가시 물체의 방향 · 속력이 선형으로 읽힌다 (SSv2 방향 0.89–0.97 vs 픽셀 0.41–0.70; v3 법칙 안 속력 R² ≈ 0.95). "encoder 가 여러 프레임을 경계 토큰에 적분한다" 는 가설이다 | 확정 (가시) · 적분은 판정 보류 | E-SSv2 · M13 · M6 |
| 한 걸음 전이 | 진짜 다음 관측을 한 튜블릿씩 넣으면 물체가 새 칸으로 간다 (기울기 0.361, 진실 0.440). 네 predictor 가 거의 같다 (0.355–0.388). 미래 프레임을 쓰는 **탐침** 이지 예측기가 아니다 | 확정 | P3 · H5 |
| 관측이 끊기면 | 릴리즈 one-shot: 약 3 튜블릿을 진실 이상 속도로 간 뒤 멈춘다 (경로 기울기 0.65 → 0.07; 멈춘 자리 b 4.60 vs 진실 6.07). 자기 예측을 이어붙이면 두 걸음 뒤 멈춘다 (−0.012 [−0.042, +0.018]). 학습된 predictor 의 사슬은 진실의 1/3–1/2 (Ariel 0.226, pv1 0.139) | 확정 (행동) | TRAINED · P3 · P3B |
| 단위 = 거리 | §0 근거 1 | 1 차 | SC1 |
| 학습 효과 | §0 근거 2. post-FT (pv1 · v11_postft) 는 1 epoch 만에 도달하고 평평하다 | 1 차 | SC1 §3-6 |
| 실영상 | §0 근거 3. 합성으로 학습한 pv1 · v11_postft 는 SSv2 에서 release 와 거의 같다 | 1 차 | SC2v2 |
| 왜 | §0 근거 4. 다른 후보는 학습 데이터 변위 분포 (pv1 학습셋 8 슬롯 최대 변위 중앙값 3.76, p75 5.67 칸) 다. 이 후보를 가르려면 학습이 필요하다 | 1 차 · 상관 | SC1 §4 |
| 평가 | §0 근거 5 | 확정 · 1 차 | H6 · SC3 |
| 처방 | reach 가 변위에 무관한 predictor. 후보: (i) 운반형 (latent warp; 학습 없는 등속 오라클이 실영상 2–4 칸에서 이미 release 를 이긴다), (ii) 자기 예측을 앵커로 되먹이는 닫힘 학습, (iii) 큰 변위 커리큘럼 | **학습 필요 → 사용자 결정** | [ORAL_DIRECTION](paper/ORAL_DIRECTION_2026-09-25.md) §4 |

## 2. 철회 · 기각 (다시 쓰지 말 것)

| 옛 문장 | 왜 | 어디 |
|---|---|---|
| "encoder 는 충분하고 predictor 가 병목이다" | 충분성 대조가 진실 위치를 써서 이를 재지 못했다. 분업 귀속은 닫히지 않았다 | 옛 README 정정 1 · TRAINED 정정 2 |
| "학습된 predictor 는 학습 지평 (8 슬롯) 까지만 외삽한다" (oral 첫 안 A) | SC1: 슬롯 모형 R² 0, Ariel 은 stride 1 에서 13–17 슬롯까지 간다 | SC1 · ORAL_DIRECTION §7 |
| "미래를 1 튜블릿씩 예측해 이어붙였더니 된다" | 된 줄 (`ar_z`) 은 매 걸음 진짜 관측을 넣은 탐침이다. 자기 예측을 이어붙이면 멈춘다 | 옛 README "미래를 튜블릿 1 개만…" 절 · P3 |
| 기전 귀속 셋: 마지막 토큰이 정한다 · 역사 토큰이 뒤로 끈다 · 속도를 한 걸음만 쓴다 | 판독 · 이음 교란 · 조향 방향 문제 (적대 검증 2 차) | P3B · M6 · M5 정정 2 |
| P2 반경 공식 "2.64 칸 + 1.89 걸음" | 사후 모형 선택. 기술식으로만 남는다 | P2 정정 |
| SC2 1 판: encoder 템플릿 궤적을 유사 정답으로 | 궤적이 슬롯 사이 평균 5.5 칸씩 뛰었다 → flow 판으로 대체 | SC2v2 §1 |

나머지 철회 · 좁힘 목록은 옛 README 의 정정 1 · 2 표에 있다.

## 3. 결정 대기 (사용자) — ⚠️ 09-25 기준. 현행 결정 대기는 정본 §3

1. **world video prediction 모델을 학습할지, 한다면 어느 후보로** (§1 처방 행). 결정 전까지 학습하지 않는다 (09-25 "새로 학습은 말고").
2. **커밋.** `auto_research/` 전체가 아직 git 추적 밖이다.

## 4. 다음 (학습 없이, 우선순위) — ⚠️ 09-25 기준. 현행은 정본 §5

| 순 | 무엇 | 왜 | 비용 |
|---|---|---|---|
| 0 | **I-scene 인과 개입 (묶음 K · L) + 학습 검정 (묶음 M)** | 병목 규명 (사용자 우선순위) → 학습으로 검정 | 진행 중 (vll6) |
| 1 | SC1 · SC2v2 · SC3 · 묶음 G 적대 검증 | 새 뼈대 전체가 1 차다 | CPU |
| 2 | Ariel attention 거리 감쇠 (hook 을 prefix 마스크와 합친다) + 인과 검정 (attention 온도 · RoPE 거리 척도를 바꾸면 reach 가 따라 움직이나) | 기전을 상관에서 인과로 올린다 | GPU 수십 분 |
| 3 | 거리 vs 경과 시간 분리 (같은 stride 에서 속도 폭이 큰 clip) | SC1 단서 1 | CPU 먼저 |
| 4 | EK100 라벨 없는 reach 곡선 (`e_flowtrack.py` 를 EK100 clip 에) | egocentric 실사용 도메인 | GPU 약 10 분 |
| 5 | 논문 초안 v3 (A′ 뼈대) + 리뷰 패널 | REVIEW_v1 은 옛 뼈대 기준 (weak reject 3/6) | CPU |

## 5. 문서 지도

| 알고 싶은 것 | 어디 |
|---|---|
| **논문 스토리 정본 (2026-10-08) · 근거 지도 · 결정 대기 · 하네스 · 금지 표현** | [`paper/PAPER_STORY_2026-10-08.md`](paper/PAPER_STORY_2026-10-08.md) |
| oral 방향 · 주제 안 · 위험 대응 | [`paper/ORAL_DIRECTION_2026-09-25.md`](paper/ORAL_DIRECTION_2026-09-25.md) |
| reach 법칙 (합성) · epoch · attention | [`Archive/SC1_REACH_NOT_HORIZON_2026-09-25.md`](Archive/SC1_REACH_NOT_HORIZON_2026-09-25.md) |
| 실영상 reach 와 flow 판독 보정 | [`Archive/SC2V2_FLOW_REACH_SSV2_2026-09-25.md`](Archive/SC2V2_FLOW_REACH_SSV2_2026-09-25.md) |
| EK100 predictor 치환 | [`Archive/SC3_EK100_PREDICTOR_SUBSTITUTION_2026-09-25.md`](Archive/SC3_EK100_PREDICTOR_SUBSTITUTION_2026-09-25.md) |
| **복사가 높은 이유 = metric 이 위치를 못 본다**: 위치만 다른 쌍 (RollOut_v2 ledge 낙하/부유 · wall 정지/통과, ViT-H) 에서 VoE 가 p 0 % · 17.6 %, 복사 7.9 · 2.3 % — 외형 위반은 복사로 풀리고 위치 위반은 둘 다 못 푼다 (p 는 틀린 쪽) | [`Archive/COPY_VS_POSITION_2026-10-07.md`](Archive/COPY_VS_POSITION_2026-10-07.md) |
| DINO-Foresight × IntPhysGen v11 (`surprise_c16t32`): 77.1 (ViT-H 73.4) 인데 **복사 76.4** — 상승은 프레임 DINOv2 타깃 공간 (비가림 shape/color 천장, 복사도 100), **가림 세 조건은 chance + vanish A/B 0/100** 로 ViT-H 와 같은 실패 | [`Archive/DINOF_V11_2026-10-07.md`](Archive/DINOF_V11_2026-10-07.md) |
| DINO-Foresight (frozen DINOv2 + Cityscapes 학습 frame predictor) × IntPhys 1 Garrido: predictor 84.4 (ViT-H 88.9) 인데 **같은 공간 복사 기준선이 88.9–92.2** — 긴 창 IntPhys 1 은 변화량 검출로 풀리고 predictor 몫은 음수; M=1 에서만 predictor 81 vs 복사 54. 공개값 87.8 (DINO-world 논문 baseline) 은 property 별 최고 규칙으로 86.1–86.7 재현, 차이는 O1 뿐; stretch 448×896 도 같음 | [`Archive/DINOF_INTPHYS1_2026-10-02.md`](Archive/DINOF_INTPHYS1_2026-10-02.md) |
| 과거만 본 문맥 encoder 에서 미래 위치 회귀 (사용자 제안): ledge 낙하 · wall 정지 · v11 late 가림 k=1–4 를 z 자가 5–10 px 로 맞추고 (≈ h), p 에 같은 자를 걸면 가림 없는 v2 에선 대부분 나오지만 (ledge 5.4 vs 3.3) v11 late 가림에선 12–30 px — 정보는 입력에 있고 predictor 가 가림 뒤로 안 옮긴다; '미래를 표현한다' 는 미래가 현재의 결정적 함수라 못 씀 | [`Archive/FUTURE_FROM_CONTEXT_2026-10-02.md`](Archive/FUTURE_FROM_CONTEXT_2026-10-02.md) |
| IntPhys 2 encoder linear probe (T×N 평균, 장면 split): 쌍 73.7 vs VoE 54.3 — immutability·permanence 는 encoder 벡터에 선형으로 남고 solidity 만 바닥; 지도 자라 VoE 와 같은 능력은 아님 | [`Archive/INTPHYS2_ENCODER_LINEAR_PROBE_2026-09-30.md`](Archive/INTPHYS2_ENCODER_LINEAR_PROBE_2026-09-30.md) |
| IntPhys 2 위반 시점 (metadata 에 없음 → 픽셀 차 실측; 후반 6.7–6.9 s, solidity 3.9 s, 32 % 는 고정 창 예측 구간 밖) | [`Archive/INTPHYS2_ONSET_2026-09-30.md`](Archive/INTPHYS2_ONSET_2026-09-30.md) |
| 학습된 predictor 넷 비교 (P2 · P3) | [`Archive/TRAINED_PREDICTORS_SIGNATURES_2026-09-25.md`](Archive/TRAINED_PREDICTORS_SIGNATURES_2026-09-25.md) |
| 되먹임 · 닫힘 (관측 vs 자기 예측) | `Archive/P3_CLOSURE_TEST_*` · `Archive/P3B_*` · `Archive/H5_*` |
| 역사 · 경계 · 조향 · 조회 반경 | `Archive/M6_*` · `M2_*` · `M2B_*` (1 차) · `M5_*` · `M4_*` · `M13_*` |
| VoE 점수가 무엇을 재나 | `Archive/H6_*` · `H9_*` |
| 적대 검증 원문 | `Archive/VERIFY_0925_*` (1 차 · 2 차) · `CROSS_REVIEW_2026-09-24.md` |
| 문헌 · 리뷰어 행태 | `Archive/LITERATURE_POSITIONING_*` · `REVIEW_CVPR_ORAL_*` |
| 초안 · 리뷰 | `paper/DRAFT_v2_*` (옛 뼈대) · `paper/REVIEW_v1_*` |
| 옛 README (정정 표 · 사슬 · 등급 · 읽기 규칙 · 현황 · 재현) | [`Archive/README_UNTIL_2026-09-25.md`](Archive/README_UNTIL_2026-09-25.md) |

## 6. 읽기 규칙 (새로 더한 것만; 1–10 은 옛 README)

11. **지평은 변위 (칸) 축으로 쓴다.** "몇 튜블릿까지 간다" 는 stride 에 따라 바뀐다. 슬롯 수는 조건 (stride · 문맥) 과 같이만 쓴다.
12. **실영상 위치 주장은 두 유사 정답 (flow · encoder 판독) 이 합의한 쌍에서만 한다.** encoder 천장이 슬롯에 따라 떨어지므로 (0.62 → 0.26) 전체 평균은 판독 질을 섞는다.
13. **학습된 predictor 끼리 비교할 때는 학습 조건을 적는다** (문맥 길이 · stride · 도메인). pv1 · v11_postft 는 문맥 4 가 학습 분포 밖이고, Ariel 은 SSv2 로 학습했다.

## 7. 원자료 · 실행 · 재현

- 원자료는 vll6 `/data2/local_datasets/world/world_analysis/cache/auto_research/` 에 있다 (`scripts/srun6.sh <cpus> <mem> <명령>` 로 읽는다). 09-25 오후 추가분:
  - `v3_p2s16_*` (SC1)
  - `flowtrack_{ssv2,v3}_*` (SC2v2)
  - `v3_m4_pv1`
  - `v3_p2c_*` · `v3_p3_{ariel,pv1,v11}e*` · `v3_p2s3_*` (묶음 G)
- 집계는 `exp_results/{p2,ssv2,m4}/` 에 있다. EK100 은 `z_research/anticipation/EK100/exp_results/action_anticipation_frozen/ek100_vith_{rel,pap}_grid8_val_*` 다.
- GPU 제출은 `scripts/run8.sbatch` + 묶음 스크립트 `scripts/chain_0925{f..j}.sh` 다. 명령은 각 문서의 `## 재현` 에 있다.
- 수치 경로: fp32 + autocast fp16, mask_index 0. Ariel 은 문맥이 block 0 에서 시작하는 팔만 유효하다.
