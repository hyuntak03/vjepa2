# 적대 검증 — TrainingEffects 세부 문서 다섯 (2026-09-25)

> 상위 [`../README.md`](../README.md). 문서마다 독립 검증자 한 명이 헤드라인 주장 6 ~ 12 개를 **자기 코드로 원시 배열에서 다시 계산**하고,
> 읽기 규칙 위반 (같은 clip 비교 · test 절반 · 고정 문턱 · 퍼진 attention · CI 단위 · 도메인 안 이득 · 원인 언어 · 단일 칸 고르기) 을 찾았다.
> 원문 (주장별 재계산 값 · 판정 · 고칠 점): `verify_raw/verify_<doc>.json`. 정정은 README §0 ~ §3 · §5 에 반영했다.
> 각 세부 문서의 스크립트를 고쳐 다시 돌렸다 (다섯 모두 fixed+rerun, 04:37 ~ 04:42). 문서 맨 위 '⚠️ 정정' 블록이 무엇이 바뀌었는지와 **미적용 항목** (GPU · 재학습이 필요한 것) 을 적는다.

| 문서 | 확인한 주장 | 그대로 | 수치 틀림 | 과장 | 틀림 |
|---|---:|---:|---:|---:|---:|
| [`scores/ANALYSIS.md`](../scores/ANALYSIS.md) | 12 | 7 | 0 | 4 | 1 |
| [`ip1_copy/IP1_COPY.md`](../ip1_copy/IP1_COPY.md) | 12 | 8 | 0 | 2 | 2 |
| [`v11_presence/PRESENCE.md`](../v11_presence/PRESENCE.md) | 11 | 6 | 1 | 3 | 1 |
| [`v3_motion/MOTION.md`](../v3_motion/MOTION.md) | 11 | 4 | 1 | 3 | 3 |
| [`v11_identity/IDENTITY.md`](../v11_identity/IDENTITY.md) | 11 | 5 | 0 | 3 | 3 |

## README 결론을 바꾼 지적 (정정 전 → 정정 후)

| # | 정정 전 문장 | 정정 후 | 근거 |
|---|---|---|---|
| 1 | 릴리즈 p 는 v11 에서 복사 기준선을 전체로 CI 로 이긴다 (+2.5) | 그 우위는 **색 위반에서만** (+9.3). 모양 −1.6 · vanish '물체→빈' −6.8, 색 빼면 −1.7 [−2.6, −0.7]. IntPhys 1 post-FT 도 색 빼면 −0.4 | verify_scores |
| 2 | 가려졌다 나오는 물체를 제자리에 두는 것은 도메인 안 둘 (v11 · Predictor_v1) | v11 post-FT 만 깨끗하다 (빈 장면 귀무 0.06 / 0.04). Predictor_v1 은 빈 장면에도 0.21 / 0.09 — 상당 부분 (flat 약 40 % · ramp 약 26 %) 이 '가림막 출구에서 물체가 나온다' prior. epoch 마다 귀무 ↑ · 순 배치 0.41 → 0.29 · '빈→물체' 100 → 73 | verify_presence |
| 3 | vanish '물체→빈' 순위가 배치 순위와 같다 | k≥2 에서만. Ariel (27 / 5 %) 은 복사 (26 / 5 %) 와 같다 — 표에 복사 행 추가 | verify_presence |
| 4 | v11 post-FT 는 문맥 속도를 무시하고 진실의 약 2 배로 옮긴다 | **틀림.** 궤적 속도를 따른다 (기울기 0.86 [0.53, 1.13]) + 고정 앞섬 +37 px. 궤적 특이성 지표 (own − null) 가 큰 공통 앞섬을 못 가른 것 | verify_v3 |
| 5 | 중력: IntPhys 2 는 더 떨어짐 (진실 쪽), Predictor_v1 은 덜 떨어짐 — 두 자 · 두 보정 일치 | '보정' (t0–1 빼기) 에 **부호 버그** (치우침을 빼지 않고 두 배로 더함). 스크립트를 고쳐 다시 돌리면 일치하는 것은 Predictor_v1 '덜 떨어짐' (Δ −17.6 / −14.5) 과 IntPhys 1 '조금 더 떨어짐' (Δ +2.6 / +7.0, 보정 가정 약함). IntPhys 2 는 빠진다. 검증자 재계산은 Predictor_v1 하나였다 — 보정에 쓰는 공통 칸 궤적 (7/14) 차이로 보인다 | verify_v3 · 재실행 |
| 6 | 가속 · 감속을 진실 쪽으로 넣는 predictor 가 없다 (전이는 안 생긴다) | '넣는다는 증거가 없다' 로 약화. ρ 는 덧셈꼴 앞섬을 못 지운다 — 서술적 | verify_v3 |
| 7 | v3 지속: Ariel 두 자 모두 ↑ | 지속 정의 ('있다' 비율 = README 정의 vs '모임' = 위치 포함) 에 따라 반대. 두 정의 모두 같은 방향은 Predictor_v1 ↑ · v11 ↓ 뿐 | verify_v3 |
| 8 | P32 t8 ~ t15 는 post-FT 넷의 학습 지평 밖 | v11 · Predictor_v1 만. IntPhys 1 (≤ 14) · IntPhys 2 (≤ 18) 는 일부 안. Predictor_v1 은 v3 와 같은 쐐기 · 등속 팔 (근접 도메인) | verify_v3 |
| 9 | 배치는 모양만 encoder 쪽으로 온다, 색은 어디서도 안 오른다 | 모양 · 색 둘 다 v11 · Predictor_v1 에서 p 가 target (h_fut) 쪽으로 온다 (고정 머리 이식 shape 40 → 87 / 70, color 47 → 85 / 75). '모양만' 은 z–p 부분공간 지표 하나의 결과. IntPhys 1 도 일부 움직인다 (cos 0.974) | verify_identity |
| 10 | 정체 헤드라인 수치 (probe-test 전체) | v11_split test block 으로 다시 인용 (학습 절반 섞임 제거). (궤적 × k) CI 에서 작은 효과는 유의하지 않음 | verify_identity |
| 11 | IntPhys 1 학습 곡선: 변화는 첫 checkpoint 에 끝나고 더 학습해서 생긴 것이 아니다 | skip2_w16 에서 v11 (e1 → e10 −5.6) · Predictor_v1 (−9.4) 은 epoch 마다 더 떨어진다 | verify_ip1 |
| 12 | Ariel 은 IntPhys 1 세 칸 모두 복사보다 낮다 | 표준 표적에서만. 인과 표적에서는 skip2_w32 한 칸. Ariel 은 ≤ 8 slot 예측으로 학습했는데 skip2_w32 (C=4) 는 14 slot | verify_ip1 |
| 13 | IntPhys 1 사건 전 튜블릿은 pos/imp 입력이 픽셀 단위로 같다 | 일부 쌍 (O1_05 · O1_27 · O1_01 · O3_22) 은 first_div_strict 전에 픽셀이 갈린다 — 튜블릿 사건 구간이 최대 1 튜블릿 어긋난다. 주 수치는 그 쌍을 빼도 유지 | verify_ip1 |
| 14 | 늦은 가림 채점 이득은 모든 predictor 에서 '다시 나온 뒤' 슬롯에 몰린다 | 슬롯 수를 맞추면 v11 post-FT 만 그렇다 (+5.9 / +12.3 / +20.3). IntPhys 2 는 재등장 슬롯이 가장 크다 | verify_scores |
| 15 | IntPhys 1 post-FT 학습 곡선: e5 최고 뒤 하락 | checkpoint 사이 ±2 pt 로 흔들린다 — 도메인 밖 이득과 같은 크기, 도메인 밖 셋끼리 순위 불가 | verify_scores |

## 문서별 '안전한 헤드라인' (검증자 작성)

### scores/ANALYSIS.md

v11_split_test (학습 안 한 block 절반, 10,752 matched pair) 에서 릴리즈 ViT-H predictor 는 예측 없는 복사 기준선 (문맥 마지막 튜블릿 복사) 보다 전체로 +2.5 [+1.6, +3.4] pt 높다. 다만 이 우위는 색 위반 (+9.3) 에서만 나오고, 모양 (−1.6 [−2.9, −0.2]) 과 물체→빈 사라짐 (−6.8) 에서는 복사보다 CI 로 낮으며 색을 빼면 −1.7 [−2.6, −0.7] 이다. predictor 만 다시 학습하면 v11 점수는 모두 오른다: v11 궤적 재생 post-FT +15.5, 가림 팔이 있는 Predictor_v1 +5.8, 도메인 밖 IntPhys1 +2.7 · IntPhys2 +3.7 · Ariel scratch +3.6 (하네스, block CI). 그러나 도메인 밖 셋은 48 칸 중 5–7 칸에서 CI 로 내려가고 (IntPhys1 은 움직이는 물체→빈 사라짐 칸이 대부분), IntPhys1 은 복사보다 새로 낮아진 칸이 3 개다. 도메인 밖 predictor 의 +3 pt 안팎 이득은 한 seed 에서 checkpoint 사이 ±2 pt 흔들림과 같은 크기라 서로 순위를 매길 수 없다. v11_e10 의 이득은 궤적 재생이라 일반화가 아니다. 드러남 뒤 슬롯에 이득이 몰린다는 결론은 슬롯 수를 맞추면 v11_e10 에서만 남는다.

주요 지적:

- The bold headline 'release 는 복사 기준선을 전체로 CI 로 이긴다' (+2.5) comes entirely from color violations (+9.3 [+7.7,+11.0]). Release is CI-below copy on shape (−1.6 [−2.9,−0.2]), on vanish obj→empty (−6.8 [−8.5,−5.2]) and on all non-color pairs together (−1.7 [−2.6,−0.7]); it is null at early and mid timing. The same composition makes ip1_e40 look like it beats copy (+5.5 overall, but −0.4 [−1.4,+0.5] without color). A reader quoting the headline would conclude the opposite of what holds for shape and object permanence.
- The '새로 copy 를 밑도는 칸' counts look only inside the 20 non-beaten cells, so drops below copy in the 17 ceiling cells are never counted. pv1_e15 also falls CI-below copy in late·flat·vanish empty→obj (79.5 vs 100), and ip1_e40 in visible·ramp·vanish obj→empty (87.5 vs 100) and mid·flat·vanish obj→empty (84.8 vs 100). The real counts are pv1 2 and ip1 3, not 1 each, so the cost of out-of-domain training is understated.
- The reveal-stage comparison ('Δ 가 가장 큰 단계: 모두 after') is not like-for-like: it scores the mean S over 5–6 slots for 'after' but 1 slot for 'reveal' and 1–2 for 'hidden'. Averaging more slots raises accuracy by itself (v11 after +28.3 is larger than any single-slot Δ, max +22.8). Averaging per-slot hits instead gives ip1 after +0.9 (was +4.5), ip2 after +2.3 < reveal +3.6 (the ordering flips), and pv1 after +4.3 ≈ reveal +3.8. Only v11_e10's after-dominance survives.
- The ip1 learning curve ('e5 최고 뒤 하락') zig-zags: e5 80.25 → e10 77.86 → e20 79.99 → e40 78.75, with ±2.1–2.4 pt swings that are CI-significant between neighbouring checkpoints. That swing is as large as ip1's whole gain over release (+2.7 harness / +3.1 extraction). With a single seed, the block-bootstrap CIs on the out-of-domain Δs (+2.7 to +3.7) do not include training noise, so ranking ip1/ip2/ariel against each other is not supported.

### ip1_copy/IP1_COPY.md

IntPhys 1 dev (180쌍, 90 scene) 에서 예측 없이 문맥 마지막 튜블릿 latent 를 복사하기만 해도 Garrido Filtered macro 가 skip2_w32 85.0 (표준 표적) / 77.8 (인과 표적) 이다. 릴리즈 predictor 의 88.89 는 같은 칸 copy 보다 +3.9 [−0.6, +8.3] 로 CI 상 구분되지 않는다. copy 를 CI 로 넘는 곳은 skip2_w16 (+9.4, 인과 +15.0) 과 skip2_w32 인과 표적 (+6.7 [+0.6, +13.3]) 뿐이다. 모든 방법이 appear 방향 ~100%, disappear 방향 40–73% 로 방향 편향이 크다. 학습한 predictor 중 skip2_w32 에서 copy 와 릴리즈를 둘 다 CI 로 넘는 것은 IntPhys 2019 train 으로 학습한 ip1_e40 (+8.9 / +5.0) 하나이고, 이는 학습 도메인 안의 결과라 일반화 증거가 아니다. v11 · Predictor_v1 · IntPhys2 post-FT 는 skip2_w32 에서 릴리즈보다 3–11pt 낮다. v11 과 Predictor_v1 은 skip2_w16 에서 epoch 이 늘수록 더 떨어진다. Ariel scratch 는 표준 표적 세 칸 모두 copy 보다 낮지만, 인과 표적에서는 skip2_w32 한 칸만 그렇다. 학습은 한 번씩 (seed 0) 뿐이고 칸 CI 는 scene 15–30 개 위의 값이라, 칸 단위 차이는 다중비교를 감안해 읽어야 한다.

주요 지적:

- The learning-curve conclusion ('변화는 대부분 첫 체크포인트 안에서 … 더 오래 학습해서 생긴 변화가 아니다') is chosen from one cell and contradicted by the doc's own numbers. On skip2_w16, v11 e1→e10 is -5.6 [-10.0, -1.7] and pv1 e1→e15 is -9.4 [-14.4, -5.0] (0 vs 17 discordant pairs), while release→e1 is not significant. On skip2_w32, ip2 goes from -0.6 (e10) to -3.3 (e80), which the same paragraph prints. On IntPhys1, v11 and pv1 degrade progressively with more training.
- The Ariel headline 'ariel_ep43 는 세 칸 모두 copy 보다 CI 로 낮다 (예측이 복사보다 못하다)' holds only for the standard target. The doc says the causal target is the better reading of the prediction's share, and under it only skip2_w32 is below copy (-7.8 [-15.0, -0.6]); skip2_w16 is +2.8 [-5.6, +11.1] and skip5_w16 is -4.4 [-11.1, +2.2]. There is also an unmentioned regime mismatch: Ariel was trained to predict at most 8 slots, at 12 fps, while skip2_w32 windows ask for up to 14 future tubelets. That is a fourth reason, beyond the four listed, not to read 'worse than copy' as a property of the attention rule.
- The pre-event tubelet reasoning states two things that are false. First, 'pos/imp 입력 프레임이 같다 / 문맥 프레임이 픽셀 단위로 같다' fails for some windows. Second, the causal-pre non-ties are blamed on fp16 noise. In fact they come from real pixel divergence before first_div_strict: 8 of 60 moving/occluded pairs (O1_05, O1_27, O1_01, O3_22) differ from first_div_sensitive, with max per-pixel diff up to 174 and ΔL1 up to 0.03. In 232 of 3352 e0 tubelets the context includes such frames, and 87% of static/occluded causal-pre tubelets are non-ties in 15 of 15 scenes. The event binning (first_div_strict) is therefore off by up to one tubelet. The main pre/e0 numbers survive excluding these pairs (ip1 pre 67.7, release e0 80.4 vs copy 62.6), but the stated verification is wrong and the static/occluded tubelet bins in the json are unusable.
- The doc breaks the CLAUDE.md §1-7 rule to always report accuracy by A/B direction: no pair-level appear/disappear split is given. Every method is near 100% on appear and 40–73% on disappear. At skip2_w32, copy scores O1 appear 100 / disappear 53.3 and release 100 / 66.7; the O1 disappear subset is n=15, so ip1_e40's O1-disappear gain over release (73.3 vs 66.7) is just 1 pair. Without this split, readers cannot tell how much of copy's 85 and p's 88.89 is a direction bias rather than violation detection.

### v11_presence/PRESENCE.md

v11 held-out block 절반 (문맥 끝 가림 k≥2, 궤적은 좌→우·우→좌 2 개뿐) 에서, 가려졌다 다시 나오는 물체 쪽으로 p 의 attention 을 확실히 모으는 것은 v11 궤적을 재생 학습한 v11_e10 하나다 (진실 3×3 질량 0.66/0.57; 빈 장면 귀무 0.06/0.04; x 오차 −13 px). pv1_e15 도 질량은 0.50/0.35 로 오르지만 물체가 없던 빈 가림막 장면에서도 같은 자리에 0.21/0.09 를 둔다. epoch 가 갈수록 이 귀무가 커지고 vanish B (빈→물체) 는 100 → 73 % 로 떨어진다. 그래서 pv1 의 상당 부분은 '문맥과 무관하게 가림막 출구에서 물체가 나온다' 는 prior 이지 영속이 아니다. 둘 다 학습 도메인 안이다. 도메인 밖 셋 (IntPhys1·IntPhys2·Ariel) 은 질량 0.04–0.13 으로 release 수준에 머문다. 고정 문턱 '있다' 비율의 변화는 자를 바꾸면 부호가 뒤집히거나 (ip2_e80) 문턱 없는 AUROC·배치와 어긋난다 (ip1_e40·Ariel flat). vanish 물체→빈 채점에서 Ariel (27/5 %) 은 복사 기준선 (26/5 %) 과 같다. 보이는 물체 (visible) 는 모든 predictor 가 AUROC ≥ 0.97 이지만 복사 기준선도 1.00 이라 예측 능력의 증거가 아니다. 뒤처짐 (release −72 px → v11_e10·pv1_e15 −16 px, visible flat t4) 은 두 궤적 안에서만 일관되며, 궤적·속도를 넘는 일반화는 이 데이터로 말할 수 없다.

주요 지적:

- No position null for 'placement' (3x3 GT attention mass). pv1_e15 puts 0.21 (flat) / 0.09 (ramp) of the mass at the would-be object location in empty occluder scenes that never contained an object, against 0.50 / 0.35 on object clips and 0.035 uniform. At late flat t4, 42% of empty clips pass 'present AND mass3>0.035'. Roughly 40% of pv1's flat 'permanence placement' is therefore an occluder-exit prior, not object permanence. Headline 1 ('only the two in-domain predictors put it back in place') is overclaimed for pv1; v11_e10's null is clean (0.057).
- The pv1_e15 learning curve is misread. Section 4 says the falling present-minus-floor with flat placement is 'ruler scale / empty-floor shift'. Two threshold-free measures contradict this: the empty-scene placement null rises 0.12 -> 0.21 over e1 -> e15, and the vanish B score (empty->object) falls 100 -> 72.6. pv1 increasingly hallucinates an emerging object regardless of context. That is a behavioural change the doc half-acknowledges in headline 4 but explains away in section 4.
- Table 7 (vanish A/B cross-check) omits the copy baseline, which the README requires. Copy A at late flat / ramp k>=2 is 26.2% / 4.8%. Ariel's A (27.4 / 4.8) is copy level, and release (0/0) and several late-static entries (pv1 39.3, Ariel 42.9, release 58.3 vs copy 64.3) are below copy. The 'top two = concentrated placement' ranking holds only at k>=2: at k=1 flat, pv1 A = 64.3, below copy 78.6, ip2 75.0 and Ariel 67.9.
- The 'robust' verdicts in Table A and headline 2 ignore the threshold-free readouts the doc itself says are needed. ip1_e40 flat and ariel_ep43 flat are 'robust down' on fixed-threshold p@h/p@p only. Their p@h AUROC Δ is about 0 (-0.002; -0.041 with CI including 0) and their placement Δ is positive (+0.02 / +0.07). This violates the stated reading rule that fixed-threshold present mixes in calibration shift.

### v3_motion/MOTION.md

RollOut_v3 (가능 2,744 clip · 궤적 112, 모든 predictor 같은 clip) 에서 릴리즈 대비 정의와 자에 관계없이 같은 방향인 변화는 두 가지뿐이다. Predictor_v1 post-FT 는 물체를 더 오래 유지한다: P32 '모임' T50 10→16 (중도절단), t8–15 p@h 0.46→0.84 · p@p 0.36→0.69, '있다' 만으로 재도 두 자 모두 ↑. v11 post-FT 는 더 일찍 잃는다: T50 10→5, 두 자 · 두 정의 모두 ↓. pv1 은 같은 10° 쐐기와 등속 팔로 학습했으니 이것은 동역학상 근접 도메인의 이득이지 일반화가 아니다. 학습한 다섯 predictor 는 모두 자유 x 법칙 전부에서 릴리즈보다 앞서 읽힌다 (공통 칸 Δ > 0). 이 앞섬은 곱셈꼴 이득이 아니라 궤적을 따르는 기울기 약 1 에 고정 오프셋이 더해진 모양이다 (v11_e10 flat_v: 기울기 0.86, +37 px). 그래서 'v11 이 문맥 속도를 따르지 않는다' 는 성립하지 않고, ρ 로 가속 · 감속 반영을 가르는 판정은 확정적이지 않다. 가속과 감속을 두 자 모두에서 GT 로 넣는 predictor 는 확인되지 않았다. arc · ledge 의 '보정' 값은 부호 규칙 때문에 치우침을 빼지 않고 두 배로 더하므로 인용할 수 없다. 올바르게 보정하면 두 자 × 두 방식에서 같은 방향인 것은 pv1_e15 (덜 떨어짐) 하나다. 위치 판독의 주 자 p@h 는 p 토큰에서 검증되지 않았다 (flat 장면에서 release p 를 19–40 px 위로 읽는다). 따라서 위 결론은 행동 수준의 비교로만 읽는다.

주요 지적:

- Sign bug in the '보정' (t0–1 subtraction) rows. axis_dir gives s_y = −1 (upward) at t0–1 for all ledge clips and half of the arc clips, then +1 in the falling tubelets. Subtracting e(t0–1) therefore adds twice the ruler bias instead of removing it. My recomputation reproduces the doc exactly (arc release −9.5/−22.7, ledge −31.2/−19.6). The sign-consistent correction gives arc −1.2/−7.7 and ledge −10.7/−9.7. After it, ip2_e80's 'robustly falls more' becomes rulers disagreeing (p@p Δ −7.3 [−9.1, −5.5]), and several ledge verdicts flip (pv1 p@h +26.2 → −0.6; ip1 p@p −19.0 → +2.8; ip2 p@p +12.4 → −10.1). Of the headline gravity claim, only pv1_e15 (falls less) survives.
- The trajectory-specificity metric (own − null displacement) cannot detect tracking when the common offset is larger than the spread of trajectory displacements. The claim that v11_e10 does not follow context speed is contradicted by a direct regression: slope 0.86 [0.53, 1.13] for p@h and 1.04 [0.13, 1.74] for p@p, both significant. The same flaw weakens the qualifier 'trajectory-specific: ip2_e80' attached to the flat_d GT verdicts.
- Every trained predictor shows an additive lead (slope ≈ 1 plus an offset of +6 to +50 px), not a multiplicative gain. So ρ = β_law/β_flat_v does not 'cancel the common shift' as claimed, and it biases ρ toward 1 (GT). This undermines the extension-versus-GT classification in the transition headline, especially the 'GT only' verdicts for flat_d (v11, ip2, ariel).
- Persistence is defined differently from the README. README §1 defines 지속 as the '있다' rate curve and its T50; MOTION uses '모임' ('있다' plus GT 3×3 mass > uniform), which mixes in position accuracy. Under the README definition, release p@h T50 at P32 is 15 (not 10). ariel_ep43 becomes a robust decrease under both rulers (Δ −0.21 / −0.07) instead of 'robust ↑', and ip1_e40 becomes a robust decrease instead of 'rulers disagree'. The doc itself reports ariel p@p T50 10 → 6, yet the headline counts ariel as increasing under both rulers. v11_presence/PRESENCE.md uses 지속 to mean presence ('nobody loses the object; what differs is placement'), so the two docs use the same capability name for different measurements.
- Factual error on training horizons. IntPhys2 post-FT trained with future horizons up to 18 tubelets (48 frames, C 12–42) and IntPhys1 post-FT up to 14 tubelets (skip2_w32, C 4–20). So 'P32 t8–15 is outside the horizon of all four post-FT models' is false for ip1 and ip2.
- The main position ruler p@h is not validated on p tokens (decoder-first rule). On flat_v it reads release p 19–40 px above GT, while the h-ruler on h is off by ≤ 3 px. RollOutV3 reports a 37.5 px error for p@h on late p tokens. The '모임' gate lets these readings through. Calling p@h 'predictor-independent' is not supported.
- Domain overclaim (rule f). 'v3 is not any predictor's training domain' does not hold for pv1_e15. Predictor_v1 has a constant-velocity arm with continuous speeds and a 10° ramp arm with continuous up-ramp and down-ramp acceleration. RollOut_v3 uses the same 10° wedge (rollout3.json ramp_theta 10.0) and the same law family; only the stride differs. pv1's persistence and reach gains are near-domain gains, not evidence of generalization.

### v11_identity/IDENTITY.md

v11_split test block (v11_postft 가 학습에 쓰지 않은 절반) 에서, 학습한 predictor 5 개 모두 p 의 조건 안 정체 decodability 는 천장 근처로 남는다 (predictor 마다 다시 맞춘 late 머리 → late clip: shape 96.1–97.8 %, color 94.0–97.5 %; release 96.8 / 97.1). v11 과 같은 생성기로 학습한 두 predictor (v11_postft 는 궤적 재생, Predictor_v1_postft) 는 p 를 절대 좌표에서 h_fut 쪽으로 크게 옮긴다. cos(p_fut, h_fut) late 는 0.931 → 0.996 / 0.991 이고, 고정 h_fut 머리를 p 에 그대로 걸면 visible 정확도가 shape 40.4 → 86.9 / 69.6 %, color 47.3 → 85.3 / 75.3 % 로 오른다 — 두 속성 모두다. shape 의 클래스 평균 부분공간도 encoder 쪽으로 간다 (z–p 0.794 → 0.881 / 0.896, encoder 끼리 0.943). color 는 z 기준 공유 몫이 모든 predictor 에서 약간 내려가지만 h_fut 기준으로는 v11 · pv1 에서 약간 오르고, 추정 자체의 노이즈가 크다. 이 이동은 학습 도메인 안의 결과다. IntPhys1 post-FT 는 일부만 움직이고 (cos 0.974), IntPhys2 와 ariel 은 거의 움직이지 않는다. visible 짝으로 맞춘 Procrustes 를 late 에 걸면 어느 predictor 에서도 45–66 % 로, late 짝 수준 (91–97 %) 에 못 미친다. ip2 만 부분 개선 (+6~11 pt) 이 있다. 약 5 pt 이하의 정확도 차와 0.01 이하의 공유 몫 차는 (궤적 × k) 단위 CI 에서 대부분 유의하지 않으므로 주장하지 않는다.

주요 지적:

- Rule (b): the headline (i)–(iii) and the subspace numbers quote the probe-test-full / 9,408-clip scope. Half of those clips (index_train blocks) are v11_postft training data, and README says v11 is read only on v11_split_test. The v11test versions exist (tables 3 and 4) and give the same conclusions (h raw vis v11 86.9 vs 86.3; z–p 0.881 vs 0.884), but the headline should cite them. Probe heads are also fit on the mixed probe-train half.
- Internal contradiction in headline (ii): 'raw 머리 이식도 같은 쪽이다' is cited to support 'shape only', but raw h_fut-head transfer rises just as much for color (v11 +38.0 vs shape +42.4; pv1 +28.4 vs +26.2). The shape/color difference exists only in the resid z–p class-mean metric.
- 'color 는 어디서도 안 오른다' / 'shape 만' is metric-selective. On h_fut–p (the training target), color share rises on v11te for v11 (+0.036) and pv1 (+0.024), both bold in the doc's own table 4. p's color split-half ceiling is 0.61–0.81, so color subspace numbers are noisy.
- Rule (e): the CI unit is block (1–2 clips, effectively clip-level), not (trajectory, k) (16 trajectories; 8 visible / 32 late clusters in test∩v11te). The doc's assurance that paired comparisons are fine is refuted: under the correct unit several bold or significant small effects become n.s. (v11 shape Procr(vis)→late +3.6 [−0.4, +8.2]; ariel h raw vis +2.7 [−0.2, +5.0]; ip2 h raw vis +4.1 [−1.7, +9.0]). The large effects survive (v11 +46.6 [+40.4, +52.7]; pv1 +29.2 [+22.2, +35.9]).
- The subspace bootstrap is biased: point estimates lie outside their own 95 % CIs in several cells (e.g. color h_fut–p late v11 0.774 [0.778, 0.810]), so the bold marks on small subspace deltas (ariel ±0.002–0.005) are unreliable.
- Rule (f): the headline labels the shift as a 'synthetic-physics post-FT' effect. It is an in-domain same-generator effect (v11 = trajectory replay; pv1 = same generator, 5/7 shapes). ip1, also synthetic physics, moves part of the way (cos 0.931 → 0.975; h raw shape +5.9 / +14.6 on v11te, surviving the (trajectory, k) CI), so 'IntPhys 계열에서는 안 보인다' is too strong.

## 재현

```bash
# 검증 원문 (에이전트 출력) 은 verify_raw/verify_<doc>.json 에 그대로 보관했다.
# README 의 정정 문장은 그 json 의 checked_claims[].recomputed 값을 옮겼다.
```
