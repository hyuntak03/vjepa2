# 10 — 기각 · 철회 · 정정 · 폐기 통합 표 ("다시 시도하지 말 것 · 다시 쓰지 말 것")

> **결론** — 레포 곳곳 (z_world_model_analysis 08-13 → z_research → auto_research 10-08) 에 흩어진 기각 · 철회 · 정정 · 폐기를 중복을 합쳐 여덟 주제 373 행 (A 46 · B 33 · C 55 · D 45 · E 65 · F 44 · G 33 · H 52) 으로 묶고, 금지 표현 36 행 (§3) 을 따로 두었다. 가장 자주 되풀이된 실패는 셋이다: (i) **학습된 자의 기본값을 predictor 의 행동으로 읽은 것** ("마지막 자리에 둔다", "슬롯 4 이후 멈춘다", "t10 절벽"), (ii) **문턱 · 귀무 · CI 단위를 잘못 잡아 생긴 인공물** (모드별 문턱, clip 단위 CI, 진실 위치를 쓴 충분성 대조), (iii) **복사 기준선 없이 점수를 능력으로 읽은 것** ("IntPhys 1 88.89 처럼 위반을 가려낸다", C30 "ledge 쌍 74–100 % 가 떠 가는 미래에 가깝다 → 낙하를 안 만든다" 를 능력 주장으로 — 복사가 59–99 % 로 재현. 정본 §7 은 "낙하하는 미래를 만들지 않는다" 를 **다중 미래 설계에서만** 대체 문장으로 허용한다).
> **근거** — CLAUDE.md §6 기각표 (24 행) · `auto_research/README.md` §2 철회표 (6 행) + `auto_research/Archive/README_UNTIL_2026-09-25.md` 정정 1 · 2 표 (21 행) · `z_research/RollOutV3` 감사 철회 사유 32 행 · `auto_research/Archive/VERIFY_0925_R2_2026-09-25.md` 모순 X1–X21 · `z_research/OVERVIEW_2026-09-22.md` §7-2 철회 9 건 · 각 원 문서 머리의 ⚠️ 정정 블록.
> **미결** — "encoder 충분 · predictor 병목" 은 09-25 에 철회됐는데 10-08 정본 P3 는 다른 근거 (z 속도 R² · FUTURE_FROM_CONTEXT · IntPhys 2 linear probe) 로 "정보" 로 한정해 다시 세웠다 — 이 재사용이 리뷰에서 서는지는 아직 검증되지 않았다 (E 절 E21).
> 이 장이 받치는 논문 문장: 모든 문단 (P2 · P3 · 목표) 의 **쓰면 안 되는 문장** 목록  ·  원재료: G01–G13 추출문의 "단서 · 정정 · 철회" 칸과 묶음 요약 (c), CLAUDE.md §6, `auto_research/README.md` §2, `auto_research/paper/PAPER_STORY_2026-10-08.md` §7 → 원 문서는 각 행 `출처` 와 끝 `## 출처`.

---

## 1. 읽는 법

### 1-1. 종류 (`종류` 열)

| 종류 | 뜻 | 예 |
|---|---|---|
| **기각** | 가설을 실험으로 반증했다 | "context 길이가 문제" — C=4/8/12/16 전부 50.9–54.3 % |
| **철회** | 우리가 쓴 문장을 거뒀다 (수치는 맞을 수 있다) | "p 가 물체를 마지막 자리에 둔다" — 자의 기본값 |
| **정정** | 수치 · 해석을 고쳤다. 하위: **좁힘** (범위를 줄임) · **강등** (확정 → 시사 · 가설) · **판정 보류** | "reach 는 고정 거리" → release 는 약 2 칸 + 1.6 걸음 |
| **폐기** | 도구 · 데이터 · 프로토콜 · 그림을 버렸다 | presence 자, Luce 7 지선다, Max 지표 |
| **금지** | 표현 규칙 (사용자 결정 · 정본 §7) | "모델이 중력을 안다", memory 프레임 |
| **번복** | 사용자 결정으로 범위 · 틀 자체를 바꿨다 | ledge · wall "정의 밖" → 10-08 포함 |
| **사고 · 함정** | 인프라 · 데이터 손실 · 조용한 오동작 | sbatch 자기제출 9,924 job, 체크포인트 삭제 |

### 1-2. 표기

- **`†`** = 원 문서 표현 — 현재 금지 (정본 §7 · 09-30 memory/state 프레임 금지 · 내부 귀속). 옛 문장을 그대로 옮길 때만 쓴다.
- `날짜` 는 철회 · 정정이 일어난 날 (원 문장의 날짜가 아님). 원 문서가 날짜를 안 밝히면 `—`.
- `출처` 경로는 레포 루트 기준. 같은 철회가 여러 문서에 있으면 `;` 로 나열한다 (정본 우선).
- 반복 약어: **README_UNTIL** = `auto_research/Archive/README_UNTIL_2026-09-25.md` · **r1** = `z_research/RollOutV3/Archive/_superseded/NEW_ARCHIVE_README_2026-09-25_r1.md` · **NAR0924** = `z_research/RollOutV3/Archive/_superseded/NEW_ARCHIVE_README_2026-09-24.md` · **V3R0925** = `z_research/RollOutV3/_superseded/README_2026-09-25.md` · **TRAINED** = `auto_research/Archive/TRAINED_PREDICTORS_SIGNATURES_2026-09-25.md` · **POSREAD** = `z_research/RollOutV2/figures/v5/summary/POSITION_READOUT_2026-09-12.md` · **DISTLIM** = `z_research/RollOutV2/Archive/PREDICTOR_DISTANCE_LIMIT_2026-09-19.md` · **정본** = `auto_research/paper/PAPER_STORY_2026-10-08.md`.
- `대신 쓸 문장` 이 `—` 이면 원 문서가 대체 문장을 주지 않았다 (그 주장 자체를 하지 않는다).
- α 분해 · 헤지 · 보정 계수 값은 이 장에서도 옮기지 않는다 (§3 마지막 행).

### 1-3. 이 장을 쓰는 규칙

1. 새 실험 · 새 문장을 제안하기 전에 이 장에서 **키워드로 먼저 찾는다** (예: "반경", "멈춘다", "충분", "절벽", "복사").
2. 철회된 문장의 **수치는 대개 유효하다** — 거둔 것은 해석이다. 수치를 다시 쓸 때는 `대신 쓸 문장` 열의 틀로 쓴다.
3. 모순 (두 문서가 다른 답) 은 지우지 않고 그대로 두었다 — 어느 쪽인지 모르는 것은 `11_OPEN_ISSUES_AND_NEXT.md` 로.

---

## 2. 주제별 표

### A. 채점 · 프로토콜 · 벤치마크

| # | 옛 문장 / 도구 | 종류 | 왜 (반증 근거 수치) | 날짜 | 대신 쓸 문장 | 출처 |
|---|---|---|---|---|---|---|
| A1 | MaxSurprise 지표 (Avg + Max 병기) | 폐기 | Garrido Table S4/S5: 쌍 비교는 Avg 오류 0.28 / 0.0 / 0.09 % < Max 4.4 / 4.4 / 12.87 %. Max 는 단일 영상 (1−AUROC) 용. 코드 분기 제거 | 09-21 | 쌍 비교 **AvgSurprise 하나** | `z_research/Benchmarks/PROTOCOLS.md` §1-b; `z_research/Benchmarks/README.md` §6; `analysis/intphys2/PROTOCOL_CHECK_2026-09-14.md` 머리 |
| A2 | IntPhys 1 보고값 = Table S3 한 칸 `skip2_w16` 83.33 | 정정 | Table S3 는 그들 모델 (fpc 16) 의 탐색 결과. V-JEPA 2 (fpc 64 @256) 는 A.8 격자를 탐색해야 → `skip2_w32` 88.89 (160/180) | 09-21 | "88.89 (`skip2_w32`, Filtered, A.8 격자 최고 — descriptive)", 공식 칸 83.33 병기 | `PROTOCOLS.md` §2; `Benchmarks/README.md` §6 (`_superseded_20260921_singlecell/`); CLAUDE.md §2-2 |
| A3 | 창 16 · 32 를 한 실행 (혼합 격자) | 폐기 | 공식은 창마다 모델을 그 프레임 수로 짓는다 | 09-21 | 창 (C+M) 마다 실행 분리 | `z_research/Benchmarks/exp_results/_invalid_20260921_mixedgrid/README.md` |
| A4 | Benchmarks 의 09-21 이전 수치 전부 | 폐기 | 8 가지 사유 (단일 칸 · 혼합 격자 · InfLevel 자작 쌍 · `MB21` 미적용 · VideoMAEv2 창 16 고정 · min 축약 · Max 지표 등). `exp_results/_invalid_*` 에 보존만 | 09-21 | 인용 금지 | `Benchmarks/README.md` §6; `z_research/OVERVIEW_2026-09-22.md` §7-2 |
| A5 | InfLevel 자작 쌍 · cross 2×2 · priming 미절단 | 폐기 | 공식 csv 한 줄 = 한 쌍, `matches=[[0,1]]`, priming 절단. 이전 인덱스 · 수치 전부 폐기 | 09-21 | 공식 쌍 | `PROTOCOLS.md` §5-3; `configs/protocols/datasets.md` (InfLevel 3 종 재작성) |
| A6 | InfLevel gravity · solidity 로 모델 비교 | 금지 (사용 제한) | priming 절단 때문에 원리적으로 못 푼다 (Garrido App. E): ViT-H solidity 50.00 · gravity 51.95 | 09-21 | continuity 만 쓴다 (ViT-H 72.85) | `Benchmarks/README.md` ①; `OVERVIEW_2026-09-22.md` §5-8 |
| A7 | GRASP · InfLevel 에 min 축약 · VideoMAEv2 창 16 고정 | 폐기 · 정정 | property 별 최고 C 여야 한다; VideoMAEv2 는 창 32 도 돈다 | 09-21 | — | `Benchmarks/README.md` §6; `configs/protocols/models.md` |
| A8 | IntPhys1 dev `0.6750` (자체 C32/M8 재구성, 360 pair) | 정정 (프로토콜 교체) | Garrido sliding · ViT-H · 180 matched pair 로 바뀌어 88.89. 08-13 문서 머리에 대체 표시 없음 | 08-16 | 프로토콜을 밝힌 88.89 만 (CLAUDE.md §1-3) | `z_world_model_analysis/intphys1_dev_2026-08-13.md`; `z_world_model_analysis/RESULTS_2026-08-16.md` |
| A9 | "future-matched (미래일치) 쌍이 공정하다" (occlusion_v2 presence 0.9427 / shape 0.6192 를 권고값으로) | 기각 | 미래일치 쌍은 p 부터 달라 누수를 흡수한다. 문맥일치 2 쌍만 통제 비교 (presence 50.00 / shape 54.30) | 08-16 | `scoring.pairing: matched` | `z_world_model_analysis/occlusion_v2_2026-08-13_PREVIOUS-DATA.md`; `z_world_model_analysis/FRAME_ABLATION_2026-08-19.md` §2; CLAUDE.md §1-2 |
| A10 | "C 를 줄이면 신호가 커진다" | 정정 | v2↔v3 교란 비교 (obj_px 24→48, divergence 33–36→19–29, t_occ 10.6→7.2 가 함께 바뀜). 동일 데이터에서는 반대 (C 길수록 신호 gap +0.0237→+0.0318, 누수 +0.0334→+0.0614) | 08-19 | — | `FRAME_ABLATION_2026-08-19.md` §5 |
| A11 | 채점 프로토콜 (sliding) 이 원인 | 기각 | IntPhys 는 고정 단창 하나로도 88.33 | 08-20 | — | CLAUDE.md §6; `z_world_model_analysis/MECHANISM_2026-08-20.md` §6 |
| A12 | IntPhys1 단창 87.78 / 88.3 / 88.33 을 sliding 88.89 와 같은 수로 | 정정 (주의) | 같은 창의 반올림 차로 보이나 문서가 안 밝힘 | 08-20 | sliding 과 구분해 쓴다 | `z_world_model_analysis/WHY_INTPHYS_WORKS_2026-08-20.md`; `VERIFICATION_2026-08-20.md`; `MECHANISM_2026-08-20.md` §6 |
| A13 | Luce 7 지선다 확률 (`fig_make_prob`, `fig_make_prob_grid_*`) | 기각 · 폐기 | `s=(1-p)/p` 가 최저 상대 하나에 지배: 등가속 · 비가림 cube 짝평균 83.3 → 7 지선다 8.7 (cylinder 에게만 6 %). k 별 판은 비대각 n=4. IIA 검정 불가 | 08-31 | 2 지선다 원자료 | CLAUDE.md §6; `z_research/IntPhysGenV11/figures/surprise/README.md` (`_superseded/`); `z_research/IntPhysGenV11/Archive/surprising_score/VOTE_GRID_2026-08-31.md` |
| A14 | 08-31 beat 4 "cone 문맥에서 cube 77 %, cone 2 %" | 철회 | Luce 산출물 (A13) | 08-31 이후 | — | `z_research/IntPhysGenV11/Archive/PAPER_STORY_2026-08-31.md`; `IntPhysGenV11/figures/surprise/README.md` |
| A15 | block 밖 7-way retrieval | 기각 (다시 시도 금지) | 같은 block 다른 모양 0.5438 / 다른 block 같은 모양 0.7602 → 신호/잡음 0.715. `h` 대조군조차 40.3 % (chance 14.3) | — | — | CLAUDE.md §6; `z_research/IntPhysGenV11/README.md`; `z_research/scripts/analysis/retrieval_confusion.py` 머리 |
| A16 | `token_subset: object` (물체 토큰만 채점) | 기각 | 25–39 배 좁혀도 v11 외형 위반에서 **더 나빠짐**. 위치 쌍에서는 안 재 봤다 | 08-19 | 기본 `all` | CLAUDE.md §6; `z_world_model_analysis/TOKEN_SUBSET_2026-08-19.md`; `auto_research/Archive/COPY_VS_POSITION_2026-10-07.md` |
| A17 | E3 "채점기가 보는 것 = 토큰 평균" | 정정 | 채점기는 토큰별 \|p−h\| 의 평균 — 등치 불가; "토큰 국소 채점" 설계 요구 삭제 | 09-16 | 풀링 폭 단조 → 가림막 토큰 혼입 (구성 효과) | `z_research/context_encoder_analysis/Archive/STATE_EVOLUTION_HYPOTHESES_2026-09-15.md` blockquote; `EXPERIMENTS_2026-09-16.md` 실험 3 |
| A18 | "IntPhys1 88.89 % 처럼 surprise 로 위반을 가려낸다" | 철회 | 마지막 문맥 latent 복사만으로 85.00 %, p − 복사 +3.9 [−0.6, +8.5] n.s. (`skip2_w32`); `skip2_w16` 은 +12.2 [+6.8, +17.8]. ⚠️ 같은 Δ 의 CI 끝이 원 문서마다 다르다 — `RESULTS_2026-09-25` 부록 C [−0.6, +8.5] · [+6.8, +17.8] / `H6_…` · `REVIEW_CVPR_ORAL` [−0.6, +8.4] · [+6.7, +17.8] / 정본 §2-2 · `IP1_COPY.md` [−0.6, +8.9] (bootstrap 차로 보임, 문서에 설명 없음) | 09-25 | "IntPhys 1 은 leakage 가 크다 — 예측 없이 85 %" (복사 기준선으로 정의) | NAR0924 (09-25 정정); `z_research/RollOutV3/Archive/RESULTS_2026-09-25.md` 부록 C; `auto_research/Archive/H6_INTPHYS1_WHAT_THE_SCORE_MEASURES_2026-09-24.md` |
| A19 | H6 "predictor 의 몫은 3.9 pt" (한 숫자) | 정정 | 가법 분해가 아니다 | 09-24 | Δ 와 CI 로만, "몫" 이라 하지 않음 | `H6_INTPHYS1_…_2026-09-24.md` 정정 |
| A20 | H6 "85.00 은 예측 없이" | 정정 (좁힘) | 표적은 미래를 본 양방향 target encoder | 09-24 | "미래를 본 표적 위에서, 예측 없이 85.00" | `H6_…` 정정 |
| A21 | H6 "v11 움직이는 물체에서 (p − 복사) 0 또는 음수" | 철회 (거짓) | 골라 인용 | 09-24 | — | `H6_…`; `auto_research/Archive/CROSS_REVIEW_2026-09-24.md` |
| A22 | H6 "이동+가림 p 가 확실히 낫다 (+20)" · IntPhys1 +20 vs v11 +4.9/−3.8 비교 | 정정 | 인과 표적에서만, 표준 표적 +5.0 n.s.; 두 비교는 표적 혼용 | 09-24 | — | `H6_…` |
| A23 | H6 "late 정지 +21.7 은 정지 물체에 몰림" | 정정 | 복사가 chance 아래라서 ("문맥이 가림막에서 끝나면 복사는 역정보") | 09-24 | — | `H6_…` |
| A24 | `copy_raw` (LN 없는 복사 77.78) 를 "복사가 약하다" 근거로 · `copyh` (누수 기준선) 인용 | 금지 | LN 복사가 표적과 같은 척도이고 예측 없는 기준선으로 알맞다; copyh 는 누수 | 09-25 | 복사는 LN 판 | `H6_…` 단서 (e); `auto_research/Archive/INTPHYS2_PERMANENCE_AUDIT_2026-09-26.md` |
| A25 | 88.89 · 85.00 을 held-out 추정으로 | 금지 | 같은 그리드에서 고른 최고 칸 = descriptive | — | descriptive 명시 | `H6_…` §5; CLAUDE.md §1-3; `TrainingEffects/ip1_copy/IP1_COPY.md` |
| A26 | IntPhys 2 "복사가 같거나 조금 낫다 / −5 pt 앞선다" | 정정 (2 차) | best-C 선택 산물. 같은 C24 에서 +1.8 [−1.8, +5.5] | 09-26 | "**복사와 구분되지 않는다**" | `INTPHYS2_PERMANENCE_AUDIT_2026-09-26.md`; `auto_research/Archive/VERIFY_ISCENE2_2026-09-26.md` §8; `auto_research/figures/story/README.md` fig3 (a) |
| A27 | "IntPhys 2 에서 chance (근처)" | 금지 | V-JEPA 2-h 57.51 (논문), human 96.4 | 09-06 | "복사와 구분되지 않는다" | `z_research/IntPhysGenV11/Archive/PAPER_STORY_2026-09-06.md`; 정본 §7 |
| A28 | IntPhys 2 수치를 한 줄에: 57.51 (Bordes) · 53.26 (ViT-L, 옛 파이프라인) · 57.94 (MATRIX, Main test 252 쌍) · 54.35 (Main 506 쌍) · 제안서 56.75 / 79.56 / 85.56 | 비교 금지 | 표본 · 프로토콜이 다르거나 미표기 | — | 54.35 (C24) · 복사 55.53 (C42) 처럼 프로토콜과 함께 | `z_training/runs/_matrix/MATRIX.md`; `z_research/IntPhysGenV11/Archive/PAPER_STORY_2026-09-06.md` 재현절; `V-JEPA 2 predictor 분석 ~하면 되지 않을까 에 답하는 실험 제안.md` |
| A29 | AR 이 "복사가 못 푸는 걸 푼다" (copy-hard 73.8) | 정정 (3 차 검증) | 무작위 겹침 귀무에서 copy-hard 정확도 = 전체 정확도 → 모두 그 아래 (release −27 · prefix −28 · full −28 · AR −10 · v11_e10 −9) | 09-29 | "AR 의 오류가 복사 오류와 덜 묶여 있다" | `auto_research/Archive/COPY_HARD_SUBSET_2026-09-29.md`; `VERIFY_R3_2026-09-29.md` C3; `auto_research/paper/RETHINK_PREDICTOR_ROLE_2026-09-29.md` §5-b |
| A30 | 복사 대비 Δ 가 작으니 "predictor 가 쓸모없다" | 금지 (해석) | DINO-F 는 w5 에선 predictor 가 27 pt 압도; Δ 작음은 평가 쪽 성질 | 09-29 | "이 벤치마크들이 예측력을 요구하지 않는다" | `auto_research/Archive/PREDICTOR_SHARE_2026-09-29.md`; `DINOF_INTPHYS1_2026-10-02.md`; `SC3_EK100_PREDICTOR_SUBSTITUTION_2026-09-25.md` |
| A31 | IntPhys 1 을 IntPhys 2 프로토콜로 (w16 67.22 / w32 66.11) | 사용 중지 (원인 미규명) | 논문 87.22 와 20 pt 차, 원인 못 찾음 | 09-21 | IntPhys 1 은 Garrido 프로토콜로만 보고 | `analysis/intphys2/configs/_superseded/README.md`; `PROTOCOLS.md` §7; `analysis/intphys2/README.md` |
| A32 | ANALYSIS_NOTES T9 "`dual_encoder` 적용하지 말 것" (67.22 < 68.89) | 뒤집힘 | 현행 `surprise_c16t32` · Garrido 하네스는 `dual_encoder: true`. 번복 근거는 이 문서들에 없음 | — | CLAUDE.md §1-7 을 따른다 | `analysis/intphys2/ANALYSIS_NOTES.md` T9 |
| A33 | v11_postft IntPhys 1 80.83 | 정정 | 철회된 Max 지표 값 | 09-25 | 77.22 | TRAINED 정정 2 (8); `auto_research/paper/REVIEW_v1_2026-09-25.md`; `auto_research/Archive/REVIEW_CVPR_ORAL_2026-09-25.md` |
| A34 | VERIFICATION "궤적 불연속 (teleport) 을 넣어라" 권고 · "궤적 · 속도 위반 추가" | 철회 · 기각 | "teleport" 는 오분류 (10 프레임 소멸 후 옳은 자리 재출현). IntPhys1 dev 에 궤적 · 속도 위반 0 건 | 08-20 | — | `z_world_model_analysis/VERIFICATION_2026-08-20.md` §2.2; CLAUDE.md §6 |
| A35 | "가려진 채 위반 70 쌍" | 정정 | 엄격 임계 (화소차 > 25, 15 픽셀 초과) 착시 → 민감 임계로 180/180 delay 0 | 08-20 | — | `VERIFICATION_2026-08-20.md` §2.3 |
| A36 | IntPhys 1 사건 전 튜블릿 "pos/imp 입력이 같다 · 차이는 fp16 잡음" | 정정 (틀림) | PNG 가 `first_div_strict` 보다 먼저 갈리는 쌍 (이동+가림 54/60, 정지+가림 30/30). 오염 쌍 빼도 결론 유지 | 09-25 | `first_div_sensitive − 1` 을 쓴다 | `z_research/TrainingEffects/ip1_copy/IP1_COPY.md` 정정 3; `z_research/Benchmarks/Archive/INTPHYS1_EVENT_POSITION_2026-09-25.md` |
| A37 | 역재생 정확도를 독립된 수로 | 정정 | `역재생 정확도 = P(정방향 맞음 ∧ 뒤집힘) + P(정방향 틀림 ∧ 안 뒤집힘)` — ssv2 tiny 93.1 = 84.7 + 8.3 | 09-20 | 정방향 정확도와 뒤집힘 비율 둘만 | `z_research/context_encoder_analysis/Archive/ENCODER_TEMPORAL_DYNAMICS_2026-09-20.md` §3 |
| A38 | EK100 "프레임 번호 vs 타임스탬프 8 비디오 0.9 %, 무해" | 철회 | val 26.7 % 가 오차 > 0.5 s, 10.9 % > 1 s (59.94 fps 도 최대 2 s 밀림; 90 fps 비디오 P18_09 있음) | 09 | `timestamp` 기본 | `z_research/anticipation/EK100/README.md` §2-B · §8; `OVERVIEW_2026-09-22.md` §7-2 |
| A39 | EK100 probe lr 1e-3 | 폐기 | 1e-3 · 3e-3 은 첫 epoch 붕괴 (상수 예측 0.45) | — | warmup 0 에서 상한 3e-4 | `EK100/README.md` §7 |
| A40 | EK100 "AR ep40 val 21.74" (09-30) | 철회 (인용 금지) | `model_kwargs.predictor_checkpoint` 키가 `pretrain_kwargs` 로 안 넘어가 **릴리즈가 돈 것** | 10-01 | 재실행: AR ep40 22.03 vs 릴리즈 21.80 (재현 잡음 안) | `auto_research/Archive/AR_PREDICTOR_2026-09-27.md` §2-9; `auto_research/README.md` 머리말 |
| A41 | "EK100 에서 못 한다" | 금지 | `paper_grid8` 19.07 은 우연 (0.45) 의 42 배 | 09-22 | — | `z_research/IntPhysGenV11/Archive/PAPER_STORY_2026-09-22.md` |
| A42 | SC5 예측 (a) "지평이 reach 를 넘으면 Δaction 이 준다" | 기각 | Δaction 1 s +2.5 · 2 s +2.1 · 3 s +2.1 (paper 2 s +2.8) | 09-26 | EK100 에 "reach 밖을 못 본다" 는 쓰지 않는다 | `auto_research/Archive/SC5_EK100_HORIZON_SWEEP_2026-09-26.md` |
| A43 | IntPhys 2 onset 방법 셋: 전역 평균차 절대 문턱 2.0 · 앞 15 프레임 MAD 상대 문턱 · 사중항 6 조합 중 앞 1 초 차 최소 조합 | 폐기 (다시 쓰지 말 것) | — | 09-30 | — | `auto_research/Archive/INTPHYS2_ONSET_2026-09-30.md` |
| A44 | `rule_auc` (파라미터 없는 대조) | 폐기 (버그) | 재계산 전까지 사용 불가 | 09-25 | "학습된 자가 규칙을 이긴다" 쓰지 않음 | V3R0925 §4-4; `z_research/RollOutV3/Archive/_superseded/PREDICTOR_FUTURE_SUMMARY_2026-09-24.md` §5 |
| A45 | 창을 합쳐 평균한 recall 곡선 | 폐기 | t12 봉우리가 어느 창에도 없는 모양 (C4 80 · C8 90 · C16 67 · C32 0 %) | 09-24 | 창별로 | `z_research/RollOutV3/Archive/_superseded/RECALL_TUBELET_LIMIT_2026-09-24.md` §6 |
| A46 | 창 가장자리 일괄 빼기 규칙 (48f 이상 마지막 칸 · C4 첫 칸) | 폐기 | C4/P4 · P8 t0 는 z · h '없다' 0–7 %, 위치 3–9 px 로 멀쩡 | 09-24 | hfail 은 `values.json` 에만 표시 | NAR0924 §1; `z_research/RollOutV3/exp_results/_superseded/windows_v6_20260923/WINDOW_SUMMARY.md` |

⚠️ A18 · A26 · A29 가 P2 (leakage) 의 근거다 — 세 행 모두 "점수 → 능력" 읽기를 거두고 **복사 대비 Δ** 로 바꿨다. IntPhys 1 의 `skip2_w16` (공식 칸) 에서는 p − 복사 +12.2 가 유의하므로, 85.0 (n.s.) 만 쓰면 반쪽이다 (`REVIEW_CVPR_ORAL` K8 권고).
⚠️ A32 (dual_encoder) 는 권고가 뒤집힌 근거 문서를 찾지 못했다 — 현행 규칙 (CLAUDE.md §1-7) 을 따른다는 것만 확실하다.

### B. 가림

| # | 옛 문장 / 도구 | 종류 | 왜 (반증 근거 수치) | 날짜 | 대신 쓸 문장 | 출처 |
|---|---|---|---|---|---|---|
| B1 | "가림 길이가 원인 ✅ 확인" (blender v4 n_hidden 3–7 64 % → 10–14 53.9 %) | 기각 | 가림 길이 · 증거 프레임 · 가림막 크기가 얽혀 분리 불가 (문서 스스로 적음). v10_occ_low 4→1 프레임 56.25→57.03 무관 · v11 k=1…4 평평 · v11 timing 양 · 문맥 안 위치 무관 | 08-20 → 08-28 → 09-01 | "물체가 가려지면 무너진다. 가려지는 **시간** 은 무관 — 무너지는 것은 가림이 예측 경계에 닿을 때뿐" | `z_world_model_analysis/OCCLUSION_LENGTH_2026-08-20.md`; `z_research/IntPhysGenV10/Archive/V10_2026-08-28.md`; `z_research/IntPhysGenV11_occlusion_timing_ablation/README.md`; CLAUDE.md §8-5 |
| B2 | "늦게 가릴수록 나쁘다" · late 를 "문맥 끝 가림" 으로 단정 | 금지 | late 는 문맥 끝 + 미래 앞 동시 가림, 문맥 끝만 가리는 팔이 없다 | 09 | "문맥 끝 가림 vs 미래 앞 가림은 못 가린다" | `IntPhysGenV11_occlusion_timing_ablation/README.md`; `PAPER_STORY_2026-09-22.md` |
| B3 | "가림막이 색면이라 색 채점을 죽인다" | 기각 | `v13_black` (검정 가림막) 에서 색 `visible→early` 하락 −14.1 → −13.5 같음. 대신 모양 −6.2 → 0.0 (나뭇결이 모양을 깎았다) | 09-03 | — | CLAUDE.md §6; `IntPhysGenV11_occlusion_timing_ablation/README.md` (본문에 옛 경쟁 가설 문단이 남아 있음) |
| B4 | "가림막은 있는데 물체를 안 가리는" 조건 신설 | 기각 (설계 결정) | "가림 유무" 는 곧 "물체가 가려지는가" 이고 가림막 존재는 그 구현 | — | — | CLAUDE.md §6 · §8-5 |
| B5 | "가림막이 화면에 있는 것 자체는 영향이 없다" | 금지 | 안 쟀다 | — | — | CLAUDE.md §8-5 |
| B6 | "가림 하에서 가속 무관" (moving_occlusion ramp vs flat) | 정정 (08-30) | 바닥 효과: ramp 54.46 / flat 56.85 둘 다 chance 근처 | 08-30 | "둘 다 바닥이라 잴 수 없다" | `z_research/IntPhysGenV11/Archive/surprising_score/RESULTS_2026-08-30.md` §3-(2); `HYPOTHESIS_SCALAR_ORDER_2026-08-30.md` §6 |
| B7 | "가림 조건은 knockout 에 덜 다친다" | 금지 | `static_occlusion` clean 54.17 (moving 과 같음) 인데 절단 후 6.25 — chance 한참 아래 = 체계적 역전 | — | — | `z_research/IntPhysGenV11/figures/knockout/README.md` |
| B8 | knockout 38 % 를 "정보가 없다" 로 | 금지 | chance 아래 (visible/vanish 13.2) = 문맥 무관 고정 예측의 외형 편향 | 09 | 절대값보다 **꺾이는 자리** 를 읽는다 | `IntPhysGenV11_occlusion_timing_ablation/README.md`; `PAPER_STORY_2026-09-22.md` |
| B9 | knockout_hidden "moving 은 이롭다" | 삭제 | moving late −2.0 / −2.6 은 대조군 범위 (−6.8 … +3.6) 안 | 09-15 | 해석 불가 | `STATE_EVOLUTION_HYPOTHESES_2026-09-15.md` blockquote; `CLAIM_NO_STATE_EVOLUTION_2026-09-15.md` §2-4 |
| B10 | "가림은 corner case" (09-03 STATUS 판정) | 판정 이력 (번복 · 범위 한정) | 09-22: 가림 없는 등속 (flat_v) 에서도 같은 성질 → corner case 아님. 09-25 I_SCENE: 실영상 움직이는 내용 약 절반 (SSv2 46 %, EK100 51 %) 이 release reach 밖. 단 **문맥 경계 가림** 자체는 corner case 로 남고 닿는 곳은 VoE 평가 방법론 | 09-03 → 09-22 → 09-25 | 판정 절차 (하네스 9) 는 정본 §4-3 이 이어받음. 주장은 비가림에서도 성립해야 한다 | `z_research/IntPhysGenV11/Archive/STATUS_AND_CRITIQUE_2026-09-03.md`; `PAPER_STORY_2026-09-22.md` §3; `auto_research/Archive/I_SCENE_CAUSAL_BOTTLENECK_2026-09-25.md` (C); `M2_BOUNDARY_PATCH_2026-09-25.md` corner case 판정; `auto_research/paper/DRAFT_v2_2026-09-25.md` §6.6 |
| B11 | "가림이 표현을 망가뜨린다" | 금지 | `env` 는 self · 이식 · k 분해 72 칸 전부 100.0 | 08-31 | — | `READING_THE_MATRIX_2026-08-31.md`; 정본 §7; CLAUDE.md §5-2 |
| B12 | "가림이 표현을 회전시킨다" / "표현이 회전했다" | 철회 → 측정 | 09-22 에 측정했고 아니었다 (저차원 · 부분 겹침). probe 가 attentive pooling + 선형이라 방향 vs 토큰 선택 구분 불가 | 08-30 → 09-22 | "직교 사상 하나로 이식이 돌아온다" 는 뜻으로만 | `RESULTS_2026-08-30.md` §12; `SYNTHESIS_2026-08-30.md`; `PAPER_STORY_2026-09-22.md`; `z_research/RollOutV3/figures/v11_probe/PROBE.md` §6 |
| B13 | "가림 조건에서 사상이 한 번 더 바뀐다 / late 의 p 는 또 다른 자리에 놓인다" (PROBE §5-3) | 철회 | encoder 쌍 (h_fut→z 65.0, z→h 42.5) 도 late 에서 같이 무너진다 | 09-25 | — | `z_research/RollOutV3/audit/v11_identity/AUDIT.md` §0-5 · §2; `PROBE.md` §5 항목 3; `RESULTS_2026-09-25.md` §6 |
| B14 | "(4) encoder 안에서도 이식 안 됨 — 정보는 있는데 자리가 다르다" | 철회 | 풀링 불일치 (8 → 1–2 튜블릿) 가 섞임; visible 단일 t7 에서도 85.5 / 56.9 | 09-25 | (4c) 를 가려진 튜블릿 이식 실패 기준선으로 쓰지 않는다 | r1 §6 각주; `audit/v11_identity/AUDIT.md` §7 |
| B15 | "p 가 정체를 지킨다" 를 영속 증거로 | 철회 | 복사 수준 — z_all 보다 낮고 [z_all, p] 로 늘지 않는다 | 09-25 | "문맥 복사 — 옮겨 싣는다, 영속 증거 아님" | `audit/v11_identity/AUDIT.md` §3; `RESULTS_2026-09-25.md` §6 |
| B16 | PROBE §4 "정체는 있는데 '여기 물체가 있다' 형태로는 없다 — 잠재적으로만 존재" | 정정 | presence 자 '있다' 0–35 % 에 기댄 문장; identity_r8 에서는 이동 끝 가림 '있음' 이 k 따라 25–62 % | 09-26 | — | `PROBE.md` §4; `z_research/RollOutV3/Archive/IDENTITY_R8_2026-09-26.md` §5-1 |
| B17 | "멈춘 물체 (static) 는 가려져도 대부분 남는다" | 철회 (판정 불가) | trapdoor 빈 장면에서도 0.23–0.92 '있다' | 09-25 | — | `z_research/RollOutV3/audit/v11_presence/AUDIT.md` (b); r1 |
| B18 | 옛 자 v11 static late '있음' 69–100 % | 정정 | 판 (가림막) 오탐; identity_r8 정지 끝 가림 17–28 % | 09-26 | — | `z_research/RollOutV3/exp_results/v11/V11_READOUT.md` 머리; `IDENTITY_R8_2026-09-26.md` |
| B19 | "t0~t4 전부 '없음'" 을 영속 증거로 | 철회 | t0 (k ≥ 3 의 t1) 은 물체가 실제로 가려진 칸 (h 도 0.11 / 0.37) | 09-25 | — | r1 §5; `RESULTS_2026-09-25.md` §5 |
| B20 | "last flat t5–t7 '있음' 은 가림막 장면 후반 오탐 (빈 장면 t6 73 %)" | 정정 | t5 는 가장자리 칸 (귀무 0.01, h 0.17); 73 % 는 다른 정의 (V11_READOUT k=1–4 합침) 의 수치 | 09-25 | — | `audit/v11_presence/AUDIT.md` (d)(e); `RESULTS_2026-09-25.md` 부록 C |
| B21 | "짧은 가림을 이어 붙인다" (r8 k=2 초반 ramp 58–69 % · flat 34–59 % '있음') | 조건부 (미결) | 자 없는 vanish block 비교에서는 이동 k=2–4 가 98–100 % 빈 미래 쪽 | 09-26 | 조건부로만 | `z_research/RollOutV3/README.md`; `IDENTITY_R8_2026-09-26.md` §7-1 |
| B22 | v11_roll_out "late 에서도 위치가 있다" | 충돌 (정본은 RollOutV2) | v11 로 학습한 자는 자리 prior 를 외운다; late 는 슬롯 0 부터 물체다운 토큰 없음 (자 없는 검사 r > 1 비율 0.54 → 0.09) | 09-07 | "가림이 경계에 걸리면 p 의 미래에는 처음부터 물체다운 토큰이 없다" | `z_research/v11_roll_out/README.md`; POSREAD §2-3 · §5-4 |
| B23 | v11_roll_out readout 이식 (`from_visible`) "자리가 바뀐다" | 정정 | 이 도구로 불가 — 천장 h 가 무너진다 | 09-07 | — | `z_research/v11_roll_out/README.md` |
| B24 | E2 "경계에 정체성 없음, 원리적으로 못 푼다 → 설계 요구 2" | 정정 (◐) | ridge 이식 실패이지 "없음" 이 아니다 (static B7−BG +35~42; late→early 36 / 41 / 73) | 09-16 | "원리적으로 못 푼다" 삭제 | `STATE_EVOLUTION_HYPOTHESES_2026-09-15.md` blockquote; `exp_results/ctxenc_boundary_identity_v2/RESULTS.md` |
| B25 | "context 에서 안 보이면 못 한다" (전부로) | 금지 | 문맥 가운데 가림은 처리한다 (vanish early 97.8 / mid 95.8) | 10-08 | "**마지막 관측에** 안 보이면" | 정본 §7; CLAUDE.md §5-5 |
| B26 | v11_occtiming pilot (6,048 pair, overall 68.11 %) | 폐기 (인용 금지) | early / mid 가 문맥을 **2k** 장 가렸다 (후속 `v11_earlymid` 는 k) | 09-01 | `v11_earlymid` · `v11_timing` 수치만 | `IntPhysGenV11_occlusion_timing_ablation/README.md`; `configs/protocols/datasets.md` (`available: false`) |
| B27 | IntPhys 1 가림 통계 v0 (58/60 쌍, 1–5 프레임, scene 단위) → v1 (O3 가짜 1 프레임 가림 5 scene) → v2 (가짜 1–1.6 초 가림 4 개) | 정정 (같은 날 두 번) | 원리별 방법 (O1 D 전체 / O2 대칭차 / O3 짝 없는 조각) 으로 교체 | 09-26 | 현행: 이동+가림 2–6 프레임 (중앙 3, skip-2 로 1–3 장) · 정지+가림 27–89 프레임 (중앙 4.9 초) | `z_research/OcclusionStats/Archive/INTPHYS1_OCCLUSION_2026-09-26.md` §1-2; `INTPHYS1_EVENT_POSITION_2026-09-25.md` §5 (대체됨) |
| B28 | IntPhys 1 "3 장 칸" 을 가림 길이 효과로 · "IntPhys 2 는 움직이는 물체를 오래 가린다" | 금지 | n=3 · 쌍 비독립 · CI 없음 / IntPhys 2 는 안 쟀다 | 09-26 | — | `INTPHYS1_OCCLUSION_2026-09-26.md` |
| B29 | IntPhys 2 저자 설명 "200 프레임 기억" | 반박 가능 (memory 프레임 주의) | k 무관 · early/mid 회복 | 09-03 | "기억" 으로 쓰지 않음 (09-30) | `STATUS_AND_CRITIQUE_2026-09-03.md` |
| B30 | AR "가림막을 더 그린다" · "AR permanence 도 같다" | 정정 | AR 은 **덜** 그림 (k=4 −.045 vs 릴리즈 −.10, k=2 n.s.); permanence 는 릴리즈보다 조금 낮고 k=2 최저 (.59) | 09-27 | — | `auto_research/Archive/VERIFY_AR_2026-09-27.md` (4)(5); `AR_PREDICTOR_2026-09-27.md` |
| B31 | AR late 정답 (v11 late 72–99) = 물체를 가림 뒤로 이어 감 | 정정 (해소) | presence-without-placement: 재등장 칸 음수는 릴리즈 · prefix 도 같고, AR 고유는 먼 칸 (ring ≥ 3) 전역 양수 오프셋 (margin 지도 ≈ 복사 지도 r .59) | 09-29 | "물체 있는 미래 쪽의 전역 편향" 뜻으로만 "presence" | `auto_research/Archive/VERIFY_PRESENCE_2026-09-29.md`; `AR40_READOUT_2026-09-30.md` (late 물체 토큰 0–10 %) |
| B32 | "presence 가 전 장면에 퍼진" | 정정 | "재등장 반경 2 칸 밖 원거리, 중앙 띠 (행 6–9) 제외" | 09-29 | — | `VERIFY_PRESENCE_2026-09-29.md` (1) |
| B33 | IntPhys 1 재등장 p − 복사 +16.4 를 무시하고 "안 보이는 것은 못 한다" | 미결 (반대 증거) | devil S5 · D8; 기준선 약함 — 정본 §4 에 답 없음 | 09-25 | README "영속: 새 증거 없음" → "미결" | `REVIEW_CVPR_ORAL_2026-09-25.md`; `11_OPEN_ISSUES_AND_NEXT.md` 로 |

⚠️ B1 · B25 가 P3 "안 보이는 것" 문장의 경계를 정한다: **"마지막 관측에" 가려진 경우만**, 그리고 **가림의 시간 (k) 은 무관**. 단 k 평평은 채점 기준이고, 자로 읽은 p 미래는 k 따라 계속 준다 (`IDENTITY_R8_2026-09-26.md` §5-4b) — 두 양은 다르다.
⚠️ B10 의 corner case 판정은 세 번 움직였다. 지금 쓸 수 있는 것은 "문맥 끝 가림은 testbed (corner case), 그러나 같은 성질이 비가림 · 실영상에서도 나온다" 까지다.

### C. 운동 · 지평 · reach

| # | 옛 문장 / 도구 | 종류 | 왜 (반증 근거 수치) | 날짜 | 대신 쓸 문장 | 출처 |
|---|---|---|---|---|---|---|
| C1 | context 길이가 문제 | 기각 | C = 4 / 8 / 12 / 16 전부 50.9–54.3 % | 08-20 | — | CLAUDE.md §6; `MECHANISM_2026-08-20.md` §6 |
| C2 | oral 첫 안 A "학습된 predictor 는 학습 지평 (8 슬롯) 까지만 외삽한다 / Horizon, not perception / stall near their own training horizon" · TRAINED "약 8 튜블릿 = 학습 지평" | 철회 | SC1: 슬롯 모형 R² 0; Ariel 은 stride 1 에서 13–17 슬롯까지 간다 (학습 지평의 2 배) | 09-25 | "멈춤의 단위는 걸음이 아니라 거리 쪽 (1 차)" | `auto_research/README.md` §2; `auto_research/paper/ORAL_DIRECTION_2026-09-25.md` §7; `auto_research/Archive/SC1_REACH_NOT_HORIZON_2026-09-25.md`; `DRAFT_v2_2026-09-25.md` §5 · §7; `auto_research/paper/STORY_EVIDENCE_2026-09-26.md` §2-4 |
| C3 | P2 "공간 조회 반경 약 2.6 칸 + 약 2 걸음" / "절벽 거리 = 2.64 칸 + 1.89 × δ (R² 0.91; 단독 모형 R² ≤ 0)" | 철회 | 사후 모형 선택; 슬롯식 +1.89 → +0.89; "R² ≤ 0" 은 1 모수와 2 모수 비교; a + b·stride 와 구분 안 됨 (ΔR² [−0.05, +0.30]); 2.6 칸은 Chebyshev ≤ 1 판독 의존 | 09-25 | 기술식 D0 2.62 [2.26, 3.12] + k 1.96 [1.52, 2.30]·δ 만, 기술로 | `auto_research/Archive/P2_STRIDE_SWEEP_HORIZON_UNIT_2026-09-25.md`; `VERIFY_0925_2026-09-25.md` (c); README_UNTIL 정정 1-3; `auto_research/README.md` §2 |
| C4 | DRAFT v0 "fixed retrieval radius (~2.6 cells)" | 철회 | 판독 의존; 멈춤 예측기 freeze-after-K (K=3) 가 더 잘 맞음 (R² 0.991 — 오라클 자신의 fit) | 09-25 | "몇 걸음 외삽 (stride 1 약 3 튜블릿)" | `auto_research/paper/DRAFT_v0_2026-09-25.md`; `DRAFT_v1_2026-09-25.md`; `VERIFY_0925_2026-09-25.md`; `REVIEW_v1_2026-09-25.md` |
| C5 | M4 "≈ 4–6 칸에서 균등, stride 무관 고정 반경" · "d 이상이면 원래대로" | 철회 | 고정 반경 없음 (사전 기준 (a) 미충족); 옛 균등 분모가 d 0–1 을 약 2 배 작게 잡음 (1–3 칸 봉우리는 분모 탓); 회복은 r ≥ d+2 | 09-25 | "조회가 닿는 거리는 칸 수로 고정되지 않음 (슬롯이 늦을수록 · stride 가 작을수록 짧다)" — 시사 | `auto_research/Archive/M4_RETRIEVAL_RADIUS_2026-09-25.md`; `VERIFY_0925_2026-09-25.md` |
| C6 | "reach 는 걸음 수 · fps · 화면 위치와 무관한 고정 거리" | 정정 (같은 날) | release 는 한 stride 안 멈춘 자리가 속도에 비례 (빠름/느림 비 1.96–2.07 vs 속도비 1.85, `REVIEW_CVPR_ORAL` K4), plateau 2.3 → 3.2 → 4.9 칸. 기각된 것은 "고정 걸음 수" 모형 (R² 0) 뿐 | 09-25 | "release 는 약 2 칸 + 1.6 걸음; 순수 거리에 가까운 건 Ariel (plateau 5–6.7 칸)" | `auto_research/README.md` §0; `SC1_…` 정정; `ORAL_DIRECTION_2026-09-25.md`; `auto_research/paper/DESIGN_CHOICES_2026-09-25.md` §4 |
| C7 | P2 "RollOutV2 2–3 칸 = 반경 항" · "RollOutV3 재현됨 (s1_C32)" | 삭제 · 철회 | 사용자 지시 "RollOutV2 가 아니라 V3 로"; 재현 철회 | 09-25 | — | `P2_…` §4 · §5 |
| C8 | P2 "s=4 에서도 지평 안 늚 → fps 반론 실패" · "문맥 길이 무관" | 강등 · 좁힘 | 문맥 가시성 (s4 중앙값 2 튜블릿) 과 섞임; hit−null 한정 (물체다움은 s1 +2.39 슬롯) | 09-25 | 시사 | `P2_…` 정정 |
| C9 | I_SCENE "거리가 주 변수" | 정정 (좁힘) | Chebyshev ≤ 1 판독에서만. 정확 칸 판독: release 슬롯 계수 −0.074 > 거리 −0.049. 연속 판독 (2 차): Ariel 실영상 시간 효과가 거리의 2–3 배 | 09-25 / 26 | 판독을 밝히고 쓴다 | `I_SCENE_CAUSAL_BOTTLENECK_2026-09-25.md` §4-1; `VERIFY_ISCENE_2026-09-25.md` C1; `VERIFY_ISCENE2_2026-09-26.md` |
| C10 | "팬에서는 슬롯 효과가 작고" | 철회 | τ 0.02 에서 모든 판 슬롯 > 칸, τ 0.1 에서 둘 다 ≈ 0; 균등 지도 귀무 ρ = 0.5 | 09-26 | "팬은 둘을 가르기 어렵다" | `VERIFY_ISCENE2_2026-09-26.md` §4-7 |
| C11 | "attention 감소는 슬롯 탓 (−0.17) > 속도 탓 (−0.08)" | 정정 (같은 날) | 주 변수는 마지막 관측 자리에서의 **이동 거리** (자 attention: 슬롯 −0.26 / 거리 100 px 당 −0.43; 자 없는 검사: −0.28 / −0.47) | 09-19 | "물체다운 토큰은 마지막 관측 자리에서 약 2–3 칸 안에서만 유지" | DISTLIM; `z_research/RollOutV2/README.md` §4c; `OVERVIEW_2026-09-22.md` §7-2 |
| C12 | H23 "지속을 정하는 것은 시간과 거리 둘 다 (슬롯 −0.28, 칸 −0.90) — 한 모형의 두 항" | 철회 | 귀무 미보정 — 슬롯 항 −0.275 는 귀무 자체의 기울기 (h23b Mn null t −0.265) | 09-24 | — | `auto_research/Archive/H23_PREDICTOR_LAYERS_TRAJECTORY_2026-09-24.md`; `auto_research/exp_results/h23/REPORT_h23.md` §1; `CROSS_REVIEW_2026-09-24.md` B3 |
| C13 | "p 가 슬롯 4 이후 멈춘다" / "문맥 속도로 조금 가다 멈추는 고정된 동작" / 제안서 "GT 보다 덜 움직이고 후반에 정지" | 철회 | p 는 멈추는 게 아니라 **물체를 놓친다** (후반 속도 0 = 자 기본값); flat_v 는 슬롯 5 까지 진실 추종; 확실히 멈추는 건 arc 수평뿐 | 09-19 | "물체다운 토큰이 마지막 관측 2–3 칸 (40–60 px) 안에서만 유지된다" | CLAUDE.md §6; DISTLIM §8; `OVERVIEW_2026-09-22.md` §7-2; `V-JEPA 2 predictor 분석 ~하면 되지 않을까 에 답하는 실험 제안.md` |
| C14 | "p 가 문맥 끝 속도를 그대로 이어 간다" | 기각 | 등속 arc 에서도 약 3 칸 (53 px, 진실 92) 가서 멈춤 | 09-19 | — | CLAUDE.md §6; DISTLIM |
| C15 | 초반 과속 (1.5–2 배) 주장 | 금지 (주장 불가) | 1 칸 안 · 자 오차 ~10 px | 09-19 | — | DISTLIM; `RollOutV2/README.md` |
| C16 | "p 는 8 프레임 뒤부터 물체를 덜 보낸다" (기울기 감소) | 철회 | 기울기 감소 시점 = '있다' 하락 시점; 자 기본값 끌림 (C13 과 같은 함정) | 09-24 | — | `PREDICTOR_FUTURE_SUMMARY_2026-09-24.md` §4 (#21) |
| C17 | "p 는 t8 근처에서 절벽처럼 물체를 놓고, 자리는 튜블릿 수가 정하며, 깜빡인다" ("C16 은 t10 에 '없다'") | 철회 | presence 자의 성질. identity_r8 에서는 절벽 없음 ('있음' 84–100 %), 대신 조합 정답이 t8 부터 80 → 17 % | 09-26 | — | `RECALL_TUBELET_LIMIT_2026-09-24.md` 머리; `IDENTITY_R8_2026-09-26.md` §4; `z_research/RollOutV3/exp_results/windows/BEHAVIOR.md`; `exp_results/windows/CROSS_HEAD.md` |
| C18 | r1 "지속 절벽은 창마다 다르다" · "지속 ❌ (8–10 튜블릿 뒤 소실)" | 되돌림 · 정정 | 2 차 검토가 "튜블릿 번호에 묶인다" 로 되돌림 (이후 절벽 자체가 새 자에서 사라짐); "소실" 이 아니라 "자가 '없다' 로 읽는다" | 09-25 | — | r1 머리 blockquote |
| C19 | v2 "거리 2–3 칸" 을 v3 "튜블릿 수" 로 대체 | 금지 | 잰 것이 다르다 (물체다운 토큰 위치 vs presence 자 recall) | 09-24 | 조건 표로 병기 | `PREDICTOR_FUTURE_SUMMARY_2026-09-24.md` §5; `RECALL_TUBELET_LIMIT_2026-09-24.md` §7 |
| C20 | "감속 → 앞섬, 가속 → 뒤처짐, 낙하 · 포물선 → 덜 떨어짐, 벽 → 계속 감 = 문맥 운동의 연장" | 철회 | 자 · 창 의존 — h 자로 flat_d −6.4, C8/P32 −0.9; 가속 반영은 판정 불가 | 09-25 | — | r1 §2; `z_research/RollOutV3/audit/v3/AUDIT.md`; `RESULTS_2026-09-25.md` 부록 C; `z_research/TrainingEffects/v3_motion/MOTION.md` (서명이 자에 걸려 있다) |
| C21 | "자유 운동 x 는 t9 까지 GT 를 거의 겹쳐 따라간다" | 철회 (틀림) | t9 오차 23.4 px (1.3 칸); 등속 연장이 모든 t 에서 더 가까움; ±10 px 는 t0–t4 까지만 | 09-25 | — | r1 §3; `audit/v3/AUDIT.md` |
| C22 | "p 속도 톱니, z 도 약하게" | 철회 (틀림) | 톱니는 p 자기 자 읽기에만, 화면 x 고정 교대 (−3.6 px/튜블릿); z 는 부호 반대 · 1/4 | 09-25 | 속도는 2–3 칸 추세로만 | r1 §3; `RESULTS_2026-09-25.md` 부록 A-2 |
| C23 | "ledge 의 덜 떨어짐 = 연장의 증거" | 정정 | 사건 순간 수직 속도 0 → 복사 = 연장 = −16.5 (p −16.6) → y 로 못 가름 | 09-25 | x 는 복사보다 +17.7–18.6 앞섬 (기록) | r1 §2; `RESULTS_2026-09-25.md` 부록 A-1 |
| C24 | "flat · wall y 오차 ~20 px 는 t0 부터 일정한 치우침" | 정정 | h 자로 읽은 p 의 y 는 +18–21 → +34–46 → +60–65 px 로 커진다 (top1 무너지는 시점) | 09-25 | — | r1 "뺀 것" 표 |
| C25 | "wall 은 벽에서 되튄다" (부호 정의 ① 판) | 철회 (틀림) | GT 멈춤 (0) 을 → 로 표시한 버그; 부호 정의가 같은 날 4 번 바뀜 | 09-24 | "정지점을 0.6–0.9 칸 지나간다 (C16)" (생존 편향 단서) | NAR0924 §4; r1; `RESULTS_2026-09-25.md` 부록 A-1 |
| C26 | "wall 에서 벽 속 한 칸 반 (+29 px) 파고들어 멈춘다" · "87 % 통과 쪽" · 속도별 침투 표 | 정정 | 슬롯 1–2 통과 + 슬롯 3–7 기본값의 평균. 멈춤은 한 번도 없다 | 09-12 | "p 는 2 슬롯 통과 뒤 물체를 잃는다" (통과 질량 0.61 / 0.48 → ≤ 0.18) | POSREAD §4-3 · §8; CLAUDE.md §5-5 · §6; `STORY_EVIDENCE_2026-09-26.md` §2-4 |
| C27 | H23 wall 방향 서술 ("초반 벽 통과 쪽, 뒤로 갈수록 정지 쪽") | 철회 | 복사 기준선이 재현 | 09-24 | — | `H23_…`; `REPORT_h23.md` |
| C28 | FUTURE_FROM_CONTEXT 의 "wall 에서 p 도 멈춤을 읽는다" ↔ POSREAD "2 슬롯 통과 뒤 잃음" | 조건 병기 (모순 아님) | 다른 자 (장면 종류 → 템플릿) | 10-02 | 자 이름을 같이 쓴다 | `auto_research/Archive/FUTURE_FROM_CONTEXT_2026-10-02.md`; POSREAD |
| C29 | "ledge 에서 수평으로 나아가다 멈춘다" | 부분 정정 | 부유 (슬롯 3–4) 는 선다; 슬롯 5–7 은 물체가 흐려진다 | 09-12 | "p 는 선반 높이를 유지한다 (부유 궤적 attention 0.43 / 0.39)" | POSREAD §8 |
| C30 | 검정 A "ledge 쌍 74–100 % 가 떠 가는 미래에 더 가깝다 — 낙하를 안 만든다" (능력 주장) | 철회 (능력 주장에서 뺌) | 복사 기준선이 59–99 % 로 재현 (떠 가는 미래는 복사와 닮음) | 09-24 | "낙하하는 미래를 만들지 않는다" + 다중 미래 설계로 | `H23_…` §3-7; `REPORT_h23.md`; `CROSS_REVIEW_2026-09-24.md`; 정본 §2-3 iii |
| C31 | FUTURE_FROM_CONTEXT "v2 ledge 에서 p 는 안 된다" | 정정 (과장) | p 자 B 도착: p-B 5.4 px · 낙하량 52.8; p 가 z 에 분명히 못 미치는 곳은 v11 late 가림뿐 | 10-02 | 결론 2 를 v11 late 로 좁혀 읽기 | `FUTURE_FROM_CONTEXT_2026-10-02.md` |
| C32 | CONTEXT_TO_FUTURE 첫 판 "가속 / 등속 / 감속 순서를 지킨다 (flat_a −50 > flat_v −65 > flat_d −95 px)" · "먼 미래에서 속도 · 가속 차이를 둘 다 30–40 % 남긴다" | 정정 | 세 법칙의 문맥 속도가 달라 생긴 순서 (flat_d 가장 빠름) — 같은 속도면 세 선이 겹침; 속도는 줄여서 남고 가속은 거의 안 남는다 | 09-26 | "가속은 거의 반영하지 않는다" | `z_research/RollOutV3/Archive/CONTEXT_TO_FUTURE_2026-09-26.md` §7; `figures/context_to_future/_superseded/first_draft_2026-09-26/README.md` |
| C33 | "p 가 감속을 만든다" · "감속에서 진실과 맞음 (90 vs 92)" | 금지 | 슬롯 7 물체 칸 질량 감속 0.15–0.17 < flat_v 0.37; 감속 일치는 거리 한계와 구분 불가 | 09-12 / 19 | — | `RollOutV2/README.md`; DISTLIM |
| C34 | "p 는 힘을 반영하지 않는다" (전부로) | 정정 | arc 상승 −27 | 09-19 | 조건을 붙여서 | DISTLIM; `PAPER_STORY_2026-09-22.md` |
| C35 | "중력을 잘 모른다" † | 철회 (틀 자체 금지) | arc · ledge y 는 복사 ≈ 연장, ramp_a y 판정 불가 | 09-25 | "낙하하는 미래를 예측하지 않는다" | r1 (#15); `RESULTS_2026-09-25.md` 부록 C |
| C36 | AR40 "방향 무관하게 느려진다" (ramp_a · 감속 포함) | 정정 (좁힘) | ramp_a · 감속은 t8–15 = rollout 9–16 블록 (학습 최대 8) **분포 밖**; 분포 안 flat_d 는 진실 추종 | 09-30 | 가속 반영 음수는 arc · ledge t0–7 에서만 확정 | `auto_research/Archive/AR40_READOUT_2026-09-30.md` 머리; `VERIFY_AR40RD_2026-09-30.md` |
| C37 | AR40 flat_a 가속 반영 음수 | 철회 | 오프셋 0.5 에서 AR +0.03 n.s. | 09-30 | — | 같음 |
| C38 | H1 C32 r8 0.76 을 "encoder 에 가속이 있다" 근거로 | 금지 | 역사 기준선 B_xfirst 0.93 · B_entry 0.90 아래; C8 0.01 · C16 0.51 | 09-25 | 가속은 미결 · 조건부 (Joseph 2026 과 충돌) | `REVIEW_CVPR_ORAL_2026-09-25.md`; `H1_ACCEL_IN_INPUT_2026-09-24.md`; `LITERATURE_POSITIONING_2026-09-25.md` |
| C39 | H5 "느린 걸음의 복리 수축 (걸음당 60 %)" | 철회 (다시 쓰지 말 것) | 60 % 는 H4 누적 진행률 | 09-24 | — | `auto_research/Archive/H5_AUTOREGRESSIVE_FEEDBACK_2026-09-24.md` |
| C40 | H5 제목 "자기 예측을 되먹이면 물체는 남지만 멈춘다" · "되먹임으로 존재는 유지된다 (0.88)" | 철회 | p10 문턱에서만; ar_z 와 모순 | 09-24 | "1 칸 해상도에서 전진을 멈춘다" (CLAUDE.md §6 철회와 다른 이유: 자 없음 · 물체다운 표본 · 속도 비례 · 귀무 아래) | `H5_…` |
| C41 | "미래를 1 튜블릿씩 예측해 이어붙였더니 된다" | 철회 | 된 줄 (`ar_z`) 은 매 걸음 진짜 관측을 넣은 **탐침**. 자기 예측을 이어붙이면 멈춘다 (−0.012 [−0.042, +0.018]) | 09-25 | "진짜 다음 관측을 넣으면 한 걸음 전이는 된다 (탐침)" | `auto_research/README.md` §2; README_UNTIL; `P3_CLOSURE_TEST_2026-09-25.md` |
| C42 | P3 "처음 세 걸음 = P2 '반경 + 약 2 걸음' 의 걸음 항" | 철회 | 숫자 맞추기 | 09-25 | — | `P3_CLOSURE_TEST_2026-09-25.md` |
| C43 | README_UNTIL "C16 j3–7 기울기 −0.009 [−0.017, −0.002]" | 정정 | clip 단위 CI; 궤적 CI 로는 −0.012 [−0.042, +0.018] (0 과 구분 안 됨) | 09-25 | — | README_UNTIL 정정 1-9 |
| C44 | README_UNTIL "이동량 (적분 이득) 은 진실 변위의 약 0.4–0.7" | 정정 | 두 국면의 평균: t0–2 진실 이상 (0.65 vs 0.46), t3–7 0.07 로 멈춤 (b 4.60 vs 진실 6.07) | 09-25 | 두 국면으로 쓴다; "3" 은 시사 | README_UNTIL 정정 2-4 (X21) |
| C45 | H4 "빠른 물체가 더 빨리 사라진다 (거리)" · "위치 밀면 사라지는 k 가 늦춰진다" · 진행률 0.5–0.67 | 정정 | 속도 3 분위가 법칙과 교락; 공통 문턱에서 k 3–13 만; 0.40–0.67 | 09-24 | — | `auto_research/Archive/H4_QUERY_AND_POSITION_2026-09-24.md` |
| C46 | 시간 정렬 "슬롯 0–3 제 시각, 4 에서 절벽, 물체가 4 슬롯 (1.5 s) 뒤 없어진다" | 정정 | 배경 토큰만으로도 대각선 (flat_v 100 %) → gated 판: 등속 3–4 / 가속 · 수직 ≤ 1–2; "없어진다" 는 진실 자리에서만, 옛 자리 유지는 미결 | 09-15 / 16 | — | `z_research/context_encoder_analysis/exp_results/time_alignment/time_alignment.md`; `exp_results/ctxenc_time_alignment_gated/RESULTS.md`; `CLAIM_NO_STATE_EVOLUTION_2026-09-15.md` v0.1 |
| C47 | 자 gain 0.4–0.5 = "시간이 절반쯤 흐른다" | 철회 | 슬롯 4–7 기본값이 섞인 값 | 09-15 | — | `STATE_EVOLUTION_HYPOTHESES_2026-09-15.md` §3-1-4 |
| C48 | 옛 "gain 0.85" (RollOutV2 릴리즈 flat_v) | 정정 | 선형 층의 과소평가 — 최댓값 칸 β_peak 1.01 | 09-16 | — | `z_research/predictor_training/predictor_IntPhysGenV11_PFT/Archive/RESULTS_2026-09-16.md` §2-1 |
| C49 | 09-06 beat 4(a) "물체 위치가 시간에 따라 진전하지 않는다" | 정정 (틀림) | 진전 / 정지 이분법 밖: 2–3 칸 진전 후 놓는다 | 09-22 | "마지막 관측 근처 2–3 칸을 그린다" | `PAPER_STORY_2026-09-22.md`; `PAPER_STORY_2026-09-06.md` |
| C50 | CLAIM v0 "낙하 시작은 슬롯 0–2 안" · "옛 자리에서 외삽하지 않는다" · "슬롯 4 이후 물체는 없다" · "EK100 1 s 는 지평 안" | 철회 · 한정 | 같은 토큰 창 / 해석 불가 개입 / 진실 자리에서만 / 4 튜블릿째 = 경계 (토폴로지 · mask 다름) | 09-15 | — | `CLAIM_NO_STATE_EVOLUTION_2026-09-15.md` §7 |
| C51 | E-SSv2 "predictor 의 미래는 시간을 따라 진전하지 않는다 (기울기 0.01)" | 철회 | 0.01 은 두 방향 평균 (정 −0.244 / 역 +0.269); 귀무 보정 슬롯 0–3 기울기 +0.46 [+0.40, +0.52] | 09-25 | — | `auto_research/Archive/E_SSV2_ENCODER_READS_PREDICTOR_STALLS_2026-09-25.md` 정정 1; README_UNTIL 정정 1-5; `STORY_EVIDENCE_2026-09-26.md` §2-4 |
| C52 | E-SSv2 "진실 변화의 약 15 % 크기로 진전" · "오차 맞춤 오라클 정렬 1.00 → 진전 부족" · "약 3 슬롯 따라간 뒤 시간 특이성이 준다" | 철회 · 좁힘 | 크기비는 진전이 아님 (진실 추세 오라클도 0.12–0.13); 독립 오차라 자명; 줄어드는 건 경계 인접 슬롯 일치 (ρ 0.143 → 0.033) | 09-25 | 분해 확정 · 해석 시사 | `E_SSV2_…` 정정 2; `VERIFY_0925_R2_2026-09-25.md` X14 · X15; README_UNTIL 정정 2-9 |
| C53 | qshift "위치를 맞춰 주면 한계가 사라진다" · "먼 거리 적중이 근거리와 같은 수준" | 철회 | 먼 적중 = base 근거리의 53–72 %, 정확 칸 .08–.09; 팬 근거리 비용 −0.2 (망가뜨림 문턱 초과); 출처 지나침 32–58 % | 09-26 | 정답 출처 **상한** (시사) | `VERIFY_ISCENE2_2026-09-26.md` §4-6; `auto_research/figures/story/README.md` fig2; `DIRECTION_2026-09-25.md` §4 |
| C54 | "더 학습해도 reach 는 안 는다" · fig5 (a) 제목 "saturates early" | 좁힘 | cosine lr 끝과 구분 안 됨 → EK100 확정 · SSv2 시사 · 팬 미결 | 09-26 | "EK100 reach unchanged by training" 류 | `VERIFY_ISCENE2_2026-09-26.md` §5-3; `auto_research/figures/story/README.md` |
| C55 | "시간 한계 (MOTION, v3)" vs "거리 한계 (DISTLIM, v2)" 를 하나로 | 금지 (모순 병기) | 자료 · 창이 다르다 | 09-25 | 조건과 같이 둘 다 | `TrainingEffects/v3_motion/MOTION.md`; DISTLIM |

⚠️ C11 의 회귀 (거리 −0.47 / 100 px, 슬롯 −0.28) 자체도 귀무 미보정 · clip 단위라는 리뷰 지적이 있다 — "재적합 전 논문 문장 금지" (`REVIEW_CVPR_ORAL_2026-09-25.md` R2, `REPORT_h23b.md`). 지금 쓸 수 있는 문장은 "2–3 칸" 이라는 **기술** 까지이고, "거리가 단위다" 는 판독 · 표본에 따라 답이 다르다 (C9 · C10 · `11_OPEN_ISSUES_AND_NEXT.md`).
⚠️ C13 · C16 · C17 · C18 은 모두 같은 함정 — **자의 기본값을 멈춤 · 절벽으로 읽음** (CLAUDE.md §7-3). 행동 문장을 쓰기 전에 그 슬롯의 3×3 attention 질량 > 균등 0.035 를 확인한다.

### D. 자 (readout) · 판독

| # | 옛 문장 / 도구 | 종류 | 왜 (반증 근거 수치) | 날짜 | 대신 쓸 문장 | 출처 |
|---|---|---|---|---|---|---|
| D1 | **presence 자 (`dcd24d8a8a47`) 전체** | 폐기 | v11 가림막을 물체로 읽음 (판 빈 장면 '있음' p 7–45 %); presence 하나로는 "만들었는데 안 움직임" vs "못 만듦" 을 못 가름 | 09-25 (사용자) | 위치 + 정체 57 분류 자 `identity_r8` (빈 장면 오탐 0 %) | `z_research/RollOutV3/figures/{v11_presence,v3_signed_error,v3_signed_error_hhead,v3_trajectory}/_superseded/presence_decoder/README.md`; `z_research/RollOutV3/README.md`; `exp_results/presence/RESULTS.md` 머리 |
| D2 | 옛 identity 자 (`aeda55874c1c`) | 폐기 | presence 자와 같이 v11 가림막을 물체로 읽음 | 09-26 | `identity_r8` | `z_research/RollOutV3/README.md`; `figures/*/_superseded/identity_aeda55874c1c/README.md` |
| D3 | presence 자 시기 판 전체 (`figures/_superseded/presence_era_2026-09-24/`, V3R0925, r1, NAR0924, RECALL_TUBELET_LIMIT, PREDICTOR_FUTURE_SUMMARY, windows/BEHAVIOR · CROSS_HEAD · WINDOW_SUMMARY, v11/V11_READOUT) | 대체 | 물러난 이유 둘: ① 자가 v11 판을 물체로 읽음 ② "t10 절벽" 이 새 자에서 안 나옴. 선반 가설 (가중치 원인) 은 검증 안 함 | 09-26 | `IDENTITY_R8_2026-09-26.md` · `RESULTS_2026-09-25.md` 정의가 정본 | `z_research/RollOutV3/figures/_superseded/presence_era_2026-09-24/README.md`; 각 문서 머리 |
| D4 | "**p 가 물체를 마지막 관측 자리에 둔 채 머문다**" (v11 가림) | 철회 | 자의 기본값 (토큰 평균 읽기 ≈ 화면 중심 ≈ 마지막 관측). 마지막 관측 3×3 질량 0.01–0.03, 읽기 = 기본값 ± 19–36 px | 09-12 | "**가림이 경계에 걸리면 p 의 미래에는 처음부터 물체다운 토큰이 없다**" | POSREAD §8; CLAUDE.md §5-5 · §6; `IntPhysGenV11/README.md`; `LITERATURE_POSITIONING_2026-09-25.md`; 정본 §7 |
| D5 | "가림막 뒤에 둔 채" | 철회 | 가림막 칸 질량 균등 수준 | 09-12 | — | POSREAD §8 |
| D6 | "early / mid 에서도 타이밍과 무관하게 멈춘다 (슬롯 7 비율 0.1)" | 정정 | early / mid 슬롯 0–2 는 k=0 처럼 정확 | 09-12 | — | POSREAD §5-4 · §8 |
| D7 | "슬롯 7 진행 비율" 을 단독 요약 지표로 | 폐기 | 물체가 사라진 뒤 기본값을 섞어 early / mid 를 과소평가 | 09-12 | 슬롯별 질량 표 (`attn_diag.md`) 와 함께 | POSREAD §5-4; `z_research/RollOutV2/figures/v5/v11_vanish/timing/timing_summary.md`; CLAUDE.md §6 |
| D8 | v11 k=0 p 만으로 위치 자 학습 | 기각 | 자리 prior 를 외움 (오차 3 px, 물체 attention 0.1; early 슬롯 0–2 에서 47–63 px) | 09-12 | 위치를 고루 덮는 학습셋 | CLAUDE.md §6; `RollOutV2/README.md` §7; POSREAD |
| D9 | 위치 자 학습셋 v3 (kink · 고속) | 기각 (재시도 금지) | 자가 학습 분포 안에서도 흐려짐 | 09-12 | — | CLAUDE.md §6; POSREAD §8 |
| D10 | v4 자 (사물 없는 학습셋) 로 ledge / wall 판정 · soft-argmax 자 · presence head + 빈 장면 학습셋 | 대체 · 취소 · 보류 | v4 대체; soft-argmax 는 사용자 취소; presence head 는 09-12 사용자 보류 | 09-12 | — | POSREAD §8; `RollOutV2/README.md` 읽는 규칙 (5) |
| D11 | 학습셋 판 뒤 튜블릿 16–20 px 읽기 · [h_ctx ; p] probe 94 / 78 % | 인용 금지 | 증거 아님 / 이음새 탐지 | 09-12 | — | POSREAD §7; `RollOutV2/README.md` §7 |
| D12 | "p 안에 물체다운 토큰이 없다" 를 p 내용의 직접 증거로 | 좁힘 | 이 자가 물체로 인식하는 토큰이 없다는 뜻 (자 없는 검사 C 로 보강됐으나 직접 증거는 아님) | 09-12 | — | POSREAD §2-3; `figures/v5/v11_vanish/timing/token_test.md` |
| D13 | 자 읽기에서 attention 질량 문턱 없이 위치 주장 | 금지 (규칙) | 자는 물체가 없어도 위치를 낸다 (attention 이 퍼지면 토큰 평균 = 기본값) | 09-12 | 그 슬롯 3×3 질량 > 균등 0.035 일 때만 | `RollOutV2/README.md` 읽는 규칙 (1)(2); CLAUDE.md §5-5 · §7-3 |
| D14 | **`v11_dh_gif`** — 토큰 차이 크기 (\|h(물체) − h(빈)\|) 로 물체 자리 찾기 | 폐기 | target encoder 가 전역으로 반응 (토큰 코사인 중앙 0.52, 1e-3 잡음에도 0.125 / 0.235 변함); 최댓값이 물체 칸인 튜블릿 15 % (19/126); PCA · 평균 제거도 0–12 % | 09-25 | 탐색용, 결론에 쓰지 않음 | `z_research/RollOutV3/figures/_superseded/v11_dh_gif/README.md`; `z_research/RollOutV3/figures/README.md` |
| D15 | `v11_presence` 이진 (과반) 판 | 폐기 | 50 % 근처 (문맥 끝 가림 flat 34–63 %) 를 깜빡임처럼 보이게 함 | 09-27 | 비율 판 | `z_research/RollOutV3/figures/v11_presence/_superseded/binary_majority_2026-09-27/README.md` |
| D16 | "두 자가 부재에 합의한다" (강한 주장) | 철회 | 둘 다 '없다' 일치 52–76 % | 09-25 | — | V3R0925 §4-3; `exp_results/windows/CROSS_HEAD.md` |
| D17 | 절벽의 정확한 위치 · 깜빡임을 숫자로 인용 | 금지 | 두 자가 중간 구간에서 서로 다름 (CROSS_HEAD 본문 "C16 t5–t12 일치 30–63 %" 는 같은 문서 표 52–96 % 와 안 맞음) | 09-24 | — | `exp_results/windows/CROSS_HEAD.md`; `exp_results/_superseded/windows_v6_20260923/CROSS_HEAD.md` |
| D18 | BEHAVIOR clip 별 깜빡임 R · 벗어남 작은 법칙의 g | 읽지 말 것 (초안) | z 도 R 이 커서 자 실수와 못 가름; g 가 자 위치 오차보다 작음 | 09-24 | — | `exp_results/windows/BEHAVIOR.md` |
| D19 | 가림 음성을 자 학습셋에 넣기 | 폐기 (사용자 결정) | — | 09-25 | 가림 판정은 자 없는 vanish 짝 비교로 | V3R0925 §4-2 |
| D20 | 새 렌더 자 학습셋 1 차 · 2 차 판 | 폐기 | 1 차: 판이 서 있거나 움직이는 빈 장면 0 개, 가림 열 틀림 (판 clip 1,296 개 `has_occlusion=1`). 2 차: 판 위치 x MI 0.184 bits · 폭×높이 0.255 (라벨 정보의 절반) | 09-25 / 26 | 3 차 통과 (위치 x 0.0033 · 폭×높이 0.0059 bits) = `training_r8` | `z_research/RollOutV3/audit/training_sets/RollOut_v2_training_v8.md`; `audit/training_v8/AUDIT.md` |
| D21 | wall · ledge 가능 / 불가능 판정을 위치 자로 | 금지 | 분리량 P=8 28.7 px 인데 자 오차 15–20 px, 좌표 편향 Δx +25 px 가 결론을 뒤집음 ("정지 예측" ↔ "통과 예측") | 09-23 | 자 없는 잠재 공간 검사 (\|p−h_pos\| vs \|p−h_imp\|) | `z_research/RollOutV3/Archive/READOUT_CHOICE_2026-09-23.md` ②; V3R0925 ⑦ |
| D22 | attention 지도 top1 로 '있음' 판정 | 금지 | v3 0.03–0.27 평평; v6 "물체 있음" 0.262 vs "화면 밖" 0.267; 균등 1/256 | 09-23 | — | `READOUT_CHOICE_2026-09-23.md` ③; `exp_results/presence/RESULTS.md` |
| D23 | 고정 문턱 '있다' 판독을 단독으로 | 금지 | 자 (p@h vs p@p) 에 따라 부호가 뒤집힌다 (IntPhys 2 학습판 영속 −0.24 vs +0.13); 척도 이동이 섞임 | 09-25 | AUROC · 순 배치 Δ 와 같이 | `z_research/TrainingEffects/README.md` §5 (2); `TrainingEffects/v11_presence/PRESENCE.md` 정정 (4) |
| D24 | p@h 를 "주 판독 · predictor 무관한 자" 로 | 정정 | p 토큰에서 따로 검증 안 됨 (평지에서 릴리즈 p 를 19–40 px 위로 읽음) — decoder-first 규칙 위반 | 09-25 | — | `TrainingEffects/v3_motion/MOTION.md` 정정 (9); `TrainingEffects/Archive/VERIFICATION_2026-09-25.md` |
| D25 | 층별 · 모드별 · 명세별 **자기 문턱** 으로 비교 (H4 · H8 · H23) | 폐기 (다시 하지 말 것) | 문턱 인공물 — 공통 문턱에서 부호가 뒤집힘 | 09-24 | 하나의 고정 문턱 + contrast 분포 병기, 적중은 hit − null | `CROSS_REVIEW_2026-09-24.md` A1; `H8_KNOCKOUT_PERSISTENCE_2026-09-24.md` "폐기한 해석" |
| D26 | RollOut_v3 를 clip 단위 (n = 2,352 독립) CI 로 | 폐기 | 유효 표본은 궤적 84 (법칙 14 × 외형 28 복사); 588 은 rank 당 수 | 09-24 | 궤적 단위 bootstrap | `H4_…`; `H8_…` 정정 9; `auto_research/exp_results/h23b/REPORT_h23b.md`; `CROSS_REVIEW_2026-09-24.md` A5 |
| D27 | H1 clip 단위 split · 법칙 **안** 회귀 | 금지 (다시 하지 말 것) | 같은 궤적이 train · test 에 들어감 (궤적 복사본 누수); within R² 음수; 법칙 안 \|v\|–\|a\| 상관 0.95–0.98 | 09-24 | 궤적 단위 split | `H1_ACCEL_IN_INPUT_2026-09-24.md`; `auto_research/Archive/PLAN_2026-09-24.md` 변경 1 |
| D28 | M2 물체다움 문턱 1 판 0.234 · "파괴 1.22" · 물체다움 절대 수준 = 존재율 | 철회 | 빈 장면 672 쌍이 섞인 문턱; 빈 장면 템플릿도 0.78–0.83 켜짐 | 09-25 | 문턱 0.313 + 0.357 | `M2_BOUNDARY_PATCH_2026-09-25.md`; `PLAN_ENCODER_PREDICTOR_2026-09-25.md`; README_UNTIL 정정 1-2 |
| D29 | SC2 1 판 — encoder 템플릿 argmax 궤적을 유사 정답으로 (`e_ssv2_track.py`, `ssv2_track*.json`) | 무효 (인용 금지) | 궤적이 슬롯 사이 평균 5.5 칸씩 뛰고 질의 선택 에너지 평평; flow 보정 첫 판 · 둘째 판도 폐기 | 09-25 | flow 판 (SC2v2) | `auto_research/Archive/SC2V2_FLOW_REACH_SSV2_2026-09-25.md` §1; `auto_research/README.md` §2; `ORAL_DIRECTION_2026-09-25.md` |
| D30 | E-SSv2 "p 는 어디서 바뀔지 안다 (변화 지도 ρ 0.33–0.44)" · "zf 기울기 1.00 = 지표 검증" · "흐림 · 희석으로 기울기 안 무너짐" · "슬롯 4 뒤 이득 0 = 진화 없음 지표" | 철회 | 공유 기준 h_7 인공물 / 자명한 인덱스 정렬 / 옛 오라클 검출력 없음 · 출처 파일 없음 / 완벽 진전 오라클도 같은 윤곽 | 09-25 | — | `E_SSV2_…` 정정 1; `VERIFY_0925_2026-09-25.md` |
| D31 | v3 법칙별 운동학 지표 (궤적 R² · z 대비 R² · 부호 검정 · 물리 파라미터 복원 · clip 간 회귀 R² · 속도별 평균 R² · 기울기) | 폐기 | 연장 ≈ 실제 (R² 0.99); z 는 공정한 상한 아님; clip 속도 SD 3 px < 자 오차 6.3 / 12.2 → z 도 R² 0.05; 설계 축이 창이지 법칙 아님 | 09-24 | — | `PREDICTOR_FUTURE_SUMMARY_2026-09-24.md` §4 (#20); `presence_era_2026-09-24/direction/DIRECTION_R2.md` |
| D32 | `fig_confusion_self` · `fig_self_arm_k` | 삭제 (다시 만들지 말 것) | 두 `p` head 미수렴 (0.931 / 0.954, sta+OCC 는 채점이 살아남는 조건) / 9 패널 중 7 개 100 | 08-30 | — | `z_research/IntPhysGenV11/figures/probing/README.md`; `RESULTS_2026-08-30.md` |
| D33 | "R² 높으면 1 차원" · "confusion 무상관이 스칼라 가설 증거" · `corr(bias, C(X→Y))` 를 근거로 · §3-2 와 §4-2 를 같은 null 로 | 폐기 | 무작위 기준선 0.286 ± 0.13; 반대칭 vs 대칭은 구조적 0 | 08-30 | `corr(s, recall) > 0` 사전 등록 | `HYPOTHESIS_SCALAR_ORDER_2026-08-30.md` |
| D34 | (조건, k) 칸 하나 · 쌍 단위 칸 단독 인용 | 금지 | 비대각 방향당 n=4 (쌍 단위 방향당 3–7) | 08-31 | 행 단위 · 21–28 칸 평균으로만 | `READING_THE_MATRIX_2026-08-31.md`; `VOTE_GRID_2026-08-31.md`; `TALK_2026-08-30.md`; CLAUDE.md §7-3 |
| D35 | "위치가 읽히니 채점도 됐어야 한다" | 논리 아님 | 채점 vs 위치는 별개 축 (vanish early 97.8 / mid 95.8 / late 57.5 인데 위치는 셋 다 3 슬롯 안 소실) | 09-07 | — | `z_research/v11_roll_out/README.md`; CLAUDE.md §5-5 |
| D36 | TrainingEffects arc · ledge '보정' 값 (이전 판, 04:10) | 인용 금지 (부호 버그) | 자 치우침을 두 배로 더했다 (이전 판 값은 원 문서가 인용 금지로 지정 — 여기에도 옮기지 않는다) → 정정판 release arc −1.2 / −7.7 · ledge −10.7 / −9.7 (px, p@h / p@p). ⚠️ 여기 '보정' 은 자 좌표 치우침 보정이며 인용 금지 α · 보정 계수와 다른 양이다 | 09-25 | 04:32 정정판 값만 | `TrainingEffects/v3_motion/MOTION.md` 정정 (2); `TrainingEffects/v3_motion/TABLES.md`; `VERIFICATION_2026-09-25.md` (5) |
| D37 | IDENTITY_R8 처음 값 "복사보다 가까움 t1–t8 88–98 %, t15 37 %" · "t15 에서 복사 수준으로 떨어진다" · ramp k=2 거리 21/29/59/67 | 정정 (되돌리지 말 것) | t0 22 · t1 49 · t2–t15 82–94 % (wall 1 %); 거리는 정의 통일 (마지막 본 자리 = 샘플 16 − k − 1) 후 24/29/60/71 | 09-26 | — | `IDENTITY_R8_2026-09-26.md` §8 |
| D38 | IDENTITY_R8 · README 표의 t7 칸 | 제외 | 이동 물체 화면 나가는 중 (h '있음' 11–31 %) 인데 p '있음' flat 78–95 % · ramp 40–100 % (진실에서 60–145 px); 학습셋에서도 p 오탐 t7 최고 | 09-26 | 결론에 안 씀 | `IDENTITY_R8_2026-09-26.md` §7-7; `RollOutV3/README.md` |
| D39 | GEOMETRY "p 가 짧고 좁은 것은 (일부) 정규화 방식에서 온다" | 철회 (원인 서술 삭제) | 토큰 LN 대조 없음 (z · h 토큰 LN, p 재 LN 없음) | 09-25 / 26 | — | `RESULTS_2026-09-25.md` §7; `z_research/RollOutV3/figures/v11_geometry/GEOMETRY.md` minor ⑤; r1 §7 |
| D40 | GEOMETRY "t-SNE 섬은 60 칸 / env 로 같은 모양" · "raw 실패 대부분은 평균 이동" · resid PC2 = shape 축 | 정정 | 섬은 env 를 따름; visible/late 가 resid 에서도 0.95 갈림; PC2 는 {sphere, torus} 대 나머지 | 09-26 | — | `GEOMETRY.md` minor ①–③ |
| D41 | 이식 크기 비교 (probe 간) | 금지 | probe 따라 최대 약 48 pt (ridge vs logistic 18–60 pt) | 09-25 / 26 | "같은 조건 안 (93–100 %) 보다 훨씬 낮다" 와 방향만 | `RESULTS_2026-09-25.md` §6; `PROBE.md` §6; `TrainingEffects/v11_identity/IDENTITY.md` |
| D42 | probe head 과대용량 (49.3M) 의 contextF · targetF self 줄을 대조로 | 금지 | `train_acc=1.000` 포화 | 08-13 | 천장에 안 닿은 이식 줄만 읽는다 | `z_world_model_analysis/probe_set_2026-08-13_PREVIOUS-DATA.md`; `occlusion_v2_2026-08-13_PREVIOUS-DATA.md` §7; CLAUDE.md §7-3 |
| D43 | `attn_probe` 와 `attn_probe_xfer` 수치를 섞기 · `overall` 을 self 로 | 금지 | 같은 self 칸이 108 중 1 개에서 6.1 pt 벌어짐 (미수렴 head); `overall` 은 12 조건 평균 (09-02 혼동 사례) | 09-02 / 03 | self = `evals[fit]["per_group"][fit_group]` | `IntPhysGenV11_occlusion_timing_ablation/README.md`; `…/figures/README.md` |
| D44 | `identity_r8/summary.json` 의 `train_set: training_v8` | 정정 (오기) | 실제 `training_r8` | 09-26 | `train_set_note` 추가 | `z_research/RollOutV3/exp_results/README.md` |
| D45 | `V11_READOUT.md` 와 `RESULTS_2026-09-25.md` §5 의 같은 칸 다른 숫자 (last flat t1 0.44 vs 0.24) | 정정 (정의 차) | — | 09-25 | **RESULTS 정의가 정본** | `exp_results/v11/V11_READOUT.md`; `RESULTS_2026-09-25.md` §5 |

⚠️ D 절이 P3 의 모든 위치 주장에 걸린다. 정본은 "검증된 위치 자 보조 + 파라미터 0 의 '후보 미래 중 어디에 가까운가'" 로 Figure 1 을 짠다 (CLAUDE.md §10-1) — 자만으로 쓰면 D4 · D13 이 되풀이된다.
⚠️ D1 의 presence 자로 낸 수치 (RollOutV3 09-24 · 09-25 판, TrainingEffects 의 presence 자 sanity 대조) 는 **방법은 유효, 수치는 지금 결론에 안 씀** 이 원 문서의 판정이다.

### E. encoder · predictor 귀속

| # | 옛 문장 / 도구 | 종류 | 왜 (반증 근거 수치) | 날짜 | 대신 쓸 문장 | 출처 |
|---|---|---|---|---|---|---|
| E1 | "**p 에 정보가 없다**" † | 철회 | v11 등가속+가림에서 p probe shape 98.46 / color 99.49 / env 100 | 08-20 → 08-30 | "맞는 미래로 놓지 않는다" | `z_world_model_analysis/MECHANISM_2026-08-20.md` §6; CLAUDE.md §6 · §5-2; 정본 §7 |
| E2 | "물체를 엉뚱한 자리에 그린다" | 철회 | Mh 가 물체 위치에 안 몰림 (상위 1 % = 1.9 %) | 08-20 | — | `MECHANISM_2026-08-20.md`; CLAUDE.md §6 |
| E3 | predictor 내부가 정보를 잃음 · 출력 부분공간 병목 | 기각 | 층별 probe 68 → 86 % 단조 증가, proj 후 88.7 % | 08-20 | — | CLAUDE.md §6; `MECHANISM_2026-08-20.md` §6 |
| E4 | "세기를 키우면 된다" (균등 증폭 · per-token std · global scalar rescale) | 기각 | 균등 증폭 천장 51–65 %, 잘 맞던 조건까지 부숨 (98.4 → 59.4); visible 정확도를 내린다 | 08-29 · 09-16 | "부족한 건 크기가 아니라 배치다" | `z_research/IntPhysGenV10/Archive/GEOMETRY_2026-08-29.md` §6a; CLAUDE.md §6; `exp_results/ctxenc_subspace_v2/RESULTS.md` |
| E5 | "우리 신호가 작아서" | 기각 | v8 RGB MAE 1.484–2.079 vs Jongseo 1.639 — 오히려 더 큼 | — | — | CLAUDE.md §6 |
| E6 | "IntPhys 물체가 2 배 크다 (6.16 vs 3.04)" | 철회 | 개수와 크기를 섞어 잰 오류; 개별 물체 vanish 4.92 % ≈ IntPhys (shape/color 2.5 % 는 여지) | 08-20 | — | `z_world_model_analysis/WHY_INTPHYS_WORKS_2026-08-20.md` §5; CLAUDE.md §6 |
| E7 | "IntPhys 는 토큰별 신호가 더 크다" | 기각 | 오차/신호 비 IntPhys 1.49 vs 우리 1.15 — 반대; 토큰 배치 어긋남은 IntPhys 에도 같음 | 08-20 | — | `MECHANISM_2026-08-20.md` §7.2 |
| E8 | 배경 복잡도 · 카메라 시점 · 물체 크기 · 렌더러 / 오클루더 | 기각 | env 8 종 51.6–62.5 %, 탑다운 vs 원근 무차이 | 08-20 | — | `RESULTS_2026-08-16.md` §5.2 (가설) → `MECHANISM_2026-08-20.md` §6; `VERIFICATION_2026-08-20.md`; CLAUDE.md §6 |
| E9 | "모델 규모가 결정적 (+24.4 pt)" · "ViT-H 가 항상 낫다" | 정정 | blender v3 L→H +2.5 pt; 자체 데이터는 ViT-L 이 이기는 축 (v8 shape 83.98 vs 73.83, 2D 75.39 vs 71.48, Jongseo 92 → 80) | 08-19 | "ViT-H 는 외부 벤치 (IntPhys 1 88.89 vs 64.44) 에서 먼저 정했다" | `FRAME_ABLATION_2026-08-19.md` §6; CLAUDE.md §1-1 |
| E10 | "cos(Δp, Δh) = 0.085 거의 직교 = 원인" | 정정 (정교화) | 개념 벡터 cos +0.823 / +0.832, 토큰 배치 corr 0.002 (측정 단위가 다름). "blender presence 축 0.834" 는 출처 미확인 | 08-20 | "방향은 정렬, 토큰 배치가 무관" | `RESULTS_2026-08-16.md` §3.4; `MECHANISM_2026-08-20.md` §2.3 · §4 |
| E11 | "표현 공간 이탈 (misalignment) 이 이식 실패의 원인" | 정정 | 원인이 아니라 증상 (α 기반 문서 — 수치 인용 금지) | 08-26 | — | `z_world_model_analysis/PREDICTOR_HEDGING_2026-08-26.md` §4 |
| E12 | GEOMETRY h→p 이식을 p 자기 통계로 재표준화 | 정정 | 차원별 재척도를 공짜로 넣음 (ALL 15.08 → 50.19) | 08-29 | 기본은 h 통계 공유 (`--restd` 는 옛 방식) | `GEOMETRY_2026-08-29.md` §4 |
| E13 | "방향 비대칭은 p 가 그 모양을 못 담아서" | 기각 | 설계 쌍 probe confusion 0 %; probe 가 잘 읽는 쪽이 채점은 0 % (torus probe 100 % vs 채점 3 %) | 08-30 | — | CLAUDE.md §6 · §5-3 |
| E14 | "특정 물체로 몰린다" / "predictor 가 cylinder 를 만든다" / "cylinder 를 잘 예측한다" / "torus 를 못 만든다" | 철회 | 최다 목적지 열 비대각 질량 21–40 % (균등 16.7); cylinder precision 51.5 = chance; torus probe 100 % | 08-31 | "**순서 위쪽으로 흘러내린다**" (과생산 / 과소생산) | `VOTE_GRID_2026-08-31.md` §2-(3); `READING_THE_MATRIX_2026-08-31.md`; `IntPhysGenV11/README.md` §4; CLAUDE.md §5-3 |
| E15 | "채점이 스칼라 읽기로 축퇴했다" (전면) · `s ∝ ‖LN(h)‖` · "s ~ 실루엣 면적" | 기각 | 합 200 쌍 반증 (`cone→sphere`, `cyan→red`); LN 이 노름을 상수로 만들어 항등적 거짓; 실루엣은 설계상 등화 | 08-30 | 편향 (반대칭) 만 1 차원, 탐지력 (대칭) 은 별개 | `HYPOTHESIS_SCALAR_ORDER_2026-08-30.md` §3-1 |
| E16 | "하삼각만 진한 건 정렬 탓" · "colour 도 1 차원 순서" (쌍 단위) | 철회 · 금지 | k=0 정지 · 등속은 21/42 반반; 색 mean\|b\| 는 잡음바닥 언저리 · 정지+가림에 순서 없음 | 08-30 / 31 | 색은 물체 단위로만 | `READING_THE_MATRIX_2026-08-31.md`; `SYNTHESIS_2026-08-30.md`; `VOTE_GRID_2026-08-31.md` |
| E17 | "팔을 가로지르는 이식 실패 = 표현이 조건별이라는 증거" (정지→이동 70.1 vs 이동→정지 17.3) | 철회 | 기대값 — attention query 가 위치 특이적 | 08-30 | — | `RESULTS_2026-08-30.md` §12; `figures/probing/README.md`; CLAUDE.md §7-3 |
| E18 | "`env` 100 % 이식이므로 용량 · 표본 문제 아님" · pooled 대조군 | 철회 (반만 맞음) | 배경은 2048 토큰 전체 | 08-30 | — | `RESULTS_2026-08-30.md` §12 |
| E19 | "z 와 h 가 같다" | 금지 | 천장 18/18 | 08-30 | — | `figures/probing/README.md`; `PAPER_STORY_2026-08-31.md` 금지표 |
| E20 | "인코더는 가림 · 조건과 무관" (shape 포함) | 좁힘 | shape 은 z 도 흔들림 (3 팔 평균 64.8, 등속 팔 42.4) | 08-30 | color · env 로만 주장 | CLAUDE.md §5-2; `SYNTHESIS_2026-08-30.md` |
| E21 | **"encoder 는 충분하고 (릴리즈) predictor 가 병목이다"** (README · PLAN 제목 · TRAINED 제목 §4-1 · DRAFT v1 "encoder is enough") | 철회 (09-25) → 10-08 에 다른 근거로 "정보" 로 한정해 다시 섬 | 충분성 대조 (M13 z_cv) 가 **진실 마지막 위치** 를 썼다; 분업 귀속은 닫히지 않았다 (결정 실험 — 경계 인코딩 길이 L × 릴리즈 · Ariel — 미제출) | 09-25 · 10-08 | "encoder 에는 정보가 있다 (z 속도 R² 0.996 · FUTURE_FROM_CONTEXT z 자 ≈ h 자 · EK100 운동 R² .69 · IntPhys 2 지도 자 73.7), 실패는 predictor 에서 일어난다" — **정보** 로 한정, "충분 · 병목" 단어는 쓰지 않는다 | README_UNTIL 정정 1-1; TRAINED 정정 2 (1); `auto_research/README.md` §2; `VERIFY_0925_R2_2026-09-25.md` X5; `M13_MEMORY_VELOCITY_SUFFICIENCY_2026-09-25.md` (2); `M5_VELOCITY_STEERING_2026-09-25.md` (10); `STORY_EVIDENCE_2026-09-26.md` §2-4; 정본 P3 |
| E22 | M13 "속도가 물체 자리로 적분되지 않는다 / 정보는 있고 진화가 없다" † · M3 1 판 "적분되지 않는다" | 철회 | 절대 위치 R² 는 무효 대조 (진실 칸도 R² .23 / .15) | 09-25 | — | `M13_…` 정정 (1); `PLAN_ENCODER_PREDICTOR_2026-09-25.md`; `VERIFY_0925_2026-09-25.md`; `E_SSV2_…` ("합성 M3 와 같은 모양" 철회) |
| E23 | M13 "p 에 속도 R² ≥ .93 → 상태를 받아 둔다" † · "속도는 p 에 실려 가지 않는다" | 좁힘 | 법칙 정체성 · 물체 비특이 (물체 없는 창에서도 같음) | 09-25 | "속도는 p 토큰에서 읽히지만 물체 비특이" | `M13_…` (3); `PLAN_ENCODER_PREDICTOR_2026-09-25.md` |
| E24 | M13 "pred_last 는 자리에 기여 안 함" · enc_last2 기전 | 철회 | — | 09-25 | — | `M13_…` (5)(6) |
| E25 | "**역사는 쓰이지 않는다**" (일반형) · DRAFT v0 "history is not used" | 철회 | v11 (M2) 은 자리를, v3 (M13 · M6) 는 있음을 역사가 바꾼다 — 문서마다 귀무 축만 봄; 반대 축이라 서로 "역사 불사용" 근거로 쓰지 않는다 | 09-25 | 범위 한정: "v3 자연 조건에서 역사 토큰은 자리를 거의 바꾸지 않고 **있음** 에 기여한다" | `VERIFY_0925_2026-09-25.md` (d); `M13_…` (4); `M2_…` (3); `PLAN_ENCODER_PREDICTOR_2026-09-25.md`; `DRAFT_v0_2026-09-25.md`; README_UNTIL 결합 이야기 4 |
| E26 | M2 "경계만 바꾸면 물체 신호 간극이 양방향으로 통째로 넘어간다 (0.94 / 1.22)" · "Eb_only − E = +0.00 → 마지막 2 튜블릿이면 충분" · "역사 기여 0.1 이하" | 좁힘 · 철회 | 넘어가는 것은 **있음** 뿐 (1.10 / 1.19), 자리 회복 0.15 [0.09, 0.21]; +0.00 은 t2–4 · 있음 · 움직이는 물체 한정; 정지 물체의 있음은 역사에도 기댐 (파괴 0.53–0.71) | 09-25 | — | `M2_…` 정정 (1)(4)(5); README_UNTIL 정정 1-2 |
| E27 | M2 "가림 타이밍만 다른 쌍둥이" · E 짝 420 채점표 | 철회 | sym_k 41 % 다름 · 등가속 짝 가림막 높이 전부 다름 · E_pos 최대 11 회 재사용; E 짝 420 은 다른 block (matched 아님, 버그 B1) | 09-25 | — | `M2_…` (10)(11); `VERIFY_0925_2026-09-25.md` |
| E28 | M2 §5 "상태를 운반하는 world model 과 반대되는 인과 증거" † | 삭제 | 경계 토큰은 문맥 16 장 전체를 본 encoder 출력 — 관측 프레임 이야기가 아니다 | 09-25 | — | `M2_…` 정정 (13) |
| E29 | P3B "**다음 한 걸음은 마지막 토큰 하나가 정한다, 역사는 무관**" · "predictor 는 운반된 역사가 아니라 최신 토큰에서 한 걸음 전이를 읽는다" | 철회 (파일 이름은 철회된 주장) | 역사와 마지막 되먹인 토큰 중 한쪽만 참이어도 격차의 약 78 % 회복 (tf_first 0.79 · tf_last 0.78); 2×2 가 깨끗하지 않음 (마지막 자기 칸 내용 p^z vs p^{pA} 다름) | 09-25 | 누적 vs 토큰 질은 판정 보류 | `auto_research/Archive/P3B_LAST_TOKEN_DRIVES_TRANSITION_2026-09-25.md`; `VERIFY_0925_R2_2026-09-25.md` X3; README_UNTIL 정정 2-8; `auto_research/README.md` §2 |
| E30 | M6 "**역사 토큰이 예측을 과거 쪽으로 끌어 뒤처짐을 만든다**" · "적분은 encoder, 조회가 과거 자리로 끌어 뒤처지게" | 철회 | 이음 점프가 이동과 공선; v3 에서 원래 역사를 지워도 진행 비율 0.547 vs base 0.542 (차 −0.004 [−0.014, +0.005]); 인코딩 방식만으로 t4 −0.56 | 09-25 | — | `M6_HISTORY_SHIFT_2026-09-25.md` 정정 (1)(4)(7); X2; README_UNTIL 정정 2-7 |
| E31 | "encoder 가 경계 토큰에 역사 (운동) 를 요약 · 적분해 넣는다" (M6 · M2b · DRAFT v1 "summarized into boundary tokens") | 강등 (판정 보류 · 가설) | M6 bndM 의 주 효과는 **있음 붕괴** (물체다움 t4 0.63 → 0.23, t8 0.25 → 0.006); 먼 슬롯 자리는 판독 기본값; M2b 는 1 차 | 09-25 | — | `M6_…` (5); `M2B_BOUNDARY_V2_2026-09-25.md`; X1; README_UNTIL 정정 2-6; `DRAFT_v1_2026-09-25.md` |
| E32 | M5 "**속도 부호를 첫 튜블릿에서만 일부 쓰고 누적하지 않는다 / 적분하지 못한다**" · "속도를 한 걸음만 쓴다" | 철회 · 좁힘 | 조향 방향이 법칙 정체와 얽힘 (법칙 안 held-out R² 0.518); 같은 clip 자연 속도 이득은 t1–t5 0.72–1.02 (진실 0.76–0.98) 로 자리에 쌓인다 → 누적 없음은 **이 조향 방향의 성질** | 09-25 | 첫 튜블릿 이동 확정, 속도 사용으로 읽기는 판정 보류 | `M5_VELOCITY_STEERING_2026-09-25.md` 정정 2; X4; README_UNTIL 정정 2-5; `auto_research/README.md` §2 |
| E33 | M4 "미래의 물체는 운반된 상태가 아니라 문맥 속 과거 자리에서 공간적으로 가져온 것 … 둘 방법이 없다" † | 철회 | 고정 반경 없음; 상태 부재는 안 쟀다; "불가능" 주장은 금지 표현 유형 | 09-25 | "2–3 칸 떨어진 진실 칸에 물체 토큰이 나오려면 문맥 물체 토큰에 대한 직접 조회가 필요 (시사)" | `M4_RETRIEVAL_RADIUS_2026-09-25.md`; `VERIFY_0925_2026-09-25.md` |
| E34 | P3 "**멈춤의 원인은 경로가 아니라 p 가 담은 내용**" · "인터페이스가 아니라 p 의 내용 때문" | 철회 | 양방향 참 토큰도 j ≥ 4 멈춤 (형식 교락): 형식 효과 −0.148 [−0.169, −0.125] · 내용 효과 −0.224 (누적 섞임) | 09-25 | — | `P3_CLOSURE_TEST_2026-09-25.md`; README_UNTIL 정정 1-4 |
| E35 | P3 "한 걸음 전이에 필요한 것은 관측 토큰에 있고, predictor 가 만든 토큰에는 없다" † · PLAN §5 "p 에 상태가 없다" † · P2 "그 자리를 채우지 못하는 것은 p 다" † | 철회 (금지 표현 계열) | — | 09-25 | 행동 기술로 ("자기 예측을 이어붙이면 j3 부터 멈춘다") | `P3_…`; `PLAN_ENCODER_PREDICTOR_2026-09-25.md`; `P2_…` §6; `VERIFY_0925_2026-09-25.md`; README_UNTIL 정정 1-8 |
| E36 | H5 "되먹인 상태에 속도가 부족" · "멈춤은 어댑터 탓 아님 → 속도 부족" · "어느 쪽도 상태를 진화시키지 않는다" † | 강등 · 철회 | ar_z 와 모순; 어댑터 3.40 % 는 재구성 오차 | 09-24 | — | `H5_…`; `CROSS_REVIEW_2026-09-24.md` |
| E37 | H5 정정 2 초판 ("길 부분 철회", "멈춤 원인은 p 의 내용", "다시 '공간 전환 비용' 으로 쓰지 말 것") | 철회 | 범위 과대 | 09-25 | "선형 채널 어댑터와 target encoder 출처만 배제" | `H5_…` 정정 2; `P3_…`; README_UNTIL 정정 1-10 |
| E38 | H1 첫 판 "가속은 입력 (z) 에 있다: 전이 실패는 읽기 실패가 아니다 / predictor 미래 계산의 실패" · "인터페이스 손실은 작다 (−0.07)" · "참 상태 대조 60 % 가 상한" | 철회 | clip 단위 split 궤적 누수. 정정: r8 z C8 0.01 · C16 0.51 · C32 0.76; a 열 0.33 / −0.03 / −0.02 (궤적 암기) | 09-24 | RollOutV2 · V3 근거는 "일관되나 교락" | `H1_ACCEL_IN_INPUT_2026-09-24.md`; `CROSS_REVIEW_2026-09-24.md` |
| E39 | H23 "물체는 느리게 연장되다가 뒤층에서 지워진다" · "최종층이 먼 미래 물체를 지운다 (L6 0.88 → L11 0.30)" | 철회 | 층별 자기 문턱 인공물 | 09-24 | — | `H23_…`; `REPORT_h23.md` §3; `CROSS_REVIEW_2026-09-24.md`; README_UNTIL |
| E40 | H23 "어느 층도 먼 슬롯 진실 자리에 물체를 안 만든다" · "진행률 0.5–0.67, 마지막 자리 ≤ 3 %" | 범위 축소 | L0–L4 는 슬롯 9–11 귀무 초과; 슬롯 12 · 13 은 15 % · 35 % | 09-24 | — | `H23_…` |
| E41 | H8 "주된 경로는 초기층 미래↔미래 섞임" · "H4 와 같은 기제" · "위치는 초기층 문맥 조회가 정한다" · "뒤층 문맥 재조회가 외형 · 대비를 만든다" · "존재 소실과 운동 느림은 다른 기제" | 철회 | 명세별 문턱 인공물; 뒤층 차단은 공통 문턱에서 부호 뒤집힘; mc_L7-11 은 L1 +64–65 % 로 목표 공간 이탈 | 09-24 | 남는 것: "뒤층 (L7–11) 미래↔미래를 끊어도 먼 슬롯 물체다움은 돌아오지 않는다" | `H8_KNOCKOUT_PERSISTENCE_2026-09-24.md` "폐기한 해석"; `CROSS_REVIEW_2026-09-24.md`; `H7_V11_LAYERWISE_2026-09-24.md` §4 |
| E42 | H8 의 "CLAUDE.md `mask→ctx` L8 −14.4" 인용 | 철회 (출처 오인) | 출처는 `OVERVIEW_2026-09-22.md` §5-3 `knockout_w1` (v11_full); 비교 불가. README 의 −18.06 은 `knockout_window__v11_vith` n_pair 288 표본 | 09-24 | 두 값을 표본과 같이 병기 | `H8_…` 정정 7; `IntPhysGenV11/figures/knockout/README.md`; `PAPER_STORY_2026-09-06.md` 재현절 |
| E43 | H4 "미래 mask 토큰끼리 상호작용이 먼 미래 물체를 지운다" · "`mask→mask` 차단이 채점 1.4 pt 와 달리 지속에는 효과" | 철회 (인과) | 미래↔미래 attention · key 수 / softmax 질량 · 학습 밖 마스크 모양 셋이 섞임 | 09-24 | 가설: "미래 토큰끼리 보면 공통 미래로 당겨진다" (미측정) | `H4_…`; `CROSS_REVIEW_2026-09-24.md` |
| E44 | "RoPE 라 절대 위치 효과는 구조상 0" | 정정 (틀림, 실측) | `rotate_queries_or_keys` 가 회전쌍 두 성분에 다른 주파수 ("subtle bug", 릴리즈가 그대로 학습) → 위치 인덱스 4 / 8 튜블릿 밀면 p 19 % / 26 % 변함 | 09-24 | — | `PLAN_2026-09-24.md` H4 정정 |
| E45 | H7 "단조 증가, L11 최고 · 중간층 무익" · "평균 풀링 선형 probe 로는 p 특이성 안 나온다" · "가림막 유무로 분포 이동" · "§5-2 인코더도 흔들린다" 인용 | 정정 | dip 건너뛴 층 선택; 틀림; 검정 안 된 원인 → 가설; §5-2 의 z 64.8 / 42.4 는 **이식** 값 (OCC self z 99.9–100) | 09-24 | — | `H7_…` 정정 12 항 |
| E46 | H9 "움직임 · 가림 조건만 오른다 / 상태를 이어 가야 하는 조건에서만" † · "정지는 그대로" · "52 = chance 로 돌아옴" · "섞임이 빈 미래 치우침을 만든다" · "정지는 복사로 설명" · "mm_L0-6 이 early/mid 깎음 = 튜블릿 안 공간 상호작용" | 철회 · 정정 | 정지는 유의 하락; 운동별 평균은 복사와 구분 안 됨; 인과 · 귀속 철회; IntPhys 1 은 오르지 않음 | 09-24 | 채점 개입 시험으로만 | `H9_FROZEN_INTERVENTION_SCORING_2026-09-24.md`; `CROSS_REVIEW_2026-09-24.md` |
| E47 | I_SCENE 매개 "95 % / 99 %" (Ariel 우위가 조회 강도로 설명) | 강등 | 거리 층화 시 팬 4–6 칸 56 % · 6–9 칸 44 %, SSv2 67 / 41 %, 토큰 짝 팬 47 % | 09-25 | "약 절반 (44–67 %), 시사" | `VERIFY_ISCENE_2026-09-25.md`; `I_SCENE_…` 정정 (8); `STORY_EVIDENCE_2026-09-26.md` §2-4 |
| E48 | I_SCENE "방향은 안다" † (진짜 출처를 2–22 배) | 좁힘 | 균등 귀무로 팬 70 → 4.5 배, SSv2 22 → 1.4 배; 토큰별 중앙값 자연 영상 < 1 (SSv2 0.8–1.2, EK100 0.6–0.9) | 09-25 | "일부 토큰만 출처 쪽으로 선택적이다" (attention 선택성 기술) | `I_SCENE_…` 정정 (7); `VERIFY_ISCENE_2026-09-25.md`; `auto_research/figures/story/README.md` fig2 (b) |
| E49 | `rule_full` "모든 칸이 같다" · "정답 없는 개입은 무효" (전부) · "규칙 (prefix) 은 병목 아님" (전부) | 좁힘 | 구간 적중 같고 argmax 99.72 % 일치; v3 anticopy4 +0.085 / temp2 +0.063 (자연 영상에서만 무효); v3 prefix 규칙 −0.06 ~ −0.10 → 팬 · SSv2 · EK100 한정 (시사) | 09-25 | — | `I_SCENE_…` 정정 (2)(4)(5); `VERIFY_ISCENE_2026-09-25.md` |
| E50 | "RoPE 가 거리 감쇠의 원인" · "정체 = 자리" | 판정 보류 · 서술로만 | RoPE 축척 해석 불가; RoPE vs 재조준 미분리 | 09-25 / 26 | — | `I_SCENE_…` §4-1 · §4-6; `VERIFY_ISCENE2_2026-09-26.md` |
| E51 | ctx_ar "움직이는 내용은 AR 과 같거나 +.03" · "문맥 인코딩은 원인 아님" | 틀림 · 강등 | 팬 1–6 칸 −.035 ~ −.074 (CI 0 제외), 정지에서 prefix 보다 낮음; ep16 (학습 중) | 09-29 | "창 단위 z 문맥으로 되돌려도 되먹임 reach · 붕괴 · permanence 는 회복되지 않는다" | `VERIFY_R3_2026-09-29.md` A2 · A6; `AR_PREDICTOR_2026-09-27.md` §2-7 |
| E52 | encoder 운동 판독 "표적 신뢰도의 천장까지 읽는다" · "카메라 몫 74 %" | 정정 | 원 값 .70 / .50 은 train = test; clip-CV 자기 일관성 .53 / .36 을 **넘는다**; 67–75 % | 09-29 | "flow 표적의 자기 일관성보다 높게 읽는다 (문맥 끝 운동을 담는다)" | `ENCODER_MOTION_READOUT_2026-09-29.md`; `VERIFY_R3_2026-09-29.md` B2 · B3 |
| E53 | "encoder 가 미래 자체를 표현한다 / 낙하 시각을 안다" | 금지 | 미래가 현재 상태의 결정적 함수인 데이터 (ledge 낙하 시각 모든 클립 샘플 22 고정, wall 정지 자리 고정) | 10-02 | 변동 시각 데이터가 필요 | `FUTURE_FROM_CONTEXT_2026-10-02.md`; 정본 §7 |
| E54 | IntPhys 2 linear probe 73.7 로 "encoder 가 물리를 이해한다" † · "solidity 는 encoder 도 못 본다" | 금지 | 지도 자 (impossible 라벨을 보고 배움) — VoE 와 같은 능력 아님; solidity 는 transient 라 T×N 평균이 지움 | 09-30 | — | `auto_research/Archive/INTPHYS2_ENCODER_LINEAR_PROBE_2026-09-30.md` |
| E55 | "문맥 encoder 가 속도를 약하게 담아서" | 기각 | z → 문맥 끝 속도 ridge R² 0.996 / 0.989 | 09-19 | — | CLAUDE.md §6; DISTLIM |
| E56 | 경계 토큰 속도가 "경계 토큰 **하나** 에 국소적" (n=30, F1 0.931 / 배경 0.39) · "해상도 아래" | 철회 | n=300 배경 창 0.871; 부분공간 판: 배경에 **다른** 코드 (겹침 0.043); ramp_a 가속 0.83 은 경사각 장면 지름길 | 09-15 / 16 | "배경에도 속도를 맞히는 다른 코드가 있다" | `STATE_EVOLUTION_HYPOTHESES_2026-09-15.md` §3-2; `exp_results/ctxenc_boundary_kinematics/n30/RESULTS.md`; `CLAIM_…` v0.2; `OVERVIEW_2026-09-22.md` §7-2 |
| E57 | 전역 정렬 사상 (Procrustes) "사실상 기각" (토큰 L1 ≤ 9 %) | 정정 | 채점 재계산 late +3 ~ +6 → **부분 효과** (visible +3.0 / late +3.2 / early +6.7 — predictor 재학습 (+25~30) 의 1/5) | 09-16 | "기각이 아니라 부분 효과" | `STATE_EVOLUTION_HYPOTHESES_2026-09-15.md` §3-4; `exp_results/ctxenc_subspace_v2/RESULTS.md`; `CLAIM_…` §2-7; `OVERVIEW_2026-09-22.md` §7-2 |
| E58 | "캐시는 LN 전" | 철회 | `target.npy` · `isolated_ctx` · `ctx_masked` 는 추출 때 이미 affine-free LN (row std 1.0000, 재적용 차 9e-5) | 09-15 | — | `STATE_EVOLUTION_HYPOTHESES_2026-09-15.md`; `time_alignment.md` Notes |
| E59 | 캐시 `ctx_masked` 를 predictor 입력으로 · `--h-from-cache` 를 v11 캐시에 | 기각 (다시 하지 말 것) | LN(z) 가 저장돼 있다 (`forward.py:163`) — 캐시 p 와 상대 L1 0.60; `target` 은 bf16 | 09-19 | predictor 를 다시 돌리려면 문맥 encoder 를 다시 돌린다 | CLAUDE.md §6; `RollOutV2/README.md` §7; DISTLIM; `analysis/attention/knockout/README.md` |
| E60 | "predictor 는 물체 정보를 없애지만 shape · color 는 읽힌다 — 맞다" † | 철회 | 자기모순 + 금지 표현 계열 | 09-25 | — | r1 "뺀 것" 표 (#12) |
| E61 | "물체의 정체성을 잃는다" | 철회 | 잰 것은 자리의 물체다운 토큰; v11 p 에 shape 98.46 / color 99.49 정보 | 09-19 | — | DISTLIM; `PAPER_STORY_2026-09-22.md` 금지표 |
| E62 | "p 가 모든 표현보다 이식이 나쁘다" | 좁힘 | 운동 합침에서만; 모든 운동에서 서는 것은 p < h_fut | 09-25 | — | `audit/v11_identity/AUDIT.md` xfer §7 |
| E63 | E-SSv2 "복사 대비 28 % 이득 중 약 2/3 가 흐림" | 정정 | 61–79 % | 09-25 | — | `E_SSV2_…` 정정 1 |
| E64 | "p ≈ E_q[h\|c]" 예측분포 질량 해석 · H1 "L1 평균이 정체성 방향을 못 센다" | 인용 금지 (α 기반) | α 분해 계열 | 08-20 · 08-30 | — | `VERIFICATION_2026-08-20.md` §1.1; `SYNTHESIS_2026-08-30.md` §8 |
| E65 | "p 가 문맥 잔상인가" (SYNTHESIS · TALK "가장 큰 구멍") | 해소 | p 는 2–3 칸 진전, 정체성 정보는 남음 (09-22 exp1–3 · RollOutV2) | 09-22 | — | `SYNTHESIS_2026-08-30.md` §9; `TALK_2026-08-30.md` §7; `PAPER_STORY_2026-09-22.md` §2-1 |

⚠️ E21 이 이 장에서 가장 조심할 행이다. 09-25 의 철회 사유 (충분성 대조가 진실 위치를 씀) 는 10-08 근거들로 **해소되지 않았다** — 새 근거는 "정보가 있다" 를 세우지만 "predictor 입력에 그 정보가 **쓰기 좋은 형태** 인가" 는 별개다 (`ENCODER_MOTION_READOUT_2026-09-29.md` · `RETHINK_PREDICTOR_ROLE_2026-09-29.md` §1). 그리고 3a (가속) 에서는 encoder 정보 자체가 미결이다 (C38).
⚠️ E29 · E30 · E32 (기전 귀속 셋) 는 `auto_research/README.md` §2 가 한 줄로 묶어 "다시 쓰지 말 것" 으로 지정했다 — 2 차 적대 검증 (`VERIFY_0925_R2_2026-09-25.md`) 의 결론.

### F. 학습 (post-FT · Ariel · AR · AC)

| # | 옛 문장 / 도구 | 종류 | 왜 (반증 근거 수치) | 날짜 | 대신 쓸 문장 | 출처 |
|---|---|---|---|---|---|---|
| F1 | "**학습 도메인 안에서는 진짜 상태 진화처럼 보인다**" † (v11 post-FT, late 91 % · 재출현 일치 · 순서 편향 소실) | 철회 (사용자 지적) | v11 은 속도 1 종 · 물체가 늘 화면 중앙을 지남 → held-out block 도 궤적이 학습셋과 같다 = 학습 분포 (고정 궤적) 재생. RollOut_v2 에서는 v11 속도 (2 배) 강요 · 경사 · 벽 무시 (β_peak flat_v 2.10, ramp_a R² −4.45, wall 69.6 px 통과) | 09-17 | "**배운 것은 v11 궤적 prior** — 도메인 안에서만 오른다" | `z_research/predictor_training/predictor_IntPhysGenV11_PFT/Archive/RESULTS_2026-09-16.md` §3-1; `…/predictor_IntPhysGenV11_PFT/README.md`; `PAPER_STORY_2026-09-22.md`; `OVERVIEW_2026-09-22.md` §5-10 · §7-2; `EXPERIMENTS_2026-09-16.md` §6-1 |
| F2 | "post-FT 가 상태 진화 / intuitive physics 를 배웠다" | 금지 | F1 | 09-22 · 10-08 | "도메인 안 궤적 prior" | `PAPER_STORY_2026-09-22.md` 금지표; 정본 §7 |
| F3 | "v11_postft 는 stride 3 으로 학습해 튜블릿당 3 배로 옮긴다 (과잉 외삽)" | 철회 | b ≤ 1, 고정 앞섬 2–3 칸; 같은 설명을 쓴 TrainingEffects §2-1b 도 철회 대상 | 09-25 | — | TRAINED 정정 2 (5); `VERIFY_0925_R2_2026-09-25.md` X8; README_UNTIL 정정 2-10 |
| F4 | "v11 post-FT 가 문맥 속도를 무시하고 진실의 2 배로 간다" · "v11_e10 은 문맥 속도를 따르지 않는다" | 철회 (틀림) | 기울기 0.86 [0.53, 1.13] + 고정 앞섬 +37 px (절편 +44); own − null 지표 한계 | 09-25 | — | `z_research/TrainingEffects/README.md` 정정; `TrainingEffects/Archive/VERIFICATION_2026-09-25.md` (4); `TrainingEffects/v3_motion/MOTION.md` 정정 (3) |
| F5 | pv1 · v11_postft "물체 유지가 크게 낫다 (물체다움 0.95–0.97)" | 철회 | 물체다움이 위치 무관; 문턱 0.357 이식 불가 | 09-25 | 근거에서 뺀다 | TRAINED (6); X11 |
| F6 | "Ariel 은 자연 영상으로만 학습 → 도메인 암기로 설명 안 됨" · "Ariel 은 도메인 밖" | 철회 | 릴리즈 사전학습 VideoMix22M 에 SSv2 · K400 포함 | 09-25 | "**렌더** 도메인 밖" | TRAINED (11); X13; README_UNTIL 정정 2-10 |
| F7 | "pv1 도 역사 교체에 둔감 → carried state 구조 처방" † | 철회 | hist_swap 은 장면 절단 교란 | 09-25 | — | TRAINED (12) |
| F8 | **Ariel 부분 문맥 팔** — M13 `pred_last*` · `enc_last*`, M6 `hist_none` · `bndM_only` (`chain_0925f.sh`, `block_causal_future_only_1e_5_ep_43`), X9 "마지막 상태만 쓴다" | 무효 (판정 보류 · 인용 안 함) | `kind=prefix` predictor 가 블록 0 에서 시작하지 않는 문맥을 미래로 분류 (`src/models/rollout_predictor.py:178` 개수 셈 vs `build_prefix_mask(block_of, 2)` 절대 블록 번호; dense 폴백) | 09-25 | 문맥은 블록 0 부터 | TRAINED (12); `M6_HISTORY_SHIFT_2026-09-25.md`; `VERIFY_0925_R2_2026-09-25.md` X9; README_UNTIL 읽기 규칙 9 |
| F9 | "가장 잘 진화하는 Ariel 이 VoE 최하 → VoE 는 상태 진화 대리 지표가 아니다" † | 좁힘 (시사) | IntPhys 1 에서만 어긋남 (지평 · fps · 표적 교락) | 09-25 | — | X10; TRAINED (14) |
| F10 | "v11_postft 는 도메인 밖에서 일반화 안 함" | 정정 | 창 의존 | 09-25 | — | TRAINED (13) |
| F11 | "거의 천장, 뒤처짐 거의 없다" | 정정 | 외형 템플릿 .67 / .73; 귀무가 predictor 마다 다름 | 09-25 | — | TRAINED (4); X6 |
| F12 | TRAINED 부록 A 초판 표 전체 | 인용 금지 | 정정 2 (18 항) 전 판 | 09-25 | — | TRAINED |
| F13 | Ariel IntPhys 1 "copy 미만" · "학습 곡선이 첫 checkpoint 안에 끝남" · "ip1 e5 최고 뒤 하락" | 정정 | 표준 표적에서만 copy 미만 (인과 표적은 skip2_w32 한 칸); Ariel 학습 지평 ≤ 8 slot 인데 skip2_w32 는 14 (다섯째 교란); skip2_w16 에서 epoch 마다 더 떨어짐; ±2 pt 진동 | 09-25 | — | `TrainingEffects/ip1_copy/IP1_COPY.md` (1)(2); `VERIFICATION_2026-09-25.md` (11)(12)(15); `TrainingEffects/scores/ANALYSIS.md` |
| F14 | "가속 · 감속을 넣는 predictor 는 없다" | 약화 | — | 09-25 | "넣는다는 증거가 없다" | `VERIFICATION_2026-09-25.md` (6) |
| F15 | "v3 는 어느 predictor 의 도메인도 아니다" · "post-FT 넷 모두 P32 t8–15 지평 밖" | 정정 | pv1 은 v3 의 근접 도메인 (같은 10° 쐐기 · 등속 팔); ip1 (≤ 14) · ip2 (≤ 18) 에는 일부 안 | 09-25 | — | `MOTION.md` 정정 (6)(8); `VERIFICATION_2026-09-25.md` (8) |
| F16 | MOTION 지속 '견고' "pv1_e15, ariel_ep43" · ρ 판정 ("flat_d GT 만: v11 · ip2 · ariel") · "모든 법칙에서 앞으로 간다" · "질량 오르는데 모임이 준다 → 문턱 이동" | 정정 | 두 정의 × 두 자로 ↑ pv1 · ↓ v11; ρ 는 더하기꼴 앞섬 때문에 1 쪽으로 끌림 (기술값); "릴리즈보다 앞서 읽힌다"; v3 에 빈 장면 없어 판정 못 함 | 09-25 | — | `MOTION.md` 정정 (1)(4)(5)(10) |
| F17 | Ariel 지속 판정 "두 자 모두 ↑" | 정정 | 정의에 따라 반대 | 09-25 | 판독 의존이라고 명시 | `VERIFICATION_2026-09-25.md` (7); G03b 묶음 (c) |
| F18 | PRESENCE "제자리에 두는 것은 v11_e10 · pv1_e15" · pv1 학습 곡선 = "척도 · 빈 바닥 이동" · 도메인 밖 "0.1 안팎, release 보다 조금 높음" · "보이는 물체는 아무도 잃지 않는다" | 정정 | 확실히 모으는 것은 v11_e10 하나 (빈 장면 귀무 뺌); "epoch 따라 문맥 무관 출현 prior 를 키운다" 로 뒤집힘; 0.04–0.13 release 수준; 복사도 1.00 이라 증거 아님 | 09-25 | — | `TrainingEffects/v11_presence/PRESENCE.md` 정정 (1)–(9) |
| F19 | IDENTITY "배치는 shape 만 / color 는 어디서도 안 오른다" · block bootstrap "짝 비교는 괜찮다" · "v11 late→late 두 속성 모두 조금 오른다 (굵게)" | 철회 | (궤적, k) 묶음으로 CI 1.5–4 배 → 작은 굵게 여럿 소실; 최적화 잡음 수준 | 09-25 | 방향만 주장 | `TrainingEffects/v11_identity/IDENTITY.md` 정정 |
| F20 | "릴리즈가 v11 복사를 전체로 CI 로 이긴다" | 정정 | 색에서만 (+9.3); 색 빼면 −1.7 [−2.6, −0.7] | 09-25 | — | `VERIFICATION_2026-09-25.md` (1); `scores/ANALYSIS.md` (1) |
| F21 | 영속 배치 "post-FT 가 개선" (Predictor_v1 포함) | 정정 | v11 post-FT 만 깨끗; Predictor_v1 은 상당 부분 (flat ~40 % · ramp ~26 %) '가림막 출구 prior'; '빈→물체' 100 → 73 교환 | 09-25 | — | `VERIFICATION_2026-09-25.md` (2); `TrainingEffects/README.md` |
| F22 | scores/ANALYSIS "L1 로 학습한 예측은 … 수축" | 삭제 | 목적함수를 원인으로 거는 문장 | 09-25 | discussion 한 문단 | `TrainingEffects/scores/ANALYSIS.md` 정정 (6) |
| F23 | `intphys1_postft` "late 그대로" | 틀림 | static late 57.4 → 69.9 (+12.5), moving −2.7 / +0.9 | 09-16 | — | `STATE_EVOLUTION_HYPOTHESES_2026-09-15.md` blockquote |
| F24 | v11 · v11_full **전체** 채점을 학습 후 점수로 | 금지 | 학습 절반과 문맥 공유 (`v11_possible` 은 v11 채점과 문맥이 픽셀 단위로 같다) | 09-10 | `v11_split_test` 로만 (출발선 75.83) | `z_training/README.md`; `z_training/RESULTS_2026-09-19.md`; CLAUDE.md §2-7 |
| F25 | "AR 이 물리를 더 안다" † | 금지 | 블록별 공간 복사만으로 IntPhys 1 83.33 (AR 90.56 = 복사 + 7, 릴리즈 = 복사 + 4) | 09-27 | — | `AR_PREDICTOR_2026-09-27.md` |
| F26 | AR "출처 < 복사 역전이 슬롯 4 부터 · 뭉개짐 · 3 걸음 안에 · AR 도 permanence 같다 · 걸음 크기 설명" | 정정 | 슬롯 **5** 이고 릴리즈도 같음; "뭉개짐" 은 시사 (릴리즈와 같은 정도); 붕괴는 슬롯 1 부터 (−.042); 조금 낮음; 0 칸 최고 · v=2 에서도 짐 → "걸음 크기" 기각 | 09-27 | — | `VERIFY_AR_2026-09-27.md` §9 (1)–(7) |
| F27 | AR40 "AR 자가 더 좋으므로 자 탓이 아니다" | 철회 | 위치는 AR 자가 더 나쁨 (1.16 vs 0.85 칸); AR–릴리즈 오차 차 +5–14 px 중 3–5 px 는 자 몫일 수 있음; 가속 반영 계수 절대값은 자 의존 (교차 head 에서 부호 반전) | 09-30 | — | `AR40_READOUT_2026-09-30.md` 머리; `VERIFY_AR40RD_2026-09-30.md` |
| F28 | AR v11 moving k1–4 '있음' 0–8 % 를 "있다" 로 | 정정 | 빈 장면 오탐 2.5 % 와 구분 안 됨 | 09-30 | "0 과 구분 안 됨"; static k1–2 t5–7 30–41 % 는 실재 | `AR40_READOUT_2026-09-30.md` |
| F29 | `z_training/ariel/context_full_ar_1e_5/latest.pt` = ctx_ar | 정정 | 실은 **AR ep40** 최종 (17:36 판; 19:00 사용자 재배치) | 09-29 | "context_full_ar" 이름 쓰지 않음 | `AR_PREDICTOR_2026-09-27.md` §2-5; `z_training/ARIEL_CHECKPOINTS.md` 1-b |
| F30 | AR +19 pt (vs A · B) 를 predictor 몫으로 · AR margin · 복사를 다른 kind 와 비교 | 금지 | 타깃 공간 (블록별 LN) 이 달라 쌍 정확도만 비교 가능; v11 `copy_blk` 는 사용자 지시로 안 돌림 (09-28) | 09-28 | — | `z_training/ARIEL_CHECKPOINTS.md`; `z_research/TrainingEffects/MODELS.md` |
| F31 | IntPhys 1 p − 복사: Ariel prefix ep43 −16.1 vs AR ep18 +7.2 를 한 비교로 | 금지 (모순 아님) | 다른 predictor · 다른 표적 공간 (표준 vs 블록별) | — | 섞어 읽지 않는다 | TRAINED §3-6; `PREDICTOR_SHARE_2026-09-29.md`; G03 묶음 (c) |
| F32 | Ariel 세 팔 vs 릴리즈를 공정 비교로 · Ariel 을 post-FT 로 | 금지 | scratch · SSv2+K400 · 40–45 ep vs VideoMix22M; attention 규칙 · 데이터 · scratch · epoch 네 가지 동시 차이 | 09-27 | A (full) vs B (prefix) 만 규칙 하나 차 대조 | `z_training/ARIEL_CHECKPOINTS.md`; `TrainingEffects/MODELS.md` |
| F33 | `ariel_ep43` 수치를 ep45 로 옮겨 적기 · `_ep5_full` 성능 보고 | 금지 | ep45 는 다른 파일; `_ep5_full` 보고 금지 | 09-27 | — | `TrainingEffects/MODELS.md`; `configs/protocols/models.md` |
| F34 | Ariel IntPhys 2 Fixed 71.2 | 인용 주의 | Moving 39.7 (우연 아래) 과 짝 · 창 선택 민감 (w16 59.6) · max 집계에서 사라짐 | 09-26 | — | `INTPHYS2_PERMANENCE_AUDIT_2026-09-26.md`; `auto_research/figures/story/README.md` |
| F35 | 팬 post-FT · Ariel "Ariel 수준" · "epoch 포화 → 학습 부족 반론 해소" | 좁힘 | 팬 88 % · SSv2 52 % · EK100 동률; 포화는 EK100 한정 (cosine lr 교락) | 09-26 | — | `VERIFY_ISCENE2_2026-09-26.md` §5-4; `DIRECTION_2026-09-25.md` §4 |
| F36 | "retraining does not repair" (ABSTRACT v1) | 1 차 → 이후 확정 | ep38 재판독 중이었음 | 09-29 | 정본: "다시 학습해도 같다 — 도메인 안에서만 오른다" | `auto_research/paper/ABSTRACT_v1_2026-09-29.md`; `DIRECTION_2026-09-25.md` §4 |
| F37 | 리뷰 시점 "forecasting 학습 predictor 가 더 멀리 간다" ↔ 10-08 "다시 학습해도 같다" | 조건 병기 | 같은 데이터를 다른 축으로 읽음 (reach 는 늘고, 문맥에 없던 운동 변화는 못 넣음) | 10-08 | 두 축을 나눠 쓴다 | `REVIEW_v1_2026-09-25.md`; 정본 P3 |
| F38 | "**action-free 학습으로 해결된다**" | 금지 | Ariel · post-FT 반례 (정본 §2-3 v) | 10-08 | 방법은 Ariel 과의 차이를 대야 한다 | 정본 §7 |
| F39 | "**AC 는 post 여야 효과가 있다**" (사실로) | 금지 | subsection 실험 (scratch vs post × AC 데이터 10/30/100 %) 전까지 가설; 10-10 사용자: AC 스토리는 아직 최종이 아니다 | 10-08 · 10-10 | 가설로만 (`09b_AC_STORY.md` §6) | 정본 §7; `z_AC_training/README.md` |
| F40 | "AC 가 action 을 안 읽는다" (Jongseo RoboCasa) | 인용 불가 | 공식 ckpt 버그 · proprio 불일치 · zero-shot | 09-20 | — | `z_AC_training/README.md`; `z_AC_training/Archive/PLANNING_AND_ROBOCASA_2026-09-20.md` |
| F41 | Jongseo 문서 "DROID 는 camera-frame state 로 학습" · "RoboCasa 는 DINO-WM 학습셋" | 정정 (둘 다 틀림) | `camera_frame` 기본 False = base 좌표; jepa-wms 도 DROID 만 학습 | 09-20 | — | `PLANNING_AND_ROBOCASA_2026-09-20.md` §4 |
| F42 | 09-06 beat 7 "AC 저자들도 pretrained predictor 를 버리고 새로 학습했다 = 암묵적 시인" (`load_predictor: false`) | 대체 | 10-08: AC scratch 레시피를 대조 팔로, 주장은 미검증 가설 | 10-08 | — | `z_AC_training/README.md` §1; `OVERVIEW_2026-09-22.md` §6 |
| F43 | 단계 O (DESIGN_CHOICES 학습 계획) | 무효 | 최적화 실패 — DC1(b) 판정 보류 | — | — | `DESIGN_CHOICES_2026-09-25.md` |
| F44 | AR · ctx_ar 의 `video_batch 8` 수치를 릴리즈 (1) 와 같은 판으로 | 주의 | ±0.5 pt 흔들림 가능 | 09-28 | — | `z_training/ARIEL_CHECKPOINTS.md` |

⚠️ F1 · F38 이 정본 "다시 학습해도 같다" 의 근거이자 한계다 — post-FT 넷과 Ariel 은 **도메인 안에서만** 오른다. 그러나 v11 은 운동 조건마다 궤적 2 개뿐이라 "처음 보는 운동에서도 영속이 좋아진다" 는 어느 쪽으로도 말할 수 없다 (`PRESENCE.md` 단서).
⚠️ F8 은 Ariel 부분 문맥 팔 전부를 막는다. `kind=prefix` 로 부분 문맥 (위치 6–7 만 등) 을 주는 실험은 설계부터 무효다.

### G. 스토리 · 프레임

| # | 옛 문장 / 도구 | 종류 | 왜 | 날짜 | 대신 쓸 문장 | 출처 |
|---|---|---|---|---|---|---|
| G1 | 08-29 "perception model 이지 world model 이 아니다 / 근본 원인은 목적함수 / 물체 영속성도 없다" (12 beat) | 대체 (두 번) | 원인 축을 재학습 없이 못 갚음; 내부 귀속 | 08-31 · 10-08 | 12 beat 의 근거 수치는 유효, 새 beat 로 재배치 | `z_research/IntPhysGenV10/Archive/PAPER_STORY_2026-08-29.md` 머리 |
| G2 | "격차는 목적함수에서 나온다 / L1 최적해 = 조건부 중앙값" (TALK 마무리) · H6 §4c "인과 표적은 p 에만 불리 — p 는 미래를 본 표적에 맞춰 학습" | 강등 | 재학습 없이 검증 안 되는 원인 → 스스로 못 갚는 빚 | 08-31 · 09-22 | **discussion 한 문단** 까지 (Mathieu 2016 조건부 중앙값 수축은 1 순위 대안 설명으로 남음) | `z_research/IntPhysGenV11/Archive/TALK_2026-08-30.md`; `PAPER_STORY_2026-08-31.md`; `H6_…`; `LITERATURE_POSITIONING_2026-09-25.md`; CLAUDE.md §0 |
| G3 | 08-31 "총점이 병목을 가린다" | 대체 | → "IntPhys 1 vs 2 의 문헌 긴장" | 08-31 | — | `PAPER_STORY_2026-08-31.md`; G06 묶음 (c) |
| G4 | 08-31 "정보는 남고 배치가 무너진다, 그 자리는 **frozen 상태에서 되돌릴 수 있다**" — "**frozen 위에서 닫힌다**" (하네스 5 번) | 폐기 | knockout `mask→mask` −1.4 · AC 가 `load_predictor: false`; beat 7 = predictor 재학습. 전역 정렬 사상은 +3 (재학습 +25–30 의 1/5) | 09-06 | 목표 자체가 predictor 학습 | `PAPER_STORY_2026-09-06.md`; `PAPER_STORY_2026-08-31.md` 머리; `IntPhysGenV11/README.md`; `z_training/RESULTS_2026-09-19.md` §3 (옛 프레임이 남은 곳); `z_research/HANDOFF_2026-09-22.md` §1 |
| G5 | 08-31 beat 5 (부분공간 / Procrustes) · beat 6 · 7 (frozen 개입) · `IntPhysGenV11/README.md` §7 "다음에 할 것" 표 | 폐기 | G4 | 09-06 | — | `PAPER_STORY_2026-09-06.md`; `IntPhysGenV11/README.md` §7; CLAUDE.md §10 말미 |
| G6 | 09-06 "**predictor 는 상태를 이어가지 않는다. 문맥을 조회할 뿐**" † (8 beat) | 대체 | 09-06 beat 4(a) 이분법 (진전 / 정지) 이 틀림 — 2–3 칸 진전 후 놓는다; state 프레임 금지 (09-30) | 09-22 · 10-08 | 하네스 · 수치는 유효 | `PAPER_STORY_2026-09-06.md`; `PAPER_STORY_2026-09-22.md`; `OVERVIEW_2026-09-22.md` §6; `HANDOFF_2026-09-22.md` §0 |
| G7 | 09-15 CLAIM "predictor 는 … 상태를 이어가지 않는다" † — 축 R 재귀 · A 지평 · B 가림 · C 동역학 전환 (구조 + 행동 혼합) | 대체 | 09-19 사용자: 행동으로 정의한 **네 능력** (1 읽기 · 2 지속 · 3 전이 · 4 영속). R 재귀는 기제 진단, 정합은 측정 조건으로 빠짐 | 09-19 | 10-08: 같은 네 칸을 intuitive physics 의 조작적 정의로 다시 읽음 | `CLAIM_NO_STATE_EVOLUTION_2026-09-15.md` §1; `STATE_EVOLUTION_CAPABILITIES_2026-09-19.md` §1 · §4 |
| G8 | 09-19 "**ledge · wall (· 꼭대기) 는 common sense, state evolution 정의 밖**" · NARRATIVE_v2 B-2 "그림 예시 한 줄" · 09-25 리뷰 "정의 밖, 복사가 재현 → 부록" | **번복** (사용자 10-08 "이것도 포함해서 논문 쓸거야") | — | 10-08 | 3 전이를 3a 추세 + **3b 사건** 으로; 쓰는 조건 넷 (복사와 닮은 미래 · 갈림 시점 누수 · 자 둘의 다른 답 · L1 교란) | `STATE_EVOLUTION_CAPABILITIES_2026-09-19.md` 머리; `auto_research/paper/NARRATIVE_v2_USER_2026-09-29.md`; `z_research/v11_realistic/README.md` §2; `configs/protocols/datasets.md` (`v11_realistic_ledge`); DISTLIM; `RESULTS_2026-09-25.md` 머리; `MOTION.md`; `audit/v3/AUDIT.md`; `IDENTITY_R8_2026-09-26.md` §4 (본문 미반영); 정본 §2-3 iii |
| G9 | 09-22 한 문장 "마지막으로 본 자리 주변을 다시 읽어 두세 칸을 그릴 뿐 — 학습시켜야 하나 궤적 외우기와 구분되게" · 8 beat · beat 8 평가축 (EK100) | 대체 | 사용자 10-08 스토리 | 10-08 | 정본 §1 한 문장 | `PAPER_STORY_2026-09-22.md`; 정본 |
| G10 | 09-25 뼈대 (reach · DC1–5 · A′) · 09-25 밤 방향 (인간 같은 latent WM, 주 평가 EK100 · IntPhys 2 · Ego4D · EgoExo4D) · 09-29 "**JEPA world model 에서 예측의 자리가 비어 있다**" | 대체 | 10-08 정본 | 10-08 | 틀 · 다음 할 일은 정본 §3 · §5 · §6; **수치 · 정정 · 철회만 유효** | `auto_research/README.md` 머리; `DIRECTION_2026-09-25.md`; `DESIGN_CHOICES_2026-09-25.md`; `ORAL_DIRECTION_2026-09-25.md`; `RETHINK_PREDICTOR_ROLE_2026-09-29.md`; `LITERATURE_PREDICTOR_ROLE_2026-09-29.md` |
| G11 | 옛 README 제목 "새 latent video predictor 의 design choice: pretrained predictor 는 마지막 관측에서 먼 내용을 attention 으로 가져오지 못해 장면을 멈춘다" | 대체 | 10-08 | 10-08 | — | `auto_research/README.md` 머리말 |
| G12 | "a true latent world model must evolve state even when it can't see" → "관측 없이 운반하는 거리가 변위와 무관해야 한다" † | 대체 (state 프레임) | 09-30 memory · state 금지 | 09-30 | 정본 P3 "보이는 것 / 안 보이는 것 / 없는 것" | `auto_research/README.md` §0 |
| G13 | 초안 제목 "…Does Not Carry the World State" (DRAFT v0) · "Exposes the State" (DRAFT v1) · 가설 이름 "frozen V-JEPA 2 는 state 를 진화시키지 않는다" † | 교체 권고 | state 프레임; 교체안 "Seeing Is Not Evolving: … Predictors Trained on the Same Frozen Encoder Go Further" (이것도 10-08 로 대체) | 09-25 | — | `VERIFY_0925_2026-09-25.md`; `VERIFY_0925_R2_2026-09-25.md`; `DRAFT_v0_2026-09-25.md`; `DRAFT_v1_2026-09-25.md`; `CROSS_REVIEW_2026-09-24.md` |
| G14 | **memory · carried state · "state 를 굴린다" · "기억 일관성" · "state 모듈 (관측 없이도 굴러가는 상태 + 렌더러)" · "상태 진화 / 갱신 / 누적 / 닫힌 상태 / 기억 어댑터"** 프레임 | 금지 (사용자 09-30) | 정보는 남아 있고, 기억이 필요 없는 곳 (가림 없는 가속) 에서도 가속을 안 넣는다 → 실패는 규칙 미적용 | 09-30 | "목표가 규칙을 요구하게" 설계; "상태 / 렌더 분리" 는 구조 선택지로만 | `RETHINK_PREDICTOR_ROLE_2026-09-29.md` §0-c; `DIRECTION_2026-09-25.md` §0-b; `LITERATURE_STATE_WORLD_MODELS_2026-09-26.md`; `REVIEW_CVPR_ORAL_2026-09-25.md`; `REVIEW_v1_2026-09-25.md`; `REVIEW_v2_2026-09-26.md`; 정본 §7 |
| G15 | "world knowledge 는 encoder 의 표현 안에 있다" · "predictor 는 world knowledge 를 모른다" · "World knowledge belongs in the predictor" † | 금지 (리뷰어가 잡는다) | 내부 귀속 | 09-29 / 30 | — | `DIRECTION_2026-09-25.md`; `RETHINK_…`; `ABSTRACT_v1_2026-09-29.md`; `RESULTS_2026-09-25.md`; NAR0924 |
| G16 | "synthetic 에서 (pretrained) predictor 가 문제다" | 금지 (사용자 09-25) | 주인공은 새 video predictor, 분석은 design choice 근거 | 09-25 | — | `DESIGN_CHOICES_2026-09-25.md` §4; `DIRECTION_2026-09-25.md` §5; `NARRATIVE_v2_USER_2026-09-29.md` B-6 |
| G17 | "observation-anchored retrieval" | 금지 (조작적 정의 없이) | 문서 곳곳 당시 용어로 남음 | 09-25 | — | `LITERATURE_POSITIONING_2026-09-25.md` |
| G18 | "최초" 문장 · "관측이 끊겨도 상태가 진화해야 한다" 개념 최초 주장 | 금지 | WebSearch 한도 소진 · OpenReview 누락; StEvo-Bench · WRBench 가 개념 선점, Human-JEPA 가 copy-last 귀무 선점 | 09-25 / 29 | 남는 delta: latent predictor · 국소화 · 점수 감사 | `LITERATURE_POSITIONING_2026-09-25.md`; `LITERATURE_PREDICTOR_ROLE_2026-09-29.md` |
| G19 | 철회 논문 2607.16274 ("The JEPA Predictor: A Transferable Operator…") 인용 · 2605.15618 을 predictor 논문으로 | 금지 | 철회 논문 / 제목 오독 | 09-25 / 29 | — | `LITERATURE_PREDICTOR_ROLE_2026-09-29.md`; `LITERATURE_STATE_WORLD_MODELS_2026-09-26.md`; `LITERATURE_POSITIONING_2026-09-25.md` |
| G20 | Garrido 2025 에 "the predictor acts as a world model" 귀속 | 금지 | 원문에 그 문장 없음 | 09-29 | — | `LITERATURE_PREDICTOR_ROLE_2026-09-29.md`; `RETHINK_…` §6-b |
| G21 | GEOPHYS 93.3 % · Human-JEPA 0.873 / 0.798 · HERA 52.57 · AVT Table 8 를 우리 수치와 나란히 | 금지 (원문 확인 전) | split / 프로토콜 미확인; 자기 predictor · K700 20 clip; 본문과 표 불일치 | 09-25 / 29 | — | `LITERATURE_PREDICTOR_ROLE_2026-09-29.md`; `LITERATURE_POSITIONING_2026-09-25.md` |
| G22 | "V-JEPA 2 에 EK100 encoder-only ablation 이 없다" | 정정 | App. D.2 Table 20 에 있다 (39.1 / 20.2 / 39.7) | 09-29 | — | `LITERATURE_PREDICTOR_ROLE_2026-09-29.md` |
| G23 | "leakage explains much" · "appearance template uses no future truth" · "30 fps only" · "non-monotone" | 폐기 · 정정 | 부정확 ("진실 칸 위치를 안 씀" 이 맞음); s2 에서 유지되므로 ✗; → "dissociable" | 09-25 | — | `REVIEW_v1_2026-09-25.md`; `VERIFY_0925_R2_2026-09-25.md` X18; G04b 묶음 (c) |
| G24 | README "영속: 새 증거 없음" 카드 | 정정 권고 | IntPhys 1 재등장 p − 복사 +16.4 (devil) | 09-25 | "미결" | `REVIEW_CVPR_ORAL_2026-09-25.md` |
| G25 | README "확정" ar_p 기울기 음수 · H23 "L1 · L4 먼 슬롯" · H8 "최대 +0.07" · H6 절대 L1 0.577 vs 0.798 (√(2/π)) · v3 holdout=1 확증 | 철회 · 강등 권고 | [−0.030, +0.011] (부호 철회); L0 만; +0.09–0.10; 삭제; 확증 부적격 | 09-25 | — | `REVIEW_CVPR_ORAL_2026-09-25.md` |
| G26 | PAPER_STORY 09-22 beat 4(b) · §2-1 "거리" | 철회 · 보류 권고 | 귀무 미보정 회귀 (C11 ⚠️) | 09-25 | — | `REVIEW_CVPR_ORAL_2026-09-25.md` |
| G27 | DIRECTION §5 "인간 비유는 첫 문장과 discussion 에만" | 바뀜 | 인간 비유가 P1 이 됨 | 10-08 | — | `DIRECTION_2026-09-25.md`; 정본 P1 |
| G28 | "직관물리는 창발하지 않았다" · "물리를 이해하지 못한다" (L1 testbed 전까지) · "정렬 제약이 없다" | 금지 (09-06) | 반박 대상은 측정 / 행동만 쟀다 / 손실이 \|p − LN(h)\| | 09-06 | 정본 §7 문장 | `PAPER_STORY_2026-09-06.md` |
| G29 | SYNTHESIS H2 "predictor 가 문맥 끝에 의존 (recency)" | 흡수 | 09-06 "조회기" 프레임으로 (이후 그 프레임도 대체) | 09-06 | "최근성 효과로 읽힌다" 는 해석 표현 | `SYNTHESIS_2026-08-30.md` §8; `IntPhysGenV11_occlusion_timing_ablation/README.md` |
| G30 | 09-06 beat 6 "학습은 공간에만 구멍, 사용은 시간 뒤쪽 비움, 축이 직교" (마스크 토폴로지 원인) · M4 "사전학습 tube 마스크" 원인 | 강등 | 목적함수 · 학습 마스크를 원인으로 거는 문장 | 09-15 · 09-25 | 인용 · 관찰 · discussion 한 문단 (사용자 09-15) | `OVERVIEW_2026-09-22.md` §6; `z_research/context_encoder_analysis/README.md`; `STATE_EVOLUTION_HYPOTHESES_2026-09-15.md` §4 P5; `M4_…` |
| G31 | beat 8 "frozen encoder + 우리 predictor 로 anticipation + intuitive physics" | 미실행 · 하네스 2 번 불통과 | fallback 필요 | 09-22 | → 정본 목표 (★ 사용자 학습) | `OVERVIEW_2026-09-22.md` §6 |
| G32 | 사용자 결정 4 (planning 동기 오류 · new predictor 범위 · 학회 · 메시지) 가 "미결" | 해소 | 09-29 "새 predictor" · 10-08 정본으로 정리 | 10-08 | — | `LITERATURE_POSITIONING_2026-09-25.md` |
| G33 | "IntPhys 1 은 쓸모없다 / 틀린 벤치다" | 금지 | 잰 것은 leakage 의 크기 | 10-08 | "예측 없이 85 % 가 풀린다" (복사 기준선으로 정의하고) | 정본 §7 |

⚠️ 스토리 판은 08-29 → 08-31 → 09-06 → 09-22 → 09-25 / 09-29 → 10-08 로 여섯 번 바뀌었다 (`00_STORY_AND_TIMELINE.md`). 매번 **틀만 대체되고 측정 수치 · 정정 · 기각표는 유효** 하다 (정본 머리 blockquote). 옛 판 문장을 다시 꺼낼 때는 이 표의 해당 행과 정본 §7 을 같이 본다.

### H. 데이터 · 인프라

| # | 옛 것 / 도구 | 종류 | 왜 | 날짜 | 대신 | 출처 |
|---|---|---|---|---|---|---|
| H1 | `z_scripts/` (구버전 run 스크립트 139 개) | 삭제 (복구 불가) | 옮긴 config 를 가리켜 이미 깨져 있었고 `run.sh` 로 대체; git 추적 0 | 09-21 | `build_intphys1_pairs.py` 만 공식 로직으로 복원 (`z_research/scripts/data/`) | CLAUDE.md §2-0 |
| H2 | `z_research/scripts/HOWTO.md` §1 · `datasets.md` `v11_full` 3 단계 (`build_intphysgen_v6_index.py` · `build_probe_index.py`) | 낡음 (지금 안 돈다) | `z_scripts/` 삭제; v11 계열 index 는 복사가 유일 | 09-21 | 현행 생성기 `z_research/scripts/data/build_intphysgen_index.py` (10-06) | `z_research/scripts/HOWTO.md`; `z_research/HANDOFF_2026-09-22.md` §2; `configs/protocols/datasets.md` |
| H3 | HOWTO §5 sbatch 예시 (`WMA_RUN=1` 없음) | 낡음 (위험) | 자기제출 폭주 전 판; 문서에 "대체됨" 표시 없음 | — | `--export=ALL,WMA_RUN=1,…` | `HOWTO.md`; CLAUDE.md §2-5 |
| H4 | sbatch 자기제출 | 사고 | `SLURM_BATCH_SCRIPT` 가 batch job 에서 비어 모든 job 이 제출 분기를 다시 탐 — 세대마다 ×7, **9,924 개** | — | `WMA_RUN=1` 로 가르고, `SLURM_JOB_ID` + `OUTDIR` 동시면 거부하는 tripwire | CLAUDE.md §7-1 |
| H5 | `--export` 에 YAML 리스트 | 함정 | 콤마 구분자라 `'[capsule'` 로 잘려 도착 | — | base64 `SET_B64` | CLAUDE.md §2-5 · §7-1 |
| H6 | `mp.Process(...).start()` 후 join 안 함 | 함정 (exit 0 on failure) | rank 가 죽어도 0 반환 → `afterok` 풀림 | — | `evals/main.py` 가 exitcode 확인 | CLAUDE.md §7-1 |
| H7 | `evals.main` 직접 호출 | 금지 | DDP bind 실패를 삼키고 `world_size=1` 로 폴백 → 두 job 섞임 (split-brain); NCCL 600 s timeout | 08-16 | `run.sh` (포트 자동 탐색 · `WMA_EXPECT_WS` · `EVAL_DDP_TIMEOUT_S` 7200) | CLAUDE.md §2-0 · §7-1; `z_world_model_analysis/RESULTS_2026-08-16.md` |
| H8 | `probing.runs` 를 줄여 prep | 함정 (과소추출) | 캐시에 뽑히는 표현도 줄어든다 | — | runs 는 두고 `num_epochs` 만 1 | CLAUDE.md §7-1 |
| H9 | 토큰 캐시 재사용 | 함정 | `TokenCache.matches()` 는 video_ids · 토큰수 · dtype 만 본다 (체크포인트 · 해상도 · autocast · 재렌더는 서명 밖); 상위 · 부분집합 재사용 불가 | 08-16 | 재렌더 시 `cache/_previous_YYYYMMDD/<tag>` 로 치움 | CLAUDE.md §7-1; `RESULTS_2026-08-16.md`; `occlusion_v2_2026-08-13_PREVIOUS-DATA.md` |
| H10 | `features:` 블록 누락 · `data.resolution != model.img_size` · CPU 스레드 과다구독 | 함정 | 채점 끝난 뒤 `cache_dir` KeyError / 크면 에러 없이 엉뚱한 구간 / decode 39 → 268 ms | — | `resolve.py` 실물 검사 | CLAUDE.md §7-1 |
| H11 | `model.predictor.kind` 를 안 줌 | 함정 | state_dict 키가 릴리즈와 같아 조용히 full attention (출력 상대차 0.5) | 09-23 | 로더가 `predictor_checkpoint` 의 `arch.kind` 를 읽고 다르면 죽는다 | CLAUDE.md §3-1; `configs/protocols/models.md` |
| H12 | `sed -i` 로 `datasets.md` 한 줄 수정 | 사고 | 섹션 4 개 통째 삭제 | 08-28 | 여러 줄 md / yaml 은 Python 으로 | CLAUDE.md §9 |
| H13 | `z_research/IntPhys-Like_data/` 리포트 · figure | 소실 (복구 불가) | git 추적 0 | 08-28 | 새 스크립트 · config · 문서는 추적되는 자리에 | CLAUDE.md §8-1 |
| H14 | 2D transit 프레임 (195M) + 생성 코드 | 소실 | 토큰 캐시 (41G) 만 남음 | 08-28 | 캐시 기반 사후 분석만 | CLAUDE.md §4-3 |
| H15 | RollOut_v1 · 학습셋 v1–v4 | 삭제 | 레지스트리에서도 뺐다 | — | 기록은 각 세트 문서 | CLAUDE.md §2-2 |
| H16 | v6–v10 · v13 · 2D · RollOut_v1 프레임 | 삭제 | `v10*` · `v13_black` 은 `resolve` 실물 검사에서 죽는다 (v13 세트 폴더 없음, 결과는 `datasets.md` 와 산출물에만) | — | — | `HANDOFF_2026-09-22.md` §3-1; `configs/protocols/datasets.md`; CLAUDE.md §6 |
| H17 | v1 데이터셋 → `z_world_model_analysis/` 의 `extract.py` · `spatial.py` · `metrics.py` · `layers.py` · `mech.py` | 깨짐 | `intphysgen_v1_vith.yaml` 을 찾는데 v1 데이터 자체가 없음; v1 결과 복구 불가 | — | — | CLAUDE.md §2-6; `z_research/scripts/_superseded/README.md` |
| H18 | Ariel ep5 · 19 · 30 · 41 · 43 체크포인트 (`vith_ariel_bc_ep5/19/30/41/43/ep5_full`, `block_causal_future_only_1e_5_ep_*`) | 삭제 (gitignore, 복구 불가) | `_resolved.yaml` 해독용 기록으로만 (`available: false`, 실행하면 죽음); I_SCENE §5-3 epoch 포화 재실행 불가 — 원자료는 `auto_research/_stage/results_vll3/scene_*_arielep/` 만 | 09-27 | ep45 (`block_causal_future_only_1e_5/latest.pt`, ep43 과 다른 파일) · full ep40 · AR | `z_training/ARIEL_CHECKPOINTS.md`; `configs/protocols/models.md`; `I_SCENE_…` §5-3; `TrainingEffects/MODELS.md` |
| H19 | `ar_future_1e_5/latest.pt` 덮어쓰기 (ep18 → 09-28 17:22 ep36 → 20:19 ep38) | 함정 | `ar_ep18` · `ar_ep36` 은 같은 파일을 가리킨다 (어느 epoch 인지 미검증) | 09-28 | ep38 만 고정 사본 | `configs/protocols/models.md`; `cvpr/README.md` §7 |
| H20 | EK100 run 들 `ek100_rel_third10_grid8_ar40` · `ek100_vith_official256_released_grid8_ar40` | 인용 금지 | 키 누락으로 릴리즈가 돈 run (A40) | 10-01 | 긴 run 은 smoke 에서 "무엇이 로드됐는지" assert | `AR_PREDICTOR_2026-09-27.md` §2-9 |
| H21 | v11 토큰 캐시 `v11_full_vith` · `v11_vanish_all_ctx32_vith` · `rollout_v2_training_v6/v8` 특징 캐시 | 삭제 | — | 09 | — | `PREDICTOR_FUTURE_SUMMARY_2026-09-24.md`; `configs/protocols/datasets.md` |
| H22 | `rollout_v2` 09-09 판 | 재생성 (09-10) | wall 오류 | 09-10 | 09-10 판 | `configs/protocols/datasets.md` |
| H23 | `gravity_realistic` "metadata `object_px_*_by_sample` 이 불가능 셋에서 똑같다" · `dataset.json` "g 101" | 정정 (10-07 18:07) | metadata 재조립 (프레임은 그대로라 채점 유효); 실제 g 170 (옛 판) / 120 (새 판); pixel check FAIL (옛 판 6.5 / 13.7 px, 새 판 6.3 / 6.4 / 16.0 px) | 10-07 | — | `configs/protocols/datasets.md`; `z_research/v11_realistic/README.md` |
| H24 | gravity 옛 판 (미래 넷, 꼭대기 apex float 93.8) | 대체 | 미래 다섯 판 apex float 33.3 · arc 62.5 — 데이터 (g 170 → 120, 꼭대기 f57/f42 → f66/f30) 가 같이 바뀌어 원인 분리 불가 | 10-07 | **두 판을 섞어 읽지 않는다** | `z_research/v11_realistic/README.md` §4 · §4-옛판; `exp_results/_previous_20261007_4futures/gravity_retrieval.md` |
| H25 | gravity "오르는 중이면 높이를 유지한다 — 위로 가는 속도를 잇지 않는다" | 정정 (10-08) | 튜블릿 분해: 앞 4 튜블릿은 arc (물체 토큰 92–100 %), 꼭대기 뒤 내려오지 않음, t7 stop | 10-08 | "위로 가는 속도는 잇는다 · 꼭대기에서 돌아 내려오지 않는다" | `v11_realistic/README.md` §4 |
| H26 | gravity 꼭대기 arc 62.5 = "꼭대기에서 내려온다고 예측" | 정정 | 공 자리는 98 % block 이 float — arc 가 이기는 건 두 미래 픽셀이 같은 칸 (나머지 85 %) 때문 | 10-08 | 공 자리 근거로는 안 선다 | `z_research/v11_realistic/exp_results/gravity_retrieval.md` §7; CLAUDE.md §5-6 |
| H27 | 배경 Δ 부호를 조건 무관하게 | 정정 | vanish · ledge 는 배경 = 물체 부호 96–100 % 인데 gravity 오르는 중 · 꼭대기는 반대 | 10-08 | 조건 차이로 읽는다 | `v11_realistic/README.md` §4; `v11_realistic/figures/token_surprise/README.md` |
| H28 | `fig_rl_occlusion` · `fig_rl_direction` · `fig_rl_ledge` | 교체 | 보기 어려움 (A/B 분해는 옛 그림에만) | 10 | `fig_rl_summary` | `z_research/v11_realistic/figures/_superseded/README.md` |
| H29 | `configs/analysis/probing/*_5fold_*.yaml` (intphys1 probing) | 레포에 없다 | config 는 산출물 `summary.json` 안에만 | 09-21 | — | `analysis/intphys1/HANDOFF_SPEC.md`; `PAPER_VS_OURS.md`; `PIPELINE_FILE_MAP.md` |
| H30 | `z_research/IntPhysGenV11/Archive/ATTENTION_2026-09-01.md` | 레포에 없다 | — | 09-21 | 해석은 산출물에서 다시 읽는다 | `analysis/attention/README.md` |
| H31 | `z_research/IntPhysGenV8/exp_results/` | 비어 있음 (4.0 K) | v8 79.10 / 73.44 출처는 `z_exp/world_model_analysis/results/` | 09-22 | — | `OVERVIEW_2026-09-22.md` §9-2 ③ |
| H32 | `_superseded/gif_C16_P32_ex{0,1}.gif` "이전 판" | 파일이 디스크에 없음 | — | 09-25 | — | r1 §4 |
| H33 | Benchmarks README §4 표 · HANDOFF "GRASP · InfLevel 미완" | 낡음 | InfLevel 은 3 모델 × 3 property 전부 나옴; GRASP 만 진짜 미실행 (`_resolved.yaml` 하나뿐) | 09-22 | — | `OVERVIEW_2026-09-22.md` §9-2 ①②; `HANDOFF_2026-09-22.md` §1-1 · §7 |
| H34 | `configs/protocols/README.md` 하단 "현재 프로토콜 4 개" | 문서 내부 불일치 | 상단 표 · OVERVIEW §2-2 는 5 종 (`attn_probe_xfer` 추가) | — | — | `configs/protocols/README.md` |
| H35 | `auto_research/scripts/README.md` 목록 | 미갱신 | 09-26 이후 스크립트 (`e_scene_ar.py` · `copy_hard_subset.py` · `e_v11_presence.py` · `intphys2_permanence_audit.py` 등) 빠짐 | — | 각 문서 `## 재현` | `auto_research/scripts/README.md` |
| H36 | OVERVIEW §9-2 ⑤ "`rollout2_presence_readout.py` 가 scripts README 에 줄 없음" | 해소 | 현재 판에는 줄이 있다 | — | — | `z_research/scripts/README.md` |
| H37 | EK100 "데이터는 vll6 에만" · "PDF 없다" · config 생성기 | 정정 · 제거 | vll5 `/data2` 에도 700 개; PDF 있음 | 09 | — | `z_research/anticipation/EK100/README.md` §3 · §8 |
| H38 | EK100 head 20 (sweep) | 정정 | 4090 24 GB 에 안 들어감 (head 1 = 82.1M) → head 8, head 마다 fwd→bwd (업데이트 차 0.0) | 09 | — | `EK100/README.md` |
| H39 | `data.py` mp4 경로 | 버그 수정 | `frames_start` / `frames_stride` 를 무시 | 09-21 | — | `z_research/Benchmarks/PROTOCOLS.md` |
| H40 | IntPhys 2 SceneIndex `"154_0"` 형 15 장면 (30 쌍) | 버그 수정 | `int("154_0")=1540` onset 파싱 — 특징은 동일, meta `onset_frame` 만 −1 | 09-30 | — | `INTPHYS2_ENCODER_LINEAR_PROBE_2026-09-30.md` |
| H41 | VERIFY_0925 코드 버그 B1–B35 (예: B1 M2 E_pos/E_imp 420 쌍 다른 block → matched 위반 · B5 역사 제거 경계 RoPE 위치 어긋남 · B25 화면 밖 문맥 튜블릿 78.6 %) | 기록 · 수정 | — | 09-25 | — | `VERIFY_0925_2026-09-25.md` |
| H42 | 런처만 죽여 rank 8 개가 고아로 GPU 점유 · `CUDA_VISIBLE_DEVICES = SLURM_LOCALID` 트릭 | 사고 · 넣지 않음 | 후자는 모든 rank 를 GPU 0 으로 몬다 | 09-10 | `pkill -f "launch.py --fname .*<run>"` | `z_training/README.md` |
| H43 | `filter_short_videos` 를 통과율 낮은 데이터에 | 함정 | 무작위 재추첨 → DDP 전체 멈춤 (Ariel 09-22: skipping 1,368 회 / step 0 회) | 09-22 | `train_min<N>.csv` 로 미리 거름 | `configs/training/datasets.md` |
| H44 | `SET` 에 YAML flow `{a:b}` | 함정 | `{'a:b': None}` 으로 읽힘 | — | JSON 식 (공백 없이) | `configs/training/README.md` |
| H45 | bf16 probing 캐시로 surprise raw 정확도 | 정정 | 근접 tie 가 뒤집힘 (static_occlusion 63.37 vs 참조 58.16; 5 조건이 0.5 pt 밖) | 09-16 | Δ 만 믿고, 정식 수치는 fp32 재추출 후 | `exp_results/ctxenc_subspace_v2/RESULTS.md` §1; `CLAIM_…` §2-7; `PAPER_STORY_2026-09-22.md` |
| H46 | bf16 predictor 출력의 batch 구성 의존 | 함정 | batch 1 vs 4 상대 L1 0.049 | — | 재현성은 `video_batch: 1` | `RollOutV2/README.md` §7; `OVERVIEW_2026-09-22.md` §8; CLAUDE.md §7-2 |
| H47 | 정밀도가 실행마다 다름 (IntPhys fp32 + autocast fp16 · occlusion/probe_set bf16; `occlusion_v2.yaml` 이 실행 후 바뀜) | 주의 | 그대로 재실행하면 수치가 움직일 수 있음 | 08-16 | surprise fp32+fp16 · probing bf16 관례를 섞지 않는다 | `RESULTS_2026-08-16.md`; CLAUDE.md §1-7 |
| H48 | occlusion_v2 · probe_set 옛 데이터 (08-13) | superseded | 재렌더 (f 교란 · 문맥 통제 수정; `divergence_frame` / `leak_maxdiff` 추가) | 08-16 | video 단위 비교는 가능 (video_id 동일) | `occlusion_v2_2026-08-13_PREVIOUS-DATA.md`; `probe_set_2026-08-13_PREVIOUS-DATA.md` |
| H49 | IntPhys1 dev 가림 판정을 "첫 가림-재등장" 기준으로 | 정정 | 90 개 중 41 개만 살고 문맥이 갈라진 쌍이 절반 넘음 | 08-13 | magic tick 기준 · `status.json` `masks` | `z_world_model_analysis/intphys1_dev_2026-08-13.md` |
| H50 | `/local_datasets/world/dev/task.txt` 의 이름 (`NN_test_visible_static_nobj1`) 을 dev 디렉토리에 대응 | 금지 | 익명화 — 90 개 중 84 개 불일치 | — | `dev_metadata/dev/{blk}/{quad}/{run}/status.json` 에서 직접 | `z_world_model_analysis/fig/intphys1_examples/README.md` |
| H51 | v11_color · v11_shape pft RESULTS 머리 제목 "v11 vanish pos_a" | 오기 (제목 복사) | n 336 / 84 로 보아 color · shape 블록 pos_a | — | — | `z_research/predictor_training/predictor_IntPhysGenV11_PFT/exp_results/{pft,release}/*/RESULTS.md` |
| H52 | `token_subset` (config 주석에만 있던 08-13 판) | 구현 · 기본 `all` | surprise 경로는 토큰을 캐시하지 않아 부분집합 채점은 재실행 필요 | 08-19 | — | `TOKEN_SUBSET_2026-08-19.md`; `occlusion_v2_2026-08-13_PREVIOUS-DATA.md` §7 |

⚠️ H18 · H19 · H20 은 **학습 비교의 재현성** 을 직접 깎는다: Ariel epoch 곡선 (ep5–43) 은 원자료로만 남았고, AR ep18 / ep36 은 파일이 같으며, 09-30 EK100 AR 수치는 릴리즈였다. 학습된 predictor 수치를 인용할 때는 체크포인트 파일 · 날짜를 같이 적는다.

---

## 3. 금지 표현

### 3-1. 정본 §7 (10-08 갱신) 그대로 + 원 문서에서 발견된 위반 예

`위반 예` 는 원 문서에 실제로 쓰였던 문장이다 (지금은 각 문서가 철회 · 정정 표시를 달았거나, 추출문이 `(원 문서 표현 — 현재 금지)` 를 붙였다). 이 장의 다른 표에서는 `†` 로 표시했다.

| # | 쓰지 말 것 | 왜 · 대신 (정본 §7) | 원 문서에서 발견된 위반 예 | 출처 |
|---|---|---|---|---|
| X1 | "world model 은 물리를 이해하지 못한다" · "intuitive physics 가 없다" (조건 없이) | 행동만 쟀다 → "**마지막 관측에 없는 X 를 따르는 미래를 예측하지 못한다**" / "배웠다고 보기 어렵다" (근거와 함께) | 08-29 "물체 영속성도 없다"; VERIFICATION "모델이 우리 위반을 '이상하다' 고 사실상 전혀 보지 않는다"; NARRATIVE_v2 B-6 이 막은 "world knowledge 가 없다" | `PAPER_STORY_2026-08-29.md`; `VERIFICATION_2026-08-20.md`; `NARRATIVE_v2_USER_2026-09-29.md` B-6 |
| X2 | "직관물리는 창발하지 않았다" | 반박 대상은 그 측정 → "창발 근거인 IntPhys 1 은 leakage 가 크다 (예측 없이 85 %)" | (09-06 금지표에서 먼저 막음 — 위반 문장은 추출문에서 못 찾음) | `PAPER_STORY_2026-09-06.md` |
| X3 | "IntPhys 1 은 쓸모없다 / 틀린 벤치다" | 잰 것은 leakage 의 크기 → "예측 없이 85 % 가 풀린다" (복사 기준선으로 정의하고) | (위반 문장 못 찾음) | 정본 §7 |
| X4 | "IntPhys 2 에서 chance" | V-JEPA 2-h 57.51 (논문) → "복사와 구분되지 않는다" | 09-06 이전 "IntPhys 2 chance 근처"; IntPhys 2 감사 원문 "복사가 −5 pt 앞선다" (A26) | `PAPER_STORY_2026-09-06.md`; `INTPHYS2_PERMANENCE_AUDIT_2026-09-26.md` |
| X5 | "모델이 중력을 안다 / 모른다 / 물리 법칙을 적용한다" | 내부 귀속 → "낙하하는 미래를 예측하지 않는다" | r1 "중력을 잘 모른다"; AR40 사용자 요청 문구 "중력을 이해했는지"; "AR 이 물리를 더 안다"; "DINO-F 가 ViT-H 보다 물리를 더 안다" · "DINO-F predictor 가 가속을 안다"; "encoder 가 물리를 이해한다" (IntPhys 2 linear probe); I_SCENE "방향은 안다 / 모른다" (attention 선택성 기술) | r1; `AR40_READOUT_2026-09-30.md`; `AR_PREDICTOR_2026-09-27.md`; `DINOF_V11_2026-10-07.md`; `INTPHYS2_ENCODER_LINEAR_PROBE_2026-09-30.md`; `I_SCENE_…`; `VERIFY_ISCENE_2026-09-25.md` |
| X6 | "predictor 에 정보가 없다" / "X 를 못 만든다" | p probe 98–100 → "맞는 미래로 놓지 않는다" | MECHANISM 08-20 "p 에 정보가 없다"; PLAN "p 에 상태가 없다"; P3 "predictor 가 만든 토큰에는 없다"; P2 "그 자리를 채우지 못하는 것은 p 다"; M4 "둘 방법이 없다"; READING_THE_MATRIX "torus 를 못 만든다"; r1 "predictor 는 물체 정보를 없앤다" | `MECHANISM_2026-08-20.md`; `PLAN_ENCODER_PREDICTOR_2026-09-25.md`; `P3_…`; `P2_…`; `M4_…`; `READING_THE_MATRIX_2026-08-31.md`; r1 |
| X7 | "encoder 가 미래를 표현한다" | 미래가 현재의 결정적 함수인 데이터다 (`FUTURE_FROM_CONTEXT` §3) | FUTURE_FROM_CONTEXT 초안 "낙하 시각을 안다" (문서가 스스로 금지); ENCODER_MOTION_READOUT 는 "담는다 (문맥 끝 운동)" 까지만 | `FUTURE_FROM_CONTEXT_2026-10-02.md`; `ENCODER_MOTION_READOUT_2026-09-29.md` |
| X8 | "기억 (memory) 문제다" · "state 를 잃는다" | 09-30 사용자. 정보는 남아 있고, 기억이 필요 없는 곳에서도 가속을 안 넣는다 | 파일 이름 `M13_MEMORY_VELOCITY_SUFFICIENCY` · `m13_memory_state_v3.py` (이름일 뿐); CLAIM "기억 길이 — k=1 실패 = 기억 ≈ 0"; H5 "어느 쪽도 상태를 진화시키지 않는다"; CROSS_REVIEW 가설 이름 "state 를 진화시키지 않는다"; AR_PREDICTOR "state 를 굴리는 모듈"; RESULTS_2026-09-25 "상태가 흐려진다 · 세계 상태"; PREDICTOR_FUTURE_SUMMARY "물체를 놓았거나 encoder 공간을 벗어난 형태로 들고 있다"; v11_roll_out "상태를 적분해 나른다"; LITERATURE_STATE "기억 일관성"; REVIEW_CVPR_ORAL "기억 어댑터"; DRAFT 제목 "Does Not Carry the World State" | `M13_…`; `auto_research/scripts/README.md`; `CLAIM_NO_STATE_EVOLUTION_2026-09-15.md` §3; `H5_…`; `CROSS_REVIEW_2026-09-24.md`; `AR_PREDICTOR_2026-09-27.md` §0; `RESULTS_2026-09-25.md`; `PREDICTOR_FUTURE_SUMMARY_2026-09-24.md`; `z_research/v11_roll_out/README.md`; `LITERATURE_STATE_WORLD_MODELS_2026-09-26.md`; `DRAFT_v0_2026-09-25.md` |
| X9 | "context 에서 안 보이면 못 한다" (전부로) | 문맥 가운데 가림은 처리한다 (early 97.8 / mid 95.8) → "**마지막 관측에** 안 보이면" | (위반 문장은 추출문에서 못 찾음 — B25) | 정본 §7; CLAUDE.md §5-1 |
| X10 | "action-free 학습으로 해결된다" | Ariel · post-FT 반례 (§2-3 v). 방법은 차이를 대야 한다 | 리뷰 시점 "forecasting 학습 predictor 가 더 멀리 간다" 를 해결로 읽는 문장 (F37) | 정본 §7; `REVIEW_v1_2026-09-25.md` |
| X11 | "AC 는 post 여야 효과가 있다" (사실로) | subsection 실험 (§2-5) 결과 전까지 가설 (10-10: AC 스토리는 아직 최종 아님, `09b`) | 09-06 beat 7 "저자들도 predictor 를 버렸다 = 암묵적 시인" (F42) | 정본 §7; `z_AC_training/README.md` |
| X12 | "ledge 에서 불가능한 연속을 예측한다" | 떠 가는 미래는 복사와 닮았다 (H23 §3-7) → "낙하하는 미래를 만들지 않는다" (다중 미래 설계로) | H23 검정 A 원판 "ledge 쌍 74–100 % 가 떠 가는 미래에 더 가깝다" 를 능력 주장으로 (C30) | `H23_…`; `REPORT_h23.md` |
| X13 | "post-FT 가 intuitive physics 를 배웠다" | 도메인 안 궤적 prior 다 | RESULTS_2026-09-16 "학습 도메인 안에서는 진짜 상태 진화처럼 보인다" (F1) | `predictor_IntPhysGenV11_PFT/Archive/RESULTS_2026-09-16.md` |
| X14 | "가림이 표현을 망가뜨린다" | env 72 칸 100 | READING_THE_MATRIX 오해표의 원 문장 | `READING_THE_MATRIX_2026-08-31.md` |
| X15 | "목적함수 때문이다" | discussion 한 문단까지 | 08-29 "근본 원인은 목적함수"; TALK "L1 최적해 = 조건부 중앙값"; H6 §4c "p 는 미래를 본 표적에 맞춰 학습"; scores/ANALYSIS "L1 로 학습한 예측은 수축"; M4 "사전학습 tube 마스크"; 09-06 beat 6 마스크 토폴로지 | `PAPER_STORY_2026-08-29.md`; `TALK_2026-08-30.md`; `H6_…`; `TrainingEffects/scores/ANALYSIS.md`; `M4_…`; `OVERVIEW_2026-09-22.md` §6 |
| X16 | "p 가 물체를 마지막 자리에 둔다" · "슬롯 N 이후 멈춘다" | 철회 — 자의 기본값 | POSREAD 원판 "마지막 관측 자리에 둔 채 머문다"; LITERATURE_POSITIONING "predictor 가 물체를 마지막 관측 자리에 둔다"; 제안서 "후반에 정지"; DISTLIM 원판 "슬롯 4 이후 멈춘다" | POSREAD; `LITERATURE_POSITIONING_2026-09-25.md`; `V-JEPA 2 predictor 분석 ~하면 되지 않을까 에 답하는 실험 제안.md`; DISTLIM |
| X17 | **α · 보정 · 헤지 계수** | **인용 금지** (사용자 지시 전까지) | `z_world_model_analysis/PREDICTOR_HEDGING_*` · `MECHANISM_*` · `VERIFICATION_2026-08-20.md` §1.1 · `GEOMETRY_2026-08-29.md` 의 α · 보정 값, `alpha_amplify.py`, `ctxenc_subspace_v2` 보정 계수, v10 α — 값은 이 아카이브 어디에도 옮기지 않는다 (α 부호 규약도 문서마다 반대) | CLAUDE.md §0 · §5-4; 정본 §7 |

### 3-2. 옛 금지표에서 이어지는 것 (정본 §7 에 따로 없는 행)

| # | 쓰지 말 것 | 왜 · 대신 | 처음 막은 곳 |
|---|---|---|---|
| Y1 | "표현이 회전했다 / 이동했다" | 09-22 측정했고 아니었다 → "저차원 부분집합", "직교 사상 하나로 이식이 돌아온다" 뜻으로만 | `PAPER_STORY_2026-08-31.md`; `SYNTHESIS_2026-08-30.md`; `PAPER_STORY_2026-09-22.md` |
| Y2 | "z 와 h 가 같다" | 천장 18/18 | `PAPER_STORY_2026-08-31.md`; `figures/probing/README.md` |
| Y3 | "팔 가로지르는 이식 실패가 증거" | 기대값 | `RESULTS_2026-08-30.md` §12 |
| Y4 | "colour 도 1 차원 순서" (쌍 단위) | 잡음바닥 — 물체 단위만 | `PAPER_STORY_2026-08-31.md`; `VOTE_GRID_2026-08-31.md` |
| Y5 | "가림 하에서 가속 무관" | 바닥 효과 | `RESULTS_2026-08-30.md` §3-(2) |
| Y6 | "위치가 진전하지 않는다" | 2–3 칸 진전 후 놓는다 | `PAPER_STORY_2026-09-22.md` |
| Y7 | "정체성을 잃는다" | 잰 것은 자리의 물체다운 토큰 | `PAPER_STORY_2026-09-22.md`; DISTLIM |
| Y8 | "p 는 힘을 반영하지 않는다" (전부로) | arc 상승 −27 | `PAPER_STORY_2026-09-22.md`; DISTLIM |
| Y9 | "EK100 에서 못 한다" | 우연의 42 배 | `PAPER_STORY_2026-09-22.md` |
| Y10 | "정렬 제약이 없다" | 손실이 \|p − LN(h)\| | `PAPER_STORY_2026-09-06.md` |
| Y11 | "synthetic 에서 predictor 가 문제다" | 주인공은 새 predictor | `DESIGN_CHOICES_2026-09-25.md` §4 |
| Y12 | "encoder 는 충분하고 predictor 가 병목" (단어 그대로) | E21 — 정본은 "정보" 로 한정 | `STORY_EVIDENCE_2026-09-26.md` §2-4 |
| Y13 | "reach 는 fps · 걸음 수와 무관한 고정 거리" | C6 | `DESIGN_CHOICES_2026-09-25.md` §4 |
| Y14 | "학습 지평 (8 슬롯) 에서 멈춘다" | C2 | `STORY_EVIDENCE_2026-09-26.md` §2-4 |
| Y15 | "observation-anchored retrieval" · "최초" | G17 · G18 | `LITERATURE_POSITIONING_2026-09-25.md` |
| Y16 | "world knowledge" (그대로) | 리뷰어가 잡는다 | `DIRECTION_2026-09-25.md` §0-b |
| Y17 | "predictor 가 쓸모없다" | A30 — "이 평가가 예측력을 요구하지 않는다" | `PREDICTOR_SHARE_2026-09-29.md`; `SC3_…` |
| Y18 | "knockout 간선이 X 를 나른다 / 정한다" · "개입이 예측을 낫게 한다" · "섞임이 X 를 지운다 / 만든다" | knockout 은 분포 밖 개입 | `H8_…` §5; `H9_…` §5 |
| Y19 | "그 층이 안다 / 모른다" | 렌즈는 반사실 (cos L0 0.32 · L6 0.70) | `H7_…` §5; `H23_…` |

---

## 4. CLAUDE.md §6 기각표 ↔ 이 장의 행

CLAUDE.md §6 의 각 행이 이 장 어디에 근거와 함께 들어갔는지 (CLAUDE.md 는 운영 규칙이라 수치가 짧다).

| CLAUDE.md §6 행 | 이 장 |
|---|---|
| context 길이 · 희석 · 채점 프로토콜 · 배경 / 카메라 · predictor 내부 정보 손실 | C1 · A16 · A11 · E8 · E3 |
| "p 에 정보가 없다" · "엉뚱한 자리" · "IntPhys 물체 2 배" · "신호가 작아서" | E1 · E2 · E6 · E5 |
| 궤적 · 속도 위반 추가 · 방향 비대칭 = p 가 못 담아서 · 세기를 키우면 · block 밖 retrieval · Luce | A34 · E13 · E4 · A15 · A13 |
| 가림막 색면 · "가림막 있는데 안 가리는" 조건 | B3 · B4 |
| "마지막 관측 자리에 둔 채 머문다" · "wall 벽 속 한 칸 반" | D4 · C26 |
| v11 k=0 자 학습 · 위치 자 학습셋 v3 | D8 · D9 |
| "문맥 끝 속도를 그대로 이어 간다" / "슬롯 4 이후 멈춘다" · "encoder 가 속도를 약하게" | C14 · C13 · E55 |
| 캐시 `ctx_masked` 를 predictor 입력으로 · 슬롯 7 진행 비율 하나로 요약 | E59 · D7 |

---

## 출처

레포 루트 기준. 이 장의 행이 인용한 원 문서 전부 (추출문 G01–G13 경유).

**운영 규칙 · 정본**
- `CLAUDE.md` (§0 · §1-1 · §1-2 · §1-3 · §1-7 · §2-0 · §2-2 · §2-5 · §2-6 · §2-7 · §3-1 · §4-3 · §5 · §6 · §7 · §8 · §9 · §10)
- `auto_research/paper/PAPER_STORY_2026-10-08.md` (§1 · §2-3 · §2-4 · §4 · §7)
- `auto_research/README.md` (머리말 · §0 · §2)
- `auto_research/Archive/README_UNTIL_2026-09-25.md` (정정 1 · 2 표 · 결합 이야기 · 읽기 규칙 9)

**auto_research — 09-24 H 시리즈 · 적대 검증**
- `auto_research/Archive/PLAN_2026-09-24.md` · `H1_ACCEL_IN_INPUT_2026-09-24.md` · `H23_PREDICTOR_LAYERS_TRAJECTORY_2026-09-24.md` · `H4_QUERY_AND_POSITION_2026-09-24.md` · `H5_AUTOREGRESSIVE_FEEDBACK_2026-09-24.md` · `H6_INTPHYS1_WHAT_THE_SCORE_MEASURES_2026-09-24.md` · `H7_V11_LAYERWISE_2026-09-24.md` · `H8_KNOCKOUT_PERSISTENCE_2026-09-24.md` · `H9_FROZEN_INTERVENTION_SCORING_2026-09-24.md`
- `auto_research/Archive/CROSS_REVIEW_2026-09-24.md` · `VERIFY_0925_2026-09-25.md` · `VERIFY_0925_R2_2026-09-25.md`
- `auto_research/exp_results/h23/REPORT_h23.md` · `auto_research/exp_results/h23b/REPORT_h23b.md` · `auto_research/exp_results/h7/h7b_scoring_tables.md`

**auto_research — 09-25 ~ 10-07 (M · P · SC · I_SCENE · AR · copy)**
- `auto_research/Archive/PLAN_ENCODER_PREDICTOR_2026-09-25.md` · `M13_MEMORY_VELOCITY_SUFFICIENCY_2026-09-25.md` · `M2_BOUNDARY_PATCH_2026-09-25.md` · `M2B_BOUNDARY_V2_2026-09-25.md` · `M4_RETRIEVAL_RADIUS_2026-09-25.md` · `M5_VELOCITY_STEERING_2026-09-25.md` · `M6_HISTORY_SHIFT_2026-09-25.md`
- `auto_research/Archive/P2_STRIDE_SWEEP_HORIZON_UNIT_2026-09-25.md` · `P3_CLOSURE_TEST_2026-09-25.md` · `P3B_LAST_TOKEN_DRIVES_TRANSITION_2026-09-25.md` · `E_SSV2_ENCODER_READS_PREDICTOR_STALLS_2026-09-25.md`
- `auto_research/Archive/SC1_REACH_NOT_HORIZON_2026-09-25.md` · `SC2V2_FLOW_REACH_SSV2_2026-09-25.md` · `SC3_EK100_PREDICTOR_SUBSTITUTION_2026-09-25.md` · `SC5_EK100_HORIZON_SWEEP_2026-09-26.md`
- `auto_research/Archive/I_SCENE_CAUSAL_BOTTLENECK_2026-09-25.md` · `VERIFY_ISCENE_2026-09-25.md` · `VERIFY_ISCENE2_2026-09-26.md`
- `auto_research/Archive/TRAINED_PREDICTORS_SIGNATURES_2026-09-25.md` · `INTPHYS2_PERMANENCE_AUDIT_2026-09-26.md` · `AR_PREDICTOR_2026-09-27.md` · `VERIFY_AR_2026-09-27.md` · `COPY_HARD_SUBSET_2026-09-29.md` · `ENCODER_MOTION_READOUT_2026-09-29.md` · `PREDICTOR_SHARE_2026-09-29.md` · `VERIFY_PRESENCE_2026-09-29.md` · `VERIFY_R3_2026-09-29.md`
- `auto_research/Archive/AR40_READOUT_2026-09-30.md` · `VERIFY_AR40RD_2026-09-30.md` · `INTPHYS2_ENCODER_LINEAR_PROBE_2026-09-30.md` · `INTPHYS2_ONSET_2026-09-30.md` · `DINOF_INTPHYS1_2026-10-02.md` · `FUTURE_FROM_CONTEXT_2026-10-02.md` · `COPY_VS_POSITION_2026-10-07.md` · `DINOF_V11_2026-10-07.md`
- `auto_research/Archive/LITERATURE_POSITIONING_2026-09-25.md` · `LITERATURE_STATE_WORLD_MODELS_2026-09-26.md` · `LITERATURE_PREDICTOR_ROLE_2026-09-29.md` · `REVIEW_CVPR_ORAL_2026-09-25.md`
- `auto_research/figures/story/README.md` · `auto_research/scripts/README.md`

**auto_research/paper — 옛 스토리 · 초안 · 리뷰**
- `auto_research/paper/NARRATIVE_v2_USER_2026-09-29.md` · `RETHINK_PREDICTOR_ROLE_2026-09-29.md` · `DIRECTION_2026-09-25.md` · `ORAL_DIRECTION_2026-09-25.md` · `DESIGN_CHOICES_2026-09-25.md` · `STORY_EVIDENCE_2026-09-26.md` · `ABSTRACT_v0_2026-09-25.md` · `ABSTRACT_v1_2026-09-29.md` · `DRAFT_v0_2026-09-25.md` · `DRAFT_v1_2026-09-25.md` · `DRAFT_v2_2026-09-25.md` · `REVIEW_v1_2026-09-25.md` · `REVIEW_v2_2026-09-26.md`

**z_research — v11 본체 · timing · v10**
- `z_research/IntPhysGenV11/README.md` · `Archive/PAPER_STORY_2026-08-31.md` · `Archive/PAPER_STORY_2026-09-06.md` · `Archive/PAPER_STORY_2026-09-22.md` · `Archive/STATUS_AND_CRITIQUE_2026-09-03.md` · `Archive/SYNTHESIS_2026-08-30.md` · `Archive/TALK_2026-08-30.md`
- `z_research/IntPhysGenV11/Archive/surprising_score/RESULTS_2026-08-30.md` · `HYPOTHESIS_SCALAR_ORDER_2026-08-30.md` · `READING_THE_MATRIX_2026-08-31.md` · `VOTE_GRID_2026-08-31.md`
- `z_research/IntPhysGenV11/figures/{attention,knockout,probing,surprise}/README.md`
- `z_research/IntPhysGenV11_occlusion_timing_ablation/README.md` · `z_research/IntPhysGenV11_occlusion_timing_ablation/figures/README.md`
- `z_research/IntPhysGenV10/Archive/V10_2026-08-28.md` · `GEOMETRY_2026-08-29.md` · `PAPER_STORY_2026-08-29.md`

**z_world_model_analysis (08-13 ~ 08-26)**
- `z_world_model_analysis/intphys1_dev_2026-08-13.md` · `occlusion_v2_2026-08-13_PREVIOUS-DATA.md` · `probe_set_2026-08-13_PREVIOUS-DATA.md` · `RESULTS_2026-08-16.md` · `FRAME_ABLATION_2026-08-19.md` · `TOKEN_SUBSET_2026-08-19.md` · `MECHANISM_2026-08-20.md` · `OCCLUSION_LENGTH_2026-08-20.md` · `VERIFICATION_2026-08-20.md` · `WHY_INTPHYS_WORKS_2026-08-20.md` · `PREDICTOR_HEDGING_2026-08-26.md` · `fig/intphys1_examples/README.md`

**z_research — RollOutV2 · v11_roll_out**
- `z_research/RollOutV2/README.md` · `Archive/PREDICTOR_DISTANCE_LIMIT_2026-09-19.md` · `figures/v5/summary/POSITION_READOUT_2026-09-12.md` · `figures/v5/v11_vanish/timing/{timing_summary,token_test,attn_diag}.md`
- `z_research/v11_roll_out/README.md`

**z_research — RollOutV3 (현행 · 감사 · 물러난 판)**
- `z_research/RollOutV3/README.md` · `Archive/IDENTITY_R8_2026-09-26.md` · `Archive/CONTEXT_TO_FUTURE_2026-09-26.md` · `Archive/RESULTS_2026-09-25.md` · `Archive/READOUT_CHOICE_2026-09-23.md`
- `z_research/RollOutV3/exp_results/README.md` · `exp_results/presence/RESULTS.md` · `exp_results/v11/V11_READOUT.md` · `exp_results/windows/{BEHAVIOR,CROSS_HEAD,WINDOW_SUMMARY}.md` · `exp_results/_superseded/windows_v6_20260923/{CROSS_HEAD,WINDOW_SUMMARY}.md`
- `z_research/RollOutV3/figures/README.md` · `figures/v11_probe/PROBE.md` · `figures/v11_geometry/GEOMETRY.md` · `figures/_superseded/v11_dh_gif/README.md` · `figures/_superseded/presence_era_2026-09-24/README.md` · `figures/_superseded/presence_era_2026-09-24/direction/DIRECTION_R2.md` · `figures/v11_presence/_superseded/binary_majority_2026-09-27/README.md` · `figures/context_to_future/_superseded/first_draft_2026-09-26/README.md` · `figures/{v11_presence,v3_signed_error,v3_signed_error_hhead,v3_trajectory}/_superseded/{presence_decoder,identity_aeda55874c1c}/README.md`
- `z_research/RollOutV3/_superseded/README_2026-09-25.md` · `Archive/_superseded/NEW_ARCHIVE_README_2026-09-24.md` · `Archive/_superseded/NEW_ARCHIVE_README_2026-09-25_r1.md` · `Archive/_superseded/PREDICTOR_FUTURE_SUMMARY_2026-09-24.md` · `Archive/_superseded/RECALL_TUBELET_LIMIT_2026-09-24.md`
- `z_research/RollOutV3/audit/v11_identity/AUDIT.md` · `audit/v11_presence/AUDIT.md` · `audit/v3/AUDIT.md` · `audit/training_v8/AUDIT.md` · `audit/training_sets/RollOut_v2_training_v8.md`

**z_research — context_encoder_analysis**
- `z_research/context_encoder_analysis/README.md` · `Archive/CLAIM_NO_STATE_EVOLUTION_2026-09-15.md` · `Archive/STATE_EVOLUTION_HYPOTHESES_2026-09-15.md` · `Archive/EXPERIMENTS_2026-09-16.md` · `Archive/STATE_EVOLUTION_CAPABILITIES_2026-09-19.md` · `Archive/ENCODER_TEMPORAL_DYNAMICS_2026-09-20.md`
- `z_research/context_encoder_analysis/exp_results/{ctxenc_boundary_kinematics/n30,ctxenc_boundary_identity_v2,ctxenc_subspace_v2,ctxenc_time_alignment_gated}/RESULTS.md` · `exp_results/time_alignment/time_alignment.md`

**z_research — TrainingEffects · predictor_training · z_training**
- `z_research/TrainingEffects/README.md` · `MODELS.md` · `Archive/VERIFICATION_2026-09-25.md` · `scores/ANALYSIS.md` · `ip1_copy/IP1_COPY.md` · `v11_identity/IDENTITY.md` · `v11_presence/PRESENCE.md` · `v3_motion/MOTION.md` · `v3_motion/TABLES.md`
- `z_research/predictor_training/predictor_IntPhysGenV11_PFT/README.md` · `Archive/RESULTS_2026-09-16.md` · `exp_results/{pft,release}/*/RESULTS.md`
- `z_training/README.md` · `z_training/RESULTS_2026-09-19.md` · `z_training/ARIEL_CHECKPOINTS.md` · `z_training/runs/_matrix/MATRIX.md`
- `configs/training/README.md` · `configs/training/datasets.md`

**z_research — Benchmarks · IntPhys · OcclusionStats · EK100 · v11_realistic**
- `z_research/Benchmarks/README.md` · `PROTOCOLS.md` · `Archive/INTPHYS1_EVENT_POSITION_2026-09-25.md` · `exp_results/_invalid_20260921_mixedgrid/README.md`
- `analysis/intphys2/README.md` · `ANALYSIS_NOTES.md` · `PROTOCOL_CHECK_2026-09-14.md` · `analysis/intphys2/configs/_superseded/README.md`
- `analysis/intphys1/HANDOFF_SPEC.md` · `PAPER_VS_OURS.md` · `PIPELINE_FILE_MAP.md`
- `analysis/attention/README.md` · `analysis/attention/knockout/README.md`
- `z_research/OcclusionStats/Archive/INTPHYS1_OCCLUSION_2026-09-26.md`
- `z_research/anticipation/EK100/README.md`
- `z_research/v11_realistic/README.md` · `exp_results/gravity_retrieval.md` · `exp_results/_previous_20261007_4futures/gravity_retrieval.md` · `figures/_superseded/README.md` · `figures/token_surprise/README.md`

**AC · 개관 · 인프라 · 레지스트리**
- `z_AC_training/README.md` · `z_AC_training/Archive/PLANNING_AND_ROBOCASA_2026-09-20.md`
- `z_research/OVERVIEW_2026-09-22.md` (§5 · §6 · §7-2 · §8 · §9-2) · `z_research/HANDOFF_2026-09-22.md`
- `z_research/scripts/README.md` · `z_research/scripts/HOWTO.md` · `z_research/scripts/_superseded/README.md`
- `configs/protocols/README.md` · `configs/protocols/datasets.md` · `configs/protocols/models.md` · `cvpr/README.md`
- `V-JEPA 2 predictor 분석 ~하면 되지 않을까 에 답하는 실험 제안.md` (레포 루트)
