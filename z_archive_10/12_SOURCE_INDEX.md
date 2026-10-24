# 12 — 원 문서 색인

> 레포의 결과 · 분석 md **264 개** 전부 (`z_research` · `auto_research` · `z_training` · `z_AC_training` · `z_world_model_analysis` · `analysis` · `configs` · `evals` · 루트 제안서). 추출문에 항목이 있는 것 264 개, 없는 것 0 개 (아래 '추출 안 됨').
> 열: **줄** = 원 문서 줄 수 · **날짜** · **상태** (원 문서 머리 기준: 정본 · 대체됨 → 정본 · superseded · 기록 · 인프라) · **장** = 이 아카이브에서 그 문서를 주로 쓰는 장 (00 스토리 · 01 P1 · 02 P2 · 03 testbed·측정 · 04 보이는 것 · 05 안 보이는 것 · 06 없는 것 · 07 encoder·predictor · 08 재학습 · 09 목표 · 09b AC).
> 상태 · 날짜는 추출 에이전트가 원 문서 머리에서 읽은 값이다. 바꾸기 전에 원 문서를 볼 것.
> ⚠️ **경로 이동 (2026-10-10)** — 아래 `z_research/<셋>` 의 세트 15 개 (anticipation · Benchmarks · context_encoder_analysis · IntPhys · IntPhysGenV8 · V10 · V11 · V11_occlusion_timing_ablation · OcclusionStats · predictor_training · RollOutV2 · RollOutV3 · TrainingEffects · v11_realistic · v11_roll_out) 와 `HANDOFF/OVERVIEW_2026-09-22.md` 는 지금 **`z_research/previous/<셋>`** 에 있다. 이 표의 경로는 10-10 당시 것이다 — `previous/` 를 끼워 읽는다. 본문 · 줄 수는 안 바뀌었다. 10-10 이후 추가된 92 개는 이 파일 **끝** 의 표.

## (루트)  (1 개)

| 문서 | 줄 | 날짜 | 상태 | 장 |
|---|---:|---|---|---|
| `V-JEPA 2 predictor 분석 ~하면 되지 않을까 에 답하는 실험 제안.md` | 150 | 2026-09-24 | 기록 (제안; 실행 여부 문서에 없음) | 03 · 04 · 06 · 08 · 09 |

## analysis  (10 개)

### analysis/attention

| 문서 | 줄 | 날짜 | 상태 | 장 |
|---|---:|---|---|---|
| `README.md` | 45 | 불명 | 인프라·방법 | 07 |
| `knockout/README.md` | 244 | 2026-09-03 | 인프라·방법 | 03 · 07 |
| `predictor_attn/README.md` | 202 | 불명 | 인프라·방법 | 03 · 07 |

### analysis/intphys1

| 문서 | 줄 | 날짜 | 상태 | 장 |
|---|---:|---|---|---|
| `HANDOFF_SPEC.md` | 373 | 2026-09-21 | 기록 — 아카이브된 파이프라인 명세, 영어 | 03 · 07 |
| `PAPER_VS_OURS.md` | 183 | 불명 | 기록 | 00 · 07 |
| `PIPELINE_FILE_MAP.md` | 218 | 불명 | 인프라·방법 (디버그 지도) | 03 |

### analysis/intphys2

| 문서 | 줄 | 날짜 | 상태 | 장 |
|---|---:|---|---|---|
| `ANALYSIS_NOTES.md` | 318 | 2026-07-12 | 기록 — 머리에 "정본은 README, 이 문서는 분석 메모", 영어 | 00 · 03 · 07 |
| `PROTOCOL_CHECK_2026-09-14.md` | 73 | 2026-09-14 | **대체됨 → `analysis/intphys2/README.md`** (2026-09-21) | 00 · 03 |
| `README.md` | 107 | 2026-09-21 | 정본 — IntPhys 2 프로토콜 시작점 | 02 · 03 |
| `configs/_superseded/README.md` | 12 | 2026-09-21 | superseded (물러난 config 사유) | 00 · 03 |

## auto_research  (72 개)

### auto_research/Archive

| 문서 | 줄 | 날짜 | 상태 | 장 |
|---|---:|---|---|---|
| `AR40_READOUT_2026-09-30.md` | 68 | 2026-09-30 | 기록 (1 차) — 본문 안 "적대 검증 정정" blockquote 로 일부 철회·축소, 검증 정본 → `auto_research/Archive/VERIFY_… | 03 · 05 · 06 · 08 · 09 |
| `AR_PREDICTOR_2026-09-27.md` | 251 | 2026-09-27 | 기록 (1 차, §2-1~2-3 은 `VERIFY_AR` 로 확정/정정, §2-4 ep38 로 "확정", §2-6 presence 는 `VERIFY_PRESE…) | 02 · 04 · 05 · 08 · 09 · 09b |
| `COPY_HARD_SUBSET_2026-09-29.md` | 165 | 2026-09-29 | 기록 (1 차; 머리에 ⚠️ 적대 검증 3 차 정정 blockquote — 수치 유지 · 읽기 문장 교체) | 02 · 05 · 08 · 09 |
| `COPY_VS_POSITION_2026-10-07.md` | 33 | 2026-10-07 | 정본 (짧은 답변 문서) | 02 · 03 · 06 |
| `CROSS_REVIEW_2026-09-24.md` | 272 | 2026-09-24 | 기록 (적대적 검증 워크플로 — 문서별 검증 7 + 교차 검토 1 — 의 교차 검토 원문; 이 검토로 각 문서 머리 ⚠️ 정정 블록이 생겼다. 파일은 고치지 않음) | 00 · 02 · 03 · 04 · 06 · 07 |
| `DINOF_INTPHYS1_2026-10-02.md` | 88 | 2026-10-02 | 정본 (DINO-Foresight × IntPhys 1) | 02 · 03 |
| `DINOF_V11_2026-10-07.md` | 45 | 2026-10-07 | 정본 (DINO-Foresight × v11) | 02 · 05 |
| `ENCODER_MOTION_READOUT_2026-09-29.md` | 48 | 2026-09-29 | 기록 (1 차; 본문에 ⚠️ `VERIFY_R3` B 정정 blockquote) | 02 · 07 |
| `E_SSV2_ENCODER_READS_PREDICTOR_STALLS_2026-09-25.md` | 496 | 2026-09-25 | 정본 (정정 1 + 정정 2 반영; 1 차 제목·파일 이름의 "predictor stalls / 시간 따라 진전 안 함" 은 철회됨) | 02 · 03 · 04 · 07 |
| `FUTURE_FROM_CONTEXT_2026-10-02.md` | 76 | 2026-10-01 | 정본 (본문 안 10-02 정정 포함) | 03 · 05 · 06 · 07 |
| `H1_ACCEL_IN_INPUT_2026-09-24.md` | 273 | 2026-09-24 | 정본 (정정판; 머리에 ⚠️ 정정 14 항 — 첫 판 결론 철회) | 03 · 06 · 07 |
| `H23_PREDICTOR_LAYERS_TRAJECTORY_2026-09-24.md` | 338 | 2026-09-24 | 정본 (정정판; 머리 ⚠️ 정정 12 항, 원문 §7 기록) | 02 · 03 · 04 · 06 · 07 |
| `H4_QUERY_AND_POSITION_2026-09-24.md` | 317 | 2026-09-24 | 정본 (정정판; 머리 ⚠️ 정정 11 항, §0-원문 "인용 금지") | 03 · 04 · 07 |
| `H5_AUTOREGRESSIVE_FEEDBACK_2026-09-24.md` | 281 | 2026-09-24 | 정본 (개정판; ⚠️ 정정 12 항 + 정정 2 → 일부는 `P3_CLOSURE_TEST_2026-09-25.md` §4 가 정본) | 04 · 07 · 08 · 09 · 09b |
| `H6_INTPHYS1_WHAT_THE_SCORE_MEASURES_2026-09-24.md` | 429 | 2026-09-24 | 정본 (정정판; 머리 ⚠️ 정정 17 항, §0-기록 = 첫 판 22:05 원문) | 02 · 03 · 05 |
| `H7_V11_LAYERWISE_2026-09-24.md` | 285 | 2026-09-24 | 기록 (본문 머리에 ⚠️ 정정 12 항, 원판 제목 "정체성은 predictor 전 층에 있고, 채점은 최종층이 가장 낫다" 를 정정해 제목 교체) | 03 · 05 · 06 · 07 |
| `H8_KNOCKOUT_PERSISTENCE_2026-09-24.md` | 264 | 2026-09-24 | 기록 (본문 머리 ⚠️ 정정 10 항 — 원판 결론 다수 **철회**, 원 제목 "무엇이 먼 미래의 물체를 흐리게 하나: 초기층의 미래↔미래 섞임") | 03 · 04 · 07 |
| `H9_FROZEN_INTERVENTION_SCORING_2026-09-24.md` | 288 | 2026-09-24 | 기록 (본문 머리 ⚠️ 정정 13 항, 초판 결론 1·2·3·5 대체; 초판 결론은 blockquote 로 남김) | 02 · 03 · 05 · 07 |
| `INTPHYS2_ENCODER_LINEAR_PROBE_2026-09-30.md` | 48 | 2026-09-30 | 기록 (단일 문서, 대체 표시 없음) | 02 · 03 · 07 |
| `INTPHYS2_ONSET_2026-09-30.md` | 55 | 2026-09-30 | 정본 (IntPhys 2 위반 시점 추정) | 02 · 03 |
| `INTPHYS2_PERMANENCE_AUDIT_2026-09-26.md` | 97 | 2026-09-26 | 기록 (1 차 · 적대 검증 전; 본문에 ⚠️ 2 차 정정 반영) — 결론은 `README` §0 근거 5 · CLAUDE.md §5-6 이 인용 | 02 · 05 |
| `I_SCENE_CAUSAL_BOTTLENECK_2026-09-25.md` | 460 | 2026-09-25 | 기록 (1 차 결과 + 적대 검증 1 차 정정표 + 절별 ⚠️ 2 차 검증 표시; §4-3 · §4-4 · §4-5 는 현재 정본이 인용) | 02 · 03 · 04 · 05 · 07 · 08 · 09 |
| `LITERATURE_POSITIONING_2026-09-25.md` | 1525 | 2026-09-25 | 기록 (웹 조사 워크플로 산출물 — 영역 4 + 종합 1, 에이전트 5. 대체 표시 없음. S-26 · P-29 는 "정의·리뷰어 행태·설계 원칙은 P-25…) | 00 · 01 · 02 · 03 · 05 · 07 · 09 · 09b |
| `LITERATURE_PREDICTOR_ROLE_2026-09-29.md` | 397 | 2026-09-29 | 기록 (문헌 조사; 머리 blockquote 에 대체 표시 없음. 본문 끝: "이 문서는 S-26 · P-25 를 대체하지 않는다 — 정의·리뷰어 행태·설계…) | 00 · 01 · 02 · 07 · 09 · 09b |
| `LITERATURE_STATE_WORLD_MODELS_2026-09-26.md` | 270 | 2026-09-26 | 기록 (문헌 조사, P-25 후속; 대체 표시 없음. "이 문서는 P-25 를 대체하지 않는다 — 정의·리뷰어 행태·설계 원칙은 그쪽이 정본") | 00 · 02 · 05 · 07 · 09 |
| `M13_MEMORY_VELOCITY_SUFFICIENCY_2026-09-25.md` | 375 | 2026-09-25 | 기록 (적대 검증 1 차 반영 정정판; 옛 제목 취소선 보존) | 03 · 04 · 06 · 07 |
| `M2B_BOUNDARY_V2_2026-09-25.md` | 71 | 2026-09-25 | 기록 (**1 차 결과 · 적대 검증 전**; README_UNTIL 은 "인용 안 함", 제목의 "encoder 가 경계 토큰에 통합한 역사" 는 가설로 읽…) | 03 · 04 · 05 · 07 |
| `M2_BOUNDARY_PATCH_2026-09-25.md` | 298 | 2026-09-25 | 기록 (적대 검증 1 차 반영 2 판; 옛 제목·1 판 본문 부록 A 에 취소선 보존) | 03 · 05 · 07 |
| `M4_RETRIEVAL_RADIUS_2026-09-25.md` | 343 | 2026-09-25 | 기록 (적대 검증 1 차 정정판, **등급 시사**; 원판 표·제목 보존) | 03 · 04 · 07 |
| `M5_VELOCITY_STEERING_2026-09-25.md` | 229 | 2026-09-25 | 기록 (적대 검증 2 차 반영 "정정 2"; 옛 제목 보존) | 03 · 04 · 07 |
| `M6_HISTORY_SHIFT_2026-09-25.md` | 259 | 2026-09-25 | 기록 (적대 검증 2 차 반영 "정정 2"; 옛 제목·옛 2×2 표 보존) | 03 · 04 · 07 |
| `P2_STRIDE_SWEEP_HORIZON_UNIT_2026-09-25.md` | 413 | 2026-09-25 | 정본 (적대 검증 VERIFY_0925 §1 P2 정정 13 건 반영; 옛 제목·해석 철회) | 03 · 04 · 07 |
| `P3B_LAST_TOKEN_DRIVES_TRANSITION_2026-09-25.md` | 205 | 2026-09-25 | 정본 2 판 (적대 검증 2 차 정정 9 건 반영; **파일 이름의 주장 "마지막 토큰이 전이를 정한다" 는 철회됨**) | 03 · 04 · 07 · 09 |
| `P3_CLOSURE_TEST_2026-09-25.md` | 269 | 2026-09-25 | 정본 (적대 검증 후 개정, 정정 12 건; 후속 판별은 P3b) | 03 · 04 · 07 · 09 |
| `PLAN_2026-09-24.md` | 231 | 2026-09-24 | 기록 (가설 트리 정본이었음 — 본문은 계획, 결과 없음; 머리 blockquote 에 변경 1·2·H4 수정·정정 4 개) | 00 · 03 |
| `PLAN_ENCODER_PREDICTOR_2026-09-25.md` | 213 | 2026-09-25 | 기록 (사전 계획서, 판정 규칙·공식 고정; 적대 검증 뒤 "→ 결과" 줄 덧붙임) — 제목 주장 "encoder 는 충분하다" 는 세워지지 않음 | 00 · 02 · 03 · 04 · 05 · 07 · 09 |
| `PREDICTOR_SHARE_2026-09-29.md` | 68 | 2026-09-29 | 기록 (1 차, 표 자체는 적대 검증 전; 행마다 재계산/전사 표시) | 02 · 09 |
| `README_UNTIL_2026-09-25.md` | 267 | 2026-09-25 | 대체됨 → `auto_research/README.md` | 00 · 02 · 03 · 04 · 05 · 07 · 08 |
| `REVIEW_CVPR_ORAL_2026-09-25.md` | 1984 | 2026-09-25 | 기록 (머리 blockquote 에 "대체됨" 표기 없음; 입력은 "적대 정정 뒤 문서 (2026-09-24 23:40 판)", "이 다음 판의 논문 초안은…) | 00 · 01 · 02 · 03 · 04 · 05 · 06 · 07 |
| `SC1_REACH_NOT_HORIZON_2026-09-25.md` | 181 | 2026-09-25 | 기록 (1 차 결과, 적대 검증 전; 같은 날 오후 자체 정정 1 건) | 00 · 04 · 07 · 08 · 09 |
| `SC2V2_FLOW_REACH_SSV2_2026-09-25.md` | 99 | 2026-09-25 | 기록 (1 차 결과, 적대 검증 전; SC2 1 판은 무효 선언) | 02 · 03 · 04 · 08 · 09 |
| `SC3_EK100_PREDICTOR_SUBSTITUTION_2026-09-25.md` | 70 | 2026-09-25 | 기록 (1 차 결과, 적대 검증 전) | 02 · 03 |
| `SC5_EK100_HORIZON_SWEEP_2026-09-26.md` | 57 | 2026-09-26 | 기록 (1 차, 적대 검증 전; 결과 §3 채움, 일부 Δ 는 VERIFY_ISCENE2 §7 에서 재현) | 02 · 03 |
| `TRAINED_PREDICTORS_SIGNATURES_2026-09-25.md` | 342 | 2026-09-25 | 정본 v3 (적대 검증 2 차 후 개정; 머리 "정정 2" 18 항, 부록 A 초판 표는 "기록용, 인용 금지") | 02 · 03 · 04 · 07 · 08 · 09 |
| `VERIFY_0925_2026-09-25.md` | 510 | 2026-09-25 | 기록 (적대 검증 워크플로 `verify-0925-results` — 검증자 6 · 교차 비평 1 · 종합 1 — 의 종합 정정 계획 + 교차 비평 원문; 레…) | 00 · 02 · 03 · 04 · 07 |
| `VERIFY_0925_R2_2026-09-25.md` | 169 | 2026-09-25 | 기록 (적대 검증 2 차 워크플로 `verify-0925-round2` 의 교차 비평 원문; 레포 파일 수정 없음) | 00 · 02 · 04 · 07 · 08 · 09 |
| `VERIFY_AR40RD_2026-09-30.md` | 81 | 2026-09-30 | 정본 (AR40_READOUT 의 적대 검증) | 03 · 05 · 06 · 07 · 08 |
| `VERIFY_AR_2026-09-27.md` | 121 | 2026-09-27 | 기록 (적대 검증 결과 — `AR_PREDICTOR` §2-1~2-3 을 확정 · 정정) | 03 · 05 · 08 |
| `VERIFY_ISCENE2_2026-09-26.md` | 210 | 2026-09-26 | 기록 (적대 검증 2 차, 독립 재계산 완료; 정정은 I_SCENE 본문 ⚠️ 표시로 반영) | 02 · 03 · 05 · 07 · 08 |
| `VERIFY_ISCENE_2026-09-25.md` | 125 | 2026-09-25 | 기록 (적대 검증 1 차; 정정 내용은 I_SCENE 머리 정정표에 반영됨) | 02 · 03 · 07 |
| `VERIFY_PRESENCE_2026-09-29.md` | 123 | 2026-09-29 | 기록 (적대 검증 결과 — `AR_PREDICTOR` §2-6 presence 검정을 확정 · 정정) | 02 · 03 · 05 · 08 |
| `VERIFY_R3_2026-09-29.md` | 106 | 2026-09-29 | 기록 (적대 검증 3 차 — `AR_PREDICTOR` §2-7 · `ENCODER_MOTION_READOUT` §2 · `COPY_HARD_SUBSET` §…) | 02 · 03 · 07 · 08 |

| 문서 | 줄 | 날짜 | 상태 | 장 |
|---|---:|---|---|---|
| `README.md` | 153 | 2026-09-25 | **정본 (결과 로그 · 철회표 · 문서 지도)** — 단 §0–§4 의 스토리 틀 · 결정 대기 · 다음 할 일은 **대체됨 → `auto_research/… | 00 · 02 · 03 · 05 · 06 · 07 · 08 · 09 |

### auto_research/exp_results

| 문서 | 줄 | 날짜 | 상태 | 장 |
|---|---:|---|---|---|
| `h23/REPORT_h23.md` | 273 | 2026-09-24 | 기록 (원시 자동 표 — 머리말 없음. 해석·정정은 `auto_research/Archive/H23_PREDICTOR_LAYERS_TRAJECTORY_2026…) | 02 · 03 · 04 · 06 · 07 |
| `h23b/REPORT_h23b.md` | 206 | 2026-09-24 | 기록 (H23 의 귀무 보정 · 공통 문턱 재분석 자동 표; 해석은 H23 Archive 정정 블록) | 03 · 04 · 07 |
| `h7/h7b_scoring_tables.md` | 1265 | 2026-09-24 | 기록 (자동 생성 표 `h7b_layer_scoring_splits.py`; H7 문서 §3 의 부록) — 1,000 줄 넘는 표 덤프라 구조 + 헤드라인만 | 02 · 03 · 05 |

### auto_research/figures

| 문서 | 줄 | 날짜 | 상태 | 장 |
|---|---:|---|---|---|
| `story/README.md` | 113 | 2026-09-26 | 기록 (그림 초안 데이터 지도, 새 결과 없음) | 00 · 02 · 03 · 05 · 07 · 08 |

### auto_research/paper

| 문서 | 줄 | 날짜 | 상태 | 장 |
|---|---:|---|---|---|
| `ABSTRACT_v0_2026-09-25.md` | 35 | 2026-09-25 | 대체됨 → `ABSTRACT_v1_2026-09-29.md` → `PAPER_STORY_2026-10-08.md` | 00 |
| `ABSTRACT_v1_2026-09-29.md` | 29 | 2026-09-29 | 대체됨 → `PAPER_STORY_2026-10-08.md` | 00 · 02 |
| `DESIGN_CHOICES_2026-09-25.md` | 98 | 2026-09-25 | 대체됨 (스토리 부분) → `PAPER_STORY_2026-10-08.md`; **§1 DC 표 · §3-b–d 설계 후보는 현행 (미검증 설계안)** | 00 · 08 · 09 · 09b |
| `DIRECTION_2026-09-25.md` | 101 | 2026-09-25 | 대체됨 → `auto_research/paper/PAPER_STORY_2026-10-08.md` | 00 · 02 · 08 · 09 |
| `DRAFT_v0_2026-09-25.md` | 115 | 2026-09-25 | 대체됨 → `PAPER_STORY_2026-10-08.md` | 00 · 02 · 04 |
| `DRAFT_v1_2026-09-25.md` | 187 | 2026-09-25 | 대체됨 → `PAPER_STORY_2026-10-08.md` | 00 · 02 · 07 · 08 |
| `DRAFT_v2_2026-09-25.md` | 179 | 2026-09-25 | 대체됨 → `PAPER_STORY_2026-10-08.md` | 00 · 02 · 04 · 08 |
| `NARRATIVE_v2_USER_2026-09-29.md` | 61 | 2026-09-29 | 대체됨 → `auto_research/paper/PAPER_STORY_2026-10-08.md` | 00 · 01 |
| `ORAL_DIRECTION_2026-09-25.md` | 132 | 2026-09-25 | 대체됨 → `DESIGN_CHOICES_2026-09-25.md` (09-25 저녁) → `PAPER_STORY_2026-10-08.md` | 00 · 02 · 04 · 08 |
| `PAPER_STORY_2026-10-08.md` | 335 | 2026-10-08 | 정본 | 00 · 03 · 09 · 09b |
| `RETHINK_PREDICTOR_ROLE_2026-09-29.md` | 110 | 2026-09-29 | 대체됨 → `auto_research/paper/PAPER_STORY_2026-10-08.md` | 00 · 02 · 07 · 08 · 09 |
| `REVIEW_v1_2026-09-25.md` | 1424 | 2026-09-25 | 대체됨 → auto_research/paper/PAPER_STORY_2026-10-08.md | 00 · 02 · 03 · 04 · 07 · 08 |
| `REVIEW_v2_2026-09-26.md` | 73 | 2026-09-26 | 대체됨 → auto_research/paper/PAPER_STORY_2026-10-08.md | 00 · 02 · 03 · 08 · 09 |
| `STORY_EVIDENCE_2026-09-26.md` | 111 | 2026-09-26 | 대체됨 → `PAPER_STORY_2026-10-08.md` (행별 수치 · 등급 · 빈틈 목록은 유효) | 00 · 02 · 03 · 04 · 05 · 07 · 08 |

### auto_research/scripts

| 문서 | 줄 | 날짜 | 상태 | 장 |
|---|---:|---|---|---|
| `README.md` | 31 | 2026-09-25 | 인프라·방법 (결과 없음) | 03 |

## configs  (6 개)

| 문서 | 줄 | 날짜 | 상태 | 장 |
|---|---:|---|---|---|
| `README.md` | 52 | 2026-09-21 | 정본 (config 지도) — 인프라 | 03 |

### configs/protocols

| 문서 | 줄 | 날짜 | 상태 | 장 |
|---|---:|---|---|---|
| `README.md` | 156 | 2026-08-29 | 정본 (실행 계약) — 인프라 · 방법 | 03 |
| `datasets.md` | 864 | 2026-08-29 | 정본 (데이터셋 레지스트리) — 경로 필드 생략, 결과/비고만 | 03 · 05 · 06 |
| `models.md` | 488 | 2026-08-29 | 정본 (모델 레지스트리) — 출처 · 비고만 | 02 · 03 · 08 · 09 |

### configs/training

| 문서 | 줄 | 날짜 | 상태 | 장 |
|---|---:|---|---|---|
| `README.md` | 103 | 2026-09-10 | 인프라·방법 (결과 없음) | 03 |
| `datasets.md` | 211 | 2026-09-10 | 인프라·방법 (학습셋 레지스트리, 결과 없음) | 03 · 09 |

## evals  (1 개)

### evals/analysis_vlm

| 문서 | 줄 | 날짜 | 상태 | 장 |
|---|---:|---|---|---|
| `modes/REPRODUCTION_PLAN.md` | 362 | 2026-07-03 | 기록 — **죽은 실험 세트의 blueprint**, 결과 없음 (config `analysis_vlm/` · `blender_toy_dataset/` 는 2…) | 01 · 07 |

## z_AC_training  (2 개)

### z_AC_training/Archive

| 문서 | 줄 | 날짜 | 상태 | 장 |
|---|---:|---|---|---|
| `PLANNING_AND_ROBOCASA_2026-09-20.md` | 151 | 2026-09-20 | 기록 (메모; 학습·추출 없음) | 03 · 08 · 09b |

| 문서 | 줄 | 날짜 | 상태 | 장 |
|---|---:|---|---|---|
| `README.md` | 194 | 2026-09-19 | 정본 (AC subsection 의 시작점) — **학습 실행 없음** | 03 · 09 · 09b |

## z_research  (155 개)

### z_research/Benchmarks

| 문서 | 줄 | 날짜 | 상태 | 장 |
|---|---:|---|---|---|
| `Archive/INTPHYS1_EVENT_POSITION_2026-09-25.md` | 114 | 2026-09-25 | 정본 (§0–4) / §5 대체됨 → `z_research/OcclusionStats/Archive/INTPHYS1_OCCLUSION_2026-09-26.md` | 02 · 03 |
| `PROTOCOLS.md` | 273 | 2026-09-21 | 정본 — 설정의 단일 출처 | 02 · 03 |
| `README.md` | 163 | 2026-09-21 | 정본 — 시작점 | 02 · 03 |
| `exp_results/_invalid_20260921_mixedgrid/README.md` | 8 | 2026-09-21 | 인프라 (무효 산출물 사유) | 03 |
| `exp_results/intphys2/bench_intphys2_main_ariel_bc_ep30_w16/column_best.md` | 38 | 불명 | 기록 | 02 · 08 |
| `exp_results/intphys2/bench_intphys2_main_ariel_bc_ep30_w32/column_best.md` | 38 | 불명 | 기록 | 02 · 08 |
| `exp_results/intphys2/bench_intphys2_main_ariel_bc_ep30_w48/column_best.md` | 38 | 불명 | 기록 | 02 · 08 |
| `exp_results/intphys2/bench_intphys2_main_vith/column_best.md` | 43 | 불명 | 기록 (산출물 표) | 02 |
| `exp_results/intphys2/bench_intphys2_main_vith_w16/column_best.md` | 38 | 불명 | 기록 | 02 |
| `exp_results/intphys2/bench_intphys2_main_vith_w32/column_best.md` | 38 | 불명 | 기록 | 02 |
| `figures/intphys1_windows/README.md` | 43 | 2026-09-25 | 인프라·방법 (그림 설명) | 02 · 03 |

| 문서 | 줄 | 날짜 | 상태 | 장 |
|---|---:|---|---|---|
| `HANDOFF_2026-09-22.md` | 342 | 2026-09-22 | 정본 (환경 구축) — 스토리 줄은 대체됨 → `auto_research/paper/PAPER_STORY_2026-10-08.md` | 03 |
| `OVERVIEW_2026-09-22.md` | 843 | 2026-09-22 | **§6 스토리 대체됨 → `auto_research/paper/PAPER_STORY_2026-10-08.md`**; 하네스 · 양식 · 결과 수치는 09-2… | 00 · 02 · 03 · 04 · 05 · 06 · 07 · 08 |

### z_research/IntPhysGenV10

| 문서 | 줄 | 날짜 | 상태 | 장 |
|---|---:|---|---|---|
| `Archive/GEOMETRY_2026-08-29.md` | 196 | 2026-08-29 | 기록 (머리에 대체 표시 없음) — **§1 α 분해 · §3 α 열 · §6c α(mu) 인용 금지** | 00 · 03 · 07 |
| `Archive/PAPER_STORY_2026-08-29.md` | 79 | 2026-08-29 | **대체됨** → `z_research/IntPhysGenV11/Archive/PAPER_STORY_2026-08-31.md` (08-31) → `auto_r… | 00 · 06 · 07 |
| `Archive/V10_2026-08-28.md` | 325 | 2026-08-28 | 기록 (머리에 대체 표시 없음; "α 분해·기하는 별도 문서" 라 적음) | 03 · 05 · 06 · 07 |
| `README.md` | 0 | 불명 | 없음 — 비어 있음 | 03 |

### z_research/IntPhysGenV11

| 문서 | 줄 | 날짜 | 상태 | 장 |
|---|---:|---|---|---|
| `Archive/PAPER_STORY_2026-08-31.md` | 335 | 2026-08-31 | 대체됨 → `auto_research/paper/PAPER_STORY_2026-10-08.md` (09-06 → 09-22 → 10-08 순). beat 3·… | 00 · 02 · 05 · 06 · 07 |
| `Archive/PAPER_STORY_2026-09-06.md` | 262 | 2026-09-06 | 대체됨 → `auto_research/paper/PAPER_STORY_2026-10-08.md` (09-22 판 경유). 근거 수치·금지 표현 유효 | 00 · 05 · 07 · 08 · 09 · 09b |
| `Archive/PAPER_STORY_2026-09-22.md` | 411 | 2026-09-22 | 대체됨 → `auto_research/paper/PAPER_STORY_2026-10-08.md`. 그대로 유효: §2 네 능력 표 (→ 정본 §3), §4-3… | 02 · 03 · 04 · 05 · 06 · 07 · 08 · 09 |
| `Archive/STATUS_AND_CRITIQUE_2026-09-03.md` | 170 | 2026-09-03 | 대체됨 → `auto_research/paper/PAPER_STORY_2026-10-08.md`. corner case 판정 (§8 · 하네스 9) 은 유효… | 00 · 02 · 03 · 05 · 07 |
| `Archive/SYNTHESIS_2026-08-30.md` | 506 | 2026-08-30 | 대체됨 → `auto_research/paper/PAPER_STORY_2026-10-08.md`. 08-30 시점 결과 사슬; 수치 유효 | 00 · 03 · 05 · 06 · 07 |
| `Archive/TALK_2026-08-30.md` | 328 | 2026-08-30 | 대체됨 → `auto_research/paper/PAPER_STORY_2026-10-08.md`. 08-30 스토리 기준 발표안 | 00 · 02 · 05 · 06 · 07 |
| `Archive/surprising_score/HYPOTHESIS_SCALAR_ORDER_2026-08-30.md` | 406 | 2026-08-30 | 기록 (가설 문서; "일부 확인, 대부분 미검증" — E1–E5 미구현). blockquote 없음 | 02 · 03 · 06 |
| `Archive/surprising_score/READING_THE_MATRIX_2026-08-31.md` | 194 | 2026-08-31 | 기록 (예제 설명서; 결론·전수는 `VOTE_GRID_2026-08-31.md`). blockquote 없음 | 03 · 05 · 06 |
| `Archive/surprising_score/RESULTS_2026-08-30.md` | 2098 | 2026-08-30 | 정본 (전수 기록) — blockquote 없음; 다른 문서들이 "전수 표 A–J" 로 참조 | 03 · 05 · 06 · 07 |
| `Archive/surprising_score/VOTE_GRID_2026-08-31.md` | 171 | 2026-08-31 | 기록 (beat 4 결론 + 전수 표). blockquote 없음 | 03 · 05 · 06 · 07 |
| `README.md` | 279 | 2026-10-08 | 정본 (세트 시작점) — 스토리 정본은 `auto_research/paper/PAPER_STORY_2026-10-08.md` | 00 · 03 · 05 · 06 · 07 |
| `figures/attention/README.md` | 65 | 불명 | 기록 (관찰; `analysis/attention/predictor_attn/plot.py` 그림 지도). blockquote 없음 | 03 · 07 |
| `figures/knockout/README.md` | 59 | 2026-09-03 | 기록 (개입; `analysis/attention/knockout/plot.py` 그림 지도). blockquote 없음 | 03 · 07 |
| `figures/probing/README.md` | 51 | 불명 | 인프라·방법 (그림 지도 + 금지 표현; 수치는 RESULTS 재인용). blockquote 없음 | 03 · 07 |
| `figures/surprise/README.md` | 81 | 2026-08-31 | 인프라·방법 (그림 지도) — 폴더는 대체된 `PAPER_STORY_2026-08-31.md` beat 기준, 그림 내용 유효. blockquote: 그 스토… | 00 · 03 |

### z_research/IntPhysGenV11_occlusion_timing_ablation

| 문서 | 줄 | 날짜 | 상태 | 장 |
|---|---:|---|---|---|
| `README.md` | 724 | 2026-09-01 | 정본 (세트 README; "옛 포인터 PAPER_STORY_2026-08-31 대체됨" 은 스토리 문서 얘기) | 03 · 05 · 07 |
| `figures/README.md` | 204 | 2026-09-02 | 인프라·방법 (그림 지도; 수치는 본 README 중복) | 03 |

### z_research/OcclusionStats

| 문서 | 줄 | 날짜 | 상태 | 장 |
|---|---:|---|---|---|
| `Archive/INTPHYS1_OCCLUSION_2026-09-26.md` | 137 | 2026-09-26 | 정본 | 02 · 05 |
| `README.md` | 37 | 2026-09-26 | 정본 — 세트 시작점 | 02 · 05 |
| `intphys1/_superseded/README.md` | 12 | 2026-09-26 | superseded | 00 |
| `intphys1/figures/_superseded/README.md` | 4 | 2026-09-26 | superseded | 00 |

### z_research/RollOutV2

| 문서 | 줄 | 날짜 | 상태 | 장 |
|---|---:|---|---|---|
| `Archive/PREDICTOR_DISTANCE_LIMIT_2026-09-19.md` | 212 | 2026-09-19 | 정본 (09-19 종합; 단 §3 의 "common sense, 정의 밖" 은 10-08 번복 표시가 본문에 있음) | 03 · 04 · 06 · 07 · 09b |
| `README.md` | 190 | 2026-09-12 | 정본 (세트 시작점; 수치 정본은 POSITION_READOUT_2026-09-12.md) | 03 · 04 · 05 · 06 · 07 |
| `exp_results/v5/locality/distance_decay.md` | 38 | 2026-09-19 | 정본 (자동 산출) | 03 · 04 · 06 |
| `exp_results/v5/locality/report.md` | 81 | 2026-09-19 | 정본 (자동 산출) | 07 |
| `figures/v5/summary/POSITION_READOUT_2026-09-12.md` | 468 | 2026-09-12 | 정본 (CLAUDE.md §5-5 지정) | 03 · 04 · 05 · 06 · 07 |
| `figures/v5/summary/_superseded/before_decel_20260919/README.md` | 5 | 2026-09-19 | superseded | 00 |
| `figures/v5/summary/_superseded/step_profile_flat_attn_20260919/README.md` | 4 | 2026-09-19 | superseded | 00 |
| `figures/v5/summary/two_futures_attn.md` | 34 | 2026-09-12 | 정본 (자동 산출 표) | 03 · 06 |
| `figures/v5/v11_vanish/emergence/_superseded/README.md` | 11 | 2026-09-14 | superseded | 00 |
| `figures/v5/v11_vanish/emergence/emergence.md` | 179 | 2026-09-14 | 정본 (자동 산출 표) | 03 · 05 |
| `figures/v5/v11_vanish/timing/attn_diag.md` | 95 | 2026-09-12 | 정본 (자동 산출 표) | 03 · 05 |
| `figures/v5/v11_vanish/timing/timing_summary.md` | 101 | 2026-09-12 | 정본 (POSITION_READOUT §5-4 의 정본 표) | 05 · 07 |
| `figures/v5/v11_vanish/timing/token_test.md` | 40 | 2026-09-12 | 정본 (자 검증 C 의 원표) | 03 · 05 |

### z_research/RollOutV3

| 문서 | 줄 | 날짜 | 상태 | 장 |
|---|---:|---|---|---|
| `Archive/CONTEXT_TO_FUTURE_2026-09-26.md` | 143 | 2026-09-26 | 정본 (전이 판정의 근거; 첫 판은 §7 에서 정정) | 03 · 04 · 06 · 07 |
| `Archive/IDENTITY_R8_2026-09-26.md` | 357 | 2026-09-26 | **정본** (지금 자 identity_r8 의 근거 + v3 · v11 결과; README 수치의 정본) | 02 · 03 · 04 · 05 · 07 |
| `Archive/READOUT_CHOICE_2026-09-23.md` | 78 | 2026-09-23 | 정본 (자 구조 선택 — 풀링은 그대로, 머리만 09-26 에 57-way 로 바뀜) · "다시 시도하지 말 것" | 03 · 06 |
| `Archive/RESULTS_2026-09-25.md` | 672 | 2026-09-25 | **부분 대체됨** → 자로 읽은 수치 (§1–§5, 결론 표 (b)·(e), 부록 A-1·A-3) 는 `Archive/IDENTITY_R8_2026-09-2… | 02 · 03 · 04 · 05 · 06 · 07 |
| `Archive/_superseded/NEW_ARCHIVE_README_2026-09-24.md` | 186 | 2026-09-24 | 대체됨 → r1 (09-25) → `README.md` → `IDENTITY_R8_2026-09-26.md` | 00 · 05 · 07 |
| `Archive/_superseded/NEW_ARCHIVE_README_2026-09-25_r1.md` | 629 | 2026-09-25 | 대체됨 (2차 검토) → `README.md` → `IDENTITY_R8_2026-09-26.md` | 00 · 02 · 03 · 04 · 05 · 06 · 07 |
| `Archive/_superseded/PREDICTOR_FUTURE_SUMMARY_2026-09-24.md` | 165 | 2026-09-24 | 대체됨 → 09-25 new_archive README → `IDENTITY_R8` | 00 · 03 · 07 |
| `Archive/_superseded/RECALL_TUBELET_LIMIT_2026-09-24.md` | 160 | 2026-09-24 | 대체됨 → `IDENTITY_R8_2026-09-26.md` §4; 결론 인용 금지 | 00 · 03 · 07 |
| `README.md` | 162 | 2026-09-26 | 정본 (세트 시작점; 수치 정본은 `Archive/IDENTITY_R8_2026-09-26.md`) | 03 · 04 · 05 · 06 · 07 |
| `_superseded/README_2026-09-25.md` | 168 | 2026-09-24 | 대체됨 → `RollOutV3/README.md` → `Archive/IDENTITY_R8_2026-09-26.md` | 00 · 03 · 07 |
| `_superseded/RERUN_2026-09-24.md` | 104 | 2026-09-24 | 인프라·방법 (결과 없음), 대체됨 → `decoder_train_redraw.sh` / `README.md` §재현 | 00 |
| `audit/README.md` | 22 | 2026-09-26 | 인프라·방법 (감사 지도) | 03 |
| `audit/training_sets/RollOut_v2_training_v8.md` | 124 | 2026-09-25 | 2026-09-25 → 09-26 3차 · 정본 (유효, 3차 통과) | 03 |
| `audit/training_v8/AUDIT.md` | 132 | 2026-09-25 | 기록 (옛 학습셋 감사; §E 가 판 오탐 원인 — 폴더의 `panel_false_alarm_*.json` 은 정본) | 03 · 05 |
| `audit/v11_identity/AUDIT.md` | 144 | 2026-09-25 | 2026-09-25 (xfer 2차) · 정본 (유효 — 자 무관) | 02 · 03 · 07 |
| `audit/v11_presence/AUDIT.md` | 57 | 2026-09-25 | 기록 (presence 자 — 지금 결론에 안 씀; 방법은 유효) | 02 · 03 · 05 |
| `audit/v3/AUDIT.md` | 68 | 2026-09-25 | 기록 — 방법 (궤적 단위 CI) 유효, 수치는 presence 자 | 03 · 04 · 06 |
| `audit/v3/tables.md` | 1812 | 2026-09-25 | 1,812 줄 · 2026-09-25 · 기록 — 자동 생성 표 (presence 자) | 03 |
| `exp_results/README.md` | 54 | 2026-09-26 | 인프라·방법 (결과 없음 — 자 · 읽은 값 폴더 지도) | 03 |
| `exp_results/_superseded/windows_v6_20260923/CROSS_HEAD.md` | 79 | 2026-09-24 | superseded (v6 학습셋 자) | 00 · 03 |
| `exp_results/_superseded/windows_v6_20260923/WINDOW_SUMMARY.md` | 118 | 2026-09-23 | superseded (v6 자) | 03 |
| `exp_results/presence/COMPARE.md` | 30 | 2026-09-26 | **대체됨** → `Archive/IDENTITY_R8_2026-09-26.md` (presence 자 시기) | 03 |
| `exp_results/presence/RESULTS.md` | 35 | 2026-09-26 | **대체됨** → `README.md` → `Archive/IDENTITY_R8_2026-09-26.md` (presence 자 시기 문서) | 03 |
| `exp_results/v11/V11_READOUT.md` | 68 | 2026-09-24 | **대체됨** → `Archive/IDENTITY_R8_2026-09-26.md`; 같은 readings 의 정의는 `RESULTS_2026-09-25.md`… | 03 · 05 · 06 |
| `exp_results/windows/BEHAVIOR.md` | 59 | 불명 | **대체됨** (presence 자 시기) + **초안 — 지표 확정 전, 사용자 결정 대기** | 03 · 06 |
| `exp_results/windows/CROSS_HEAD.md` | 303 | 2026-09-24 | **대체됨** (presence 자 시기) — 자 의존성 (이식) 검사 | 03 · 07 |
| `exp_results/windows/WINDOW_SUMMARY.md` | 129 | 불명 | **대체됨** (presence 자 시기) — 자 전이 검증 문서 | 03 · 04 |
| `figures/README.md` | 58 | 2026-09-26 | 인프라·방법 (결과 없음 — 그림 폴더 · 스크립트 · 자 도장 표) | 03 |
| `figures/_superseded/presence_era_2026-09-24/README.md` | 111 | 2026-09-24 | 대체됨 → `README.md` → `IDENTITY_R8` | 00 · 03 |
| `figures/_superseded/presence_era_2026-09-24/direction/DIRECTION_R2.md` | 126 | 2026-09-24 | superseded (presence 자, 탐색 표) | 00 · 06 |
| `figures/_superseded/presence_era_2026-09-24/examples/README.md` | 44 | 2026-09-24 | superseded (presence 자 그림 설명) | 00 |
| `figures/_superseded/presence_era_2026-09-24/readout/ERRORS.md` | 30 | 2026-09-24 | superseded (presence 자 그림 설명) | 00 |
| `figures/_superseded/presence_era_2026-09-24/readout/README.md` | 68 | 2026-09-24 | superseded (presence 자 그림 설명) | 00 |
| `figures/_superseded/presence_era_2026-09-24/v11/README.md` | 23 | 2026-09-24 | superseded (presence 자 그림 설명) | 00 |
| `figures/_superseded/presence_era_2026-09-24/windows/README.md` | 41 | 2026-09-24 | superseded (presence 자 그림 설명) | 00 |
| `figures/_superseded/presence_era_2026-09-24/windows_gif/README.md` | 33 | 2026-09-24 | superseded (presence 자 그림 설명) | 00 |
| `figures/_superseded/v11_dh_gif/README.md` | 22 | 2026-09-25 | superseded — "쓰지 않는다" | 03 · 07 |
| `figures/context_to_future/README.md` | 100 | 2026-09-27 | 정본 (그림 설명 · 캡션; 수치 정본은 `Archive/CONTEXT_TO_FUTURE_2026-09-26.md`) | 03 · 04 · 06 · 07 |
| `figures/context_to_future/_superseded/first_draft_2026-09-26/README.md` | 6 | 2026-09-26 | 2026-09-26/27 · superseded (그림 양식) | 06 |
| `figures/context_to_future/_superseded/simple_draft_2026-09-26/README.md` | 4 | 2026-09-26 | 2026-09-26/27 · superseded (그림 양식) | 06 |
| `figures/v11_geometry/GEOMETRY.md` | 335 | 2026-09-24 | **기록 (감사 반영 정본 → `Archive/RESULTS_2026-09-25.md` §7)** — G1 · G5 · G6/G7 만 결론으로; CKA · P… | 00 · 03 · 07 |
| `figures/v11_presence/_superseded/binary_majority_2026-09-27/README.md` | 5 | 2026-09-27 | superseded (표시 방식) | 03 |
| `figures/v11_presence/_superseded/identity_aeda55874c1c/README.md` | 5 | 2026-09-26 | 각 5 줄 · 2026-09-26 01:25–01:27 · 기록 (자동 내림) | 03 |
| `figures/v11_presence/_superseded/presence_decoder/README.md` | 18 | 2026-09-25 | 각 18 줄 · 2026-09-25 · superseded (옛 자 판) | 00 · 03 |
| `figures/v11_probe/PROBE.md` | 134 | 2026-09-24 | **기록 (감사 반영 정본 → `Archive/RESULTS_2026-09-25.md` §6)**; §5 항목 3 **철회** | 03 · 05 · 07 |
| `figures/v3_signed_error/_superseded/identity_aeda55874c1c/README.md` | 5 | 2026-09-26 | 각 5 줄 · 2026-09-26 01:25–01:27 · 기록 (자동 내림) | 03 |
| `figures/v3_signed_error/_superseded/presence_decoder/README.md` | 18 | 2026-09-25 | 각 18 줄 · 2026-09-25 · superseded (옛 자 판) | 00 · 03 |
| `figures/v3_signed_error_hhead/_superseded/identity_aeda55874c1c/README.md` | 5 | 2026-09-26 | 각 5 줄 · 2026-09-26 01:25–01:27 · 기록 (자동 내림) | 03 |
| `figures/v3_signed_error_hhead/_superseded/presence_decoder/README.md` | 18 | 2026-09-25 | 각 18 줄 · 2026-09-25 · superseded (옛 자 판) | 00 · 03 |
| `figures/v3_trajectory/_superseded/identity_aeda55874c1c/README.md` | 5 | 2026-09-26 | 각 5 줄 · 2026-09-26 01:25–01:27 · 기록 (자동 내림) | 03 |
| `figures/v3_trajectory/_superseded/presence_decoder/README.md` | 18 | 2026-09-25 | 각 18 줄 · 2026-09-25 · superseded (옛 자 판) | 00 · 03 |

### z_research/TrainingEffects

| 문서 | 줄 | 날짜 | 상태 | 장 |
|---|---:|---|---|---|
| `Archive/VERIFICATION_2026-09-25.md` | 103 | 2026-09-25 | 기록 (적대 검증 결과, 정정은 README 와 세부 문서에 반영) | 03 · 08 |
| `MODELS.md` | 75 | 2026-09-25 | 정본 (모델 목록) | 03 · 08 · 09 |
| `README.md` | 253 | 2026-09-25 | 정본 (TrainingEffects 세트 시작점, 적대 검증 반영본) | 02 · 03 · 04 · 05 · 06 · 07 · 08 · 09 |
| `ip1_copy/IP1_COPY.md` | 391 | 2026-09-25 | 정본 (맨 위 '⚠️ 정정' 10 항목 반영본) | 02 · 08 · 09 |
| `scores/ANALYSIS.md` | 477 | 2026-09-25 | 정본 (맨 위 '⚠️ 정정' 블록 = 적대 검증 반영본) | 02 · 03 · 05 · 08 |
| `scores/SCORES.md` | 120 | 2026-09-25 | 정본 (점수표, 손으로 고치지 않음) | 02 · 08 · 09 |
| `v11_identity/IDENTITY.md` | 458 | 2026-09-25 | 정본 (머리 blockquote 에 ⚠️ 정정 2026-09-25 적대 검증 반영) | 03 · 05 · 07 · 08 · 09 |
| `v11_presence/PRESENCE.md` | 572 | 2026-09-25 | 정본 (머리 blockquote ⚠️ 정정 2026-09-25 적대 검증 반영, 9 항목) | 02 · 03 · 04 · 05 · 08 · 09 |
| `v11_presence/TABLES.md` | 1566 | 2026-09-25 | 기록 (PRESENCE.md 의 전 칸 표, 같은 스크립트 산출) | 03 · 05 · 08 |
| `v3_motion/MOTION.md` | 311 | 2026-09-25 | 정본 (머리 blockquote ⚠️ 정정 2026-09-25 적대 검증 반영, 10 항목; 같은 날 04:10 판을 대체) | 03 · 04 · 06 · 08 · 09 |
| `v3_motion/TABLES.md` | 4257 | 2026-09-25 | 기록 (MOTION.md 정정판과 같은 실행의 전 칸 표) | 03 · 04 · 06 · 08 |

### z_research/anticipation

| 문서 | 줄 | 날짜 | 상태 | 장 |
|---|---:|---|---|---|
| `EK100/README.md` | 329 | 2026-09-14 | 정본 — 시작점 | 02 · 03 |

### z_research/context_encoder_analysis

| 문서 | 줄 | 날짜 | 상태 | 장 |
|---|---:|---|---|---|
| `Archive/CLAIM_NO_STATE_EVOLUTION_2026-09-15.md` | 179 | 2026-09-15 | **대체됨 → `Archive/STATE_EVOLUTION_CAPABILITIES_2026-09-19.md`** (정의 축이 §4 대응표로 흡수; 수치·철회표…) | 00 · 03 · 04 · 05 · 06 · 07 · 08 |
| `Archive/ENCODER_TEMPORAL_DYNAMICS_2026-09-20.md` | 119 | 2026-09-20 | **정본** (실험 6) | 02 · 03 · 07 |
| `Archive/EXPERIMENTS_2026-09-16.md` | 169 | 2026-09-16 | **정본** (이 세트의 실험 1~5 + 기존 run 재읽기 정리; README 가 "먼저 볼 것" 으로 지정; 실험 6 은 포함 안 됨) | 03 · 04 · 05 · 06 · 07 · 08 |
| `Archive/STATE_EVOLUTION_CAPABILITIES_2026-09-19.md` | 80 | 2026-09-19 | **정본 (정의)** — 10-08 에 읽는 틀만 바뀜 (정의 그대로), ledge/wall 번복 | 00 · 03 · 04 · 05 · 06 · 07 |
| `Archive/STATE_EVOLUTION_HYPOTHESES_2026-09-15.md` | 403 | 2026-09-15 | **대체됨 → `STATE_EVOLUTION_CAPABILITIES_2026-09-19.md`** (정의) · 판정은 `CLAIM` v0.1/v0.2 가 정본… | 00 · 01 · 02 · 03 · 04 · 05 · 07 · 08 · 09 · 09b |
| `README.md` | 30 | 2026-09-20 | 인프라·방법 (결과 없음) — 시작점·포인터 | 07 |
| `exp_results/ctxenc_boundary_identity_occlusion/RESULTS.md` | 127 | 2026-09-15 | **기록** — `ctxenc_boundary_identity_v2/` 가 같은 6 셀을 bit-identical 로 재현하고 확장 (v2 가 정본) | 03 · 05 · 07 |
| `exp_results/ctxenc_boundary_identity_v2/RESULTS.md` | 547 | 2026-09-16 | **정본 (E2 확장)** | 03 · 05 · 07 |
| `exp_results/ctxenc_boundary_kinematics/RESULTS.md` | 433 | 2026-09-16 | **정본 (E1)** | 03 · 06 · 07 |
| `exp_results/ctxenc_boundary_kinematics/n30/RESULTS.md` | 433 | 2026-09-15 | **superseded → `../RESULTS.md` (n=300)** — 표본 artifact 기록용 | 00 · 03 |
| `exp_results/ctxenc_state_subspaces_kinematics/RESULTS.md` | 744 | 2026-09-16 | **정본 (실험 5 운동 편)** | 03 · 06 · 07 |
| `exp_results/ctxenc_subspace_overlap/RESULTS.md` | 322 | 2026-09-15 | **기록** — `ctxenc_subspace_v2/` 가 같은 (a)(b)(c) 를 ramp 포함 9 조건으로 재실행 + 채점 재계산 (v2 가 정본); 이… | 03 · 07 |
| `exp_results/ctxenc_subspace_v2/RESULTS.md` | 738 | 2026-09-16 | **정본 (E4 정렬 사상 + matched-pair 채점 재계산)** — 단 raw 정확도는 bf16 캐시라 fp32 참조와 다름 (Δ 만 믿을 것) | 00 · 03 · 07 |
| `exp_results/ctxenc_time_alignment_gated/RESULTS.md` | 548 | 2026-09-15 | **정본 (실험 1 gated 판)** — v1 pilot 의 gate 수치는 직접 비교 불가 | 03 · 04 · 06 |
| `exp_results/encoder_temporal_dynamics/ntu/RESULTS.md` | 53 | 2026-09-20 | **정본 산출물** (실험 6 ntu) | 02 · 07 |
| `exp_results/encoder_temporal_dynamics/ssv2/RESULTS.md` | 53 | 2026-09-20 | **정본 산출물** (실험 6 ssv2) | 02 · 07 |
| `exp_results/time_alignment/time_alignment.md` | 205 | 2026-09-15 | **기록 (pilot)** — gated 판이 보완; "argmin 대각선 = 물체가 제 시각 자리" 해석은 시각 서명 때문에 철회 | 03 · 04 |

### z_research/predictor_training

| 문서 | 줄 | 날짜 | 상태 | 장 |
|---|---:|---|---|---|
| `predictor_IntPhysGenV11_PFT/Archive/RESULTS_2026-09-16.md` | 86 | 2026-09-16 | 정본 (§3-1 은 철회) | 03 · 04 · 06 · 08 |
| `predictor_IntPhysGenV11_PFT/Archive/SCORING_2026-09-16.md` | 221 | 2026-09-16 | 정본 (v11 held-out 채점 케이스별 분해) | 01 · 04 · 05 · 08 · 09 |
| `predictor_IntPhysGenV11_PFT/README.md` | 39 | 2026-09-16 | 정본 (세트 시작점) | 03 · 06 · 08 · 09 |
| `predictor_IntPhysGenV11_PFT/exp_results/pft/rollout_v2/RESULTS.md` | 13 | 2026-09-16 | 기록 (스크립트 자동 출력 표, `RESULTS_2026-09-16.md` 의 원자료) | 06 · 08 |
| `predictor_IntPhysGenV11_PFT/exp_results/pft/v11/RESULTS.md` | 38 | 2026-09-16 | 기록 (스크립트 자동 출력 표, `RESULTS_2026-09-16.md` 의 원자료) | 06 · 08 |
| `predictor_IntPhysGenV11_PFT/exp_results/pft/v11_color/RESULTS.md` | 38 | 2026-09-16 | 기록 (스크립트 자동 출력 표, `RESULTS_2026-09-16.md` 의 원자료) | 06 · 08 |
| `predictor_IntPhysGenV11_PFT/exp_results/pft/v11_shape/RESULTS.md` | 38 | 2026-09-16 | 기록 (스크립트 자동 출력 표, `RESULTS_2026-09-16.md` 의 원자료) | 06 · 08 |
| `predictor_IntPhysGenV11_PFT/exp_results/release/rollout_v2/RESULTS.md` | 13 | 2026-09-16 | 기록 (자동 출력 표, 릴리즈 대조군 · 파이프라인 검증용) | 03 · 04 · 05 · 06 |
| `predictor_IntPhysGenV11_PFT/exp_results/release/v11/RESULTS.md` | 38 | 2026-09-16 | 기록 (자동 출력 표, 릴리즈 대조군 · 파이프라인 검증용) | 03 · 04 · 05 · 06 |
| `predictor_IntPhysGenV11_PFT/exp_results/release/v11_color/RESULTS.md` | 38 | 2026-09-16 | 기록 (자동 출력 표, 릴리즈 대조군 · 파이프라인 검증용) | 03 · 04 · 05 · 06 |
| `predictor_IntPhysGenV11_PFT/exp_results/release/v11_shape/RESULTS.md` | 38 | 2026-09-16 | 기록 (자동 출력 표, 릴리즈 대조군 · 파이프라인 검증용) | 03 · 04 · 05 · 06 |

### z_research/scripts

| 문서 | 줄 | 날짜 | 상태 | 장 |
|---|---:|---|---|---|
| `HOWTO.md` | 89 | 2026-08-29 | **낡음 — 일부 명령이 삭제된 스크립트를 가리킨다** (인프라 · 방법, 결과 없음) | 03 |
| `README.md` | 193 | 2026-08-29 | 정본 (스크립트 목록) — 인프라 · 방법 (결과 없음) | 03 |
| `_superseded/README.md` | 6 | 2026-09-13 | superseded (실행 불가, 기록만) | 00 · 03 |

### z_research/v11_realistic

| 문서 | 줄 | 날짜 | 상태 | 장 |
|---|---:|---|---|---|
| `README.md` | 228 | 2026-10-06 | 정본 — 세트 시작점 | 03 · 04 · 05 · 06 |
| `exp_results/_previous_20261007_4futures/README.md` | 5 | 2026-10-07 | superseded 기록 | 00 |
| `exp_results/_previous_20261007_4futures/gravity_retrieval.md` | 51 | 2026-10-07 | **대체됨 → `exp_results/gravity_retrieval.md`** (20:36 재렌더) | 00 · 06 |
| `exp_results/gravity_retrieval.md` | 103 | 2026-10-07 | 정본 산출물 — README §4 의 원표 | 04 · 06 · 07 |
| `exp_results/tables.md` | 85 | 2026-10-06 | 기록 — 산출물 표, README §1–2 의 원표 | 05 · 06 |
| `figures/_superseded/README.md` | 14 | 2026-10-06 | superseded (그림 사유) | 00 |
| `figures/token_surprise/README.md` | 96 | 2026-10-07 | 정본 (토큰별 surprise 차 분석) | 03 · 06 · 07 |
| `figures/token_surprise/background_checks.md` | 35 | 2026-10-07 | 기록 — 산출물 표 | 03 · 07 |
| `figures/token_surprise/object_vs_background.md` | 15 | 2026-10-07 | 기록 — 산출물 표 | 07 |

### z_research/v11_roll_out

| 문서 | 줄 | 날짜 | 상태 | 장 |
|---|---:|---|---|---|
| `README.md` | 159 | 2026-09-07 | 기록 (본문 머리에 대체 표시 없음; 그러나 결론이 이후 RollOutV2 정본과 충돌 — 아래) | 00 · 03 |

## z_training  (5 개)

| 문서 | 줄 | 날짜 | 상태 | 장 |
|---|---:|---|---|---|
| `ARIEL_CHECKPOINTS.md` | 242 | 2026-09-27 | 정본 (Ariel 세 팔의 지금 판). 옛 `z_training/ariel/README.md` (09-23~24) 는 소실 — 결론은 `configs/prot… | 02 · 03 · 05 · 08 · 09 |
| `README.md` | 153 | 2026-09-10 | 인프라·방법 (결과는 링크·요약만) | 03 · 08 · 09 |
| `RESULTS_2026-09-19.md` | 82 | 2026-09-19 | 정본 (post-FT 세 run 결과 종합) | 02 · 08 · 09 |

### z_training/runs

| 문서 | 줄 | 날짜 | 상태 | 장 |
|---|---:|---|---|---|
| `_matrix/MATRIX.md` | 57 | 2026-09-19 | 정본 (post-FT 네 run 의 학습 데이터 × 채점 bench 전이 행렬, 자동 생성 `z_training/harness/matrix.py`) | 08 · 09 |
| `_matrix/predictor_v1_holdout_l1.md` | 17 | 2026-09-19 | 정본 (MATRIX L1 열의 출처) | 08 |

## z_world_model_analysis  (12 개)

| 문서 | 줄 | 날짜 | 상태 | 장 |
|---|---:|---|---|---|
| `FRAME_ABLATION_2026-08-19.md` | 256 | 2026-08-19 | 기록 (머리에 대체 표시 없음; §5 에 자기 정정) | 02 · 03 · 05 |
| `MECHANISM_2026-08-20.md` | 333 | 2026-08-20 | 기록 (머리에 대체 표시 없음; §6 폐기 가설표 · 철회 2건 포함) — **§7 α 분해 수치 인용 금지** | 00 · 02 · 03 · 07 |
| `OCCLUSION_LENGTH_2026-08-20.md` | 154 | 2026-08-20 | 기록 (머리에 대체 표시 없음) — **핵심 결론 "가림 길이가 레버" 는 이후 v10 · v11 에서 뒤집힘** | 00 · 02 · 03 · 05 |
| `PREDICTOR_HEDGING_2026-08-26.md` | 205 | 2026-08-26 | 기록 (머리에 대체 표시 없음) — **⚠️ 헤지 · α · 보정 계수 문서: 수치 인용 금지, 존재만 기록** | 00 · 05 · 07 |
| `RESULTS_2026-08-16.md` | 577 | 2026-08-16 | 기록 (머리에 대체 표시 없음; "현재 유효한 최신 실행 4건만" 담은 당시 종합) — IntPhys1 88.89 는 지금도 정본 수치 | 02 · 03 · 05 · 07 |
| `TOKEN_SUBSET_2026-08-19.md` | 99 | 2026-08-19 | 기록 (기각 결과; 머리에 대체 표시 없음) | 02 · 03 · 05 |
| `VERIFICATION_2026-08-20.md` | 334 | 2026-08-20 | 기록 (머리에 대체 표시 없음; 정정·철회 다수 포함) — **§1 α 항등식 · 표준화 편향 · "153배" 수치 인용 금지** | 00 · 02 · 03 · 05 |
| `WHY_INTPHYS_WORKS_2026-08-20.md` | 247 | 2026-08-20 | 기록 (머리에 대체 표시 없음; §5 철회 포함) | 02 · 03 · 05 |
| `intphys1_dev_2026-08-13.md` | 167 | 2026-08-13 | 기록 (머리에 대체 표시 없음; 단 `RESULTS_2026-08-16.md` 가 "이전 실행은 *_PREVIOUS-DATA.md 참조" 라 하고 프로토콜이…) | 02 · 03 |
| `occlusion_v2_2026-08-13_PREVIOUS-DATA.md` | 306 | 2026-08-13 | **superseded** (머리 blockquote: 영상이 2026-08-15 재렌더, 수치는 재렌더 이전 영상 기준) | 00 · 03 · 07 |
| `probe_set_2026-08-13_PREVIOUS-DATA.md` | 56 | 2026-08-13 | **superseded** (머리 blockquote: 영상 2026-08-15 16:02 재렌더) | 03 · 07 |

### z_world_model_analysis/fig

| 문서 | 줄 | 날짜 | 상태 | 장 |
|---|---:|---|---|---|
| `intphys1_examples/README.md` | 41 | 불명 | 인프라·방법 (예시 GIF; 결과 수치는 예시 3쌍뿐) | 02 |

---

## 2026-10-10 이후 추가 — 92 개 (갱신 2026-10-21; 쓰는 장은 `13_SINCE_2026-10-10.md` 의 절)

> 10-10 스냅샷 뒤 생기거나 바뀐 md. **줄 · 날짜** 는 2026-10-21 파일 실물 (mtime), **상태** 는 추출문이 원 문서 머리에서 읽은 것, **장** 은 13 장의 절 (§) 또는 부록. `configs/protocols/*.md` · `auto_research/README.md` · `PAPER_STORY_2026-10-08.md` · `z_training/README.md` · `MATRIX.md` · EK100 README 는 10-10 이전 색인에도 있는 문서인데 10-10 이후 바뀐 곳이 있어 여기 다시 적었다 (위 표의 행은 그대로 둔다).

### (루트)  (1 개)

| 문서 | 줄 | 날짜 | 상태 | 장 |
|---|---:|---|---|---|
| `RESEARCH_SUMMARY_FOR_COLLAB_2026-10-10.md` | 175 | 2026-10-10 | 기록 (협업 제안용 — 새 수치 없음, 10-13 이전 설계 후보) | 00 · 13 §8 |

### auto_research  (7 개)

| 문서 | 줄 | 날짜 | 상태 | 장 |
|---|---:|---|---|---|
| `Archive/PREDICTOR_BOTTLENECK_2026-10-13.md` | 178 | 2026-10-13 | 기록 — 결론 1 · 3 **철회**, §3 는 '어디 상태 분리' 로 → 정본 `paper/DESIGN_CHOICES_2026-10-13.md` | 13 §6 · §10 |
| `README.md` | 159 | 2026-10-13 | 현행 결과 로그 — ⚠️ 머리 blockquote 철회 전 문장, 표 행에만 철회 | 13 §10 |
| `paper/DESIGN_CHOICES_2026-10-13.md` | 206 | 2026-10-13 | 정본 (방법 절 설계 근거 · 팔 A0–A3) | 13 §6 · §8 |
| `paper/DESIGN_CHOICES_2026-10-13_EASY.md` | 233 | 2026-10-13 | 정본의 요약판 (합격선 추가 둘) | 13 §8 |
| `paper/EXPERIMENTS_WHAT_WHERE_2026-10-13.md` | 586 | 2026-10-13 | 정본 (실험 × 주장 지도; E1a/E1b 교락 단서) | 13 §4 · §6 |
| `paper/PAPER_STORY_2026-10-08.md` | 339 | 2026-10-13 | 정본 — ⚠️ 10-13 blockquote 가 철회 전 문장 | 13 §8 · §10 |
| `paper/STATUS_2026-10-13.md` | 197 | 2026-10-14 | 정본 (현재 상황; 10-14 B13 정정 · post-FT attention) | 13 §5 · §6 |

### configs  (1 개)

| 문서 | 줄 | 날짜 | 상태 | 장 |
|---|---:|---|---|---|
| `training/datasets.md` | 516 | 2026-10-21 | 레지스트리 (학습셋 25 개 추가) | 13 부록 |

### cvpr  (3 개)

| 문서 | 줄 | 날짜 | 상태 | 장 |
|---|---:|---|---|---|
| `eval/ek100/cache/README.md` | 102 | 2026-10-17 | 인프라 (encoder 특징 캐시 — 논문 수치 없음) | — |
| `eval/intphys1_late/README.md` | 152 | 2026-10-18 | 정본 (새 하네스 + 결과; 등급 없음 → 1 차) | 13 §9 |
| `results/eval/intphys2/_breakdown_ar_state_2026-10-12/breakdown.md` | 132 | 2026-10-12 | 기록 (표만 · 해설 없음) | 13 §5 |

### z_research/VOBench_v1  (59 개)

| 문서 | 줄 | 날짜 | 상태 | 장 |
|---|---:|---|---|---|
| `Archive/EVAL_MATRIX_2026-10-15.md` | 76 | 2026-10-17 | 정본 (평가 행렬; 10-15 · 10-17 지표 정정) | 13 §3 · §7 |
| `Archive/RESULTS_SUMMARY_2026-10-12.md` | 352 | 2026-10-13 | 정본 (종합, 10-13 갱신) | 13 §2 · §5 · §7 |
| `README.md` | 418 | 2026-10-14 | 정본 — 세트 시작점 (10-11 7 차 · 10-14 갱신) | 13 §2–§5 |
| `analysis/attention_map/README.md` | 193 | 2026-10-14 | 정본 (해석 정본은 STATUS B13) | 13 §6 |
| `analysis/attention_map/compare/_superseded/README.md` | 8 | 2026-10-14 | 인프라 (물러난 그림) | 13 §6 |
| `ariel_ar_ep40/gravity/README.md` | 26 | 2026-10-11 | 인프라 (정리본 안내 · 캡션 초안) | 13 §3 |
| `ariel_ar_ep40/gravity/tables.md` | 233 | 2026-10-11 | 정본 표 (재렌더판) | 13 §3 |
| `ariel_ar_ep40/surprise/README.md` | 31 | 2026-10-11 | 정본 (토큰 Δ — presence without placement) | 13 §5 |
| `ariel_ar_ep40/surprise/TRACKING_2026-10-12.md` | 66 | 2026-10-12 | 정본 — ⚠️ 같은 날 공간 분포 분석 철회 (다시 시도 금지) | 13 §5 · §10 |
| `ariel_ar_ep40/surprise/tables.md` | 245 | 2026-10-11 | 정본 표 (AR 공간 복사 `copy_blk`) | 13 §5 |
| `ariel_ar_state_ep40/postft_vobench/README.md` | 36 | 2026-10-12 | 기록 — ⚠️ 첫 split 누수 (인용은 v2 판 `vjepa2_pft/`) | 13 §7 · §10 |
| `ariel_ar_state_ep40/postft_vobench/c16t24/compare.md` | 86 | 2026-10-12 | 기록 — ⚠️ 첫 split 누수 (인용은 v2 판 `vjepa2_pft/`) | 13 §7 · §10 |
| `ariel_ar_state_ep40/postft_vobench/compare.md` | 86 | 2026-10-12 | 기록 — ⚠️ 첫 split 누수 (인용은 v2 판 `vjepa2_pft/`) | 13 §7 · §10 |
| `ariel_ar_state_ep40/surprise/README.md` | 32 | 2026-10-11 | 정본 (ar 와 구분 안 됨) | 13 §5 |
| `ariel_ar_state_ep40/surprise/tables.md` | 245 | 2026-10-11 | 정본 (ar 와 구분 안 됨) | 13 §5 |
| `bottleneck/README.md` | 21 | 2026-10-14 | 인프라 (산출물 지도 — 해석은 auto_research) | 13 §6 |
| `dino_foresight/surprise/README.md` | 23 | 2026-10-11 | 인프라 (폴더 안내 · 재현) | 13 §2 |
| `dino_foresight/surprise/tables.md` | 245 | 2026-10-10 | 정본 표 (vll7 판은 README · 해석 없음, 복사 없음) | 13 §2 |
| `exp_results/_previous_20261010_gravity_g981/README.md` | 5 | 2026-10-11 | 기록 (옛 중력 판 — 재렌더판과 섞지 말 것) | 13 §3 |
| `headroom/README.md` | 12 | 2026-10-13 | 인프라 (산출물 지도 — 해석은 auto_research) | 13 §6 |
| `postft_split_v2/compare.md` | 86 | 2026-10-14 | 정본 표 (split v2 test 기준값) | 13 §7 |
| `readout/README.md` | 67 | 2026-10-12 | 정본 (AR 자 — 같은 날 '물체다운 것 없다' 정정) | 13 §5 |
| `readout/horizon_ar40/README.md` | 32 | 2026-10-12 | 정본 (1 차 — 감쇠는 블록 순서) | 13 §5 |
| `vast_ar_state_ep70/surprise/README.md` | 50 | 2026-10-14 | 정본 (10-13 이후 ar_state 기준 · post-FT 포함) | 13 §5 · §7 |
| `vast_ar_state_ep70/surprise/tables.md` | 245 | 2026-10-14 | 정본 (10-13 이후 ar_state 기준 · post-FT 포함) | 13 §5 · §7 |
| `videomae_v2/surprise/README.md` | 23 | 2026-10-11 | 인프라 (폴더 안내 · 재현) | 13 §2 |
| `videomae_v2/surprise/tables.md` | 190 | 2026-10-10 | 정본 표 (vll7 판은 README · 해석 없음, 복사 없음) | 13 §2 |
| `vjepa2/encoder_voe_probe/README.md` | 51 | 2026-10-14 | 정본 (encoder 위반 선형 probe) | 13 §4 |
| `vjepa2/gravity/README.md` | 27 | 2026-10-11 | 인프라 (정리본 안내 · 캡션 초안) | 13 §3 |
| `vjepa2/gravity/tables.md` | 233 | 2026-10-11 | 정본 표 (재렌더판) | 13 §3 |
| `vjepa2/gravity_horizon/README.md` | 32 | 2026-10-11 | 인프라 (설계 · 캡션) | 13 §3 |
| `vjepa2/gravity_horizon/figures/_superseded/README.md` | 6 | 2026-10-11 | 인프라 (그림 1 · 2 차 물러남, 수치 동일) | 13 §3 |
| `vjepa2/gravity_horizon/tables.md` | 183 | 2026-10-11 | 정본 표 | 13 §3 |
| `vjepa2/probing/README.md` | 47 | 2026-10-13 | 정본 (튜블릿 정체 probe · 16 층 절) | 13 §4 |
| `vjepa2/probing/concept/RESULTS.md` | 46 | 2026-10-10 | 기록 (무효 판정 — 개념 벡터 희석) | 13 §10 |
| `vjepa2/probing/results/L16_e35_s0/RESULTS.md` | 64 | 2026-10-13 | 정본 run (수치 run 별) | 13 §4 |
| `vjepa2/probing/results/L16_e35_s1/RESULTS.md` | 64 | 2026-10-13 | 정본 run (수치 run 별) | 13 §4 |
| `vjepa2/probing/results/_invalid_run1_realistic_collapsed/NOTE.md` | 5 | 2026-10-10 | 무효 (realistic head 붕괴 — 인용 금지) | 13 §10 |
| `vjepa2/probing/results/_invalid_run1_realistic_collapsed/RESULTS.md` | 60 | 2026-10-10 | 무효 (realistic head 붕괴 — 인용 금지) | 13 §10 |
| `vjepa2/probing/results/e10_s0/RESULTS.md` | 64 | 2026-10-10 | 정본 run (수치 run 별) | 13 §4 |
| `vjepa2/probing/results/e10_s1/RESULTS.md` | 64 | 2026-10-10 | 정본 run (수치 run 별) | 13 §4 |
| `vjepa2/probing/results/e35_s0_realistic_only/RESULTS.md` | 36 | 2026-10-10 | 정본 run (수치 run 별) | 13 §4 |
| `vjepa2/probing/results/e35_s0_user/RESULTS.md` | 64 | 2026-10-10 | 정본 run (수치 run 별) | 13 §4 |
| `vjepa2/probing/results/e35_s1/RESULTS.md` | 64 | 2026-10-10 | 정본 run (수치 run 별) | 13 §4 |
| `vjepa2/probing/results/p_only_e10_s0/RESULTS.md` | 64 | 2026-10-10 | 정본 run (수치 run 별) | 13 §4 |
| `vjepa2/probing/results/p_only_e10_s1/RESULTS.md` | 64 | 2026-10-10 | 정본 run (수치 run 별) | 13 §4 |
| `vjepa2/probing/results/p_only_e35_s0/RESULTS.md` | 64 | 2026-10-10 | 정본 run (수치 run 별) | 13 §4 |
| `vjepa2/probing/results/p_only_e35_s1/RESULTS.md` | 64 | 2026-10-10 | 정본 run (수치 run 별) | 13 §4 |
| `vjepa2/probing/temporal_generalization/RESULTS.md` | 160 | 2026-10-10 | 정본 (1 차 — 형식이 시점마다 다름) | 13 §4 |
| `vjepa2/surprise/README.md` | 23 | 2026-10-11 | 인프라 (폴더 안내 · 재현) | 13 §2 |
| `vjepa2/surprise/tables.md` | 245 | 2026-10-10 | 정본 표 (vll7 판은 README · 해석 없음, 복사 없음) | 13 §2 |
| `vjepa2/surprise_object_region/README.md` | 34 | 2026-10-13 | 정본 (보조 분석 — 물체 자리 토큰만 채점) | 13 §2 · §5 |
| `vjepa2/surprise_object_region/compare.md` | 79 | 2026-10-13 | 정본 (보조 분석 — 물체 자리 토큰만 채점) | 13 §2 · §5 |
| `vjepa2_pft/README.md` | 103 | 2026-10-14 | 정본 (릴리즈 VOBench post-FT + 자 + 상한선) | 13 §7 |
| `vjepa2_pft/SETTINGS.md` | 85 | 2026-10-14 | 인프라 (설정 정리; split v1 누수 기록) | 13 §7 |
| `vjepa2_pft/readout_figures/motion_fidelity_late.md` | 58 | 2026-10-14 | 정본 표 (1 차 · 해석 문장 없음) | 13 §3 · §7 |
| `vjepa2_pft/readout_figures/motion_fidelity_seen.md` | 58 | 2026-10-14 | 정본 표 (1 차 · 해석 문장 없음) | 13 §3 · §7 |
| `vjepa2_pft/surprise/compare.md` | 86 | 2026-10-14 | 정본 표 | 13 §7 |
| `vll7_ctxt_tgtvideo/surprise/tables.md` | 190 | 2026-10-17 | 정본 표 (vll7 판은 README · 해석 없음, 복사 없음) | 13 §2 |

### z_research/previous  (7 개)

| 문서 | 줄 | 날짜 | 상태 | 장 |
|---|---:|---|---|---|
| `README.md` | 30 | 2026-10-10 | 인프라 (10-10 세트 이동 기록) | 13 부록 |
| `TrainingEffects/Archive/PHYS2D_V1_POSTFT_2026-10-18.md` | 342 | 2026-10-19 | 정본 (§9 가 결론; §7-b 추론 정정) | 13 §7 |
| `anticipation/EK100/Archive/CAP16METRIC_SUBSET_2026-10-21.md` | 109 | 2026-10-21 | 정본 (규칙 사후 복원, sha256) | 13 §7 |
| `anticipation/EK100/Archive/CHEAP_PROXY_2026-10-15.md` | 67 | 2026-10-15 | 기록 (중간 결과 · 팔 A 취소) | 13 §7 |
| `anticipation/EK100/Archive/GROUP4_CAP16METRIC_2026-10-18.md` | 77 | 2026-10-21 | 정본 (jsonl 재확인) | 13 §7 |
| `anticipation/EK100/Archive/GROUP5_VAST_VARIANTS_2026-10-18.md` | 112 | 2026-10-18 | 정본 (run 간 노이즈 0.98) | 13 §7 |
| `anticipation/EK100/README.md` | 331 | 2026-10-21 | 정본 (시작점; cap16metric 주석 2 줄) | 13 §7 |

### z_training  (8 개)

| 문서 | 줄 | 날짜 | 상태 | 장 |
|---|---:|---|---|---|
| `README.md` | 165 | 2026-10-19 | 정본 (시작점; 10-18 IntPhys 2 LoRA 절) | 13 §7 |
| `docs/PHYS2D_DATASET_2026-10-18.md` | 141 | 2026-10-19 | **대체됨 → `PHYS2D_SUP_DATASET_2026-10-18.md` §V** (전환이 문맥 끝 — 사용자 결정 반대) | 13 §7 · §10 |
| `docs/PHYS2D_SUP_DATASET_2026-10-18.md` | 221 | 2026-10-19 | 정본 (§V 상태 라벨 판 v2; v1 창 규칙 깨짐 기록) | 13 §7 |
| `docs/STATE_HEAD_2026-10-19.md` | 136 | 2026-10-19 | 정본 · 진행 중 (대조군 셋 미실행) | 13 §7 · §8 |
| `runs/_matrix/MATRIX.md` | 75 | 2026-10-14 | 정본 (자동 생성 — 제목 09-19 · 10-14 갱신 75 줄) | 13 §7 |
| `runs/phys2d_sup_vast_ar_state_headonly/eval/state_head_latest.md` | 33 | 2026-10-19 | 산출물 (자동 채점표 — 해석 문서 없음) | 13 §7 |
| `runs/phys2d_sup_vast_ar_state_pft/eval/state_head_latest.md` | 33 | 2026-10-19 | 산출물 (자동 채점표 — 해석 문서 없음) | 13 §7 |
| `vast/README.md` | 25 | 2026-10-10 | 기록 · 인프라 (반입 체크포인트; 10-13 이후 폴더 미기재) | 13 §7 |

### z_vll7_training  (3 개)

| 문서 | 줄 | 날짜 | 상태 | 장 |
|---|---:|---|---|---|
| `README.md` | 253 | 2026-10-17 | 인프라 + §6–§7 결과 (영상 타깃 predictor 48.33) | 13 §7 |
| `data/README.md` | 10 | 2026-10-16 | 인프라 | — |
| `tests/README.md` | 10 | 2026-10-16 | 인프라 | — |

### z_pixel_decoder  (3 개)

| 문서 | 줄 | 날짜 | 상태 | 장 |
|---|---:|---|---|---|
| `README.md` | 78 | 2026-10-21 | 인프라 (latent → 픽셀 decoder · 렌더 — predictor 수치 없음) | — |
| `runs/vob_grav_i1_vitb_both/_superseded/README.md` | 3 | 2026-10-21 | 인프라 (latent → 픽셀 decoder · 렌더 — predictor 수치 없음) | — |
| `runs/vob_grav_i1_vitb_both/cosmos_ar4b/_superseded/README.md` | 3 | 2026-10-21 | 인프라 (latent → 픽셀 decoder · 렌더 — predictor 수치 없음) | — |

