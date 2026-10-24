# 출처 목록 (SOURCES) — 근거표가 읽은 곳과 읽은 시점

**질문**: 근거표 v1 은 어느 branch · 어느 파일을, 어느 시점 판으로 읽었나.
**답**: main 작업 트리 (96024d9 + 미추적 산출물, 2026-10-21 20:40 KST 기준), origin/vast f6e7934, origin/ariel 7417f68, origin/intphys1-late-uni 2bd0d5d (로컬 6090fca), 그리고 다른 팀 회신 네 번을 읽었다. Vast 에만 있는 것은 이 머신에서 산출물로 확인할 수 없다.
**다음 할 일**: 라운드 2 에서는 아래 "기준점" 이후 바뀐 것만 다시 읽는다.

판: **v2.1 (라운드 2 단계 0 수집 반영, 2026-10-22 00:15)** · 작성: 진행자 1 · 이 문서는 출처 목록이라 파일 · run 이름을 그대로 쓴다 · v2 에서 §8 을 더했다 (리뷰 · 다른 팀 회신 5–8 · 새 산출물)

---

## 0. 다음 라운드 기준점

| 무엇 | 값 |
|---|---|
| 라운드 1 수집 시각 | 2026-10-21 20:15 – 20:45 KST (`git fetch --all` 20:16, 새 커밋 없음) |
| main (HEAD = origin/main) | `96024d9` 2026-10-21 19:49 "EK100 cap16metric 부분집합 규칙 복원" |
| origin/vast | `f6e7934` 2026-10-20 10:08 UTC (main 과 38ae9fb 09-22 에서 갈라짐) |
| origin/ariel | `7417f68` 2026-10-17 19:50 (main 과 38ae9fb 에서 갈라짐) |
| origin/intphys1-late-uni | `2bd0d5d` 2026-10-21 18:46 (main 4950aa7 + 커밋 하나) · 로컬 `6090fca` = 그 위에 EK100 커밋 (main 96024d9 와 같은 내용) |
| 다음 라운드 명령 | `git -C vjepa2 log --all --since="2026-10-21 20:15" --name-only` + `find <폴더> -newermt "2026-10-21 20:15"` (runs · cvpr/results · z_research · z_training · z_vll7_training · z_pixel_decoder · auto_research · paper_lab/_team) |
| 진행 중이라 다음 라운드에 다시 볼 것 | 다른 팀 [P] (기준 블록 AR 문맥 자르기) · [E] (표적 떨림 VOBench · EK100) · [T] (잡음 줄인 표적) · 진행자 2 의 측정 E1 (job 227495) 과 E3 · Cosmos latent 결정판 · `vobench_release_postft_objholdout` 채점 · `natural_ar_vw_state_phys3d_only` (Vast, 학습 중) |

## 1. main 작업 트리 — 추적되는 문서

| 경로 | 마지막 commit · mtime | 한 줄 요약 |
|---|---|---|
| `CLAUDE.md` | c627386 10-12 · 작업 트리 수정 10-21 20:07 (문서 작성 규칙 +30 줄, 미커밋) | 레포 운영 규칙 · 10-08 스토리 요약 · 확립된 결과 · 기각 가설 · 금지 표현 |
| `auto_research/README.md` | 4950aa7 10-19 · 10-13 16:39 | predictor 분석 결과 로그 · 철회표 · 문서 지도 |
| `auto_research/paper/PAPER_STORY_2026-10-08.md` | 4950aa7 · 10-13 04:19 | 스토리 정본 P1–P3 · 목표 · AC, 합격선 표 (§2-4) |
| `auto_research/paper/STATUS_2026-10-13.md` | 4950aa7 · 10-14 21:31 | 릴리즈 vs Ariel 블록 AR 행동 비교 · 병목 B1–B13v |
| `auto_research/paper/DESIGN_CHOICES_2026-10-13.md` | 4950aa7 · 10-13 14:46 | "무엇은 이어지고 어디는 마지막 관측에 묶인다", 설계 ① 공간 대조 ② 마지막 관측 가리기 ③ 어디 상태 |
| `auto_research/paper/EXPERIMENTS_WHAT_WHERE_2026-10-13.md` | 4950aa7 · 10-13 16:39 | 위 설계의 실험 E1–E8 세팅 · 타당성 위협 |
| `auto_research/Archive/PREDICTOR_BOTTLENECK_2026-10-13.md` | 4950aa7 · 10-13 16:39 | 진행이 4–5 걸음 뒤 0, 속도 코드도 같이 감쇠 (철회 · 정정 포함) |
| `auto_research/Archive/{FUTURE_FROM_CONTEXT_2026-10-02, COPY_VS_POSITION_2026-10-07, AR_PREDICTOR_2026-09-27, I_SCENE_CAUSAL_BOTTLENECK_2026-09-25, INTPHYS2_PERMANENCE_AUDIT_2026-09-26, DINOF_*}.md` | f754be3 10-09 | 문맥 토큰의 미래 위치 판독 · 위치 쌍 · Ariel AR · 자연 영상 도달 거리 · IntPhys 2 복사 감사 · DINO-F |
| `z_research/VOBench_v1/Archive/EVAL_MATRIX_2026-10-15.md` | 4950aa7 · 10-17 11:48 | 모델 4 × 벤치마크 6 행렬 (23 칸 채움, 기준 블록 AR × GRASP 빈 칸) |
| `z_research/VOBench_v1/Archive/RESULTS_SUMMARY_2026-10-12.md` | 4950aa7 · 10-13 19:44 | VOBench · IntPhys 1 · 2 · AR 판독 · 미세조정 종합 |
| `z_research/VOBench_v1/README.md` · `vjepa2_pft/README.md` | 4950aa7 · 10-14 | VOBench 세트 · 릴리즈 VOBench 미세조정 표 |
| `z_research/previous/TrainingEffects/Archive/PHYS2D_V1_POSTFT_2026-10-18.md` | 4950aa7 · 10-19 04:57 (§7-b 정정 ~ §9 미커밋) | 2D 합성 물리 미세조정 두 팔, 팔 A 는 1 차 외삽 |
| `z_training/README.md` · `RESULTS_2026-09-19.md` · `ARIEL_CHECKPOINTS.md` | 425d67d · df972a3 · f754be3 | 학습 하네스 · 미세조정은 도메인 안에서만 · Ariel 세 팔 채점 정본 |
| `z_training/docs/PHYS2D_SUP_DATASET_2026-10-18.md` | 425d67d · 10-19 15:44 | 2D 합성 상태 라벨 학습셋 (v2 정본) |
| `z_vll7_training/README.md` | 4950aa7 · 10-17 15:18 | 영상 문맥 vs 블록 문맥 학습 하네스, 영상 표적 판 결과 (IntPhys 1 48.33) |
| `cvpr/eval/intphys1_late/README.md` | 4950aa7 · 10-18 02:25 | IntPhys 1 끝 가림 판들 (고르게 뽑은 판 절은 없음) |
| `z_research/previous/anticipation/EK100/README.md` · `Archive/{CAP16METRIC_SUBSET_2026-10-21, GROUP4_CAP16METRIC_2026-10-18, GROUP5_VAST_VARIANTS_2026-10-18}.md` | 96024d9 10-21 · 4950aa7 | EK100 축소 학습판 규칙 복원 · predictor 넷 / 변형 다섯 구분 불가 |
| `z_research/previous/{OcclusionStats, Benchmarks}/README.md` | c627386 | IntPhys 1 가림 통계 · Garrido 프로토콜 (GRASP · InfLevel 칸은 낡음) |
| `z_archive_10/README.md` | c627386 · 10-10 | 10-10 까지 결과 문서 264 개 지도 (이후 결과 없음) |

## 2. main 작업 트리 — 미추적 문서 · 산출물

| 경로 | mtime | 한 줄 요약 |
|---|---|---|
| `paper_lab/_team/SFA_REPLY_2026-10-21.md` | 10-21 (회신 네 번 누적) | 다른 팀 진단 사슬 · 측정 목록 · 정정 (운반 이득 순서) · 표적 떨림 자연 영상 1 차 |
| `z_training/docs/STATE_HEAD_2026-10-19.md` · `PHYS2D_DATASET_2026-10-18.md` | 10-19 23:38 · 15:44 | 고정 predictor 위 head 셋이 복사보다 2.5–3 배 나쁨 · 2D 합성 v1/v2 (대체됨) |
| `z_pixel_decoder/README.md` · `runs/vob_grav_i1_vitb_both/compare/{late,gravity}/summary.json` | 10-21 17:24 · 15:21–15:30 | 블록 latent → 픽셀 decoder (ViT-B, Vast 학습) 와 렌더 비교 PSNR |
| `z_research/scripts/analysis/sheet_row_report.py` | 10-19 14:57 | 사용자 시트 한 행 (IntPhys 1 · VOBench 끝 가림 · 중력 · InfLevel) 재계산 규칙 |
| `z_training/runs/*` (30 개) | 09-10 – 10-21 18:41 | 미세조정 · 라벨 head · IntPhys 1 끝 가림 fold · reach 등. 상태는 근거표 표 1-8a, 미평가: `vobench_release_postft_objholdout` · `intphys2_postft_pair_e40` · `phys2d_single_v1_pft_mix` |
| `z_training/runs/phys2d_sup_vast_ar_state_pft/eval/state_head_latest.md` · `_vast_ar_ep70_probe/head_fit/*.json` | 10-19 | 라벨 head 의 가려진 위치 판독 (근거 #22) |
| `z_vll7_training/runs/*` (10 개) | 10-17 – 10-21 13:52 | 영상 문맥 (영상 표적) 70/70 · 블록 문맥 + 2D 합성 13/70 멈춤 · 2D 합성 물체 하나 미세조정 넷 (두 개 9/15 중단) |
| `cvpr/results/eval/{intphys1, intphys1_late, intphys2, testbed, ek100}/` | 10-09 – 10-21 20:14 | 정식 채점 산출물 (gitignore). 10-21 에도 GRASP · InfLevel 실행이 쌓이는 중 |
| `z_research/VOBench_v1/{headroom, bottleneck, exp_results, */surprise}/` | 10-10 – 10-17 | 진단 산출물 (조립 상한 · 표적 국소성 · 바꿔치기 · 역사 값 · 속도 코드) 과 VOBench 채점표 |
| `z_research/VOBench_v1/vjepa2/ssv2_straightness/*.json` | 10-15 · 10-17 | SSv2 예측 궤적 곧음 (해석 문서 없음, 근거 #15) |
| `cvpr/results/eval/testbed/cosmos_ar4b_latent{,_topk1,_smoke}` | 10-21 | Cosmos 4B latent (부분 · 비결정 판, 인용 보류) |
| `z_research/previous/TrainingEffects/exp_results/phys2d_heldout/summary.json` | 10-19 04:05 | 2D 합성 도메인 안 예측 L1 (복사 포함) |

## 3. origin/vast (`wt_vast`, f6e7934) — 산출물은 Vast 에만

| 경로 | 마지막 commit | 한 줄 요약 |
|---|---|---|
| `CLAUDE.md` | e130003 10-18 (§2-7 한 줄만 바꿈) | 거의 09-22 main 판. 10-08 스토리 · 복사 기준선 규칙 없음. 10-08 이후 vast 결과 기록 없음 |
| `z_training_config/{README, NATURAL_AR_MS_2026-10-14, CTXT_VIDEO_TGTVIDEO_HOLE_2026-10-16, DESIGN_OPEN_CONTEXT_TARGET_2026-10-16}.md` | e130003 10-18 | 여러 시작점 rollout 설계 (학습 안 함) · 영상 문맥 · 영상 표적 · 문맥 구멍 · 문맥/정답 설계 선택지 |
| `z_training/README.md` · `VARIANTS.md` | cf8e528 10-19 · 909f1b3 10-18 | 시작점 · 모든 변형의 철학 · 구조 · run 표 |
| `z_training/docs/RESULTS_2026-10-{12,15,16}.md` · `RESULTS_GATING_2026-10-18.md` | e130003 | 가변 창 · 힌트 · 변화량 L1 · VOBench 학습 · GRASP · InfLevel · EK100 · 영상 문맥 · 마지막 관측 섞기 (모두 기준 이하) |
| `z_training/docs/PROBLEMS_AND_DIRECTION_2026-10-18.md` | e130003 | 능력별 표 · 문제 P1–P6 · 방향 D1–D5 |
| `z_training/docs/{PHYS2D_SINGLE_DATASET_2026-10-19, PHYS3D_DATASET_2026-10-19}.md` | cf8e528 | 2D 합성 물체 하나 · 3D 합성 (생성 중단 18,063 / 30,800) |
| `z_training/docs/STATE_SUPERVISION_2026-10-20.md` | f6e7934 10-20 | 라벨 보조 손실 변천 · head 는 읽게 되지만 채점은 기준과 같음 (Vast 쪽 수정본은 미푸시) |
| `z_research/VOBench_v1/Archive/DIAG_CTXT_VIDEO_VS_AR_2026-10-17.md` | 061d07c 10-17 | 영상 문맥 vs 블록 문맥 진단 (약 100 표본): 위치 오차 · 속도 배율 · 같은 칸 회귀 · 물체 자리 채점 |
| `z_research/VOBench_v1/vast_*/surprise/tables.md` | 73ae468 / 061d07c | vast 변형별 VOBench 채점표 (요약과 일치 검증 문구) |
| `z_pixel_decoder/README.md` | ebabbf7 10-20 | 블록 latent → 픽셀 (사용법만) |
| 커밋 메시지 e4ba067 · b888c61 · 167db20 · 177291c · 36f4921 | 10-19 – 10-20 | 잡음 바닥 수치 · head 단독 학습 수치 (문서 없이 메시지에만) |
| **Vast 에만 있고 푸시 안 된 것** | — | `auto_research/` 전체 (README · LOG · TEAM_BOARD · scripts · results) · `CTXT_VS_BLOCK_MECHANISM_2026-10-21.md` · `PHYS3D_SINGLE(_V2)` 문서 · STATE_SUPERVISION 수정본 · run `natural_ar_vw_state_phys3d_only` |

## 4. origin/ariel (`wt_ariel`, 7417f68)

| 경로 | 마지막 commit | 한 줄 요약 |
|---|---|---|
| `CLAUDE.md` | 3c09415 09-29 | 기계 이전 (ariel-k2) 운영 규칙만 더함. 스토리는 08-31 판 · 09-19 정의 그대로 |
| `z_research/action_planning/README.md` | 56ee24d 10-15 | Push-T · Wall planning 하네스, 미세조정 0.60 = 처음부터 0.60 |
| `configs/training/datasets.md` · `z_research/scripts/README.md` | 56ee24d · c973560 | ariel-k2 학습셋 등록 · 노드 이름 |
| 영상 문맥 · 마지막 관측 섞기 config 머리말 | 7417f68 10-17 | 결과 문서 없음. "gres" 이름이 main 판과 다른 모델을 가리킨다 |

## 5. origin/intphys1-late-uni (`wt_intphys1-late-uni`, 2bd0d5d) · 로컬 6090fca

| 경로 | 마지막 commit | 한 줄 요약 |
|---|---|---|
| `CLAUDE.md` | main 과 같음 | 차이 없음 |
| `configs/training/datasets.md` · `cvpr/registry/{datasets,models}.yaml` | 2bd0d5d 10-21 | 2D 합성 · IntPhys 1 끝 가림 fold · 물체 떼어 둔 VOBench 학습셋 등록. registry 결함 둘 (GRASP 키 이동 · 중력 키 중복) |
| `data_csv/intphys1_late_uni{,_cv}` | 2bd0d5d | worktree 에만 있다 (main 작업 트리에 없음) — fold 겹침을 main 에서 확인 못 함 |
| `z_research/previous/anticipation/EK100/Archive/CAP16METRIC_SUBSET_2026-10-21.md` | 6090fca (= main 96024d9) | train 만 클래스당 16 개 상한, val 은 공식 전수 |

## 6. 다른 팀 (slow_feature_analysis, Vast)

| 회신 | Vast 기준 시각 | 내용 |
|---|---|---|
| 첫 번째 | 10-21 11:15 | 원인 사슬 (고리 1–5), 측정 목록, 미푸시 목록 |
| 두 번째 | 11:30 | 잡음 줄인 표적 [T] 방법 · 진행자 2 측정과 겹침 |
| 세 번째 | 11:40 | 운반 이득 순서 정정 (블록 0.29 · 영상 0.59) · 고리 3 = 기준 블록 AR · 3D 합성 · 최근 블록 위주 가중 평균 + 일부 운반 |
| 네 번째 | (리더 전달 10-21 저녁) | 표적 떨림 자연 영상 1 차 (K710 32 % · SSv2 39 % · HowTo 33 %), 블록마다 독립 몫 96–97 % · 역사 값 정의 요청 → 진행자 1 이 리더 통해 답함 |
| 조정 (리더) | 10-21 저녁 | 기준 블록 AR 문맥 자르기는 그쪽 [P] 가 같은 판에서 맡는다 |

## 7. 이 머신에 없는 것

- 사용자 원 작업 폴더 `/workspace/world_model/vjepa2` (Vast) 와 `z_autoresearch/` — 어느 branch 에도 없다.
- "MotionJEPA" PDF — `papers/` 에 없다 (진행자 2 가 웹에서 찾는다).
- Ariel 판 Push-T 산출물 (ariel-k2 `/nas2`).

## 8. v2 에서 더 읽은 것 (라운드 1 리뷰 뒤, 2026-10-21 21:45–22:10)

| 경로 | 시점 | 한 줄 요약 |
|---|---|---|
| `paper_lab/REVIEWS/round_1/{R1,R2,R5,SUMMARY}.md` | 21:36–21:46 | 리뷰 셋 (accept 2 · oral 1) · 지적 F1–F4 · I1–I13 · 배분 |
| `paper_lab/scripts/review_r2/{r2_check_gravity,r2_check_ip1_blockcopy,r2_check_late108,r2_check_vobench}.py` | 21:30–21:42 | R2 검산 스크립트 (읽기 전용). 중력 · IntPhys 1 블록 복사 짝 bootstrap · 끝 가림 108 쌍. 진행자 1 이 중력 판을 다시 돌려 일치 확인 |
| `paper_lab/_team/SFA_REPLY_2026-10-21.md` 다섯–여덟 번째 회신 | 21:00–21:50 | 운반 이득 정의 · 표적 비국소 독립 재현 · 칸 미만 이동 · 기준 블록 AR 문맥 자르기 (K710 보류) · 마지막 블록 수축 · SSv2 학습 목록 겹침 98 % |
| `papers/Interpreting Physics in Video World Models.pdf` (arXiv 2602.07050) | 로컬 | Joseph et al. 2026 — 속도 · 가속도 앞 층 판독 (R1 확인, #62) |
| `papers/Intuitive physics understanding emerges from self-supervised pretraining on natural videos.pdf` (arXiv 2502.11831) | 로컬 | Garrido 2025 — GRASP 지지 98.1 · 중력 74.9 (R1 확인, #61) |
| Xie & Alanwar, arXiv 2610.07355 | 웹 (R1 실물 확인) | 물체 자리 판독으로 존재 · 합성 학습 처방 (#60) |
| `/data2/local_datasets/something-something/something-something-v2-mp4/{train,val}.csv` | 로컬 | SSv2 공식 분할 (168,913 · 24,777). `data_csv/ssv2/val_min48.csv` 대조 (#59) |
| `z_training/ariel/ar_future_1e_5/latest.pt` (`config`) | 09-29 | Ariel 판 학습 목록 = SSv2 자체 train 목록 + K400 (#59) |
| `z_research/VOBench_v1/headroom/vobench_r*.npz` · `gravity_r*.npz` | 10-13 | 칸별 오차 지도 (복사 · Ariel 판 · 정답 자리 오라클) — v2 의 물체 자리 채점 (#56 · #57) 원자료 |
| `paper_lab/scripts/f4_baselines.py` → `paper_lab/_scratch/f4_baselines.json` | 21:55 (새) | 같은 공간 기준선 셋 중 복사 · 오라클 + 물체 자리 채점 (진행자 1) |
| `paper_lab/scripts/verify_evidence_v2.py` | 22:05 (새) | v1 32 + v2 22 항목, 54 일치 |
| `paper_lab/_scratch/tspace/` (진행자 2 의 E1 · E2, job 227497) | 21:56 VOBench 추출 끝 (`vobench_r0.npz`), SSv2 · 분석은 진행 중 | 문맥 평균 복사 (세 번째 기준선) · 걸음별 평균 보상 — VOBench 몫을 진행자 1 이 재계산해 #16 · #57 · #64 에 넣었다 (스모크 n = 3–4 는 쓰지 않음) |
| `paper_lab/_scratch/evidence_pending_r1.md` | 21:45 | 리뷰 동결 중 대기 메모 — v2 에 전부 반영 |

## 9. 라운드 2 단계 0 지속 수집 (2026-10-22 00:14 KST, 기준점 10-21 20:15)

`git fetch --all` → 모든 branch 에 새 커밋 없음 (main 96024d9 · origin/vast f6e7934 · origin/ariel 7417f68 · origin/intphys1-late-uni 2bd0d5d 그대로). paper_lab 밖의 새 · 바뀐 미추적 파일:

| 경로 | mtime | 한 줄 요약 | 근거표 |
|---|---|---|---|
| `cvpr/results/eval/testbed/cosmos_ar4b_latent/` (16 + 3 조각, `vobench_pairs.json` · `gravity_pick.json`) | 10-21 21:01 – 10-22 00:09 | 사용자 run. Cosmos-Predict1-4B 결정판 (argmax) VOBench · 중력 — 끝 가림 이동 52.9, 선반 0 / 144 | #71 |
| `z_research/scripts/analysis/cosmos_vobench_latent.py` | 10-22 | 위 run 의 스크립트, `--set intphys1_uni` 추가 (IntPhys 1 끝 가림을 Cosmos 창으로) | — |
| `z_pixel_decoder/runs/_frames/IntPhys1_uni/` · `slurm_logs/` | 10-21 – 10-22 | 위 IntPhys 1 끝 가림 Cosmos 판의 프레임 준비 (결과 아직 없음) | — |
| `z_vll7_training/configs/ar_vw_shortctx.yaml` · `runs/ar_vw_shortctx/` · `monitor_ar_vw.sh` | 10-21 20:25 – 10-22 00:14 | 사용자 학습: **짧은 문맥 (1–2 블록) · 긴 예측 (1–16 블록)** 블록 AR (출력 되먹임, 처음부터, K400 + SSv2), 30 epoch 째, 체크포인트 e5–e25. config 머리말이 "C 축소와 K 확대를 못 가른다 · 데이터 비율 다름 · 복사 기준선 필수" 를 미리 적었다. 평가 결과 아직 없음 → 하네스로 재면 된다 | (다음 판) |
| `cvpr/results/eval/intphys1/grasp_level2_vith_pft_intphys1_e40_wgarrido_w32` | 10-21 | IntPhys 1 미세조정 predictor 의 GRASP 창 32 (옛 모델, 행렬 밖) | — |
| `paper_lab/_team/SFA_REPLY_2026-10-21.md` 아홉 · 열 번째 회신 | 10-21 23:35 | κ (#70) · A1 정답 의존 가중의 분위수 편향 [이론] (하네스 B1 · B2 로 대응) | #70 |
