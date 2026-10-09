# z_research/scripts

**최상위에는 사람이 직접 치는 것만 둔다.** 나머지는 역할별 폴더로 내린다.

```
run.sh          표준 진입점.  GPUS=8 bash z_research/scripts/run.sh <프로토콜> <데이터셋> [모델]
sbatch.sh       위를 SLURM 으로 제출
monitor.sh      watch -n 1 bash z_research/scripts/monitor.sh [job id | 로그경로]

harness/        run.sh 가 부르는 것
  resolve.py      프로토콜 + datasets.md + models.md 를 병합. --set / extends / auto sweep

data/           인덱스·데이터 준비
  build_grasp_index.py        GRASP level2 인덱스 (4096 영상 / 2048 쌍). 공식은 P_<prop>/<scene> vs
                              IP_<prop>/<scene> 한 쌍 (2026-09-20)
  build_inflevel_index.py     InfLevel 인덱스. **공식 CSV(auxiliary_data_loading_files/inflevel/)가 정본**이고
                              한 줄이 한 쌍. priming 구간을 잘라낸 시작점을 frame_start 컬럼에 쓴다.
                              쌍 1116/1182/450 (2026-09-21 전면 재작성)
  build_rollout2_index.py     RollOut_v2 인덱스 — 라벨은 metadata 가 아니라 plan 에서 (검증 15항목)
  build_probe_imp_index.py    불가능 변이를 probing 대상으로 여는 인덱스
  build_intphysgen_index.py   IntPhysGen 렌더 metadata.csv -> 표준 index.csv (v11 열 + sym_k) + 문맥 무결성 전수 감사 (v11_realistic · ledge, 2026-10-06)
  build_ek100_resized.py      EPIC-KITCHENS 비디오 짧은변 256 -> 가운데 256x256, 프레임을 N/fps 로 재번호해 원본 decord 인덱스와 1:1 (파일마다 프레임 수 대조) -> /data2/local_datasets/EPIC-KITCHENS_resized (vll5 에서 확인)
  ek100_resized_progress.py   위 전처리 진행률 (완료 파일 + 도는 ffmpeg 읽은 양 -> %, 남은 시간). 로그가 비어 있는 초반에 쓴다
  verify_ek100_resized_frames.py  재인코딩본을 dataloader 와 같은 인덱스로 decord 로 읽어 원본 프레임과 픽셀 대조 (밀림 d=-2..2, 동률·중복 프레임 구분)
  verify_ek100_resized_headers.py  원본 vs 재인코딩본 파일을 직접 다시 읽어 대조: ffprobe 헤더 700 개 + decord 프레임 수·fps·색 (표본)
  build_ek100_val_official.py  EK100 validation 138 개를 공식 V-JEPA 2 val 입력 (짧은변 292 INTER_LINEAR -> 가운데 256, BT.709 태그) 과 같게 -> /data2/local_datasets/epic_test (--split train|both 로 train 495 도 같은 기하)
  sbatch_build_ek100_official.sh  위 빌드를 vll5 SLURM (GPU 8 · CPU 128 · mem 전체 할당, 기본 train crf 15). sbatch 로만 부른다

figures/        논문 그림. 전부 산출물에서 재계산해 summary.json 과 대조 검증한다
  plot_v11_surprise.py      v11 채점 그림 전부.  상단 FIGDIR 표가 하위 폴더를 배정한다
                            (01_condition / 02_occlusion_k / 03_direction / 04_object_order)
                            표에 없는 이름은 최상위에 떨어진다 — 새 그림이 눈에 띄라고 일부러
  gif_token_surprise.py     토큰별 surprise 차 Δ 를 원본 위에 겹친 GIF + 물체/배경 기여 표 -> z_research/v11_realistic/figures/token_surprise/ (2026-10-07)
  plot_v11_realistic.py     v11_realistic 그림 3 장 (가림 유무 · k × 방향 · ledge) -> z_research/v11_realistic/figures/ (2026-10-06)
  plot_v11_probing.py       v11 probing 그림.  324칸 전수 대조 후 by_condition/ by_k/ 로
  plot_v11_occtiming.py     가림 타이밍(early/mid/late) 그림.  **두 run 을 합친 report** 를 받는다
                            (v11 본체 + v11_earlymid). FIGDIR: 01_timing / 02_timing_k / 03_object_order
  plot_rollout2_readout.py  RollOut_v2 위치 readout overlay·궤적 (--pooling --rep p|z|h)
  plot_rollout2_traj_gif.py  위 traj PNG 의 GIF 판: 32 샘플 프레임 위에 정답(흰)·읽기(주황) 궤적 누적 + time bar -> <pooling>/<rep>/traj_gif/
  plot_rollout2_summary.py  RollOut_v2 fig_l2 / fig_err_xy / fig_motion_gain / fig_motion_xy (p·z·h 막대)
  plot_rollout2_step_profile.py  RollOut_v2 p 의 걸음 모양: 한 축 사영, 슬롯별 이동·누적 위치 (운동 9 종 / 11 패널) -> summary/fig_step_profile
  plot_rollout2_xy_profile.py    RollOut_v2 화면 x·y 속도-시간 (기본, -> summary/fig_xy_velocity) / 위치-시간 (--pos, -> fig_xy_profile). 문맥 끝 등속 대조, 공통 세로 스케일, 11 패널
  plot_v11_vanish_readout.py  v11 pos_a 에 위치 자 적용: GIF·overlay·readout.npz (--rep p|z --motion flat|ramp|static --timing late|early|mid --k)
  plot_v11_vanish_pair_gif.py  encoder z vs predictor p 나란히 GIF (발표용; --dataset v11|v2)
  plot_v11_vanish_timing.py  v11 전체 timing × k 표·그림 (readout.npz 모음 → v11_vanish/timing/)
  plot_v11_position_k1.py  v11 visible / early·mid·late k=1 미래 위치: 진실 vs z vs p + 출구 모서리 (→ v11_vanish/emergence/fig_position_k1)
  plot_v11_emergence.py  v11 가림: 미래에서 물체가 가림막 출구를 넘는가, context encoder z vs predictor p (→ v11_vanish/emergence/, --plot-only)
  plot_probe_pos_imp_confusion.py  ledge/wall pos·imp probe confusion 2×4 (probe_pos_imp.json → figures/probe_pos_imp/fig_confusion_all_*)
  plot_v11_position.py      v11_full 위치·속도 readout 그림 (v11_roll_out/figures/position/). --fit-by condition|motion|all
  plot_v11_knockout.py      attention knockout 그림 (06_knockout/ + by_violation/ by_motion/ by_timing/ by_k/).
                            입력은 knockout merge.py 의 results.json 하나. 축별은 `breakdown` 을 block 가중으로 합쳐 쓴다
  plot_violation_bars.py    채점, 조건 x 위반          (--flat-style green|hatch)
  plot_direction_bias.py    방향 비대칭                (--by-condition 로 2x2)
  plot_probing_bars.py      z/h/p probing              (--line, --group-fit, --head)
  plot_confusion.py         predictions.json -> confusion matrix
  plot_vanish_direction.py  vanish 방향별
  plot_intphys1_bars.py     IntPhys1 채점
  plot_occlusion_intphys1.py IntPhys1 가림 통계 그림 (쌍 × 프레임 가시성 래스터 · 길이/이동/정답 3 패널) — occlusion_intphys1.py 산출물만 읽음 -> OcclusionStats/intphys1/figures/ (2026-09-26)
  plot_intphys1_target_inputs.py IntPhys1 88.89 칸에서 target encoder 가 받는 32 장 (모델 입력 텐서, 256) — --simple <video> 로 창 9 × 32 장 한 장 -> Benchmarks/figures/intphys1_windows/ (2026-09-25)
  plot_intphys1_windows.py  IntPhys1 88.89 칸 (skip2_w32) 창마다 context / future / target encoder 가 받는 raw 프레임 지도 + 한 쌍의 9 창 썸네일 (Filtered 가 고른 C 경계) -> z_research/Benchmarks/figures/intphys1_windows/ (2026-09-25)
  plot_ek100_anticipation_gif.py  EK100 anticipation 과제 GIF (CONTEXT 4s / GAP 1s / ACTION 라벨 + 모델 입력 영역) -> z_research/anticipation/EK100/figures/samples/
  plot_ctxenc_direction.py  실험 6 그림: 좌/우 방향 혼동행렬 4 개 (데이터셋 2 x 정방향·역재생) + probe 별 요약 막대 → context_encoder_analysis/figures/encoder_temporal_dynamics/ (2026-09-20)
  plot_v3_l2_error.py       RollOut_v3 p 위치 오차 16 창 × 8 법칙. --signed = 축마다 앞섬/뒤처짐 + 등속 연장·복사 기준선 → RollOutV3/figures/v3_signed_error[_hhead]/ (--head h = h 자로 p 읽기). 플래그 없는 |오차| 판은 new_archive 에서 뺐다 (2026-09-25)
  plot_v3_trajectory.py     RollOut_v3 마지막 관측에서 옮긴 위치·속도 (GT · p · z) → figures/v3_trajectory/
  plot_v3_decoder_gif.py    RollOut_v3 영상 위 진실 · p@h · h@h (P ≤ 16 창 12 개, 삽화) → figures/v3_gif/
  plot_v11_presence.py      v11 visible / mid / last (k≥2) × 운동: p '있다' 비율 · 위치 L2 → figures/v11_presence/ (판정은 _audit/v11_presence/)
  plot_v11_signed_error.py  v11 조건 9 개 (가림 없음 · 문맥 끝 k1–4 · 중간 k1–4) × 정지/flat/ramp: p 앞섬/뒤처짐 (x·y), 기준선 = 마지막으로 본 자리 멈춤 · 본 속도 연장, h 기준선 → figures/v11_signed_error/ (2026-09-26)
  plot_context_to_future.py     v3 16 창 · v11: 문맥 속도 · 가속 · 시작 위치 · 외형이 p 의 미래를 바꾸나 (궤적 bootstrap 회귀 · η²) + 정체 유지 → RollOutV3/figures/context_to_future/ (2026-09-26)
  plot_train_readout.py         정체 자를 **자기 학습셋 held-out test** 에서: ROC · 정체 · 위치 오차 · 혼동, 튜블릿별 · 무대 가족별 · 화면 위치별 → RollOutV3/figures/train_readout/ (2026-09-26, new_archive_redraw 등록)
  plot_v11_readout_overlay.py  v11 정지/flat/ramp × 가림 없음·문맥 끝 k1–4: p 가 '있음' 일 때 읽힌 위치를 빈 장면 프레임 위에 (색 = 튜블릿), 진실 궤적 · 마지막 본 자리 → figures/v11_readout_overlay/ (2026-09-26)
  new_archive_redraw.py         **자 하나로 RollOutV3/figures 그림 여섯 폴더 다시 그리기** (2026-09-25, 이름은 옛 new_archive 시절 그대로): check → bias → extract (이 자로 안 읽힌 readings 만, v3 GPU 6 + v11 GPU 2 동시, v3 프레임 캐시 예열) → figs (옛 그림은 도장 보고 _superseded/<자>_<지문>/ 로)

analysis/       산출물·토큰 캐시 기반 분석. **전부 GPU 불필요**
  check_prefix_predictor.py   **`kind: prefix` predictor 구조 검사** (Ariel block-causal 체크포인트).
                              mask 의미 + **인과 불변**(미래 slot 을 잘라도 앞 slot 출력 불변) +
                              prefix vs default 출력 차. 모델 로드 없음, 난수 문맥. 시작점
                              `z_training/ariel/README.md` (2026-09-23)
  check_oracle.py             **Garrido 하네스 기준값 검사.** IntPhys1 dev x ViT-H = 88.89 (skip2_w32,
                              Filtered, 160/180). config 격자·창별 모델 생성·하네스 값·per_window 독립 재계산까지
                              29 항목. **하네스/config/로더를 건드리면 이것부터 돌린다** (2026-09-21)
  garrido_rescore.py          공식 축약 규칙으로 **재실행 없이 다시 채점**. IntPhys=Filtered(min),
                              GRASP·InfLevel=property 마다 최고 C. 창 16/32 실행을 합쳐 A.8 최고를 고른다 (2026-09-21)
  intphys2_grid_best.py       IntPhys 2 — 창 16/32/48 실행을 **하나의 격자**로 합쳐 열 (Easy/Medium/Hard/Overall) 마다 최고.
                              C 는 공식 비율 (창 × ¼…⅞) 로 제한. pair_acc 는 intphys2_column_best 것 (2026-09-27)
  bench_table.py              위 재채점 결과로 z_research/Benchmarks/README.md §4 표를 생성 (--readme) (2026-09-21)
  report.py                   summary.json 검증 -> report.json (그림·문서의 단일 입력)
  token_surprise_background.py  배경 토큰 Δ 검사 — 픽셀 차 구간별 분해 · 쌍마다 배경/물체 부호 일치 -> token_surprise/background_checks.md (2026-10-07)
  token_surprise_maps.py      surprise 채점을 토큰별로 다시 (같은 _resolved.yaml · forward) → <run>/token_surprise.npz. 토큰 평균 = per_block.json 검증 · 쌍 판정 대조 (2026-10-07)
  gravity_retrieval.py        gravity_realistic — 미래 넷 (arc · rise · float · line) 중 p 가 가장 가까운 것 (argmin surprise), 쌍 정확도 · 순위 (2026-10-07)
  v11_realistic_tables.py     v11_realistic · ledge 채점표 (조건 × 방향 · k · 물체, 원본 v11 vanish 대조). summary.json 과 overall 대조 (2026-10-06)
  merge_probe_runs.py         쪼개서 제출한 probing job 을 합침. val_video_ids 가 다르면 죽는다
  probing_md.py               RESULTS_*.md 의 probing 절(표 G-J)을 재생성
  knockout_md.py              knockout results.json -> 축별 markdown 표. `--write` 로 ablation README 전수 기록 11 교체
  rollout2_test_readout.py    위치 readout 공통 설정 (ROLLOUT2_TRAIN=v5 학습셋·캐시·결과 경로) + spatial p test
  rollout2_fit_readout.py     학습셋 p 공간평균 → (1281, 2) OLS 자 (spatial_pooling/p/fit/w_p.npy)
  rollout2_attn_readout.py    쿼리 1개 attentive 자 (3,842 파라미터), 학습셋 p 로 학습 → v2 test (attentive_pooling/p/). GPU 샤딩 Loader
  rollout2_encoder_readout.py 같은 자를 encoder z (context 32 frames) / h (target) 에 (--encoder z|h)
  rollout2_distance_decay.py  p 물체다운 토큰의 거리 × 슬롯 분해: 자 attention + 자 없는 검사 (cos 템플릿) -> RollOutV2/exp_results/v5/locality/distance_decay
  rollout2_predictor_locality.py  릴리즈 predictor attention hook (진실 칸 query 의 문맥 조회), 공간 반경 knockout, z/h 문맥 끝 속도 ridge -> v5/locality/report.md
  rollout2_ceiling.py         v2 30% p-fit 천장 (spatial)
  rollout2_probe_pos_imp.py   ledge/wall pos·imp attentive probe: h (32 frames) 학습 → [z16 ; p] test (exp_results/probe_pos_imp/)
  v11_readout_attn_diag.py    v11 에서 p 자 attention 이 슬롯별로 어디에 실리는가 (물체/마지막관측/가림막 3×3, 균등 읽기 거리) → v11_vanish/timing/attn_diag
  v11_token_object_test.py    자 없는 검사: 진실 물체 칸에서 |p−h_imp|/|p−h_pos| (>1 = p 에 물체 있음) → v11_vanish/timing/token_test
  rollout2_two_futures_attn.py  ledge/wall 슬롯별: 가능/불가능 궤적 attention 질량, 읽기−기본값 거리 → figures/<train>/summary/two_futures_attn
  v11_position.py             v11_full 미래 8슬롯에서 물체 위치·속도·가속 readout (RollOut 측정의 v11 이식, 조건별 + 공유 fit)
  ctxenc_time_alignment.py    시간 정렬 행렬 D[i][j] = |p_i − LN(h_j)| (물체 3×3 합집합), z–h 대조군·null gate → context_encoder_analysis/exp_results/time_alignment/ (2026-09-15)
  ctxenc_boundary_kinematics.py  경계 튜블릿 (t=7) 물체 3×3 토큰에서 속도·가속 ridge readout (z16 = predictor 입력; 지름길·셔플·h·p 슬롯 0 대조) → ctxenc_boundary_kinematics/
  ctxenc_boundary_identity_occlusion.py  경계 창 B7 / 튜블릿 전체 / 문맥 전체 에서 shape·color ridge; self · visible→late · visible→early 이식 (가림막 존재 vs 가려짐) → ctxenc_boundary_identity_occlusion/
  ctxenc_subspace_overlap.py  z/p/h 주각 겹침 (k 16/64) · token-matched CKA · Procrustes/ridge p→h 정렬 (visible fit → late 이식) → ctxenc_subspace_overlap/  (CLAUDE.md §10 1번)
  ctxenc_time_alignment_gated.py  위 행렬에 물체 특이 gate (Δc = c_S − c_BG, 시점별 S_j, null A/B) → ctxenc_time_alignment_gated/
  ctxenc_accel_crosspair.py   flat_v × flat_a 교차 쌍으로 경계 토큰의 가속 검정 (정지 대조 쌍이 갈려 시나리오 지름길로 무효) → ctxenc_accel_crosspair/
  ctxenc_boundary_identity_v2.py  E2 확장: ramp 가족·concat 9 토큰·가림막 제외 평균·역이식·bootstrap → ctxenc_boundary_identity_v2/
  ctxenc_subspace_v2.py       E4 확장: ramp·per-k·z→h 대조 + **정렬 사상 걸고 matched-pair 채점 재계산** → ctxenc_subspace_v2/
  ctxenc_state_subspaces_kinematics.py  경계 토큰의 위치·속도·가속·시나리오·시각 부분공간 겹침, 물체 창 vs 배경 창, 채점 거리 분해 → ctxenc_state_subspaces_kinematics/
  ctxenc_state_subspaces_v11.py  같은 것을 v11 정체성·가림막·시각 서명으로 (스크립트만, 미실행)
  ctxenc_direction_probe.py   실제 영상 (ssv2_VP · ntu_direction) 에서 문맥 encoder 의 좌/우 운동 방향 — 균등 32 장 → attentive probe 를 **정방향으로만** 학습하고 **역재생** clip 으로 시험 (뒤집힘 비율 = 라벨 없는 검사). 캐시 없음 → context_encoder_analysis/exp_results/encoder_temporal_dynamics/ (2026-09-20)
  ctxenc_direction_sbatch.sh  위 스크립트의 SLURM 런처 (vll3, GPU 1 장; 영상이 그 노드의 /local_datasets 에 있다). ANA_RUN=1 로 자기제출 방지
  rollout2_presence_readout.py  위치 + **있음/없음 (presence)** 을 같이 내는 자 (**attn** 하나로 확정, 2026-09-23) — RollOut_v2_training v6 (빈 장면 288 clip) 로, 디스크 캐시 없이 /dev/shm 에 p·z·h 를 뽑아 학습 → RollOutV3/exp_results/presence/. 기각한 후보 6 종은 RollOutV3/Archive/READOUT_CHOICE_2026-09-23.md
  rollout2_presence_sbatch.sh   위 스크립트의 SLURM 런처 (vll5 8 GPU, /dev/shm 122 GB, 죽어도 지우는 trap). ANA_RUN=1 로 자기제출 방지
  presence_readout_compare.py   위 자의 **존재 판정 오류와 좌표 오류를 따로, 그리고 교차해서** 본다. `--bias` 는 attn 의 좌표 치우침을 v6 test 에서 재어 attn_bias_px.json 에 남긴다 (쓰는 쪽이 빼는 값)
  rollout3_window_readout.py    RollOut_v3 **16 개 (문맥, 예측) 창**에서 p·z·h 의 위치·존재를 읽는다. 창마다 모델을 다시 짓는다 (window_size → RoPE 격자) → RollOutV3/exp_results/windows/readings.npz (2026-09-23)
  plot_readout_errorbars.py     (figures/) 자의 존재 판정·위치 오차를 v6 / v3 / 튜블릿별 recall 세 장으로 + ERRORS.md
  rollout3_paths.py             RollOutV3 경로·자 지문을 **한 곳에서** (R3_DECODER / R3_OUT). readings 는 지문이 다르면 check() 가 막는다
  rollout3_doc_numbers.py       RollOutV3 README · IDENTITY_R8 문서의 **모든 표를 산출물에서 다시 찍는다** (CPU 몇 초, 2026-09-26). 문서 수치를 고치면 이것과 대조
  rollout3_rerun.sh             자를 새로 배운 뒤 v3 결과 전부를 다시 (치우침 → 16 창 → 표 → 그림 → GIF). R3_OUT 필수
  rollout3_behavior_metrics.py  predictor 행동 성적표 초안 (H · T50 · F · s · g + z 로 자 검증). 지표 확정 전 → exp_results/windows/BEHAVIOR.md
  rollout3_decoder_compare.py   같은 v3 에서 옛 자 vs 새 자 (각자 자기 문턱·치우침) → exp_results/<새 readings>/DECODER_COMPARE.md
  rollout3_cross_head.py        남의 자를 p 에 걸어 본다 (이식) — '자가 고장 났나' 를 가른다 → windows*/CROSS_HEAD.md
  rollout3_window_summary.py    위 읽은 값으로 **자가 그 창에 전이됐는지** 표를 낸다 (z·h 가 자의 천장, v3 는 음성 0 개라 '없다' 비율이 곧 오작동률) → windows/WINDOW_SUMMARY.md
  pft_ruler_direct.py         post-FT / 릴리즈 predictor 를 캐시 없이 forward 해 p 에 v5 위치 자를 건다 (RollOut_v2 + v11 vanish) → predictor_training/predictor_IntPhysGenV11_PFT/exp_results/
  alpha_amplify.py            증폭 개입의 천장 (--anchor mu|z).  기각된 개입 (천장 51~65%)
  concept_separability.py     Fisher / ridge / 개념 벡터 정렬 (--align)
  confusion_vs_surprise.py    probe confusion x 채점 방향 상관
  is_p_just_context.py        p 가 문맥의 복사인가
  pca_spectrum.py             주성분 스펙트럼
  step_direction.py           걸음의 방향
  typicality.py               전형성 가설 (기각됨)
  intphys1_direction_audit.py IntPhys1 방향 균형 감사
  occlusion_intphys1.py      (CPU 3 분) IntPhys1 사건 물체 가림 통계 — 두 possible 영상 mask 를 픽셀 단위로 맞대 원리별 (O1 있음/없음 · O2 대칭차 · O3 영상별) 가시성, 쌍마다 사건 가림 길이 (초 · skip2/5 장수) · 가려진 동안 이동 (물체 폭) · 사건 전 미관측 · V-JEPA 2 정답 → OcclusionStats/intphys1/ (2026-09-26, 정본)
  intphys1_occlusion_stats.py (CPU) ⚠️ 대체됨 → occlusion_intphys1.py. IntPhys1 사건 물체가 몇 프레임 가려지나 — 두 possible 영상의 mask 면적 목록 비교 (파라미터 없음), 그룹별 raw · skip2 · skip5 가림 길이 → Benchmarks/exp_results/intphys1_event_position/occlusion_stats.json (2026-09-25)
  intphys1_event_position_acc.py (CPU) IntPhys1 Garrido 창마다 pos/imp 첫 픽셀 차 (PNG) 가 context 안 / 예측 구간 안 / 창 뒤 중 어디인지 분류 → 분류별 창 정답 · Filtered 가 고른 창의 분류 · 예측 구간 안 창만 쓴 점수 (V1) · p vs copy (H6h 원자료) → Benchmarks/exp_results/intphys1_event_position/, 문서 Benchmarks/Archive/INTPHYS1_EVENT_POSITION_2026-09-25.md (2026-09-25)
  audit_v11_identity.py       v11 정체 probe · 풀링 기하 감사 (2026-09-25): 궤적 반복 · twin/장면 hold-out · logistic vs 닫힌 해 ridge 재도출 · 복사 기준선 (z_t7/z_all) · 부분공간 block bootstrap/순열 → RollOutV3/audit/v11_identity/AUDIT.md
  audit_v3_motion.py          v3 운동 주장 궤적 단위 감사 (2026-09-25): 복사·등속 연장·지연 추적 기준선, 법칙 안 변위 기울기, 자 끌림 통제, p@h 대비, 짝/홀 교대 → RollOutV3/audit/v3/ (CPU ~2 분)
  audit_v11_presence.py       v11 presence 감사 (2026-09-25): extract = 문맥만 본 복사 기준선 (GPU) · analyze = 빈 장면 귀무·h 검사·(궤적,k) CI·자 없는 vanish 비교 (CPU) → audit/v11_presence/
  audit_training_v8.py        자 학습셋 (training_v8) 감사 (2026-09-25): 라벨 산술 · 음성 출처 (가장자리 잘림 35 %) · 판 자세↔라벨 MI · split 궤적 누수 · v11 가림막 vs 학습 판 (재질·깊이·크기·자세) → audit/training_v8/ (CPU 수 초)
  audit_training_set.py       **렌더된 자 학습셋 감사** (--root <세트 폴더>, CPU 수 분): 구조 · 방해물×물체 빈 칸 · 빈 장면 짝 · 물체/빈 전용 자세 · 라벨 · 튜블릿 P(있음|방해물) · 가중치 · v11 판 비교 · 누수 R² · 커버리지 · 가림 열 → audit/training_sets/<세트>.json
  rollout2_identity_readout.py  위치 + 정체 (모양×색 56 조합 + 없음, 57 분류) 자 학습 (2026-09-25). --launch 는 p·z·h 를 GPU 2 장씩 동시에, --name 으로 자 폴더 exp_results/<이름>/, 끝나면 치우침까지 (약 6.5 분)
  decoder_bias.py               자 폴더의 좌표 치우침 attn_bias_px.json — presence 자 · 정체 자 공통 (test × 양성 평균 pred − truth)
  v11_panel_false_alarm.py      v11 빈 장면 '있음' 오탐이 가림막 픽셀을 가리키는가 (CPU) → audit/training_v8/panel_false_alarm_<자>.json
  v11_readout_online.py       (GPU) v11 가능 clip 에 자 (p·z·h, p@h) 를 걸어 readings.npz (--timing mid) → RollOutV3/exp_results/v11{,_mid}/
  v11_pooled_features.py      (GPU) v11 튜블릿 평균 풀링 z/p/h → cache/v11_pooled_vith/ ; v11_pooled_probe.py (정체 선형 probe) · v11_pooled_geometry.py (풀링 기하) → figures/v11_{probe,geometry}/
  retrieval_confusion.py      block 밖 7-way retrieval. **기각됨** — 최상단 주석을 읽을 것
  retrieval_pooled.py         위의 pooled 판. 같이 기각
  training_effects_run.sh     TrainingEffects: 14 모델 × (v11_split_test surprise_c16t32 · IntPhys1 창 16/32) 표준 하네스 채점을 GPU lane 별로 (run.sh 를 감쌈)
  training_effects_scores.py  (CPU) 그 산출물 (per_block · per_window) 에서 v11 · IntPhys1 점수표를 다시 계산 → TrainingEffects/scores/SCORES.md (block bootstrap CI, 방향별)
  training_effects_queue.sbatch  TrainingEffects 추출을 이어 도는 SLURM 큐 (vll5 8 GPU, 모든 단계 resume)
  te_intphys1_score.py        **(GPU 추출 + CPU 채점)** TrainingEffects 그룹 A — IntPhys1 dev Garrido 격자 (skip2_w16·skip2_w32·skip5_w16, 창 140/영상) 에서 predictor 여러 개 (--preset final|curves) 를 encoder 한 번으로: 창×튜블릿 L1 vs 표준 표적 h · 인과 표적 h^c (H6e 정의), 복사 LN(z)_문맥끝 기준선. 영상별 shard (이어 돌리기) → /data2/.../cache/training_effects/ip1score_<preset>/, --score (garrido_rescore 함수 그대로 → 칸별 macro) · --validate (공식 per_window · H6e) · --selftest (로더 동치 + H6b/H6e 비트 재현) → TrainingEffects/ip1score/ (2026-09-25). 2026-09-27: Ariel 태그 ariel_prefix_ep45 · ariel_full_ep40 · ariel_ar_ep18 (옛 ariel_ep* 는 파일 지워짐), kind=ar 는 블록별 표적 (ar_scoring) 으로 p_std · copy_blk, p_causal=NaN
  te_v11_readout.py           TrainingEffects: v11 (visible·late·mid 가능 16,128 clip) 에서 **predictor 여러 개** (--preset final|curves, kind 는 ckpt arch) 의 p@p·p@h 읽기 (x,y,logit,top1,진실 3×3 질량) + z·h 자 검사 + 튜블릿 평균 풀링 z/h/p_tag. kind=ar 는 거부 (2026-09-27). encoder 는 clip 당 한 번, 재개 가능 memmap → cache/training_effects/v11readout_<preset>/ (GPU 필요, 2026-09-25)
  te_v3_readout.py            TrainingEffects: rollout3_window_readout.py 를 **predictor 목록** (--preset final|curves, --preds tag=path; kind 는 ckpt arch) 으로 일반화. RollOut_v3 가능 2,744 clip (kind=ar 는 거부, 2026-09-27), C16/P16·C16/P32 (+--c8) 에서 p_<tag>·p@h_<tag>·p@z_<tag> + z·h 를 (x,y,logit,top1,GT 3×3 질량) 로. encoder 는 clip·창당 한 번, 재개 가능 memmap, 내장 검사 (release 동치 · prefix 창 독립) + --validate (정본 readings 대조) · --check-loader (build_from_config 동치) → cache/training_effects/v3readout_<preset>/ (GPU 필요, 2026-09-25)
  te_analyze_identity.py      TrainingEffects: v11 정체 (읽기) · 배치 — v11_pooled_probe.py · v11_pooled_geometry.py 를 **predictor 마다** (코드 수정 없이 V11P_CACHE) 다시 돌려 릴리즈와 나란히. prep (v11readout_curves pool_p → cache/training_effects/v11pooled_<tag>/, z·h·meta 는 릴리즈 symlink) · probe (원 main + clip 별 정오 기록, GPU) · geometry (GPU) · ridge / curves / report (CPU): p 무관 잎 = 릴리즈 확인, v11_split test 부분집합, 짝 block bootstrap, 부분공간 bootstrap → TrainingEffects/v11_identity/IDENTITY.md (2026-09-25)
  te_v11_score.py             TrainingEffects: v11_split_test 21,504 clip 을 surprise_c16t32 그대로 **predictor 여러 개** (--preset final|curves|own, --preds tag=path; kind 는 ckpt arch) 로 채점. clip × predictor × 미래 슬롯 8 의 l1 (슬롯 평균 = 표준 surprise) + 복사 기준선 copy = |LN(z)_7 − h_t| + hcopy·hstep·pz·pstep. 그룹 B (--model vitb_*/vittiny_*) 는 own. 2026-09-27: Ariel 태그 갱신 (ariel_prefix_ep45 · ariel_full_ep40 · ariel_ar_ep18), kind=ar 는 블록별 LN(target) 공간 l1 + AR 전용 copy_blk (쌍 정확도만 비교). encoder 는 clip 당 한 번, 재개 가능 memmap + done 플래그, 끝나면 matched-pair summary → cache/training_effects/v11score_<model>_<preset>/. --validate: p 비트 대조 (build_from_config) + 하네스 per_video_surprise 대조 → TrainingEffects/v11score/ (GPU 필요, 2026-09-25)
  te_analyze_scores.py        (CPU) te_v11_score.py 배열 (done 행만) 분석: 하네스 재현 검증 · 복사 기준선 (p − copy, 그룹 B 는 own copy) · 학습 곡선 (e0 = release) · 슬롯별 · 드러남 단계 · '예측한 변화량' (pz/hcopy, pstep/hstep) · Δ vs release 칸 목록. block bootstrap (칸마다 공유 resample = 짝지은 차이) → TrainingEffects/scores/{ANALYSIS.md, analysis.json, fig_*.png} (2026-09-25)
  te_analyze_v3.py            (CPU) v3readout_curves (te_v3_readout.py) 의 RollOut_v3 운동 — 지속 (튜블릿별 '있다' · '모임' (+진실 3×3 질량) · T50, C16/P16·C16/P32) · 전이 (법칙별 부호 오차 · 이득 β · 복사/등속 연장 기준선 · 변위/위치 귀무 · law−flat_v 대비 · 이득 비 ρ · y 치우침 보정) · 도달 거리 (flat_v reach · plateau · 시간/거리 상관) · 학습 곡선, predictor 22 개. p@h 주 · p@p 보조, 두 자 견고 판정, 법칙 층화 궤적 bootstrap (짝지은 Δ, 공통 '모임' 칸) + release sanity (정본 readings · values.json · CROSS_HEAD) → TrainingEffects/v3_motion/{MOTION.md, TABLES.md, motion.json, fig_*.png} (2026-09-25)
  te_analyze_presence.py      (CPU) v11readout_curves (done 행, v11_split test 절반) 의 있음 읽기 — 영속 (late k≥2 t1..t4 present − 빈 바닥 · AUROC) · 지속 (visible 읽을 수 있는 칸 t3..) 을 predictor 별 (p@h 주 · p@p 보조 + h@h · z@z · 복사 참조) · 학습 곡선 · 위치 ('있다' ∧ mass3 > 0.035 의 L2 · 부호 x 오차) · vanish A/B (scores.json). (궤적,k) 두 단계 cluster bootstrap (칸마다 공유 가중치 = 짝지은 Δ) + release 전 block sanity (values.json) → TrainingEffects/v11_presence/{PRESENCE.md, TABLES.md, presence.json, fig_*.png} (2026-09-25)
  te_analyze_ip1.py           (CPU) te_intphys1_score.py 배열 (ip1score_final, 읽기만) 로 IntPhys1 그룹 A predictor 마다 **p vs 복사 기준선**: 하네스 per_window 재현 검증 (뒤집힌 쌍) · 칸 3 개 (skip2_w32/w16 · skip5_w16) × 표준/인과 표적의 p − copy · Δ vs release · O1/O2/O3 · 운동×가림 · 이동+가림 사건 튜블릿 pre/e0/post (H6e/H6d 정의) · 그룹 B 하네스 표 · 학습 곡선 (curves 배열 또는 하네스 점). principle 층화 scene bootstrap (cluster 집합마다 공유 resample) → TrainingEffects/ip1_copy/{IP1_COPY.md, ip1_copy.json, fig_*.png} (2026-09-25)

slurm_logs/     .gitignore
```

## 새 스크립트를 만들 때

1. **위 네 폴더 중 하나에 넣는다.** 최상위에 새 파일을 두지 않는다.
   맞는 폴더가 없으면 폴더를 새로 만들고 이 README 에 한 줄 추가한다.
2. **docstring 첫 줄에 무엇을 재는지, 마지막에 실행 예시**를 적는다.
3. **수치를 내는 스크립트는 산출물과 대조 검증**한다 (`figures/` 는 전부 그렇게 한다).
4. 임시 출력은 레포 루트에 흘리지 말고 `/tmp` 나 스크래치패드에 쓴다.

## 그림을 새로 만들 때

`plot_v11_surprise.py` 는 상단 **`FIGDIR`** 표로 하위 폴더를 배정한다.
파일만 옮기면 다음 실행에서 도로 흩어지므로 **표를 같이 고친다.**
대체된 그림은 지우지 않고 `_superseded` 로 보내고, `figures/surprise/README.md` 에
**왜 물러났는지** 적는다.

## 실험 케이스마다 프로토콜 yaml 을 만들지 않는다

`configs/protocols/README.md` 참고. 요약:

- `fit_groups_sweep: auto` — 데이터의 group 을 읽어 sweep 자동 생성
- `SET="a.b=1 c.d=null"` — 병합된 config 를 점 경로로 덮어씀 (`null` = 키 삭제)
- `extends:` — **재는 방식 자체가 다를 때만** 새 yaml

_superseded/     RollOut_v1 (프레임·캐시·결과 삭제) 과 2026-09-10 이전 RollOut_v2 pipeline (pooled p-fit · token decoder · 옛 궤적 그림) 스크립트. 실행 불가, 기록만.
