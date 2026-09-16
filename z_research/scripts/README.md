# z_research/scripts

**최상위에는 사람이 직접 치는 것만 둔다.** 나머지는 역할별 폴더로 내린다.

```
run.sh          표준 진입점.  GPUS=8 bash z_research/scripts/run.sh <프로토콜> <데이터셋> [모델]
sbatch.sh       위를 SLURM 으로 제출
monitor.sh      watch -n 1 bash z_research/scripts/monitor.sh [job id | 로그경로]

harness/        run.sh 가 부르는 것
  resolve.py      프로토콜 + datasets.md + models.md 를 병합. --set / extends / auto sweep

data/           인덱스·데이터 준비
  build_rollout2_index.py     RollOut_v2 인덱스 — 라벨은 metadata 가 아니라 plan 에서 (검증 15항목)
  build_probe_imp_index.py    불가능 변이를 probing 대상으로 여는 인덱스
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
  plot_v11_probing.py       v11 probing 그림.  324칸 전수 대조 후 by_condition/ by_k/ 로
  plot_v11_occtiming.py     가림 타이밍(early/mid/late) 그림.  **두 run 을 합친 report** 를 받는다
                            (v11 본체 + v11_earlymid). FIGDIR: 01_timing / 02_timing_k / 03_object_order
  plot_rollout2_readout.py  RollOut_v2 위치 readout overlay·궤적 (--pooling --rep p|z|h)
  plot_rollout2_traj_gif.py  위 traj PNG 의 GIF 판: 32 샘플 프레임 위에 정답(흰)·읽기(주황) 궤적 누적 + time bar -> <pooling>/<rep>/traj_gif/
  plot_rollout2_summary.py  RollOut_v2 fig_l2 / fig_err_xy / fig_motion_gain / fig_motion_xy (p·z·h 막대)
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
  plot_ek100_anticipation_gif.py  EK100 anticipation 과제 GIF (CONTEXT 4s / GAP 1s / ACTION 라벨 + 모델 입력 영역) -> z_research/anticipation/EK100/figures/samples/

analysis/       산출물·토큰 캐시 기반 분석. **전부 GPU 불필요**
  report.py                   summary.json 검증 -> report.json (그림·문서의 단일 입력)
  merge_probe_runs.py         쪼개서 제출한 probing job 을 합침. val_video_ids 가 다르면 죽는다
  probing_md.py               RESULTS_*.md 의 probing 절(표 G-J)을 재생성
  knockout_md.py              knockout results.json -> 축별 markdown 표. `--write` 로 ablation README 전수 기록 11 교체
  rollout2_test_readout.py    위치 readout 공통 설정 (ROLLOUT2_TRAIN=v5 학습셋·캐시·결과 경로) + spatial p test
  rollout2_fit_readout.py     학습셋 p 공간평균 → (1281, 2) OLS 자 (spatial_pooling/p/fit/w_p.npy)
  rollout2_attn_readout.py    쿼리 1개 attentive 자 (3,842 파라미터), 학습셋 p 로 학습 → v2 test (attentive_pooling/p/). GPU 샤딩 Loader
  rollout2_encoder_readout.py 같은 자를 encoder z (context 32 frames) / h (target) 에 (--encoder z|h)
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
  pft_ruler_direct.py         post-FT / 릴리즈 predictor 를 캐시 없이 forward 해 p 에 v5 위치 자를 건다 (RollOut_v2 + v11 vanish) → predictor_training/predictor_IntPhysGenV11_PFT/exp_results/
  alpha_amplify.py            증폭 개입의 천장 (--anchor mu|z).  기각된 개입 (천장 51~65%)
  concept_separability.py     Fisher / ridge / 개념 벡터 정렬 (--align)
  confusion_vs_surprise.py    probe confusion x 채점 방향 상관
  is_p_just_context.py        p 가 문맥의 복사인가
  pca_spectrum.py             주성분 스펙트럼
  step_direction.py           걸음의 방향
  typicality.py               전형성 가설 (기각됨)
  intphys1_direction_audit.py IntPhys1 방향 균형 감사
  retrieval_confusion.py      block 밖 7-way retrieval. **기각됨** — 최상단 주석을 읽을 것
  retrieval_pooled.py         위의 pooled 판. 같이 기각

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
