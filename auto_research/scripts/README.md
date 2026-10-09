# auto_research/scripts

| 파일 | 무엇 | GPU |
|---|---|---|
| `arlib.py` | 공용: config 로드, bundle 빌드, **층별 forward** (문맥 encoder 블록 · predictor 블록 · 렌즈 · LN(h)), 토큰 인덱스 | — |
| `validate_arlib.py` | arlib 검증 3 종 (hook p 비트 동일, 렌즈 L11 = p, 표준 채점 surprise 재현) | 1 |
| `stage_vll6.sh` | vll6 `/data2` 에 v11 · IntPhys1 dev · RollOut_v2 풀기 (sbatch, CPU) | 0 |
| `stage_vll6_intphys1_masks.sh` | IntPhys1 dev 의 masks/ · status.json 추가 | 0 |
| `h5b_copy_vs_extrapolate.py` | H5 개정: 되먹임 팔의 위치 귀무 · stay/copy/cv 기준선 · T_j vs T_{j−1} 착지 · along-track 기울기 · 속도 3 분위 멈춤 자리 · 물체다움 문턱 p10/p25/p50 (h5.npy, CPU, clip bootstrap CI) | 0 |
| `h8b_fixed_threshold.py` | H8 개정: knockout 명세별 물체다움을 공통 고정 문턱 (clean_null 슬롯 0–2 · 슬롯 0 P10) 으로 · 명세별 문턱 분해 · 적중−귀무 · 마지막 칸 적중 · 봉우리 자리 (길위 귀무) · L1/cos (h8.npy, CPU, clip paired bootstrap CI) | 0 |
| `h9c_bootstrap_v11.py` | H9 적대 검증: 개입 변형 − clean 의 block bootstrap CI (timing×운동×위반×방향, 슬롯별), H6f 복사 기준선 대조, Holm → `exp_results/h9/h9c_bootstrap_v11.{json,txt}` (vll6 CPU) | 0 |
| `h9d_intphys1_bootstrap.py` | H9b IntPhys 1 개입 채점 (h9b_analyze 와 같은 Garrido 규칙) + 원리 층화 장면 bootstrap CI → `exp_results/h9/h9d_intphys1_bootstrap.json` | 0 |
| `h6g_intphys1_cell_ci.py` | H6 적대 검증: IntPhys 1 Garrido 칸 3 개 × 표준/인과 표적 × p/복사, p − 복사 · p − chance (principle 층화 scene bootstrap CI), 그룹·방향·창 j 표·튜블릿 사건 전/분기/후 · H6d (tie 0.5) · 공식 하네스 대비 편차 → `exp_results/h6/h6g_intphys1.json` (NFS 입력, CPU) | 0 |
| `h6g_v11_cell_ci.py` | H6 적대 검증: v11 H6f 12 칸 × 표준/인과 × p/복사, p − 복사 · p − chance (block bootstrap CI), A/B · 이동 합침 · late 재등장 전/재등장/후 슬롯 → `exp_results/h6/h6g_v11.json` (vll6 CPU) | 0 |
| `h6g_v11_hidden_pixel_check.py` | H6 적대 검증: v11 late 가려진 미래 프레임이 pos/imp 에서 픽셀 단위로 같은가 (300 쌍) → `exp_results/h6/h6g_v11_hidden_pixels.json` (CPU, /local_datasets 필요) | 0 |
| `h4b_threshold_invariant.py` | H4 적대 검증: 단일 · joint · 위치 이동의 물체다움을 공통 문턱 (single 문턱) · raw 대비 짝 비교 · 두 템플릿 (진실 시각 / 마지막 관측) · 진실 적중 · L1 로, 법칙 안 속도 3 분위 · 법칙 FE 선형 확률 모형 (거리 β / 시간 γ) · 진행률 무조건 중앙값 (h4.npy, CPU, 궤적 84 군집 bootstrap CI) → `exp_results/h4/h4b_threshold_invariant.{json}`, `h4b_summary.txt` | 0 |
| `h23b_null_objectness_testA.py` | H23 적대 검증 (1/2): 행 단위 정확 귀무 (같은 법칙 다른 clip 진실 칸, 전쌍) · 공통 문턱 (θ_L6 · θ_L11) 층별 물체다움 · 법칙별 진행률 vs 등속 외삽 · 검정 A 복사 LN(z)_T/h_T 기준선 · 쌍 갈림 프레임 (vll6 CPU, loc/l1/vec) → `exp_results/h23b/{rows_null.npz, objectness.json, progress_law.json, testA_copy.json}` | 0 |
| `h23b_regress_tables.py` | H23 적대 검증 (2/2): 귀무 보정 회귀 (logit 귀무 공변량 · 적중−귀무 OLS · C 범주 · t·d 교호, clip bootstrap) · 적중−귀무 층×슬롯 · 거리×슬롯 (clip·법칙 수) · P32 문맥 곡선 · 공통 문턱 L11−L8/L6 CI · A−Ash 층별 · 법칙별 S/B (NFS 입력, CPU) → `exp_results/h23b/{h23b.json, REPORT_h23b.md}` | 0 |
| `h7b_layer_scoring_splits.py` | H7 적대 검증: v11 층별 렌즈 채점 12 층 × timing × motion × violation × A/B 전 칸, 층 차이 block 군집 bootstrap CI · held-out 층 선택, 입력 없는 기준선 (풀링 mean-p · block 안 A/B 분해 X1/BC/BW · H6f 복사), target encoder 문맥 번짐 (vll6 CPU, l1/lens/h) → `exp_results/h7/h7b_scoring_splits.json`, `h7b_scoring_tables.md` | 0 |
| `h7c_centered_transfer.py` | H7 적대 검증: 풀링 ridge 비가림 → early/mid/late 이식을 raw · 영역 평균 중심화 · 영역 z-score 로 (p 렌즈 12 층 · LN encoder 8 층 · h, 8 슬롯 평균 포함), p 특이 격차 (h−p, z−p) block bootstrap CI, 팔 안 이식·self (vll6 CPU) → `exp_results/h7/h7c_centered_transfer.json` | 0 |
| `h1c_traj_split.py` | H1 적대 검증: h1b 를 **궤적 단위 split** (group = law\|primary\|secondary, split · λ-CV 모두) 으로 — xfit (법칙당 12 궤적 학습, 궤적 bootstrap CI) · half (7 궤적 × 20 split) · 학습 곡선 (`--curve`), 기준선 B_state · B_entry · B_xfirst · B_kin3 ([RBF+선형] 커널), 법칙 분류 + 구간 최빈 기준선 (vll6 CPU, v3_h1) → `exp_results/h1/h1c_traj_C{8,16,32}.json` | 0 |
| `h1d_p_side_v3.py` | H1 적대 검증 p 쪽: (b) 같은 (v_T, x_T) 에서 p argmax 변위의 법칙 효과 / 참 가속 효과 (flat_v 기준 모형 lin·int·quad·공통 지지, 궤적 bootstrap), 칸 적중 + 다른 궤적 귀무 (c) 같은 창 p 미래 토큰 @W_T vs LN(z)_T@W_T 궤적 단위 probe (vll6 CPU, v3_h23 loc · vec) → `exp_results/h1/h1d_p_side.json` | 0 |
| **2026-09-25 (encoder 충분 · predictor 병목 인과)** | | |
| `sub.sh` · `srun6.sh` · `run4.sbatch` · `run8.sbatch` · `chain_0925{a,b,c}.sh` | 제출 도우미 (SLURM_* 제거) · 묶음 실행 (스모크 통과 뒤 본 실행) | — |
| `m2_boundary_patch_v11.py` → `m2_analyze.py` | M2: v11 late↔early 쌍둥이 (metadata 내용 키) 의 경계 m 튜블릿 교체 · 역사 삭제 6 팔 → 채점 · 물체 신호 (템플릿 = early pos h 진실 칸, 고정 문턱) · 닫힘 (late block 군집 bootstrap) | 8 |
| `p2_stride_sweep_v3.py` → `p2_analyze.py`, `p2b_units.py` | P2: v3 stride 1·2·4 팔 5 개, 적중−귀무 · 물체다움 · 외형 템플릿 (p / h 천장) · 복사 대비 → 절벽 단위 (슬롯 · raw 시간 · 거리) 변동계수 · 속도 3 분위 · 두 항 모형 | 8 |
| `p3_closure_v3.py` → `p3_analyze.py` | P3: 닫힘 검정 — 같은 사상 (채널 A · 능형 W) 으로 진짜 관측 (인과 표적 hc) 대 자기 예측 되먹임, 착지 · 기울기 · 사상 품질 | 8 |
| `m13_memory_state_v3.py` → `m13_analyze.py` | M1 (유효 기억: 역사 삭제 · 다른 궤적 역사 교체) · M3 (속도 R²: z_last · h_last · h_obj · p_obj, 궤적 group CV) · 충분성 (z 에서 읽은 속도로 등속 외삽 vs p) | 8 |
| `m4_locality_stride_v3.py` → `m4_analyze.py` | M4: stride 별 진실 칸 미래 질의의 attention (물체 궤적 · 마지막 물체 · 자기 기둥) × 변위 거리 · 미래→문맥 반경 r knockout (hook = rollout2_predictor_locality 의 것, 재현 0.0) | 8 |
| `m5_steer_velocity_v3.py` → `m5_analyze.py` | M5: z 경계 물체 토큰을 능형 속도 방향으로 조향 → p 물체 이동 이득 (무작위 방향 대조) | 8 |
| `e_ssv2_extract.py` → `e_ssv2_analyze.py` | E-SSv2: SSv2 좌/우 실영상 — 방향 probe (정+역재생 학습) · 시간 정렬 (encoder 대조 · 귀무 · 양성 대조 oracle) · 복사/흐림 기준선 · 변화 지도 | 8 |
