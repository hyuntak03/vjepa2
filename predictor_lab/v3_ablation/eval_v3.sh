#!/bin/bash
# BlenderTrack v3 ablation 채점 (2026-11-01) — 패널 여섯 × predictor 여럿 + 복사 기준선을 transfer_gap_te.py 한 번 통과로.
#   패널 = 시트 trianing_ablation 의 열:
#     v3test   v3 VoE 시험 1,719 장면 · 3,438 짝 (data_csv/blender_track_v3/voe_test, mp4 64 장 → stride 2 로 32 장)
#     vob_late VOBench_v1 split2 test late 108 장면 (permanence 36 장면 = 72 짝, 나머지는 모양 · 색 바뀜) — 표에서 permanence 만 고른다
#     ip1uni   IntPhys 1 dev late (균일 창) 180 짝 — 표에서 가림 (occluded) 90 짝만
#     ip2uni   IntPhys 2 Main late (균일 창) 500 짝
#     clevrer  CLEVRER 식 permanence 792 짝 (static · behind · cross · overtake)
#     rubric   Rubric office 실영상 40 짝 (uniform 창)
#   사용: PREDS="release=…/latest.pt,v3_mix1600_s0=…" OUTSFX=b1 GPUS=2 bash predictor_lab/v3_ablation/eval_v3.sh
#         PANELS="v3test rubric" 로 패널을 고를 수 있다. 결과 predictor_lab/results/v3_ablation/te/<패널>_<OUTSFX>/
#   PY 기본은 envs/vjepa2 (sm_90 이하). vll7 (Blackwell) 에서는 PY=/data/hyuntak/anaconda3/envs/vjepa2_bw/bin/python
set -uo pipefail
R=/data/hyuntak/project/2026/2027_cvpr/vjepa2
cd "$R"
P=${PY:-/data/hyuntak/anaconda3/envs/vjepa2/bin/python}
# ⚠️ sbatch --export 는 쉼표가 변수 구분자라 PREDS (쉼표 목록) 가 첫 항목에서 잘린다 (11-01 b1 에서 실제로 release 만 채점됨).
#    sbatch 로 낼 때는 PREDS_FILE (한 줄에 tag=path 하나) 로 넘긴다.
if [[ -n ${PREDS_FILE:-} ]]; then PREDS=$(grep -v '^[[:space:]]*$' "$PREDS_FILE" | paste -sd,); fi
NFS=$R/z_pixel_decoder/runs/_frames
OUT=$R/predictor_lab/results/v3_ablation/te
mkdir -p "$OUT"
echo "[eval_v3] $(hostname) $(date '+%F %T') panels=${PANELS:-all} preds=${PREDS}"
for PN in ${PANELS:-v3test vob_late ip1uni ip2uni clevrer rubric}; do
  case $PN in
    v3test)   DS=vobench_v1;             ST="data.root=$R/data_csv/blender_track_v3/voe_test data.index_csv=index.csv data.frames_root=null window.stride=2 window.frames_start=0" ;;
    vob_late) DS=vobench_v1_split2_test; ST="data.root=data_csv/transfer_gap_r4_vob data.index_csv=index_test_late.csv data.frames_root=$NFS/VOBench_v1_256full" ;;
    ip1uni)   DS=intphys1_late_uni;      ST="data.frames_root=$NFS/tg4_IntPhys1_late_uni" ;;
    ip2uni)   DS=intphys2_late_uni;      ST="data.frames_root=$NFS/IntPhys2_late_uni" ;;
    clevrer)  DS=vobench_v1;             ST="data.root=/data/hyuntak/project/2026/2027_cvpr/testbed/cleverer_permanence_v1 data.index_csv=$R/data_csv/clevrer_perm_v1/index.csv data.frames_root=null window.stride=1 window.frames_start=0" ;;
    rubric)   DS=vobench_v1;             ST="data.root=$R/data_csv/rubric_quads_uniform data.index_csv=index.csv data.frames_root=/data/hyuntak/project/2026/2027_cvpr/nfs_cache/rubric_quads_frames/uniform window.stride=1 window.frames_start=0" ;;
    # 렌더러 대조 (11-01): VOBench 와 같은 UE 렌더 · 같은 궤적인데 가림막만 움직이는 상자 (v3 에 있는 가림 방식). 프레임 0,3,…,93 만 NFS 사본
    vomo)     DS=vo_moving_occluder;     ST="data.frames_root=$NFS/VO_moving_occluder_s3" ;;
    # 11-01 v3t (v3 + trapdoor) 시험 — v3test 와 같은 창 (mp4 64 장 → stride 2)
    v3ttest)  DS=vobench_v1;             ST="data.root=$R/data_csv/blender_track_v3t/voe_test data.index_csv=index.csv data.frames_root=null window.stride=2 window.frames_start=0" ;;
    # 11-02 v4_velocity_control 시험 — 같은 장면 · 속도 4 단계 묶음 322 개 (속도 바꿔치기 · 사라짐 · 나타남), 같은 창 (mp4 64 장 → stride 2)
    # 11-02 2 차: v4 빗면 시험 — 같은 빗면 · 속도 4 단계 묶음 (내려감 가속 · 올라감 감속/되돌아옴 × 판 · 벽 · trapdoor)
    v4vcrtest) DS=vobench_v1;            ST="data.root=$R/data_csv/blender_track_v4vcr/voe_test data.index_csv=index.csv data.frames_root=null window.stride=2 window.frames_start=0" ;;
    v4vctest) DS=vobench_v1;             ST="data.root=$R/data_csv/blender_track_v4vc/voe_test data.index_csv=index.csv data.frames_root=null window.stride=2 window.frames_start=0" ;;
    # 11-02 Real Bench = real_world_permanence (실제 영상 95 × 4 = 380 clip, 짝 190). 창 = Rubric uniform (t_splice 앞뒤 균등 16 + 16), mp4 + 영상별 프레임 번호
    #   index: z_research/scripts/data/build_real_world_permanence_index.py
    # 11-02 VOBench_v2 late permanence 216 block (짝 432; replica 0 = v1). 프레임은 vll5 /data2 원본에서 보폭 3 의 32 장만 NFS 사본 (원본 해상도)
    #   z_research/scripts/data/build_vobench_v2_late_perm.py
    vob2late) DS=vobench_v1;             ST="data.root=$R/data_csv/vobench_v2_late_perm data.index_csv=index.csv data.frames_root=/data/hyuntak/project/2026/2027_cvpr/nfs_cache/VOBench_v2_late_perm" ;;
    rwp)      DS=vobench_v1;             ST="data.root=/data/hyuntak/project/2026/2027_cvpr/testbed/real_bench/real_world_permanence data.index_csv=$R/data_csv/real_world_permanence_uniform/index.csv data.frames_root=null data.frames_column=frames window.stride=1 window.frames_start=0" ;;
    *) echo "모르는 패널 $PN"; exit 1 ;;
  esac
  t0=$(date +%s)
  # shellcheck disable=SC2086
  $P z_research/scripts/analysis/transfer_gap_te.py --name "v3abl_${PN}_${OUTSFX:-x}" --dataset $DS --set $ST \
     --preds "$PREDS" --out "$OUT/${PN}_${OUTSFX:-x}" --gpus "${GPUS:-1}" --batch 8 --workers "${WORKERS:-6}" --threads 4 \
     || { echo "FAIL $PN"; continue; }
  echo "[eval_v3] $PN 끝 $(( $(date +%s) - t0 ))s"
done
echo EVAL_V3_DONE
