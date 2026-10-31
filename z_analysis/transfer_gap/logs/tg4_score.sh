#!/usr/bin/env bash
# [X2] R4 — 전이 시험판 다섯을 predictor 여럿으로 한 번에 채점 (transfer_gap_te.py; 표준 surprise 식 + 복사 기준선).
#   vll3 등 다른 노드에서 NFS 사본으로 돈다.  PANELS · PREDS · OUTSFX 를 env 로 준다.
#   sbatch -p batch_vll -w vll3 --gres=gpu:4 --cpus-per-gpu=8 --mem-per-gpu=30G --time=0-02:00:00 \
#     -o z_analysis/transfer_gap/logs/%x_%j.out -J tg4_score --export=ALL,PANELS="ip1uni v11late",PREDS=base,OUTSFX=base \
#     z_analysis/transfer_gap/logs/tg4_score.sh
set -uo pipefail
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
source cvpr/env.sh
P=$CVPR_PY
NFS=$PWD/z_pixel_decoder/runs/_frames
R=$PWD/z_training/runs
OUT=$PWD/z_analysis/transfer_gap/results/r4_x2/te
G=${GPUS:-4}
case "${PREDS:-base}" in
  base) PL="release=$R/release_vith/latest.pt,vobftg=$R/vobench_release_postft_g/latest.pt" ;;
  all)  PL="release=$R/release_vith/latest.pt,vobftg=$R/vobench_release_postft_g/latest.pt,t0s0=$R/tg4_t0_s0/latest.pt,t0s1=$R/tg4_t0_s1/latest.pt,t1=$R/tg4_t1_endscatter/latest.pt,t2=$R/tg4_t2_genmix/latest.pt" ;;
  *)    PL="$PREDS" ;;
esac
echo "[tg4_score] $(hostname) $(date '+%F %T') panels=${PANELS} preds=${PL}"
for PN in ${PANELS:-vob_late ip1uni ip2uni v11late v11real}; do
  case $PN in
    vob_late) DS=vobench_v1_split2_test; ST="data.root=data_csv/transfer_gap_r4_vob data.index_csv=index_test_late.csv data.frames_root=$NFS/VOBench_v1_256full" ;;
    ip1uni)   DS=intphys1_late_uni;      ST="data.frames_root=$NFS/tg4_IntPhys1_late_uni" ;;
    ip2uni)   DS=intphys2_late_uni;      ST="data.frames_root=$NFS/IntPhys2_late_uni" ;;
    v11late)  DS=v11_split_test;         ST="data.root=data_csv/transfer_gap_r4_v11 data.index_csv=index_test_latevanish.csv data.frames_root=$NFS/tg4_IntPhysGen_v11" ;;
    v11real)  DS=v11_realistic;          ST="data.root=data_csv/transfer_gap_r4_v11real data.index_csv=index_occ.csv data.frames_root=$NFS/tg4_IntPhysGen_v11_realistic" ;;
    *) echo "unknown panel $PN"; exit 1 ;;
  esac
  $P z_research/scripts/analysis/transfer_gap_te.py --name "r4_${PN}_${OUTSFX:-x}" --dataset $DS --set $ST \
     --preds "$PL" --out "$OUT/${PN}_${OUTSFX:-x}" --gpus $G --batch 8 --workers 6 --threads 4 ${LIMIT:+--limit $LIMIT} || { echo "FAIL $PN"; exit 1; }
done
echo TG4_SCORE_DONE
