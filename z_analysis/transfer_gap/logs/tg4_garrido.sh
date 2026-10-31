#!/usr/bin/env bash
# [X2] R4 점검 — IntPhys 1 dev Garrido 칸 skip2_w32 (릴리즈 88.89 칸) 을 진단 PFT 넷 + 릴리즈로 (vll3, NFS 바이트 사본). 결론 축 아님 ("안 망가지나").
#   sbatch -p batch_vll -w vll3 --gres=gpu:1 ... z_analysis/transfer_gap/logs/tg4_garrido.sh
set -uo pipefail
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
source cvpr/env.sh
NFS=$PWD/z_pixel_decoder/runs/_frames
R=$PWD/z_training/runs
OUT=$PWD/z_analysis/transfer_gap/results/r4_x2/garrido
mkdir -p $OUT
i=0
for m in release tg4_t0_s0 tg4_t0_s1 tg4_t1_endscatter tg4_t2_genmix; do
  i=$((i + 1))
  O=$OUT/ip1dev_skip2w32_$m
  [[ -f $O/summary.json ]] && { echo "skip $O"; continue; }
  CK=$R/$m/latest.pt; [[ $m == release ]] && CK=$R/release_vith/latest.pt
  EVAL_DDP_PORT=$(( 32400 + i )) WINDOW=garrido_skip2_w32 \
    SET="data.root=$NFS/tg4_IntPhys1_dev_videos data.frames_root=$NFS/tg4_IntPhys1_dev_frame_png model.predictor_checkpoint=$CK" \
    TAG=tg4_ip1dev_$m OUTDIR=$O GPUS=1 bash cvpr/harness/launch.sh cvpr/eval/intphys1/protocol.yaml intphys1_dev vith
done
echo TG4_GARRIDO_DONE
