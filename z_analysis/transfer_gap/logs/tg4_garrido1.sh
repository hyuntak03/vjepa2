#!/usr/bin/env bash
# [X2] R4 점검 — IntPhys 1 dev Garrido 칸 skip2_w32 를 모델 하나로 (MODEL env, PORT env). tg4_garrido.sh 의 한 모델 판 (병렬로 띄우려고).
set -uo pipefail
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
source cvpr/env.sh
NFS=$PWD/z_pixel_decoder/runs/_frames
R=$PWD/z_training/runs
m=${MODEL:?}
O=$PWD/z_analysis/transfer_gap/results/r4_x2/garrido/ip1dev_skip2w32_$m
[[ -f $O/summary.json ]] && { echo "skip $O"; exit 0; }
CK=$R/$m/latest.pt; [[ $m == release ]] && CK=$R/release_vith/latest.pt
EVAL_DDP_PORT=${PORT:?} WINDOW=garrido_skip2_w32 \
  SET="data.root=$NFS/tg4_IntPhys1_dev_videos data.frames_root=$NFS/tg4_IntPhys1_dev_frame_png model.predictor_checkpoint=$CK" \
  TAG=tg4_ip1dev_$m OUTDIR=$O GPUS=1 bash cvpr/harness/launch.sh cvpr/eval/intphys1/protocol.yaml intphys1_dev vith
echo TG4_GARRIDO1_DONE $m
