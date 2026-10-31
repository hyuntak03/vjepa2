#!/usr/bin/env bash
set -uo pipefail
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
S=z_research/scripts/analysis/transfer_gap_offset.py
IFS=',' read -r G0 G1 <<< "$CUDA_VISIBLE_DEVICES"
echo "gpus $G0 $G1"
CUDA_VISIBLE_DEVICES=$G0 $P $S --phase train || exit 1
CUDA_VISIBLE_DEVICES=$G0 $P $S --phase score --sets vob_test,intphys1_dev,inflevel_continuity_s5 > z_analysis/transfer_gap/logs/E27_g0.log 2>&1 &
CUDA_VISIBLE_DEVICES=$G1 $P $S --phase score --sets v11sub,grasp_level2_s5 > z_analysis/transfer_gap/logs/E27_g1.log 2>&1 &
wait
$P $S --phase report
echo E27_DONE
