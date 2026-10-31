#!/usr/bin/env bash
set -euo pipefail
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
S=z_research/scripts/analysis/transfer_gap_offset.py
$P $S --phase train --smoke 16
$P $S --phase score --sets vob_test,intphys1_dev,grasp_level2_s5 --smoke 16
$P $S --phase report --smoke 16
echo E27_SMOKE_DONE
