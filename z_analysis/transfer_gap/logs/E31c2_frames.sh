#!/usr/bin/env bash
set -euo pipefail
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
$P z_research/scripts/analysis/transfer_gap_srframes.py --seeds 1,2,3,4 --sets ip1,vob --procs 22
$P z_research/scripts/analysis/transfer_gap_srframes.py --check 1
$P z_research/scripts/analysis/transfer_gap_srframes.py --check 4
echo E31C2_FRAMES_DONE
