#!/usr/bin/env bash
set -uo pipefail
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
for D in -5 -20 -30 -60; do
  $P cvpr/eval/intphys2_late/build.py --labels data_csv/intphys2_late_uni/labels_snapshot.csv --d-offset $D \
     --frames-out /data2/local_datasets/world/world_analysis/cache/transfer_gap/ip2late_d$D --out data_csv/transfer_gap_ip2late_d$D --workers 22 || echo "FAIL $D"
done
echo E29_BUILD_DONE
