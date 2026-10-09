#!/bin/bash
# vll6 에 RollOut_v3 (20 GB, 3,136 clip × 64 프레임, 사건 f32) 를 푼다. H1~H5 의 주 데이터 (사용자 지시 2026-09-24).
# rollout3_window_readout.py 가 SRC=/data2/local_datasets/world/world_analysis/RollOut_v3 를 그대로 쓰므로 같은 자리에 둔다.
set -euo pipefail
[ "$(hostname)" = "vll6" ] || { echo "vll6 에서만"; exit 1; }
DST=/data2/local_datasets/world/world_analysis
if [ ! -f "$DST/RollOut_v3/.staged" ]; then
  tar xf /data/dataset/world/RollOut_v3.tar -C "$DST"; touch "$DST/RollOut_v3/.staged"
fi
L=/local_datasets/world/world_analysis/RollOut_v3
[ -e "$L" ] || ln -sfn "$DST/RollOut_v3" "$L"
ls "$DST/RollOut_v3"; ls "$DST/RollOut_v3/Images" | head; find "$DST/RollOut_v3/Images" -maxdepth 3 -mindepth 3 -type d | wc -l
echo DONE
