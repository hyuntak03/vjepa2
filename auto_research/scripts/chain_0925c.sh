#!/bin/bash
# 2026-09-25 묶음 C: M5 속도 조향 개입 → E-SSv2 양성 대조 재추출.
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python; S=auto_research/scripts; T=/data2/local_datasets/world/world_analysis/cache/auto_research/_smoke
run() { echo "[$(date)] >>> $*"; "$@"; r=$?; echo "[$(date)] <<< exit $r"; return $r; }
( run $P $S/m5_steer_velocity_v3.py --gpus 1 --limit 4 --fit-clips 24 --out $T/m5 && run $P $S/m5_steer_velocity_v3.py --gpus 8 )
run $P $S/e_ssv2_extract.py --gpus 8 --out /data2/local_datasets/world/world_analysis/cache/auto_research/ssv2_e2   # 검증 중인 ssv2_e 를 덮지 않게
