#!/bin/bash
# 2026-09-25 묶음: M1/M3 → P2 stride sweep → E-SSv2. 각자 스모크 (GPU 1장, 4 clip) 가 통과해야 본 실행.
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python; S=auto_research/scripts; T=/data2/local_datasets/world/world_analysis/cache/auto_research/_smoke
run() { echo "[$(date)] >>> $*"; "$@"; r=$?; echo "[$(date)] <<< exit $r"; return $r; }
( run $P $S/m13_memory_state_v3.py --gpus 1 --limit 4 --out $T/m13 && run $P $S/m13_memory_state_v3.py --gpus 8 )
( run $P $S/p2_stride_sweep_v3.py --gpus 1 --limit 4 --out $T/p2 && run $P $S/p2_stride_sweep_v3.py --gpus 8 )
( run $P $S/e_ssv2_extract.py --gpus 1 --limit 4 --out $T/ssv2 && run $P $S/e_ssv2_extract.py --gpus 8 )
