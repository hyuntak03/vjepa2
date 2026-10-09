#!/bin/bash
# 2026-09-29 EK100 encoder 충분성 (ego-motion) — EK100 판 (vll3/5/6).
set -uo pipefail
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python; S=auto_research/scripts
ROOT=/data/hyuntak/project/2026/2027_cvpr/vjepa2; OUT=$ROOT/auto_research/_stage/results_ar
run() { echo "[$(date)] >>> $*"; "$@"; r=$?; echo "[$(date)] <<< exit $r"; return $r; }
cd $ROOT; unset ARLIB_PRED_CKPT
run $P $S/e_ek_egomotion.py --source ek100 --gpus 1 --limit 3 --out $OUT/_smoke/ego_ek || exit 1
run $P $S/e_ek_egomotion.py --source ek100 --gpus ${G:-2} --out $OUT/ego_ek100
echo "[$(date)] CHAIN DONE"
