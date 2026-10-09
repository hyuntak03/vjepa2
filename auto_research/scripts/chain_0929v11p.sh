#!/bin/bash
# 2026-09-29 v11 late 가림 · vanish 쌍 presence vs placement (AR_PREDICTOR §2-6 모순 검정). vll1, 스테이징 auto_research/_stage/v11_late.
set -uo pipefail
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python; S=auto_research/scripts
ROOT=/data/hyuntak/project/2026/2027_cvpr/vjepa2; OUT=$ROOT/auto_research/_stage/results_ar
run() { echo "[$(date)] >>> $*"; "$@"; r=$?; echo "[$(date)] <<< exit $r"; return $r; }
cd $ROOT; unset ARLIB_PRED_CKPT SCENE_EXTRA SCENE_ONLY_EXTRA; export SCENE_V11_STAGE=$ROOT/auto_research/_stage/v11_late
run $P $S/e_v11_presence.py --gpus 1 --limit 4 --out $OUT/_smoke/v11p || exit 1
run $P $S/e_v11_presence.py --gpus ${G:-2} --out $OUT/v11_presence
echo "[$(date)] CHAIN DONE"
