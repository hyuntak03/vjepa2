#!/bin/bash
# 2026-09-27 AR 묶음 2/2: EK100 · v3 reach (데이터가 vll3 · vll5 · vll6 의 /data2 에만 있다). chain_0927ar.sh (팬 · SSv2, vll2) 의 짝.
set -uo pipefail
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python; S=auto_research/scripts
ROOT=/data/hyuntak/project/2026/2027_cvpr/vjepa2; OUT=$ROOT/auto_research/_stage/results_ar
run() { echo "[$(date)] >>> $*"; "$@"; r=$?; echo "[$(date)] <<< exit $r"; return $r; }
cd $ROOT; unset ARLIB_PRED_CKPT SCENE_EXTRA SCENE_ONLY_EXTRA
run $P $S/e_scene_ar.py --task reach --source ek100 --gpus 1 --limit 4 --out $OUT/_smoke/ar_ek || exit 1
run $P $S/e_scene_ar.py --task reach --source ek100 --gpus ${G:-4} --out $OUT/scene_ar_ek100
[[ -d /data2/local_datasets/world/world_analysis/RollOut_v3 ]] && run $P $S/e_scene_ar.py --task reach --source v3 --gpus ${G:-4} --out $OUT/scene_ar_v3
echo "[$(date)] CHAIN DONE"
