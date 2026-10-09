#!/bin/bash
# 2026-09-29 EK100 을 스테이징 (auto_research/_stage/ek100_800) 으로 vll1 에서: ego-motion 판 + AR ep38 reach 재판독.
set -uo pipefail
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python; S=auto_research/scripts
ROOT=/data/hyuntak/project/2026/2027_cvpr/vjepa2; OUT=$ROOT/auto_research/_stage/results_ar
run() { echo "[$(date)] >>> $*"; "$@"; r=$?; echo "[$(date)] <<< exit $r"; return $r; }
cd $ROOT; unset ARLIB_PRED_CKPT SCENE_EXTRA SCENE_ONLY_EXTRA
export SCENE_EK_STAGE=$ROOT/auto_research/_stage/ek100_800
export SCENE_AR_CKPT=$ROOT/z_training/ariel_eval/ckpt/ar_ep38.pt
run $P $S/e_ek_egomotion.py --source ek100 --gpus 1 --limit 3 --out $OUT/_smoke/ego_ek || exit 1
run $P $S/e_ek_egomotion.py --source ek100 --gpus ${G:-2} --out $OUT/ego_ek100
run $P $S/e_scene_ar.py --task reach --source ek100 --gpus ${G:-2} --out $OUT/scene_ar38_ek100
echo "[$(date)] CHAIN DONE"
