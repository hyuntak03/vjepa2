#!/bin/bash
# 2026-09-29 EK100 encoder 충분성 (ego-motion) — 팬 (양성 대조) · SSv2 는 vll1 에서; EK100 은 데이터가 vll3/5/6 에만 있어 chain_0929ego_ek.sh 로.
set -uo pipefail
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python; S=auto_research/scripts
ROOT=/data/hyuntak/project/2026/2027_cvpr/vjepa2; OUT=$ROOT/auto_research/_stage/results_ar
run() { echo "[$(date)] >>> $*"; "$@"; r=$?; echo "[$(date)] <<< exit $r"; return $r; }
cd $ROOT; export SCENE_PAN_LIST=$ROOT/auto_research/exp_results/scene/pan_items_1600.json
STG=/tmp/hyuntak_ar_stage; mkdir -p $STG
if [[ ! -d $STG/world/world_analysis/ssv2_VP_default/left ]]; then run tar xf $ROOT/auto_research/_stage/ssv2_vp.tar -C $STG; fi
export SCENE_SSV2=$STG/world/world_analysis/ssv2_VP_default; unset ARLIB_PRED_CKPT
run $P $S/e_ek_egomotion.py --source pan --gpus 1 --limit 3 --out $OUT/_smoke/ego_pan || exit 1
run $P $S/e_ek_egomotion.py --source pan --gpus ${G:-2} --n 800 --out $OUT/ego_pan
run $P $S/e_ek_egomotion.py --source ssv2 --gpus ${G:-2} --out $OUT/ego_ssv2
echo "[$(date)] CHAIN DONE"
