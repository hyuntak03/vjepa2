#!/bin/bash
# 2026-09-29 AR ep38 (고정 사본) 재판독: 팬 reach · SSv2 reach · 팬 perm. vll1 (K400 val /data2 ✓, SSv2 tar 스테이징). chain_0927ar.sh 의 ep38 판.
set -uo pipefail
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python; S=auto_research/scripts
ROOT=/data/hyuntak/project/2026/2027_cvpr/vjepa2; OUT=$ROOT/auto_research/_stage/results_ar
run() { echo "[$(date)] >>> $*"; "$@"; r=$?; echo "[$(date)] <<< exit $r"; return $r; }
cd $ROOT
export SCENE_PAN_LIST=$ROOT/auto_research/exp_results/scene/pan_items_1600.json
export SCENE_AR_CKPT=$ROOT/z_training/ariel_eval/ckpt/ar_ep38.pt
STG=/tmp/hyuntak_ar_stage; mkdir -p $STG
if [[ ! -d $STG/world/world_analysis/ssv2_VP_default/left ]]; then run tar xf $ROOT/auto_research/_stage/ssv2_vp.tar -C $STG; fi
export SCENE_SSV2=$STG/world/world_analysis/ssv2_VP_default
unset ARLIB_PRED_CKPT SCENE_EXTRA SCENE_ONLY_EXTRA
run $P $S/e_scene_ar.py --task reach --source pan --gpus 1 --limit 4 --out $OUT/_smoke/ar38_pan || exit 1
run $P $S/e_scene_ar.py --task reach --source pan --gpus ${G:-4} --n 800 --out $OUT/scene_ar38_pan
run $P $S/e_scene_ar.py --task reach --source ssv2 --gpus ${G:-4} --out $OUT/scene_ar38_ssv2
run $P $S/e_scene_ar.py --task perm --source pan --gpus ${G:-4} --n 900 --out $OUT/scene_ar38_perm
echo "[$(date)] CHAIN DONE"
