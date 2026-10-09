#!/bin/bash
# 2026-09-27 AR 묶음: Ariel 자기회귀 predictor (ep18) + full ep40 + prefix ep45 를 reach (팬 · SSv2) · permanence (팬) 판독에.
#   vll2 (K400 val 은 /data2 에 있음, SSv2 는 tar 로 /tmp 에 푼다). EK100 · v3 는 vll3/5/6 에만 있어 뒤에 따로 (chain_0927ar_ek.sh).
#   결과는 NFS auto_research/_stage/results_ar/ 에 바로 쓴다 (작다). 학습 없음.
set -uo pipefail
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python; S=auto_research/scripts
ROOT=/data/hyuntak/project/2026/2027_cvpr/vjepa2; OUT=$ROOT/auto_research/_stage/results_ar
run() { echo "[$(date)] >>> $*"; "$@"; r=$?; echo "[$(date)] <<< exit $r"; return $r; }
cd $ROOT
export SCENE_PAN_LIST=$ROOT/auto_research/exp_results/scene/pan_items_1600.json
STG=/tmp/hyuntak_ar_stage; mkdir -p $STG
if [[ ! -d $STG/world/world_analysis/ssv2_VP_default/left ]]; then run tar xf $ROOT/auto_research/_stage/ssv2_vp.tar -C $STG; fi
export SCENE_SSV2=$STG/world/world_analysis/ssv2_VP_default
unset ARLIB_PRED_CKPT SCENE_EXTRA SCENE_ONLY_EXTRA
run $P $S/e_scene_ar.py --task reach --source pan --gpus 1 --limit 4 --out $OUT/_smoke/ar_pan || exit 1
run $P $S/e_scene_ar.py --task perm --source pan --gpus 1 --limit 3 --out $OUT/_smoke/ar_perm || exit 1
run $P $S/e_scene_ar.py --task reach --source pan --gpus ${G:-4} --n 800 --out $OUT/scene_ar_pan
run $P $S/e_scene_ar.py --task reach --source ssv2 --gpus ${G:-4} --out $OUT/scene_ar_ssv2
run $P $S/e_scene_ar.py --task perm --source pan --gpus ${G:-4} --n 900 --out $OUT/scene_ar_perm
echo "[$(date)] CHAIN DONE"
