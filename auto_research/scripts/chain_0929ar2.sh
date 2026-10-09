#!/bin/bash
# 2026-09-29 (19:10 재제출) AR ep40 최종 (AR40) 과 ctx_ar ep16 (CX16, 학습 중 16/40) 을 AR ep38 과 나란히: 팬 · SSv2 reach, 팬 perm. vll1. 고정 사본 z_training/ariel_eval/ckpt/.
set -uo pipefail
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python; S=auto_research/scripts
ROOT=/data/hyuntak/project/2026/2027_cvpr/vjepa2; OUT=$ROOT/auto_research/_stage/results_ar
run() { echo "[$(date)] >>> $*"; "$@"; r=$?; echo "[$(date)] <<< exit $r"; return $r; }
cd $ROOT; export SCENE_PAN_LIST=$ROOT/auto_research/exp_results/scene/pan_items_1600.json
export SCENE_AR_CKPT=$ROOT/z_training/ariel_eval/ckpt/ar_ep38.pt
export SCENE_AR_EXTRA="AR40=$ROOT/z_training/ariel_eval/ckpt/ar_ep40.pt,CX16=$ROOT/z_training/ariel_eval/ckpt/ctx_ar_ep16.pt"
STG=/tmp/hyuntak_ar_stage; mkdir -p $STG
if [[ ! -d $STG/world/world_analysis/ssv2_VP_default/left ]]; then run tar xf $ROOT/auto_research/_stage/ssv2_vp.tar -C $STG; fi
export SCENE_SSV2=$STG/world/world_analysis/ssv2_VP_default; unset ARLIB_PRED_CKPT SCENE_EXTRA SCENE_ONLY_EXTRA
run $P $S/e_scene_ar.py --task reach --source pan --gpus 1 --limit 4 --out $OUT/_smoke/ar2_pan || exit 1
run $P $S/e_scene_ar.py --task reach --source pan --gpus ${G:-2} --n 800 --out $OUT/scene_ar2_pan
run $P $S/e_scene_ar.py --task reach --source ssv2 --gpus ${G:-2} --out $OUT/scene_ar2_ssv2
run $P $S/e_scene_ar.py --task perm --source pan --gpus ${G:-2} --n 900 --out $OUT/scene_ar2_perm
echo "[$(date)] CHAIN DONE"
