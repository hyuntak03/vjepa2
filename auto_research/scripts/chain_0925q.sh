#!/bin/bash
# 묶음 Q: 장면 단위 permanence (인공 팬 + 마지막 k 프레임 화면 고정 가림 띠). release · Ariel · pv1 (+ 학습 run 들).
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python; S=auto_research/scripts; C=/data2/local_datasets/world/world_analysis/cache/auto_research
R=/data/hyuntak/project/2026/2027_cvpr/vjepa2/z_training/runs
run() { echo "[$(date)] >>> $*"; "$@"; r=$?; echo "[$(date)] <<< exit $r"; return $r; }
unset ARLIB_PRED_CKPT SCENE_ONLY_EXTRA
export SCENE_EXTRA="N1=$R/reachN_prefix_pan/latest.pt,T1=$R/reach_k400_pan/latest.pt"
run $P $S/e_scene_permanence.py --gpus 1 --limit 6 --out $C/_smoke/perm || exit 1
run $P $S/e_scene_permanence.py --gpus ${G:-8} --n 900 --out $C/scene_perm
