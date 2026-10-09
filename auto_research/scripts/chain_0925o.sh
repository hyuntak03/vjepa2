#!/bin/bash
# 2026-09-25 묶음 O (DESIGN_CHOICES §3 단계 O): mret predictor (DC1(b)) post-FT from Ariel ep43, K400 40k, 팬 증강 — N1 과 구조만 다르다.
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python; S=auto_research/scripts; C=/data2/local_datasets/world/world_analysis/cache/auto_research
R=/data/hyuntak/project/2026/2027_cvpr/vjepa2/z_training/runs
run() { echo "[$(date)] >>> $*"; "$@"; r=$?; echo "[$(date)] <<< exit $r"; return $r; }
unset ARLIB_PRED_CKPT
GPUS=8 NAME=reachO_smoke SET="optimization.epochs=1 optimization.ipe=5 val.at_start=false" run bash z_training/train.sh reach_k400_mret || exit 1
GPUS=8 NAME=reachO_mret_pan run bash z_training/train.sh reach_k400_mret
export SCENE_EXTRA="O1=$R/reachO_mret_pan/latest.pt,N1=$R/reachN_prefix_pan/latest.pt" SCENE_ONLY_EXTRA=1
run $P $S/e_scene_reach.py --gpus 8 --n 800 --out $C/scene_pan_trainedO
for src in ssv2 ek100; do run $P $S/e_scene_reach_nat.py --source $src --gpus 8 --out $C/scene_nat_${src}_trainedO; done
