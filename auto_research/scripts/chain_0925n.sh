#!/bin/bash
# 2026-09-25 묶음 N (DESIGN_CHOICES §3 단계 N): Ariel ep43 (prefix) post-FT on K400 40k × {팬 증강, 고정 창} 3 epoch → 같은 개입 판으로 평가.
#   묶음 M (release post-FT) 뒤에 돈다. 하네스 병합 (kind · prefix_window · sync_masks) 이 검증된 뒤에만 제출할 것.
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python; S=auto_research/scripts; C=/data2/local_datasets/world/world_analysis/cache/auto_research
R=/data/hyuntak/project/2026/2027_cvpr/vjepa2/z_training/runs
run() { echo "[$(date)] >>> $*"; "$@"; r=$?; echo "[$(date)] <<< exit $r"; return $r; }
unset ARLIB_PRED_CKPT
GPUS=8 NAME=reachN_smoke SET="data.aug.window.pan_p=0.7 optimization.epochs=1 optimization.ipe=5 val.at_start=false" run bash z_training/train.sh reach_k400_prefix || exit 1
GPUS=8 NAME=reachN_prefix_pan SET="data.aug.window.pan_p=0.7" run bash z_training/train.sh reach_k400_prefix
GPUS=8 NAME=reachN_prefix_static SET="data.aug.window.pan_p=0.0" run bash z_training/train.sh reach_k400_prefix
export SCENE_EXTRA="N0=$R/reachN_prefix_static/latest.pt,N1=$R/reachN_prefix_pan/latest.pt" SCENE_ONLY_EXTRA=1
run $P $S/e_scene_reach.py --gpus 8 --n 800 --out $C/scene_pan_trainedN
for src in ssv2 ek100; do run $P $S/e_scene_reach_nat.py --source $src --gpus 8 --out $C/scene_nat_${src}_trainedN; done
