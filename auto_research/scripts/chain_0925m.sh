#!/bin/bash
# 2026-09-25 묶음 M: reach 학습 검정 (I_SCENE_CAUSAL_BOTTLENECK §5). 릴리즈 predictor post-FT on K400 train 40k, 3 epoch, 두 팔:
#   T1 인공 카메라 팬 증강 (pan_p 0.7, v ≤ 12 px/프레임)  vs  T0 고정 창 (같은 crop 통계, 팬 없음). 그 뒤 두 run 을 같은 개입 판으로 평가.
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python; S=auto_research/scripts; C=/data2/local_datasets/world/world_analysis/cache/auto_research
R=/data/hyuntak/project/2026/2027_cvpr/vjepa2/z_training/runs
run() { echo "[$(date)] >>> $*"; "$@"; r=$?; echo "[$(date)] <<< exit $r"; return $r; }
unset ARLIB_PRED_CKPT
GPUS=8 NAME=reach_smoke SET="data.aug.window.pan_p=0.7 optimization.epochs=1 optimization.ipe=5 val.at_start=false" run bash z_training/train.sh reach_k400_postft || exit 1
GPUS=8 NAME=reach_k400_pan SET="data.aug.window.pan_p=0.7" run bash z_training/train.sh reach_k400_postft
GPUS=8 NAME=reach_k400_static SET="data.aug.window.pan_p=0.0" run bash z_training/train.sh reach_k400_postft
export SCENE_EXTRA="T0=$R/reach_k400_static/latest.pt,T1=$R/reach_k400_pan/latest.pt" SCENE_ONLY_EXTRA=1
run $P $S/e_scene_reach.py --gpus 8 --n 800 --out $C/scene_pan_trained
for src in ssv2 ek100; do run $P $S/e_scene_reach_nat.py --source $src --gpus 8 --out $C/scene_nat_${src}_trained; done
