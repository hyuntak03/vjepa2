#!/bin/bash
# 묶음 M2 의 평가만 다시 (학습은 끝남; 첫 평가는 --cosmaps argparse 누락으로 실패). T0 고정 창 · T1 팬 증강 (release post-FT).
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python; S=auto_research/scripts; C=/data2/local_datasets/world/world_analysis/cache/auto_research
R=/data/hyuntak/project/2026/2027_cvpr/vjepa2/z_training/runs
run() { echo "[$(date)] >>> $*"; "$@"; r=$?; echo "[$(date)] <<< exit $r"; return $r; }
unset ARLIB_PRED_CKPT
export SCENE_EXTRA="T0=$R/reach_k400_static/latest.pt,T1=$R/reach_k400_pan/latest.pt" SCENE_ONLY_EXTRA=1
run $P $S/e_scene_reach.py --gpus ${G:-8} --n 800 --out $C/scene_pan_trained
for src in ssv2 ek100; do run $P $S/e_scene_reach_nat.py --source $src --gpus ${G:-8} --out $C/scene_nat_${src}_trained; done
