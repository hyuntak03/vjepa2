#!/bin/bash
# 2026-09-25 묶음 L: 장면 단위 reach 인과 개입 — 자연 움직임 (SSv2 · EK100: 역방향 flow 출처, v3: GT 출처). e_scene_reach_nat.py
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python; S=auto_research/scripts; C=/data2/local_datasets/world/world_analysis/cache/auto_research
run() { echo "[$(date)] >>> $*"; "$@"; r=$?; echo "[$(date)] <<< exit $r"; return $r; }
unset ARLIB_PRED_CKPT
run $P $S/e_scene_reach_nat.py --source v3 --gpus 1 --limit 4 --out $C/_smoke/nat_v3 || exit 1
for src in ssv2 v3 ek100; do run $P $S/e_scene_reach_nat.py --source $src --gpus 8 --out $C/scene_nat_$src; done
