#!/bin/bash
# 2026-09-25 묶음 K: 장면 단위 reach 인과 개입 (인공 팬, K400 val 1,600 clip, release · Ariel · pv1 × 규칙 · RoPE · 온도 · 조회 보조 · 억제 · 반복사).
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python; S=auto_research/scripts; C=/data2/local_datasets/world/world_analysis/cache/auto_research
run() { echo "[$(date)] >>> $*"; "$@"; r=$?; echo "[$(date)] <<< exit $r"; return $r; }
unset ARLIB_PRED_CKPT
run $P $S/e_scene_reach.py --gpus 1 --limit 4 --out $C/_smoke/scene_k || exit 1
run $P $S/e_scene_reach.py --gpus 8 --n 1600 --out $C/scene_pan
