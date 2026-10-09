#!/bin/bash
# 2026-09-26 묶음 T: 장면 단위 closure (인공 팬 K400) — os · ar_z (관측 되먹임) · ar_pA (자기 예측 되먹임). release · Ariel · pv1 (+N1). 학습 없음.
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python; S=auto_research/scripts; C=/data2/local_datasets/world/world_analysis/cache/auto_research
R=/data/hyuntak/project/2026/2027_cvpr/vjepa2/z_training/runs
run() { echo "[$(date)] >>> $*"; "$@"; r=$?; echo "[$(date)] <<< exit $r"; return $r; }
unset ARLIB_PRED_CKPT SCENE_ONLY_EXTRA; export SCENE_EXTRA="N1=$R/reachN_prefix_pan/latest.pt"
run $P $S/e_scene_closure.py --gpus 1 --limit 4 --out $C/_smoke/closure || exit 1
run $P $S/e_scene_closure.py --gpus ${G:-8} --n 400 --out $C/scene_closure
