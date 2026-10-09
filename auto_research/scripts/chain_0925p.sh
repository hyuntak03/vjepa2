#!/bin/bash
# 2026-09-25 묶음 P (분석 A): base 팔 cos 지도 전체 저장 → 연속 위치 판독 (soft-argmax · top-k) 으로 거리 vs 걸음 재검정. 팬 800 · SSv2 · EK100 · v3.
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python; S=auto_research/scripts; C=/data2/local_datasets/world/world_analysis/cache/auto_research
run() { echo "[$(date)] >>> $*"; "$@"; r=$?; echo "[$(date)] <<< exit $r"; return $r; }
unset ARLIB_PRED_CKPT SCENE_EXTRA SCENE_ONLY_EXTRA
run $P $S/e_scene_reach.py --gpus 1 --limit 4 --cosmaps --out $C/_smoke/scene_cm || exit 1
run $P $S/e_scene_reach.py --gpus ${G:-8} --n 800 --cosmaps --out $C/scene_pan_cm
for src in ssv2 ek100 v3; do run $P $S/e_scene_reach_nat.py --source $src --gpus ${G:-8} --cosmaps --out $C/scene_nat_${src}_cm; done
