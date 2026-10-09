#!/bin/bash
# Ariel 세 팔 (prefix ep45 · full ep40 · ar ep18) 표준 채점 — 2026-09-27. vll5 GPU 4 장 (job 안).
#   bash z_training/ariel_eval/run_ariel_eval.sh
# v11: surprise_c16t32 × v11_split_test (GPU 0·1·2 한 모델씩) · IntPhys1: intphys1_sliding 창 16/32 따로 (GPU 3 순서대로)
set -uo pipefail
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
L=z_training/ariel_eval; BX=z_research/Benchmarks/exp_results
v11() { GPU_IDS="$2" EVAL_DDP_PORT=$3 WMA_BAR=off bash z_research/scripts/run.sh surprise_c16t32 v11_split_test $1 > $L/v11_$1.log 2>&1; echo "$(date +%T) v11 $1 exit $?"; }
ip1() { for w in 16 32; do out=$BX/intphys1_sliding__intphys1_dev_${1}_w$w; [[ -f $out/summary.json ]] && continue
  GPU_IDS="3" EVAL_DDP_PORT=29733 WMA_BAR=off TAG=intphys1_dev_${1}_w$w OUTDIR=$out \
  SET="surprise.intphys1.window_sizes=[$w] model.window_size=$w" \
  bash z_research/scripts/run.sh intphys1_sliding intphys1_dev $1 > $L/ip1_${1}_w$w.log 2>&1; echo "$(date +%T) ip1 $1 w$w exit $?"; done; }
v11 vith_ariel_ar_ep18 0 29730 &
v11 vith_ariel_prefix_ep45 1 29731 &
v11 vith_ariel_full_ep40 2 29732 &
( ip1 vith_ariel_ar_ep18; ip1 vith_ariel_prefix_ep45; ip1 vith_ariel_full_ep40 ) &
wait; echo "$(date +%T) ALL DONE"
