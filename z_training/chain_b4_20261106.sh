#!/bin/bash
# 2026-11-06 자동 탐색 묶음 4 채점 (z_training/docs/AUTO_EXPLORE_2026-11-06.md). 판마다 마지막 epoch → 시트 벤치 (chain_full_eval) + Surrogate. 채점만.
cd /workspace/world_model/vjepa2
ts() { TZ=Asia/Seoul date '+%F %T KST'; }
G="0 1 2 3 4 5 6 7"
for row in v4bc4_ar_vw_state_negimp_w075:vast_ar_vw_state_v4bc4_negimp_w075:20 v4bc4_ar_vw_state_negimp_w085:vast_ar_vw_state_v4bc4_negimp_w085:20 \
           v4bc2_ar_vw_state_negimp_w085:vast_ar_vw_state_v4bc2_negimp_w085:20 v4bc2_ar_vw_state_negimp_w075_long:vast_ar_vw_state_v4bc2_negimp_w075_long:40; do
  IFS=: read R M EP <<<"$row"
  echo "[chain] $(ts) $M 기다림 (e$EP) → 채점"
  RUN=$R MODEL=$M EP=$EP GPU_IDS="$G" bash z_training/chain_full_eval.sh > z_training/logs/chain_full_eval_$M.log 2>&1 || { echo "[chain] $(ts) ⚠️ $M 학습이 e$EP 없이 끝남 — 건너뜀"; continue; }
  GPU_IDS="$G" GPUS=8 bash cvpr/eval/testbed/run.sh intphys1_surrogate $M > z_training/logs/surrogate_eval_$M.log 2>&1
  echo "[chain] $(ts) $M 채점 끝"
done
echo "CHAIN_B4_DONE $(ts)"
