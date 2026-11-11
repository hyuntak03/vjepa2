#!/bin/bash
# 2026-11-06 사용자 요청: v4bc4 L1 (음의 손실 없음) 채점 — e20 → 시트 벤치 (chain_full_eval) + Surrogate.
cd /workspace/world_model/vjepa2
ts() { TZ=Asia/Seoul date '+%F %T KST'; }
M=vast_ar_vw_state_v4bc4
RUN=v4bc4_ar_vw_state_postft MODEL=$M EP=20 GPU_IDS="0 1 2 3 4 5 6 7" bash z_training/chain_full_eval.sh > z_training/logs/chain_full_eval_$M.log 2>&1 || { echo "⚠️ e20 없이 끝남"; exit 1; }
GPU_IDS="0 1 2 3 4 5 6 7" GPUS=8 bash cvpr/eval/testbed/run.sh intphys1_surrogate $M > z_training/logs/surrogate_eval_$M.log 2>&1
echo "CHAIN_V4BC4_L1_DONE $(ts)"
