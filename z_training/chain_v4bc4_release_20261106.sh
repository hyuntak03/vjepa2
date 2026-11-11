#!/bin/bash
# 2026-11-06 사용자: v4bc4 릴리즈 L1 · 음의 손실 λ 1 채점 — 판마다 e20 → 시트 벤치 (chain_full_eval) + Surrogate.
cd /workspace/world_model/vjepa2
ts() { TZ=Asia/Seoul date '+%F %T KST'; }
for pair in "v4bc4_release_postft:vith_release_v4bc4" "v4bc4_release_negimp:vith_release_v4bc4_negimp"; do
  RUN=${pair%%:*}; M=${pair##*:}
  RUN=$RUN MODEL=$M EP=20 GPU_IDS="0 1 2 3 4 5 6 7" bash z_training/chain_full_eval.sh > z_training/logs/chain_full_eval_$M.log 2>&1 || { echo "⚠️ $RUN e20 없이 끝남 $(ts)"; continue; }
  GPU_IDS="0 1 2 3 4 5 6 7" GPUS=8 bash cvpr/eval/testbed/run.sh intphys1_surrogate $M > z_training/logs/surrogate_eval_$M.log 2>&1
  echo "$M 채점 끝 $(ts)"
done
echo "CHAIN_V4BC4_RELEASE_DONE $(ts)"
