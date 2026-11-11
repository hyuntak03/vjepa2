#!/bin/bash
# 2026-11-06 사용자: 사라짐 : 나타남 1 : 1 · λ 1 · batch 96 (데이터 맞춤) — AR · 릴리즈. 판마다 e20 → 시트 벤치 (chain_full_eval) + Surrogate.
#   (앞의 λ 1 데이터 맞춤 두 판 · λ 0.75 대기는 사용자 지시로 취소 — z_training/runs/_aborted_v4bc4_*_negimp_dm_20261106)
cd /workspace/world_model/vjepa2
ts() { TZ=Asia/Seoul date '+%F %T KST'; }
for pair in "v4bc4_ar_vw_state_negimp_va11_dm:vast_ar_vw_state_v4bc4_negimp_va11_dm" "v4bc4_release_negimp_va11_dm:vith_release_v4bc4_negimp_va11_dm"; do
  RUN=${pair%%:*}; M=${pair##*:}
  RUN=$RUN MODEL=$M EP=20 GPU_IDS="0 1 2 3 4 5 6 7" bash z_training/chain_full_eval.sh > z_training/logs/chain_full_eval_$M.log 2>&1 || { echo "⚠️ $RUN e20 없이 끝남 $(ts)"; continue; }
  GPU_IDS="0 1 2 3 4 5 6 7" GPUS=8 bash cvpr/eval/testbed/run.sh intphys1_surrogate $M > z_training/logs/surrogate_eval_$M.log 2>&1
  echo "$M 채점 끝 $(ts)"
done
echo "CHAIN_V4BC4_VA11_DONE $(ts)"
