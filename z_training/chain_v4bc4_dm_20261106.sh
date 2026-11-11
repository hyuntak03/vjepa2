#!/bin/bash
# 2026-11-06 사용자: 데이터 맞춤 음의 손실 (batch 96) — λ 1 두 판 (AR · 릴리즈) 먼저, 끝나면 λ 0.75 두 판을 올린다 (GPU 메모리: 동시에 두 판).
#   판마다 e20 → 시트 벤치 (chain_full_eval) + Surrogate.
cd /workspace/world_model/vjepa2
ts() { TZ=Asia/Seoul date '+%F %T KST'; }
score() {  # score <run> <model>
  RUN=$1 MODEL=$2 EP=20 GPU_IDS="0 1 2 3 4 5 6 7" bash z_training/chain_full_eval.sh > z_training/logs/chain_full_eval_$2.log 2>&1 || { echo "⚠️ $1 e20 없이 끝남 $(ts)"; return; }
  GPU_IDS="0 1 2 3 4 5 6 7" GPUS=8 bash cvpr/eval/testbed/run.sh intphys1_surrogate $2 > z_training/logs/surrogate_eval_$2.log 2>&1
  echo "$2 채점 끝 $(ts)"
}
wait_done() { until [[ -f z_training/runs/$1/e20.pt ]] && ! ps -eo args | grep -qE "z_training/harness/launch\.py --fname \S*/runs/$1/config\.yaml"; do sleep 60; done; }
wait_done v4bc4_ar_vw_state_negimp_dm; wait_done v4bc4_release_negimp_dm
echo "λ 1 두 판 학습 끝 $(ts) → λ 0.75 두 판 올림"
GPUS=8 bash z_training/tmux_train.sh v4bc4_ar_vw_state_negimp_w075_dm
GPUS=8 bash z_training/tmux_train.sh v4bc4_release_negimp_w075_dm
score v4bc4_ar_vw_state_negimp_dm vast_ar_vw_state_v4bc4_negimp_dm
score v4bc4_release_negimp_dm vith_release_v4bc4_negimp_dm
score v4bc4_ar_vw_state_negimp_w075_dm vast_ar_vw_state_v4bc4_negimp_w075_dm
score v4bc4_release_negimp_w075_dm vith_release_v4bc4_negimp_w075_dm
echo "CHAIN_V4BC4_DM_DONE $(ts)"
