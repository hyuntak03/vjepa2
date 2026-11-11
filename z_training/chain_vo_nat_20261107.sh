#!/bin/bash
# 2026-11-07 사용자: 시간을 가져오기로 (VO) — AR 기본판에서 자연 영상 5 epoch 이어 학습 세 판을 차례로 (GPU 메모리 81 GB 라 한 번에 한 판).
#   판마다 학습 → 채점 (시트 벤치: VOB v3 · IP1 late · IP2 late · real + Surrogate) → 다음 판. 순서 ours → control → notime.
cd /workspace/world_model/vjepa2
ts() { TZ=Asia/Seoul date '+%F %T KST'; }
for pair in "vo_nat_ours:vast_ar_vo_ours" "vo_nat_control:vast_ar_vo_control" "vo_nat_notime:vast_ar_vo_notime"; do
  RUN=${pair%%:*}; M=${pair##*:}
  echo "$(ts) $RUN 학습 시작"
  GPUS=8 bash z_training/tmux_train.sh $RUN
  RUN=$RUN MODEL=$M EP=5 GPU_IDS="0 1 2 3 4 5 6 7" bash z_training/chain_full_eval.sh > z_training/logs/chain_full_eval_$M.log 2>&1 || { echo "⚠️ $RUN e5 없이 끝남 $(ts)"; continue; }
  GPU_IDS="0 1 2 3 4 5 6 7" GPUS=8 bash cvpr/eval/testbed/run.sh intphys1_surrogate $M > z_training/logs/surrogate_eval_$M.log 2>&1
  echo "$(ts) $M 채점 끝"
done
echo "CHAIN_VO_NAT_DONE $(ts)"
