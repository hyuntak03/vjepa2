#!/bin/bash
# 2026-11-09 사용자 "GPU util 최대한 꽉 채워서": 전이 탐색 여섯째 판 (v4bc4f · 음의 손실 바꿔치기 없음) 을 메모리가 비는 대로 올린다.
#   조건: 모든 GPU 사용 메모리 < 78 GB (판 하나 ≈ 16.5 GB, 98 GB 중) 가 1 분 간격 3 번 연속. 사슬 (chain_transfer_search_20261109) 의
#   나중 올리기는 tmux 세션이 이미 있으면 그냥 넘어간다 (채점은 사슬이 한다).
cd /workspace/world_model/vjepa2
ts() { TZ=Asia/Seoul date '+%F %T KST'; }
R=v4bc4f_ar_vw_state_negimp_noswap
ok=0
while (( ok < 3 )); do
  tmux has-session -t "=train_$R" 2>/dev/null && { echo "이미 올라가 있음 $(ts)"; exit 0; }
  full=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 >= 78000' | wc -l)
  if (( full == 0 )); then ok=$((ok + 1)); else ok=0; fi
  sleep 60
done
GPUS=8 bash z_training/tmux_train.sh $R && echo "TRAIN_LAUNCHED $R $(ts)"
