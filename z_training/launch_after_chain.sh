#!/bin/bash
# =============================================================================
# 지금 도는 학습 묶음이 다 끝난 뒤에 새 학습 하나를 8 GPU 로 올린다 (2026-11-02, 사용자: "이거 학습 후에 ar 학습할 때 v4 껴서 … scratch 부터").
#   끝남 = (CHAIN_LOG 에 DONE_MARK 가 찍힘 또는 그 chain 프로세스가 없음) 그리고 z_training/harness/launch.py 학습 프로세스가 하나도 없음.
#   chain 이 학습 뒤에 하는 채점까지 기다린다 (채점과 겹치면 GPU 메모리가 모자라 채점이 죽은 적이 있다).
#
#   CONFIG=natural_ar_vw_state_v4vc setsid nohup bash z_training/launch_after_chain.sh > z_training/logs/launch_after_chain_<CONFIG>.log 2>&1 &
# =============================================================================
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${CONFIG:?}" "${CHAIN_LOG:=z_training/logs/chain_lora_pft.log}" "${DONE_MARK:=CHAIN_LORA_PFT_DONE}" "${CHAIN_PAT:=chain_lora_pft\.py}"
ts() { TZ=Asia/Seoul date '+%F %T KST'; }
chain_alive() { ps -eo args | grep -qE "^\S*python\S* \S*${CHAIN_PAT}"; }
train_alive() { ps -eo args | grep -qE '^\S*python\S* \S*z_training/harness/launch\.py'; }
echo "[launch_after_chain] $(ts) 기다림: $CHAIN_LOG 의 $DONE_MARK (또는 chain 종료) + 학습 프로세스 0 → $CONFIG 올림"
until { grep -q "$DONE_MARK" "$CHAIN_LOG" 2>/dev/null || ! chain_alive; } && ! train_alive; do sleep 120; done
sleep 60
echo "[launch_after_chain] $(ts) 비었다. GPU 메모리:"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
GPUS=8 bash z_training/tmux_train.sh "$CONFIG"
echo "LAUNCH_AFTER_CHAIN_DONE $CONFIG $(ts)"
