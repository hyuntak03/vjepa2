#!/bin/bash
# 곡률 손실 판을 tmux 로 올린다 (2026-11-07). tmux_train.sh 와 같고 train_curv.sh 를 부른다.
#   GPUS=8 bash z_training/tmux_train_curv.sh v4bc4_ar_vw_state_curv [NAME]
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../z_research/scripts/harness" && pwd)/env.sh"
cd "$VJEPA2_ROOT"
CONFIG=${1:?"사용법: tmux_train_curv.sh <config> [NAME]"}; NAME=${2:-$CONFIG}; SESS="train_$NAME"
if tmux has-session -t "=$SESS" 2>/dev/null; then echo "이미 있다: tmux attach -t $SESS"; exit 1; fi
mkdir -p z_training/runs/"$NAME"
tmux new-session -d -s "$SESS" -c "$VJEPA2_ROOT" \
  "source \"$CONDA_ACTIVATE\" \"$CONDA_ENV\"; env GPUS=${GPUS:-8} bash z_training/train_curv.sh $CONFIG $NAME; \
   echo; echo '=== train_curv.sh 종료 (exit '\$?') ==='; exec bash"
echo "올렸다: tmux 세션 $SESS"
