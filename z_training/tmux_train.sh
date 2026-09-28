#!/bin/bash
# =============================================================================
# 학습을 **tmux 세션**으로 올린다 (SLURM 없이, 이 노드에서). 세션이 셸과 무관하게 살아남는다.
#
#   bash z_training/tmux_train.sh <config> [NAME]              # 기본 GPUS=8
#   GPUS=8 NAME=nat_prefix SET="..." bash z_training/tmux_train.sh natural_prefix
#
#   보기:   watch -c -n 1 bash z_training/monitor.sh <NAME>
#   붙기:   tmux attach -t train_<NAME>        (떼기: Ctrl-b d)
#   중단:   tmux kill-session -t train_<NAME>  (rank 는 PDEATHSIG 로 같이 죽는다)
#   재개:   같은 NAME 으로 다시 올리면 <run>/latest.pt 에서 이어 돈다 (meta.auto_resume)
# =============================================================================
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../z_research/scripts/harness" && pwd)/env.sh"
cd "$VJEPA2_ROOT"
CONFIG=${1:?"사용법: tmux_train.sh <config> [NAME]"}
NAME=${2:-${NAME:-$(basename "${CONFIG%.yaml}")}}
SESS="train_$NAME"
if tmux has-session -t "$SESS" 2>/dev/null; then echo "이미 있다: tmux attach -t $SESS"; exit 1; fi
mkdir -p z_training/runs/"$NAME"
ENVS="GPUS=${GPUS:-8} NAME=$NAME"
[[ -n "${SET:-}" ]]   && ENVS="$ENVS SET=$(printf %q "$SET")"
[[ -n "${LIMIT:-}" ]] && ENVS="$ENVS LIMIT=$LIMIT"
tmux new-session -d -s "$SESS" -c "$VJEPA2_ROOT" \
  "source \"$CONDA_ACTIVATE\" \"$CONDA_ENV\"; env $ENVS bash z_training/train.sh $CONFIG; \
   echo; echo '=== train.sh 종료 (exit '\$?') — 이 창은 로그 확인용으로 남겨 둔다 ==='; exec bash"
echo "올렸다: tmux 세션 $SESS  (config $CONFIG, GPUS ${GPUS:-8}${SET:+, SET=\"$SET\"})"
echo "  watch -c -n 1 bash z_training/monitor.sh $NAME"
echo "  tmux attach -t $SESS"
