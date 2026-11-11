#!/bin/bash
# 곡률 손실 판 전용 (2026-11-07). train.sh 와 순서가 같고 런처만 launch_curv.py 다. train.sh 는 slot 두 판이 쓰는 중이라 안 고쳤다.
#   GPUS=8 bash z_training/train_curv.sh v4bc4_ar_vw_state_curv [NAME]
#   LIMIT=64 DEBUG=1 SET="optimization.epochs=1" bash z_training/train_curv.sh v4bc4_ar_vw_state_curv <스모크 이름>
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../z_research/scripts/harness" && pwd)/env.sh"
PY=$VJEPA2_PY
cd "$VJEPA2_ROOT"; export PYTHONPATH="$VJEPA2_ROOT:${PYTHONPATH:-}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CONFIG=${1:?"사용법: train_curv.sh <config> [NAME]"}; NAME=${2:-$CONFIG}
GPUS=${GPUS:-8}; [[ -n "${DEBUG:-}" ]] && GPUS=1
FOLDER=$VJEPA2_ROOT/z_training/runs/$NAME; mkdir -p "$FOLDER"; CFG=$FOLDER/config.yaml
SET_ARGS=(); for kv in ${SET:-}; do SET_ARGS+=(--set "$kv"); done
[[ -n "${LIMIT:-}" ]] && SET_ARGS+=(--limit "$LIMIT")
"$PY" z_training/harness/resolve_train.py "$CONFIG" -o "$CFG" --folder "$FOLDER" "${SET_ARGS[@]}"
DEVICES=${DEVICES:-$(for i in $(seq 0 $((GPUS-1))); do echo -n "cuda:$i "; done)}
BASE=$(( 30000 + ( $$ % 10000 ) ))
TRAIN_DDP_PORT=${TRAIN_DDP_PORT:-$("$PY" -c "
import socket, sys
for p in range($BASE, $BASE + 200):
    s = socket.socket()
    try:
        s.bind(('', p)); s.close(); print(p); sys.exit(0)
    except OSError:
        s.close()
sys.exit(1)")}
export TRAIN_DDP_PORT TRAIN_EXPECT_WS=$GPUS TRAIN_DDP_TIMEOUT_S=${TRAIN_DDP_TIMEOUT_S:-7200}
T=$(( $(nproc) / GPUS )); [[ $T -gt 12 ]] && T=12; [[ $T -lt 1 ]] && T=1
export OMP_NUM_THREADS=${THREADS:-$T} MKL_NUM_THREADS=${THREADS:-$T}
echo "  devices $DEVICES | port $TRAIN_DDP_PORT | threads $OMP_NUM_THREADS | log $FOLDER/train.log"
LAUNCH=z_training/harness/launch_curv.py
if [[ -n "${DEBUG:-}" ]]; then
  "$PY" "$LAUNCH" --fname "$CFG" --debugmode 2>&1 | tee -a "$FOLDER/train.log"
else
  "$PY" "$LAUNCH" --fname "$CFG" --devices $DEVICES 2>&1 | tee -a "$FOLDER/train.log"
fi
exit "${PIPESTATUS[0]}"
