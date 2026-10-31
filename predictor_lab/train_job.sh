#!/bin/bash
# predictor_lab/train.sh 의 본체 (sbatch 안 또는 LOCAL=1 · DRYRUN=1). 직접 부르지 말고 train.sh 를 쓴다.
# 입력 (환경변수): CONFIG · NAME · GPUS · SET_B64 [· LIMIT · DRYRUN · THREADS · TRAIN_DDP_PORT]
# z_training/train.sh 와 같은 순서 (resolve_train → DDP 포트 → harness/launch.py) 인데 파이썬만 paths.env 의 LAB_PY 를 쓴다.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[[ -f $HERE/paths.env ]] || HERE=/data/hyuntak/project/2026/2027_cvpr/vjepa2/predictor_lab   # sbatch 가 스크립트를 복사해 돌릴 때
source "$HERE/paths.env"
cd "$LAB_REPO"
export PYTHONPATH="$LAB_REPO${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TRAIN_CFG_DIR=$HERE/configs                     # resolve_train.py 가 여기를 먼저 보고 configs/training 으로 떨어진다
SET=$(printf %s "${SET_B64:-}" | base64 -d)
GPUS=${GPUS:-1}
FOLDER=$LAB_REPO/z_training/runs/$NAME
if [[ -n ${DRYRUN:-} ]]; then CFGOUT=$(mktemp /tmp/lab_train_XXXXXX.yaml); trap 'rm -f "$CFGOUT"' EXIT; else mkdir -p "$FOLDER"; CFGOUT=$FOLDER/config.yaml; fi
ARGS=(); for kv in $SET; do ARGS+=(--set "$kv"); done
[[ -n ${LIMIT:-} ]] && ARGS+=(--limit "$LIMIT")
echo "=================================================="
echo "  node $(hostname) | config $CONFIG | run $NAME | GPU $GPUS | python $LAB_PY"
"$LAB_PY" z_training/harness/resolve_train.py "$CONFIG" -o "$CFGOUT" --folder "$FOLDER" "${ARGS[@]}" || exit 1
[[ -n ${DRYRUN:-} ]] && { echo "--- DRYRUN: 병합된 config ---"; cat "$CFGOUT"; exit 0; }
[[ -f $FOLDER/latest.pt ]] && echo "  resume: $FOLDER/latest.pt 가 있다 -> 이어서 돈다 (새로 시작하려면 다른 NAME)"
if [[ -z ${TRAIN_DDP_PORT:-} ]]; then                 # job id 로 갈라 같은 노드의 다른 job 과 안 겹치게
  BASE=$(( 30000 + ( ${SLURM_JOB_ID:-$$} % 10000 ) ))
  TRAIN_DDP_PORT=$("$LAB_PY" -c "
import socket, sys
for p in range($BASE, $BASE + 200):
    s = socket.socket()
    try:
        s.bind(('', p)); s.close(); print(p); sys.exit(0)
    except OSError:
        s.close()
sys.exit(1)") || { echo "ERROR: 빈 DDP 포트를 못 찾았다"; exit 1; }
fi
export TRAIN_DDP_PORT TRAIN_EXPECT_WS=$GPUS TRAIN_DDP_TIMEOUT_S=${TRAIN_DDP_TIMEOUT_S:-7200}
T=$(( $(nproc) / GPUS )); (( T > 12 )) && T=12; (( T < 1 )) && T=1
export OMP_NUM_THREADS=${THREADS:-$T} MKL_NUM_THREADS=${THREADS:-$T}
DEVICES=$(for i in $(seq 0 $((GPUS - 1))); do echo -n "cuda:$i "; done)
echo "  devices $DEVICES | ddp port $TRAIN_DDP_PORT | threads/rank $OMP_NUM_THREADS | log $FOLDER/train.log"
echo "=================================================="
"$LAB_PY" z_training/harness/launch.py --fname "$CFGOUT" --devices $DEVICES 2>&1 | tee -a "$FOLDER/train.log"
exit "${PIPESTATUS[0]}"
