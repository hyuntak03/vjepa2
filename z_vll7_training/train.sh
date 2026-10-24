#!/bin/bash
# -----------------------------------------------------------------------------
# vll7 frozen encoder + predictor 학습 진입점 (6 x RTX PRO 6000 Blackwell)
#
#   bash z_vll7_training/train.sh --list                          # config · 데이터셋 · run 목록
#   DRYRUN=1 bash z_vll7_training/train.sh ctxt_video_tgtvideo    # 병합 · 실물 검사만 (몇 초, GPU 0 장)
#   GPUS=6 bash z_vll7_training/train.sh ctxt_video_tgtvideo      # 본 학습
#   GPUS=1 LIMIT=... SET="optimization.ipe=3 optimization.epochs=1" bash z_vll7_training/train.sh ctxt_video_tgtvideo   # 스모크
#   GPUS=2 bash z_vll7_training/train.sh --smoke-ddp              # 모델 · 데이터 없이 spawn / 포트 / join 점검
#
# 하는 일 (z_training/train.sh 의 vll7 판 — 경로 정본과 config 폴더만 다르다)
#   1. z_vll7_training/env.sh 로 경로 정본 (paths.env) 을 읽는다. python = vjepa2_bw (sm_120)
#   2. TRAIN_CFG_DIR=z_vll7_training/configs 를 먼저 찾고, 없으면 configs/training 으로 떨어진다 (`extends:` 가 공용 부모를 가져온다)
#   3. resolve_train.py 로 병합 + **모델 로드 전에** 경로 · 인덱스 · 프레임 예산 · kind 전제를 검사 -> <run>/config.yaml
#   4. 빈 DDP 포트를 찾아 TRAIN_DDP_PORT 로, TRAIN_EXPECT_WS 로 split-brain 을 막는다
#   5. rank 당 CPU 스레드를 제한한다 (rank 마다 전 코어를 잡으면 decode 가 7 배 느려진다)
#
# 환경변수: GPUS(6) GPU_IDS="2 3" NAME FOLDER SET="a.b=1 c.d=null" LIMIT DEBUG DRYRUN THREADS TRAIN_DDP_PORT
# 이어 돌리기: 같은 NAME 으로 다시 부르면 <run>/latest.pt 에서 재개 (meta.auto_resume)
# -----------------------------------------------------------------------------
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
PY=$VJEPA2_PY
PROJ=$VJEPA2_ROOT
export TRAIN_CFG_DIR

if [[ "${1:-}" == "--list" || $# -eq 0 ]]; then
  echo "config (z_vll7_training/configs, _ 로 시작하면 부모라 직접 못 돌린다):"
  ls "$TRAIN_CFG_DIR"/*.yaml | xargs -n1 basename | sed 's/\.yaml$//;s/^/  /'
  echo "공용 부모 (configs/training):"; ls "$PROJ"/configs/training/*.yaml | xargs -n1 basename | sed 's/\.yaml$//;s/^/  /' | head -20
  echo "vll7 학습 데이터셋 (configs/training/datasets.md):"; grep '^## vll7_' "$PROJ/configs/training/datasets.md" | sed 's/^## /  /'
  echo "run (z_vll7_training/runs):"; ls "$VLL7_RUNS" 2>/dev/null | sed 's/^/  /' || echo "  (없음)"
  exit 0
fi

cd "$PROJ"
if [[ "${1:-}" == "--smoke-ddp" ]]; then
  GPUS=${GPUS:-2}
  export TRAIN_DDP_PORT=${TRAIN_DDP_PORT:-$(( 30000 + ( ${SLURM_JOB_ID:-$$} % 10000 ) ))} TRAIN_EXPECT_WS=$GPUS
  exec "$PY" z_training/harness/launch.py --smoke_ddp --devices $(for i in $(seq 0 $((GPUS-1))); do echo -n "cuda:$i "; done)
fi
CONFIG=${1:?"사용법: train.sh <config>   (목록: train.sh --list)"}

GPUS=${GPUS:-6}
[[ -n "${DEBUG:-}" ]] && GPUS=1
NAME_GIVEN=${NAME:-}
NAME=${NAME:-$(basename "${CONFIG%.yaml}")}
[[ -n "${LIMIT:-}" && -z "$NAME_GIVEN" ]] && NAME="${NAME}_smoke${LIMIT}"
FOLDER=${FOLDER:-$VLL7_RUNS/$NAME}
if [[ -n "${DRYRUN:-}" ]]; then
  CFG=$(mktemp /tmp/vll7train_XXXXXX.yaml); trap 'rm -f "$CFG"' EXIT
else
  mkdir -p "$FOLDER"; CFG=$FOLDER/config.yaml
fi

echo "=================================================="
SET_ARGS=()
for kv in ${SET:-}; do SET_ARGS+=(--set "$kv"); done
[[ -n "${LIMIT:-}" ]] && SET_ARGS+=(--limit "$LIMIT")
"$PY" z_training/harness/resolve_train.py "$CONFIG" -o "$CFG" --folder "$FOLDER" "${SET_ARGS[@]}"
# GPU_IDS="2 3" 로 쓸 GPU 를 직접 고른다 (같은 노드에 run 둘을 나눠 올릴 때). 주면 GPUS 대신 이게 이긴다.
#   launch.py 가 rank 마다 CUDA_VISIBLE_DEVICES 를 덮으므로 바깥 CUDA_VISIBLE_DEVICES 는 안 먹는다 (CLAUDE.md §2-3 과 같은 함정).
if [[ -n "${GPU_IDS:-}" ]]; then
  DEVICES=$(for i in $GPU_IDS; do echo -n "cuda:$i "; done)
  GPUS=$(echo $GPU_IDS | wc -w)
else
  DEVICES=$(for i in $(seq 0 $((GPUS-1))); do echo -n "cuda:$i "; done)
fi
echo "  devices  : $DEVICES (world_size=$GPUS)"
echo "  python   : $PY"
[[ -f "$FOLDER/latest.pt" ]] && echo "  resume   : $FOLDER/latest.pt 가 있다 -> 이어서 돈다 (새로 시작하려면 다른 NAME)"
echo "=================================================="
if [[ -n "${DRYRUN:-}" ]]; then echo "--- DRYRUN: 병합된 config ---"; cat "$CFG"; exit 0; fi

if [[ -z "${TRAIN_DDP_PORT:-}" ]]; then
  BASE=$(( 30000 + ( ${SLURM_JOB_ID:-$$} % 10000 ) ))
  TRAIN_DDP_PORT=$("$PY" -c "
import socket, sys
for p in range($BASE, $BASE + 200):
    s = socket.socket()
    try:
        s.bind(('', p)); s.close(); print(p); sys.exit(0)
    except OSError:
        s.close()
sys.exit(1)
") || { echo 'ERROR: 빈 DDP 포트를 못 찾았다'; exit 1; }
fi
export TRAIN_DDP_PORT TRAIN_EXPECT_WS=$GPUS TRAIN_DDP_TIMEOUT_S=${TRAIN_DDP_TIMEOUT_S:-7200}
NPROC=$(nproc); T=$(( NPROC / GPUS )); [[ $T -gt 12 ]] && T=12; [[ $T -lt 1 ]] && T=1
export OMP_NUM_THREADS=${THREADS:-$T} MKL_NUM_THREADS=${THREADS:-$T}
echo "  ddp port : $TRAIN_DDP_PORT | threads/rank: $OMP_NUM_THREADS | log: $FOLDER/train.log"

LAUNCH=z_training/harness/launch.py
if [[ -n "${DEBUG:-}" ]]; then
  "$PY" "$LAUNCH" --fname "$CFG" --debugmode 2>&1 | tee -a "$FOLDER/train.log"
else
  "$PY" "$LAUNCH" --fname "$CFG" --devices $DEVICES 2>&1 | tee -a "$FOLDER/train.log"
fi
exit "${PIPESTATUS[0]}"
