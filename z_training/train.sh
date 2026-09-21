#!/bin/bash
# -----------------------------------------------------------------------------
# frozen encoder + predictor 학습 진입점
#
#   ★ 학습 데이터는 미정이라 config 의 data.datasets 가 비어 있다. 항상 SET 으로 준다 (후보: configs/training/datasets.md).
#   GPUS=8 SET="data.datasets=[<이름>]" bash z_training/train.sh <config>            # configs/training/<config>.yaml
#   GPUS=8 NAME=my_run SET="data.datasets=[<이름>] optimization.lr=1e-4" bash z_training/train.sh frozen_predictor_postft
#   LIMIT=16 SET="data.datasets=[<이름>]" DEBUG=1 bash z_training/train.sh debug     # 단일 프로세스, GPU 1장 (--debugmode)
#   DRYRUN=1 SET="data.datasets=[<이름>]" bash z_training/train.sh frozen_predictor_scratch   # 병합·검사만 (몇 초)
#   GPUS=2 bash z_training/train.sh --smoke-ddp                  # 모델·데이터 없이 spawn/포트/join 점검
#
# 하는 일
#   1. resolve_train.py 로 configs/training/<config>.yaml (+extends) + 레지스트리를 병합해
#      <run>/config.yaml 로 남긴다 (모델 로드 전에 경로·프레임·정렬을 검사한다)
#   2. 빈 DDP 포트를 찾아 TRAIN_DDP_PORT 로 넘기고 TRAIN_EXPECT_WS 로 split-brain 을 막는다
#   3. rank 당 CPU 스레드를 제한한다 (rank 마다 전체 코어를 잡으면 decode 가 7배 느려진다)
#   4. python z_training/harness/launch.py --fname <run>/config.yaml --devices cuda:0..N-1
#      (app/main.py 는 건드리지 않는다 — 포트·join 이 필요해서 런처를 따로 뒀다)
#
# 환경변수
#   GPUS=N        (기본 1)          NAME=      run 이름 (기본 config 이름)   -> z_training/runs/<NAME>
#   FOLDER=       run 폴더 직접 지정                SET="a.b=1 c.d=null"   병합 config 점 경로 덮어쓰기
#   LIMIT=N       모든 frames_index 데이터셋을 앞 N 클립으로 (스모크). NAME 을 안 주면 <config>_smoke{N} 폴더로 간다
#   SET 값에 dict/list 를 쓸 때는 JSON 식으로 따옴표를 쓴다 (공백 금지, YAML flow 의 {a:b} 는 잘못 읽힌다):
#     SET='data.datasets=[{"name":"v11_possible","exclude":{"condition":["static_occlusion"]}}]'
#   DEBUG=1       --debugmode (프로세스 1개)        DRYRUN=1   병합만
#   THREADS=      rank 당 OMP 스레드 (기본 min(12, nproc/GPUS))
#   TRAIN_DDP_PORT=   지정 시 탐색 생략
#
# 이어 돌리기: 같은 NAME 으로 다시 부르면 <run>/latest.pt 에서 자동 재개한다 (meta.auto_resume).
# 채점:      bash z_training/eval.sh <NAME>
# -----------------------------------------------------------------------------
set -euo pipefail
PY=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
PROJ=/data/hyuntak/project/2026/2027_cvpr/vjepa2

if [[ "${1:-}" == "--list" || $# -eq 0 ]]; then
  echo "config (configs/training):"; ls "$PROJ"/configs/training/*.yaml | xargs -n1 basename | sed 's/\.yaml$//;s/^/  /'
  echo "학습 데이터셋 (configs/training/datasets.md):"; grep '^## ' "$PROJ/configs/training/datasets.md" | sed 's/^## /  /'
  echo "run (z_training/runs):"; ls "$PROJ/z_training/runs" 2>/dev/null | sed 's/^/  /'
  exit 0
fi

if [[ "${1:-}" == "--smoke-ddp" ]]; then          # 모델·데이터 없이 spawn/포트/join 점검
  cd "$PROJ"; export PYTHONPATH="$PROJ:${PYTHONPATH:-}"; GPUS=${GPUS:-2}
  export TRAIN_DDP_PORT=${TRAIN_DDP_PORT:-$(( 30000 + ( ${SLURM_JOB_ID:-$$} % 10000 ) ))} TRAIN_EXPECT_WS=$GPUS
  exec "$PY" z_training/harness/launch.py --smoke_ddp --devices $(for i in $(seq 0 $((GPUS-1))); do echo -n "cuda:$i "; done)
fi
CONFIG=${1:?"사용법: train.sh <config>   (목록: train.sh --list)"}
cd "$PROJ"; export PYTHONPATH="$PROJ:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GPUS=${GPUS:-1}
[[ -n "${DEBUG:-}" ]] && GPUS=1
NAME_GIVEN=${NAME:-}
NAME=${NAME:-$(basename "${CONFIG%.yaml}")}
[[ -n "${LIMIT:-}" && -z "$NAME_GIVEN" ]] && NAME="${NAME}_smoke${LIMIT}"   # 스모크가 본 run 의 latest.pt 를 안 건드리게 (run.sh 와 같다)
FOLDER=${FOLDER:-$PROJ/z_training/runs/$NAME}
if [[ -n "${DRYRUN:-}" ]]; then                    # DRYRUN 은 run 폴더를 만들지 않는다
  CFG=$(mktemp /tmp/train_XXXXXX.yaml); trap 'rm -f "$CFG"' EXIT
else
  mkdir -p "$FOLDER"; CFG=$FOLDER/config.yaml
fi

echo "=================================================="
SET_ARGS=()
for kv in ${SET:-}; do SET_ARGS+=(--set "$kv"); done
[[ -n "${LIMIT:-}" ]] && SET_ARGS+=(--limit "$LIMIT")
"$PY" z_training/harness/resolve_train.py "$CONFIG" -o "$CFG" --folder "$FOLDER" "${SET_ARGS[@]}"
DEVICES=$(for i in $(seq 0 $((GPUS-1))); do echo -n "cuda:$i "; done)
echo "  devices  : $DEVICES (world_size=$GPUS)"
[[ -f "$FOLDER/latest.pt" ]] && echo "  resume   : $FOLDER/latest.pt 가 있다 -> 이어서 돈다 (새로 시작하려면 다른 NAME)"
echo "=================================================="
if [[ -n "${DRYRUN:-}" ]]; then echo "--- DRYRUN: 병합된 config ---"; cat "$CFG"; exit 0; fi

# ── DDP 포트: run.sh 와 같은 방식 (SLURM job id 로 갈라 잡아 같은 노드의 다른 job 과 안 겹친다) ──
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
