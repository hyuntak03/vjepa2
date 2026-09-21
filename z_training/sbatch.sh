#!/bin/bash
#SBATCH --job-name=train
#SBATCH --partition=batch_vll
#SBATCH -w vll5
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=45G
#SBATCH --time=1-00:00:00
#SBATCH --output=/data/hyuntak/project/2026/2027_cvpr/vjepa2/z_training/slurm_logs/%x_%j.out
#SBATCH --error=/data/hyuntak/project/2026/2027_cvpr/vjepa2/z_training/slurm_logs/%x_%j.err
# -----------------------------------------------------------------------------
# train.sh 를 SLURM 으로 제출한다. z_research/scripts/sbatch.sh 와 같은 **두 역할 한 파일** 구조.
#
#   bash z_training/sbatch.sh <config>                          사람이 실행 -> 스스로를 제출한다
#   GPUS=8 NAME=run1 SET="optimization.lr=1e-4" bash z_training/sbatch.sh frozen_predictor_postft
#   TIME=0-12:00:00 GPUS=4 bash z_training/sbatch.sh frozen_predictor_scratch
#
# **우리가 직접 붙이는 TRAIN_RUN=1** 로 제출/본체가 갈린다 (SLURM 변수에 기대지 않는다 —
# 2026-08-30 자기제출 폭주 사고 참고, z_research/scripts/sbatch.sh 머리말).
# SET 은 콤마를 담을 수 있어 base64 로 싣는다 (--export 는 콤마가 구분자다).
# 데이터가 vll5 로컬(/local_datasets)이라 기본 -w vll5. 다른 노드는 NODE=vll3 (명령줄 -w 가 #SBATCH -w 를 이긴다) —
# 그 노드에 학습 데이터 경로가 있는지는 직접 확인할 것. --gres 와 GPUS 를 맞춘다.
#   NODE=vll3 GPUS=8 CPUS_PER_GPU=10 MEM_PER_GPU=36G bash z_training/sbatch.sh intphys2_postft
#   (vll3 = CPU 96 · 메모리 336G 라 기본 12 CPU · 45G/GPU × 8 은 제출이 거부된다 — 2026-09-19)
# -----------------------------------------------------------------------------
set -euo pipefail
PROJECT=/data/hyuntak/project/2026/2027_cvpr/vjepa2

if [[ -z "${TRAIN_RUN:-}" ]]; then
  cd "$PROJECT"
  if [[ -n "${SLURM_JOB_ID:-}" && -n "${FOLDER:-}" && -z "${TRAIN_FORCE:-}" ]]; then
    echo "거부: SLURM job($SLURM_JOB_ID) 안에서 FOLDER=$FOLDER 를 들고 제출 분기에 들어왔다 (TRAIN_RUN 누락 = 자기제출 폭주 신호)." >&2
    exit 1
  fi
  unset SLURM_JOB_ID SLURM_JOBID SLURM_NODELIST SLURM_JOB_NODELIST \
        SLURM_NNODES SLURM_JOB_NUM_NODES SLURM_NTASKS SLURM_TASKS_PER_NODE \
        SLURM_GPUS_ON_NODE SLURM_JOB_GPUS SLURM_MEM_PER_NODE SLURM_CPUS_ON_NODE
  CONFIG=${1:?"사용법: sbatch.sh <config>"}
  G=${GPUS:-8}; NAME_GIVEN=${NAME:-}
  NAME=${NAME:-$(basename "${CONFIG%.yaml}")}
  [[ -n "${LIMIT:-}" && -z "${NAME_GIVEN:-}" ]] && NAME="${NAME}_smoke${LIMIT}"
  FOLDER=${FOLDER:-$PROJECT/z_training/runs/$NAME}       # 트립와이어의 지문 — 본체에 넘기고, 되돌아오면 거부된다
  mkdir -p "$PROJECT/z_training/slurm_logs"
  _b64() { printf %s "$1" | base64 -w0; }
  sbatch --job-name="${JOBNAME:-tr_$NAME}" --gres=gpu:"$G" ${TIME:+--time="$TIME"} ${NODE:+-w "$NODE"} ${CPUS_PER_GPU:+--cpus-per-gpu="$CPUS_PER_GPU"} ${MEM_PER_GPU:+--mem-per-gpu="$MEM_PER_GPU"} \
         --export=ALL,TRAIN_RUN=1,CONFIG="$CONFIG",NAME="$NAME",FOLDER="$FOLDER",GPUS="$G",LIMIT="${LIMIT:-}",SET_B64="$(_b64 "${SET:-}")" \
         z_training/sbatch.sh
  echo "진행: tail -f z_training/runs/$NAME/train.log   |   squeue -u \$USER"
  exit 0
fi

# ── SLURM 이 실행하는 본체 ────────────────────────────────────────────────────
CONFIG=${CONFIG:?"--export=ALL,TRAIN_RUN=1,CONFIG=<config> 필요"}
source /data/hyuntak/anaconda3/bin/activate vjepa2
cd "$PROJECT"; export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"
[[ -n "${SET_B64:-}" ]] && SET=$(printf %s "$SET_B64" | base64 -d)
echo "node $(hostname) | gres gpus=${SLURM_GPUS_ON_NODE:-?} | config $CONFIG | NAME=${NAME:-} | GPUS=${GPUS:-8} | SET: ${SET:-(없음)}"
GPUS=${GPUS:-8} NAME="${NAME:-}" FOLDER="${FOLDER:-}" LIMIT="${LIMIT:-}" SET="${SET:-}" bash z_training/train.sh "$CONFIG"
