#!/bin/bash
#SBATCH --job-name=presence
#SBATCH --partition=batch_vll
#SBATCH -w vll5
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=/data/hyuntak/project/2026/2027_cvpr/vjepa2/z_research/scripts/slurm_logs/%x_%j.out
# rollout2_presence_readout.py 를 SLURM 으로. 사람이 `bash` 로 부르면 스스로를 제출하고, SLURM 안에서는 본체를 돈다.
#
#   bash z_research/scripts/analysis/rollout2_presence_sbatch.sh
#   ARGS="--reps p --epochs 100" bash z_research/scripts/analysis/rollout2_presence_sbatch.sh
#   NODE=vll3 bash z_research/scripts/analysis/rollout2_presence_sbatch.sh
#
# 두 역할은 우리가 붙이는 ANA_RUN=1 로 갈린다 (SLURM 변수에 기대면 자기제출 폭주가 난다, CLAUDE.md §7-1).
# -w vll5 고정 — 프레임이 그 노드의 /local_datasets/world/world_analysis/RollOut_v2_training 에 있다 (노드 로컬).
# --mem 200G: 특징을 디스크가 아니라 /dev/shm 에 든다 (8,352 clip x 3 표현 x 8 튜블릿 = 122 GB). tmpfs 도 cgroup 에 잡힌다.
set -euo pipefail
PROJ=/data/hyuntak/project/2026/2027_cvpr/vjepa2
PY=/data/hyuntak/anaconda3/envs/vjepa2/bin/python

if [[ -z "${ANA_RUN:-}" ]]; then
  [[ -n "${SLURM_JOB_ID:-}" ]] && echo "note: 바깥 SLURM_JOB_ID=$SLURM_JOB_ID 는 제출에 쓰지 않는다" >&2
  EXPORTS="ALL,ANA_RUN=1"
  [[ -n "${ARGS:-}" ]] && EXPORTS="$EXPORTS,ARGS_B64=$(printf %s "$ARGS" | base64 -w0)"
  mkdir -p "$PROJ/z_research/scripts/slurm_logs"
  exec env -u SLURM_JOB_ID sbatch ${NODE:+-w "$NODE"} --export="$EXPORTS" "$PROJ/z_research/scripts/analysis/rollout2_presence_sbatch.sh"
fi

source /data/hyuntak/anaconda3/bin/activate vjepa2
[[ -n "${ARGS_B64:-}" ]] && ARGS=$(printf %s "$ARGS_B64" | base64 -d)
cd "$PROJ"
trap 'rm -rf /dev/shm/rollout2_presence' EXIT      # 죽어도 122 GB 를 남기지 않는다
echo "[sbatch] job=$SLURM_JOB_ID node=$(hostname) ARGS=${ARGS:-}"
nvidia-smi --query-gpu=index,name --format=csv,noheader | head -8
exec "$PY" z_research/scripts/analysis/rollout2_presence_readout.py ${ARGS:-}
