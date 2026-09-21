#!/bin/bash
#SBATCH --job-name=ctxdir
#SBATCH --partition=batch_vll
#SBATCH -w vll3
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=90G
#SBATCH --time=06:00:00
#SBATCH --output=/data/hyuntak/project/2026/2027_cvpr/vjepa2/z_research/scripts/slurm_logs/%x_%j.out
# ctxenc_direction_probe.py 를 SLURM 으로. 사람이 `bash` 로 부르면 스스로를 제출하고, SLURM 안에서는 본체를 돈다.
#
#   bash z_research/scripts/analysis/ctxenc_direction_sbatch.sh                     # 두 데이터셋 (ssv2, ntu)
#   ARGS="--sets ssv2 --smoke 8" bash z_research/scripts/analysis/ctxenc_direction_sbatch.sh
#   NODE=vll5 bash z_research/scripts/analysis/ctxenc_direction_sbatch.sh
#
# 두 역할은 우리가 붙이는 ANA_RUN=1 로 갈린다 (SLURM 변수에 기대면 자기제출 폭주가 난다, CLAUDE.md §7-1).
# -w vll3 고정 — 영상이 그 노드의 /local_datasets/vlm_direction 에 있다 (노드 로컬).
# 메모리: 특징을 디스크가 아니라 RAM 에 든다 (정방향+역재생 = ssv2 15 GB / ntu 9 GB).
set -euo pipefail
PROJ=/data/hyuntak/project/2026/2027_cvpr/vjepa2
PY=/data/hyuntak/anaconda3/envs/vjepa2/bin/python

if [[ -z "${ANA_RUN:-}" ]]; then
  [[ -n "${SLURM_JOB_ID:-}" ]] && echo "note: 바깥 SLURM_JOB_ID=$SLURM_JOB_ID 는 제출에 쓰지 않는다" >&2
  EXPORTS="ALL,ANA_RUN=1"
  [[ -n "${ARGS:-}" ]] && EXPORTS="$EXPORTS,ARGS_B64=$(printf %s "$ARGS" | base64 -w0)"
  mkdir -p "$PROJ/z_research/scripts/slurm_logs"
  exec env -u SLURM_JOB_ID sbatch ${NODE:+-w "$NODE"} --export="$EXPORTS" "$PROJ/z_research/scripts/analysis/ctxenc_direction_sbatch.sh"
fi

source /data/hyuntak/anaconda3/bin/activate vjepa2
[[ -n "${ARGS_B64:-}" ]] && ARGS=$(printf %s "$ARGS_B64" | base64 -d)
cd "$PROJ"
echo "[sbatch] job=$SLURM_JOB_ID node=$(hostname) ARGS=${ARGS:-}"
exec "$PY" z_research/scripts/analysis/ctxenc_direction_probe.py ${ARGS:-}
