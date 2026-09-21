#!/bin/bash
#SBATCH --job-name=ek100
#SBATCH --partition=batch_vll
#SBATCH -w vll5
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=45G
#SBATCH --time=2-00:00:00
#SBATCH --output=/data/hyuntak/project/2026/2027_cvpr/vjepa2/z_research/anticipation/EK100/slurm_logs/%x_%j.out
# run.sh 를 SLURM 으로. 사람이 `bash sbatch.sh` 로 부르면 스스로를 제출하고, SLURM 안에서는 run.sh 를 돈다.
#
#   GPUS=8 TAG=ek100_vith SET="..." bash z_research/anticipation/EK100/sbatch.sh
#
# 두 역할은 우리가 붙이는 ANT_RUN=1 로 갈린다 (SLURM 변수에 기대면 자기제출 폭주가 난다, CLAUDE.md §7-1).
# --export 는 콤마로 끊기므로 SET 은 base64 로 싣는다. 비디오가 노드 로컬 /data2 에 있어 -w 를 고정한다.
# resume_checkpoint 가 켜져 있어 walltime 에 걸려도 같은 TAG 로 다시 제출하면 이어진다.
set -euo pipefail
HERE=/data/hyuntak/project/2026/2027_cvpr/vjepa2/z_research/anticipation/EK100

if [[ -z "${ANT_RUN:-}" ]]; then
  [[ -n "${SLURM_JOB_ID:-}" ]] && { echo "ERROR: job 안인데 ANT_RUN 이 없다 (자기제출 방지)" >&2; exit 3; }
  GPUS=${GPUS:-8}; EXPORTS="ALL,ANT_RUN=1,GPUS=$GPUS"
  for v in TAG HEADS SMOKE VAL_ONLY; do [[ -n "${!v:-}" ]] && EXPORTS="$EXPORTS,$v=${!v}"; done
  [[ -n "${SET:-}" ]] && EXPORTS="$EXPORTS,SET_B64=$(printf %s "$SET" | base64 -w0)"
  # NODE=vll3 처럼 주면 #SBATCH -w vll5 를 덮는다 (비디오는 그 노드의 /data2 에 있어야 한다)
  # MEM_PER_GPU=40G: vll3 는 RealMemory 329 GB 라 기본 45G x 8 = 360G 로는 영원히 대기한다
  exec sbatch --job-name="ek100_${TAG:-ek100_vith}" ${NODE:+-w "$NODE"} ${MEM_PER_GPU:+--mem-per-gpu="$MEM_PER_GPU"} \
       --gres=gpu:"$GPUS" --export="$EXPORTS" "$HERE/sbatch.sh"
fi

source /data/hyuntak/anaconda3/bin/activate vjepa2
[[ -n "${SET_B64:-}" ]] && export SET=$(printf %s "$SET_B64" | base64 -d)
echo "[sbatch] job=$SLURM_JOB_ID node=$(hostname) GPUS=$GPUS TAG=${TAG:-} SET=${SET:-}"
exec bash "$HERE/run.sh"
