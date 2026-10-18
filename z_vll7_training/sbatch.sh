#!/bin/bash
# =============================================================================
# SLURM 제출 (vll7, 6 GPU). **한 파일이 두 역할**을 한다 — 사람이 bash 로 부르면 스스로를 제출하고,
# SLURM 이 실행하면 본체를 돈다. 우리가 직접 붙이는 `TRAIN_RUN=1` 로 갈린다.
#
#   GPUS=6 bash z_vll7_training/sbatch.sh ctxt_video_tgtvideo
#   squeue -u $USER ; tail -f z_vll7_training/slurm_logs/<jobid>.out
#
# ⚠️ **TRAIN_RUN=1 없이 직접 sbatch 하면 본체가 안 돌고 제출만 한다** (z_training/sbatch.sh 와 같은 규약).
#    2026 초에 SLURM_BATCH_SCRIPT 가 batch job 에서 비어 자기제출이 폭주해 9,924 개가 쌓인 적이 있다 —
#    **SLURM 이 무엇을 넣어 주는지에 기대지 않는다. 표시는 우리가 붙인다.**
# ⚠️ `--export` 는 콤마가 구분자다. SET 에 리스트가 있으면 잘린다 -> base64 (SET_B64) 로 싣는다.
# =============================================================================
set -euo pipefail
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${TRAIN_RUN:-}" ]]; then                       # ── 제출 분기 ──
  source "$_HERE/env.sh"
  CONFIG=${1:?"사용법: GPUS=6 bash z_vll7_training/sbatch.sh <config>"}
  GPUS=${GPUS:-$WM_SLURM_GPUS}
  NAME=${NAME:-$(basename "${CONFIG%.yaml}")}
  mkdir -p "$_HERE/slurm_logs" "$VLL7_RUNS/$NAME"
  EXPORTS="ALL,TRAIN_RUN=1,CONFIG=$CONFIG,NAME=$NAME,GPUS=$GPUS"
  [[ -n "${SET:-}"   ]] && EXPORTS="$EXPORTS,SET_B64=$(printf %s "$SET" | base64 -w0)"
  [[ -n "${LIMIT:-}" ]] && EXPORTS="$EXPORTS,LIMIT=$LIMIT"
  exec sbatch --job-name="v7_$NAME" --partition="$WM_SLURM_PARTITION" -w "$WM_SLURM_NODE" \
       --gres=gpu:$GPUS --cpus-per-task=$(( GPUS * 12 )) --mem=0 --time=7-00:00:00 \
       --output="$_HERE/slurm_logs/%j.out" --export="$EXPORTS" "$_HERE/sbatch.sh"
fi

# ── 본체 (SLURM 안) ──
if [[ -n "${SLURM_JOB_ID:-}" && -n "${OUTDIR:-}" ]]; then
  echo "tripwire: SLURM_JOB_ID 와 OUTDIR 이 함께 있다 — 자기제출 폭주 방지로 거부한다" >&2; exit 1
fi
source /data/hyuntak/anaconda3/bin/activate vjepa2_bw
source "$_HERE/env.sh"
cd "$VJEPA2_ROOT"
[[ -n "${SET_B64:-}" ]] && export SET="$(printf %s "$SET_B64" | base64 -d)"
echo "node $(hostname) | job ${SLURM_JOB_ID:-?} | GPUS=$GPUS | config=$CONFIG | NAME=$NAME${SET:+ | SET=$SET}"
exec bash z_vll7_training/train.sh "$CONFIG"
