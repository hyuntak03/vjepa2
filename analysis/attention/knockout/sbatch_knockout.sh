#!/usr/bin/env bash
#SBATCH --job-name=ko
#SBATCH --partition=batch_vll
#SBATCH -w vll5
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=45G
#SBATCH --time=0-04:00:00
#SBATCH --output=/data/hyuntak/project/2026/2027_cvpr/vjepa2/z_research/scripts/slurm_logs/%x_%j.out
#SBATCH --error=/data/hyuntak/project/2026/2027_cvpr/vjepa2/z_research/scripts/slurm_logs/%x_%j.err
# knockout 을 SLURM 으로. run_sharded.sh 의 환경변수를 --export 로 그대로 넘긴다.
# 이 스크립트는 **스스로를 제출하지 않는다** — sbatch 로만 띄운다.
#
#   sbatch --job-name=ko_w1 --export=ALL,W=3,LAYERS=each,window,OUT=... analysis/attention/knockout/sbatch_knockout.sh
#
# ⚠️ --export 는 콤마가 구분자다. LAYERS=each,window 처럼 콤마가 든 값은 LAYERS_B64 로 싣는다.
set -euo pipefail
source /data/hyuntak/anaconda3/bin/activate vjepa2
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
[ -n "${LAYERS_B64:-}" ] && export LAYERS="$(printf %s "$LAYERS_B64" | base64 -d)"
[ -n "${EDGES_B64:-}" ] && export EDGES="$(printf %s "$EDGES_B64" | base64 -d)"
[ -n "${SPECS_B64:-}" ] && export SPECS="$(printf %s "$SPECS_B64" | base64 -d)"
export N=${N:-8}
echo "job $SLURM_JOB_ID on $(hostname)  N=$N  W=${W:-}  LAYERS=${LAYERS:-}  EDGES=${EDGES:-}  COND_RE=${COND_RE:-}  OUT=${OUT:-}"
bash analysis/attention/knockout/run_sharded.sh
