#!/bin/bash
#SBATCH --job-name=pl_e3anchor
#SBATCH --partition=batch_vll
#SBATCH --nodelist=vll5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=/data/hyuntak/project/2026/2027_cvpr/vjepa2/paper_lab/_scratch/e3/slurm_%j.log
# paper_lab 진행자 2 — E3 (기준 블록 AR 마지막 관측 바꿔치기, 학습 없음). 제출: sbatch [--dependency=afterany:<E1 job>] paper_lab/scripts/run_e3.sbatch
set -uo pipefail
REPO=/data/hyuntak/project/2026/2027_cvpr/vjepa2; PY=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
cd $REPO; export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=6
echo "[$(date '+%F %T')] job $SLURM_JOB_ID on $(hostname)"
nvidia-smi --query-gpu=index,pci.bus_id,memory.used --format=csv
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [ "${USED:-99999}" -gt 1000 ]; then echo "할당 GPU 가 비어 있지 않다 (${USED} MiB) — 멈춘다"; exit 3; fi
$PY paper_lab/scripts/e3_anchor_vast70.py --bs 8 --max-n 600 2>&1 | grep -vE "warn|Warning|xFormers|self.gen|predictor kind"
echo "[$(date '+%F %T')] done"
