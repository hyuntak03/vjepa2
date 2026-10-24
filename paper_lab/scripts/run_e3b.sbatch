#!/bin/bash
#SBATCH --job-name=pl_e3bavg
#SBATCH --partition=batch_vll
#SBATCH --nodelist=vll5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=/data/hyuntak/project/2026/2027_cvpr/vjepa2/paper_lab/_scratch/e3/slurm_e3b_%j.log
# paper_lab 진행자 2 — E3b (E3 판정 단위 맞추기: 학습 없는 '가중 평균' · '마지막 블록만' 가설의 Δs). sbatch [--dependency=afterany:<E3 job>] paper_lab/scripts/run_e3b.sbatch
set -uo pipefail
REPO=/data/hyuntak/project/2026/2027_cvpr/vjepa2; PY=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
cd $REPO; export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=6
echo "[$(date '+%F %T')] job $SLURM_JOB_ID on $(hostname)"
nvidia-smi --query-gpu=index,pci.bus_id,memory.used --format=csv
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [ "${USED:-99999}" -gt 1000 ]; then echo "할당 GPU 가 비어 있지 않다 (${USED} MiB) — 멈춘다"; exit 3; fi
$PY paper_lab/scripts/e3b_avg_hypothesis.py --max-n 600 2>&1 | grep -vE "warn|Warning|xFormers|self.gen|predictor kind"
echo "[$(date '+%F %T')] done"
