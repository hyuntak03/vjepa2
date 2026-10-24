#!/bin/bash
#SBATCH --job-name=pl_e1b2x2
#SBATCH --partition=batch_vll
#SBATCH --nodelist=vll5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/data/hyuntak/project/2026/2027_cvpr/vjepa2/paper_lab/_scratch/tspace/slurm_e1b_%j.log
# paper_lab 진행자 2 — E1b (입력 × 정답 떨림 2 × 2, 학습 없음, R5 반박 대응). sbatch [--dependency=afterany:<job>] paper_lab/scripts/run_e1b.sbatch
set -uo pipefail
REPO=/data/hyuntak/project/2026/2027_cvpr/vjepa2; PY=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
export TSR_OUT=$REPO/paper_lab/_scratch/tspace PYTHONUNBUFFERED=1 OMP_NUM_THREADS=6
cd $REPO
echo "[$(date '+%F %T')] job $SLURM_JOB_ID on $(hostname)"
nvidia-smi --query-gpu=index,pci.bus_id,memory.used --format=csv
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [ "${USED:-99999}" -gt 1000 ]; then echo "할당 GPU 가 비어 있지 않다 (${USED} MiB) — 멈춘다"; exit 3; fi
xargs -a $TSR_OUT/_frames.txt -P 32 -n 200 cat > /dev/null 2>&1
echo "[$(date '+%F %T')] vobench"; $PY paper_lab/scripts/tspace_2x2.py extract --set vobench || exit 5
echo "[$(date '+%F %T')] ssv2"; $PY paper_lab/scripts/tspace_2x2.py extract --set ssv2 --n 200 > $TSR_OUT/x2_analyze.txt 2>&1 || exit 6
cat $TSR_OUT/x2_analyze.txt | grep -vE "warn|Warning"
echo "[$(date '+%F %T')] done"
