#!/bin/bash
#SBATCH -J ar_h1h6b
#SBATCH -w vll6
#SBATCH -p batch_vll
#SBATCH --gres=gpu:8
#SBATCH -c 64
#SBATCH --mem=320G
#SBATCH -t 8:00:00
#SBATCH -o /data/hyuntak/project/2026/2027_cvpr/vjepa2/auto_research/logs/h1_h6b_%j.out
# GPU 0-3: H1 추출 (h1_extract_v3.py) · GPU 4-7: H6b IntPhys1 기준선 (h6b_intphys1_baselines.py, rank 4 개)
set -uo pipefail
source /data/hyuntak/anaconda3/bin/activate vjepa2
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
G=(${CUDA_VISIBLE_DEVICES//,/ })
CUDA_VISIBLE_DEVICES=${G[0]},${G[1]},${G[2]},${G[3]} python auto_research/scripts/h1_extract_v3.py --gpus 4 --workers 6 > auto_research/logs/h1_extract_${SLURM_JOB_ID}.log 2>&1 &
P1=$!
for r in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=${G[$((r+4))]} python auto_research/scripts/h6b_intphys1_baselines.py --rank $r --world 4 > auto_research/logs/h6b_r${r}_${SLURM_JOB_ID}.log 2>&1 &
done
wait $P1; e1=$?
wait
echo "h1 exit $e1"; tail -2 auto_research/logs/h6b_r*_${SLURM_JOB_ID}.log
