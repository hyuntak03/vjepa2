#!/bin/bash
#SBATCH -w vll6
#SBATCH -p batch_vll
#SBATCH --gres=gpu:8
#SBATCH -c 48
#SBATCH --mem=200G
#SBATCH -t 12:00:00
# vll6 8 GPU 판. CMD 를 그대로 실행한다.
set -uo pipefail
source /data/hyuntak/anaconda3/bin/activate vjepa2
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
echo "[$(date)] $(hostname) $CMD"; eval "$CMD"; echo "[$(date)] exit $?"
