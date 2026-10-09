#!/bin/bash
#SBATCH -w vll1
#SBATCH -p batch_vll
#SBATCH --gres=gpu:2
#SBATCH -c 12
#SBATCH --mem=64G
#SBATCH -t 8:00:00
# vll1 4 GPU 판. CMD 안의 스크립트는 G=4 를 읽는다.
set -uo pipefail
source /data/hyuntak/anaconda3/bin/activate vjepa2
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
export G=2
echo "[$(date)] $(hostname) $CMD"; eval "$CMD"; echo "[$(date)] exit $?"
