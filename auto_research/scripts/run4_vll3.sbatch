#!/bin/bash
#SBATCH -w vll3
#SBATCH -p batch_vll
#SBATCH --gres=gpu:4
#SBATCH -c 32
#SBATCH --mem=120G
#SBATCH -t 8:00:00
# vll3 4 GPU 판. CMD 안의 스크립트는 G=4 를 읽는다 (chain_*.sh 의 --gpus ${G:-8}).
set -uo pipefail
source /data/hyuntak/anaconda3/bin/activate vjepa2
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
export G=4
echo "[$(date)] $(hostname) $CMD"; eval "$CMD"; echo "[$(date)] exit $?"
