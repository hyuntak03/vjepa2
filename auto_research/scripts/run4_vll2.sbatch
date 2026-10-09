#!/bin/bash
#SBATCH -w vll2
#SBATCH -p batch_vll
#SBATCH --gres=gpu:4
#SBATCH -c 24
#SBATCH --mem=96G
#SBATCH -t 8:00:00
# vll2 4 GPU 판 (run4_vll3.sbatch 의 vll2 판). CMD 안의 스크립트는 G=4 를 읽는다.
set -uo pipefail
source /data/hyuntak/anaconda3/bin/activate vjepa2
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
export G=4
echo "[$(date)] $(hostname) $CMD"; eval "$CMD"; echo "[$(date)] exit $?"
