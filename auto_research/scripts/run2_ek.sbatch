#!/bin/bash
#SBATCH --exclude=vll1,vll2,vll4,vll7
#SBATCH -p batch_vll
#SBATCH --gres=gpu:2
#SBATCH -c 12
#SBATCH --mem=64G
#SBATCH -t 8:00:00
# vll3/5/6 아무 데나 4 GPU (EK100 · v3 가 있는 노드). CMD 안의 스크립트는 G=4 를 읽는다.
set -uo pipefail
source /data/hyuntak/anaconda3/bin/activate vjepa2
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
export G=2
echo "[$(date)] $(hostname) $CMD"; eval "$CMD"; echo "[$(date)] exit $?"
