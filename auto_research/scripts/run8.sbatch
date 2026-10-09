#!/bin/bash
#SBATCH -w vll6
#SBATCH -p batch_vll
#SBATCH --gres=gpu:8
#SBATCH -c 64
#SBATCH --mem=160G   # 300G 는 CPU 분석 (srun6) 자리를 막았다 — 실사용은 100G 안팎
#SBATCH -t 8:00:00
# 범용: GPU 4 장으로 CMD 를 돈다.  sbatch -J <이름> -o <로그> --export=ALL,CMD="python ..." run4.sbatch
set -uo pipefail
source /data/hyuntak/anaconda3/bin/activate vjepa2
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
echo "[$(date)] $CMD"; eval "$CMD"; echo "[$(date)] exit $?"
