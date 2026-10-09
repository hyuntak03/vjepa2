#!/bin/bash
#SBATCH -J ar_h23
#SBATCH -w vll6
#SBATCH -p batch_vll
#SBATCH --gres=gpu:8
#SBATCH -c 64
#SBATCH --mem=320G
#SBATCH -t 8:00:00
#SBATCH -o /data/hyuntak/project/2026/2027_cvpr/vjepa2/auto_research/logs/h23_extract_%j.out
# H2+H3 추출 — RollOut_v3 16 창 × predictor 12 층 렌즈. 본 스크립트: h23_extract_v3.py
set -euo pipefail
source /data/hyuntak/anaconda3/bin/activate vjepa2
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
python auto_research/scripts/h23_extract_v3.py --gpus 8 --workers 6 "$@"
