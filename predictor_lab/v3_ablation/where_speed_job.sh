#!/bin/bash
# 가르는 실험 두 개 (자리 확인 · 속도 묶음) — GPU 1 장 (2026-11-01). 설명은 where_speed.py 머리.
#   sbatch -p batch_vll -w vll4 --gres=gpu:1 -c 12 --mem=60G -t 2:00:00 -J where_speed \
#          -o predictor_lab/v3_ablation/logs/%j_where.out predictor_lab/v3_ablation/where_speed_job.sh
set -u
R=/data/hyuntak/project/2026/2027_cvpr/vjepa2
cd "$R"
cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
if [[ ${cc%%.*} -ge 12 ]]; then P=/data/hyuntak/anaconda3/envs/vjepa2_bw/bin/python; else P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python; fi
echo "[where_speed] $(hostname) $(date '+%F %T') GPU compute_cap $cc → $P"
$P predictor_lab/v3_ablation/where_speed.py extract ${SETS_LIST:-}
echo WHERE_SPEED_DONE
