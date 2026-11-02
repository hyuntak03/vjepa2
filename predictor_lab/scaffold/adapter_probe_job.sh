#!/usr/bin/env bash
# 운동 adapter 학습 없이 채점 (sbatch 본체, NFS). vll7 은 쓰지 않는다 (--exclude=vll7).
#   sbatch -p batch_vll --exclude=vll7 --gres=gpu:1 ... predictor_lab/scaffold/adapter_probe_job.sh [패널 …]
set -euo pipefail
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
nvidia-smi --query-gpu=name --format=csv,noheader
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
$P -u predictor_lab/scaffold/adapter_probe.py run "$@"
$P -u predictor_lab/scaffold/adapter_probe.py summary
