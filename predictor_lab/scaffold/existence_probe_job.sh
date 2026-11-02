#!/usr/bin/env bash
# 자리 허용 채점 (sbatch 본체, NFS). vll7 은 쓰지 않는다 (--exclude=vll7).
set -euo pipefail
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
nvidia-smi --query-gpu=name --format=csv,noheader
exec /data/hyuntak/anaconda3/envs/vjepa2/bin/python -u predictor_lab/scaffold/existence_probe.py run
