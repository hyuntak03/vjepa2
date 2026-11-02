#!/usr/bin/env bash
# 경로 선생 진단 1/2 sbatch 본체 (NFS 에 둔다)
#   GPUS=1 NODE=vll2 JOBNAME=route_extract bash cvpr/harness/submit.sh predictor_lab/scaffold/route_job.sh 200
set -euo pipefail
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
PY=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
[[ $(hostname) == vll7* ]] && PY=/data/hyuntak/anaconda3/envs/vjepa2_bw/bin/python   # Blackwell
exec $PY -u predictor_lab/scaffold/route_extract.py "$@"
