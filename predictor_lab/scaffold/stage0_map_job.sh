#!/usr/bin/env bash
# scaffold 2 판 0 단계 sbatch 본체 (NFS 에 둔다 — /tmp 는 노드 로컬이라 다른 노드 job 에서 안 보인다)
#   STEPS=20 EVAL_LIMIT=48 TAG=smoke GPUS=1 NODE=vll2 JOBNAME=scaf_map_smoke bash cvpr/harness/submit.sh predictor_lab/scaffold/stage0_map_job.sh train
set -euo pipefail
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
exec /data/hyuntak/anaconda3/envs/vjepa2/bin/python -u predictor_lab/scaffold/stage0_map.py "$@"
