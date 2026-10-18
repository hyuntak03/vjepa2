#!/bin/bash
# vll5 sbatch GPU 4 장 job 안에서 두 줄 (2026-10-17). submit.sh 가 부른다:
#   NODE=vll5 GPUS=4 JOBNAME=<이름> bash cvpr/harness/submit.sh z_research/scripts/harness/eval_model_job4.sh <모델>
#   줄 1 (GPU 0 1): IntPhys 2 ip2_w48      줄 2 (GPU 2 3): IntPhys 1 (garrido_w16 · w32) → IntPhys 2 ip2_w16
# 같은 창이 다른 줄 (eval_model_lanes.sh) 에서 돌고 있으면 launch.sh 가드 (RUNNING_MIN) 가 건너뛴다.
set -uo pipefail
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
M=${1:?모델}; export GPUS=2
( export GPU_IDS="0 1" EVAL_DDP_PORT=25401; echo "[job4 1 $(date +%T)] ip2_w48"; WINDOWS=ip2_w48 bash cvpr/eval/intphys2/run.sh intphys2_main "$M"; echo "[job4 1 $(date +%T)] 끝 rc=$?" ) &
( export GPU_IDS="2 3"
  EVAL_DDP_PORT=25501 bash cvpr/eval/intphys1/run.sh intphys1_dev "$M"; echo "[job4 2 $(date +%T)] intphys1 끝 rc=$?"
  EVAL_DDP_PORT=25502 WINDOWS=ip2_w16 bash cvpr/eval/intphys2/run.sh intphys2_main "$M"; echo "[job4 2 $(date +%T)] ip2_w16 끝 rc=$?" ) &
wait
