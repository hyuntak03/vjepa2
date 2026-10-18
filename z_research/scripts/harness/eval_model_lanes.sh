#!/bin/bash
# 모델 하나를 평가 행렬 벤치에 빠르게 돌린다 — vll5 내 GPU 4 장을 두 줄로 (2026-10-17).
#   setsid nohup bash z_research/scripts/harness/eval_model_lanes.sh <모델> A|B > cvpr/logs/<이름>.log 2>&1 &
#   줄 A (GPU 0 1): gravity → VOBench 전체 → IntPhys 1 (garrido_w16 · w32)
#   줄 B (GPU 2 3): IntPhys 2 ip2_w32 → ip2_w16          (ip2_w48 은 다른 노드 sbatch: eval_matrix_node.sh + WINDOWS=ip2_w48)
# 칸 세는 법은 z_research/VOBench_v1/Archive/EVAL_MATRIX_2026-10-15.md (VOBench late = split v2 test 블록, gravity = vobench_gravity_pick.py --test).
set -uo pipefail
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
M=${1:?모델}; LANE=${2:?A|B}
log() { echo "[lane $LANE $(date +%m-%d_%H:%M:%S)] $*"; }
run() { local name=$1; shift; NJOB=$((NJOB + 1)); export EVAL_DDP_PORT=$(( PORT_BASE + NJOB ))
        log "시작 $name (port $EVAL_DDP_PORT)"; "$@"; log "  끝 $name rc=$?"; }
NJOB=0; export GPUS=2
case $LANE in
  A) export GPU_IDS="0 1"; PORT_BASE=25100
     run gravity  env RESULTS_ROOT=legacy bash cvpr/eval/testbed/run.sh vobench_v1_gravity "$M"
     run vobench  env RESULTS_ROOT=legacy bash cvpr/eval/testbed/run.sh vobench_v1 "$M"
     run intphys1 bash cvpr/eval/intphys1/run.sh intphys1_dev "$M" ;;
  B) export GPU_IDS="2 3"; PORT_BASE=25200
     for w in ip2_w32 ip2_w16; do run "intphys2_$w" env WINDOWS=$w bash cvpr/eval/intphys2/run.sh intphys2_main "$M"; done ;;
esac
log "줄 $LANE 전부 끝"
