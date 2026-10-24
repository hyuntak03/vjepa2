#!/bin/bash
# [T] 낮은 우선순위 장난감: 블록 노름 energy score (결합 점수) 가 갈래를 나누나 — 대기열 N 뒤 (GPU 1, 하나씩)
cd /workspace/world_model/vjepa2
export OMP_NUM_THREADS=8
while pgrep -f "t_laneN.sh" > /dev/null; do sleep 15; done
run() { n=$1; shift; CUDA_VISIBLE_DEVICES=1 /venv/vjepa/bin/python auto_research/scripts/toy_factor.py --name $n --rope 1 --obj_fixed 1.0 --steps 6000 --p_stop 0.4 --vmin 1.0 --vmax 2.0 --branch_eval 1 --jit 0 "$@" > auto_research/logs/toy_factor_$n.log 2>&1; }
run b_j0_es2 --loss es --ns 2
run b_j0_es4 --loss es --ns 4
