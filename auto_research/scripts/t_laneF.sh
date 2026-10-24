#!/bin/bash
# [T] 갈래 세계 (간다 0.6 / 멈춘다 0.4, 문맥으로 구별 불가): 칸별 L1 vs 분포 예측. GPU 1, 2 줄.
cd /workspace/world_model/vjepa2
run() { n=$1; shift; CUDA_VISIBLE_DEVICES=1 /venv/vjepa/bin/python auto_research/scripts/toy_factor.py --name $n --rope 1 --obj_fixed 1.0 --steps 6000 --p_stop 0.4 --vmin 1.0 --vmax 2.0 --branch_eval 1 "$@" > auto_research/logs/toy_factor_$n.log 2>&1; }
run b_j06_l1 --jit 0.6
run b_j0_l1 --jit 0
