#!/bin/bash
# [T] 갈래 세계: 분포 예측 최적화 개선판 (표본 4 개 · 9000 step, ξ 척도 참값 고정). 대기열 L 뒤.
cd /workspace/world_model/vjepa2
export OMP_NUM_THREADS=8
while pgrep -f "t_laneL.sh" > /dev/null; do sleep 10; done
CUDA_VISIBLE_DEVICES=1 /venv/vjepa/bin/python auto_research/scripts/toy_factor.py --name b_j06_crpsxi_fix_m4 --rope 1 --obj_fixed 1.0 --steps 9000 --p_stop 0.4 --vmin 1.0 --vmax 2.0 --branch_eval 1 --jit 0.6 --loss crps --xi gauss --xi_fix 0.6 --ns 4 > auto_research/logs/toy_factor_b_j06_crpsxi_fix_m4.log 2>&1
