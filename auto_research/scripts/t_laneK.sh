#!/bin/bash
# [T] 갈래 세계: ξ 척도를 참값 (0.6) 으로 고정한 분포 예측 — 분해가 되면 g 가 갈래를 진하게 그리나. 대기열 I 의 b_j06_crpsxi 뒤.
cd /workspace/world_model/vjepa2
while pgrep -f "name b_j06_crpsxi " > /dev/null; do sleep 10; done
CUDA_VISIBLE_DEVICES=1 /venv/vjepa/bin/python auto_research/scripts/toy_factor.py --name b_j06_crpsxi_fix --rope 1 --obj_fixed 1.0 --steps 6000 --p_stop 0.4 --vmin 1.0 --vmax 2.0 --branch_eval 1 --jit 0.6 --loss crps --xi gauss --xi_fix 0.6 > auto_research/logs/toy_factor_b_j06_crpsxi_fix.log 2>&1
CUDA_VISIBLE_DEVICES=1 /venv/vjepa/bin/python auto_research/scripts/toy_factor.py --name b_j0_crps --rope 1 --obj_fixed 1.0 --steps 6000 --p_stop 0.4 --vmin 1.0 --vmax 2.0 --branch_eval 1 --jit 0 --loss crps > auto_research/logs/toy_factor_b_j0_crps.log 2>&1
