#!/bin/bash
# [T] 갈래 세계 장난감 이어서 (GPU 1): b_j06_l1 이 끝나면 → 분포 예측 (g + s⊙ξ) → 잡음 0 판들
cd /workspace/world_model/vjepa2
while pgrep -f "name b_j06_l1 " > /dev/null; do sleep 10; done
run() { n=$1; shift; CUDA_VISIBLE_DEVICES=1 /venv/vjepa/bin/python auto_research/scripts/toy_factor.py --name $n --rope 1 --obj_fixed 1.0 --steps 6000 --p_stop 0.4 --vmin 1.0 --vmax 2.0 --branch_eval 1 "$@" > auto_research/logs/toy_factor_$n.log 2>&1; }
run b_j06_crpsxi --jit 0.6 --loss crps --xi gauss
run b_j0_crps --jit 0 --loss crps
run b_j0_l1 --jit 0
