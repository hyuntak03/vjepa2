#!/bin/bash
# [T] 갈래 세계 대조군 (GPU 1, 한 번에 하나): 다중 가설 (승자 독식) → 잡음 0 분포 예측. CPU 스레드 제한.
cd /workspace/world_model/vjepa2
export OMP_NUM_THREADS=8
while pgrep -f "name b_j06_crpsxi_fix" > /dev/null; do sleep 10; done
run() { n=$1; shift; CUDA_VISIBLE_DEVICES=1 /venv/vjepa/bin/python auto_research/scripts/toy_factor.py --name $n --rope 1 --obj_fixed 1.0 --steps 6000 --p_stop 0.4 --vmin 1.0 --vmax 2.0 --branch_eval 1 "$@" > auto_research/logs/toy_factor_$n.log 2>&1; }
run b_j06_wta --jit 0.6 --loss wta --nh 3
run b_j0_crps --jit 0 --loss crps
