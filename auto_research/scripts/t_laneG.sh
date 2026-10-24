#!/bin/bash
cd /workspace/world_model/vjepa2
while pgrep -f "name r_s05_j0_crps" > /dev/null; do sleep 10; done
run() { n=$1; shift; CUDA_VISIBLE_DEVICES=1 /venv/vjepa/bin/python auto_research/scripts/toy_factor.py --name $n --rope 1 --obj_fixed 1.0 --steps 6000 --p_stop 0.4 --vmin 1.0 --vmax 2.0 --branch_eval 1 "$@" > auto_research/logs/toy_factor_$n.log 2>&1; }
run b_j0_crps --jit 0 --loss crps
run b_j06_crps --jit 0.6 --loss crps
