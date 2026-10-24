#!/bin/bash
# [P] H1 장난감 대조 — 하나씩 (GPU 하나 · 스레드 8)
cd /workspace/world_model/vjepa2
P=auto_research/scripts/toy_disp.py
B="--rope 1 --obj_fixed 1.0 --steps 6000 --head disp"
run() { n=$1; shift; OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=${GPU:-5} /venv/vjepa/bin/python $P --name $n $B "$@" > auto_research/logs/toy_disp_$n.log 2>&1; echo "done $n $(date +%H:%M)"; }
run dl_b_j06     --disp_loss 1 --jit 0.6 --p_stop 0.4 --vmin 1 --vmax 2 --branch_eval 1
run dl_b_j06_p06 --disp_loss 1 --jit 0.6 --p_stop 0.6 --vmin 1 --vmax 2 --branch_eval 1
run d_b_j06_p06  --jit 0.6 --p_stop 0.6 --vmin 1 --vmax 2 --branch_eval 1
run d_r_s05_j0   --jit 0 --sig_a 0.5
run dl_r_s05_j0  --disp_loss 1 --jit 0 --sig_a 0.5
