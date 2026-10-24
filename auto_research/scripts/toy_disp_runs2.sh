#!/bin/bash
cd /workspace/world_model/vjepa2
P=auto_research/scripts/toy_disp.py
B="--rope 1 --obj_fixed 1.0 --steps 6000 --head disp"
run() { n=$1; shift; OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=${GPU:-5} /venv/vjepa/bin/python $P --name $n $B "$@" > auto_research/logs/toy_disp_$n.log 2>&1; echo "done $n $(date +%H:%M)"; }
run dl_det_j06   --disp_loss 1 --jit 0.6 --vmin 1 --vmax 2
run dl2_b_j06    --disp_loss 2 --jit 0.6 --p_stop 0.4 --vmin 1 --vmax 2 --branch_eval 1
run dl2_det_j06  --disp_loss 2 --jit 0.6 --vmin 1 --vmax 2
