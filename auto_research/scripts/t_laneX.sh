#!/bin/bash
# [T] 흐름 공간 조건화 장난감 (관성 / 제멋대로), GPU 7 하나씩 · 스레드 8
cd /workspace/world_model/vjepa2
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
P=auto_research/scripts/toy_inertia.py
B="--rope 1 --obj_fixed 1.0 --jit 0.6 --vmin 1 --vmax 2 --steps 6000"
run() { n=$1; shift; CUDA_VISIBLE_DEVICES=7 /venv/vjepa/bin/python -W ignore $P --name $n $B "$@" > auto_research/logs/toy_inertia_$n.log 2>&1; echo "done $n $(date -u +%H:%M)"; }
run in_tok_25 --inert_frac 0.25
run in_disp_25 --inert_frac 0.25 --head disp --disp_loss 3
run in_tok_10 --inert_frac 0.10
run in_disp_10 --inert_frac 0.10 --head disp --disp_loss 3
