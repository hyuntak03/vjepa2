#!/bin/bash
# [T] 관성 5 % 판이 끝나면 2 % 판 (GPU 7). B′ 흐름 쪽 준비는 리더 지시로 하지 않는다 (KST 03:43).
cd /workspace/world_model/vjepa2
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
while pgrep -f "toy_inertia.py --name in_disp_05" > /dev/null; do sleep 10; done
TB="--rope 1 --obj_fixed 1.0 --jit 0.6 --vmin 1 --vmax 2 --steps 6000 --head disp --disp_loss 3"
grep -q "done" auto_research/logs/toy_inertia_in_disp_05.log 2>/dev/null || env CUDA_VISIBLE_DEVICES=7 /venv/vjepa/bin/python -W ignore auto_research/scripts/toy_inertia.py --name in_disp_05 --inert_frac 0.05 $TB > auto_research/logs/toy_inertia_in_disp_05.log 2>&1
env CUDA_VISIBLE_DEVICES=7 /venv/vjepa/bin/python -W ignore auto_research/scripts/toy_inertia.py --name in_disp_02 --inert_frac 0.02 $TB > auto_research/logs/toy_inertia_in_disp_02.log 2>&1
echo "toys done $(date -u +%H:%M)"
