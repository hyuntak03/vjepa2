#!/bin/bash
# [T] 변위 공간 문턱 위치 (관성 2 · 5 %, 변위 팔만) — t_laneZ 끝난 뒤 GPU 7
cd /workspace/world_model/vjepa2
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
while pgrep -f "t_laneZ.sh" > /dev/null; do sleep 15; done
TB="--rope 1 --obj_fixed 1.0 --jit 0.6 --vmin 1 --vmax 2 --steps 6000 --head disp --disp_loss 3"
for f in 05 02; do env CUDA_VISIBLE_DEVICES=7 /venv/vjepa/bin/python -W ignore auto_research/scripts/toy_inertia.py --name in_disp_$f --inert_frac 0.$f $TB > auto_research/logs/toy_inertia_in_disp_$f.log 2>&1; echo "done in_disp_$f $(date -u +%H:%M)"; done
