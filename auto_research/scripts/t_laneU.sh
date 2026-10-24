#!/bin/bash
# [T] R5 · R6 규제 장난감 (GPU 1, 스레드 4, 하나씩 — 리더 KST 04:5x). GPU 7 에서 렌더와 같이 돌리자 렌더가 느려져 옮겼다.
cd /workspace/world_model/vjepa2
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
B="--rope 1 --obj_fixed 1.0 --vmin 1 --vmax 2 --jit 0.6 --steps 6000"
run() { local nm=$1; shift; env CUDA_VISIBLE_DEVICES=1 /venv/vjepa/bin/python -W ignore "$@" > auto_research/logs/toy_reg56_$nm.log 2>&1; echo "done $nm $(date -u +%H:%M)"; }
for arm in "c:" "r5a:--r5 0.1" "r5b:--r5 0.5" "r6a:--r6 0.5" "r6b:--r6 2"; do
  n=${arm%%:*}; f=${arm#*:}
  [ -f auto_research/results/toy_reg56/rs10_$n.json ] || run rs_$n auto_research/scripts/toy_factor.py --name rs10_$n $B --soft_c 0.3 --rigid_frac 0.1 $f --out auto_research/results/toy_reg56
  [ -f auto_research/results/toy_reg56/in10_$n.json ] || run in_$n auto_research/scripts/toy_inertia.py --name in10_$n $B --inert_frac 0.1 $f --out auto_research/results/toy_reg56
done
echo "all done $(date -u +%H:%M)"
