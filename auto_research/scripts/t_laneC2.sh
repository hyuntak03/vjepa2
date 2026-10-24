#!/bin/bash
# [T] 장난감 요인 실험 이어서 (GPU 1). r_j0 과 3D 합성 흐름 측정이 끝나길 기다린다 (GPU 1 동시 2 개 이하).
cd /workspace/world_model/vjepa2
busy() { pgrep -f "name r_j0 --rope" > /dev/null || pgrep -f "src phys3d --out" > /dev/null; }
while busy; do sleep 10; done
run() { n=$1; shift; CUDA_VISIBLE_DEVICES=1 /venv/vjepa/bin/python auto_research/scripts/toy_factor.py --name $n --rope 1 --obj_fixed 1.0 --steps 6000 "$@" > auto_research/logs/toy_factor_$n.log 2>&1; }
run r_j06 --jit 0.6
run r_j10 --jit 1.0
run r_s05_j0 --jit 0 --sig_a 0.5
run r_j10_tclean --jit 1.0 --jit_tgt 0
run r_j03 --jit 0.3
run r_s05_j06 --jit 0.6 --sig_a 0.5
