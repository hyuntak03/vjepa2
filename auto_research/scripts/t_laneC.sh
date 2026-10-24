#!/bin/bash
# [T] 장난감 요인 실험 (GPU 1, 차례로): 물체 방향 고정 + 회전 위치 부호화 = 운반을 배울 수 있는 장난감. 잡음 비 × 운동 불확실성 × 정답 잡음.
cd /workspace/world_model/vjepa2
run() { n=$1; shift; CUDA_VISIBLE_DEVICES=1 /venv/vjepa/bin/python auto_research/scripts/toy_factor.py --name $n --rope 1 --obj_fixed 1.0 --steps 6000 "$@" > auto_research/logs/toy_factor_$n.log 2>&1; }
run r_j0 --jit 0
run r_j06 --jit 0.6
run r_j10 --jit 1.0
run r_j03 --jit 0.3
run r_j10_tclean --jit 1.0 --jit_tgt 0
run r_s05_j0 --jit 0 --sig_a 0.5
run r_s05_j06 --jit 0.6 --sig_a 0.5
