#!/bin/bash
# [T] 장난감: 분포 예측 (잡음 입력 + 공정 CRPS) 이 불확실한 운동에서 물체를 또렷하게 옮기나 — L1 짝과 대조. 대기열 C2 뒤.
cd /workspace/world_model/vjepa2
while pgrep -f "t_laneC2.sh" > /dev/null; do sleep 10; done
run() { n=$1; shift; CUDA_VISIBLE_DEVICES=1 /venv/vjepa/bin/python auto_research/scripts/toy_factor.py --name $n --rope 1 --obj_fixed 1.0 --steps 6000 "$@" > auto_research/logs/toy_factor_$n.log 2>&1; }
run r_s05_j0_crps --jit 0 --sig_a 0.5 --loss crps
run r_s05_j06_crps --jit 0.6 --sig_a 0.5 --loss crps
