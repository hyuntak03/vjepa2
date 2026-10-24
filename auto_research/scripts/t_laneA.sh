#!/bin/bash
# [T] 장난감 요인 실험 대기열 (GPU 1, 한 줄씩 차례로). 앞선 run 이 끝나길 기다린 뒤 시작.
cd /workspace/world_model/vjepa2
while pgrep -f "toy_factor.py --name f_j10 " > /dev/null; do sleep 5; done
run() { n=$1; shift; CUDA_VISIBLE_DEVICES=1 /venv/vjepa/bin/python auto_research/scripts/toy_factor.py --name $n "$@" --steps 8000 > auto_research/logs/toy_factor_$n.log 2>&1; }
run f_j0 --jit 0
run f_j03 --jit 0.3
run f_s05_j0 --jit 0 --sig_a 0.5
run f_j0_p05 --jit 0 --p0 0.5
