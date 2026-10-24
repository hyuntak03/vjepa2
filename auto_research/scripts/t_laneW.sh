#!/bin/bash
# [T] 2 순위 이어서 (GPU 7 · 하나씩): 대조 aux_w0 끝나면 → 보조 (변위만, 안정도 끔) 2 판 → 판정
cd /workspace/world_model/vjepa2
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
while pgrep -f "name aux_w0 --w 0" > /dev/null; do sleep 15; done
run() { env CUDA_VISIBLE_DEVICES=7 /venv/vjepa/bin/python -W ignore "$@"; }
A=auto_research/scripts/aux_ft.py
run $A --name aux_d005 --w 0.05 --stab 0 > auto_research/logs/aux_d005.log 2>&1
run $A --name aux_d02 --w 0.2 --stab 0 > auto_research/logs/aux_d02.log 2>&1
run auto_research/scripts/ft_eval_reliability.py --ft aux_w0 aux_d005 aux_d02 --out auto_research/results/aux_reliability.json > auto_research/logs/aux_reliability.log 2>&1
run auto_research/scripts/ar_crps_eval.py --ft aux_w0 aux_d005 aux_d02 --K 1 > auto_research/logs/ar_crps_eval_aux.log 2>&1
cp auto_research/results/ar_crps_eval.json auto_research/results/ar_crps_eval_aux.json
