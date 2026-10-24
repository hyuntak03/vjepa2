#!/bin/bash
# [T] 2 순위 (GPU 7 · 하나씩 · 스레드 8): 합성 캐시 끝 → 라벨 확인 (CPU) → 이어 학습 3 판 → 판정
cd /workspace/world_model/vjepa2
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
while pgrep -f "synth_cache.py --n 800" > /dev/null; do sleep 15; done
/venv/vjepa/bin/python auto_research/scripts/synth_label_check.py > auto_research/logs/synth_label_check.log 2>&1
run() { env CUDA_VISIBLE_DEVICES=7 /venv/vjepa/bin/python -W ignore "$@"; }
A=auto_research/scripts/aux_ft.py
run $A --name aux_w0 --w 0 > auto_research/logs/aux_w0.log 2>&1
run $A --name aux_w005 --w 0.05 > auto_research/logs/aux_w005.log 2>&1
run $A --name aux_w02 --w 0.2 > auto_research/logs/aux_w02.log 2>&1
run auto_research/scripts/ft_eval_reliability.py --ft aux_w0 aux_w005 aux_w02 --out auto_research/results/aux_reliability.json > auto_research/logs/aux_reliability.log 2>&1
run auto_research/scripts/ar_crps_eval.py --ft aux_w0 aux_w005 aux_w02 --K 1 > auto_research/logs/ar_crps_eval_aux.log 2>&1
cp auto_research/results/ar_crps_eval.json auto_research/results/ar_crps_eval_aux.json
