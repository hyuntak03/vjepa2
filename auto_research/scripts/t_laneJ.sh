#!/bin/bash
# [T] 실제 latent 이어 학습: 캐시 (대기열 H · H2) 가 끝나면 L1 대조 → 공정 CRPS (g + s⊙ξ) → 판정
cd /workspace/world_model/vjepa2
while pgrep -f "t_laneH.sh" > /dev/null || pgrep -f "t_laneH2.sh" > /dev/null; do sleep 15; done
P=auto_research/scripts/ar_crps_ft.py
CUDA_VISIBLE_DEVICES=1 /venv/vjepa/bin/python -W ignore $P --loss l1 --name ft_l1 --steps 3000 > auto_research/logs/ar_ft_l1.log 2>&1
CUDA_VISIBLE_DEVICES=1 /venv/vjepa/bin/python -W ignore $P --loss crps --xi laplace --name ft_crpsxi --steps 3000 > auto_research/logs/ar_ft_crpsxi.log 2>&1
CUDA_VISIBLE_DEVICES=1 /venv/vjepa/bin/python -W ignore auto_research/scripts/ar_crps_eval.py --ft ft_l1 ft_crpsxi --K 8 > auto_research/logs/ar_crps_eval.log 2>&1
