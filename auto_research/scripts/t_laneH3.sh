#!/bin/bash
# [T] 자연 영상 학습 캐시 (소스당 300 창, 블록 8 개; CPU 가 붐벼 줄임) → 이어 학습 L1 · CRPS → 판정. GPU 1.
cd /workspace/world_model/vjepa2
for s in k710_320 ht100m_1pct ssv2; do
  CUDA_VISIBLE_DEVICES=1 /venv/vjepa/bin/python -W ignore auto_research/scripts/cache_natural.py --src $s --n 300 --nb 8 --flow 0 --seed 7 --out /workspace/auto_research_cache/natural_train > auto_research/logs/cache_natural_train_$s.log 2>&1
done
P=auto_research/scripts/ar_crps_ft.py
CUDA_VISIBLE_DEVICES=1 /venv/vjepa/bin/python -W ignore $P --loss l1 --name ft_l1 --steps 2000 > auto_research/logs/ar_ft_l1.log 2>&1
CUDA_VISIBLE_DEVICES=1 /venv/vjepa/bin/python -W ignore $P --loss crps --xi laplace --name ft_crpsxi --steps 2000 > auto_research/logs/ar_ft_crpsxi.log 2>&1
CUDA_VISIBLE_DEVICES=1 /venv/vjepa/bin/python -W ignore auto_research/scripts/ar_crps_eval.py --ft ft_l1 ft_crpsxi --K 8 > auto_research/logs/ar_crps_eval.log 2>&1
