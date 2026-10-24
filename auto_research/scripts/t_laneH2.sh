#!/bin/bash
# [T] SSv2 학습 캐시 (디코더 스레드 1 로 다시) — 대기열 H 가 끝난 뒤
cd /workspace/world_model/vjepa2
while pgrep -f "t_laneH.sh" > /dev/null; do sleep 15; done
CUDA_VISIBLE_DEVICES=1 /venv/vjepa/bin/python -W ignore auto_research/scripts/cache_natural.py --src ssv2 --n 700 --nb 8 --flow 0 --seed 7 --out /workspace/auto_research_cache/natural_train > auto_research/logs/cache_natural_train_ssv2.log 2>&1
