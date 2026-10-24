#!/bin/bash
# [T] 실제 latent 방법 개입: 자연 영상 학습 캐시 → (ar_crps_ft 는 따로) ; 장난감 갈래 CRPS 는 그 뒤
cd /workspace/world_model/vjepa2
while pgrep -f "name r_s05_j0_crps" > /dev/null; do sleep 10; done
for s in ssv2 k710_320 ht100m_1pct; do
  CUDA_VISIBLE_DEVICES=1 /venv/vjepa/bin/python -W ignore auto_research/scripts/cache_natural.py --src $s --n 700 --nb 8 --flow 0 --seed 7 --out /workspace/auto_research_cache/natural_train > auto_research/logs/cache_natural_train_$s.log 2>&1
done
