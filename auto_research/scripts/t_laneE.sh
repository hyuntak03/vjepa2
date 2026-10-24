#!/bin/bash
# [T] 렌더 끝 → PNG 인코딩 (train · test, GPU 7) → 렌더 요약 (CPU)
cd /workspace/world_model/vjepa2
while pgrep -f "cf_render_main.sh" > /dev/null; do sleep 30; done
for sp in train test; do env OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=7 /venv/vjepa/bin/python -W ignore auto_research/scripts/cf_encode.py --split $sp > auto_research/logs/cf_encode_$sp.log 2>&1; done
/venv/vjepa/bin/python auto_research/scripts/cf_summary.py > auto_research/logs/cf_summary.log 2>&1
echo "encode done $(date -u +%H:%M)"
