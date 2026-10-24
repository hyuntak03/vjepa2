#!/bin/bash
# [T] 대기열 B 다음: [E] 캐시 (블록 8 개 창) 위 흐름 오라클 — SSv2 · K710 · HT100M, 문맥 5 블록
cd /workspace/world_model/vjepa2
while kill -0 204041 2>/dev/null; do sleep 10; done
O=auto_research/results/natural_transport_ctx5
for s in ssv2 k710 ht100m; do
  CUDA_VISIBLE_DEVICES=1 /venv/vjepa/bin/python -W ignore auto_research/scripts/natural_transport.py --src enc_noise --en_src $s --ctx 5 --out $O > auto_research/logs/natural_transport_ctx5_$s.log 2>&1
done
/venv/vjepa/bin/python auto_research/scripts/natural_transport.py --report $O > auto_research/logs/natural_transport_ctx5_report.log 2>&1
