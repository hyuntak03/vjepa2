#!/bin/bash
# [T] π̄ 용 흐름 (칸 흐름 저장 판) — [E] 캐시 SSv2 · HT100M (K710 은 앞선 실행), 문맥 5 블록
cd /workspace/world_model/vjepa2
while pgrep -f "en_src k710" > /dev/null; do sleep 10; done
O=auto_research/results/natural_transport_ctx5
for s in ssv2 ht100m; do
  CUDA_VISIBLE_DEVICES=1 /venv/vjepa/bin/python -W ignore auto_research/scripts/natural_transport.py --src enc_noise --en_src $s --ctx 5 --out $O > auto_research/logs/natural_transport_ctx5_$s.log 2>&1
done
/venv/vjepa/bin/python auto_research/scripts/persistence.py auto_research/results/natural_transport_ctx5 > auto_research/logs/persistence_ctx5.log 2>&1
