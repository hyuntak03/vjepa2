#!/bin/bash
# [T] π̄ 용 흐름 K710 (칸 흐름 저장 판) — 대기열 D 가 끝난 뒤
cd /workspace/world_model/vjepa2
while pgrep -f "t_laneD.sh" > /dev/null; do sleep 10; done
O=auto_research/results/natural_transport_ctx5
CUDA_VISIBLE_DEVICES=1 /venv/vjepa/bin/python -W ignore auto_research/scripts/natural_transport.py --src enc_noise --en_src k710 --ctx 5 --out $O > auto_research/logs/natural_transport_ctx5_k710.log 2>&1
/venv/vjepa/bin/python auto_research/scripts/persistence.py auto_research/results/natural_transport_ctx5 > auto_research/logs/persistence_ctx5.log 2>&1
