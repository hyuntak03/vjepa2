#!/bin/bash
# [T] 대기열 B (GPU 1): 흐름 오라클 (3D 합성 검증 → 자연 영상) → 정답 잡음 줄인 인코딩 점검
cd /workspace/world_model/vjepa2
O=auto_research/results/natural_transport
CUDA_VISIBLE_DEVICES=1 /venv/vjepa/bin/python -W ignore auto_research/scripts/natural_transport.py --src natural --n_per 150 --out $O > auto_research/logs/natural_transport_natural.log 2>&1
CUDA_VISIBLE_DEVICES=1 /venv/vjepa/bin/python -W ignore auto_research/scripts/natural_transport.py --src phys3d --out $O > auto_research/logs/natural_transport_phys3d.log 2>&1
/venv/vjepa/bin/python auto_research/scripts/natural_transport.py --report $O > auto_research/logs/natural_transport_report.log 2>&1
CUDA_VISIBLE_DEVICES=1 /venv/vjepa/bin/python -W ignore auto_research/scripts/smooth_target_check.py --n 24 > auto_research/logs/smooth_target_check.log 2>&1
