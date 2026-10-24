#!/bin/bash
# [P] IntPhys 2 Main 실행이 끝나면 최종 표 → [E] 캐시가 다 차면 블록 흐름 U · V ([T] 요청). GPU 5 하나, 한 번에 한 프로세스.
cd /workspace/world_model/vjepa2
while pgrep -f "ip2_phys_surprise.py --split Main" > /dev/null; do sleep 20; done
echo "Main 끝 $(date -u)"
OMP_NUM_THREADS=8 /venv/vjepa/bin/python auto_research/scripts/ip2_phys_report.py --split Main --tag 최종 > auto_research/logs/ip2_phys_report_final.log 2>&1
echo "최종 표 끝 $(date -u)"
until /venv/vjepa/bin/python -c "import numpy as np,sys; d='/workspace/auto_research_cache/intphys2/Main/'; sys.exit(0 if (np.load(d+'done_s0.npy')|np.load(d+'done_s1.npy')).all() else 1)"; do sleep 30; done
echo "[E] 캐시 다 참 $(date -u)"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=5 /venv/vjepa/bin/python -W ignore auto_research/scripts/ip2_block_flow.py --split Main
echo "U · V 끝 $(date -u)"
