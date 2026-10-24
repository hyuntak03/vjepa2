#!/bin/bash
# [T] 반사실 이어 학습 3 팔 + 판정 (GPU 7 하나, 하나씩). ⚠️ 사용자 결정 뒤에만 올린다 (리더 KST 06:0x).
cd /workspace/world_model/vjepa2
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
run() { env CUDA_VISIBLE_DEVICES=7 /venv/vjepa/bin/python -W ignore "$@"; }
for arm in a b c; do run auto_research/scripts/cf_ft.py --name cf_$arm --arm $arm --steps 3000 > auto_research/logs/cf_ft_$arm.log 2>&1; echo "train $arm $(date -u +%H:%M)"; done
run auto_research/scripts/cf_style.py > auto_research/logs/cf_style.log 2>&1
run auto_research/scripts/cf_eval.py --models base cf_a cf_b cf_c > auto_research/logs/cf_eval.log 2>&1
/venv/vjepa/bin/python auto_research/scripts/cf_compare.py --base cf_a --arms cf_b cf_c > auto_research/logs/cf_compare.log 2>&1
echo "all done $(date -u +%H:%M)"
