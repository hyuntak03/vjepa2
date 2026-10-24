#!/bin/bash
# [P] pred_mean_solution.py 를 모델 목록에 차례로 (GPU 하나). 사용: bash pm_queue.sh <gpu> <run:ckpt[:attn]> ...
G=$1; shift
cd /workspace/world_model/vjepa2
for spec in "$@"; do
  IFS=: read -r run ck at <<< "$spec"
  tag="${run}__${ck%.pt}"
  if [ -f auto_research/results/pred_mean/${tag}.json ]; then echo "skip $tag"; continue; fi
  echo "== $(date +%H:%M:%S) $tag" 
  CUDA_VISIBLE_DEVICES=$G /venv/vjepa/bin/python -W ignore auto_research/scripts/pred_mean_solution.py --run $run $ck --attn ${at:-0} \
     > auto_research/logs/pm_${tag}.log 2>&1
  echo "   done $(date +%H:%M:%S) rc $?"
done
