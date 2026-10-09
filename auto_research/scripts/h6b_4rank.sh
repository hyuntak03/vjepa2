#!/bin/bash
# H6b 를 GPU 4 장에 rank 4 개로
G=(${CUDA_VISIBLE_DEVICES//,/ })
for r in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=${G[$r]} python auto_research/scripts/h6b_intphys1_baselines.py --rank $r --world 4 > auto_research/logs/h6b_r${r}.log 2>&1 &
done
wait; tail -n 2 auto_research/logs/h6b_r*.log
