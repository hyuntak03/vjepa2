#!/bin/bash
# 단일 GPU 스크립트를 GPU 8 장에 rank 8 개로:  ranks8.sh <script.py> <logtag> [추가 인자...]
S=$1; T=$2; shift 2
G=(${CUDA_VISIBLE_DEVICES//,/ })
for r in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=${G[$r]} python "$S" --rank $r --world 8 "$@" > auto_research/logs/${T}_r${r}.log 2>&1 &
done
wait; tail -n 2 auto_research/logs/${T}_r*.log
