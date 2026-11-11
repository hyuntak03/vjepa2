#!/bin/bash
# 2026-11-07 구조 판 공식 IntPhys1 · IntPhys2 를 비교할 대조판 (v4bc4 L1 만) 의 공식 채점이 없어서 — 구조 판 채점 사슬이 끝난 뒤 돈다.
cd /workspace/world_model/vjepa2
while pgrep -f "chain_v4bc4_struct_20261107.sh" > /dev/null; do sleep 120; done
GPUS=8 bash z_training/eval_bench.sh v4bc4_ar_vw_state_postft > z_training/logs/eval_bench_v4bc4_ar_vw_state_postft.log 2>&1
echo "L1_OFFICIAL_DONE $(TZ=Asia/Seoul date '+%F %T KST')"
