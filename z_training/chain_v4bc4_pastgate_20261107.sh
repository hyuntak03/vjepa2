#!/bin/bash
# 2026-11-07 now/past gate 판 채점: e20 + 학습 끝 → 시트 벤치 (VOB v3 · IP1 late · IP2 late · real) → Surrogate → 시트 표 (대조 v4bc4 L1).
#   GPU 8 장 채점이 다른 사슬과 겹치면 느려질 뿐 (같이 돌아도 된다). 공식 IP1 · IP2 는 구조 판 공식 채점 사슬이 끝난 뒤.
cd /workspace/world_model/vjepa2
ts() { TZ=Asia/Seoul date '+%F %T KST'; }
RUN=v4bc4_ar_vw_state_pastgate; M=vast_ar_vw_state_v4bc4_pastgate
RUN=$RUN MODEL=$M EP=20 GPU_IDS="0 1 2 3 4 5 6 7" bash z_training/chain_full_eval.sh > z_training/logs/chain_full_eval_$M.log 2>&1 || { echo "⚠️ $RUN e20 없이 끝남 $(ts)"; exit 1; }
GPU_IDS="0 1 2 3 4 5 6 7" GPUS=8 bash cvpr/eval/testbed/run.sh intphys1_surrogate $M > z_training/logs/surrogate_eval_$M.log 2>&1
echo "SHEET_TABLE $(ts)"
/venv/vjepa/bin/python z_research/scripts/analysis/sheet_table.py --md vast_ar_vw_state_v4bc4 $M
while pgrep -f "chain_v4bc4_struct_20261107.sh|chain_v4bc4_l1_official_20261107.sh" > /dev/null; do sleep 120; done
GPUS=8 bash z_training/eval_bench.sh $RUN > z_training/logs/eval_bench_$RUN.log 2>&1
echo "CHAIN_PASTGATE_DONE $(ts)"
