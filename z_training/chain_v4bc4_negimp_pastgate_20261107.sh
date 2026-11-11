#!/bin/bash
# 2026-11-07 now/past gate + 음의 손실 판 채점: e20 + 학습 끝 → 시트 벤치 → Surrogate → 시트 표 (대조 = 음의 손실 기준판) → 공식 IP1 · IP2 (다른 공식 채점 뒤).
cd /workspace/world_model/vjepa2
ts() { TZ=Asia/Seoul date '+%F %T KST'; }
RUN=v4bc4_ar_vw_state_negimp_pastgate; M=vast_ar_vw_state_v4bc4_negimp_pastgate
RUN=$RUN MODEL=$M EP=20 GPU_IDS="0 1 2 3 4 5 6 7" bash z_training/chain_full_eval.sh > z_training/logs/chain_full_eval_$M.log 2>&1 || { echo "⚠️ $RUN e20 없이 끝남 $(ts)"; exit 1; }
GPU_IDS="0 1 2 3 4 5 6 7" GPUS=8 bash cvpr/eval/testbed/run.sh intphys1_surrogate $M > z_training/logs/surrogate_eval_$M.log 2>&1
echo "SHEET_TABLE $(ts)"
/venv/vjepa/bin/python z_research/scripts/analysis/sheet_table.py --md vast_ar_vw_state_v4bc4_negimp $M
while pgrep -f "chain_v4bc4_struct_20261107.sh|chain_v4bc4_l1_official_20261107.sh|chain_v4bc4_pastgate_20261107.sh" > /dev/null; do sleep 120; done
GPUS=8 bash z_training/eval_bench.sh $RUN > z_training/logs/eval_bench_$RUN.log 2>&1
echo "CHAIN_NEGIMP_PASTGATE_DONE $(ts)"
