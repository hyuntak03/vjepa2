#!/bin/bash
# 2026-11-07 AR 처음부터 (자연 + 매 step v4bc4 음의 손실) 채점: e70 + 학습 끝 → 시트 벤치 → Surrogate → 시트 표 (AR 기본판 · v4bc4 음의 손실 이어 학습과 비교) → 공식 IP1 · IP2.
cd /workspace/world_model/vjepa2
ts() { TZ=Asia/Seoul date '+%F %T KST'; }
RUN=natural_ar_vw_state_v4bc4_negimp_mix; M=vast_ar_vw_state_natmix_v4bc4_negimp
RUN=$RUN MODEL=$M EP=70 GPU_IDS="0 1 2 3 4 5 6 7" bash z_training/chain_full_eval.sh > z_training/logs/chain_full_eval_$M.log 2>&1 || { echo "⚠️ $RUN e70 없이 끝남 $(ts)"; exit 1; }
GPU_IDS="0 1 2 3 4 5 6 7" GPUS=8 bash cvpr/eval/testbed/run.sh intphys1_surrogate $M > z_training/logs/surrogate_eval_$M.log 2>&1
echo "SHEET_TABLE $(ts)"
/venv/vjepa/bin/python z_research/scripts/analysis/sheet_table.py --md vast_ar_vw_state vast_ar_vw_state_v4bc4_negimp $M
GPUS=8 bash z_training/eval_bench.sh $RUN > z_training/logs/eval_bench_$RUN.log 2>&1
grep -E "격자 최고|열별 최고" z_training/logs/eval_bench_$RUN.log
echo "CHAIN_MIXNEG_DONE $(ts)"
