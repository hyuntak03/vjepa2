#!/bin/bash
# natural_ar_vw_state_v5id_negimp_mix 학습이 끝나면 (e70.pt + 프로세스 종료) 시트 칸 채점 → 공식 IntPhys1 · 2 (2026-11-10).
#   setsid nohup bash z_training/chain_v5id_mix_eval_20261110.sh > z_training/logs/chain_v5id_mix_eval_20261110.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
ts() { TZ=Asia/Seoul date '+%F %T KST'; }
RUN=natural_ar_vw_state_v5id_negimp_mix; M=vast_ar_vw_state_v5id_negimp_mix
RUN=$RUN MODEL=$M EP=70 GPU_IDS="0 1 2 3 4 5 6 7" bash z_training/chain_full_eval.sh > z_training/logs/chain_full_eval_$M.log 2>&1 || { echo "학습 실패 또는 중단 $(ts)"; exit 1; }
GPU_IDS="0 1 2 3 4 5 6 7" GPUS=8 bash cvpr/eval/testbed/run.sh intphys1_surrogate $M > z_training/logs/surrogate_eval_$M.log 2>&1
echo "SHEET_EVAL_DONE $(ts)"
/venv/vjepa/bin/python z_research/scripts/analysis/sheet_table.py vast_ar_vw_state vast_ar_vw_state_v4bc4_negimp_vanish_app02 $M --md
GPUS=8 bash z_training/eval_bench.sh $RUN > z_training/logs/eval_bench_$RUN.log 2>&1
echo "OFFICIAL_DONE $(ts)"; tail -12 z_training/runs/$RUN/eval/BENCH_latest.txt 2>/dev/null
