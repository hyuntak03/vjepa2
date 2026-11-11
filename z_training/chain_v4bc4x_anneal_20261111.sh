#!/bin/bash
# 감쇠 구간에만 음의 손실 판 (natural_ar_vw_state_v4bc4x_negimp_anneal) 학습 → e70 뒤 시트 네 벤치 + Surrogate → 표 (2026-11-11).
cd "$(dirname "$0")/.." || exit 1
RUN=natural_ar_vw_state_v4bc4x_negimp_anneal; M=vast_ar_vw_state_v4bc4x_negimp_anneal; GP="0 1 2 3 4 5 6 7"
GPUS=8 bash z_training/tmux_train.sh $RUN || exit 1
RUN=$RUN MODEL=$M EP=70 GPU_IDS="$GP" bash z_training/chain_full_eval.sh > z_training/logs/chain_full_eval_$M.log 2>&1 || { echo "학습 실패 또는 중단"; exit 1; }
GPU_IDS="$GP" GPUS=8 bash cvpr/eval/testbed/run.sh intphys1_surrogate $M > z_training/logs/surrogate_eval_$M.log 2>&1
echo "SHEET_EVAL_DONE $(TZ=Asia/Seoul date '+%F %T KST')"
/venv/vjepa/bin/python z_research/scripts/analysis/sheet_table.py vast_ar_vw_state vast_ar_vw_state_v4bc4_negimp_vanish_app02 vast_ar_vw_state_v4bc4x_negimp_vanish_app02_mix $M --md
