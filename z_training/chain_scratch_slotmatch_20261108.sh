#!/bin/bash
# 2026-11-08 사용자 "채점까지 해야지": AR scratch + slot + 맞추기 (v4bc4 10 %) — e70 + 학습 종료 → 시트 벤치 (chain_full_eval) + Surrogate → 표.
cd /workspace/world_model/vjepa2
ts() { TZ=Asia/Seoul date '+%F %T KST'; }
RUN=natural_ar_vw_state_v4bc4_slotmatch; M=vast_ar_vw_state_v4bc4_slotmatch_scratch; G="0 1 2 3 4 5 6 7"
echo "[chain_scratch_slot] $(ts) 기다림 ($RUN e70)"
RUN=$RUN MODEL=$M EP=70 GPU_IDS="$G" bash z_training/chain_full_eval.sh > z_training/logs/chain_full_eval_$M.log 2>&1 || { echo "[chain_scratch_slot] $(ts) ⚠️ e70 없이 끝났거나 채점 실패 — z_training/logs/chain_full_eval_$M.log"; exit 1; }
GPU_IDS="$G" GPUS=8 bash cvpr/eval/testbed/run.sh intphys1_surrogate $M > z_training/logs/surrogate_eval_$M.log 2>&1
/venv/vjepa/bin/python z_research/scripts/analysis/sheet_table.py vast_ar_vw_state vast_ar_vw_state_v4vc_scratch vast_ar_vw_state_v4bc4_slotmatch $M --md
echo "CHAIN_SCRATCH_SLOT_DONE $(ts)"
