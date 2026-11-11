#!/bin/bash
# 2026-11-07 v4bc4f 음의 손실 판 (v4f 덮어 미는 상자 추가) 채점: 학습이 끝나면 시트 벤치 + Surrogate → 시트 표 → realbench 사건별 표.
cd /workspace/world_model/vjepa2
ts() { TZ=Asia/Seoul date '+%F %T KST'; }
RUN=v4bc4f_ar_vw_state_negimp; M=vast_ar_vw_state_v4bc4f_negimp
echo "[chain_v4bc4f] $(ts) 시작"
RUN=$RUN MODEL=$M EP=20 GPU_IDS="0 1 2 3 4 5 6 7" bash z_training/chain_full_eval.sh > z_training/logs/chain_full_eval_$M.log 2>&1
GPU_IDS="0 1 2 3 4 5 6 7" GPUS=8 bash cvpr/eval/testbed/run.sh intphys1_surrogate $M > z_training/logs/surrogate_eval_$M.log 2>&1
/venv/vjepa/bin/python z_research/scripts/analysis/sheet_table.py --md vast_ar_vw_state_v4bc4 vast_ar_vw_state_v4bc4_negimp vast_ar_vw_state_v4bc4de_negimp $M
/venv/vjepa/bin/python z_research/scripts/analysis/realbench_by_case.py vast_ar_vw_state_v4bc4_negimp vast_ar_vw_state_v4bc4de_negimp $M
echo "CHAIN_V4BC4F_DONE $(ts)"
