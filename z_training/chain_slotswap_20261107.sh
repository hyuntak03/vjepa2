#!/bin/bash
# 2026-11-07 형제 slot 맞바꾸기 짧은 판 채점: 학습이 끝나면 시트 벤치 + Surrogate → slot 진단 (판정 1 = 맞바꾸기 효과) → 시트 표. 공식 IP1 · IP2 는 판정 1 통과 뒤 따로.
cd /workspace/world_model/vjepa2
ts() { TZ=Asia/Seoul date '+%F %T KST'; }
RUN=v4bc4_ar_vw_state_slotswap; M=vast_ar_vw_state_v4bc4_slotswap
echo "[chain_slotswap] $(ts) 시작"
RUN=$RUN MODEL=$M EP=5 GPU_IDS="0 1 2 3 4 5 6 7" bash z_training/chain_full_eval.sh > z_training/logs/chain_full_eval_$M.log 2>&1
GPU_IDS="0 1 2 3 4 5 6 7" GPUS=8 bash cvpr/eval/testbed/run.sh intphys1_surrogate $M > z_training/logs/surrogate_eval_$M.log 2>&1
echo "[chain_slotswap] $(ts) 시트 벤치 끝 → slot 진단"
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 /venv/vjepa/bin/python z_research/scripts/analysis/slot_diag.py --model $M --clips 512 > z_analysis/results/slot_diag_20261107/$M.log 2>&1
/venv/vjepa/bin/python z_research/scripts/analysis/sheet_table.py --md vast_ar_vw_state_v4bc4 vast_ar_vw_state_v4bc4_slotmatch vast_ar_vw_state_v4bc4_negimp $M
echo "CHAIN_SLOTSWAP_DONE $(ts)"
