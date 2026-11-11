#!/bin/bash
# 2026-11-07 곡률 손실 판 채점: e20 + 학습 종료 → 시트 벤치 (chain_full_eval) + Surrogate → 대조 v4bc4 L1 과 표.
cd /workspace/world_model/vjepa2
ts() { TZ=Asia/Seoul date '+%F %T KST'; }
RUN=v4bc4_ar_vw_state_curv; M=vast_ar_vw_state_v4bc4_curv; G="0 1 2 3 4 5 6 7"
echo "[chain_curv] $(ts) 기다림"
until [[ -f z_training/runs/$RUN/e20.pt ]] && ! pgrep -f "launch_curv.py --fname .*/runs/$RUN/config.yaml" >/dev/null; do
  pgrep -f "launch_curv.py --fname .*/runs/$RUN/config.yaml" >/dev/null || [[ -f z_training/runs/$RUN/e20.pt ]] || { echo "[chain_curv] $(ts) ⚠️ 학습이 e20 없이 끝남 — 채점 안 함"; exit 1; }
  sleep 120
done
sleep 30
RUN=$RUN MODEL=$M EP=20 GPU_IDS="$G" bash z_training/chain_full_eval.sh > z_training/logs/chain_full_eval_$M.log 2>&1 || { echo "⚠️ 채점 실패"; exit 1; }
GPU_IDS="$G" GPUS=8 bash cvpr/eval/testbed/run.sh intphys1_surrogate $M > z_training/logs/surrogate_eval_$M.log 2>&1
/venv/vjepa/bin/python z_research/scripts/analysis/sheet_table.py vast_ar_vw_state vast_ar_vw_state_v4bc4 $M --md
echo "CHAIN_V4BC4_CURV_DONE $(ts)"
