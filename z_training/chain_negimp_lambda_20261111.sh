#!/bin/bash
# 음의 손실 p 0.2 기준판의 λ 비교 (0.5 · 2.0) — 두 판을 GPU 8 장에 같이 올리고 (one-shot 학습과도 공유), 끝나는 대로 시트 네 벤치 + Surrogate → 표 (2026-11-11, 사용자 "0.5 2 이렇게 돌려봐 · 같이 8 gpu 로").
#   공식 IntPhys1 · 2 는 이 사슬에서 안 한다 (학습이 도는 동안이라). 필요하면 학습이 없을 때 따로.
#   setsid nohup bash z_training/chain_negimp_lambda_20261111.sh > z_training/logs/chain_negimp_lambda_20261111.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
ts() { TZ=Asia/Seoul date '+%F %T KST'; }
GP="0 1 2 3 4 5 6 7"
for w in w05 w2; do
  GPUS=8 bash z_training/tmux_train.sh v4bc4_ar_vw_state_negimp_vanish_app02_$w || { echo "[$(ts)] $w 올리기 실패"; exit 1; }
done
echo "[$(ts)] 두 판 올림"
for w in w05 w2; do
  R=v4bc4_ar_vw_state_negimp_vanish_app02_$w; M=vast_ar_vw_state_v4bc4_negimp_vanish_app02_$w
  RUN=$R MODEL=$M EP=20 GPU_IDS="$GP" bash z_training/chain_full_eval.sh > z_training/logs/chain_full_eval_$M.log 2>&1 || { echo "[$(ts)] ⚠️ $R 학습 실패 또는 중단"; continue; }
  GPU_IDS="$GP" GPUS=8 bash cvpr/eval/testbed/run.sh intphys1_surrogate $M > z_training/logs/surrogate_eval_$M.log 2>&1
  echo "[$(ts)] $M 시트 채점 끝"
done
echo "SHEET_EVAL_DONE $(ts)"
/venv/vjepa/bin/python z_research/scripts/analysis/sheet_table.py vast_ar_vw_state_v4bc4 vast_ar_vw_state_v4bc4_negimp_vanish_app02_w05 vast_ar_vw_state_v4bc4_negimp_vanish_app02 vast_ar_vw_state_v4bc4_negimp_vanish_app02_w2 --md
