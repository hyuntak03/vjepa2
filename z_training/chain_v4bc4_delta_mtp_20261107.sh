#!/bin/bash
# 2026-11-07 사용자 노트 1 (Δ 예측) · 3 (다중 블록 예측) — v4bc4 L1 post-train 두 판의 채점 사슬.
#   두 판이 e20 + 학습 프로세스 종료 → slot 사슬 (chain_v4bc4_slot_20261107.sh) 이 끝날 때까지 기다림 (GPU 8 장 채점이 겹치지 않게) →
#   판마다 시트 벤치 (chain_full_eval: VOB v3 · IP1 late · IP2 late · real) → Surrogate → 공식 IntPhys1 A.8 · IntPhys2 Main (eval_bench).
cd /workspace/world_model/vjepa2
ts() { TZ=Asia/Seoul date '+%F %T KST'; }
alive() { ps -eo args | grep -qE "^\S*python\S* \S*z_training/harness/launch\.py --fname \S*/runs/$1/config\.yaml"; }
echo "[chain_delta_mtp] $(ts) 두 판 학습 끝을 기다림"
for r in v4bc4_ar_vw_state_delta v4bc4_ar_vw_state_mtp; do
  until [[ -f z_training/runs/$r/e20.pt ]] && ! alive $r; do
    if ! tmux has-session -t "=train_$r" 2>/dev/null && ! alive $r && [[ ! -f z_training/runs/$r/e20.pt ]]; then echo "⚠️ $r 가 e20 없이 끝남 $(ts)"; break; fi
    sleep 120
  done
done
echo "[chain_delta_mtp] $(ts) 학습 끝 — slot 사슬 끝을 기다림"
while pgrep -f "chain_v4bc4_slot_20261107.sh" > /dev/null; do sleep 120; done
for pair in "v4bc4_ar_vw_state_delta:vast_ar_vw_state_v4bc4_delta" "v4bc4_ar_vw_state_mtp:vast_ar_vw_state_v4bc4_mtp"; do
  RUN=${pair%%:*}; M=${pair##*:}
  [[ -f z_training/runs/$RUN/e20.pt ]] || { echo "⚠️ $RUN e20 없음 — 건너뜀 $(ts)"; continue; }
  RUN=$RUN MODEL=$M EP=20 GPU_IDS="0 1 2 3 4 5 6 7" bash z_training/chain_full_eval.sh > z_training/logs/chain_full_eval_$M.log 2>&1
  GPU_IDS="0 1 2 3 4 5 6 7" GPUS=8 bash cvpr/eval/testbed/run.sh intphys1_surrogate $M > z_training/logs/surrogate_eval_$M.log 2>&1
  GPUS=8 bash z_training/eval_bench.sh $RUN > z_training/logs/eval_bench_$RUN.log 2>&1
  echo "$M 채점 끝 $(ts)"
done
/venv/vjepa/bin/python z_research/scripts/analysis/sheet_table.py --md vast_ar_vw_state_v4bc4 vast_ar_vw_state_v4bc4_slot vast_ar_vw_state_v4bc4_slotmatch vast_ar_vw_state_v4bc4_delta vast_ar_vw_state_v4bc4_mtp
echo "CHAIN_V4BC4_DELTA_MTP_DONE $(ts)"
