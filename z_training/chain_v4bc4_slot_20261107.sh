#!/bin/bash
# 2026-11-07 사용자 "Slot 메모리 post-training … v4bc4 로 2000 step, 아이디어 최대한 잘 살려서 올려".
#   (11-07 04:2x 사용자 "L1 만으로, 둘 다 다시": 음의 손실을 얹었던 첫 두 판은 runs/_aborted_v4bc4_ar_vw_state_negimp_slot*_20261107)
#   두 판 (slot · slot+맞추기, 예측 L1 만) 이 GPU 0–3 / 4–7 에서 동시에 돈다 → 둘 다 e20 이 나오고 학습 프로세스가 끝나면 8 장으로 순서대로 채점:
#   시트 벤치 (chain_full_eval: VOB v3 · IP1 late · IP2 late · real) → Surrogate → 공식 IntPhys1 A.8 · IntPhys2 Main (eval_bench).
cd /workspace/world_model/vjepa2
ts() { TZ=Asia/Seoul date '+%F %T KST'; }
RUNS="v4bc4_ar_vw_state_slot v4bc4_ar_vw_state_slotmatch"
alive() { ps -eo args | grep -qE "^\S*python\S* \S*z_training/harness/launch\.py --fname \S*/runs/$1/config\.yaml"; }
echo "[chain_slot] $(ts) 두 판 학습 끝을 기다림"
for r in $RUNS; do
  until [[ -f z_training/runs/$r/e20.pt ]] && ! alive $r; do
    if ! tmux has-session -t "=train_$r" 2>/dev/null && ! alive $r && [[ ! -f z_training/runs/$r/e20.pt ]]; then echo "⚠️ $r 가 e20 없이 끝남 $(ts)"; break; fi
    sleep 120
  done
done
for pair in "v4bc4_ar_vw_state_slot:vast_ar_vw_state_v4bc4_slot" "v4bc4_ar_vw_state_slotmatch:vast_ar_vw_state_v4bc4_slotmatch"; do
  RUN=${pair%%:*}; M=${pair##*:}
  [[ -f z_training/runs/$RUN/e20.pt ]] || { echo "⚠️ $RUN e20 없음 — 건너뜀 $(ts)"; continue; }
  RUN=$RUN MODEL=$M EP=20 GPU_IDS="0 1 2 3 4 5 6 7" bash z_training/chain_full_eval.sh > z_training/logs/chain_full_eval_$M.log 2>&1
  GPU_IDS="0 1 2 3 4 5 6 7" GPUS=8 bash cvpr/eval/testbed/run.sh intphys1_surrogate $M > z_training/logs/surrogate_eval_$M.log 2>&1
  GPUS=8 bash z_training/eval_bench.sh $RUN > z_training/logs/eval_bench_$RUN.log 2>&1
  echo "$M 채점 끝 $(ts)"
done
/venv/vjepa/bin/python z_research/scripts/analysis/sheet_table.py --md vast_ar_vw_state_v4bc4 vast_ar_vw_state_v4bc4_slot vast_ar_vw_state_v4bc4_slotmatch
echo "CHAIN_V4BC4_SLOT_DONE $(ts)"
