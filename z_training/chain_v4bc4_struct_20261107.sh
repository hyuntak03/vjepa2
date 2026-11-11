#!/bin/bash
# 2026-11-07 v4bc4 L1 구조 판 4 개 (slot · slot+맞추기 · Δ 예측 · 다중 블록 예측) 채점 사슬 — 앞의 두 사슬 (chain_v4bc4_slot / chain_v4bc4_delta_mtp) 을 대체.
#   공식 IntPhys1 · IntPhys2 (eval_bench) 가 판당 ~1.5 시간이라 맨 뒤로: ① 판이 끝나는 대로 시트 벤치 + Surrogate (판당 ~10 분) → ② 시트 표 → ③ 공식 채점.
cd /workspace/world_model/vjepa2
ts() { TZ=Asia/Seoul date '+%F %T KST'; }
alive() { ps -eo args | grep -qE "^\S*python\S* \S*z_training/harness/launch\.py --fname \S*/runs/$1/config\.yaml"; }
PAIRS="v4bc4_ar_vw_state_slot:vast_ar_vw_state_v4bc4_slot v4bc4_ar_vw_state_slotmatch:vast_ar_vw_state_v4bc4_slotmatch v4bc4_ar_vw_state_delta:vast_ar_vw_state_v4bc4_delta v4bc4_ar_vw_state_mtp:vast_ar_vw_state_v4bc4_mtp"
declare -A DONE
echo "[chain_struct] $(ts) 시작 — 판이 끝나는 대로 시트 벤치 + Surrogate"
while [[ ${#DONE[@]} -lt 4 ]]; do
  for pair in $PAIRS; do
    RUN=${pair%%:*}; M=${pair##*:}
    [[ -n ${DONE[$RUN]:-} ]] && continue
    if [[ -f z_training/runs/$RUN/e20.pt ]] && ! alive $RUN; then
      sleep 30
      echo "[chain_struct] $(ts) $RUN 학습 끝 → 시트 벤치"
      RUN=$RUN MODEL=$M EP=20 GPU_IDS="0 1 2 3 4 5 6 7" bash z_training/chain_full_eval.sh > z_training/logs/chain_full_eval_$M.log 2>&1
      GPU_IDS="0 1 2 3 4 5 6 7" GPUS=8 bash cvpr/eval/testbed/run.sh intphys1_surrogate $M > z_training/logs/surrogate_eval_$M.log 2>&1
      echo "[chain_struct] $(ts) $M 시트 벤치 · Surrogate 끝"
      DONE[$RUN]=1
    elif ! tmux has-session -t "=train_$RUN" 2>/dev/null && ! alive $RUN; then
      echo "[chain_struct] ⚠️ $(ts) $RUN 가 e20 없이 끝남 — 건너뜀"; DONE[$RUN]=skip
    fi
  done
  [[ ${#DONE[@]} -lt 4 ]] && sleep 60
done
echo "SHEET_TABLE $(ts)"
/venv/vjepa/bin/python z_research/scripts/analysis/sheet_table.py --md vast_ar_vw_state_v4bc4 vast_ar_vw_state_v4bc4_slot vast_ar_vw_state_v4bc4_slotmatch vast_ar_vw_state_v4bc4_delta vast_ar_vw_state_v4bc4_mtp
for pair in $PAIRS; do
  RUN=${pair%%:*}
  [[ ${DONE[$RUN]} == 1 ]] || continue
  echo "[chain_struct] $(ts) $RUN 공식 IntPhys1 · IntPhys2 시작"
  GPUS=8 bash z_training/eval_bench.sh $RUN > z_training/logs/eval_bench_$RUN.log 2>&1
  echo "[chain_struct] $(ts) $RUN 공식 채점 끝"
done
echo "CHAIN_V4BC4_STRUCT_DONE $(ts)"
