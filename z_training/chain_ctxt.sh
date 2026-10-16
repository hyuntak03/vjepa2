#!/bin/bash
# =============================================================================
# ctxt_video → op → ctxt_video + op 순서 학습 chain (2026-10-16, 사용자 결정)
#   0. 지금 도는 GRASP 채점 (tmux grasp_vith32) 이 끝나고 GPU 가 비기를 기다린다
#   1. run 마다: 학습 (tmux train_<run>) → done 확인 → IntPhys1 A.8 + IntPhys 2 채점 (eval_bench.sh)
#      학습이 done 없이 끝나면 그 run 은 건너뛰고 다음 run 으로 간다 (op 는 ctx_state 와 무관)
#
#   tmux new-session -d -s chain_ctxt "bash z_training/chain_ctxt.sh"
# 로그: z_training/logs/chain_ctxt.log
# =============================================================================
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../z_research/scripts/harness" && pwd)/env.sh"
cd "$VJEPA2_ROOT"
RUNS=${RUNS:-"natural_ar_vw_state_ctxt_video natural_ar_vw_state_op natural_ar_vw_state_ctxt_video_op"}
WAIT_SESS=${WAIT_SESS:-grasp_vith32}
LOG=z_training/logs/chain_ctxt.log; mkdir -p z_training/logs
say() { echo "[$(TZ=Asia/Seoul date '+%F %T KST')] $*" | tee -a "$LOG"; }
gpu_free() { until ! nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 20; done; }

wait_done() {   # $1 = run 이름. done 이면 0
  until grep -qE '\] done$' "z_training/runs/$1/train.log" 2>/dev/null || ! tmux has-session -t "=train_$1" 2>/dev/null; do sleep 60; done
  grep -qE '\] done$' "z_training/runs/$1/train.log" 2>/dev/null || return 1
  gpu_free
  tmux kill-session -t "=train_$1" 2>/dev/null
  return 0
}

say "0 대기: $WAIT_SESS 끝 + GPU 비기"
while tmux has-session -t "=$WAIT_SESS" 2>/dev/null; do sleep 60; done
gpu_free
for r in $RUNS; do
  say "학습 올림: $r"
  GPUS=8 bash z_training/tmux_train.sh "$r" 2>&1 | tee -a "$LOG"
  sleep 120
  if ! wait_done "$r"; then say "$r 가 done 없이 끝났다 — 채점 건너뛰고 다음"; gpu_free; continue; fi
  say "채점: $r (IntPhys1 A.8 + IntPhys 2)"
  bash z_training/eval_bench.sh "$r" latest.pt 2>&1 | tee -a "$LOG"
  gpu_free
done
say "끝"
