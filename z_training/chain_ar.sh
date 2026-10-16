#!/bin/bash
# =============================================================================
# AR 순서 실행 (2026-10-11, 사용자 지시 "순서대로 바로 올려"):
#   1. natural_ar_vw_hint 학습이 done 으로 끝나기를 기다린다 (done 없이 끝나면 멈춤)
#   2. 그 run 을 IntPhys1 A.8 + IntPhys 2 로 채점 (z_training/eval_bench.sh). 채점이 실패해도 3 은 올린다 (채점은 다시 돌릴 수 있다)
#   3. natural_ar_vw (힌트 없는 기준선) 학습을 올린다
#   4. 그 학습이 끝나면 같은 채점
#
#   tmux new-session -d -s chain_ar "bash z_training/chain_ar.sh"
# 로그: z_training/logs/chain_ar.log
# =============================================================================
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../z_research/scripts/harness" && pwd)/env.sh"
cd "$VJEPA2_ROOT"
A=${A:-natural_ar_vw_hint}; B=${B-natural_ar_vw}
LOG=z_training/logs/chain_ar.log; mkdir -p z_training/logs
say() { echo "[$(TZ=Asia/Seoul date '+%F %T KST')] $*" | tee -a "$LOG"; }

wait_done() {   # $1 = run 이름. done 이면 0
  until grep -qE '\] done$' "z_training/runs/$1/train.log" 2>/dev/null || ! tmux has-session -t "=train_$1" 2>/dev/null; do sleep 60; done
  grep -qE '\] done$' "z_training/runs/$1/train.log" 2>/dev/null || return 1
  until ! nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 20; done
  tmux kill-session -t "=train_$1" 2>/dev/null
  return 0
}

say "1/4 대기: $A 학습"
wait_done "$A" || { say "$A 가 done 없이 끝났다 — 멈춤"; exit 1; }
say "2/4 채점: $A (IntPhys1 A.8 + IntPhys 2)"
bash z_training/eval_bench.sh "$A" latest.pt 2>&1 | tee -a "$LOG"
[[ -z "$B" ]] && { say "B 없음 — 끝"; exit 0; }       # 2026-10-13: A 만 기다려 채점하는 용도 (B= 로 비움)
say "3/4 올림: $B"
GPUS=8 bash z_training/tmux_train.sh "$B" 2>&1 | tee -a "$LOG"
sleep 120
wait_done "$B" || { say "$B 가 done 없이 끝났다 — 멈춤"; exit 1; }
say "4/4 채점: $B"
bash z_training/eval_bench.sh "$B" latest.pt 2>&1 | tee -a "$LOG"
say "끝"
