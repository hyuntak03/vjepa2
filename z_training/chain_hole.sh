#!/bin/bash
# =============================================================================
# op → 문맥 구멍 chain (2026-10-16, 사용자 결정 — chain_ctxt 의 ctxt_video_op 를 취소하고 교체)
#   1. 지금 도는 natural_ar_vw_state_op 학습이 done 으로 끝나기를 기다린다 → IntPhys1 A.8 + IntPhys 2 채점
#   2. natural_ar_vw_state_ctxt_video_tgtvideo_hole 학습 (문맥 · 타깃 영상 + encoding 전 문맥 구멍) → done → 같은 채점
#
#   tmux new-session -d -s chain_hole "bash z_training/chain_hole.sh"
# 로그: z_training/logs/chain_hole.log
# =============================================================================
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../z_research/scripts/harness" && pwd)/env.sh"
cd "$VJEPA2_ROOT"
A=${A:-natural_ar_vw_state_op}; B=${B:-natural_ar_vw_state_ctxt_video_tgtvideo_hole}   # 2026-10-16: 구멍은 encoding 전, 문맥 · 타깃 영상
LOG=z_training/logs/chain_hole.log; mkdir -p z_training/logs
say() { echo "[$(TZ=Asia/Seoul date '+%F %T KST')] $*" | tee -a "$LOG"; }
gpu_free() { until ! nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 20; done; }
wait_done() {   # $1 = run 이름. done 이면 0
  until grep -qE '\] done$' "z_training/runs/$1/train.log" 2>/dev/null || ! tmux has-session -t "=train_$1" 2>/dev/null; do sleep 60; done
  grep -qE '\] done$' "z_training/runs/$1/train.log" 2>/dev/null || return 1
  gpu_free
  tmux kill-session -t "=train_$1" 2>/dev/null
  return 0
}

say "1 대기: $A 학습"
if wait_done "$A"; then
  say "채점: $A"; bash z_training/eval_bench.sh "$A" latest.pt 2>&1 | tee -a "$LOG"; gpu_free
else
  say "$A 가 done 없이 끝났다 — 채점 건너뜀"; gpu_free
fi
say "2 학습 올림: $B"
GPUS=8 bash z_training/tmux_train.sh "$B" 2>&1 | tee -a "$LOG"
sleep 120
wait_done "$B" || { say "$B 가 done 없이 끝났다 — 멈춤"; exit 1; }
say "채점: $B"; bash z_training/eval_bench.sh "$B" latest.pt 2>&1 | tee -a "$LOG"
say "끝"
