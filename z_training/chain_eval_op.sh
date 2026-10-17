#!/bin/bash
# =============================================================================
# natural_ar_vw_state_op 학습이 끝나면 평가 묶음 (2026-10-16, 사용자 지시)
#   1. IntPhys1 A.8 + IntPhys 2 (eval_bench.sh)
#   2. VOBench 전체 · VOBench gravity (cvpr testbed, 레지스트리 vast_ar_vw_state_op, RESULTS_ROOT=legacy)
#   3. InfLevel 3 속성 · GRASP × 창 16 · 32 (chain_bench.sh, Garrido A.8) — GRASP 는 2026-10-16 사용자 추가
#   tmux new-session -d -s chain_eval_op "bash z_training/chain_eval_op.sh"     로그: z_training/logs/chain_eval_op.log
# =============================================================================
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../z_research/scripts/harness" && pwd)/env.sh"
cd "$VJEPA2_ROOT"
RUN=natural_ar_vw_state_op; M=vast_ar_vw_state_op
LOG=z_training/logs/chain_eval_op.log; mkdir -p z_training/logs
say() { echo "[$(TZ=Asia/Seoul date '+%F %T KST')] $*" | tee -a "$LOG"; }
gpu_free() { until ! nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 20; done; }

say "대기: $RUN 학습 done"
until grep -qE '\] done$' "z_training/runs/$RUN/train.log" 2>/dev/null || ! tmux has-session -t "=train_$RUN" 2>/dev/null; do sleep 60; done
grep -qE '\] done$' "z_training/runs/$RUN/train.log" || { say "$RUN 가 done 없이 끝났다 — 멈춤"; exit 1; }
gpu_free; tmux kill-session -t "=train_$RUN" 2>/dev/null
say "1/3 VOBench · VOBench gravity (2026-10-16 사용자: IntPhys 2 보다 먼저)"
for ds in vobench_v1 vobench_v1_gravity; do
  RESULTS_ROOT=legacy GPUS=8 bash cvpr/eval/testbed/run.sh $ds $M > z_training/logs/eval_op_$ds.log 2>&1 \
    && say "  $ds 완료" || say "  !! $ds 실패 — z_training/logs/eval_op_$ds.log"
  gpu_free
done
say "2/3 IntPhys1 A.8 + IntPhys 2 (이미 있는 칸은 건너뜀)"
bash z_training/eval_bench.sh "$RUN" latest.pt 2>&1 | tee -a "$LOG"; gpu_free
say "3/3 InfLevel · GRASP (창 16 · 32)"
NOWAIT=1 RUNS=$RUN BENCHES="inflevel_continuity inflevel_solidity inflevel_gravity grasp_level2" bash z_training/chain_bench.sh 2>&1 | tail -20 | tee -a "$LOG"
say "끝"
