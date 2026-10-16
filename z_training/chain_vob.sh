#!/bin/bash
# VOBench held-out 학습 chain (2026-10-15):
#   1. EK100 평가 (tmux ek100_ar) 가 끝나기를 기다린다
#   2. natural_ar_vw_state_vob 학습 (scratch)
#   3. 채점: IntPhys1 A.8 · IntPhys 2 (eval_bench.sh, IntPhys 2 는 video_batch 8 · 창 batch ×6)
#      + VOBench held-out (본판 · gravity) — 새 run 과 기준 (vast_ar_vw_state) 둘 다
# 로그: z_training/logs/chain_vob.log
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../z_research/scripts/harness" && pwd)/env.sh"
cd "$VJEPA2_ROOT"
RUN=natural_ar_vw_state_vob
LOG=z_training/logs/chain_vob.log; mkdir -p z_training/logs
say() { echo "[$(TZ=Asia/Seoul date '+%F %T KST')] $*" | tee -a "$LOG"; }
say "1/4 대기: EK100 (ek100_ar)"
while tmux has-session -t '=ek100_ar' 2>/dev/null; do sleep 60; done
until ! nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 20; done
say "2/4 학습: $RUN"
GPUS=8 bash z_training/tmux_train.sh "$RUN" 2>&1 | tee -a "$LOG"
sleep 120
until grep -qE '\] done$' "z_training/runs/$RUN/train.log" 2>/dev/null || ! tmux has-session -t "=train_$RUN" 2>/dev/null; do sleep 60; done
grep -qE '\] done$' "z_training/runs/$RUN/train.log" || { say "$RUN 가 done 없이 끝났다 — 멈춤"; exit 1; }
until ! nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 20; done
tmux kill-session -t "=train_$RUN" 2>/dev/null
say "3/4 채점: IntPhys1 · IntPhys 2"
VIDEO_BATCH=8 WIN_BATCH_SCALE=6 GPUS=8 bash z_training/eval_bench.sh "$RUN" latest.pt 2>&1 | tee -a "$LOG"
say "4/4 채점: VOBench held-out (새 run · 기준)"
for m in vast_ar_vw_state_vob vast_ar_vw_state; do
  for d in vobench_v1_heldout vobench_v1_gravity_heldout; do
    RESULTS_ROOT=legacy GPUS=8 bash cvpr/eval/testbed/run.sh "$d" "$m" > "cvpr/logs/${d}_${m}.log" 2>&1 \
      && say "  $d / $m 완료" || say "  !! $d / $m 실패"
  done
done
say "끝"
