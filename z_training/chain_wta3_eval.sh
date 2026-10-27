#!/bin/bash
# 다중 가설 3 (승자독식 이어 학습) 의 가설별 벤치마크 채점 chain (2026-10-25, 사용자: "다중 가설로 IntPhys2 · IntPhys2 late · IntPhys1 측정, 8 gpu, InfLevel 도").
#   가설 체크포인트: z_training/tools/bake_wta_hypotheses.py 가 만든 runs/natural_ar_vw_state_wta3_h{0,1,2} (+ L1 이어 학습 대조 _ftl1)
#   집합 채점 (가설 중 최소 surprise) 은 채점 뒤 z_research/scripts/analysis/wta_set_rescore.py 가 창 단위로 합친다.
#   순서: A. 가설마다 IntPhys1 A.8 · IntPhys2 Main · IntPhys2 late occ → B. 가설마다 InfLevel 3 property → C. 대조 (ftl1) 전부
#
#   nohup bash z_training/chain_wta3_eval.sh > /dev/null 2>&1 &     로그: z_training/logs/chain_wta3_eval.log
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../z_research/scripts/harness" && pwd)/env.sh"
cd "$VJEPA2_ROOT"
LOG=z_training/logs/chain_wta3_eval.log; mkdir -p z_training/logs
say() { echo "[$(TZ=Asia/Seoul date '+%F %T KST')] $*" | tee -a "$LOG"; }
HYP="natural_ar_vw_state_wta3_h0 natural_ar_vw_state_wta3_h1 natural_ar_vw_state_wta3_h2"
CTRL="natural_ar_vw_state_ftl1"

stage_a() {   # run
  local run=$1 R=z_training/runs/$1
  say "A. $run: IntPhys1 A.8 + IntPhys2 Main"
  GPUS=8 bash z_training/eval_bench.sh "$run" latest.pt >> "$LOG" 2>&1; say "   eval_bench $?"
  local sfx=${run#natural_ar_vw_state_}
  local out=cvpr/results/eval/intphys1_late/intphys2_violation_uni_test_vast_ar_vw_state_$sfx
  if [[ -f $out/summary.json ]]; then say "   IntPhys2 late 있음"; return; fi
  say "A. $run: IntPhys2 late occ (시험 절반 252 쌍)"
  DS=intphys2_violation_uni_test SUFFIX=_$sfx SET="model.predictor_checkpoint=$PWD/$R/latest.pt" GPU_IDS="0 1 2 3 4 5 6 7" \
    bash cvpr/eval/intphys1_late/run.sh vast_ar_vw_state >> "$LOG" 2>&1; say "   intphys2 late $?"
}
stage_b() {   # runs...
  say "B. InfLevel: $*"
  NOWAIT=1 SKIP_GPU_WAIT=1 RUNS="$*" BENCHES="inflevel_continuity inflevel_solidity inflevel_gravity" bash z_training/chain_bench.sh >> "$LOG" 2>&1
  say "   chain_bench $?"
}

say "===== 시작 ====="
for r in $HYP; do stage_a "$r"; done
stage_b $HYP
stage_a "$CTRL"
stage_b "$CTRL"
say "끝"
