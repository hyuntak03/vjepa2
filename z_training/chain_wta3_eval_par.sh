#!/bin/bash
# chain_wta3_eval.sh 의 남은 작업을 병렬로 (2026-10-25 KST 18 시, 사용자: "vram 남는데 그냥 한번에 돌리지").
#   그때 GPU 당 17 GB / 98 GB, 사용률 낮음 (디코딩이 병목). 순차 chain 은 바깥 스크립트만 멈췄고 돌던 가설 1 IntPhys2 w48 은 그대로 둔다.
#   작업 넷을 45 초 간격으로 띄운다 (DDP 포트 경합 방지). 각 채점기는 결과가 있으면 건너뛴다. InfLevel 은 batch 절반 (MB 72 · VB 8), OOM 이면 chain_bench 가 다시 반으로.
#   J1: 가설 1 IntPhys2 late → 가설 2 IntPhys1 · IntPhys2 · late
#   J2: InfLevel 가설 0 → InfLevel 대조 (ftl1)
#   J3: InfLevel 가설 1 → 대조 IntPhys1 · IntPhys2 · late
#   J4: InfLevel 가설 2
#   nohup setsid bash z_training/chain_wta3_eval_par.sh > /dev/null 2>&1 &     로그: z_training/logs/chain_wta3_eval.log (같은 파일에 이어 쓴다)
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../z_research/scripts/harness" && pwd)/env.sh"
cd "$VJEPA2_ROOT"
LOG=z_training/logs/chain_wta3_eval.log
say() { echo "[$(TZ=Asia/Seoul date '+%F %T KST')] $*" | tee -a "$LOG"; }
P=natural_ar_vw_state

late() {   # run
  local sfx=${1#${P}_} out=cvpr/results/eval/intphys1_late/intphys2_violation_uni_test_vast_ar_vw_state_${1#${P}_}
  [[ -f $out/summary.json ]] && { say "   [$1] IntPhys2 late 있음"; return; }
  say "   [$1] IntPhys2 late 시작"
  DS=intphys2_violation_uni_test SUFFIX=_$sfx SET="model.predictor_checkpoint=$PWD/z_training/runs/$1/latest.pt" GPU_IDS="0 1 2 3 4 5 6 7" \
    bash cvpr/eval/intphys1_late/run.sh vast_ar_vw_state >> "z_training/runs/$1/eval/logs/late.log" 2>&1; say "   [$1] IntPhys2 late 끝 $?"
}
bench() {  # run
  say "   [$1] IntPhys1 · IntPhys2 시작"
  GPUS=8 bash z_training/eval_bench.sh "$1" latest.pt >> "z_training/runs/$1/eval/logs/bench.log" 2>&1; say "   [$1] IntPhys1 · IntPhys2 끝 $?"
}
inflevel() {  # run
  say "   [$1] InfLevel 시작"
  NOWAIT=1 SKIP_GPU_WAIT=1 MB=72 VB=8 RUNS="$1" BENCHES="inflevel_continuity inflevel_solidity inflevel_gravity" bash z_training/chain_bench.sh \
    >> "z_training/runs/$1/eval/logs/inflevel_chain.log" 2>&1; say "   [$1] InfLevel 끝 $?"
}

say "===== 병렬 시작 ====="
( late ${P}_wta3_h1   # 가설 2 IntPhys2 w48 은 순차 chain 이 띄운 eval_bench 가 돌고 있다 → 끝나길 기다린 뒤 (있는 칸은 건너뜀) late
  while pgrep -f "eval_bench.sh ${P}_wta3_h2" > /dev/null; do sleep 30; done
  bench ${P}_wta3_h2; late ${P}_wta3_h2 ) & sleep 45
( inflevel ${P}_wta3_h0; inflevel ${P}_ftl1 ) & sleep 45
( inflevel ${P}_wta3_h1; bench ${P}_ftl1; late ${P}_ftl1 ) & sleep 45
( inflevel ${P}_wta3_h2 ) &
wait
say "병렬 끝"
