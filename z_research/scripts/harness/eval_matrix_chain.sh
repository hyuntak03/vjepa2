#!/bin/bash
# 평가 행렬 (z_research/VOBench_v1/Archive/EVAL_MATRIX_2026-10-15.md) 의 PFT 행을 vll5 GPU 4 장에서 채운다 (2026-10-15).
# PFT = VOBench split v2 train + gravity train 의 가능 clip (configs/training/vobench_{release,vast_ar_state}_postft_g.yaml).
# 순서 (사용자 10-15): gravity → (VOBench test · IntPhys 1) → IntPhys 2 → InfLevel → GRASP. 두 줄 (GPU 0·1 / 2·3) 이 순서 큐 하나에서 앞에서부터 꺼낸다.
#   init : 큐 파일을 만든다 (이미 있으면 그대로)
#   T    : 두 PFT 학습 (GPU 4 장, 차례로)
#   L1   : GPU 0 1 — 줄 T 가 끝나면 큐에서 꺼낸다      L2 : GPU 2 3 — 같은
#   LOG_DIR=<dir> bash z_research/scripts/harness/eval_matrix_chain.sh init|T|L1|L2
# vll5 밖 노드에서 도는 칸 (vast AR · release 의 IntPhys 2 · InfLevel · GRASP) 은 eval_matrix_node.sh.
set -uo pipefail
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
D=${LOG_DIR:-/tmp}; LOG=$D/eval_matrix_$1.log; Q=$D/eval_matrix_queue_g.txt
log() { echo "[matrix $1 $(date +%m-%d_%H:%M:%S)] ${*:2}" | tee -a "$LOG"; }
# DDP 포트를 줄마다 따로 준다 — 10-15 03:38 두 줄이 같은 초에 시작해 launch.sh 의 빈 포트 탐색이 둘 다 20434 를 골라 gather 에서 엉켜 멈췄다.
NJOB=0
run() { local lane=$1 name=$2; shift 2; NJOB=$((NJOB + 1)); export EVAL_DDP_PORT=$(( ${PORT_BASE:-29000} + NJOB ))
        log "$lane" "시작 $name (port $EVAL_DDP_PORT)"; "$@" >> "$D/eval_matrix.$name.log" 2>&1; log "$lane" "  끝 $name rc=$?"; }
INF="inflevel_continuity inflevel_gravity inflevel_solidity"
RUNOF() { case $1 in vith_release_vobftg) echo vobench_release_postft_g ;; vith_vast_ar_state_ep70_vobftg) echo vobench_vast_ar_state_postft_g ;; esac; }

pop() {   # 첫 줄을 꺼낸다 (형식: 벤치|모델)
  exec 9>"$Q.lock"; flock 9
  local line; line=$(head -n1 "$Q")
  if [ -n "$line" ]; then tail -n +2 "$Q" > "$Q.tmp"; mv "$Q.tmp" "$Q"; fi
  flock -u 9; exec 9>&-
  echo "$line"
}

work() {
  local lane=$1 line b m
  while line=$(pop); [ -n "$line" ]; do
    IFS='|' read -r b m <<<"$line"
    case $b in
      gravity)  run "$lane" "gravity_$m"  env RESULTS_ROOT=legacy bash cvpr/eval/testbed/run.sh vobench_v1_gravity "$m" ;;
      vobtest)  run "$lane" "vobtest_$m"  bash z_training/eval.sh "$(RUNOF "$m")" vobench_v1_split2_test ;;
      intphys1) run "$lane" "intphys1_$m" bash cvpr/eval/intphys1/run.sh intphys1_dev "$m" ;;
      intphys2) run "$lane" "intphys2_$m" bash cvpr/eval/intphys2/run.sh intphys2_main "$m" ;;
      inflevel) run "$lane" "inflevel_$m" bash cvpr/eval/intphys1/run.sh "$INF" "$m" ;;
      grasp)    run "$lane" "grasp_$m"    bash cvpr/eval/intphys1/run.sh grasp_level2 "$m" ;;
    esac
  done
}

case $1 in
  init)
    [ -e "$Q" ] && { echo "큐가 이미 있다: $Q"; cat "$Q"; exit 0; }
    for b in gravity vobtest intphys1 intphys2 inflevel grasp; do
      for m in vith_release_vobftg vith_vast_ar_state_ep70_vobftg; do echo "$b|$m"; done
    done > "$Q"
    cat "$Q"; exit 0 ;;
  T)
    export GPUS=4
    for c in vobench_release_postft_g vobench_vast_ar_state_postft_g; do run T "train_$c" bash z_training/train.sh "$c"; done ;;
  L1|L2)
    while pgrep -f "eval_matrix_chain.sh T" > /dev/null || pgrep -f "eval_matrix_T.log" > /dev/null || pgrep -f "z_training/train.sh vobench_.*_postft_g" > /dev/null; do sleep 60; done   # 10-15 학습은 손으로 띄운 루프 (로그 eval_matrix_T.log) 로 돌았다
    [ "$1" = L1 ] && export GPU_IDS="0 1" PORT_BASE=23000 || export GPU_IDS="2 3" PORT_BASE=24000
    export GPUS=2; work "$1" ;;
esac
log "$1" "줄 $1 전부 끝"
