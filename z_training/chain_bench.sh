#!/bin/bash
# GRASP · InfLevel (Garrido A.8, intphys1_sliding) 채점 chain (2026-10-15).
#   chain_vob 가 끝나면 (로그에 "끝") 모델마다 · 벤치마크마다 · 창 (16, 32) 마다 따로 돈다 (run_all.sh 규칙).
#   batch 를 크게 (기본 max_batch 48 · video_batch 6 → MB · VB). OOM 등으로 실패하면 batch 를 반으로 줄여 최대 2 번 다시.
# 결과: z_training/runs/<run>/eval/intphys1_sliding__<bench>_latest_w<W>/   로그: z_training/logs/chain_bench.log
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../z_research/scripts/harness" && pwd)/env.sh"
cd "$VJEPA2_ROOT"
RUNS=${RUNS:-"natural_ar_vw_state_vob natural_ar_vw_state"}
BENCHES=${BENCHES:-"inflevel_continuity inflevel_solidity inflevel_gravity grasp_level2"}
MB=${MB:-144}; VB=${VB:-16}
LOG=z_training/logs/chain_bench.log; mkdir -p z_training/logs
say() { echo "[$(TZ=Asia/Seoul date '+%F %T KST')] $*" | tee -a "$LOG"; }
skips() { case $1 in grasp_level2) echo "[2,5,10]" ;; inflevel_*) echo "[5,10,20]" ;; esac; }
if [[ -z ${NOWAIT:-} ]]; then
  say "대기: chain_vob 끝"
  until grep -q "끝$" z_training/logs/chain_vob.log 2>/dev/null; do sleep 60; done
fi
until ! nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 20; done
for run in $RUNS; do
  R=z_training/runs/$run; CK=$PWD/$R/latest.pt
  KS=$("$VJEPA2_PY" -c "import yaml;pc=yaml.safe_load(open('$R/config.yaml'))['model']['predictor'];print(' '.join(f'model.predictor.{k}={pc[k]}' for k in ('kind','n_registers','ar_feedback') if k in pc))")
  for b in $BENCHES; do for w in 16 32; do
    OUT=$R/eval/intphys1_sliding__${b}_latest_w$w
    [[ -f $OUT/summary.json ]] && { say "있음: $run $b w$w"; continue; }
    mb=$MB; vb=$VB
    for try in 1 2 3; do
      say "$run / $b / w$w 시작 (max_batch $mb, video_batch $vb, 시도 $try)"
      SET="model.predictor_checkpoint=$CK $KS surprise.intphys1.frame_skips=$(skips $b) surprise.intphys1.window_sizes=[$w] model.window_size=$w surprise.intphys1.max_batch=$mb surprise.intphys1.video_batch=$vb" \
        OUTDIR=$OUT TAG="${b}_${run}_w$w" GPUS=8 bash z_research/scripts/run.sh intphys1_sliding "$b" vith > "$R/eval/logs/${b}_w$w.log" 2>&1
      if [[ -f $OUT/summary.json ]]; then
        acc=$("$VJEPA2_PY" -c "import json;s=json.load(open('$OUT/summary.json'));b=s.get('best_avg') or {};print(f\"{100*b.get('block_pairwise',float('nan')):.2f}% ({(b.get('key') or '?').split('/')[0]})\")")
        say "  완료: $acc"; break
      fi
      say "  !! 실패 ($(grep -m1 -oE 'OutOfMemoryError|Error[^ ]*' "$R/eval/logs/${b}_w$w.log" || echo '로그 확인'))"
      mb=$((mb / 2)); vb=$(( vb > 1 ? vb / 2 : 1 ))
      until ! nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 10; done
    done
  done; done
done
say "끝"
