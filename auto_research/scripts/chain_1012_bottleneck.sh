#!/bin/bash
# 2026-10-12 predictor 병목 진단 — 자 두 개가 끝나면 세 모델 추출 → 분석. vll5 GPU 4 (job 220434).
set -uo pipefail; cd /data/hyuntak/project/2026/2027_cvpr/vjepa2; P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
log() { echo "[bottleneck $(date +%H:%M:%S)] $*"; }
until grep -q "READOUTS DONE" auto_research/logs/chain_1012_readouts.log; do sleep 30; done
log "자 준비됨"; grep -E "exit|정체" auto_research/logs/chain_1012_readouts.log | tail -6
S=z_research/scripts/analysis/predictor_bottleneck.py
for spec in "ar40 z_training/ariel/ar_future_1e_5/latest.pt" "ar_state40 z_training/ariel/ar_state_future_1e_5/latest.pt" "release -"; do
  set -- $spec; n=$1; ck=$2; [[ $ck == - ]] && ck=""
  if ls z_research/VOBench_v1/bottleneck/$n/shard3.npz >/dev/null 2>&1; then log "skip $n"; continue; fi
  log "추출 $n"; $P -u $S extract --name $n ${ck:+--ckpt $ck} --gpus 4 --n-train 1600 --n-test 1200 --bs 8 > auto_research/logs/bottleneck_$n.log 2>&1; log "$n exit $?"
  grep -E "재구현|done" auto_research/logs/bottleneck_$n.log | head -5
done
$P -u z_research/scripts/analysis/predictor_bottleneck_analyze.py --names ar40 ar_state40 release > auto_research/logs/bottleneck_analyze.log 2>&1; log "analyze exit $?"
log "BOTTLENECK DONE"
