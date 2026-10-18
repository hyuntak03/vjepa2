#!/bin/bash
# 2026-10-13 자연 영상 (SSv2 val) 학습 신호 배분 — 병목 체인이 끝나면 세 모델. vll5 GPU 4.
set -uo pipefail; cd /data/hyuntak/project/2026/2027_cvpr/vjepa2; P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
log() { echo "[natural $(date +%H:%M:%S)] $*"; }
until grep -q "BOTTLENECK DONE" auto_research/logs/chain_1012_bottleneck.log; do sleep 30; done
for spec in "ar40 z_training/ariel/ar_future_1e_5/latest.pt" "ar_state40 z_training/ariel/ar_state_future_1e_5/latest.pt" "release -"; do
  set -- $spec; n=$1; ck=$2; [[ $ck == - ]] && ck=""
  log "$n"; $P -u z_research/scripts/analysis/predictor_signal_natural.py --name $n ${ck:+--ckpt $ck} --gpus 4 --n 800 --bs 8 > auto_research/logs/natural_$n.log 2>&1; log "$n exit $?"
  tail -16 auto_research/logs/natural_$n.log
done
log "NATURAL DONE"
