#!/bin/bash
# 2026-10-13 VOBench late 정체 — 자연 영상 체인 뒤. vll5 GPU 4.
set -uo pipefail; cd /data/hyuntak/project/2026/2027_cvpr/vjepa2; P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
log() { echo "[late $(date +%H:%M:%S)] $*"; }
until grep -q "NATURAL DONE" auto_research/logs/chain_1013_natural.log; do sleep 30; done
S=z_research/scripts/analysis/predictor_late_identity.py
for spec in "release -" "ar40 z_training/ariel/ar_future_1e_5/latest.pt" "ar_state40 z_training/ariel/ar_state_future_1e_5/latest.pt"; do
  set -- $spec; n=$1; ck=$2; [[ $ck == - ]] && ck=""
  log "$n"; $P -u $S extract --name $n ${ck:+--ckpt $ck} --gpus 4 --bs 8 > auto_research/logs/late_$n.log 2>&1; log "$n exit $?"
done
$P -u $S analyze --names release ar40 ar_state40 > auto_research/logs/late_analyze.log 2>&1; log "analyze exit $?"
log "LATE DONE"
