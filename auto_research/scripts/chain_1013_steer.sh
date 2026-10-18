#!/bin/bash
set -uo pipefail; cd /data/hyuntak/project/2026/2027_cvpr/vjepa2; P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
log() { echo "[steer $(date +%H:%M:%S)] $*"; }
until grep -q "MASTER DONE" auto_research/logs/chain_1013_master.log; do sleep 30; done
for spec in "ar_state40 z_training/ariel/ar_state_future_1e_5/latest.pt" "ar40 z_training/ariel/ar_future_1e_5/latest.pt"; do
  set -- $spec; log "$1"; $P -u z_research/scripts/analysis/predictor_velocity_steer.py --name $1 --ckpt $2 --gpus 4 --bs 8 > auto_research/logs/steer_$1.log 2>&1; log "$1 exit $?"
  grep -vE "warn|Warning|xFormers|\[r0\]|self.gen|predictor kind" auto_research/logs/steer_$1.log | tail -16
done
log "STEER DONE"
