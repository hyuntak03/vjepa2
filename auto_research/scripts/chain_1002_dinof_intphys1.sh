#!/bin/bash
# 2026-10-02 DINO-Foresight (highres) × IntPhys 1 dev, Garrido sliding (A.8 격자 skip [2,5,10]) — C=4 고정이라 window 마다 실행을 나눈다 (w5: M=1 · w8: M=4 · w16: M=12).
set -uo pipefail; cd /data/hyuntak/project/2026/2027_cvpr/vjepa2; source /data/hyuntak/anaconda3/bin/activate vjepa2
log() { echo "[chain $(date +%H:%M:%S)] $*"; }
MODEL=${MODEL:-dinof_highres}; GPUS=${GPUS:-4}; MB=${MB:-16}; VB=${VB:-2};   # stretch (S=2048) 는 MB=4 VB=1 (24 GB 에서 max_batch 16 은 OOM, 10-03)
 COPY=${COPY:-0}; STRETCH=${STRETCH:-0}; SFX=""; XSET=""; [[ $STRETCH == 1 ]] && { SFX="_stretch"; XSET="model.dinof_stretch=true"; }; [[ $COPY == 1 ]] && { SFX="${SFX}_copy"; XSET="$XSET model.dinof_copy=true"; }
for spec in "5 13" "8 8" "16 4"; do set -- $spec; w=$1; m=$2
  tag=intphys1_dev_${MODEL}_w${w}${SFX}; out=z_research/IntPhys/exp_results/intphys1_sliding__intphys1_dev_${MODEL}_w${w}${SFX}
  if [[ -f $out/summary.json ]]; then log "skip $tag (있음)"; continue; fi
  log "run $tag (window $w, context_mult $m → C=4, M=$((w-4)))"
  GPUS=$GPUS TAG=$tag OUTDIR=$out SET="surprise.intphys1.frame_skips=[2,5,10] surprise.intphys1.window_sizes=[$w] surprise.intphys1.context_mult=[$m] model.window_size=$w surprise.intphys1.max_batch=${MB:-16} surprise.intphys1.video_batch=${VB:-2} $XSET" \
    bash z_research/scripts/run.sh intphys1_sliding intphys1_dev $MODEL > auto_research/logs/${tag}.log 2>&1; log "$tag exit $?"
  grep -E "block_pairwise|BEST|O1 |O2 |O3 " auto_research/logs/${tag}.log | tail -8
done
log "CHAIN DONE"
