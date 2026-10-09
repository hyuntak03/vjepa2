#!/bin/bash
# 2026-10-06 DINO-Foresight (highres, 정방형 448) × IntPhysGen v11 — 표준 채점 surprise_c16t32 (matched 10,752 쌍). vll3 (사용자 할당 job 안에서 bash).
#   모델은 문맥 4 프레임만 받으므로 문맥 16 장 중 **마지막 4 장**으로 미래 16 장을 자기회귀 unroll (16 걸음). 복사 기준선 (마지막 문맥 프레임 × 16) 도 같은 채점으로.
set -uo pipefail; ROOT=/data/hyuntak/project/2026/2027_cvpr/vjepa2; cd $ROOT; source /data/hyuntak/anaconda3/bin/activate vjepa2
log() { echo "[chain $(date +%H:%M:%S)] $*"; }
D=/local_datasets/world/world_analysis; TAR=$ROOT/auto_research/_stage/v11_six.tar
if [[ ! -d $D/IntPhysGen_v11/Images/moving_occlusion ]]; then log "untar v11 (6 조건)"; mkdir -p $D; tar xf $TAR -C $D || exit 1; fi
ls $D/IntPhysGen_v11/Images; df -h / | tail -1
MODEL=${MODEL:-dinof_highres}; GPUS=${GPUS:-8}; BS=${BS:-8}
log "smoke (GPU 1, 4 clip)"
GPUS=1 LIMIT=4 SET="surprise.batch_size=$BS surprise.decode_workers=8" bash z_research/scripts/run.sh surprise_c16t32 v11 $MODEL > auto_research/logs/v11_${MODEL}_smoke.log 2>&1 || { tail -20 auto_research/logs/v11_${MODEL}_smoke.log; exit 1; }
grep -E "dinof predictor|block_pairwise|overall" auto_research/logs/v11_${MODEL}_smoke.log | head -5
for mode in main copy; do
  if [[ $mode == copy ]]; then TAG=v11_${MODEL}_copy; OUT=z_research/IntPhysGenV11/exp_results/surprise_c16t32__v11_${MODEL}_copy; X="model.dinof_copy=true"; else TAG=v11_${MODEL}; OUT=z_research/IntPhysGenV11/exp_results/surprise_c16t32__v11_${MODEL}; X=""; fi
  [[ -f $OUT/summary.json ]] && { log "skip $TAG (있음)"; continue; }
  log "run $TAG"; GPUS=$GPUS TAG=$TAG OUTDIR=$OUT SET="surprise.batch_size=$BS surprise.decode_workers=8 $X" bash z_research/scripts/run.sh surprise_c16t32 v11 $MODEL > auto_research/logs/${TAG}.log 2>&1; log "$TAG exit $?"
  grep -E "block_pairwise|overall|vanish|shape|color" auto_research/logs/${TAG}.log | tail -12
done
log "CHAIN DONE"
