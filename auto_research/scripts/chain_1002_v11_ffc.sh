#!/bin/bash
# 2026-10-02 v11 late 가림 — 과거 16 frames 문맥 encoder 로 미래 위치 회귀 (z · p · h, 자 A · B). vll2 GPU 1 장 (3090).
set -uo pipefail; ROOT=/data/hyuntak/project/2026/2027_cvpr/vjepa2; cd $ROOT; source /data/hyuntak/anaconda3/bin/activate vjepa2
log() { echo "[chain $(date +%H:%M:%S)] $*"; }
ST=/dev/shm/v11stage; FE=/dev/shm/v11_ffc; S=auto_research/scripts/v11_future_from_context.py
if [[ ! -d $ST/IntPhysGen_v11/Images ]]; then log "untar v11 subset (12.4 GB NFS)"; mkdir -p $ST; tar xf auto_research/_stage/v11_readout_subset.tar -C $ST || exit 1; fi
ls $ST/IntPhysGen_v11/Images | head -3; log "extract (ViT-H, 1,344 clip)"
python $S extract --frames $ST/IntPhysGen_v11 --out $FE --bs 4 || exit 1
log "probe (GPU)"; python $S probe --feat $FE --out auto_research/exp_results/future_from_context/v11 --device cuda --epochs 60 || exit 1
log "copy features to NFS (21 GB)"; mkdir -p auto_research/_stage/v11_ffc_feats; cp $FE/*.npy $FE/labels.npz $FE/index.csv auto_research/_stage/v11_ffc_feats/ && log "copied"
log "CHAIN DONE"
