#!/bin/bash
# 2026-10-13 병목 진단 전체 (재시작판). ⚠️ job 220434 는 메모리 160 GB 제한 — /dev/shm 특징 캐시도 여기에 잡힌다 (10-13 00:10 SIGKILL 두 번).
#   그래서 한 번에 캐시 하나만, 자 학습이 끝나면 바로 지운다.
set -uo pipefail; cd /data/hyuntak/project/2026/2027_cvpr/vjepa2; P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
log() { echo "[master $(date +%H:%M:%S)] $*"; }
# 1. release 자 (VOB_readout p) — 특징 46 GB
FR=/data2/local_datasets/world/world_analysis/VOB_readout_primitive; FEAT=/dev/shm/vobro_release; OUT=$PWD/z_research/VOBench_v1/readout/identity_vobprim_release
mkdir -p "$OUT"; rm -rf "$FEAT"
log "release 특징"; PRESENCE_SET=vob_readout_primitive PRESENCE_STRIDE=1 PRESENCE_FRAMES=$FR PRESENCE_REPS=p $P -u z_research/scripts/analysis/rollout2_presence_readout.py \
   --set vob_readout_primitive --feat-dir "$FEAT" --extract-only --extract-bs 32 --gpus 4 > "$OUT/extract.log" 2>&1 || { log "release 특징 실패"; tail -5 "$OUT/extract.log"; exit 1; }
$P - <<PY || { log "특징이 비었다"; exit 1; }
import numpy as np; p=np.load("$FEAT/p.npy", mmap_mode="r"); x=np.asarray(p[[0,4000,8800]]).astype("float32"); assert np.abs(x).mean()>0.1, np.abs(x).mean(); print("특징 검사 통과", np.abs(x).mean())
PY
log "release 자 학습"; PRESENCE_SET=vob_readout_primitive IDENTITY_FEAT_DIR=$FEAT IDENTITY_OUT=$OUT $P -u z_research/scripts/analysis/rollout2_identity_readout.py --launch --reps p --plan "p=0,1,2,3" > "$OUT/train.log" 2>&1
grep -E "\[p\] 정체" "$OUT/train.log" | tail -1; rm -rf "$FEAT"
# 2. 병목 추출 · 분석
S=z_research/scripts/analysis/predictor_bottleneck.py
for spec in "ar40 z_training/ariel/ar_future_1e_5/latest.pt" "ar_state40 z_training/ariel/ar_state_future_1e_5/latest.pt" "release -"; do
  set -- $spec; n=$1; ck=$2; [[ $ck == - ]] && ck=""
  log "병목 추출 $n"; $P -u $S extract --name $n ${ck:+--ckpt $ck} --gpus 4 --n-train 1600 --n-test 1200 --bs 8 > auto_research/logs/bottleneck_$n.log 2>&1; log "$n exit $?"
  grep -E "재구현|done" auto_research/logs/bottleneck_$n.log | head -5
done
$P -u z_research/scripts/analysis/predictor_bottleneck_analyze.py --names ar40 ar_state40 release > auto_research/logs/bottleneck_analyze.log 2>&1; log "analyze exit $?"
log "BOTTLENECK DONE"
# 3. 자연 영상 학습 신호
for spec in "ar40 z_training/ariel/ar_future_1e_5/latest.pt" "ar_state40 z_training/ariel/ar_state_future_1e_5/latest.pt" "release -"; do
  set -- $spec; n=$1; ck=$2; [[ $ck == - ]] && ck=""
  log "자연 $n"; $P -u z_research/scripts/analysis/predictor_signal_natural.py --name $n ${ck:+--ckpt $ck} --gpus 4 --n 800 --bs 8 > auto_research/logs/natural_$n.log 2>&1; log "$n exit $?"
done
log "NATURAL DONE"
# 4. late 정체
L=z_research/scripts/analysis/predictor_late_identity.py
for spec in "release -" "ar40 z_training/ariel/ar_future_1e_5/latest.pt" "ar_state40 z_training/ariel/ar_state_future_1e_5/latest.pt"; do
  set -- $spec; n=$1; ck=$2; [[ $ck == - ]] && ck=""
  log "late $n"; $P -u $L extract --name $n ${ck:+--ckpt $ck} --gpus 4 --bs 8 > auto_research/logs/late_$n.log 2>&1; log "$n exit $?"
done
$P -u $L analyze --names release ar40 ar_state40 > auto_research/logs/late_analyze.log 2>&1; log "late analyze exit $?"
log "MASTER DONE"
