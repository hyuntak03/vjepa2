#!/bin/bash
# 2026-10-12 병목 진단용 자 두 개 (VOB_readout_primitive, identity_r8 레시피): ar_state ep40 p · release p. vll5 GPU 4 (job 220434 안).
set -uo pipefail; cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
log() { echo "[readouts $(date +%H:%M:%S)] $*"; }
NAME=ar_state40 CKPT=z_training/ariel/ar_state_future_1e_5/latest.pt REPS="p" PLAN="p=0,1,2,3" bash z_research/scripts/analysis/vob_readout_chain.sh; log "ar_state40 exit $?"
# release: 체인은 CKPT 를 요구하므로 같은 두 단계를 PRED_CKPT 없이 직접
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python; FR=/data2/local_datasets/world/world_analysis/VOB_readout_primitive; FEAT=/dev/shm/vobro_release
OUT=$PWD/z_research/VOBench_v1/readout/identity_vobprim_release; mkdir -p "$OUT"
export PRESENCE_SET=vob_readout_primitive PRESENCE_STRIDE=1 PRESENCE_FRAMES=$FR PRESENCE_REPS=p; unset PRED_CKPT
if [[ ! -f $FEAT/done ]]; then log "release 추출"; $P -u z_research/scripts/analysis/rollout2_presence_readout.py --set vob_readout_primitive --feat-dir "$FEAT" --extract-only --extract-bs 32 --gpus 4 > "$OUT/extract.log" 2>&1 && touch "$FEAT/done"; fi
log "release 자 학습"; IDENTITY_FEAT_DIR=$FEAT IDENTITY_OUT=$OUT $P -u z_research/scripts/analysis/rollout2_identity_readout.py --launch --reps p --plan "p=0,1,2,3" > "$OUT/train.log" 2>&1; log "release exit $?"
grep -E "조합|정체" z_research/VOBench_v1/readout/identity_vobprim_ar_state40/train.log | tail -3; grep -E "조합|정체" "$OUT/train.log" | tail -3
log "READOUTS DONE"
