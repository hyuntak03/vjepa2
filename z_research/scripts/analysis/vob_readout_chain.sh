#!/bin/bash
# VOB_readout_primitive 학습셋으로 위치 + 정체 (+ 없음) 자를 한 predictor 에 맞춰 학습 (2026-10-12).
#   index: z_research/scripts/data/build_vob_readout_index.py → data_csv/rollout_v2_vob_readout_primitive/
#   특징: rollout2_presence_readout.py (--extract-only; kind=ar 면 p = 블록별 LN(h) 문맥 8 → rollout 8, h = 블록별 LN(h))
#   자:   rollout2_identity_readout.py (identity_r8 과 같은 레시피, split block 단위 50/15/35)
#
#   NAME=ar40 CKPT=z_training/ariel/ar_future_1e_5/latest.pt REPS="p h" PLAN="p=0,1 h=2,3" bash z_research/scripts/analysis/vob_readout_chain.sh
#   NAME=rel CKPT=release FEAT_ROOT=/data2/local_datasets/world/world_analysis/cache ... bash ...   # 릴리즈 predictor (oneshot) · 특징은 디스크
#   → z_research/VOBench_v1/readout/identity_vobprim_<NAME>/  (특징은 /dev/shm/vobro_<NAME>, 끝나면 남겨 둔다)
set -euo pipefail
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
NAME=${NAME:?}; CKPT=${CKPT:?}; REPS=${REPS:-"p h"}; PLAN=${PLAN:-"p=0,1 h=2,3"}
FR=/data2/local_datasets/world/world_analysis/VOB_readout_primitive
FEAT=${FEAT_ROOT:-/dev/shm}/vobro_$NAME          # FEAT_ROOT=/data2/... 이면 디스크 (job 메모리 한도에 /dev/shm 이 잡힌다, 2026-10-13)
OUT=$PWD/z_research/VOBench_v1/readout/identity_vobprim_$NAME
mkdir -p "$OUT"
log() { echo "[vobro $(date +%H:%M:%S)] $*"; }

while [[ $(ls "$FR/Frames" | wc -l) -lt 8820 ]] || pgrep -f "cp -r .*VOB_readout_primitive" > /dev/null; do sleep 20; done
log "프레임 $(ls "$FR/Frames" | wc -l) clip · $(du -sh "$FR" | cut -f1)"

export PRESENCE_SET=vob_readout_primitive PRESENCE_STRIDE=1 PRESENCE_FRAMES=$FR PRESENCE_REPS=${REPS// /,}
if [[ $CKPT == release ]]; then unset PRED_CKPT; else export PRED_CKPT=$PWD/$CKPT; fi   # release = 릴리즈 predictor 그대로
if [[ ! -f $FEAT/done ]]; then
  log "추출 → $FEAT"
  $P -u z_research/scripts/analysis/rollout2_presence_readout.py --set vob_readout_primitive --feat-dir "$FEAT" --extract-only \
      --extract-bs 32 --gpus "$(nvidia-smi -L | wc -l)" > "$OUT/extract.log" 2>&1
  touch "$FEAT/done"
fi
log "  $(grep -E '^\[extract\] 끝' "$OUT/extract.log" | tail -1)"

log "자 학습 ($REPS, $PLAN)"
IDENTITY_FEAT_DIR=$FEAT IDENTITY_OUT=$OUT $P -u z_research/scripts/analysis/rollout2_identity_readout.py \
    --launch --reps $REPS --plan "$PLAN" > "$OUT/train.log" 2>&1
log "  $(grep -E '조합|정체' "$OUT/train.log" | tail -3 | tr '\n' ' ')"
log "끝 → $OUT"
