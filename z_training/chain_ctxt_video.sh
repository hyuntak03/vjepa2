#!/bin/bash
# nat_ar_state_ksh_scratch 가 끝나면 vast 의 ctxt_video (문맥 = 영상, ctx_target block, 가변 창) 를 올린다 (2026-10-17).
#   tmux new-session -d -s chain_ctxt_video "bash z_training/chain_ctxt_video.sh; exec bash"
# config configs/training/natural_ar_vw_state_ctxt_video.yaml (vast 061d07c 그대로, K710 → K400)
# 1) VRAM: 최악 창 C32K32 (64 장, rank B 14) 를 GPU 8 장에서 — vast 와 같은 token_budget 950 · accum 1 부터,
#    OOM 이면 budget 475 × grad_accum 2 (optimizer step 당 토큰 수가 같다)
# 2) 학습 nat_ar_vw_state_ctxt_video. 로그 z_training/logs/chain_ctxt_video.log
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source /nas2/data/takhyun03/anaconda3/bin/activate vjepa2
LOG=z_training/logs/chain_ctxt_video.log; mkdir -p z_training/logs
WAIT_FOR=${WAIT_FOR:-nat_ar_state_ksh_scratch}; NAME=${NAME:-nat_ar_vw_state_ctxt_video}; CFG=natural_ar_vw_state_ctxt_video
say() { echo "[$(date '+%m-%d %H:%M')] $*" | tee -a $LOG; }
gpus_free() { until [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -eq 0 ]; do sleep 30; done; }

say "chain 시작 — $WAIT_FOR 종료 대기"
until grep -q "\] done$" z_training/runs/$WAIT_FOR/train.log 2>/dev/null; do sleep 120; done
say "$WAIT_FOR 끝"
gpus_free
SMOKE=/tmp/${USER}_ctxt_video_smoke; rm -rf $SMOKE
SETS=""
for ba in "950 1" "475 2"; do
  set -- $ba
  say "VRAM 점검: token_budget $1 × accum $2, 최악 창 C32K32, GPU 8 장"
  GPUS=8 FOLDER=$SMOKE/b$1 SET="data.var_window.token_budget=$1 optimization.grad_accum=$2 data.var_window.context_frames=[32] data.var_window.predict_frames=[32] optimization.epochs=1 optimization.ipe=3 optimization.warmup=0 optimization.anneal_epochs=0 val=null" \
    timeout 2400 bash z_training/train.sh $CFG > $SMOKE.b$1.log 2>&1
  if grep -q "\] done$" $SMOKE/b$1/train.log 2>/dev/null && ! grep -q OutOfMemory $SMOKE/b$1/train.log; then
    say "  통과: $(grep -oE 'mem [0-9.]+G' $SMOKE/b$1/train.log | sort -t' ' -k2 -n | tail -1) (max allocated)"
    [ "$1" = 950 ] || SETS="data.var_window.token_budget=$1 optimization.grad_accum=$2"
    OK=1; break
  fi
  say "  실패 (OOM 또는 에러) — 로그 $SMOKE.b$1.log"
  gpus_free
done
[ -z "${OK:-}" ] && { say "⚠️ VRAM 점검 모두 실패 — 학습 안 올림"; exit 1; }
gpus_free
say "$NAME 학습 시작 (config $CFG${SETS:+, $SETS})"
GPUS=8 NAME=$NAME SET="$SETS" bash z_training/train.sh $CFG > z_training/logs/$NAME.chain.log 2>&1
if grep -q "\] done$" z_training/runs/$NAME/train.log 2>/dev/null; then say "$NAME 학습 끝"; else say "⚠️ $NAME 학습 실패 — z_training/logs/$NAME.chain.log"; fi
