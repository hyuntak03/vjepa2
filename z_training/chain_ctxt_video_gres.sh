#!/bin/bash
# nat_ar_state_ksh_scratch 가 끝나면 ctxt_video 10K step 두 판을 연달아 올린다 (2026-10-17, chain_ctxt_video.sh 대체).
#   tmux new-session -d -s chain_ctxt_video "bash z_training/chain_ctxt_video_gres.sh; exec bash"
# 1) VRAM: 최악 창 C32K32 를 GPU 8 장에서 (gating_residual 판으로) — token_budget 950 · accum 1, OOM 이면 475 × accum 2
# 2) nat_ar_vw_state_ctxt_video_gres   (natural_ar_vw_state_ctxt_video_gres, gate 지름길, 34 ep = 9,996 step)
# 3) nat_ar_vw_state_ctxt_video_10k    (gate 없는 대조군, 같은 34 ep · anneal 3)
# 로그 z_training/logs/chain_ctxt_video.log
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source /nas2/data/takhyun03/anaconda3/bin/activate vjepa2
LOG=z_training/logs/chain_ctxt_video.log; mkdir -p z_training/logs
WAIT_FOR=${WAIT_FOR:-nat_ar_state_ksh_scratch}
say() { echo "[$(date '+%m-%d %H:%M')] $*" | tee -a $LOG; }
gpus_free() { until [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -eq 0 ]; do sleep 30; done; }
train() {  # $1 NAME  $2 CFG  $3 SET
  gpus_free
  say "$1 학습 시작 (config $2${3:+, $3})"
  GPUS=8 NAME=$1 SET="$3" bash z_training/train.sh $2 > z_training/logs/$1.chain.log 2>&1
  if grep -q "\] done$" z_training/runs/$1/train.log 2>/dev/null; then say "$1 학습 끝"; else say "⚠️ $1 학습 실패 — z_training/logs/$1.chain.log"; fi
}

if [ -n "${SKIP_WAIT:-}" ]; then
  say "chain (gres 판) 시작 — 대기 없이 (SKIP_WAIT)"
else
  say "chain (gres 판) 시작 — $WAIT_FOR 종료 대기"
  until grep -q "\] done$" z_training/runs/$WAIT_FOR/train.log 2>/dev/null; do sleep 120; done
  say "$WAIT_FOR 끝"
fi
gpus_free
SMOKE=/tmp/${USER}_ctxt_video_smoke; rm -rf $SMOKE
SETS=""
PAIRS=("950 1" "475 2"); [ -n "${ONLY_BUDGET:-}" ] && PAIRS=("$ONLY_BUDGET")   # 예: ONLY_BUDGET="475 2" (950 은 48 GB 에서 OOM, 10-17 실측)
for ba in "${PAIRS[@]}"; do
  read -r BUD ACC <<< "$ba"
  say "VRAM 점검: token_budget $BUD × accum $ACC, 최악 창 C32K32, GPU 8 장 (gating_residual)"
  GPUS=8 FOLDER=$SMOKE/b$BUD SET="data.var_window.token_budget=$BUD optimization.grad_accum=$ACC data.var_window.context_frames=[32] data.var_window.predict_frames=[32] optimization.epochs=1 optimization.ipe=3 optimization.warmup=0 optimization.anneal_epochs=0 val=null" \
    timeout 2400 bash z_training/train.sh natural_ar_vw_state_ctxt_video_gres > $SMOKE.b$BUD.log 2>&1
  if grep -q "\] done$" $SMOKE/b$BUD/train.log 2>/dev/null && ! grep -q OutOfMemory $SMOKE/b$BUD/train.log; then
    say "  통과: $(grep -oE 'mem [0-9.]+G' $SMOKE/b$BUD/train.log | sort -t' ' -k2 -n | tail -1) (max allocated)"
    [ "$BUD" = 950 ] || SETS="data.var_window.token_budget=$BUD optimization.grad_accum=$ACC"
    OK=1; break
  fi
  say "  실패 (OOM 또는 에러) — 로그 $SMOKE.b$BUD.log"
  gpus_free
done
[ -z "${OK:-}" ] && { say "⚠️ VRAM 점검 모두 실패 — 학습 안 올림"; exit 1; }
train nat_ar_vw_state_ctxt_video_gres natural_ar_vw_state_ctxt_video_gres "$SETS"
train nat_ar_vw_state_ctxt_video_10k  natural_ar_vw_state_ctxt_video      "optimization.epochs=34 optimization.anneal_epochs=3${SETS:+ $SETS}"
