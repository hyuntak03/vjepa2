#!/bin/bash
# =============================================================================
# 앞 run 이 끝나면 다음 config 를 올린다 (2026-09-30). tmux 로 띄운다:
#   tmux new-session -d -s chain_tube "bash z_training/chain_after.sh natural_ctx_state natural_tube_k710"
#
# 1. <앞 run>/train.log 에 'done' 이 찍히거나 tmux 세션 train_<앞 run> 이 사라질 때까지 기다린다
#    (실패로 끝났으면 다음을 올리지 않는다 — train.log 에 done 이 없으면 멈춤)
# 2. 다음 config 를 GPU 1 장으로 몇 step 돌려 **rank 당 batch 가 들어가는지** 본다 (가장 큰 것부터 BATCHES).
#    낮추면 본 clip 수를 유지하도록 ipe 와 lr 을 같이 맞춘다: ipe = CLIPS / (40 × 8 × B), lr = 3e-4 × sqrt(8B / 168)
# 3. 8 GPU 로 올린다 (tmux_train.sh)
# 로그: z_training/logs/chain_<다음 config>.log
# =============================================================================
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../z_research/scripts/harness" && pwd)/env.sh"
cd "$VJEPA2_ROOT"
PREV=${1:?앞 run 이름}; NEXT=${2:?다음 config}; NAME=${3:-$NEXT}
BATCHES=${BATCHES:-"64 56 48 40 32"}; GPUS=${GPUS:-8}; EPOCHS=${EPOCHS:-40}; CLIPS=${CLIPS:-6021120}   # 512 × 11,760
LOG=z_training/logs/chain_${NAME}.log; mkdir -p z_training/logs
say() { echo "[$(TZ=Asia/Seoul date '+%F %T KST')] $*" | tee -a "$LOG"; }

say "대기: $PREV 끝나기를 (다음 $NEXT -> run $NAME)"
until grep -qE '\] done$' "z_training/runs/$PREV/train.log" 2>/dev/null || ! tmux has-session -t "train_$PREV" 2>/dev/null; do sleep 60; done
if ! grep -qE '\] done$' "z_training/runs/$PREV/train.log" 2>/dev/null; then
  say "$PREV 가 done 없이 끝났다 — 다음을 올리지 않는다"; exit 1
fi
say "$PREV 완료. 세션 정리 대기"
until ! nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 20; done
tmux kill-session -t "train_$PREV" 2>/dev/null

PICK=
for B in $BATCHES; do
  say "메모리 검사: rank 당 batch $B (GPU 1 장, 4 step)"
  rm -rf "z_training/runs/_probe_$NAME"
  if DEBUG=1 NAME="_probe_$NAME" SET="data.batch_size=$B optimization.epochs=1 optimization.ipe=4 val=null meta.save_every_freq=-1" \
     PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True timeout 1800 bash z_training/train.sh "$NEXT" > "z_training/logs/_probe_${NAME}_b$B.log" 2>&1 \
     && grep -qE '\] done$' "z_training/runs/_probe_$NAME/train.log"; then
    mem=$(grep -oE 'mem [0-9.]+G' "z_training/logs/_probe_${NAME}_b$B.log" | tail -1)
    say "  통과 ($mem)"; PICK=$B; rm -rf "z_training/runs/_probe_$NAME"; break
  fi
  say "  실패 ($(grep -m1 -oE 'OutOfMemoryError|Error[^ ]*' "z_training/logs/_probe_${NAME}_b$B.log" || echo '원인 로그 확인'))"
  rm -rf "z_training/runs/_probe_$NAME"
done
[[ -z "$PICK" ]] && { say "들어가는 batch 가 없다 — 멈춤"; exit 1; }

# KEEP_OPT=1 (2026-10-01): batch 만 검사하고 ipe·lr 은 config 그대로 (cooldown 처럼 앞 단계와 lr 을 맞춰야 할 때)
if [[ -n "${KEEP_OPT:-}" ]]; then
  say "선택: rank 당 $PICK (global $((GPUS * PICK))), ipe·lr 은 config 그대로 (KEEP_OPT)"
  SETV="data.batch_size=$PICK"
else
  IPE=$(( (CLIPS + EPOCHS * GPUS * PICK - 1) / (EPOCHS * GPUS * PICK) ))
  LR=$("$VJEPA2_PY" -c "print(f'{3e-4 * ($GPUS * $PICK / 168) ** 0.5:.2e}')")
  say "선택: rank 당 $PICK (global $((GPUS * PICK))), ipe $IPE × $EPOCHS = $((IPE * EPOCHS)) step, lr $LR"
  SETV="data.batch_size=$PICK optimization.ipe=$IPE optimization.lr=$LR"
fi
if ! DRYRUN=1 SET="$SETV" bash z_training/train.sh "$NEXT" >> "$LOG" 2>&1; then say "DRYRUN 실패 — 멈춤"; exit 1; fi
GPUS=$GPUS SET="$SETV" bash z_training/tmux_train.sh "$NEXT" "$NAME" 2>&1 | tee -a "$LOG"
say "올림: tmux train_$NAME"
