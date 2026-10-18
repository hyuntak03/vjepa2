#!/bin/bash
# 측정판 감시 (2026-10-18). 학습 중인 run 의 e{N}.pt 가 생기면 (N % EVERY == 0) 측정판을 계산해 panel.jsonl 에 붙인다.
#   tmux new-session -d -s panel_<run> "bash z_training/tools/panel_watch.sh <run> [EVERY=10] [GPU=7]"
# 학습이 끝나면 (train.log 의 done) latest.pt 를 한 번 더 재고 끝난다. 이미 잰 checkpoint 는 건너뛴다.
RUN=$1; EVERY=${2:-10}; GPU=${3:-7}
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
D=z_training/runs/$RUN; PY=/venv/vjepa/bin/python
done_ck() { [[ -f $D/panel.jsonl ]] && grep -q "\"ckpt\": \"$1\"" $D/panel.jsonl; }
while true; do
  for f in $(ls $D 2>/dev/null | grep -E '^e[0-9]+\.pt$' | sort -V); do
    n=${f#e}; n=${n%.pt}
    (( n % EVERY == 0 )) && ! done_ck $f && CUDA_VISIBLE_DEVICES=$GPU $PY z_training/tools/panel.py $RUN $f 2>&1 | grep -vE "INFO|Warn|warn" >> $D/panel.log
  done
  if grep -qE '\] done$' $D/train.log 2>/dev/null; then
    done_ck latest.pt || CUDA_VISIBLE_DEVICES=$GPU $PY z_training/tools/panel.py $RUN latest.pt 2>&1 | grep -vE "INFO|Warn|warn" >> $D/panel.log
    exit 0
  fi
  sleep 300
done
