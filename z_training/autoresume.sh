#!/bin/bash
# =============================================================================
# 부팅 시 학습 자동 재개 (2026-09-29). 호스트가 예고 없이 재부팅되는 인스턴스용.
#
#   켜기:  touch z_training/runs/<NAME>/.autoresume        (tmux_train.sh 로 한 번 올린 run)
#   끄기:  rm    z_training/runs/<NAME>/.autoresume
#   훅:    /etc/rc.local 에  nohup bash <레포>/z_training/autoresume.sh >> /var/log/autoresume.log 2>&1 &
#   로그:  /var/log/autoresume.log
#
# 하는 일: .autoresume 표시가 있는 run 마다, **끝나지 않았고**(train.log 에 "done" 없음)
#   tmux 세션 train_<NAME> 이 없으면 tmux_train.sh 로 다시 올린다. 학습은 latest.pt 에서 이어 돈다
#   (meta.auto_resume). 재개 인자(config·GPUS·SET)는 run 폴더의 .autoresume 안에 적어 둔다:
#     CONFIG=natural_tube_pretrain
#     GPUS=8
#     SET="..."            (없으면 생략)
# =============================================================================
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "[$(date '+%F %T')] autoresume 시작 (ROOT=$ROOT)"
sleep "${AUTORESUME_DELAY:-90}"          # GPU·네트워크·supervisor 가 올라올 시간
if [[ -f "$ROOT/z_training/logs/.resize_pending" ]] && ! tmux has-session -t resize_k400 2>/dev/null; then
  echo "  K400 재인코딩이 끝나지 않았다 -> resize_k400 다시 올림 (끝나면 학습을 스스로 올린다)"
  tmux new-session -d -s resize_k400 -c "$ROOT" "bash z_training/data/finalize_k400_320.sh 2>&1 | tee -a z_training/logs/resize_k400.log; exec bash"
fi
for mark in "$ROOT"/z_training/runs/*/.autoresume; do
  [[ -f "$mark" ]] || continue
  run_dir=$(dirname "$mark"); NAME=$(basename "$run_dir")
  if grep -q "\] done$" "$run_dir/train.log" 2>/dev/null; then
    echo "  $NAME: 이미 끝났다 -> 표시 제거"; rm -f "$mark"; continue
  fi
  if tmux has-session -t "train_$NAME" 2>/dev/null; then
    echo "  $NAME: tmux 세션이 이미 있다 -> 건너뜀"; continue
  fi
  CONFIG=""; GPUS=8; SET=""
  # shellcheck source=/dev/null
  source "$mark"
  [[ -n "$CONFIG" ]] || { echo "  $NAME: .autoresume 에 CONFIG 가 없다 -> 건너뜀"; continue; }
  echo "  $NAME: 재개 (config $CONFIG, GPUS $GPUS)"
  echo "[$(date '+%F %T')] ===== autoresume: 재부팅 뒤 재개 =====" >> "$run_dir/train.log"
  ( cd "$ROOT" && GPUS=$GPUS NAME=$NAME SET="$SET" bash z_training/tmux_train.sh "$CONFIG" "$NAME" )
done
echo "[$(date '+%F %T')] autoresume 끝"
