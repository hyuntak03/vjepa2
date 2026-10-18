#!/bin/bash
# =============================================================================
# 학습을 **tmux 세션**으로 올린다 (이 노드에서 직접, SLURM 없이). 세션이 셸과 무관하게 살아남는다.
#
#   GPUS=6 bash z_vll7_training/tmux_train.sh ctxt_video_tgtvideo [NAME]
#
#   보기:   watch -c -n 2 bash z_vll7_training/monitor.sh <NAME>
#   붙기:   tmux attach -t train_<NAME>        (떼기: Ctrl-b d)
#   중단:   tmux kill-session -t train_<NAME>  (rank 는 PDEATHSIG 로 같이 죽는다)
#   재개:   같은 NAME 으로 다시 올리면 <run>/latest.pt 에서 이어 돈다
# ⚠️ vll7 은 SLURM 노드다 — GPU 할당 없이 로그인 셸에서 띄우면 다른 job 과 GPU 를 다툰다.
#    할당 안에서 쓰거나 z_vll7_training/sbatch.sh 를 쓸 것.
# =============================================================================
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
cd "$VJEPA2_ROOT"
CONFIG=${1:?"사용법: tmux_train.sh <config> [NAME]"}
NAME=${2:-${NAME:-$(basename "${CONFIG%.yaml}")}}
SESS="train_$NAME"
if tmux has-session -t "=$SESS" 2>/dev/null; then echo "이미 있다: tmux attach -t $SESS"; exit 1; fi
mkdir -p "$VLL7_RUNS/$NAME"
ENVS="GPUS=${GPUS:-6} NAME=$NAME"
[[ -n "${SET:-}" ]]   && ENVS="$ENVS SET=$(printf %q "$SET")"
[[ -n "${LIMIT:-}" ]] && ENVS="$ENVS LIMIT=$LIMIT"
tmux new-session -d -s "$SESS" -c "$VJEPA2_ROOT" \
  "source \"$CONDA_ACTIVATE\" \"$CONDA_ENV\"; env $ENVS bash z_vll7_training/train.sh $CONFIG; \
   echo; echo '=== train.sh 종료 (exit '\$?') — 이 창은 로그 확인용으로 남겨 둔다 ==='; exec bash"
echo "올렸다: tmux 세션 $SESS  (config $CONFIG, GPUS ${GPUS:-6}${SET:+, SET=\"$SET\"})"
echo "  watch -c -n 2 bash z_vll7_training/monitor.sh $NAME"
echo "  tmux attach -t $SESS"
