#!/bin/bash
# 자연 영상 predictor 사전학습 — 미래만 · full attention · mask token (kind oneshot), K400 + SSv2 + HowTo100M 1 % (2026-10-12, 사용자 지시).
#   tmux new-session -d -s chain_ksh "bash z_research/action_planning/chain_full_ksh.sh; exec bash"
# config configs/training/natural_full_ksh.yaml (70 epoch × 294 step × global 504 = 1,037 만 clip ≈ vast k710_ssv2_how_ep70).
# chain_ars_e100.sh (Push-T e100) 가 끝나고 GPU 가 비면 시작한다. 같은 NAME 으로 다시 부르면 latest.pt 에서 이어 간다.
# 로그 z_research/action_planning/exp_results/chain_full_ksh.log
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source /nas2/data/takhyun03/anaconda3/bin/activate vjepa2
OUT=z_research/action_planning/exp_results; mkdir -p $OUT z_training/logs; LOG=$OUT/chain_full_ksh.log
NAME=${NAME:-nat_full_ksh_scratch}; CFG=natural_full_ksh
say() { echo "[$(date '+%m-%d %H:%M')] $*" | tee -a $LOG; }
gpus_free() { until [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -eq 0 ]; do sleep 30; done; }

say "chain 시작 — chain_ars_e100.sh / chain_state_ft.sh 종료 대기"
while pgrep -f "bash z_research/action_planning/chain_(ars_e100|state_ft).sh" > /dev/null; do sleep 120; done
say "앞 chain 끝"

if grep -q "\] done$" z_training/runs/$NAME/train.log 2>/dev/null; then
  say "$NAME 학습 이미 끝 — 건너뜀"; exit 0
fi
if ! DRYRUN=1 GPUS=8 NAME=$NAME bash z_training/train.sh $CFG > z_training/logs/$NAME.dryrun.log 2>&1; then
  say "⚠️ DRYRUN 실패 — z_training/logs/$NAME.dryrun.log"; exit 1
fi
gpus_free; say "$NAME 학습 시작 (config $CFG, 70 epoch = 20,580 step × 504)"
GPUS=8 NAME=$NAME bash z_training/train.sh $CFG > z_training/logs/$NAME.chain.log 2>&1
if grep -q "\] done$" z_training/runs/$NAME/train.log 2>/dev/null; then
  say "$NAME 학습 끝: $(grep -E 'val\[' z_training/runs/$NAME/train.log | tail -n 1 | cut -c1-200)"
else
  say "⚠️ $NAME 학습 실패 — 로그 z_training/logs/$NAME.chain.log"; exit 1
fi
say "chain 끝"
