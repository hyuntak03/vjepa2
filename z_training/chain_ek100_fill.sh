#!/bin/bash
# =============================================================================
# 2026-10-29 사용자: "이거 학습 끝나면 google sheet 에서 EK100 subset 비워져 있는 거 다 채울 수 있도록. 모델 여러 개 한 번에 8 GPU 로"
#
#   tmux new-session -d -s chain_ek100_fill "bash z_training/chain_ek100_fill.sh 2>&1 | tee -a z_training/logs/chain_ek100_fill.log"
#
#   1. release_postft_pretrain 학습이 끝나기를 기다린다 (train.log 의 "done")
#   2. GPU 8 장이 빌 때까지 기다린다 (다른 세션 작업과 겹치지 않게)
#   3. EK100 cap16metric (시트 "EK100 subset" 규약) — encoder 1 개 + predictor 6 팔 (cvpr/registry/models.yaml 의 vast_ek100_fill1029)
#      8 GPU × 12 = global 96 (cap16b96 과 같은 global batch)
#   4. 팔별 값 정리 (collect_group.py) — 시트 기입은 사람이 / 대화에서 한다
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=z_training/runs/release_postft_pretrain/train.log
OUTD=cvpr/results/eval/ek100/action_anticipation_frozen/ek100_vast_ek100_fill1029
say() { echo "[$(TZ=Asia/Seoul date '+%m-%d %H:%M KST')] $*"; }

say "1/4 학습 끝 대기 ($LOG)"
until grep -q "\] done$" "$LOG" 2>/dev/null; do sleep 120; done
say "학습 끝 — $(grep -c 'epoch .* avg loss' "$LOG") 줄의 epoch 손실 기록"

say "2/4 GPU 가 빌 때까지 대기 (모든 장 2 GB 미만)"
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 2000' | wc -l)" -eq 0 ]; do sleep 60; done

say "3/4 EK100 subset 시작 (vast_ek100_fill1029, 6 팔)"
GPUS=8 BS=12 bash cvpr/eval/ek100/run_group.sh vast_ek100_fill1029
say "EK100 끝 (exit $?)"

say "4/4 팔별 값"
/venv/vjepa/bin/python cvpr/eval/ek100/collect_group.py "$OUTD"
say "체인 끝"
