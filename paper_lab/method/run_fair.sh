#!/bin/bash
# paper_lab 공정한 시험 한 run (진행자 2, 2026-10-22, PLAN §6 · D9 C 안). run_arm.sh 와 같은 준비 + 벽시계 마감 + 끝나면 같은 GPU 에서 평가.
#   DEADLINE=<epoch 초> SEED=<0|1> bash paper_lab/method/run_fair.sh <팔 이름> <cuda 번호> '<PL_METHOD JSON>'
#   tmux 세션 pl_fair_<팔> 안에서 paper_lab/method/fair_chain.sh 가 돈다: 마감까지 학습 → 남은 프로세스 정리 →
#   가장 늦은 체크포인트부터 하네스 빠른 판 평가 (그다음 나머지 체크포인트 = 학습 곡선).
set -euo pipefail
ARM=$1; DEV=$2; METHOD=${3:-'{}'}
REPO=/data/hyuntak/project/2026/2027_cvpr/vjepa2; PY=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
CONFIG=pl_fair; SEED=${SEED:-0}; : "${DEADLINE:?DEADLINE (epoch 초) 가 필요하다}"
RUN=$REPO/paper_lab/_scratch/runs/${CONFIG}_$ARM
cd $REPO
USED=$(CUDA_DEVICE_ORDER=PCI_BUS_ID nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $DEV)
if [ "$USED" -gt 1000 ]; then echo "GPU $DEV 가 비어 있지 않다 (${USED} MiB) — 멈춘다"; exit 3; fi
[ -e "$RUN" ] && { echo "$RUN 이 이미 있다 — 멈춘다 (덮어쓰지 않음)"; exit 5; }
mkdir -p $RUN
TRAIN_CFG_DIR=paper_lab/method/configs $PY z_training/harness/resolve_train.py $CONFIG -o $RUN/config.yaml --folder $RUN --set meta.seed=$SEED --quiet
printf '%s' "$METHOD" > $RUN/pl_method_in.json
$PY -c "import json,sys; json.load(open('$RUN/pl_method_in.json'))" || { echo "PL_METHOD JSON 이 깨졌다"; exit 4; }
echo "$DEADLINE" > $RUN/deadline_epoch
tmux new-session -d -s pl_fair_$ARM "bash $REPO/paper_lab/method/fair_chain.sh $ARM $DEV $DEADLINE"
echo "launched pl_fair_$ARM on cuda:$DEV  seed=$SEED  method=$METHOD  deadline=$(date -d @$DEADLINE '+%F %T')  run=$RUN"
