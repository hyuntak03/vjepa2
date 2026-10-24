#!/bin/bash
# paper_lab 파일럿 팔 하나 띄우기 (진행자 2, 2026-10-21). 같은 config (pl_pilot) + 팔 설정 PL_METHOD 만 다르다.
#   [CONFIG=pl_pilot] [SEED=0] bash paper_lab/method/run_arm.sh <팔 이름> <cuda 번호 (이 셸에서 보이는 번호)> '<PL_METHOD JSON>'
#   확대: CONFIG=pl_scale SEED=1 bash paper_lab/method/run_arm.sh A1s1 1 '<...>'   (run 폴더 = pl_<config 뒤>_<팔>)
#   예: bash paper_lab/method/run_arm.sh AX 1 '{"occ": {"p": 0.5, "area": [0.15, 0.4]}}'
# tmux 세션 pl_pilot<팔> 로 띄운다. 띄우기 전에 그 GPU 메모리가 1 GB 넘게 차 있으면 멈춘다.
set -euo pipefail
ARM=$1; DEV=$2; METHOD=${3:-'{}'}
REPO=/data/hyuntak/project/2026/2027_cvpr/vjepa2; PY=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
CONFIG=${CONFIG:-pl_pilot}; SEED=${SEED:-0}
RUN=$REPO/paper_lab/_scratch/runs/${CONFIG}_$ARM
cd $REPO
USED=$(CUDA_DEVICE_ORDER=PCI_BUS_ID nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $DEV)
if [ "$USED" -gt 1000 ]; then echo "GPU $DEV 가 비어 있지 않다 (${USED} MiB) — 멈춘다"; exit 3; fi
mkdir -p $RUN
TRAIN_CFG_DIR=paper_lab/method/configs $PY z_training/harness/resolve_train.py $CONFIG -o $RUN/config.yaml --folder $RUN --set meta.seed=$SEED --quiet
PORT=$((37300 + DEV * 7 + RANDOM % 50))
printf '%s' "$METHOD" > $RUN/pl_method_in.json
$PY -c "import json,sys; json.load(open('$RUN/pl_method_in.json'))" || { echo "PL_METHOD JSON 이 깨졌다"; exit 4; }
EVAL_CMD=""
if [ "${EVAL_AFTER:-0}" = 1 ]; then   # 학습이 끝나면 같은 GPU 에서 하네스 e10 빠른 판 (진행자 1 의 run_pilot.sh)
  EVAL_CMD=" && grep -q '^EXIT 0' $RUN/train.log && CUDA_VISIBLE_DEVICES=$DEV PASSLINE_FAST=1 bash paper_lab/eval/run_pilot.sh $ARM e10 > paper_lab/eval/results/run_pilot_$ARM.log 2>&1"
fi
tmux new-session -d -s pl_${CONFIG#pl_}$ARM "cd $REPO && export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1 TRAIN_DDP_PORT=$PORT PL_WSTAT_FILE=$RUN/weight_stats.jsonl && export PL_METHOD=\"\$(cat $RUN/pl_method_in.json)\" && $PY paper_lab/method/launch.py --fname $RUN/config.yaml --devices cuda:$DEV > $RUN/train.log 2>&1; echo EXIT \$? >> $RUN/train.log$EVAL_CMD"
echo "launched pl_${CONFIG#pl_}$ARM on cuda:$DEV  seed=$SEED  method=$METHOD  run=$RUN"
