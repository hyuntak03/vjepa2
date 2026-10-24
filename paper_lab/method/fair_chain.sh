#!/bin/bash
# run_fair.sh 가 tmux 안에서 부른다. <팔> <cuda 번호> <마감 epoch 초>
set -uo pipefail
ARM=$1; DEV=$2; DEADLINE=$3
REPO=/data/hyuntak/project/2026/2027_cvpr/vjepa2; PY=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
RUN=$REPO/paper_lab/_scratch/runs/pl_fair_$ARM
cd $REPO
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1 TRAIN_DDP_PORT=$((37300 + DEV * 7 + RANDOM % 50)) PL_WSTAT_FILE=$RUN/weight_stats.jsonl
export PL_METHOD="$(cat $RUN/pl_method_in.json)"
REM=$(( DEADLINE - $(date +%s) ))
echo "[$(date '+%F %T')] train start, ${REM}s until deadline" > $RUN/chain.log
timeout --signal=TERM --kill-after=120 $REM $PY paper_lab/method/launch.py --fname $RUN/config.yaml --devices cuda:$DEV > $RUN/train.log 2>&1
RC=$?
echo "EXIT $RC" >> $RUN/train.log
echo "[$(date '+%F %T')] train end rc=$RC (124 = 마감으로 멈춤)" >> $RUN/chain.log
[ -f $RUN/pids.txt ] && kill $(cat $RUN/pids.txt) 2>/dev/null
sleep 20
# 평가: 가장 늦은 체크포인트 먼저, 그다음 나머지 (학습 곡선). 하네스 빠른 판 = run_pilot.sh 와 같은 인자.
CKS=(); for e in 8 6 4 2; do [ -f $RUN/e$e.pt ] && CKS+=(e$e); done
ORDER=("${CKS[@]:0:1}"); for c in $(printf '%s\n' "${CKS[@]:1}" | sort -V); do ORDER+=($c); done
echo "[$(date '+%F %T')] eval order: ${ORDER[*]}" >> $RUN/chain.log
for ck in "${ORDER[@]}"; do
  name=fair${ARM}_$ck
  echo "[$(date '+%F %T')] extract $name" >> $RUN/chain.log
  CUDA_VISIBLE_DEVICES=$DEV $PY -u paper_lab/eval/passline.py extract --ckpt $RUN/$ck.pt --name $name --gpus 1 --bs 8 \
      --sets vobench,gravity,natural,ip1late,natcarry --n-natural 200 --fps 4 --fast >> $RUN/eval.log 2>&1 || { echo "EXTRACT $name FAILED" >> $RUN/chain.log; continue; }
  $PY -u paper_lab/eval/passline.py analyze --name $name --split all >> $RUN/eval.log 2>&1
  $PY -u paper_lab/eval/passline.py analyze --name $name --split test >> $RUN/eval.log 2>&1
  echo "[$(date '+%F %T')] done $name" >> $RUN/chain.log
done
echo "[$(date '+%F %T')] FAIR $ARM DONE" >> $RUN/chain.log
