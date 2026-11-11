#!/bin/bash
# v5id + 자연 영상 처음부터 학습 올리기 (2026-11-10 00:xx KST, 사용자 "scratch 부터 negative loss p = 0.2, 1M 데이터셋").
#   AR_full_tabl 채점 (chain_ar_full_fill_seq_20261109.sh) 이 끝나면 (AR_FULL_FILL_DONE) GPU 8 장으로 올린다.
#   처음 20 분 안에 OOM 이면 합성 batch 를 6 → 4 로 줄여 다시 올린다.
cd "$(dirname "$0")/.." || exit 1
ts() { TZ=Asia/Seoul date '+%F %T KST'; }
RUN=natural_ar_vw_state_v5id_negimp_mix
echo "대기: AR_full_tabl 채점 끝 $(ts)"
until grep -q AR_FULL_FILL_DONE z_training/logs/chain_ar_full_fill_20261109.log 2>/dev/null; do sleep 60; done
until ! pgrep -f "z_research/scripts/ru[n].sh" > /dev/null; do sleep 30; done
sleep 30
GPUS=8 bash z_training/tmux_train.sh $RUN && echo "TRAIN_LAUNCHED $RUN $(ts)"
for i in $(seq 1 20); do
  sleep 60
  if grep -qE "OutOfMemoryError|CUDA out of memory" z_training/runs/$RUN/train.log 2>/dev/null; then
    echo "OOM → 합성 batch 5 로 다시 $(ts)"; tmux kill-session -t train_$RUN 2>/dev/null; sleep 30
    mv z_training/runs/$RUN z_training/runs/_oom6_${RUN}_20261110
    GPUS=8 SET="negimp.mix_batch_size=4 data.var_window.token_budget=822" bash z_training/tmux_train.sh $RUN && echo "TRAIN_LAUNCHED (mix 4) $RUN $(ts)"
    break
  fi
done
echo "CHAIN_V5ID_MIX_DONE $(ts)"
