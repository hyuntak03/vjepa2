#!/bin/bash
# AR predictor 의 rollout attention — 과거 문맥을 보나 (2026-10-13). 산출 z_research/VOBench_v1/analysis/attention_map/<model>/
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
PY=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
S=z_research/scripts/analysis/ar_attention_history.py
for m in "vast_ar_state_ep70 z_training/vast/ar_state/latest.pt" "ariel_ar_ep40 z_training/ariel/ar_future_1e_5/latest.pt" "ariel_ar_state_ep40 z_training/ariel/ar_state_future_1e_5/latest.pt"; do
  set -- $m
  for s in vob_visible vob_late ssv2; do
    echo "[$(date +%T)] $1 $s"
    $PY $S --name $1 --ckpt $2 --set $s --gpus 4 --n 400 2>&1 | grep -v "Warning\|warn(" | tail -40
  done
done
# 릴리즈 V-JEPA 2 predictor (2026-10-14 추가, --ckpt 없음 = 릴리즈, mask token 일괄 예측)
for s in vob_visible vob_late ssv2; do
  echo "[$(date +%T)] vjepa2 $s"
  $PY $S --name vjepa2 --set $s --gpus 4 --n 400 2>&1 | grep -v "Warning\|warn(" | tail -40
done
echo "ATTENTION DONE"
