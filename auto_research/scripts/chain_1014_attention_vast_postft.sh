#!/bin/bash
# vast AR-state 를 VOBench (split_v2 train) 로 post-FT 한 것의 attention vs vast AR-state ep70 (2026-10-14, 사용자 요청).
# vast vob 판은 clip id (vid) · 같은 좌표 대조 (om_ql) 를 저장하려고 다시 돈다 (나머지 값은 같아야 한다).
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
PY=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
S=z_research/scripts/analysis/ar_attention_history.py
for s in vob_visible vob_late ssv2; do
  echo "[$(date +%T)] vobench_vast_ar_state_postft_ep30 $s"
  $PY $S --name vobench_vast_ar_state_postft_ep30 --ckpt z_training/runs/vobench_vast_ar_state_postft/latest.pt --set $s --gpus 4 --bs 2 --n 400 2>&1 | grep -v "Warning\|warn(" | grep "^==\|rel\|Error\|error"
done
for s in vob_visible vob_late; do
  echo "[$(date +%T)] vast_ar_state_ep70 $s"
  $PY $S --name vast_ar_state_ep70 --ckpt z_training/vast/ar_state/latest.pt --set $s --gpus 4 --bs 2 --n 400 2>&1 | grep -v "Warning\|warn(" | grep "^==\|rel\|Error\|error"
done
echo "VAST POSTFT ATTENTION DONE"
