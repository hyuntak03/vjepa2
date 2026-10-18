#!/bin/bash
# 릴리즈 post-FT (VOBench split_v2 train, 30 ep) 의 attention vs 릴리즈 (2026-10-14, 사용자 요청).
# 릴리즈 vob 판은 clip id (vid) 를 저장하려고 다시 돈다 (값은 같고 train / test 요약이 추가된다).
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
PY=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
S=z_research/scripts/analysis/ar_attention_history.py
for s in vob_visible vob_late ssv2; do
  echo "[$(date +%T)] vobench_release_postft_ep30 $s"
  $PY $S --name vobench_release_postft_ep30 --ckpt z_training/runs/vobench_release_postft/latest.pt --set $s --gpus 4 --n 400 2>&1 | grep -v "Warning\|warn(" | tail -60
done
for s in vob_visible vob_late; do
  echo "[$(date +%T)] vjepa2 $s"
  $PY $S --name vjepa2 --set $s --gpus 4 --n 400 2>&1 | grep -v "Warning\|warn(" | tail -60
done
echo "POSTFT ATTENTION DONE"
