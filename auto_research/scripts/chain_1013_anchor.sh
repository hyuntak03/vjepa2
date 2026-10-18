#!/bin/bash
# 2026-10-13 anchor 이식: 마지막 문맥 블록만 픽셀 조작 → 미래 자리가 역사를 따르나 마지막 내용을 따르나
set -uo pipefail; cd /data/hyuntak/project/2026/2027_cvpr/vjepa2; P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
for spec in "ar40 z_training/ariel/ar_future_1e_5/latest.pt" "release -" "ar_state40 z_training/ariel/ar_state_future_1e_5/latest.pt"; do
  set -- $spec; ck=$2; [ "$ck" = "-" ] && ck=""
  $P -u z_research/scripts/analysis/predictor_anchor_test.py --name $1 --ckpt "$ck" --gpus 4 --bs 8 > auto_research/logs/anchor_$1.log 2>&1
  echo "[$1 exit $?]"; grep -vE "warn|Warning|xFormers|\[r0\]|self.gen|predictor kind" auto_research/logs/anchor_$1.log | tail -12
done
echo ANCHOR DONE
