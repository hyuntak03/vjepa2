#!/bin/bash
# 2026-10-13 역사 가치: 문맥 8 vs 4 · 2 · 1 블록 손실 + 데이터의 역사 필요 몫 (anchor 다음)
set -uo pipefail; cd /data/hyuntak/project/2026/2027_cvpr/vjepa2; P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
for spec in "ar40 z_training/ariel/ar_future_1e_5/latest.pt ssv2" "ar40 z_training/ariel/ar_future_1e_5/latest.pt vob_late" "ar40 z_training/ariel/ar_future_1e_5/latest.pt vob_visible" \
            "ar_state40 z_training/ariel/ar_state_future_1e_5/latest.pt ssv2"; do
  set -- $spec; ck=$2; [ "$ck" = "-" ] && ck=""
  $P -u z_research/scripts/analysis/predictor_history_value.py --name $1 --ckpt "$ck" --set $3 --gpus 4 --n 800 > auto_research/logs/history_$1_$3.log 2>&1
  echo "[$1 $3 exit $?]"; grep -vE "warn|Warning|xFormers|\[r0\]|self.gen|predictor kind" auto_research/logs/history_$1_$3.log | tail -16
done
echo HISTORY DONE
