#!/bin/bash
# paper_lab pass-line in the METHOD'S OWN TRAINING SPACE (PASSLINE_TSPACE=train; targets and every baseline built from the
# pl_method.json target view), 진행자 1, 2026-10-22, 진행자 2 요청.  Results: paper_lab/eval/results/pilot<arm>_<ck>_tsp/.
# Usage (one GPU via CUDA_VISIBLE_DEVICES):  [PASSLINE_FAST=1] bash paper_lab/eval/run_tspace.sh <arm>:<ck> [<arm>:<ck> ...]
set -uo pipefail
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
E=paper_lab/eval/passline.py
for job in "$@"; do
  arm=${job%%:*}; ck=${job##*:}
  ckpt=paper_lab/_scratch/runs/pl_pilot_${arm}/${ck}.pt
  name=pilot${arm}_${ck}_tsp
  [ -f "$ckpt" ] || { echo "missing $ckpt"; continue; }
  echo "[$(date '+%F %T')] extract $name (training space)"
  PASSLINE_TSPACE=train $P -u $E extract --ckpt "$ckpt" --name "$name" --gpus 1 --bs 8 --sets vobench,gravity,natural,ip1late,natcarry --n-natural 200 --fps 4 ${PASSLINE_FAST:+--fast} || { echo "EXTRACT $name FAILED"; continue; }
  $P -u $E analyze --name "$name" --split all
  $P -u $E analyze --name "$name" --split test
  echo "[$(date '+%F %T')] done $name"
done
echo "[$(date '+%F %T')] TSPACE DONE"
