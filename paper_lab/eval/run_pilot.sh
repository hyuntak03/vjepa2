#!/bin/bash
# paper_lab pass-line for pilot arms (진행자 1, 2026-10-21). Usage (one GPU via CUDA_VISIBLE_DEVICES):
#   [PASSLINE_FAST=1] bash paper_lab/eval/run_pilot.sh <arm> [ckpt names ...]   e.g.  PASSLINE_FAST=1 bash paper_lab/eval/run_pilot.sh A0 e5 e10
#   PASSLINE_FAST=1 = VOBench late + visible-moving only (primary + carry cells; ~35 min per checkpoint instead of ~60)
# Each checkpoint: paper_lab/_scratch/runs/pl_pilot_<arm>/<ck>.pt (pl_method.json next to it -> eval-time components only).
set -uo pipefail
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
E=paper_lab/eval/passline.py
ARM=$1; shift
for ck in "$@"; do
  ckpt=paper_lab/_scratch/runs/pl_pilot_${ARM}/${ck}.pt
  name=pilot${ARM}_${ck}
  [ -f "$ckpt" ] || { echo "missing $ckpt"; continue; }
  echo "[$(date '+%F %T')] extract $name"
  $P -u $E extract --ckpt "$ckpt" --name "$name" --gpus 1 --bs 8 --sets vobench,gravity,natural,ip1late,natcarry --n-natural 200 --fps 4 ${PASSLINE_FAST:+--fast} || { echo "EXTRACT $name FAILED"; continue; }
  $P -u $E analyze --name "$name" --split all
  $P -u $E analyze --name "$name" --split test
  echo "[$(date '+%F %T')] done $name"
done
echo "[$(date '+%F %T')] PILOT $ARM DONE"
