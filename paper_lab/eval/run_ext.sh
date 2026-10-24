#!/bin/bash
# paper_lab pass-line for an EXTERNAL block-AR checkpoint (a user run), 진행자 1, 2026-10-22.
# The checkpoint is copied read-only beforehand into paper_lab/_scratch/runs/ext_<run>/<ck>.pt (cp -p + cmp), so the
# training run folder is never read during the eval nor written to.  Results go to paper_lab/eval/results/ext_<run>_<ck>_c<n>/.
# Usage (one GPU via CUDA_VISIBLE_DEVICES):
#   [PASSLINE_FAST=1] bash paper_lab/eval/run_ext.sh <run> <ck> <ctx_in> [<ctx_in> ...]
#   e.g.  PASSLINE_FAST=1 bash paper_lab/eval/run_ext.sh ar_vw_shortctx e30 8 2
# <ctx_in> = number of context blocks fed to the predictor (PASSLINE_CTX_IN; 8 = standard, same as every other row).
set -uo pipefail
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
E=paper_lab/eval/passline.py
RUN=$1; CK=$2; shift 2
ckpt=paper_lab/_scratch/runs/ext_${RUN}/${CK}.pt
[ -f "$ckpt" ] || { echo "missing $ckpt"; exit 1; }
for ci in "$@"; do
  name=ext_${RUN}_${CK}_c${ci}
  echo "[$(date '+%F %T')] extract $name (ctx_in $ci)"
  PASSLINE_CTX_IN=$ci $P -u $E extract --ckpt "$ckpt" --name "$name" --gpus 1 --bs 8 --sets vobench,gravity,natural,ip1late,natcarry --n-natural 200 --fps 4 ${PASSLINE_FAST:+--fast} || { echo "EXTRACT $name FAILED"; continue; }
  $P -u $E analyze --name "$name" --split all
  $P -u $E analyze --name "$name" --split test
  echo "[$(date '+%F %T')] done $name"
done
echo "[$(date '+%F %T')] EXT $RUN $CK DONE"
