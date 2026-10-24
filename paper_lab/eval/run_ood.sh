#!/bin/bash
# paper_lab pass-line, out-of-domain sets (진행자 1, 2026-10-21): IntPhys 1 late occlusion + natural-video carry (flow pseudo labels)
# + false-carry maps (vobd = VOBench late + visible-moving, D maps only) for the two baseline rows. Run AFTER run_baselines.sh (same GPU). Detach with tmux:
#   tmux new -d -s pl_passline_ood "CUDA_VISIBLE_DEVICES=0 bash paper_lab/eval/run_ood.sh 2>&1 | tee paper_lab/eval/results/run_ood.log"
set -uo pipefail
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
E=paper_lab/eval/passline.py
echo "[$(date '+%F %T')] start · CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
for spec in "vast70 z_training/vast/ar_state/latest.pt" "ariel_ar40 z_training/ariel/ar_future_1e_5/latest.pt"; do
  set -- $spec
  echo "[$(date '+%F %T')] ood extract $1"
  # natcarry re-runs the natural loop (same 200 K400 val clips, same seed) and adds the flow-tracked carry readout
  $P -u $E extract --ckpt "$2" --name "$1" --gpus 1 --bs 8 --sets ip1late,natcarry,vobd --n-natural 200 --fps 4 || { echo "OOD EXTRACT $1 FAILED"; continue; }
  $P -u $E analyze --name "$1" --split all
  echo "[$(date '+%F %T')] ood done $1"
done
$P -u $E table --names vast70 ariel_ar40 --split all > paper_lab/eval/results/passline_baselines_all.md
echo "[$(date '+%F %T')] OOD ALL DONE"
