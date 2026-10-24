#!/bin/bash
# paper_lab pass-line baselines (진행자 1, 2026-10-21): smoke -> baseline block AR (vast ep70) -> Ariel block AR (output feedback, ep40).
# Runs on ONE GPU given by CUDA_VISIBLE_DEVICES (set by the caller). Detach with tmux:
#   tmux new -d -s pl_passline "CUDA_VISIBLE_DEVICES=0 bash paper_lab/eval/run_baselines.sh 2>&1 | tee paper_lab/eval/results/run_baselines.log"
set -uo pipefail
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
E=paper_lab/eval/passline.py
mkdir -p paper_lab/eval/results
echo "[$(date '+%F %T')] start · CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# 1) smoke (16 clips per synthetic set, 4 natural clips) — must finish before the full runs
$P -u $E extract --ckpt z_training/vast/ar_state/latest.pt --name _smoke --gpus 1 --limit 16 --n-natural 4 --bs 8 || { echo "SMOKE EXTRACT FAILED"; exit 1; }
$P -u $E analyze --name _smoke --split all || { echo "SMOKE ANALYZE FAILED"; exit 1; }
echo "[$(date '+%F %T')] smoke ok"

# 2) full runs
for spec in "vast70 z_training/vast/ar_state/latest.pt" "ariel_ar40 z_training/ariel/ar_future_1e_5/latest.pt"; do
  set -- $spec
  echo "[$(date '+%F %T')] extract $1"
  $P -u $E extract --ckpt "$2" --name "$1" --gpus 1 --bs 8 --n-natural 200 --fps 4 || { echo "EXTRACT $1 FAILED"; continue; }
  $P -u $E analyze --name "$1" --split all
  $P -u $E analyze --name "$1" --split test
  echo "[$(date '+%F %T')] done $1"
done
$P -u $E table --names vast70 ariel_ar40 --split all > paper_lab/eval/results/passline_baselines_all.md
$P -u $E table --names vast70 ariel_ar40 --split test > paper_lab/eval/results/passline_baselines_test.md
echo "[$(date '+%F %T')] ALL DONE"
