#!/bin/bash
# 진행자 1, 2026-10-22: wait for a pilot arm's harness eval (run_pilot.sh writes "PILOT <arm> DONE"), then write the paired
# comparisons (CPU only, no GPU): <arm> vs A0 e10 and vs AX e10, cells late/moving (main) and late/static (B1).
# Usage: bash paper_lab/eval/run_compare_after.sh <arm> [<arm> ...]
set -uo pipefail
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
E=paper_lab/eval/passline.py
for arm in "$@"; do
  until grep -q "PILOT ${arm} DONE" paper_lab/eval/results/run_pilot_${arm}.log 2>/dev/null; do sleep 60; done
  for base in pilotA0_e10 pilotAX_e10; do
    for cell in late/moving late/static; do
      $P $E compare --names $base pilot${arm}_e10 --split all --cell $cell > /dev/null && echo "[$(date +%T)] compare $base vs pilot${arm}_e10 $cell"
    done
  done
done
echo "[$(date +%T)] COMPARE DONE $*"
