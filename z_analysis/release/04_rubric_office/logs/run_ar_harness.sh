#!/bin/bash
# Rubric Office · vast AR (ar_state) 표준 하네스 채점 — 창 하나 ($W) 에 vast70 · vast70g · AR 공간 복사 (copy_last)
set -u
R=/data/hyuntak/project/2026/2027_cvpr/vjepa2; cd $R
B=$R/z_analysis/release/04_rubric_office
S="data.root=$R/data_csv/rubric_quads_$W data.index_csv=index.csv data.frames_root=/data/hyuntak/project/2026/2027_cvpr/nfs_cache/rubric_quads_frames/$W window.stride=1 window.frames_start=0"
for spec in "vast70 vith_vast_ar_state_ep70 " "vast70g vith_vast_ar_state_ep70_vobftg " "vast70_copyar vith_vast_ar_state_ep70 surprise.ar_baseline=copy_last"; do
  set -- $spec; tag=$1; model=$2; extra=${3:-}
  echo "### $W $tag ($(date +%T))"
  GPUS=1 TAG=rubric_${W}_$tag OUTDIR=$B/harness/${W}_$tag SET="$S $extra" bash cvpr/eval/testbed/run.sh vobench_v1 $model
done
