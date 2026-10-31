#!/bin/bash
# CLEVRER 영속성 v1 · 표준 하네스 채점 (레지스트리 모델 하나, 환경변수 M; COPYAR=1 이면 AR 공간 복사 기준선)
set -u
R=/data/hyuntak/project/2026/2027_cvpr/vjepa2; cd $R
D=/data/hyuntak/project/2026/2027_cvpr/testbed/cleverer_permanence_v1
S="data.root=$D data.index_csv=$R/data_csv/clevrer_perm_v1/index.csv data.frames_root=null window.stride=1 window.frames_start=0"
TAGN=$M; [[ -n ${COPYAR:-} ]] && { S="$S surprise.ar_baseline=copy_last"; TAGN=${M}_copyar; }
GPUS=1 TAG=clevrer_$TAGN OUTDIR=$R/z_analysis/release/07_clevrer_perm/harness/$TAGN SET="$S" bash cvpr/eval/testbed/run.sh vobench_v1 $M
