#!/bin/bash
# predictor_lab 평가 — 학습한 predictor 하나를 다섯 벤치로 채점한다 (시트 열) (encoder 는 릴리즈 ViT-H 고정, predictor 만 바꿔 끼운다).
#
#   bash predictor_lab/eval.sh <run 이름 | run 폴더 | predictor .pt>            # sbatch 잡 하나 (paths.env 의 LAB_NODE, GPU 1 장)
#   bash predictor_lab/eval.sh lab_bind vobench rubric                           # 벤치 골라서
#   GPUS=2 NODE=vll7 bash predictor_lab/eval.sh lab_bind
#   LOCAL=1 bash predictor_lab/eval.sh lab_bind                                  # 지금 셸 (GPU 있는 노드) 에서
#   DRYRUN=1 bash predictor_lab/eval.sh lab_bind                                 # 병합 config 와 경로 검사만 (GPU 0 장)
#
# 벤치 (기본 다섯 다, 시트 열 순서)
#   vobench        VOBench_v1 split v2 test 의 late 216 쌍 (primitive · realistic) — 표준 채점 c16t32 (보폭 3)        → VOBench P / R
#   gravity        VOBench gravity test 108 block — 고른 미래 (surprise 최소) = 진짜                                → VOBench Gravity P / R
#   intphys1       IntPhys 1 dev Garrido A.8 — 창 16 · 32 (각각 skip 2 · 5 · 10), Filtered, 180 쌍. 표는 격자 최고 칸 (descriptive)
#   intphys1_late  IntPhys 1 균일 창 (문맥 16 = 위반 전 균일 · 미래 16 = 위반 뒤 균일) — 표는 가림 쌍 90 개
#   rubric         Rubric Office 실제 영상 40 쌍 (violation 앞뒤 균일 16 + 16 장)
# 결과: <run>/eval_lab/<벤치 폴더>/{summary.json, per_block.json, _resolved.yaml}. 끝난 벤치는 건너뛴다 (FORCE=1 이면 다시).
# 표:   python predictor_lab/table.py <run> [<run> ...]
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/paths.env"
ARG=${1:?"사용법: bash predictor_lab/eval.sh <run|run 폴더|.pt> [vobench gravity rubric intphys1 intphys1_late]"}; shift
BENCHES=${*:-vobench gravity rubric intphys1 intphys1_late}
if [[ $ARG == *.pt ]]; then
  CK=$(cd "$(dirname "$ARG")" && pwd)/$(basename "$ARG"); E=$(dirname "$CK")/eval_lab_$(basename "${CK%.pt}"); NAME=$(basename "$(dirname "$CK")")_$(basename "${CK%.pt}")
else
  RUN=$ARG; [[ -d $RUN ]] || RUN=$LAB_REPO/z_training/runs/$ARG
  RUN=$(cd "$RUN" && pwd); CK=$RUN/${CKPT:-latest.pt}; E=$RUN/eval_lab; NAME=$(basename "$RUN")
fi
[[ -f $CK ]] || { echo "ERROR: predictor 체크포인트가 없다 -> $CK"; exit 1; }
echo "predictor $CK | 결과 $E | 벤치 $BENCHES"
if [[ -n ${LOCAL:-} || -n ${DRYRUN:-} ]]; then
  exec env CK="$CK" E="$E" NAME="$NAME" BENCHES="$BENCHES" GPUS=${GPUS:-1} bash "$HERE/eval_job.sh"
fi
mkdir -p "$E/logs"
sbatch -p "$LAB_PARTITION" -w "${NODE:-$LAB_NODE}" --gres=gpu:${GPUS:-1} -c $((8 * ${GPUS:-1})) --mem=$((48 * ${GPUS:-1}))G --time=${TIME:-0-04:00:00} \
  -J "lab_ev_$NAME" -o "$E/logs/%j.out" \
  --export=ALL,CK="$CK",E="$E",NAME="$NAME",BENCHES="$BENCHES",GPUS=${GPUS:-1} "$HERE/eval_job.sh"
echo "진행: tail -f $E/logs/<job>.out   |   다 끝나면: $LAB_PY predictor_lab/table.py $E"
