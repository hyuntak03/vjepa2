#!/bin/bash
# predictor_lab 학습 — BlenderTrack 으로 predictor 를 학습한다 (기본: 릴리즈 ViT-H predictor post-FT, encoder 고정).
#
#   bash predictor_lab/train.sh <config>                 # predictor_lab/configs/<config>.yaml (없으면 configs/training/)
#   NAME=myidea_s1 SET="meta.seed=1" bash predictor_lab/train.sh lab_bind
#   GPUS=2 NODE=vll7 bash predictor_lab/train.sh lab_base   # 노드 · GPU 수 (기본 paths.env 의 LAB_NODE, GPU 2)
#   LOCAL=1 GPUS=1 LIMIT=64 SET="optimization.epochs=1" bash predictor_lab/train.sh lab_bind_occ   # 지금 셸 (GPU 있는 노드) 에서 스모크
#   DRYRUN=1 bash predictor_lab/train.sh lab_base        # 병합 config 만 출력 (GPU 0 장, 몇 초)
#
# 학습 데이터 경로는 paths.env 의 BT_TRAIN_META · BT_CLIPS 를 data.datasets 로 넣는다 (config 에는 경로를 안 적는다).
# 파이썬은 paths.env 의 LAB_PY (vll7 Blackwell 은 vjepa2_bw). 본체는 train_job.sh.
# 결과: z_training/runs/<NAME>/{latest.pt, config.yaml, train.log, metrics.jsonl}. 평가: bash predictor_lab/eval.sh <NAME>
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/paths.env"
CFG=${1:?"사용법: bash predictor_lab/train.sh <config>   (목록: ls predictor_lab/configs)"}
DS="data.datasets=[{\"type\":\"blender_track\",\"meta\":\"$BT_TRAIN_META\",\"root\":\"$BT_CLIPS\"}]"
NAME_GIVEN=${NAME:-}
NAME=${NAME:-$(basename "${CFG%.yaml}")}
[[ -n ${LIMIT:-} && -z $NAME_GIVEN ]] && NAME="${NAME}_smoke${LIMIT}"   # 스모크가 본 run 을 안 건드리게
G=${GPUS:-2}
SETALL="$DS ${SET:-}"
if [[ -n ${LOCAL:-} || -n ${DRYRUN:-} ]]; then
  exec env CONFIG="$CFG" NAME="$NAME" GPUS="$G" LIMIT="${LIMIT:-}" SET_B64="$(printf %s "$SETALL" | base64 -w0)" bash "$HERE/train_job.sh"
fi
mkdir -p "$LAB_REPO/z_training/slurm_logs"
sbatch -p "$LAB_PARTITION" -w "${NODE:-$LAB_NODE}" --gres=gpu:"$G" -c $(( ${CPUS_PER_GPU:-14} * G )) --mem=$(( ${MEM_GB_PER_GPU:-60} * G ))G \
  --time="${TIME:-0-08:00:00}" -J "lab_tr_$NAME" -o "$LAB_REPO/z_training/slurm_logs/lab_tr_${NAME}_%j.out" \
  --export=ALL,CONFIG="$CFG",NAME="$NAME",GPUS="$G",LIMIT="${LIMIT:-}",SET_B64="$(printf %s "$SETALL" | base64 -w0)" "$HERE/train_job.sh"
echo "진행: tail -f $LAB_REPO/z_training/runs/$NAME/train.log   |   끝나면: bash predictor_lab/eval.sh $NAME"
