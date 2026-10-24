#!/usr/bin/env bash
# =============================================================================
# EK100 action anticipation — cap16metric · predictor_group (vast 2026-10-21, main z_vll7_training/eval_ek100.sh 이식)
#
#   GPUS=6 bash cvpr/eval/ek100/run_group.sh vast_ek100_g5           # 레지스트리의 predictor_group 모델 하나 = 팔 여러 개를 한 번에
#   DRYRUN=1 bash cvpr/eval/ek100/run_group.sh vast_ek100_g5         # 병합 config 만
#   LIMIT=8 GPUS=1 bash cvpr/eval/ek100/run_group.sh vast_ek100_g5   # 배관 스모크 (영상 8 편 · 1 epoch 은 SET 으로)
#
# 규약 (시트 "EK100 subset" = main GROUP4_CAP16METRIC_2026-10-18.md 와 같게 둔다 — 바꾸면 비교가 깨진다):
#   * train / val = data_csv/ek100/EPIC_100_{train,validation}_cap16metric_seed0.csv (z_research/scripts/data/build_ek100_cap16metric.py)
#     ⚠️ 원래 CSV 는 git 에 없어 정의로 다시 만들었다 — 행 수 · 편 수 · 클래스별 개수는 같고, 16 개 넘는 클래스에서 고른 구간은 다를 수 있다
#   * HEADS=cap16b96 (lr 3e-4 · 6e-4 w1 · 1.2e-3 w2, wd 1e-4) · rank 16 × 6 GPU = global 96 · 20 epoch · val/ckpt 5 epoch 마다
#   * anticipation_point_mode=released · encoder checkpointing 끔 (classifier.use_activation_checkpointing false)
#   * 모듈 = modelcustom/vit_encoder_predictor_concat_ar_group (main nonsquare + vast kind=ar 빌드)
#   * 영상 = ${CVPR_EK100} (256x256 재인코딩, main 은 EPIC-KITCHENS_resized — 기하가 같게 만든 판)
# 결과: $CVPR_RESULTS/eval/ek100/action_anticipation_frozen/ek100_<모델>/val_metrics__<팔>.jsonl
#   추출: /venv/vjepa/bin/python cvpr/eval/ek100/collect_group.py <결과 폴더>
# =============================================================================
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); source "$HERE/../../env.sh"
MODEL=${1:?"사용법: GPUS=6 bash cvpr/eval/ek100/run_group.sh <predictor_group 모델>"}
export GPUS=${GPUS:-6}
BS=${BS:-16}; NW=${NW:-6}; VAL_EVERY=${VAL_EVERY:-5}; CKPT_EVERY=${CKPT_EVERY:-5}
TRAIN_CSV=${TRAIN_CSV:-$CVPR_REPO/data_csv/ek100/EPIC_100_train_cap16metric_seed0.csv}
VAL_CSV=${VAL_CSV:-$CVPR_REPO/data_csv/ek100/EPIC_100_validation_cap16metric_seed0.csv}
[[ -f $TRAIN_CSV && -f $VAL_CSV ]] || { echo "CSV 없음 — /venv/vjepa/bin/python z_research/scripts/data/build_ek100_cap16metric.py"; exit 1; }
# 호스트 RAM 가드 (main 2026-10-18 실측 150 MB/clip). 이 기계는 cgroup 캡이 없으니 MemAvailable 의 70 % 로 본다
CLIPS=$(( BS * NW * 2 * GPUS )); EST_GB=$(( CLIPS * 150 / 1024 )); AVAIL_GB=$(( $(awk '/MemAvailable/{print $2}' /proc/meminfo) / 1024 / 1024 ))
echo "  메모리 : in-flight clip ${CLIPS} -> 추정 ${EST_GB} GB / 사용 가능 ${AVAIL_GB} GB"
if [[ "${SKIP_MEM_GUARD:-0}" != 1 ]] && (( EST_GB * 100 > AVAIL_GB * 70 )); then echo "ERROR: 호스트 RAM 부족 위험 (SKIP_MEM_GUARD=1 로 무시)"; exit 1; fi
S="model_kwargs.module_name=evals.action_anticipation_frozen.modelcustom.vit_encoder_predictor_concat_ar_group"
S="$S experiment.classifier.use_activation_checkpointing=false experiment.data.anticipation_point_mode=released"
S="$S experiment.optimization.batch_size=$BS experiment.data.num_workers=$NW experiment.evaluation.val_every=$VAL_EVERY experiment.evaluation.ckpt_every=$CKPT_EVERY"
S="$S experiment.data.dataset_train=$TRAIN_CSV experiment.data.dataset_val=$VAL_CSV"
echo "  모델 $MODEL · HEADS ${HEADS:-cap16b96} · batch $BS x $GPUS = $((BS*GPUS)) · workers $NW · CSV $(basename "$TRAIN_CSV") / $(basename "$VAL_CSV")"
SET="${SET:-} $S" HEADS=${HEADS:-cap16b96} exec bash "$HERE/run.sh" ek100 "$MODEL"
