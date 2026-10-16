#!/usr/bin/env bash
# =============================================================================
# eval/ek100 — EK100 action anticipation (frozen encoder + predictor + attentive probe). 엔진 evals/action_anticipation_frozen.
#
#   GPUS=8 bash cvpr/eval/ek100/run.sh                         # ek100 x vith (논문 설정, head 1 개)
#   GPUS=8 HEADS=sweep bash cvpr/eval/ek100/run.sh ek100 vith  # 논문의 probe head 20 개
#   GPUS=8 bash cvpr/eval/ek100/run.sh ek100 vith_ariel_prefix_ep45    # 다른 predictor (모델 레지스트리)
#   GPUS=1 SMOKE=1 bash cvpr/eval/ek100/run.sh                 # 배관 점검 (비디오 4 개, 1 epoch)
#   VAL_ONLY=1 ...                                             # val 만
#   DRYRUN=1 bash cvpr/eval/ek100/run.sh
#
# ⚠️ 영상은 노드 로컬 ($CVPR_DATA2/local_datasets/EPIC-KITCHENS_resized) — 다른 노드면 resolve 가 죽는다.
# ⚠️ 같은 결과 폴더에 다른 설정으로 이어 돌리지 말 것 (resume_checkpoint: true). 설정을 바꾸면 TAG 도 바꾼다.
# 결과: $CVPR_RESULTS/eval/ek100/action_anticipation_frozen/<ds>_<model>[_w<창>]/
# =============================================================================
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); source "$HERE/../../env.sh"
DATASETS=${1:-${DATASETS:-ek100}}
MODELS=${2:-${MODELS:-vith}}
S="${SET:-}"; [[ -n ${HEADS:-} ]] && S="$S heads=$HEADS"
fail=0
for d in $DATASETS; do for m in $MODELS; do
  echo "### ek100  $d / $m  ($(date '+%H:%M:%S'))"
  SET="$S" bash "$CVPR_ROOT/harness/launch.sh" "$HERE/protocol.yaml" "$d" "$m" || { echo "!! 실패 $d/$m"; fail=1; }
done; done
exit $fail
