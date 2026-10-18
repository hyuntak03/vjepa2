#!/usr/bin/env bash
# =============================================================================
# eval/intphys1_late — IntPhys 1 dev 가림 샘플 (90 쌍), 위반 직전 (d − 1) 이 문맥 끝인 고정 창 c16t32 (skip 2). 채점 규칙은 testbed 와 같다.
#
#   GPU_IDS="0 1" bash cvpr/eval/intphys1_late/run.sh vith                 # 모델 (기본 vith)
#   GPU_IDS="0 1" bash cvpr/eval/intphys1_late/run.sh "vith vith_release_vobftg"
#   COPY=1 CUDA_VISIBLE_DEVICES=0 bash cvpr/eval/intphys1_late/run.sh vith   # 복사 기준선 (그 모델 결과 폴더에 copy_baseline.json)
#   DRYRUN=1 bash cvpr/eval/intphys1_late/run.sh vith
# 데이터를 다시 만들려면: python cvpr/eval/intphys1_late/build.py   (README)
# 결과: $CVPR_RESULTS/eval/intphys1_late/intphys1_late_<모델>/{summary.json, per_block.json, copy_baseline.json}
# =============================================================================
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); source "$HERE/../../env.sh"
MODELS=${1:-${MODELS:-vith}}
DS=${DS:-intphys1_late}          # DS=intphys1_late_hid → 문맥 끝 튜블릿에 물체가 안 보이는 판 (build.py --hidden-end, 2026-10-17)
fail=0
for m in $MODELS; do
  if [[ ${COPY:-0} == 1 ]]; then
    "$CVPR_PY" "$HERE/copy_baseline.py" --model-run "$CVPR_RESULTS/eval/intphys1_late/${DS}_$m" || fail=1; continue
  fi
  echo "### $DS / $m  ($(date '+%H:%M:%S'))"
  bash "$CVPR_ROOT/harness/launch.sh" "$HERE/protocol.yaml" "$DS" "$m" || { echo "!! 실패 $m"; fail=1; }
done
exit $fail
