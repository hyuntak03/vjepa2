#!/usr/bin/env bash
# =============================================================================
# eval/testbed — 자체 testbed surprise 채점 (surprise = mean |p − LN(h)[future]|, matched pair, chance 50 %)
#
#   GPUS=8 bash cvpr/eval/testbed/run.sh v11 vith                      # 데이터셋 · 모델
#   GPUS=8 bash cvpr/eval/testbed/run.sh "v11 v11_realistic" "vith vitl"   # 행렬 (공백 구분)
#   WINDOW=c8t32 GPUS=8 bash cvpr/eval/testbed/run.sh v11 vith         # 창 preset 바꾸기 (cvpr/registry/windows.yaml)
#   SET="window.context=12" GPUS=8 bash cvpr/eval/testbed/run.sh v11   # 임의 창 (tag 에 _wc12t32 가 붙는다)
#   SET="scoring.pairing=cross" ...                                    # 채점 옵션 (gravity_realistic 은 레지스트리가 알아서 cross)
#   DRYRUN=1 bash cvpr/eval/testbed/run.sh v11                         # 병합 config 만 (GPU 0 장)
#   LIMIT=2 GPUS=1 bash cvpr/eval/testbed/run.sh v11                   # 스모크 (tag/결과에 _smoke2)
#   GPUS=8 bash cvpr/eval/testbed/run.sh v11 dinof_highres              # DINO-Foresight (문맥 16 중 마지막 4 장으로 unroll)
#   COPY=1 GPUS=8 bash cvpr/eval/testbed/run.sh v11 dinof_highres       # 복사 기준선 (_copy)
#   RESULTS_ROOT=legacy ...                                            # 옛 폴더 (z_research/<셋>/exp_results/surprise_c16t32__<ds>_<model>) 로
#
# 등록된 데이터셋 · 모델 · 창:  python cvpr/harness/resolve.py --list
# 결과: $CVPR_RESULTS/eval/testbed/<ds>_<model>[_w<창>]/{summary.json, per_block.json, _resolved.yaml}
# 복사 기준선 (마지막 문맥 latent) 은 같은 폴더의 copy_baseline.sh (있으면) — README 참고
# =============================================================================
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); source "$HERE/../../env.sh"
DATASETS=${1:-${DATASETS:-v11}}
MODELS=${2:-${MODELS:-vith}}
copy_set() { case $1 in dinof_*) echo "model.dinof_copy=true" ;; *) echo "" ;; esac; }   # COPY=1: 복사 기준선 (dinof 로더 플래그). V-JEPA 계열은 README 의 copy 절차
fail=0
for d in $DATASETS; do for m in $MODELS; do
  S="${SET:-}"; SFX="${SUFFIX:-}"
  if [[ ${COPY:-0} == 1 ]]; then cs=$(copy_set "$m"); [[ -z $cs ]] && { bash "$HERE/copy_baseline.sh" "$d" "$m" || { echo "!! 실패 copy $d/$m"; fail=1; }; continue; }; S="$S $cs"; SFX="${SFX}_copy"; fi   # V-JEPA 계열 → copy_baseline.sh (te_v11_score.py)
  echo "### testbed  $d / $m$SFX  ($(date '+%H:%M:%S'))"
  SET="$S" SUFFIX="$SFX" bash "$CVPR_ROOT/harness/launch.sh" "$HERE/protocol.yaml" "$d" "$m" || { echo "!! 실패 $d/$m"; fail=1; }
done; done
exit $fail
