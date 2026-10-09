#!/usr/bin/env bash
# =============================================================================
# analysis/readout — 학습된 자 (readout, "자") 분석의 단일 진입점. 과학은 z_research/scripts/ 의 기존 스크립트 그대로.
#
#   bash cvpr/analysis/readout/run.sh --list                              # 스텝 · 노브 목록
#   STEP=fit_position DRYRUN=1 bash cvpr/analysis/readout/run.sh          # 명령 · 환경변수 · 입력 실물만 (GPU 0 장)
#   STEP=attn_position GPUS=2 bash cvpr/analysis/readout/run.sh           # 실행 (GPU 는 스크립트 자신이 쓴다)
#   STEP="fit_position test_position" bash cvpr/analysis/readout/run.sh   # 여러 스텝을 차례로 (실패하면 멈춘다)
#   STEP=cache_ctx32 GPUS=8 bash cvpr/analysis/readout/run.sh rollout_v2  # [DATASET] [MODEL] 인자 (launch 스텝 · lock 아닌 스텝만)
#   bash cvpr/analysis/readout/run.sh window_readout                      # STEP 없이 첫 인자를 스텝으로 (그 뒤 [DATASET] [MODEL])
#   STEP=window_readout HELP=1 bash cvpr/analysis/readout/run.sh          # 그 스크립트의 --help (argparse 없는 스텝은 docstring)
#   STEP=presence_extract MODEL=vith_ariel_prefix_ep45 GPUS=8 bash …      # 다른 predictor (PRED_CKPT 로 넘어간다)
#   GPUS=8 bash cvpr/harness/submit.sh cvpr/analysis/readout/run.sh       # sbatch (STEP= 등 env 는 --export=ALL 로 그대로 간다)
#
# 노브 (env): GPUS GPU_IDS DRYRUN HELP FORCE OUTDIR ARGS + config.yaml knobs: (TRAIN DECODER R3_OUT R3_EXP PSET FRAMES FEAT_DIR NAME PLAN …)
# 결과: 스크립트 자신의 자리 (z_research/RollOutV2 · RollOutV3/exp_results, 토큰 캐시) — 명령 · 로그 사본은
#       $CVPR_RESULTS/analysis/readout/<step>__<dataset>_<model>/{_cmd.json,stdout.log}
# =============================================================================
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); source "$HERE/../../env.sh"
RO="$HERE/readout.py"
if [[ -z ${STEP:-} ]]; then
  case ${1:-} in ""|-l|--list) exec "$CVPR_PY" "$RO" --list ;; esac
  STEP=$1; shift
fi
[[ ${1:-} == --list ]] && exec "$CVPR_PY" "$RO" --list
DATASET=${1:-${DATASET:-}}
MODEL=${2:-${MODEL:-}}
fail=0
for s in $STEP; do
  echo "### readout/$s  ${DATASET:-(기본)} / ${MODEL:-(기본)}  ($(date '+%H:%M:%S'))"
  "$CVPR_PY" "$RO" "$s" ${DATASET:+--dataset "$DATASET"} ${MODEL:+--model "$MODEL"} ${DRYRUN:+--dry} ${HELP:+--help-script} \
    || { echo "!! 실패 $s"; fail=1; break; }
done
exit $fail
