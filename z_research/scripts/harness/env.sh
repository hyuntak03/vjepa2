# shellcheck shell=bash
# -----------------------------------------------------------------------------
# 경로 정본(paths.env)을 셸에 export 한다. **source 전용.**
#
#   source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../z_research/scripts/harness" && pwd)/env.sh"
#
# 내보내는 것: VJEPA2_ROOT (이 파일 위치에서 파생) + paths.env 의 모든 키
#   VJEPA2_PY CONDA_ACTIVATE CONDA_ENV WORLD_ROOT BENCH_ROOT TRAIN_DATA_ROOT WMA_CACHE_DIR
#   BENCH_CODE_ROOT CKPT_ROOT DATA_CSV WM_SLURM_PARTITION WM_SLURM_NODE
# paths.py 가 같은 파일을 같은 규칙으로 읽는다 — 값을 여기서 덮어쓰지 말 것.
# -----------------------------------------------------------------------------
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "env.sh 는 source 로만 쓴다: source ${BASH_SOURCE[0]}" >&2
  exit 1
fi
_WM_HARNESS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VJEPA2_ROOT="$(cd "$_WM_HARNESS/../../.." && pwd)"
if [[ ! -f "$_WM_HARNESS/paths.env" ]]; then
  echo "ERROR: $_WM_HARNESS/paths.env 가 없다 (경로 정본)" >&2
  return 1
fi
set -a
# shellcheck source=/dev/null
source "$_WM_HARNESS/paths.env"
set +a
if [[ ! -x "$VJEPA2_PY" ]]; then
  echo "ERROR: VJEPA2_PY=$VJEPA2_PY 가 실행 파일이 아니다 (harness/paths.env 를 고칠 것)" >&2
  return 1
fi
unset _WM_HARNESS
