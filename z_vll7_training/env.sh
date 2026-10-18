# shellcheck shell=bash
# -----------------------------------------------------------------------------
# vll7 경로 정본(paths.env)을 셸에 export 한다. **source 전용** (`--show` 로 표만 볼 수 있다).
#
#   source z_vll7_training/env.sh
#   bash   z_vll7_training/env.sh --show      # 값 + 실재 여부
#
# 내보내는 것: VJEPA2_ROOT (이 파일 위치에서 파생) + paths.env 의 모든 키
# -----------------------------------------------------------------------------
_V7_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export VJEPA2_ROOT="$(cd "$_V7_DIR/.." && pwd)"
if [[ ! -f "$_V7_DIR/paths.env" ]]; then
  echo "ERROR: $_V7_DIR/paths.env 가 없다 (경로 정본)" >&2
  return 1 2>/dev/null || exit 1
fi
_V7_PY_OVERRIDE="${VJEPA2_PY_OVERRIDE:-}"     # 바깥에서 인터프리터만 갈아끼울 때 (예: env 설치 전 임시판)
set -a
# shellcheck source=/dev/null
source "$_V7_DIR/paths.env"
set +a
[[ -n "$_V7_PY_OVERRIDE" ]] && export VJEPA2_PY="$_V7_PY_OVERRIDE" && echo "VJEPA2_PY override -> $VJEPA2_PY" >&2
export PYTHONPATH="$VJEPA2_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export PYTHONUNBUFFERED=1

if [[ "${1:-}" == "--show" ]]; then
  printf '%-22s %s\n' VJEPA2_ROOT "$VJEPA2_ROOT"
  while IFS='=' read -r k v; do
    [[ -z "$k" || "$k" == \#* ]] && continue
    val="${!k}"; tag=""
    [[ "$val" == /* ]] && { [[ -e "$val" ]] && tag="  ok" || tag="  (없음)"; }
    printf '%-22s %s%s\n' "$k" "${val:-(비어 있음)}" "$tag"
  done < "$_V7_DIR/paths.env"
  exit 0
fi
if [[ ! -x "$VJEPA2_PY" ]]; then
  echo "ERROR: VJEPA2_PY=$VJEPA2_PY 가 실행 파일이 아니다 (z_vll7_training/paths.env 를 고칠 것)" >&2
  return 1 2>/dev/null || exit 1
fi
unset _V7_DIR
