# =============================================================================
# 셸 진입점들이 공통으로 읽는 환경. **source 전용** (실행 파일이 아니다).
#
#   source "$(dirname "${BASH_SOURCE[0]}")/env.sh"          # harness 안에서
#   source "$PROJ/z_research/scripts/harness/env.sh"        # 밖에서
#
# 하는 일
#   1. VJEPA2_ROOT 를 **이 파일 위치에서** 잡는다 (레포를 옮겨도 안 깨진다)
#   2. paths.env 를 source 한다 (기계별 절대경로의 정본)
#   3. 파생 경로를 export 한다
#
# 이미 환경에 있는 값이 이긴다 (paths.env 가 `KEY=${KEY:-...}` 라서).
# =============================================================================
_WM_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VJEPA2_ROOT="${VJEPA2_ROOT:-$(cd "$_WM_ENV_DIR/../../.." && pwd)}"

# shellcheck source=paths.env
. "$_WM_ENV_DIR/paths.env"

export VJEPA2_PY CONDA_ACTIVATE CONDA_ENV
export WORLD_ROOT BENCH_ROOT TRAIN_DATA_ROOT WMA_CACHE_DIR BENCH_CODE_ROOT
export WM_SLURM_PARTITION WM_SLURM_NODE WM_SLURM_GPUS

# 레포 위치에서 파생되는 것 (기계와 무관하므로 paths.env 에 두지 않는다)
export CKPT_ROOT="${CKPT_ROOT:-$VJEPA2_ROOT/checkpoints}"
export DATA_CSV="${DATA_CSV:-$VJEPA2_ROOT/data_csv}"
export SLURM_LOGS="${SLURM_LOGS:-$VJEPA2_ROOT/z_research/scripts/slurm_logs}"

unset _WM_ENV_DIR
