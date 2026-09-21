#!/bin/bash
# EK100 action anticipation — 진입점.
#
#   GPUS=8 bash z_research/anticipation/EK100/run.sh                  # 본 설정 (configs/ek100_vith.yaml)
#   GPUS=1 SMOKE=1 bash z_research/anticipation/EK100/run.sh          # 배관 점검 (비디오 4개, 1 epoch)
#   DRYRUN=1 bash z_research/anticipation/EK100/run.sh                # 검사만 (GPU 0장, 몇 초)
#
# 환경변수
#   GPUS=N        GPU 수 (기본 1)
#   TAG=name      결과 폴더 이름 (기본 config 의 tag). 설정을 바꾸면 TAG 도 바꾼다 — 같은 TAG 에 다른 설정이면 resolve 가 막는다
#   SET="a.b=1 c.d=2"   config 덮어쓰기 (data./optimization./classifier./evaluation. 는 experiment. 생략 가능)
#   HEADS=sweep   논문의 probe head 20개
#   SMOKE=1  VAL_ONLY=1  DRYRUN=1
#
# evals.main 을 직접 부르지 않는다: 여기서 빈 DDP 포트를 찾고 ANT_EXPECT_WS 를 넘긴다
# (init_distributed 는 포트 충돌을 삼키고 world_size=1 로 폴백한다, CLAUDE.md §7-1).
set -euo pipefail
PY=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
PROJ=/data/hyuntak/project/2026/2027_cvpr/vjepa2
HERE=$PROJ/z_research/anticipation/EK100
cd "$PROJ"
GPUS=${GPUS:-1}

ARGS=(--gpus "$GPUS")
for kv in ${SET:-}; do ARGS+=(--set "$kv"); done
[[ -n "${TAG:-}"    ]] && ARGS+=(--tag "$TAG")
[[ -n "${HEADS:-}"  ]] && ARGS+=(--heads "$HEADS")
[[ -n "${SMOKE:-}"  ]] && ARGS+=(--smoke)
[[ -n "${VAL_ONLY:-}" ]] && ARGS+=(--val-only)
if [[ -n "${DRYRUN:-}" ]]; then exec "$PY" "$HERE/resolve.py" ${CONFIG:-ek100_vith} "${ARGS[@]}" --dry-run; fi
CFG=$("$PY" "$HERE/resolve.py" ${CONFIG:-ek100_vith} "${ARGS[@]}" | tee /dev/stderr | tail -1)

PORT=${EVAL_DDP_PORT:-$("$PY" -c "
import socket
for p in range(20000 + ${SLURM_JOB_ID:-$$} % 10000, 30200):
    s = socket.socket()
    try: s.bind(('', p)); print(p); break
    except OSError: pass
    finally: s.close()")}
export EVAL_DDP_PORT=$PORT ANT_EXPECT_WS=$GPUS EVAL_DDP_TIMEOUT_S=${EVAL_DDP_TIMEOUT_S:-7200}
export PYTHONPATH="$PROJ:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-6}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOG="$(dirname "$CFG")/stdout.log"
echo "port $PORT  log $LOG"
"$PY" -m evals.main --fname "$CFG" --devices $(seq -f "cuda:%g" 0 $((GPUS-1))) ${VAL_ONLY:+--val_only} 2>&1 | tee -a "$LOG"
