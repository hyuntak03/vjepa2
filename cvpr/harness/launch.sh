#!/usr/bin/env bash
# =============================================================================
# cvpr/harness/launch.sh — 모든 cvpr/**/run.sh 가 부르는 **공용 런처** (엔진 셋 공통).
#
#   bash cvpr/harness/launch.sh <프로토콜> <데이터셋> [모델]
#     <프로토콜> = eval/testbed  (-> cvpr/eval/testbed/protocol.yaml) | analysis/probing/attn_probe (-> …/attn_probe.yaml) | yaml 경로   (`cvpr/` 접두사는 안 된다 — resolve.find_protocol)
#
# 하는 일 (옛 run.sh · EK100/run.sh · intphys2/run_grid.sh 가 각자 하던 것을 한 곳에)
#   1. resolve.py 로 config 하나를 만든다 (레지스트리 · 창 · SET · 실물 검사)  — 모델 로딩 전에 죽는다
#   2. 결과 폴더에 _resolved.yaml · _meta.json 을 남긴다
#   3. 빈 DDP 포트를 찾고 split-brain 가드 (*_EXPECT_WS) 를 export 한다
#   4. 엔진에 맞는 명령으로 띄운다: wma/ek100 -> evals.main, intphys2 -> torchrun analysis.intphys2.eval
#
# 환경변수 (전부 선택)
#   GPUS=N            GPU 수 (기본 1)            GPU_IDS="1 2 3"   쓸 GPU 를 직접 (GPUS 보다 우선)
#   WINDOW=c8t32      창 preset               SET="a.b=1 c.d=null"  점 경로 덮어쓰기 (공백 구분; 값에 공백이 필요하면 SET_FILE)
#   SET_FILE=f        한 줄에 KEY=VALUE 하나     LIMIT=N  SMOKE=1  RECACHE=1  VAL_ONLY=1
#   TAG= OUTDIR=      이름을 직접   SUFFIX=_copy  이름 접미사    RESULTS_ROOT=cvpr|legacy|<경로>  (기본 cvpr → $CVPR_RESULTS/<task>/...)
#   DRYRUN=1          병합·검사만              EVAL_DDP_PORT  EVAL_DDP_TIMEOUT_S  WMA_BAR  OMP_NUM_THREADS
#
# ⚠️ 외부 CUDA_VISIBLE_DEVICES 는 evals.main 이 rank 마다 덮어쓴다 (evals/main.py:51) — wma/ek100 은 GPU_IDS 로 고른다.
#    intphys2 (torchrun) 는 CUDA_VISIBLE_DEVICES 를 GPU_IDS 로 만들어 준다.
# =============================================================================
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../env.sh"
PROTO=${1:?"사용법: launch.sh <프로토콜> <데이터셋> [모델]   (목록: resolve.py --list)"}
DATASET=${2:?"사용법: launch.sh <프로토콜> <데이터셋> [모델]"}
MODEL=${3:-${MODEL:-vith}}
cd "$CVPR_REPO"

GPUS=${GPUS:-1}
GPU_IDS=${GPU_IDS:-$(seq -s ' ' 0 $((GPUS - 1)))}
GPUS=$(wc -w <<<"$GPU_IDS")
DEVICES=$(for i in $GPU_IDS; do printf 'cuda:%s ' "$i"; done)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$(( $(nproc) / GPUS < 12 ? $(nproc) / GPUS : 12 ))}   # rank 마다 전체 코어를 잡으면 decode 가 7 배 느려진다

TMP=$(mktemp -d /tmp/cvpr_XXXXXX)
ARGS=(--meta "$TMP/meta.json")
[[ -n ${WINDOW:-} ]] && ARGS+=(--window "$WINDOW")
for kv in ${SET:-}; do ARGS+=(--set "$kv"); done
if [[ -n ${SET_FILE:-} ]]; then
  while IFS= read -r kv; do [[ -n $kv && $kv != \#* ]] && ARGS+=(--set "$kv"); done < "$SET_FILE"
fi
[[ -n ${LIMIT:-} ]]   && ARGS+=(--limit "$LIMIT")
[[ -n ${SMOKE:-} ]]   && ARGS+=(--smoke)
[[ -n ${RECACHE:-} ]] && ARGS+=(--recache)
[[ -n ${SUFFIX:-} ]]  && ARGS+=(--suffix "$SUFFIX")
[[ -n ${TAG:-} ]]     && ARGS+=(--tag "$TAG")
[[ -n ${OUTDIR:-} ]]  && ARGS+=(--outdir "$OUTDIR")
[[ -n ${RESULTS_ROOT:-} ]] && ARGS+=(--results "$RESULTS_ROOT")
[[ -n ${VAL_ONLY:-} ]] && ARGS+=(--val-only)

echo "=================================================="
"$CVPR_PY" "$CVPR_ROOT/harness/resolve.py" "$PROTO" "$DATASET" "$MODEL" -o "$TMP/config.yaml" "${ARGS[@]}"
read -r ENGINE OUT TAGV < <("$CVPR_PY" -c 'import json,sys;m=json.load(open(sys.argv[1]));print(m["engine"],m["output_dir"],m["tag"])' "$TMP/meta.json")
echo "  devices  : ${DEVICES}(world_size=$GPUS)  omp=$OMP_NUM_THREADS"
echo "=================================================="

if [[ -n ${DRYRUN:-} ]]; then
  echo "--- DRYRUN: 병합된 config (engine=$ENGINE) ---"; cat "$TMP/config.yaml"; rm -rf "$TMP"; exit 0
fi

mkdir -p "$OUT"
cp "$TMP/config.yaml" "$OUT/_resolved.yaml"; cp "$TMP/meta.json" "$OUT/_meta.json"; rm -rf "$TMP"
CFG="$OUT/_resolved.yaml"

# ── DDP 포트 (한 노드에서 job 둘이 같은 포트를 쓰면 뒤 job 의 rank0 이 조용히 world_size=1 로 폴백해 두 job 이 섞인다) ──
if [[ -z ${EVAL_DDP_PORT:-} ]]; then
  BASE=$(( 20000 + ( ${SLURM_JOB_ID:-$$} % 10000 ) ))
  EVAL_DDP_PORT=$("$CVPR_PY" - "$BASE" <<'PY'
import socket, sys
b = int(sys.argv[1])
for p in range(b, b + 200):
    s = socket.socket()
    try:
        s.bind(("", p)); s.close(); print(p); sys.exit(0)
    except OSError:
        s.close()
sys.exit(1)
PY
) || { echo "ERROR: 빈 DDP 포트를 못 찾았다"; exit 1; }
fi
export EVAL_DDP_PORT
echo "  ddp port : $EVAL_DDP_PORT   log: $OUT/stdout.log"

case "$ENGINE" in
  wma)
    export WMA_EXPECT_WS=$GPUS
    exec "$CVPR_PY" -m evals.main --fname "$CFG" --devices $DEVICES 2>&1 | tee -a "$OUT/stdout.log"
    ;;
  ek100)
    export ANT_EXPECT_WS=$GPUS
    exec "$CVPR_PY" -m evals.main --fname "$CFG" --devices $DEVICES ${VAL_ONLY:+--val_only} 2>&1 | tee -a "$OUT/stdout.log"
    ;;
  intphys2)
    export CUDA_VISIBLE_DEVICES=$(tr ' ' ',' <<<"$GPU_IDS")
    exec "$CVPR_TORCHRUN" --nproc-per-node="$GPUS" --master_port="$EVAL_DDP_PORT" \
      -m analysis.intphys2.eval --config "$CFG" 2>&1 | tee -a "$OUT/stdout.log"
    ;;
  *) echo "ERROR: 모르는 engine $ENGINE"; exit 1 ;;
esac
