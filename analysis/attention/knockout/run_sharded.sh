#!/usr/bin/env bash
# GPU N 장에 block 을 나눠 knockout 을 돌리고, 전부 끝나면 merge.py 로 합친다.
#
#   bash analysis/attention/knockout/run_sharded.sh                 # v11_full, vith, 8 GPU, 전수
#   BPC=1 N=2 OUT=/tmp/x bash analysis/attention/knockout/run_sharded.sh   # 스모크
#
#   D / M      데이터셋 / 모델 (datasets.md / models.md 이름)
#   N          GPU 수 = shard 수 (cuda:0..N-1)
#   BPC        조건당 block 수 (0 = 전부)
#   LAYERS     each | window | all | prefix | suffix, 콤마로 여러 개 (기본 window)
#   W          층 창 폭 (홀수, 콤마 여러 개). 3 = ±1, 7 = ±3
#   EDGES      끊을 간선 (`..` 표기). 각 간선마다 `<간선>@all` 도 붙는다
#   SPECS      명세를 통째로 더 얹을 때 (공백 구분)
#   COND_RE    조건 정규식 필터 (예: occlusion)
#   HCACHE     토큰 캐시 폴더 — target 인코더 대신 target.npy 를 읽는다 (문맥 인코더는 돈다)
#   OUT        출력 폴더. 기본은 하네스 관례 <results_root>/knockout__<D>_<M>
set -euo pipefail
cd "$(dirname "$0")/../../.."
D=${D:-v11_full}; M=${M:-vith}; N=${N:-8}; BPC=${BPC:-0}; BS=${BS:-4}; W=${W:-3}
LAYERS=${LAYERS:-window}
EDGES=${EDGES:-"mask..ctx mask..mask ctx..ctx"}
OUT=${OUT:-z_research/IntPhysGenV11_occlusion_timing_ablation/exp_results/knockout__${D}_${M}}
PY=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
mkdir -p "$OUT/logs"
args=()
for e in $EDGES; do args+=(--edge "$e" --spec "${e}@all"); done
for s in ${SPECS:-}; do args+=(--spec "$s"); done
[ -n "${COND_RE:-}" ] && args+=(--cond-re "$COND_RE")
[ -n "${HCACHE:-}" ] && args+=(--h-from-cache "$HCACHE")
echo "[$(date '+%F %T')] $D/$M  N=$N  BPC=$BPC  LAYERS=$LAYERS  W=$W  EDGES=($EDGES)  SPECS=(${SPECS:-})  COND_RE=${COND_RE:-}  HCACHE=${HCACHE:-}  -> $OUT"
pids=()
for i in $(seq 0 $((N-1))); do
  OMP_NUM_THREADS=8 PYTHONPATH=. "$PY" -m analysis.attention.knockout.run \
    --dataset "$D" --model "$M" --set data.type_column=condition \
    "${args[@]}" --layers "$LAYERS" --window "$W" \
    --metrics surprise_acc,surprise_l1,pred_drift \
    --blocks-per-cond "$BPC" --batch-size "$BS" --workers 8 \
    --device "cuda:$i" --shard "$i/$N" -o "$OUT" > "$OUT/logs/shard$i.log" 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
if [ "$fail" != 0 ]; then echo "[$(date '+%F %T')] shard 실패 — $OUT/logs 확인"; exit 1; fi
echo "[$(date '+%F %T')] shard 완료, merge"
PYTHONPATH=. "$PY" -m analysis.attention.knockout.merge -o "$OUT" | tee "$OUT/logs/merge.log"
echo "[$(date '+%F %T')] done"
