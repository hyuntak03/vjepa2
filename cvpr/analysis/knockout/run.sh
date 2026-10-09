#!/usr/bin/env bash
# =============================================================================
# analysis/knockout — predictor attention knockout. GPU N 장에 block 을 나눠 (--shard) 돌리고 merge.py 로 합친다.
#
#   N=8 bash cvpr/analysis/knockout/run.sh v11_full vith                  # 옛 run_sharded.sh 기본 (BPC=0 전수 · LAYERS=window · W=3)
#   BPC=1 N=2 bash cvpr/analysis/knockout/run.sh v11 vith                 # 스모크 (조건당 block 1)
#   GPU_IDS="4 5 6 7" bash cvpr/analysis/knockout/run.sh v11_full vith    # GPU 를 고른다 (N = GPU 수)
#   W=1,3,7 LAYERS=window EDGES="mask..ctx mask..mask" bash ...           # 층 창 폭 · 간선 (각 간선마다 <간선>@all 도 붙는다)
#   EDGES= SPECS="mask..hidden@all mask..hidden_ctrl@all" COND_RE=occlusion bash ...   # 클립별 집합 명세만 (EDGES 빈 값 = 간선 스윕 없음)
#   HCACHE=$CVPR_CACHE/<tag> bash ...                                     # target.npy 로 target 인코더 대체 (⚠️ v11 계열 캐시 금지 — README)
#   WINDOW=c8t32 ... / SET="scoring.pairing=cross" ...                    # 창 preset · 점 경로 덮어쓰기 (cvpr/harness/resolve.py 와 같다)
#   DRYRUN=1 bash cvpr/analysis/knockout/run.sh v11_full vith             # 병합 config + shard · merge 명령만 (GPU 0 장, 폴더 안 만든다)
#   RESULTS_ROOT=legacy ...                                               # 옛 폴더 <legacy_results_root>/knockout__<ds>_<model>
#   GPUS=8 bash cvpr/harness/submit.sh cvpr/analysis/knockout/run.sh v11_full vith   # SLURM (env 는 --export=ALL 로 통째로 넘어간다)
#
# 흐름:  resolve.py -> <out>/_resolved.yaml  ->  run.py --protocol <out>/_resolved.yaml --shard i/N  (x N, 병렬)  ->  merge.py
#   run.py 는 analysis/attention/predictor_attn/extract.py:resolve_config() 으로 옛 resolver 를 다시 부르는데 그 resolver 는
#   yaml **경로**를 프로토콜로 받고 TAG/OUTDIR env 로 이름을 짓는다. 그래서 여기서 TAG/OUTDIR 을 export 해 이름이 안 바뀌게 한다.
#   옛 resolver 가 돌려주는 config 가 cvpr 의 것과 같은지는 check.py 가 검사한다 (GPU 불필요).
# 엔진 코드 (analysis/attention/knockout/*) 는 손대지 않았다. 엔진이 launch.sh 의 wma/intphys2/ek100 분기에 없어 여기서 직접 띄운다.
# =============================================================================
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); source "$HERE/../../env.sh"
cd "$CVPR_REPO"
DATASETS=${1:-${DATASETS:-${D:-v11_full}}}
MODELS=${2:-${MODELS:-${M:-vith}}}

# ── shard = GPU. GPU_IDS 가 있으면 그 수가 N, 없으면 N (기본 8 = 옛 run_sharded.sh) 으로 cuda:0..N-1 ──
if [[ -n ${GPU_IDS:-} ]]; then
  [[ -n ${N:-} && $N != $(wc -w <<<"$GPU_IDS") ]] && echo "note: GPU_IDS 가 N 을 이긴다 (N=$N -> $(wc -w <<<"$GPU_IDS"))"
  N=$(wc -w <<<"$GPU_IDS")
else
  N=${N:-${GPUS:-8}}; GPU_IDS=$(seq -s ' ' 0 $((N - 1)))
fi

# ── knockout 노브 (기본값 = 옛 run_sharded.sh) ──
BPC=${BPC:-0}; BS=${BS:-4}; W=${W:-3}; LAYERS=${LAYERS:-window}; WORKERS=${WORKERS:-8}; OMP=${OMP_NUM_THREADS:-8}
EDGES=${EDGES-"mask..ctx mask..mask ctx..ctx"}          # EDGES= (빈 값) 도 허용 — SPECS 만 돌릴 때
METRICS=${METRICS:-surprise_acc,surprise_l1,pred_drift}
args=()
for e in $EDGES; do args+=(--edge "$e" --spec "${e}@all"); done
for s in ${SPECS:-}; do args+=(--spec "$s"); done
[[ -n ${COND_RE:-} ]] && args+=(--cond-re "$COND_RE")
[[ -n ${HCACHE:-} ]]  && args+=(--h-from-cache "$HCACHE")
[[ -n ${EXTRA:-} ]]   && args+=($EXTRA)                  # run.py 의 나머지 인자 그대로 (예: EXTRA="--seed 1 --no-null-baseline")

# ── resolve.py 인자 (cvpr/harness/launch.sh 와 같은 env -> 인자 대응. 병합·--set·이름 규칙은 전부 resolve.py 의 것) ──
RARGS=(--quiet)
[[ -n ${WINDOW:-} ]] && RARGS+=(--window "$WINDOW")
for kv in ${SET:-}; do RARGS+=(--set "$kv"); done
if [[ -n ${SET_FILE:-} ]]; then
  while IFS= read -r kv; do [[ -n $kv && $kv != \#* ]] && RARGS+=(--set "$kv"); done < "$SET_FILE"
fi
[[ -n ${LIMIT:-} ]]   && RARGS+=(--limit "$LIMIT")
[[ -n ${SUFFIX:-} ]]  && RARGS+=(--suffix "$SUFFIX")
[[ -n ${TAG:-} ]]     && RARGS+=(--tag "$TAG")
[[ -n ${OUTDIR:-} ]]  && RARGS+=(--outdir "$OUTDIR")
[[ -n ${RESULTS_ROOT:-} ]] && RARGS+=(--results "$RESULTS_ROOT")

fail=0
for d in $DATASETS; do for m in $MODELS; do
  TMP=$(mktemp -d /tmp/cvpr_ko_XXXXXX)
  echo "=================================================="
  echo "### knockout  $d / $m   N=$N gpus=($GPU_IDS)  BPC=$BPC BS=$BS LAYERS=$LAYERS W=$W METRICS=$METRICS"
  echo "    EDGES=($EDGES)  SPECS=(${SPECS:-})  COND_RE=${COND_RE:-}  HCACHE=${HCACHE:-}  EXTRA=${EXTRA:-}   ($(date '+%F %T'))"
  if ! "$CVPR_PY" "$CVPR_ROOT/harness/resolve.py" "$HERE/protocol.yaml" "$d" "$m" \
        -o "$TMP/config.yaml" --meta "$TMP/meta.json" "${RARGS[@]}"; then
    echo "!! resolve 실패 $d/$m"; fail=1; rm -rf "$TMP"; continue
  fi
  read -r OUT TAGV < <("$CVPR_PY" -c 'import json,sys;m=json.load(open(sys.argv[1]));print(m["output_dir"],m["tag"])' "$TMP/meta.json")
  CFG="$OUT/_resolved.yaml"
  echo "  output   : $OUT"
  echo "  tag      : $TAGV   (TAG/OUTDIR 로 run.py 안의 옛 resolver 에 그대로 넘긴다)"
  echo "=================================================="
  base=("$CVPR_PY" -m analysis.attention.knockout.run --protocol "$CFG" --dataset "$d" --model "$m"
        ${args[@]+"${args[@]}"} --layers "$LAYERS" --window "$W" --metrics "$METRICS"
        --blocks-per-cond "$BPC" --batch-size "$BS" --workers "$WORKERS" -o "$OUT")
  merge=("$CVPR_PY" -m analysis.attention.knockout.merge -o "$OUT")

  if [[ -n ${DRYRUN:-} ]]; then
    echo "--- DRYRUN: 병합된 config ($CFG 에 놓일 것) ---"; cat "$TMP/config.yaml"
    echo "--- DRYRUN: shard 명령  (env TAG=$TAGV OUTDIR=$OUT OMP_NUM_THREADS=$OMP, cwd $CVPR_REPO, 병렬) ---"
    i=0; for g in $GPU_IDS; do echo "  ${base[*]} --device cuda:$g --shard $i/$N  > $OUT/logs/shard$i.log 2>&1 &"; i=$((i + 1)); done
    echo "--- DRYRUN: merge 명령 (shard 전부 끝난 뒤) ---"
    echo "  ${merge[*]}  | tee $OUT/logs/merge.log"
    rm -rf "$TMP"; continue
  fi

  mkdir -p "$OUT/logs" "$OUT/shards"
  cp "$TMP/config.yaml" "$CFG"; cp "$TMP/meta.json" "$OUT/_meta.json"; rm -rf "$TMP"
  pids=(); i=0
  for g in $GPU_IDS; do
    TAG="$TAGV" OUTDIR="$OUT" OMP_NUM_THREADS="$OMP" "${base[@]}" --device "cuda:$g" --shard "$i/$N" \
      > "$OUT/logs/shard$i.log" 2>&1 &
    pids+=($!); i=$((i + 1))
  done
  sfail=0; for p in "${pids[@]}"; do wait "$p" || sfail=1; done
  if [[ $sfail != 0 ]]; then echo "!! shard 실패 $d/$m — $OUT/logs/shard*.log 확인"; fail=1; continue; fi
  echo "[$(date '+%F %T')] shard $N 개 완료, merge"
  if ! "${merge[@]}" 2>&1 | tee "$OUT/logs/merge.log"; then echo "!! merge 실패 $d/$m"; fail=1; continue; fi
  echo "[$(date '+%F %T')] done  $OUT/results.json"
done; done
exit $fail
