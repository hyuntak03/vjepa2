#!/usr/bin/env bash
# fps 4 인접 쌍의 optical flow 캐시를 4 GPU 로 나눠 뽑는다 (2026-10-22).
#   GPU g = 데이터셋마다 shard g. 결과는 $OUT/<데이터셋>/s<g>/ (flow.f16 · valid.u8 · index.tsv · meta.json).
#   학습 config 는 roots: {k400: [.../s0, .../s1, ...]} 로 shard 를 전부 적는다 (FlowCache 가 합친다).
# 진행 보기:  bash z_vll7_training/monitor_flow.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-/data/hyuntak/anaconda3/envs/vjepa2_bw/bin/python}
OUT=${OUT:-/data/hyuntak/flow_cache}
GRID=${GRID:-32}
ARCH=${ARCH:-large}
ITERS=${ITERS:-12}
CONS=${CONS:-fb}
RES=${RES:-256}
CHUNK=${CHUNK:-32}
DW=${DW:-10}
NSH=${NSH:-4}
SESS=${SESS:-flow_extract}
DATASETS=${DATASETS:-"vll7_k400_320 vll7_ssv2"}

mkdir -p "$OUT/logs"
tmux has-session -t "$SESS" 2>/dev/null && { echo "이미 $SESS 가 있다. tmux kill-session -t $SESS 로 지우고 다시"; exit 1; }
tmux new-session -d -s "$SESS" -n main "echo launcher; sleep 1"

for g in $(seq 0 $((NSH - 1))); do
  CMD=""
  for DS in $DATASETS; do
    O="$OUT/$DS/s$g"
    CMD+="$PY z_training/data/build_flow_cache.py --table data_csv/$DS/fps4_table.tsv --out $O "
    CMD+="--grid $GRID --res $RES --arch $ARCH --iters $ITERS --consistency $CONS --chunk $CHUNK "
    CMD+="--decode-workers $DW --shard $g --num-shards $NSH --device cuda:$g "
    CMD+="2>&1 | tee $OUT/logs/${DS}_s$g.log; "
  done
  CMD+="echo ALLDONE_s$g"
  tmux new-window -t "$SESS" -n "g$g" "$CMD"
done
echo "띄움: tmux attach -t $SESS   ·   로그 $OUT/logs/"
