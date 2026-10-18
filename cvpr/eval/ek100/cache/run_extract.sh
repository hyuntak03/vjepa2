#!/usr/bin/env bash
# EK100 encoder 특징 캐시 추출 (extract.py) — split 마다 GPU 수만큼 shard 를 띄우고 끝나면 meta 를 합친다.
#   CFG=<_resolved.yaml> GPU_IDS="2 3" BASE=/data2/local_datasets/EPIC-KITCHENS_resized SPLITS="val train" \
#     setsid nohup bash cvpr/eval/ek100/cache/run_extract.sh > cvpr/logs/ek100_cache_extract.log 2>&1 &
# 이어하기: 같은 명령을 다시 — shard 마다 meta_shard<i>.jsonl 에 적힌 chunk 는 건너뛴다.
set -uo pipefail
cd "$(dirname "$0")/../../../.."
source /data/hyuntak/anaconda3/bin/activate vjepa2
: "${CFG:?CFG=<_resolved.yaml>}"; GPU_IDS=${GPU_IDS:-"0"}; SPLITS=${SPLITS:-"val train"}; VIEWS=${VIEWS:-1}; WORKERS=${WORKERS:-8}
B=(); [[ -n ${BASE:-} ]] && B=(--base "$BASE")
read -ra G <<< "$GPU_IDS"
for sp in $SPLITS; do
  python cvpr/eval/ek100/cache/extract.py --config "$CFG" --split $sp "${B[@]}" --views $VIEWS --prepare 2>&1 | grep -v "file path not found"
  i=0; pids=()
  for g in "${G[@]}"; do
    CUDA_VISIBLE_DEVICES=$g python cvpr/eval/ek100/cache/extract.py --config "$CFG" --split $sp "${B[@]}" --views $VIEWS \
      --shard $i --nshard ${#G[@]} --workers $WORKERS 2>&1 | grep -vE "file path not found|^INFO|key \"" | sed "s/^/[$sp g$g] /" &
    pids+=($!); i=$((i+1))
  done
  wait "${pids[@]}"
  python cvpr/eval/ek100/cache/extract.py --config "$CFG" --split $sp "${B[@]}" --views $VIEWS --merge 2>&1 | grep -v "file path not found"
  echo "=== $sp 끝 $(date '+%m-%d %H:%M')"
done
