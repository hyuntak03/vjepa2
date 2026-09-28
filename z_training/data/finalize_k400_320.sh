#!/bin/bash
# K400_320 리사이즈(tmux resize_k400)가 끝나면: 실재 파일만으로 csv 를 만들고, DRYRUN 검사 뒤 학습을 tmux 로 올린다.
#   bash z_training/data/finalize_k400_320.sh [--no-launch]
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../z_research/scripts/harness" && pwd)/env.sh"
cd "$VJEPA2_ROOT"
LOG=z_training/logs/resize_k400.log
until grep -q RESIZE_DONE "$LOG" 2>/dev/null; do
  printf "\r[%s] 리사이즈 대기: %s / 95453   " "$(date +%H:%M:%S)" "$(find $TRAIN_DATA_ROOT/K400_320 -name '*.mp4' ! -name '*.tmp.mp4' | wc -l)"; sleep 30
done; echo
mkdir -p data_csv/k400_320
for n in 64 96; do for split in train val; do
  src=data_csv/k400/${split}_min$n.csv; dst=data_csv/k400_320/${split}_min$n.csv
  "$VJEPA2_PY" - "$src" "$dst" "$TRAIN_DATA_ROOT" <<'PY'
import os, sys
src, dst, root = sys.argv[1:4]
n = m = 0
with open(src) as f, open(dst, "w") as g:
    for ln in f:
        path, lab = ln.rstrip("\n").rsplit(" ", 1); n += 1
        q = path.replace(f"{root}/K400/videos/", f"{root}/K400_320/videos/")
        if os.path.getsize(q) > 0 if os.path.exists(q) else False:
            g.write(f"{q} {lab}\n"); m += 1
print(f"{dst}: {m:,} / {n:,}")
PY
done; done
echo "failed: $(wc -l < $TRAIN_DATA_ROOT/K400_320/videos/_failed.txt 2>/dev/null || echo 0)"
DRYRUN=1 bash z_training/train.sh natural_prefix >/dev/null && echo "DRYRUN OK"
[[ "${1:-}" == "--no-launch" ]] && exit 0
GPUS=8 NAME=natural_prefix bash z_training/tmux_train.sh natural_prefix
