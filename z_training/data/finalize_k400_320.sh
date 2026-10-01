#!/bin/bash
# =============================================================================
# K400 → K400_320 재인코딩 파이프라인 (2026-09-29, 이 기계 판). **재실행 안전** (이미 만든 파일은 건너뛴다).
#
#   tmux new-session -d -s resize_k400 "bash z_training/data/finalize_k400_320.sh"
#   bash z_training/data/finalize_k400_320.sh --no-launch      # 학습은 안 올린다
#   SET=k710 bash z_training/data/finalize_k400_320.sh --no-launch   # K710 → K710_320 (2026-09-29)
#     SET 은 data_csv/<SET>/{train,val}_min112.csv 를 읽고 ${TRAIN_DATA_ROOT}/<SRC_DIR> → <SRC_DIR>_320 으로 쓴다
#
# 1. data_csv/k400/{train,val}_min112.csv 의 영상을 짧은 변 320 · keyint 24 로 재인코딩
#    (z_research/scripts/data/resize_videos.sh, CPU 전부). ${TRAIN_DATA_ROOT}/K400/train → K400_320/train
# 2. 실재하는 출력만으로 data_csv/k400_320/{train,val}_min112.csv 를 만든다 (실패분은 빠진다)
# 3. DRYRUN 검사 뒤 natural_tube_pretrain 을 tmux 로 올리고 .autoresume 표시를 단다
#
# 왜: K400 원본의 ~25 % 가 720p 이고 keyint 가 길어 학습 step 의 절반 이상이 CPU 디코드 대기였다
#     (2026-09-29 실측 6.4 s/step, GPU 0~100 % 진동). GPU(NVENC) 인코딩은 파일마다 CUDA 초기화가
#     CPU sys ~2 s 라 CPU 인코딩(1.2 s)보다 비쌌다 → CPU 로만 한다.
# 부팅 훅: z_training/logs/.resize_pending 이 있으면 autoresume.sh 가 이 스크립트를 다시 올린다.
# =============================================================================
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../z_research/scripts/harness" && pwd)/env.sh"
cd "$VJEPA2_ROOT"
mkdir -p z_training/logs
touch z_training/logs/.resize_pending
SET=${SET:-k400}
case $SET in
  k400) SRC=$TRAIN_DATA_ROOT/K400;        DST=$TRAIN_DATA_ROOT/K400_320 ;;
  k710) SRC=$TRAIN_DATA_ROOT/K710/videos; DST=$TRAIN_DATA_ROOT/K710_320/videos ;;
  *) echo "SET=$SET 모름 (k400|k710)"; exit 1 ;;
esac
LIST=z_training/logs/${SET}_min112_all.csv
cat data_csv/$SET/train_min112.csv data_csv/$SET/val_min112.csv > "$LIST"
echo "[$(date '+%F %T')] 재인코딩 시작: $(wc -l < "$LIST") 편 -> $DST"
find "$DST" -name '*.tmp.mp4' -delete 2>/dev/null
bash z_research/scripts/data/resize_videos.sh "$SRC" "$DST" "$LIST" "${P:-128}" 320
echo "[$(date '+%F %T')] RESIZE_DONE"

mkdir -p data_csv/${SET}_320
for split in train val; do
  "$VJEPA2_PY" - "data_csv/$SET/${split}_min112.csv" "data_csv/${SET}_320/${split}_min112.csv" "$SRC" "$DST" <<'PY'
import os, sys
src, dst, a, b = sys.argv[1:5]
n = m = 0
with open(src) as f, open(dst, "w") as g:
    for ln in f:
        path, lab = ln.rstrip("\n").rsplit(" ", 1); n += 1
        q = path.replace(a + "/", b + "/", 1)
        if os.path.exists(q) and os.path.getsize(q) > 0:
            g.write(f"{q} {lab}\n"); m += 1
print(f"{dst}: {m:,} / {n:,}")
PY
done
echo "failed: $(wc -l < "$DST/_failed.txt" 2>/dev/null || echo 0)"
rm -f z_training/logs/.resize_pending
[[ "${1:-}" == "--no-launch" || $SET != k400 ]] && exit 0   # 자동 학습은 k400 설정만

DRYRUN=1 bash z_training/train.sh natural_tube_pretrain > /dev/null || { echo "DRYRUN 실패 — 학습을 안 올린다"; exit 1; }
echo "DRYRUN OK"
GPUS=8 bash z_training/tmux_train.sh natural_tube_pretrain natural_tube_pretrain
printf 'CONFIG=natural_tube_pretrain\nGPUS=8\n' > z_training/runs/natural_tube_pretrain/.autoresume
echo "[$(date '+%F %T')] 학습 올림 (tmux train_natural_tube_pretrain)"
