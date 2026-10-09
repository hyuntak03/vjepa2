#!/bin/bash
# vll6 에 auto_research 가 쓰는 프레임을 푼다 (GPU 불필요, NFS 읽기).
# 레지스트리 (configs/protocols/datasets.md) 의 frames_root 는 전부 /local_datasets/world/world_analysis/<세트> 라서
# 같은 경로를 vll6 에서 /data2 로 가는 심볼릭 링크로 세우면 datasets.md 를 한 줄도 안 고쳐도 된다 (HANDOFF §3-3).
#
# 제출:  sbatch -w vll6 -p batch_vll -c 8 --mem 16G -t 6:00:00 \
#          -o auto_research/logs/stage_vll6_%j.out auto_research/scripts/stage_vll6.sh
set -euo pipefail
[ "$(hostname)" = "vll6" ] || { echo "vll6 에서만 돈다 (지금 $(hostname))"; exit 1; }

REPO=/data/hyuntak/project/2026/2027_cvpr/vjepa2
SRC=/data/dataset/world
DST=/data2/local_datasets/world/world_analysis
LOCAL=/local_datasets/world/world_analysis
mkdir -p "$DST"

link() {  # link <이름> : $LOCAL/<이름> -> $DST/<이름>
  if [ -L "$LOCAL/$1" ] || [ ! -e "$LOCAL/$1" ]; then ln -sfn "$DST/$1" "$LOCAL/$1"; fi
}

# /local_datasets/world/world_analysis 가 실제 폴더인지 링크인지에 상관없이 하위 이름 단위로 링크한다
mkdir -p "$LOCAL"

t0=$(date +%s)
# 1) IntPhysGen v11 (81 GB, 12 조건)
if [ ! -f "$DST/IntPhysGen_v11/.staged" ]; then
  echo "[v11] untar $(date)"; tar xf "$SRC/IntPhysGen_v11.tar" -C "$DST"
  touch "$DST/IntPhysGen_v11/.staged"
fi
link IntPhysGen_v11
echo "[v11] $(ls $DST/IntPhysGen_v11/Images | wc -l) 조건 폴더  ($(( $(date +%s) - t0 ))s)"

# 2) IntPhys 1 dev — 프레임 PNG (scene 만) + mp4 폴더 (index.csv 가 거기 있다)
if [ ! -f "$DST/IntPhys1_dev_frame_png/.staged" ]; then
  echo "[intphys1] untar frames $(date)"; mkdir -p "$DST/IntPhys1_dev_frame_png"
  tar xzf "$SRC/Benchmarks/IntPhys1_dev.tar.gz" -C "$DST/IntPhys1_dev_frame_png" --strip-components=1 --wildcards '*/scene/*'
  touch "$DST/IntPhys1_dev_frame_png/.staged"
fi
if [ ! -f "$DST/IntPhys1_dev_videos/.staged" ]; then
  tar xf "$SRC/IntPhys/IntPhys1_dev_videos.tar" -C "$DST"
  cp "$REPO/auto_research/_stage/intphys1_csv/"*.csv "$DST/IntPhys1_dev_videos/"   # vll5 판 (keystones.csv 포함)
  touch "$DST/IntPhys1_dev_videos/.staged"
fi
link IntPhys1_dev_frame_png; link IntPhys1_dev_videos
echo "[intphys1] done ($(( $(date +%s) - t0 ))s)"

# 3) RollOut_v2 (19 GB, 7 시나리오, 위치 자 test 셋)
if [ ! -f "$DST/RollOut_v2/.staged" ]; then
  echo "[rollout_v2] untar $(date)"; tar xf "$SRC/RollOut_v2.tar" -C "$DST"
  touch "$DST/RollOut_v2/.staged"
fi
link RollOut_v2
echo "[rollout_v2] done ($(( $(date +%s) - t0 ))s)"

mkdir -p "$DST/cache"; [ -e "$LOCAL/cache" ] || ln -sfn "$DST/cache" "$LOCAL/cache"
ls -la "$LOCAL"; df -h /data2
echo "ALL DONE $(date)"
