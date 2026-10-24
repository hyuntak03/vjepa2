#!/bin/bash
# pixel decoder 학습을 세션과 분리해 올린다.   [GPU_IDS="4 5 6 7"] bash z_pixel_decoder/run.sh [config]
#   기본 config z_pixel_decoder/configs/vobench_vits.yaml. 로그 z_pixel_decoder/runs/<name>/train.log · pid runs/<name>/train.pid
#   main 이식 (2026-10-21): vast 판의 tmux + /venv/vjepa 대신 setsid nohup + cvpr/env.sh (CVPR_TORCHRUN · CVPR_VOBENCH · CVPR_LOCAL).
#   GPU_IDS 는 **물리 번호** (CUDA_DEVICE_ORDER=PCI_BUS_ID). vll5 는 공용 노드라 내 job 의 GPU 를 먼저 본다: scontrol show job -d <id> | grep IDX
cd "$(dirname "$0")/.."
source cvpr/env.sh
CFG=${1:-z_pixel_decoder/configs/vobench_vits.yaml}
GPU_IDS=${GPU_IDS:-0}
G=$(wc -w <<<"$GPU_IDS")
NAME=$("$CVPR_PY" -c "import yaml;print(yaml.safe_load(open('$CFG'))['name'])"); D=z_pixel_decoder/runs/$NAME; mkdir -p "$D"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$(tr ' ' ',' <<<"$GPU_IDS") PYTHONPATH=$PWD
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
setsid nohup "$CVPR_TORCHRUN" --nproc-per-node="$G" --master_port=$((25000 + RANDOM % 5000)) \
  z_pixel_decoder/train.py --config "$CFG" >> "$D/train.log" 2>&1 < /dev/null &
echo $! > "$D/train.pid"
echo "올렸다: pid $(cat "$D/train.pid") · GPU $GPU_IDS  ·  tail -f $D/train.log  ·  watch -c -n 2 bash z_pixel_decoder/monitor.sh $NAME"
