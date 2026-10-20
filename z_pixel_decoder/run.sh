#!/bin/bash
# pixel decoder 학습을 tmux 로 올린다.   bash z_pixel_decoder/run.sh [config] [GPUS]
#   기본 config z_pixel_decoder/configs/vobench_vits.yaml, GPU 8. 로그 z_pixel_decoder/runs/<name>/train.log
cd "$(dirname "$0")/.."
CFG=${1:-z_pixel_decoder/configs/vobench_vits.yaml}; G=${2:-8}
NAME=$(python3 -c "import yaml;print(yaml.safe_load(open('$CFG'))['name'])"); mkdir -p z_pixel_decoder/runs/$NAME
tmux new-session -d -s "pixdec_$NAME" "cd $PWD; PYTHONPATH=$PWD /venv/vjepa/bin/torchrun --nproc-per-node=$G --master_port=$((25000 + RANDOM % 5000)) z_pixel_decoder/train.py --config $CFG 2>&1 | tee -a z_pixel_decoder/runs/$NAME/train.log"
echo "올렸다: tmux pixdec_$NAME  ·  tail -f z_pixel_decoder/runs/$NAME/train.log"
