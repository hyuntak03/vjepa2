#!/bin/bash
# BlenderTrack 렌더 — GPU 마다 작업자 PER_GPU 개 (sbatch 로만 부른다; 노드 로컬 /tmp 는 다른 노드에서 안 보이니 OUT 은 NFS 나 공유 경로)
#   sbatch -p batch_vll -w vll2 --gres=gpu:8 -c 48 --mem=160G -o <OUT>/logs/%j.out \
#     --export=ALL,OUT=<OUT>,SEED=0,START=0,END=4000,GPUS=8,PER_GPU=3,SAMPLES=16 z_research/scripts/data/blender_track/run_render.sh
#SBATCH --job-name=btrack
#SBATCH --time=2-00:00:00
set -u
B=/data/hyuntak/anaconda3/envs/blender/bin/python
HERE=/data/hyuntak/project/2026/2027_cvpr/vjepa2/z_research/scripts/data/blender_track
: "${OUT:?}" "${END:?}"
SEED=${SEED:-0}; START=${START:-0}; GPUS=${GPUS:-8}; PER_GPU=${PER_GPU:-3}; SAMPLES=${SAMPLES:-16}; THREADS=${THREADS:-4}
NW=$((GPUS * PER_GPU))
mkdir -p "$OUT/logs"
echo "host $(hostname) gpus $GPUS per_gpu $PER_GPU workers $NW clips [$START,$END) seed $SEED samples $SAMPLES"
for ((w = 0; w < NW; w++)); do
  g=$((w % GPUS))
  CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=2 nohup $B "$HERE/render.py" --out "$OUT" --seed "$SEED" --start "$START" --end "$END" \
    --worker $w --nworkers $NW --samples "$SAMPLES" --threads "$THREADS" > "$OUT/logs/w${w}_s${SEED}.log" 2>&1 &
done
wait
echo "done $(date)"
