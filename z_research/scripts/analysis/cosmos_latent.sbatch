#!/bin/bash
#SBATCH --job-name=cosmos_latent
#SBATCH --partition=batch_vll
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=3
#SBATCH --mem-per-gpu=40G
#SBATCH --time=0-04:00:00
#SBATCH --output=/data/hyuntak/project/2026/2027_cvpr/vjepa2/z_pixel_decoder/runs/slurm_logs/%x_%A_%a.out
# cosmos_vobench_latent.py 를 shard 마다 GPU 1 장으로 (2026-10-21). 배열 번호 = shard.
#   sbatch -w vll1 --array=0-7 --export=ALL,SET=vobench,NSHARD=8,BS=8 z_research/scripts/analysis/cosmos_latent.sbatch
#   (스모크: --array=0 --export=ALL,SET=vobench,NSHARD=1,BS=8,ARGS="--limit 8")
REPO=/data/hyuntak/project/2026/2027_cvpr/vjepa2
source /data/hyuntak/anaconda3/bin/activate cosmos-predict1
cd /data/hyuntak/project/2026/2027_cvpr/cosmos-predict1
export CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$PWD
echo "node $(hostname) · shard ${SLURM_ARRAY_TASK_ID:-0}/${NSHARD:-1} · set ${SET:-vobench} · bs ${BS:-8} · args ${ARGS:-}"
python $REPO/z_research/scripts/analysis/cosmos_vobench_latent.py --set ${SET:-vobench} --shard ${SLURM_ARRAY_TASK_ID:-0} --nshard ${NSHARD:-1} --bs ${BS:-8} ${ARGS:-}
