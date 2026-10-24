#!/bin/bash
#SBATCH --job-name=cosmos_render
#SBATCH --partition=batch_vll
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=60G
#SBATCH --time=0-01:00:00
#SBATCH --output=/data/hyuntak/project/2026/2027_cvpr/vjepa2/z_pixel_decoder/runs/slurm_logs/%x_%j.out
# render_cosmos.py 를 다른 노드에서 (2026-10-21). 환경 = cosmos-predict1 (cosmos-predict1/INSTALL.md 추론 절), 프레임 = NFS 사본 runs/_frames/VOBench_v1
#   python z_pixel_decoder/render_cosmos.py --list-frames (vll5, vjepa2 환경) → rsync 로 runs/_frames/VOBench_v1 에 →
#   sbatch -w vll2 --export=ALL,SETS="late" z_pixel_decoder/cosmos.sbatch
#   ⚠️ --export 는 쉼표로 변수를 끊는다 → 목록은 공백으로
REPO=/data/hyuntak/project/2026/2027_cvpr/vjepa2
source /data/hyuntak/anaconda3/bin/activate cosmos-predict1
cd /data/hyuntak/project/2026/2027_cvpr/cosmos-predict1
export CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$PWD
echo "node $(hostname) · CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES · sets ${SETS:-late gravity} · args ${ARGS:-}"
python $REPO/z_pixel_decoder/render_cosmos.py --root $REPO/z_pixel_decoder/runs/_frames/VOBench_v1 \
    --sets $(tr ' ' ',' <<<"${SETS:-late gravity}") ${ARGS:-}
