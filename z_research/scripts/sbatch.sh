#!/bin/bash
#SBATCH --job-name=wma
#SBATCH --partition=batch_vll
#SBATCH -w vll5
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=45G
#SBATCH --time=0-08:00:00
#SBATCH --output=/data/hyuntak/project/2026/2027_cvpr/vjepa2/z_research/scripts/slurm_logs/%x_%j.out
#SBATCH --error=/data/hyuntak/project/2026/2027_cvpr/vjepa2/z_research/scripts/slurm_logs/%x_%j.err
# -----------------------------------------------------------------------------
# run.sh 를 SLURM 으로 제출한다. 인자는 run.sh 와 같다.
#
#   sbatch --job-name=ip1 --export=ALL,P=intphys1_sliding,D=intphys1_dev z_research/scripts/sbatch.sh
#   sbatch --job-name=sur --export=ALL,P=surprise_c16t32,D=v8           z_research/scripts/sbatch.sh
#   sbatch --job-name=prb --export=ALL,P=attn_probe,D=v8,M=vith         z_research/scripts/sbatch.sh
#
#   P=프로토콜 (필수) / D=데이터셋 (필수) / M=모델 (기본 vith) / GPUS=N (기본 4)
#
# ★ --gres 와 GPUS 를 반드시 맞출 것:
#     sbatch --gres=gpu:2 --export=ALL,P=attn_probe,D=v8,GPUS=2 z_research/scripts/sbatch.sh
#
# 데이터가 vll5 노드 로컬(/local_datasets)이라 -w vll5 로 고정한다.
#
# ⚠️ -w vll5 라서 이미 vll5 에서 돌고 있는 run 과 같은 노드에 떨어진다. DDP 포트는
#    run.sh 가 SLURM_JOB_ID 로 갈라 잡으므로 겹쳐 돌려도 된다 (직접 지정 불필요).
#    다만 GPU 는 --gres 로 별도 할당을 받아야 한다. bash 로 띄우면 남의 run 이 쓰는
#    GPU 를 그대로 나눠 쓰게 되어 OOM 이 난다.
#
# ⚠️ probing 의 batch_size 는 global 이라 DDP 가 rank 당 1/ws 로 나눈다.
#    surprise 의 batch_size 는 rank 당이라 effective = batch_size x world_size 다.
#    단일 GPU 결과와 비교하려면 surprise 쪽만 GPU 수로 나눌 것.
#
#   진행 상황:  watch -n 1 bash z_scripts/world_model_analysis/monitor.sh
# -----------------------------------------------------------------------------
set -euo pipefail
PROJECT=/data/hyuntak/project/2026/2027_cvpr/vjepa2
P=${P:?"--export=ALL,P=<프로토콜>,D=<데이터셋> 필요  (목록: bash z_research/scripts/run.sh --list)"}
D=${D:?"--export=ALL,P=<프로토콜>,D=<데이터셋> 필요  (목록: bash z_research/scripts/run.sh --list)"}
M=${M:-vith}
source /data/hyuntak/anaconda3/bin/activate vjepa2
cd "$PROJECT"; export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"
mkdir -p "$PROJECT/z_research/scripts/slurm_logs"

echo "node $(hostname) | gres gpus=${SLURM_GPUS_ON_NODE:-?} | $P / $D / $M | GPUS=${GPUS:-4}"
GPUS=${GPUS:-4} bash z_research/scripts/run.sh "$P" "$D" "$M"
