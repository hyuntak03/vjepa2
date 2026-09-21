#!/usr/bin/env bash
#SBATCH --job-name=bench_ip2
#SBATCH --partition=batch_vll
#SBATCH --nodelist=vll3
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=/data/hyuntak/project/2026/2027_cvpr/vjepa2/z_research/Benchmarks/exp_results/intphys2/ip2_%j.out
# =============================================================================
# IntPhys 2 Main — 기존 하네스(analysis/intphys2) 로 채점한다.
#   sbatch z_research/Benchmarks/intphys2.sbatch                     # 기본 = ViT-H
#   CFG=<다른 config> sbatch z_research/Benchmarks/intphys2.sbatch
# 데이터는 노드 로컬 /data2/local_datasets/world/Benchmark/IntPhys2/Main (stage.sh 가 푼다).
# =============================================================================
set -uo pipefail
REPO=/data/hyuntak/project/2026/2027_cvpr/vjepa2
CFG=${CFG:-$REPO/configs/analysis/intphys2/bench_intphys2_main_vith.yaml}
cd "$REPO"
source /data/hyuntak/anaconda3/bin/activate vjepa2
mkdir -p "$REPO/z_research/Benchmarks/exp_results/intphys2"

echo "job $SLURM_JOB_ID  host $(hostname)  $(date)"
echo "config $CFG"
ls -d /data2/local_datasets/world/Benchmark/IntPhys2/Main || { echo "데이터 없음 — stage.sbatch 먼저"; exit 1; }

torchrun --nproc-per-node=${NG:-8} --master_port=$((20000 + RANDOM % 20000)) \
  -m analysis.intphys2.eval --config "$CFG"
echo "exit=$?  $(date)"
