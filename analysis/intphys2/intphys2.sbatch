#!/usr/bin/env bash
#SBATCH --job-name=ip2
#SBATCH --partition=batch_vll
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=/data/hyuntak/project/2026/2027_cvpr/vjepa2/z_research/Benchmarks/exp_results/logs/ip2_%j.out
# =============================================================================
# IntPhys 2 — 논문 D.3 격자. 설정의 정본은 `analysis/intphys2/README.md`.
#
#   sbatch --nodelist=vll5 --export=ALL,MODELS="vith",WINDOWS="16 32 48" \
#          analysis/intphys2/intphys2.sbatch
#
# 데이터는 노드 로컬 /data2/local_datasets/world/Benchmark/IntPhys2/Main (stage.sh 가 푼다).
# =============================================================================
set -uo pipefail
REPO=/data/hyuntak/project/2026/2027_cvpr/vjepa2
cd "$REPO"
source /data/hyuntak/anaconda3/bin/activate vjepa2

echo "job ${SLURM_JOB_ID:-?}  host $(hostname)  $(date)"
echo "MODELS=${MODELS:-기본}  WINDOWS=${WINDOWS:-기본}  GPUS=${GPUS:-8}"
ls -d /data2/local_datasets/world/Benchmark/IntPhys2/Main || { echo "데이터 없음 — stage.sbatch 먼저"; exit 1; }

# 실행 중 편집으로 깨지지 않게 스냅샷을 뜬다 (2026-09-21: 실제로 한 번 당했다)
SNAP=${TMPDIR:-/tmp}/ip2_grid_${SLURM_JOB_ID:-$$}.sh
cp "$REPO/analysis/intphys2/run_grid.sh" "$SNAP"
bash "$SNAP"
echo "exit=$?  $(date)"
