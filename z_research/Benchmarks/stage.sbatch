#!/usr/bin/env bash
#SBATCH --job-name=bench_stage
#SBATCH --partition=debug_vll
#SBATCH --nodelist=vll3
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/data/hyuntak/project/2026/2027_cvpr/vjepa2/z_research/Benchmarks/exp_results/_stage/stage_%j.out
# =============================================================================
# 벤치마크 tar 를 vll3 로컬 디스크로 푼다.  sbatch 로만 부른다 (자기제출 분기 없음).
#   sbatch z_research/Benchmarks/stage.sbatch
#   watch -c -n 1 bash z_research/Benchmarks/monitor.sh
# 다른 노드에 풀려면:  sbatch --nodelist=vllN z_research/Benchmarks/stage.sbatch
# =============================================================================
set -uo pipefail
REPO=/data/hyuntak/project/2026/2027_cvpr/vjepa2
echo "job $SLURM_JOB_ID on $(hostname)  $(date)"
bash "$REPO/z_research/Benchmarks/stage.sh"
echo "exit=$?  $(date)"
