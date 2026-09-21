#!/usr/bin/env bash
#SBATCH --job-name=bench
#SBATCH --partition=batch_vll
#SBATCH --time=12:00:00
#SBATCH --output=/data/hyuntak/project/2026/2027_cvpr/vjepa2/z_research/Benchmarks/exp_results/logs/bench_%j.out
# =============================================================================
# 벤치마크 칸을 노드 하나에 맡긴다. 칸끼리 독립이라 노드별로 쪼개 동시에 돌리면 된다.
#
#   sbatch --nodelist=vll3 --gres=gpu:6 --cpus-per-task=48 --mem=200G \
#          --export=ALL,BENCHES="inflevel_continuity",MODELS="vith vjepa21g videomae2g",GPUS=6 \
#          z_research/Benchmarks/bench.sbatch
#
# ⚠️ 데이터가 어디 있어야 하는가
#     inflevel_*   : /data (NFS) 직독 -> 아무 노드나 된다
#     grasp_level2 : /data2 노드 로컬 -> stage.sh 돌린 노드만 (vll3, vll5)
#     intphys1_dev : /local_datasets 노드 로컬 -> vll5 만
# 같은 칸을 두 노드가 동시에 잡지 않도록 BENCHES 를 겹치지 않게 나눈다
# (run_all.sh 가 summary.json 있는 칸은 건너뛰지만, 동시에 시작하면 못 거른다).
# =============================================================================
set -uo pipefail
REPO=/data/hyuntak/project/2026/2027_cvpr/vjepa2
cd "$REPO"
source /data/hyuntak/anaconda3/bin/activate vjepa2
echo "job ${SLURM_JOB_ID:-local}  host $(hostname)  $(date)"
echo "BENCHES=${BENCHES:-?}  MODELS=${MODELS:-?}  GPUS=${GPUS:-?}"
# ⚠️ 실행 중인 스크립트를 편집하면 bash 가 바이트 오프셋으로 늦게 읽어 실행이 깨진다
#    (2026-09-21: vll3 job 이 그렇게 멈췄다). job 전용 사본을 떠서 돈다.
SNAP=${TMPDIR:-/tmp}/run_all_${SLURM_JOB_ID:-$$}.sh
cp z_research/Benchmarks/run_all.sh "$SNAP"
bash "$SNAP"
echo "exit=$?  $(date)"
