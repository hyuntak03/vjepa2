#!/usr/bin/env bash
#SBATCH --job-name=ip2copy
#SBATCH --partition=batch_vll
#SBATCH -w vll3
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=100G
#SBATCH --time=8:00:00
#SBATCH --output=/data/hyuntak/project/2026/2027_cvpr/vjepa2/z_research/Benchmarks/exp_results/logs/ip2copy_%j.out
# IntPhys 2 Main 복사 기준선 (IP2_COPY=1, predictor 대신 문맥 마지막 튜블릿 LN 복사) — 2026-09-26 permanence 감사. w48 → w32.
set -uo pipefail
REPO=/data/hyuntak/project/2026/2027_cvpr/vjepa2; cd "$REPO"; source /data/hyuntak/anaconda3/bin/activate vjepa2
export IP2_COPY=1 PYTHONPATH=$REPO
OUT=$REPO/z_research/Benchmarks/exp_results/intphys2; L=$OUT
base=$REPO/analysis/intphys2/configs/bench_intphys2_main_vith.yaml
echo "job ${SLURM_JOB_ID:-?} host $(hostname) $(date)"; ls -d /data2/local_datasets/world/Benchmark/IntPhys2/Main || exit 1
# smoke 20 영상
torchrun --nproc-per-node=4 --master_port=$((24000 + RANDOM % 10000)) -m analysis.intphys2.eval --config "$base" \
  --set tag=bench_intphys2_main_vith_copy_smoke --set surprise.window_size=48 --set "surprise.context_length_sweep=[12,18,24,30,36,42]" \
  --set surprise.context_length=24 --set surprise.max_window_batch=8 --set evaluation.limit_videos=20 > $L/ip2copy_smoke.log 2>&1 || { echo "smoke 실패"; exit 1; }
for w in 48 32; do
  C=$(python -c "print([int($w*f) for f in (0.25,0.375,0.5,0.625,0.75,0.875)])"); mb=$([[ $w == 48 ]] && echo 8 || echo 12)
  tag=bench_intphys2_main_vith_copy_w${w}; echo "=== copy w$w C=$C $(date '+%H:%M:%S')"
  torchrun --nproc-per-node=4 --master_port=$((24000 + RANDOM % 10000)) -m analysis.intphys2.eval --config "$base" \
    --set tag=$tag --set surprise.window_size=$w --set "surprise.context_length_sweep=$C" --set surprise.context_length=$((w/2)) \
    --set surprise.max_window_batch=$mb > $L/${tag}.log 2>&1; echo "   rc=$? $(date '+%H:%M:%S')"
done
echo "exit $(date)"
