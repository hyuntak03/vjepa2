#!/bin/bash
#SBATCH --job-name=pl_tspace
#SBATCH --partition=batch_vll
#SBATCH --nodelist=vll5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/data/hyuntak/project/2026/2027_cvpr/vjepa2/paper_lab/_scratch/tspace/slurm_%j.log
# paper_lab 진행자 2 — E1 · E2 (정답 공간과 문맥 평균 vs 운반 보상, 학습 없음). 제출:
#   sbatch paper_lab/scripts/run_tspace.sbatch
set -uo pipefail
REPO=/data/hyuntak/project/2026/2027_cvpr/vjepa2
PY=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
S=$REPO/paper_lab/scripts/target_space_reward.py
export TSR_OUT=$REPO/paper_lab/_scratch/tspace
export OMP_NUM_THREADS=6 PYTHONUNBUFFERED=1
cd $REPO
echo "[$(date '+%F %T')] job $SLURM_JOB_ID on $(hostname), CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi --query-gpu=index,pci.bus_id,memory.used,utilization.gpu --format=csv
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [ "${USED:-99999}" -gt 1000 ]; then echo "할당 GPU 가 비어 있지 않다 (${USED} MiB) — 멈춘다"; exit 3; fi

# VOBench 에서 읽을 프레임만 (clip 당 32 장, stride 3) 페이지 캐시에 올린다 (HDD 랜덤 읽기 회피)
$PY - <<'EOF' > $TSR_OUT/_frames.txt
import csv
for r in csv.DictReader(open("data_csv/vobench_v1/index.csv")):
    for f in range(0, 96, 3): print(f"/data2/local_datasets/world/Benchmark/VOBench_v1/{r['file_name']}/{f:06d}.png")
EOF
( xargs -a $TSR_OUT/_frames.txt -P 32 -n 200 cat > /dev/null ) &
WARM=$!

# 스모크 (모델 적재 · 속도) — 별도 폴더
echo "[$(date '+%F %T')] smoke"
TSR_OUT=$TSR_OUT/_smoke TSR_LIMIT=4 $PY $S calib --n 8 --bs 4 || exit 4
TSR_OUT=$TSR_OUT/_smoke $PY $S basis || exit 5
TSR_OUT=$TSR_OUT/_smoke TSR_LIMIT=2 $PY $S extract --set vobench || exit 6
TSR_OUT=$TSR_OUT/_smoke $PY $S analyze > $TSR_OUT/_smoke/analyze.txt 2>&1 || exit 7
echo "[$(date '+%F %T')] smoke ok"
wait $WARM

echo "[$(date '+%F %T')] calib (SSv2 0..199)"; $PY $S calib --n 200 --bs 4 || exit 10
echo "[$(date '+%F %T')] basis"; $PY $S basis || exit 11
echo "[$(date '+%F %T')] extract vobench"; $PY $S extract --set vobench || exit 12
echo "[$(date '+%F %T')] extract ssv2 (200..399)"; $PY $S extract --set ssv2 --n 200 --bs 4 || exit 13
echo "[$(date '+%F %T')] analyze"; $PY $S analyze > $TSR_OUT/analyze.txt 2>&1 || exit 14
cat $TSR_OUT/analyze.txt
echo "[$(date '+%F %T')] done"
