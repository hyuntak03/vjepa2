#!/bin/bash
#SBATCH --job-name=pl_e2bgrav
#SBATCH --partition=batch_vll
#SBATCH --nodelist=vll5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/data/hyuntak/project/2026/2027_cvpr/vjepa2/paper_lab/_scratch/tspace/slurm_e2b_%j.log
# paper_lab 진행자 2 — E2b (중력 세트, 표적 공간별 학습 없는 예측의 고른 미래). E1 의 basis 가 있어야 한다.
#   sbatch [--dependency=afterany:<E3 job>] paper_lab/scripts/run_e2b.sbatch
set -uo pipefail
REPO=/data/hyuntak/project/2026/2027_cvpr/vjepa2; PY=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
S=$REPO/paper_lab/scripts/target_space_reward.py
export TSR_OUT=$REPO/paper_lab/_scratch/tspace PYTHONUNBUFFERED=1 OMP_NUM_THREADS=6
cd $REPO
echo "[$(date '+%F %T')] job $SLURM_JOB_ID on $(hostname)"
nvidia-smi --query-gpu=index,pci.bus_id,memory.used --format=csv
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [ "${USED:-99999}" -gt 1000 ]; then echo "할당 GPU 가 비어 있지 않다 (${USED} MiB) — 멈춘다"; exit 3; fi
[ -f $TSR_OUT/basis.npz ] || { echo "basis 없음 — E1 이 먼저"; exit 4; }
$PY - <<'PYEOF' > $TSR_OUT/_frames_grav.txt
import csv
for r in csv.DictReader(open("data_csv/vobench_v1_gravity/index.csv")):
    for f in range(0, 96, 3): print(f"/data2/local_datasets/world/Benchmark/VOBench_v1/gravity/{r['file_name']}/{f:06d}.png")
PYEOF
xargs -a $TSR_OUT/_frames_grav.txt -P 32 -n 200 cat > /dev/null 2>&1
echo "[$(date '+%F %T')] extract gravity"; $PY $S extract --set gravity || exit 5
echo "[$(date '+%F %T')] analyze"; $PY $S analyze > $TSR_OUT/analyze.txt 2>&1 || exit 6
grep -A 20 "중력 세트" $TSR_OUT/analyze.txt
echo "[$(date '+%F %T')] done"
