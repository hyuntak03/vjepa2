#!/bin/bash
#SBATCH --job-name=ek100_build
#SBATCH --partition=batch_vll
#SBATCH -w vll5
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/data/hyuntak/project/2026/2027_cvpr/vjepa2/z_research/anticipation/EK100/slurm_logs/%x_%j.out
# EK100 train (기본) 을 공식 val 기하 256x256 으로 /data2/local_datasets/EPIC-KITCHENS_resized (옛 이름 epic_test) 에 만든다. CPU 작업이라 GPU 는 잡지 않는다.
# (2026-09-15: 처음 판은 GPU 8 · CPU 128 · mem 전체를 잡아 다른 사용자 GPU job 뒤에서 대기했다 → CPU·mem 만)
#
#   sbatch z_research/scripts/data/sbatch_build_ek100_official.sh
#   sbatch --export=ALL,SPLIT=both,JOBS=32 z_research/scripts/data/sbatch_build_ek100_official.sh
#
# 스스로를 제출하지 않는다 (sbatch 로만 부른다). 이미 검증된 비디오는 건너뛰므로 다시 제출하면 이어진다.
# ffmpeg 디코딩 worker 는 메모리를 거의 안 쓴다 (decord 로 자동 재시도하는 깨진 비디오만 수십 GB).
set -euo pipefail
source /data/hyuntak/anaconda3/bin/activate vjepa2
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
SPLIT=${SPLIT:-train}; JOBS=${JOBS:-32}; CRF=${CRF:-12}; COLOR_TOL=${COLOR_TOL:-1.0}; MIN_FREE_GB=${MIN_FREE_GB:-4}
echo "[build] job=${SLURM_JOB_ID:-} node=$(hostname) split=$SPLIT jobs=$JOBS crf=$CRF color_tol=$COLOR_TOL  $(date '+%F %T')"
df -h /data2 | tail -1
python z_research/scripts/data/build_ek100_val_official.py --split "$SPLIT" --jobs "$JOBS" --crf "$CRF" \
    --color-tol "$COLOR_TOL" --min-free-gb "$MIN_FREE_GB"
echo "[build] done $(date '+%F %T')"; df -h /data2 | tail -1
