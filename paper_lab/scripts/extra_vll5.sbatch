#!/bin/bash
#SBATCH --job-name=pl_extra
#SBATCH -p batch_vll
#SBATCH -w vll5
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/data/hyuntak/project/2026/2027_cvpr/vjepa2/paper_lab/_scratch/extra_vll5_%j.log
# 리더 (2026-10-22 18:05): vll5 의 빈 GPU 4 장으로 내일 새벽까지 기다리던 일을 앞당긴다. 학습 없음.
#   GPU 슬롯 0: PLAN §12-1 (DINOv2 vs V-JEPA 2 · 주성분 64)          — 대기 세션 pl_p121_wait 를 대신한다
#   GPU 슬롯 1: PLAN §11-1 (#81 확인) — 지금 있는 모델만. 공정한 시험 최종 체크포인트는 pl_p111_wait 가 나중에 더한다
#   GPU 슬롯 2 · 3: 공정한 시험 중간 체크포인트 (e2 · e4 · e6) 평가, seed 0 · 1. fair_chain.sh 와 같은 인자.
#     주 판정 (PLAN §6, 사전 등록) 은 바뀌지 않는다. 중간 값은 학습 곡선 = 서술용.
set -uo pipefail
REPO=/data/hyuntak/project/2026/2027_cvpr/vjepa2; PY=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
cd $REPO
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1
LOGD=$REPO/paper_lab/_scratch/extra_vll5; mkdir -p $LOGD
echo "[$(date '+%F %T')] job $SLURM_JOB_ID on $(hostname) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" | tee $LOGD/job.log
nvidia-smi --query-gpu=index,pci.bus_id,memory.used --format=csv,noheader | tee -a $LOGD/job.log
IFS=, read -ra G <<< "$CUDA_VISIBLE_DEVICES"

interim() {   # <seed> <gpu>
  local s=$1 g=$2 ck arm RUN name
  for ck in e2 e4 e6; do
    for arm in B0s$s AXs$s; do
      RUN=$REPO/paper_lab/_scratch/runs/pl_fair_$arm
      # 체크포인트가 다 써질 때까지 (파일이 생기고 2 분 지남). 학습이 끝나 chain 이 평가를 시작하면 그만둔다
      until [ -f $RUN/$ck.pt ] && [ $(( $(date +%s) - $(stat -c %Y $RUN/$ck.pt) )) -gt 120 ]; do
        grep -q "train end" $RUN/chain.log 2>/dev/null && { echo "[$(date '+%F %T')] $arm train ended before $ck — skip" >> $LOGD/interim_s$s.log; continue 2; }
        sleep 60
      done
      name=fair${arm}_$ck
      echo "[$(date '+%F %T')] extract $name (gpu $g)" >> $LOGD/interim_s$s.log
      CUDA_VISIBLE_DEVICES=$g $PY -u paper_lab/eval/passline.py extract --ckpt $RUN/$ck.pt --name $name --gpus 1 --bs 8 \
          --sets vobench,gravity,natural,ip1late,natcarry --n-natural 200 --fps 4 --fast >> $LOGD/interim_s$s.eval.log 2>&1 \
        || { echo "EXTRACT $name FAILED" >> $LOGD/interim_s$s.log; continue; }
      $PY -u paper_lab/eval/passline.py analyze --name $name --split all >> $LOGD/interim_s$s.eval.log 2>&1
      $PY -u paper_lab/eval/passline.py analyze --name $name --split test >> $LOGD/interim_s$s.eval.log 2>&1
      echo "[$(date '+%F %T')] done $name" >> $LOGD/interim_s$s.log
    done
  done
  echo "[$(date '+%F %T')] INTERIM s$s DONE" >> $LOGD/interim_s$s.log
}

CUDA_VISIBLE_DEVICES=${G[0]} bash paper_lab/scripts/run_p12_1.sh > $LOGD/p12_1.out 2>&1 &
CUDA_VISIBLE_DEVICES=${G[1]} bash paper_lab/scripts/run_p11_1.sh > $LOGD/p11_1.out 2>&1 &
interim 0 ${G[2]} &
interim 1 ${G[3]} &
wait
echo "[$(date '+%F %T')] ALL DONE" | tee -a $LOGD/job.log
