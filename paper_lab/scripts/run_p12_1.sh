#!/bin/bash
# PLAN §12-1 — E1 · E1b 틀을 DINOv2 한 장 공간에 + 두 encoder 의 주성분 64 대조 (학습 없음, 진행자 2, 2026-10-22). GPU 한 장.
# 판정 규칙은 PLAN §12-1 (결과 전, 해시 사본 _scratch/prereg/PLAN_2026-10-22_dino_1551.*).
set -uo pipefail
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1 OMP_NUM_THREADS=6
LOGF=paper_lab/_scratch/run_p12_1.log
echo "[$(date '+%F %T')] start on CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >> $LOGF
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader >> $LOGF
[ -f paper_lab/_scratch/tspace/_frames.txt ] && xargs -a paper_lab/_scratch/tspace/_frames.txt -P 32 -n 200 cat > /dev/null 2>&1   # /data2 HDD 캐시 예열
# (1) DINOv2 한 장 — E1 다섯 공간 + 주성분 64, E1b
export TSR_ENC=dinov2 TSR_OUT=paper_lab/_scratch/tspace_dino; mkdir -p $TSR_OUT
for ph in "calib --n 200" "basis" "extract --set ssv2 --n 200" "extract --set vobench"; do
  echo "[$(date '+%F %T')] dinov2 $ph" >> $LOGF
  $P paper_lab/scripts/target_space_reward.py $ph --gpus 1 >> $LOGF 2>&1 || echo "DINO $ph FAILED" >> $LOGF
done
$P paper_lab/scripts/target_space_reward.py analyze > $TSR_OUT/analyze.txt 2>&1
for ph in "extract --set vobench" "extract --set ssv2 --n 200"; do
  echo "[$(date '+%F %T')] dinov2 E1b $ph" >> $LOGF
  $P paper_lab/scripts/tspace_2x2.py $ph >> $LOGF 2>&1 || echo "DINO E1b $ph FAILED" >> $LOGF
done
$P paper_lab/scripts/tspace_2x2.py analyze > $TSR_OUT/x2_analyze.txt 2>&1
# (2) V-JEPA 2 블록 — 주성분 64 를 같은 판에서 (보정은 E1 의 것을 그대로, 투영 공간만 다시)
export TSR_ENC=vjepa2 TSR_OUT=paper_lab/_scratch/tspace_pca64 TSR_SPACES=raw,snr64,pca64,snr256,pca256; mkdir -p $TSR_OUT
cp -n paper_lab/_scratch/tspace/calib_r0.npz $TSR_OUT/
for ph in "basis" "extract --set ssv2 --n 200" "extract --set vobench"; do
  echo "[$(date '+%F %T')] vjepa2 pca64 $ph" >> $LOGF
  $P paper_lab/scripts/target_space_reward.py $ph --gpus 1 >> $LOGF 2>&1 || echo "VJEPA $ph FAILED" >> $LOGF
done
$P paper_lab/scripts/target_space_reward.py analyze > $TSR_OUT/analyze.txt 2>&1
echo "[$(date '+%F %T')] P12_1 DONE" >> $LOGF
