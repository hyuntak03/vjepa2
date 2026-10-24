#!/bin/bash
# PLAN §11-1 — #81 사전 등록 확인 (학습 없음, 진행자 2, 2026-10-22). GPU 한 장 (CUDA_VISIBLE_DEVICES 로 받음).
# 판정 규칙은 PLAN §11-1 (결과 전, 해시 사본 _scratch/prereg/PLAN_2026-10-22_m2lb_1548.*).
#   CUDA_VISIBLE_DEVICES=<g> bash paper_lab/scripts/run_p11_1.sh            (공정한 시험 AX 체크포인트는 끝난 뒤 MODELS_EXTRA 로 추가)
set -uo pipefail
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1 CARRY_THREADS=4 OMP_NUM_THREADS=4
LOGF=paper_lab/_scratch/carry/run_p11_1.log; mkdir -p paper_lab/_scratch/carry
echo "[$(date '+%F %T')] start on CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >> $LOGF
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader >> $LOGF
for st in natural_s1 natural8_s1; do
  if [ ! -f paper_lab/_scratch/carry/$st/_data.npz ]; then
    echo "[$(date '+%F %T')] prep $st" >> $LOGF
    $P paper_lab/scripts/carry_ext.py prep --set $st --enc vjepa2 --n 300 >> $LOGF 2>&1 || echo "PREP $st FAILED" >> $LOGF
  fi
done
latest() { for e in 8 6 4 2; do [ -f paper_lab/_scratch/runs/pl_fair_$1/e$e.pt ] && { echo "fair$1_e$e:paper_lab/_scratch/runs/pl_fair_$1/e$e.pt"; return; }; done; }
MODELS="vast70:z_training/vast/ar_state/latest.pt ariel_ar40:z_training/ariel/ar_future_1e_5/latest.pt pilotA0_e10:paper_lab/_scratch/runs/pl_pilot_A0/e10.pt pilotAX_e10:paper_lab/_scratch/runs/pl_pilot_AX/e10.pt state_nosup:z_vll7_training/runs/phys2d_single_v1_pft_nosup/latest.pt $(latest B0s0) $(latest B0s1) ${MODELS_EXTRA:-}"
for m in $MODELS; do
  name=${m%%:*}; ck=${m#*:}
  [ -f paper_lab/_scratch/carry/natural8_s1/$name.npz ] && continue
  echo "[$(date '+%F %T')] model $name <- $ck" >> $LOGF
  $P paper_lab/scripts/carry_coef.py model --ckpt $ck --name $name --sets natural_s1,natural8_s1 >> $LOGF 2>&1 || echo "MODEL $name FAILED" >> $LOGF
done
for st in natural_s1 natural8_s1; do
  for name in _data $(for m in $MODELS; do echo ${m%%:*}; done); do
    [ -f paper_lab/_scratch/carry/$st/$name.npz ] && $P paper_lab/scripts/carry_coef.py placebo --set $st --name $name >> paper_lab/_scratch/carry/placebo_p11_1.txt 2>&1
  done
done
echo "[$(date '+%F %T')] P11_1 DONE" >> $LOGF
