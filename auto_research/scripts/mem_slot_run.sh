#!/bin/bash
# [물리] 실험 1 차례 실행 (GPU 2 · 프로세스 하나씩 · 스레드 8): [T] 캐시 대기 → 학습 A → 학습 P → IntPhys 2 채점 → VOBench 캐시 대기 → VOBench 채점
cd /workspace/world_model/vjepa2
L=auto_research/logs
export CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=8
PY="/venv/vjepa/bin/python -W ignore"
echo "[$(TZ=Asia/Seoul date '+%F %T KST')] [T] 캐시 대기" >> $L/mem_slot_run.log
until [ -f /workspace/auto_research_cache/state_obs/phys3d_single_v1_val_lab.npz ]; do sleep 30; done
sleep 30
for ARM in A P; do
  echo "[$(TZ=Asia/Seoul date '+%F %T KST')] 학습 $ARM 시작" >> $L/mem_slot_run.log
  $PY auto_research/scripts/mem_slot.py --arm $ARM --steps 3000 > $L/mem_slot_$ARM.log 2>&1
  echo "[$(TZ=Asia/Seoul date '+%F %T KST')] 학습 $ARM 끝 ($?)" >> $L/mem_slot_run.log
done
$PY auto_research/scripts/mem_slot_eval_ip2.py --ckpts A P > $L/mem_slot_ip2.log 2>&1
echo "[$(TZ=Asia/Seoul date '+%F %T KST')] IntPhys 2 채점 끝 ($?)" >> $L/mem_slot_run.log
until [ -f /workspace/auto_research_cache/vobench_late/meta.json ]; do sleep 30; done
$PY auto_research/scripts/mem_slot_eval_vob.py --ckpts A P > $L/mem_slot_vob.log 2>&1
echo "[$(TZ=Asia/Seoul date '+%F %T KST')] VOBench 채점 끝 ($?)" >> $L/mem_slot_run.log
echo 끝 >> $L/mem_slot_run.log
