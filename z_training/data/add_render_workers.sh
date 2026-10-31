#!/bin/bash
# 돌고 있는 생성 (gen_phys3d.py --defer-render) 에 렌더 워커를 더 붙인다 (2026-10-31). 같은 claims 대기열을 함께 비우고, 라벨이 끝나 대기열이 비면 스스로 끝난다.
#   GPU_PER=4 GPUS=0,1,2,3,4,5,6,7 CPU_N=0 bash z_training/data/add_render_workers.sh /workspace/dataset/training_data/phys3d_perm_v2/train
#   ⚠️ 그 단계 (train/val) 의 생성이 시작된 뒤에 붙인다 (시작할 때 done 표시 · 남은 claims 를 지운다)
D=${1:?"사용법: add_render_workers.sh <단계 폴더 (…/train)>"}
BL=${BLENDER:-/workspace/tools/blender-4.2.3-linux-x64/blender}
Q=/workspace/world_model/vjepa2/z_training/data/render_queue_blender.py
export PHYS3D_GPU_DENOISER=${PHYS3D_GPU_DENOISER:-OPENIMAGEDENOISE} PHYS3D_DIRECT_MP4=1 PYTHONHASHSEED=0
[ -d "$D/_work/rclaims" ] || { echo "아직 렌더 대기열이 없다: $D/_work/rclaims"; exit 1; }
n=0
for g in ${GPUS//,/ }; do for j in $(seq 1 ${GPU_PER:-4}); do
  CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=2 nohup nice -n 5 $BL -t 2 -b -noaudio -P $Q -- --root $D/clips --claims $D/_work/rclaims --done $D/_work/labels_done --engine cycles \
    > $D/_work/rlog_extra_g${g}_$j.txt 2>&1 &
  n=$((n + 1)); done; done
for j in $(seq 1 ${CPU_N:-0}); do
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 nohup nice -n 15 $BL -t 1 -b -noaudio -P $Q -- --root $D/clips --claims $D/_work/rclaims --done $D/_work/labels_done --engine cpu \
    > $D/_work/rlog_extra_cpu_$j.txt 2>&1 &
  n=$((n + 1)); done
echo "렌더 워커 $n 개 추가 → $D"
