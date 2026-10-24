#!/bin/bash
# [T] 반사실 쌍 본 렌더 (KST 2026-10-22, [물리] 생성기 gen_phys3d_cf_blender.py 그대로; 고치지 않음).
#   bash auto_research/scripts/cf_render_main.sh            # GPU 7, 워커 2 개 (blender -t 4), Cycles OptiX
# train: 장면 0..299 × (W_pos, N_pos) = 600 clip  (seed-base 7100000)
# test : 장면 0..49  × (W_pos, N_pos, W_imp, N_imp) = 200 clip  (seed-base 7900000; pilot 은 기본 7000000 — 세 split 이 겹치지 않게)
# 라벨 추가 (리더 KST 04:08, 점 단위 후보): 장마다 world 행렬 · 카메라 (_poses.npz, 기본) + 깊이 (--depth, _depth.npz).
# --keep-frames: PNG (무손실) 을 남기고 cf_encode.py 가 PNG 에서 인코딩한다 (mp4 압축이 짝 안 같은 문맥을 흔든다).
# 이어 하기: 생성기가 json 있는 장면을 건너뛴다. 멈추려면 이 스크립트와 blender 워커를 끈다.
cd /workspace/world_model/vjepa2
BL=/workspace/tools/blender-4.2.3-linux-x64/blender
G=auto_research/scripts/gen_phys3d_cf_blender.py
OUT=/workspace/dataset/training_data/phys3d_cf_v1
export CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=4
seq_csv() { python3 -c "import sys; a,b,s,o=map(int,sys.argv[1:]); print(','.join(str(k) for k in range(a,b) if k%s==o))" "$@"; }
worker() {
  w=$1
  $BL -t 4 -b -noaudio -P $G -- --scenes $(seq_csv 0 300 2 $w) --split train --seed-base 7100000 --variants W_pos,N_pos --out $OUT --keep-frames --depth \
      > auto_research/logs/cf_main_train_w$w.log 2>&1
  $BL -t 4 -b -noaudio -P $G -- --scenes $(seq_csv 0 50 2 $w) --split test --seed-base 7900000 --variants W_pos,N_pos,W_imp,N_imp --out $OUT --keep-frames --depth \
      > auto_research/logs/cf_main_test_w$w.log 2>&1
  echo "worker $w done $(date -u +%H:%M)"
}
worker 0 &
worker 1 &
wait
echo "render done $(date -u +%H:%M)"
