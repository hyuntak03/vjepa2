#!/bin/bash
# 2026-11-06 자동 탐색: 묶음 4 의 2,000 step 판 셋 + v4bc4 L1 이 e20 을 내면 (4,000 step 판은 계속 돌게 두고) 가속 데이터 두 벌을 렌더 → 목록.
#   v4d = 가파른 빗면 (configs/v4d_ramp.yaml, 1,131 묶음), v4e = 평면 가속 (configs/v4e.yaml, gen_design_v4e, 1,190 묶음). 작업자 GPU 당 8 (취소 뒤 학습이 남지 않는다).
cd /workspace/world_model/vjepa2
H=z_research/scripts/data/blender_track; B=/workspace/dataset/training_data/blender_track
for r in v4bc4_ar_vw_state_negimp_w075 v4bc4_ar_vw_state_postft; do   # 11-06 16:50 KST 사용자가 w085 두 판 · 4,000 step 판을 취소 → 남은 두 판만 기다린다
  until [ -f z_training/runs/$r/e20.pt ] || grep -q Traceback z_training/runs/$r/train.log 2>/dev/null; do sleep 30; done
done
echo "2,000 step 판 끝 $(TZ=Asia/Seoul date '+%T KST') → 렌더"
CFG=$B/v4d_steep/config_v4dr.json SPLIT=rtrain_fam PER_GPU=8 bash $H/run_v4a.sh $B/v4d_steep /nonexistent
CFG=$B/v4e_flat_accel/config_v4e.json SPLIT=train_fam PER_GPU=8 bash $H/run_v4a.sh $B/v4e_flat_accel /nonexistent
source $H/env.sh
$BT_PY $H/v4a_static.py index_root --out $B/v4d_steep --cfg-ramp $B/v4d_steep/config_v4dr.yaml --split-ramp rtrain_fam --name blender_track_v4d
$BT_PY $H/v4a_static.py index_root --out $B/v4e_flat_accel --cfg $B/v4e_flat_accel/config_v4e.yaml --split train_fam --name blender_track_v4e
echo "V4DE_DONE $(TZ=Asia/Seoul date '+%T KST')"
