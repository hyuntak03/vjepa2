#!/bin/bash
# 2026-11-06 자동 탐색: 묶음 2 학습 네 판이 e20 을 낼 때까지 기다린 뒤 v4c (지나가는 판) 2,472 편 렌더 → 목록. 학습과 GPU 를 안 겹치게.
cd /workspace/world_model/vjepa2
R=/workspace/dataset/training_data/blender_track/v4c_slide; H=z_research/scripts/data/blender_track
for r in v4b_ar_vw_state_negimp_w085 v4b_ar_vw_state_negimp_w092 v4bnat_ar_vw_state_negimp_w075 v4bnat_ar_vw_state_negimp; do
  until [ -f z_training/runs/$r/e20.pt ] || grep -q Traceback z_training/runs/$r/train.log 2>/dev/null; do sleep 30; done
done
echo "학습 끝 $(TZ=Asia/Seoul date '+%T KST') → 렌더"
CFG=$R/config_v4c.json SPLIT=slide bash $H/run_v4a.sh $R $R/extra_slide.txt
source $H/env.sh && $BT_PY $H/v4c_slide.py index --out $R --name blender_track_v4c_slide
echo "V4C_DONE $(TZ=Asia/Seoul date '+%T KST')"
