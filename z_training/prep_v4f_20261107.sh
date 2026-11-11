#!/bin/bash
# 2026-11-07 v4f 렌더가 끝나면 목록 · 묶음 정보 · 합친 목록 · 미리보기 · DRYRUN 까지 (학습은 사용자 확인 뒤).
cd /workspace/world_model/vjepa2
H=z_research/scripts/data/blender_track; O=/workspace/dataset/training_data/blender_track/v4f_cover
source $H/env.sh
until grep -q V4A_RENDER_DONE $O/render_main.log 2>/dev/null; do sleep 30; done
(cd $H && $BT_PY v4f_cover.py index --out $O --name blender_track_v4f_cover)
$BT_PY $H/concat_negimp_lists.py --name blender_track_v4bc4f --part v4bc4:data_csv/blender_track_v4bc4_negimp/families.json:1 --part v4f:data_csv/blender_track_v4f_cover_negimp/families.json:4
(cd $H && $BT_PY v4f_cover.py preview --out $O --n 8)
DRYRUN=1 NAME=_dry_v4bc4f bash z_training/train.sh v4bc4f_ar_vw_state_negimp 2>&1 | tail -4; rm -rf z_training/runs/_dry_v4bc4f
echo "PREP_V4F_DONE $(TZ=Asia/Seoul date '+%F %T KST')"
