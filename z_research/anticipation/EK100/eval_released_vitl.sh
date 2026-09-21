#!/bin/bash
# 릴리즈 ViT-L EK100 probe (ek100-vitl-256.pt) 를 val 만 평가한다 — 릴리즈 anchor + frame_fixfps + mask 1.
#
#   bash z_research/anticipation/EK100/eval_released_vitl.sh                     # epic_test (공식 val 입력 기하)
#   DATA=/data2/local_datasets/EPIC-KITCHENS_resized TAG=ek100_vitl_rel_fixfps_resized bash z_research/anticipation/EK100/eval_released_vitl.sh
#
# 2026-09-14 결과: resized 29.50 action (9296 clip). epic_test 는 공식 1080p 변환과 픽셀 평균 차 ~2 (압축 노이즈).
set -euo pipefail
PROJ=/data/hyuntak/project/2026/2027_cvpr/vjepa2
cd "$PROJ"
DATA=${DATA:-/data2/local_datasets/EPIC-KITCHENS_resized}   # 2026-09-15: epic_test 를 이 이름으로 옮겼다 (옛 짧은변 256 데이터는 삭제)
TAG=${TAG:-ek100_vitl_rel_fixfps_official256}
PROBE=$PROJ/checkpoint/ek100_probes/ek100-vitl-256.pt
ENC=$PROJ/checkpoint/models--facebook--vjepa2-vitl-fpc64-256/snapshots/b3c1679b7c34d3255ef3547f27c7b226aefab26f/original/model.pth

R=$PROJ/z_research/anticipation/EK100/exp_results/action_anticipation_frozen/$TAG
mkdir -p "$R" && ln -sfn "$PROBE" "$R/latest.pt"

SET="data.base_path=$DATA"
SET+=" data.anticipation_point_mode=released"
SET+=" data.time_source=frame_fixfps"
SET+=" data.num_workers=4"
SET+=" optimization.batch_size=8"
SET+=" model_kwargs.wrapper_kwargs.mask_index=1"
SET+=" model_kwargs.pretrain_kwargs.encoder.model_name=vit_large"
SET+=" model_kwargs.checkpoint=$ENC"

VAL_ONLY=1 TAG=$TAG SET="$SET" GPUS=${GPUS:-1} bash z_research/anticipation/EK100/run.sh
