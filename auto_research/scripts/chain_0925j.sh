#!/bin/bash
# 2026-09-25 묶음 J: 도달 거리 (SC1) 의 기전 후보 (a) attention 조회 반경 — M4 attention · knockout 을 pv1 에 (release 는 v3_m4).
# 예측: 행동 도달 거리가 긴 predictor (pv1 plateau 4–6.8 칸 vs release 2.3–3.5) 는 진실 칸 미래 query → 문맥 물체 key 질량이 거리에 따라 더 천천히 준다.
# Ariel (prefix) 은 hook 이 prefix 마스크를 덮어쓰므로 여기서 빼고 hook 을 고친 뒤 따로 돈다.
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python; S=auto_research/scripts; C=/data2/local_datasets/world/world_analysis/cache/auto_research
Z=/data/hyuntak/project/2026/2027_cvpr/vjepa2/z_training
run() { echo "[$(date)] >>> $*"; "$@"; r=$?; echo "[$(date)] <<< exit $r"; return $r; }
export ARLIB_PRED_CKPT=$Z/runs/predictor_v1_postft/latest.pt
run $P $S/m4_locality_stride_v3.py --gpus 8 --out $C/v3_m4_pv1
