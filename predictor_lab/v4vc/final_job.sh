#!/bin/bash
# v4vc 마지막 단계 (2026-11-02): 속도 따라감 (where_speed, WS_MODE=v4vc) → v3 ablation 표 → v4vc 분석. GPU 1 장.
set -u
R=/data/hyuntak/project/2026/2027_cvpr/vjepa2
cd "$R"
cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
if [[ ${cc%%.*} -ge 12 ]]; then P=/data/hyuntak/anaconda3/envs/vjepa2_bw/bin/python; else P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python; fi
WS_MODE=v4vc $P predictor_lab/v3_ablation/where_speed.py extract
$P predictor_lab/v3_ablation/table_v3.py > predictor_lab/v4vc/logs/table_v3.out 2>&1
$P predictor_lab/v4vc/analyze_v4vc.py
echo V4VC_FINAL_DONE
