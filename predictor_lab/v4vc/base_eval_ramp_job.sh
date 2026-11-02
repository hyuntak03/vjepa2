#!/bin/bash
# v4 빗면 시험 (v4vcrtest) 을 기준 모델 + 1 차 v4 (평면 묶음) 로 채점 (2026-11-02 2 차).
set -u
R=/data/hyuntak/project/2026/2027_cvpr/vjepa2
cd "$R"
cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
if [[ ${cc%%.*} -ge 12 ]]; then export PY=/data/hyuntak/anaconda3/envs/vjepa2_bw/bin/python; else export PY=/data/hyuntak/anaconda3/envs/vjepa2/bin/python; fi
PF=$R/predictor_lab/v4vc/logs/preds_base_ramp.txt
Z=$R/z_training/runs
printf '%s\n' "release=$Z/release_vith/latest.pt" "vobftg=$Z/vobench_release_postft_g/latest.pt" "v3_full_s0=$Z/v3_full_s0/latest.pt" \
  "v3t_only_s0=$Z/v3t_only_s0/latest.pt" "v4vc_fam_s0=$Z/v4vc_fam_s0/latest.pt" "v4vc_fam6k_s0=$Z/v4vc_fam6k_s0/latest.pt" > "$PF"
PANELS="v4vcrtest" PREDS_FILE=$PF OUTSFX=v4rbase GPUS=$(nvidia-smi -L | wc -l) bash "$R/predictor_lab/v3_ablation/eval_v3.sh"
echo V4VC_RAMP_BASE_DONE
