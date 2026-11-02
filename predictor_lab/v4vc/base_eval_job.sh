#!/bin/bash
# v4_velocity_control 시험 (v4vctest) 을 기준 모델로 채점 (2026-11-02) — 학습 없이 바로 (vll7 GPU 2 장).
#   release · VOBench PFT · v3 전부 · trapdoor 만 · v3 섞음 1,600 (+ 복사 기준선은 채점기가 같이 낸다)
#   sbatch -p batch_vll -w vll7 --gres=gpu:2 -c 24 --mem=80G -t 3:00:00 -J v4vc_base -o predictor_lab/v4vc/logs/%j_base.out predictor_lab/v4vc/base_eval_job.sh
set -u
R=/data/hyuntak/project/2026/2027_cvpr/vjepa2
cd "$R"
cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
if [[ ${cc%%.*} -ge 12 ]]; then export PY=/data/hyuntak/anaconda3/envs/vjepa2_bw/bin/python; else export PY=/data/hyuntak/anaconda3/envs/vjepa2/bin/python; fi
PF=$R/predictor_lab/v4vc/logs/preds_base.txt
Z=$R/z_training/runs
printf '%s\n' "release=$Z/release_vith/latest.pt" "vobftg=$Z/vobench_release_postft_g/latest.pt" "v3_full_s0=$Z/v3_full_s0/latest.pt" \
  "v3t_only_s0=$Z/v3t_only_s0/latest.pt" "v3_mix1600_s0=$Z/v3_mix1600_s0/latest.pt" > "$PF"
PANELS="v4vctest" PREDS_FILE=$PF OUTSFX=v4base GPUS=2 bash "$R/predictor_lab/v3_ablation/eval_v3.sh"
echo V4VC_BASE_DONE
