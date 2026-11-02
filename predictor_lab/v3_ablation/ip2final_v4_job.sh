#!/bin/bash
# IntPhys 2 late final (사용자 재라벨 판, 500 쌍) 을 V4 평면+빗면 4k (v4vc_all_s0) 로 채점 (2026-11-02) — 시트 Object_permanence 의 IntPhys 2 열.
#   release · VOBench PFT · AR 은 이미 있다 (cvpr/results/eval/intphys2_late/intphys2_late_final_vith*). 설정은 VOBench PFT 실행과 predictor 경로만 다르다 (DRYRUN diff 0).
#   sbatch -p batch_vll --exclude=vll7 --gres=gpu:1 -c 8 --mem=48G -t 2:00:00 -J ip2f_v4 -o predictor_lab/v3_ablation/logs/%j_ip2f_v4.out predictor_lab/v3_ablation/ip2final_v4_job.sh
set -u
R=/data/hyuntak/project/2026/2027_cvpr/vjepa2
cd "$R"
DS=intphys2_late_final TAG=intphys2_late_final_vith_v4vc_all_s0 OUTDIR=$R/cvpr/results/eval/intphys2_late/intphys2_late_final_vith_v4vc_all_s0 \
  SET="model.predictor_checkpoint=$R/z_training/runs/v4vc_all_s0/latest.pt" GPUS=1 bash cvpr/eval/intphys2_late/run.sh vith
echo IP2F_V4_DONE
