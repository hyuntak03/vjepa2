#!/bin/bash
# VOBench_v2 late permanence (짝 432) 를 시트 Object_permanence 의 네 모델로 채점 (2026-11-02, 사용자: "VOBench 는 v2 기준으로").
#   ① te: release · VOBench PFT (vobftg) · V4 평면+빗면 4k → predictor_lab/results/v3_ablation/te/vob2late_s1/
#   ② 표준 하네스: vast AR-state ep70 (te 로더가 ar_feedback=state 를 못 읽는다) → predictor_lab/results/v3_ablation/harness/vob2late_vast70/
#   sbatch -p batch_vll --exclude=vll7 --gres=gpu:1 -c 8 --mem=48G -t 2:00:00 -J vob2_eval -o predictor_lab/v3_ablation/logs/%j_vob2.out predictor_lab/v3_ablation/vob2_job.sh
#   집계 z_research/scripts/analysis/object_permanence_cells.py
set -u
R=/data/hyuntak/project/2026/2027_cvpr/vjepa2
cd "$R"
cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
if [[ ${cc%%.*} -ge 12 ]]; then export PY=/data/hyuntak/anaconda3/envs/vjepa2_bw/bin/python; else export PY=/data/hyuntak/anaconda3/envs/vjepa2/bin/python; fi
PF=$R/predictor_lab/v3_ablation/logs/preds_vob2.txt
Z=$R/z_training/runs
printf '%s\n' "release=$Z/release_vith/latest.pt" "vobftg=$Z/vobench_release_postft_g/latest.pt" "v4vc_all_s0=$Z/v4vc_all_s0/latest.pt" > "$PF"
PANELS="vob2late" PREDS_FILE=$PF OUTSFX=s1 GPUS=1 bash "$R/predictor_lab/v3_ablation/eval_v3.sh"
TAG=vob2late_vast70 OUTDIR=$R/predictor_lab/results/v3_ablation/harness/vob2late_vast70 GPUS=1 \
  SET="data.root=$R/data_csv/vobench_v2_late_perm data.index_csv=index.csv data.frames_root=/data/hyuntak/project/2026/2027_cvpr/nfs_cache/VOBench_v2_late_perm" \
  bash "$R/cvpr/eval/testbed/run.sh" vobench_v1 vith_vast_ar_state_ep70 || echo "FAIL ar harness"
echo VOB2_DONE
