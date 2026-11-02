#!/bin/bash
# Real Bench (real_world_permanence, 짝 190) 를 시트 Object_permanence 의 네 모델로 채점 (2026-11-02).
#   ① te 한 번 통과: release · VOBench PFT (vobftg) · V4 평면+빗면 4k (v4vc_all_s0) → predictor_lab/results/v3_ablation/te/rwp_s2/
#   ② 표준 하네스: vast AR-state ep70 (ar_feedback=state 라 te 로더가 못 읽는다 — type_embed. 다른 벤치 AR 칸과 같은 채점 분기)
#      → predictor_lab/results/v3_ablation/harness/rwp_uniform_vast70/
#   sbatch -p batch_vll --exclude=vll7 --gres=gpu:1 -c 8 --mem=48G -t 2:00:00 -J rwp_eval -o predictor_lab/v3_ablation/logs/%j_rwp.out predictor_lab/v3_ablation/rwp_job.sh
#   집계 z_research/scripts/analysis/object_permanence_cells.py
#   (11-02 첫 제출 238145 는 ① 에 AR 을 같이 넣어 load_state_dict 에서 죽었다 — 출력 rwp_s1 은 0/380 이라 지웠다)
set -u
R=/data/hyuntak/project/2026/2027_cvpr/vjepa2
cd "$R"
cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
if [[ ${cc%%.*} -ge 12 ]]; then export PY=/data/hyuntak/anaconda3/envs/vjepa2_bw/bin/python; else export PY=/data/hyuntak/anaconda3/envs/vjepa2/bin/python; fi
PF=$R/predictor_lab/v3_ablation/logs/preds_rwp.txt
Z=$R/z_training/runs
printf '%s\n' "release=$Z/release_vith/latest.pt" "vobftg=$Z/vobench_release_postft_g/latest.pt" "v4vc_all_s0=$Z/v4vc_all_s0/latest.pt" > "$PF"
PANELS="rwp" PREDS_FILE=$PF OUTSFX=s2 GPUS=1 bash "$R/predictor_lab/v3_ablation/eval_v3.sh"
SRC=/data/hyuntak/project/2026/2027_cvpr/testbed/real_bench/real_world_permanence
TAG=rwp_uniform_vast70 OUTDIR=$R/predictor_lab/results/v3_ablation/harness/rwp_uniform_vast70 GPUS=1 \
  SET="data.root=$SRC data.index_csv=$R/data_csv/real_world_permanence_uniform/index.csv data.frames_root=null data.frames_column=frames window.stride=1 window.frames_start=0" \
  bash "$R/cvpr/eval/testbed/run.sh" vobench_v1 vith_vast_ar_state_ep70 || echo "FAIL ar harness"
echo RWP_DONE
