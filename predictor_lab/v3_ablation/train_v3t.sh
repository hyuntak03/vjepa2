#!/bin/bash
# v3t (trapdoor) 학습 + 채점 (2026-11-01) — vll7 GPU 6 장. v3t 렌더 · 네 팔 합성이 끝난 뒤 sbatch 로:
#   sbatch -p batch_vll -w vll7 --gres=gpu:6 -c 100 --mem=300G --dependency=afterok:<합성 잡> \
#          -o predictor_lab/v3_ablation/logs/%j_v3t.out predictor_lab/v3_ablation/train_v3t.sh
#   1 학습 목록 (t_only · mix1600_t25) + v3t 시험 목록
#   2 기존 팔 전부 (release · VOBench PFT · v3 20 팔) 를 v3t 시험으로 채점 (GPU 4–5) ‖ 새 두 팔 학습 (GPU 0–5 를 같이 씀, 같은 설정 2,000 step · 전역 batch 24)
#   3 새 두 팔을 모든 패널로 채점 (v3 시험 GPU 0–3 ‖ 외부 + v3t 시험 GPU 4–5) → table_v3.py
set -u
R=/data/hyuntak/project/2026/2027_cvpr/vjepa2
D=$R/predictor_lab/v3_ablation
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
export PY=/data/hyuntak/anaconda3/envs/vjepa2_bw/bin/python
cd "$R"
$P z_research/scripts/data/blender_track/make_v3t_meta.py t_only mix1600_t25 || exit 1
$P z_research/scripts/data/blender_track/make_voe_index.py --name v3t || exit 1
PF0=$D/logs/preds_v3t_old.txt
{ echo "release=$R/z_training/runs/release_vith/latest.pt"; echo "vobftg=$R/z_training/runs/vobench_release_postft_g/latest.pt"
  for f in $D/queue/v3_*_s0.done; do r=$(basename "$f" .done); echo "$r=$R/z_training/runs/$r/latest.pt"; done; } > "$PF0"
CUDA_VISIBLE_DEVICES=4,5 PANELS=v3ttest PREDS_FILE=$PF0 OUTSFX=t0 GPUS=2 bash "$D/eval_v3.sh" > "$D/logs/v3t_eval_old.out" 2>&1 &
SLOTS="0 1 2 3 4 5|0 1 2 3 4 5" BATCH=4 bash "$D/queue_node.sh"
wait
PF1=$D/logs/preds_v3t_new.txt
for r in v3t_only_s0 v3_mix1600_t25_s0; do [[ -f $D/queue/$r.done ]] && echo "$r=$R/z_training/runs/$r/latest.pt"; done > "$PF1"
CUDA_VISIBLE_DEVICES=0,1,2,3 PANELS=v3test PREDS_FILE=$PF1 OUTSFX=t1 GPUS=4 bash "$D/eval_v3.sh" > "$D/logs/v3t_eval_new_v3test.out" 2>&1 &
CUDA_VISIBLE_DEVICES=4,5 PANELS="v3ttest vob_late vomo ip1uni ip2uni clevrer rubric" PREDS_FILE=$PF1 OUTSFX=t1 GPUS=2 bash "$D/eval_v3.sh" > "$D/logs/v3t_eval_new_ext.out" 2>&1 &
wait
$P "$D/table_v3.py"
echo V3T_TRAIN_EVAL_DONE
