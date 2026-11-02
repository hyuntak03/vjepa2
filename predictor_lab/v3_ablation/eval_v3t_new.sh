#!/bin/bash
# v3t 새 두 팔 (v3t_only_s0 · v3_mix1600_t25_s0) 채점만 — vll7 GPU 2 장 (2026-11-01, 사용자 요청: 6 장 job 237517 취소 후 2 장으로).
#   결과 폴더는 train_v3t.sh 와 같은 t1 — transfer_gap_te.py 가 done.npy 로 이어서 채점한다 (끝난 영상은 다시 안 한다).
#   sbatch -p batch_vll -w vll7 --gres=gpu:2 -c 32 --mem=100G -o predictor_lab/v3_ablation/logs/%j_v3t_eval3.out \
#          predictor_lab/v3_ablation/eval_v3t_new.sh
set -u
R=/data/hyuntak/project/2026/2027_cvpr/vjepa2
D=$R/predictor_lab/v3_ablation
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
export PY=/data/hyuntak/anaconda3/envs/vjepa2_bw/bin/python
cd "$R"
echo "[v3t_eval3] $(hostname) $(date '+%F %T') CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
PANELS="v3test v3ttest vob_late vomo ip1uni ip2uni clevrer rubric" PREDS_FILE=$D/logs/preds_v3t_new.txt OUTSFX=t1 GPUS=2 \
  bash "$D/eval_v3.sh"
$P "$D/table_v3.py"
echo V3T_EVAL3_DONE
