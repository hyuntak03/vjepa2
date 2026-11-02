#!/bin/bash
# v4_velocity_control (같은 장면 · 다른 속도) 학습 네 팔 → 채점 (2026-11-02, 사용자 요청). 노드 하나 GPU 8 장: 팔마다 GPU 2 장.
#   v4vc_fam_s0    묶음 (장면 하나 × 속도 4 단계)          2,000 step   ↔ v4vc_ind_s0   대조 (장면마다 속도 하나, 영상 수 같음)  2,000 step
#   v4vc_fam6k_s0  묶음                                    6,000 step   ↔ v4vc_ind6k_s0 대조                                  6,000 step
#   설정은 v3 ablation 과 같다 (lab_v3_base · 전역 batch 24 = GPU 2 × 12 · seed 0), 6k 팔만 epoch 60 (× ipe 100).
#   2k 팔 둘이 끝나면 GPU 0–3 에서 모든 패널 채점, 6k 팔이 끝나면 GPU 0–7 에서 채점 → 표 · 분석.
#   sbatch -p batch_vll -w vll4 --gres=gpu:8 -c 80 --mem=300G -t 1-00:00:00 -J v4vc_train \
#          -o predictor_lab/v4vc/logs/%j_train.out predictor_lab/v4vc/train_eval_job.sh
set -u
R=/data/hyuntak/project/2026/2027_cvpr/vjepa2
D=$R/predictor_lab/v4vc
A=$R/predictor_lab/v3_ablation
cd "$R"
mkdir -p "$D/logs" "$D/queue"
cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
if [[ ${cc%%.*} -ge 12 ]]; then export LAB_PY=/data/hyuntak/anaconda3/envs/vjepa2_bw/bin/python; else export LAB_PY=/data/hyuntak/anaconda3/envs/vjepa2/bin/python; fi
export PY=$LAB_PY
echo "[v4vc] $(hostname) $(date '+%F %T') GPU compute_cap $cc → $LAB_PY"
port0=$(( 32000 + (${SLURM_JOB_ID:-0} % 50) * 10 ))
train() {
  local name=$1 list=$2 ep=$3 devs=$4 k=$5
  [[ -f $D/queue/$name.done ]] && { echo "이미 끝남 $name"; return 0; }
  local META=$R/data_csv/blender_track_v4vc/arms/$list.npz
  local ROOT; ROOT=$($LAB_PY -c "import numpy as np,sys; print(np.load(sys.argv[1])['root'])" "$META")
  local SET="data.datasets=[{\"type\":\"blender_track\",\"meta\":\"$META\",\"root\":\"$ROOT\"}] meta.seed=0 data.batch_size=12 optimization.epochs=$ep"
  echo "학습 시작 $name ($list, epoch $ep × ipe 100, GPU $devs) $(date '+%F %T')"
  CONFIG=lab_v3_base NAME=$name GPUS=2 DEVS="$devs" TRAIN_DDP_PORT=$((port0 + k)) THREADS=4 SET_B64=$(printf %s "$SET" | base64 -w0) \
    bash $R/predictor_lab/train_job.sh > "$D/queue/$name.$(hostname).log" 2>&1
  if [[ -f $R/z_training/runs/$name/latest.pt ]] && grep -q "" "$R/z_training/runs/$name/train.log"; then touch "$D/queue/$name.done"; echo "학습 끝 $name $(date '+%T')"
  else echo "학습 실패 $name (로그 $D/queue/$name.$(hostname).log)"; fi
}
train v4vc_fam_s0   fam 20 "0 1" 0 & p1=$!; sleep 30
train v4vc_ind_s0   ind 20 "2 3" 1 & p2=$!; sleep 30
train v4vc_fam6k_s0 fam 60 "4 5" 2 & p3=$!; sleep 30
train v4vc_ind6k_s0 ind 60 "6 7" 3 & p4=$!
wait $p1 $p2
ALL="v4vctest vob_late vomo v3ttest v3test clevrer ip1uni ip2uni rubric"
PF=$D/logs/preds_2k.txt
for n in v4vc_fam_s0 v4vc_ind_s0; do [[ -f $R/z_training/runs/$n/latest.pt ]] && echo "$n=$R/z_training/runs/$n/latest.pt"; done > "$PF"
CUDA_VISIBLE_DEVICES=0,1,2,3 PANELS="$ALL" PREDS_FILE=$PF OUTSFX=v4a GPUS=4 bash "$A/eval_v3.sh"
echo V4VC_2K_EVAL_DONE
wait $p3 $p4
PF=$D/logs/preds_6k.txt
for n in v4vc_fam6k_s0 v4vc_ind6k_s0; do [[ -f $R/z_training/runs/$n/latest.pt ]] && echo "$n=$R/z_training/runs/$n/latest.pt"; done > "$PF"
PANELS="$ALL" PREDS_FILE=$PF OUTSFX=v4b GPUS=8 bash "$A/eval_v3.sh"
echo V4VC_6K_EVAL_DONE
cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
CUDA_VISIBLE_DEVICES=0 WS_MODE=v4vc $LAB_PY "$A/where_speed.py" extract
$LAB_PY "$A/table_v3.py"
$LAB_PY "$D/analyze_v4vc.py" || true
echo V4VC_ALL_DONE
