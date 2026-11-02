#!/bin/bash
# v3 + v3t 대조 실험 세 팔 (2026-11-01, 사용자 요청) — vll7 GPU 2 장에서 차례로 학습 → 채점.
#   1 v3t_voblike_s0  (vob_like)    VOBench 와 가장 닮은 조각만 (trapdoor · 정면 · 같은 재등장 타이밍, 389 장면) — "비슷한 것만 모으면 VOBench 가 풀리나"
#   2 v3_timingvob_s0 (timing_vob)  재등장 타이밍만 VOBench 와 같게, v3 1,200 + v3t 400 장면 (비교: v3_mix1600_t25_s0)
#   3 v3_oracle_s0    (oracle_test) 시험 장면의 가능 영상을 그대로 학습 — "다시 나올 자리를 외울 수 있으면" 의 상한
#   설정은 v3 ablation 과 같다: lab_v3_base · 2,000 step · 전역 batch 24 (= GPU 2 × 12) · seed 0
#   1 번이 끝나면 VOBench · VO_moving_occluder · trapdoor 시험만 먼저 채점하고, 셋 다 끝나면 모든 패널을 한 번에 채점한다.
#   목록: python z_research/scripts/data/blender_track/make_ctrl_meta.py vob_like timing_vob oracle_test
#   sbatch -p batch_vll -w vll7 --gres=gpu:2 -c 40 --mem=150G -t 7:00:00 -J v3_ctrl \
#          -o predictor_lab/v3_ablation/logs/%j_ctrl.out predictor_lab/v3_ablation/ctrl_job.sh
set -u
R=/data/hyuntak/project/2026/2027_cvpr/vjepa2
D=$R/predictor_lab/v3_ablation
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
export PY=/data/hyuntak/anaconda3/envs/vjepa2_bw/bin/python
cd "$R"
ARMS="v3t_voblike_s0:vob_like v3_timingvob_s0:timing_vob v3_oracle_s0:oracle_test"
port0=$(( 31500 + (${SLURM_JOB_ID:-0} % 50) * 10 ))
train() {
  local name=$1 arm=$2 k=$3
  [[ -f $D/queue/$name.done ]] && { echo "이미 끝남 $name"; return 0; }
  local META=$R/data_csv/blender_track_v3/arms/$arm.npz
  local ROOT; ROOT=$($P -c "import numpy as np,sys; print(np.load(sys.argv[1])['root'])" "$META")
  local SET="data.datasets=[{\"type\":\"blender_track\",\"meta\":\"$META\",\"root\":\"$ROOT\"}] meta.seed=0 data.batch_size=12"
  echo "학습 시작 $name ($arm) $(date '+%F %T')"
  CONFIG=lab_v3_base NAME=$name GPUS=2 DEVS="0 1" TRAIN_DDP_PORT=$((port0 + k)) THREADS=4 SET_B64=$(printf %s "$SET" | base64 -w0) \
    bash $R/predictor_lab/train_job.sh > "$D/queue/$name.$(hostname).log" 2>&1
  if [[ -f $R/z_training/runs/$name/latest.pt ]]; then touch "$D/queue/$name.done"; echo "학습 끝 $name $(date '+%T')"
  else echo "학습 실패 $name (로그 $D/queue/$name.$(hostname).log)"; fi
}
k=0
for a in $ARMS; do
  name=${a%%:*}; arm=${a##*:}
  train "$name" "$arm" $k; k=$((k + 1))
  if [[ $name == v3t_voblike_s0 && -f $R/z_training/runs/$name/latest.pt ]]; then
    echo "$name=$R/z_training/runs/$name/latest.pt" > "$D/logs/preds_ctrl1.txt"
    PANELS="vob_late vomo v3ttest" PREDS_FILE=$D/logs/preds_ctrl1.txt OUTSFX=c1 GPUS=2 bash "$D/eval_v3.sh"
    echo CTRL1_EVAL_DONE
  fi
done
PF=$D/logs/preds_ctrl_all.txt
for a in $ARMS; do n=${a%%:*}; [[ -f $R/z_training/runs/$n/latest.pt ]] && echo "$n=$R/z_training/runs/$n/latest.pt"; done > "$PF"
PANELS="vob_late vomo v3ttest v3test clevrer ip1uni ip2uni rubric" PREDS_FILE=$PF OUTSFX=c2 GPUS=2 bash "$D/eval_v3.sh"
$P "$D/table_v3.py"
echo CTRL_ALL_DONE
