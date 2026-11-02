#!/bin/bash
# v4_velocity_control 학습 팔 하나 (또는 같은 노드의 둘) → 끝나면 같은 GPU 로 모든 패널 채점 (2026-11-02).
#   입력 (환경변수): ARMS ("이름:목록:epoch:GPU묶음 ..." 예 "v4vc_fam_s0:fam:20:0+1+2+3 v4vc_ind_s0:ind:20:4+5+6+7") · OUTSFX (채점 배치 이름)
#   ⚠️ GPU 묶음은 + 로 잇는다 — sbatch --export 는 쉼표에서 변수를 자른다 (11-02 첫 제출이 "0,1,2,3" → "0" 으로 잘려 GPU 1 장 · batch 24 로 OOM)
#   전역 batch 는 24 로 고정 (GPU 당 = 24 / GPU 수). 설정은 v3 ablation 과 같다 (lab_v3_base · seed 0), epoch × ipe 100 = step.
#   sbatch -p batch_vll -w vll3 --gres=gpu:6 -c 40 --mem=150G -t 1-00:00:00 -J v4vc_arm --export=ALL,ARMS="v4vc_fam6k_s0:fam:60:0+1+2+3+4+5",OUTSFX=v4c \
#          -o predictor_lab/v4vc/logs/%j_arm.out predictor_lab/v4vc/train_arm_job.sh
set -u
R=/data/hyuntak/project/2026/2027_cvpr/vjepa2
D=$R/predictor_lab/v4vc
A=$R/predictor_lab/v3_ablation
cd "$R"
mkdir -p "$D/logs" "$D/queue"
cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
if [[ ${cc%%.*} -ge 12 ]]; then export LAB_PY=/data/hyuntak/anaconda3/envs/vjepa2_bw/bin/python; else export LAB_PY=/data/hyuntak/anaconda3/envs/vjepa2/bin/python; fi
export PY=$LAB_PY
echo "[v4vc arm] $(hostname) $(date '+%F %T') GPU $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) cc $cc → $LAB_PY | ARMS=$ARMS"
port0=$(( 33000 + (${SLURM_JOB_ID:-0} % 50) * 10 ))
train() {
  local name=$1 list=$2 ep=$3 devs=$4 k=$5
  [[ -f $D/queue/$name.done ]] && { echo "이미 끝남 $name"; return 0; }
  local ng; ng=$(wc -w <<<"$devs")
  local META=$R/data_csv/blender_track_v4vc/arms/$list.npz
  local ROOT; ROOT=$($LAB_PY -c "import numpy as np,sys; print(np.load(sys.argv[1])['root'])" "$META")
  local SET="data.datasets=[{\"type\":\"blender_track\",\"meta\":\"$META\",\"root\":\"$ROOT\"}] meta.seed=0 data.batch_size=$((24 / ng)) optimization.epochs=$ep"
  echo "학습 시작 $name ($list, epoch $ep × ipe 100, GPU $devs, GPU 당 batch $((24 / ng))) $(date '+%F %T')"
  CONFIG=lab_v3_base NAME=$name GPUS=$ng DEVS="$devs" TRAIN_DDP_PORT=$((port0 + k)) THREADS=4 SET_B64=$(printf %s "$SET" | base64 -w0) \
    bash $R/predictor_lab/train_job.sh > "$D/queue/$name.$(hostname).log" 2>&1
  if [[ -f $R/z_training/runs/$name/latest.pt ]]; then touch "$D/queue/$name.done"; echo "학습 끝 $name $(date '+%T')"
  else echo "학습 실패 $name (로그 $D/queue/$name.$(hostname).log)"; fi
}
# 11-02 2 차: 목록 "all" = 평면 묶음 (blender_track_v4vc/arms/fam.npz) + 빗면 묶음 (blender_track_v4vcr/arms/fam.npz). 같은 렌더 폴더
if [[ $ARMS == *":all:"* ]]; then
  $LAB_PY - <<'PY'
import numpy as np
R = "/data/hyuntak/project/2026/2027_cvpr/vjepa2/data_csv"
a, b = np.load(f"{R}/blender_track_v4vc/arms/fam.npz"), np.load(f"{R}/blender_track_v4vcr/arms/fam.npz")
assert str(a["root"]) == str(b["root"]), (a["root"], b["root"])
np.savez(f"{R}/blender_track_v4vc/arms/all.npz", clip=np.concatenate([a["clip"], b["clip"]]), vis=np.concatenate([a["vis"], b["vis"]]),
         root=a["root"], layout=a["layout"], labels_root=a["labels_root"])
print("목록 all: 평면", len(a["clip"]), "+ 빗면", len(b["clip"]))
PY
fi
k=0; pids=()
for a in $ARMS; do
  IFS=: read -r name list ep devs <<<"$a"
  train "$name" "$list" "$ep" "${devs//+/ }" $k & pids+=($!); k=$((k + 1)); sleep 30
done
wait "${pids[@]}"
PF=$D/logs/preds_${OUTSFX}.txt
for a in $ARMS; do n=${a%%:*}; [[ -f $R/z_training/runs/$n/latest.pt ]] && echo "$n=$R/z_training/runs/$n/latest.pt"; done > "$PF"
NG=$(nvidia-smi -L | wc -l)
[[ -s $PF ]] && PANELS="${EVAL_PANELS:-v4vctest v4vcrtest vob_late vomo v3ttest v3test clevrer ip1uni ip2uni rubric}" PREDS_FILE=$PF OUTSFX=$OUTSFX GPUS=$NG bash "$A/eval_v3.sh"
echo "V4VC_ARM_DONE $OUTSFX $(date '+%F %T')"
