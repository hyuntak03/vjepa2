#!/bin/bash
#SBATCH --job-name=te_queue
#SBATCH --partition=batch_vll
#SBATCH -w vll5
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=300G
#SBATCH --time=0-10:00:00
#SBATCH --output=/data/hyuntak/project/2026/2027_cvpr/vjepa2/z_research/TrainingEffects/logs/te_queue_%j.out
# TrainingEffects — 세션 (job 216129, 05:34 종료) 이 못 끝낸 GPU 추출을 이어서 돈다 (2026-09-25).
# 모든 단계가 이어쓰기 (resume) 를 지원하므로 이미 끝난 것은 건너뛴다. vll5 로컬 데이터 (/local_datasets, /data2) 를 쓴다.
#   sbatch z_research/scripts/analysis/training_effects_queue.sbatch
# 로그: z_research/TrainingEffects/logs/te_queue_<jobid>.out, 단계별 로그는 같은 폴더 te_*.log
set -uo pipefail
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
PY=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
L=z_research/TrainingEffects/logs
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
step() { echo "== $(date '+%F %T') START $1"; shift; "$@" && echo "== $(date '+%F %T') DONE" || echo "== $(date '+%F %T') FAILED ($?)"; }

# 1. v11 per-epoch 채점 (22 predictor) — 끝났으면 즉시 종료
step v11score_curves $PY z_research/scripts/analysis/te_v11_score.py --model vith --preset curves --gpus 8 --batch 4 > $L/te_q_v11score_curves.log 2>&1
# 2. 그룹 B 복사 기준선 (encoder 마다)
for m in vittiny_k400 vittiny_synphys_30k_e225 vittiny_k400ssv2_30k_e225 vittiny_k400_5k vittiny_synphys_5k vitb_k400_e240 vitb_synphys_30k vitb_k400_e100; do
  step "v11score_$m" $PY z_research/scripts/analysis/te_v11_score.py --model $m --gpus 8 --batch 8 > $L/te_q_v11score_$m.log 2>&1
done
# 3. IntPhys1 창별 (final → curves)
step ip1score_final  $PY z_research/scripts/analysis/te_intphys1_score.py --preset final  --gpus 8 > $L/te_q_ip1score_final.log 2>&1
step ip1score_curves $PY z_research/scripts/analysis/te_intphys1_score.py --preset curves --gpus 8 > $L/te_q_ip1score_curves.log 2>&1
# 4. RollOut_v3 (전이) — V3_CMD 가 비어 있으면 건너뛴다
V3_CMD="$PY z_research/scripts/analysis/te_v3_readout.py --preset curves --gpus 8 --validate"
[[ "$V3_CMD" != "__V3_CMD__" ]] && step v3readout bash -c "$V3_CMD > $L/te_q_v3readout.log 2>&1"
echo "== $(date '+%F %T') QUEUE END"
