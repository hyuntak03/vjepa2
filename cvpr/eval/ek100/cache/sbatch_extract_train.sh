#!/bin/bash
#SBATCH --job-name=ek_cache
#SBATCH --partition=batch_vll
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=45G
#SBATCH --time=1-00:00:00
#SBATCH --output=/data/hyuntak/project/2026/2027_cvpr/vjepa2/cvpr/logs/%x_%j.out
# EK100 encoder 특징 캐시 추출 (val · train, int8) → merge → 그 캐시 위에서 probe 학습 한 번 (2026-10-16).
# 캐시는 그 노드의 /data2 에 쓴다 — 학습도 같은 노드에서 해야 한다. 영상도 그 노드의 /data2/local_datasets/EPIC-KITCHENS_resized.
#
#   sbatch -w vll6 --export=ALL,EK_RUN=1,CFG=cvpr/eval/ek100/cache/configs/ek100_rel_1of7strat.yaml,VIEWS=20 cvpr/eval/ek100/cache/sbatch_extract_train.sh
#   (TRAIN=0 이면 추출만; PRED=<predictor ckpt> · BLOCKS=1 이면 AR 모델 학습; TAG= · EXTRA_SET= · SKIP_CHECK=1)
# 이어하기: 같은 명령을 다시 — 이미 한 chunk 는 meta_shard*.jsonl 을 보고 건너뛴다.
set -uo pipefail
[[ "${EK_RUN:-}" == 1 ]] || { echo "EK_RUN=1 을 붙여 sbatch 할 것 (자기제출 방지)"; exit 1; }
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
source /data/hyuntak/anaconda3/bin/activate vjepa2
: "${CFG:?}"; VIEWS=${VIEWS:-20}; NG=${NG:-8}; TRAIN=${TRAIN:-1}; WORKERS=${WORKERS:-10}
NAME=$(python -c "
import yaml,pathlib; d=yaml.safe_load(open('$CFG'))['experiment']['data']
print(f\"{d.get('anticipation_point_mode')}_{d.get('time_source')}_{pathlib.Path(d['dataset_train']).stem}\")")
C=/data2/local_datasets/world/world_analysis/cache/ek100_feats/$NAME
echo "node $(hostname)  cache $C  views $VIEWS  $(date)"; df -h /data2 | tail -1
for sp in val train; do
  if [[ -f $C/$sp/meta.npz ]]; then echo "$sp 이미 있음"; continue; fi
  python cvpr/eval/ek100/cache/extract.py --config $CFG --split $sp --views $VIEWS --dtype q8 --prepare 2>&1 | grep -v "file path"
  for i in $(seq 0 $((NG-1))); do
    CUDA_VISIBLE_DEVICES=$i python cvpr/eval/ek100/cache/extract.py --config $CFG --split $sp --views $VIEWS --shard $i --nshard $NG --workers $WORKERS \
      > cvpr/logs/ek_cache_${NAME}_${sp}_s$i.log 2>&1 &
  done
  wait
  python cvpr/eval/ek100/cache/extract.py --config $CFG --split $sp --views $VIEWS --merge 2>&1 | grep -v "file path"
  echo "=== $sp 끝 $(date)"; df -h /data2 | tail -1
done
[[ "${SKIP_CHECK:-0}" == 1 ]] || CUDA_VISIBLE_DEVICES=0 python cvpr/eval/ek100/cache/check_train_q8.py --config $CFG 2>&1 | grep -E "재현|상대"
[[ "$TRAIN" == 1 ]] || exit 0
TAG=${TAG:-$(python -c "import yaml; print(yaml.safe_load(open('$CFG'))['tag'])")}
S="data.cache_dir=$C optimization.batch_size=$((48 / NG))"
[[ -n ${PRED:-} ]] && S="$S model_kwargs.predictor_checkpoint=$PRED"
[[ "${BLOCKS:-0}" == 1 ]] && S="$S data.cache_blocks=true"
S="$S ${EXTRA_SET:-}"     # 예: EXTRA_SET="optimization.num_epochs=60 evaluation.val_every=10" (콤마 금지 — --export 구분자)
CONFIG=$PWD/$CFG GPUS=$NG TAG=$TAG SET="$S" bash z_research/previous/anticipation/EK100/run.sh
echo "학습 끝 $(date) rc $?"
