#!/usr/bin/env bash
# =============================================================================
# IntPhys 2 Main — Ariel 팔 C (kind=ar, ep18) 를 릴리즈와 같은 격자로 채점한다 (2026-09-27).
#
#   창 16 / 32 / 48, C = 창 x {1/4, 3/8, 1/2, 5/8, 3/4, 7/8}, growing prefix, video_batch 8 (아래 batch_for 주석)
#   (analysis/intphys2/run_grid.sh 와 같은 --set; 다른 것은 predictor_checkpoint 와 encoder 키 둘뿐)
#
#   AR 채점 = 블록별 LN(target_encoder) 문맥 → rollout, 타깃도 블록별 (analysis/predictors/ar_scoring.py).
#   학습 때처럼 encoder 는 target_encoder 하나 (context_encoder_key=target_encoder, dual_encoder=false).
#   ⚠️ 표준 채점과 타깃 공간이 달라 릴리즈와는 **쌍 정확도만** 비교한다.
#
#   GPU_W16=0 GPU_W32=1 GPU_W48=2,3 bash z_training/ariel_eval/run_ip2_ar.sh     # 창마다 따로 띄운다
# =============================================================================
set -uo pipefail
REPO=/data/hyuntak/project/2026/2027_cvpr/vjepa2
cd "$REPO"
PY=/data/hyuntak/anaconda3/envs/vjepa2/bin
L=$REPO/z_training/ariel_eval
# CK / ARTAG 로 판을 고른다 (2026-09-28: latest.pt 가 계속 덮여 고정 사본을 쓴다)
CK=${CK:-$REPO/z_training/ariel/ar_future_1e_5/latest.pt}
ARTAG=${ARTAG:-ar_ep18}
BASE=$REPO/analysis/intphys2/configs/bench_intphys2_main_vith.yaml

# video_batch 1 / 창 배치 16·12·8 (run_grid.sh 기본) 로는 AR rollout 이 GPU 를 못 채워 w32·w48 이 3 시간이었다.
# 2026-09-27 사용자 지시로 video_batch 8 · 창 배치 64·48·16 으로 올린다 (w48 은 32 에서 OOM — 16). ⚠️ 배치 모양이 릴리즈 실행 (video_batch 1) 과
# 달라 surprise 가 ~1e-4 흔들리고 IntPhys 2 쌍 정확도가 ±0.5pt 정도 흔들릴 수 있다 (run_grid.sh 주석의 실측).
VIDEO_BATCH=${VIDEO_BATCH:-8}
batch_for() { case "$1" in 16) echo ${WB16:-64} ;; 32) echo ${WB32:-48} ;; 48) echo ${WB48:-16} ;; esac; }

for w in 16 32 48; do
  var=GPU_W$w; gpus=${!var:-}
  [[ -z $gpus ]] && continue
  n=$(( $(tr -cd ',' <<<"$gpus" | wc -c) + 1 ))
  tag=bench_intphys2_main_ariel_${ARTAG}_w$w
  C=$($PY/python -c "print([int($w*f) for f in (0.25,0.375,0.5,0.625,0.75,0.875)])")
  echo "=== w$w  GPU $gpus ($n)  C=$C  -> $tag"
  CUDA_VISIBLE_DEVICES=$gpus PYTHONPATH=$REPO setsid nohup $PY/torchrun --nproc-per-node=$n \
    --master_port=$((29850 + w)) -m analysis.intphys2.eval --config "$BASE" \
      --set tag=$tag \
      --set model.predictor_checkpoint=$CK \
      --set model.context_encoder_key=target_encoder --set model.dual_encoder=false \
      --set surprise.window_size=$w \
      --set "surprise.context_length_sweep=$C" \
      --set surprise.context_length=$((w/2)) \
      --set surprise.max_window_batch=$(batch_for $w) \
      --set surprise.video_batch=$VIDEO_BATCH \
    > "$L/ip2_${ARTAG}_w$w.log" 2>&1 < /dev/null &
done
wait
