#!/bin/bash
# Ariel predictor 세 팔 (z_training/ariel: full · prefix · ar) 로 RollOutV3 v3 읽기를 한 번에 (2026-09-28).
#   출력 뿌리 z_research/RollOutV3/ariel/ — exp_results/<자> · figures/<팔> · logs (릴리즈 결과는 건드리지 않는다)
#
#   1 프레임 페이지 캐시 예열 (학습셋 · v3)
#   2 학습셋 (training_r8) 특징 → /dev/shm. full · prefix = p 만 (encoder 가 릴리즈와 같아 z · h 는 identity_r8 것을 쓴다),
#     ar = p (블록별 LN(target_encoder) 문맥 8 블록 → rollout 8 블록) + h (블록별 LN(target_encoder))
#   3 정체 자 학습 (identity_r8 과 같은 레시피), 세 팔 동시에 (GPU 2 장씩)
#   4 full · prefix 자 폴더에 identity_r8 의 z · h 자를 복사 (같은 encoder) → 치우침
#   5 v3 16 창 읽기 (팔마다 GPU 8 장). full · prefix 는 p 만 읽고 z 읽은 값은 릴리즈 판에서 가져온다 (같은 encoder · 같은 z 자)
#   6 그림: train_readout · context_to_future (v11 없이), 실제 프레임 기준선 = z (full · prefix) / h (ar)
#
#   srun --jobid=<GPU job> --overlap --cpu-bind=none -n1 -c 96 --mem=0 --gres=gpu:8 bash z_research/scripts/analysis/ariel_readout_chain.sh
set -euo pipefail
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
A=$PWD/z_research/RollOutV3/ariel
export R3_EXP=$A/exp_results
LOG=$A/logs; mkdir -p "$LOG" "$R3_EXP"
SET=training_r8
TRF=/data2/local_datasets/world/world_analysis/RollOut_v2_training_v8
V3F=/data2/local_datasets/world/world_analysis/RollOut_v3/Images
R8=$PWD/z_research/RollOutV3/exp_results/identity_r8
NG=$(nvidia-smi -L | wc -l)
# 팔: 이름 | 체크포인트 | 자 이름 | 학습셋 특징 | v3 표현 | 기준선
ARMS=(
  "ar|z_training/ariel/ar_future_1e_5/latest.pt|identity_ar18|p,h|p h|h"
  "full|z_training/ariel/full_attn_future_pred_1e_5/latest.pt|identity_full40|p|p|z"
  "prefix|z_training/ariel/block_causal_future_only_1e_5/latest.pt|identity_prefix45|p|p|z"
)
t0=$(date +%s); log() { echo "[ariel $(date +%H:%M:%S) +$(( ($(date +%s)-t0)/60 ))m] $*"; }
log "GPU $NG 장 · CPU $(nproc) · 출력 $A"

log "1 프레임 캐시 예열"
find "$TRF/Images" "$V3F" -name "*.png" -print0 | xargs -0 -P 64 -n 200 cat > /dev/null
log "  끝"

for arm in "${ARMS[@]}"; do
  IFS='|' read -r name ck dec treps vreps ref <<< "$arm"
  if [[ -f /dev/shm/ariel_${name}/p.npy.done ]]; then log "2 [$name] 특징 있음 — 건너뜀"; continue; fi
  log "2 [$name] 학습셋 특징 ($treps) → /dev/shm/ariel_$name"
  PRED_CKPT=$PWD/$ck PRESENCE_REPS=$treps PRESENCE_SET=$SET $P -u z_research/scripts/analysis/rollout2_presence_readout.py \
      --set "$SET" --feat-dir /dev/shm/ariel_$name --extract-only --extract-bs 32 --gpus "$NG" > "$LOG/extract_$name.log" 2>&1
  touch /dev/shm/ariel_${name}/p.npy.done
  log "  $(grep -E '^\[extract\] 끝' "$LOG/extract_$name.log" || tail -1 "$LOG/extract_$name.log")"
done

log "3 자 학습 (세 팔 동시에)"
PLANS=("p=0,1 h=2,3" "p=4,5" "p=6,7")
pids=(); i=0
for arm in "${ARMS[@]}"; do
  IFS='|' read -r name ck dec treps vreps ref <<< "$arm"
  reps=${treps//,/ }
  PRESENCE_SET=$SET IDENTITY_FEAT_DIR=/dev/shm/ariel_$name IDENTITY_OUT=$R3_EXP/$dec $P -u z_research/scripts/analysis/rollout2_identity_readout.py \
      --launch --reps $reps --plan "${PLANS[$i]}" > "$LOG/train_$name.log" 2>&1 &
  pids+=($!); i=$((i+1))
done
for pid in "${pids[@]}"; do wait "$pid"; done
for arm in "${ARMS[@]}"; do IFS='|' read -r name _ dec _ _ _ <<< "$arm"; log "  [$name] $(grep -E '정체 57|조합' "$LOG/train_$name.log" | tail -2 | tr '\n' ' ')"; done

log "4 full · prefix 에 identity_r8 의 z · h 자 복사 → 치우침"
for arm in "${ARMS[@]}"; do
  IFS='|' read -r name ck dec treps vreps ref <<< "$arm"
  if [[ "$treps" == "p" ]]; then
    cp -r "$R8/z" "$R8/h" "$R3_EXP/$dec/"
    $P - "$R8/summary.json" "$R3_EXP/$dec/summary.json" <<'PYEOF'
import json, sys
r8, new = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
for rep in ("z", "h"):
    new["reps"][rep] = r8["reps"][rep]
new["z_h_from"] = "identity_r8 (같은 릴리즈 encoder · 같은 학습셋 — 2026-09-28 복사)"
json.dump(new, open(sys.argv[2], "w"), indent=1, ensure_ascii=False)
PYEOF
  fi
  R3_DECODER=$dec $P z_research/scripts/analysis/decoder_bias.py > "$LOG/bias_$name.log" 2>&1
done

for arm in "${ARMS[@]}"; do
  IFS='|' read -r name ck dec treps vreps ref <<< "$arm"
  log "5 [$name] v3 16 창 읽기 ($vreps) → $R3_EXP/windows_$dec"
  R3_DECODER=$dec R3_OUT=$dec PRED_CKPT=$PWD/$ck R3_TMP=/dev/shm/r3w_$name $P -u z_research/scripts/analysis/rollout3_window_readout.py \
      --reps $vreps --gpus "$NG" --token-budget 65536 --max-bs 32 > "$LOG/windows_$name.log" 2>&1
  if [[ "$ref" == "z" ]]; then                      # 릴리즈 판의 z 읽은 값 (같은 encoder · 같은 z 자) 을 붙인다
    $P - "$PWD/z_research/RollOutV3/exp_results/windows_identity_r8/readings.npz" "$R3_EXP/windows_$dec/readings.npz" <<'PYEOF'
import sys, numpy as np
rel, new = np.load(sys.argv[1], allow_pickle=True), dict(np.load(sys.argv[2], allow_pickle=True))
assert (rel["video_id"] == new["video_id"]).all(), "clip 순서가 다르다"
add = [k for k in rel.files if k.endswith("_z") or k.endswith("_z_prob")]
for k in add:
    new[k] = rel[k]
new["z_from"] = np.array("RollOutV3/exp_results/windows_identity_r8 (identity_r8 z 자 = 이 자 폴더의 z 자와 같은 파일)")
np.savez(sys.argv[2], **new); print("z 붙임", len(add))
PYEOF
  fi
  log "  $(tail -1 "$LOG/windows_$name.log")"
  log "6 [$name] 그림 → $A/figures/$name"
  for fig in plot_train_readout plot_context_to_future; do
    R3_DECODER=$dec R3_OUT=$dec R3_FIGROOT=$A/figures/$name R3_REF=$ref R3_NO_V11=1 $P -u z_research/scripts/figures/$fig.py \
        > "$LOG/fig_${fig}_$name.log" 2>&1 || log "  ⚠️ $fig 실패 — $LOG/fig_${fig}_$name.log"
  done
done
rm -rf /dev/shm/ariel_* /dev/shm/r3w_*
log "끝"
