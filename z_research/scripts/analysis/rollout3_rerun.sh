#!/usr/bin/env bash
# RollOutV3 — **자를 새로 학습한 뒤** v3 결과 전부를 그 자로 다시 만든다 (RERUN.md §2~4 를 실행 파일로).
#
#   R3_OUT=training_v8 bash z_research/scripts/analysis/rollout3_rerun.sh
#
# 쓰는 곳:  exp_results/windows_<R3_OUT>/   (readings.npz · WINDOW_SUMMARY.md · CROSS_HEAD.md)
#           figures/<R3_OUT>/{readout,windows,examples,windows_gif}/
#           exp_results/<R3_DECODER>/{attn_bias_px.json, COMPARE.md}   (자 폴더, 기본 presence)
# ⚠️ R3_OUT 을 반드시 준다 — 안 주면 옛 결과 폴더 (windows/, figures/<하위>/) 를 덮어쓴다.
# ⚠️ readings.npz 에 자 지문이 실리고, 그림 스크립트는 지문이 다르면 죽는다 (rollout3_paths.check).
# 단계만 골라 돌리려면 STEPS="bias extract tables figs gifs" 에서 빼면 된다.
set -euo pipefail
: "${R3_OUT:?R3_OUT 을 줄 것 (예: R3_OUT=training_v8) — 안 주면 옛 결과를 덮어쓴다}"
export R3_OUT
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
STEPS="${STEPS:-bias extract tables figs gifs}"
GPUS="${GPUS:-8}"
has() { [[ " $STEPS " == *" $1 "* ]]; }
t0=$(date +%s); log() { echo "[rerun $(date +%H:%M:%S) +$(( ($(date +%s)-t0)/60 ))m] $*"; }

if has bias; then
  log "1/5 자의 치우침 + 오류표"
  $P z_research/scripts/analysis/presence_readout_compare.py --bias
  $P z_research/scripts/analysis/presence_readout_compare.py --md
fi
if has extract; then
  log "2/5 v3 16 창 읽기 (p z h + 이식 h z), GPU $GPUS 장"
  rm -rf /dev/shm/rollout3_windows
  $P z_research/scripts/analysis/rollout3_window_readout.py --reps p z h --cross h z --gpus "$GPUS"
  rm -rf /dev/shm/rollout3_windows
fi
if has tables; then
  log "3/5 표 (창 전이 검증 · 이식)"
  $P z_research/scripts/analysis/rollout3_window_summary.py > /dev/null
  $P z_research/scripts/analysis/rollout3_cross_head.py > /dev/null
  # 옛 자와 같은 v3 에서 나란히 — OLD 를 줄 때만
  OLD="${OLD-}"                                        # 비교할 옛 자 (자폴더:readings폴더). 비우면 건너뛴다
  if [[ -n "$OLD" ]]; then                               # → exp_results/windows_<R3_OUT>/DECODER_COMPARE.md
    $P z_research/scripts/analysis/rollout3_decoder_compare.py --old "$OLD" \
       --new "${R3_DECODER:-presence}:windows_${R3_OUT}" > /dev/null
  fi
fi
if has figs; then
  log "4/5 그림 (오류 3 장 · 프로필 16 x 2)"
  $P z_research/scripts/figures/plot_readout_errorbars.py > /dev/null
  for c in 4 8 16 32; do for p in 4 8 16 32; do
    $P z_research/scripts/figures/plot_rollout3_profile.py --ctx $c --prd $p > /dev/null
    $P z_research/scripts/figures/plot_rollout3_windows.py --ctx $c --prd $p > /dev/null
  done; done
fi
if has gifs; then
  log "5/5 GIF (z vs p 예시 · 창 격자)"
  $P z_research/scripts/figures/plot_rollout3_example_gif.py --ctx 16 --prd 32 --reps z p
  $P z_research/scripts/figures/plot_rollout3_windows_gif.py
fi
log "끝 → exp_results/windows_${R3_OUT}/ · figures/${R3_OUT}/"
