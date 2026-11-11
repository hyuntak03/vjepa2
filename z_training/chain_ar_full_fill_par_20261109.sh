#!/bin/bash
# AR_full_tabl 남은 칸 병렬 채점 (2026-11-09 23:5x KST, 사용자 "GRASP 랑 InfLevel 빨리, GPU VRAM 남잖아").
#   chain_bench.sh (8 장 · 한 번에 하나) 를 대신한다. 같은 SET (chain_bench 와 같은 frame_skips · 창) 에 batch 만 줄여 GPU 를 나눠 쓴다.
#   1) L1 GRASP 창 16 · 32 를 4 장씩 동시에 (p 0.2 GRASP 창 32 가 8 장에서 도는 중 — 장당 61 / 98 GB)
#   2) InfLevel 12 개 (L1 · p 0.2 × 3 세트 × 창 2) 를 2 장씩 4 개 동시에
#   3) 표 다시 계산 → AR_FULL_FILL_DONE (chain_ar_full_fill_20261109.log 에)
cd "$(dirname "$0")/.." || exit 1
source z_research/scripts/harness/env.sh
LOG=z_training/logs/chain_ar_full_fill_20261109.log
say() { echo "[$(TZ=Asia/Seoul date '+%F %T KST')] $*" | tee -a "$LOG"; }
skips() { case $1 in grasp_level2) echo "[2,5,10]" ;; inflevel_*) echo "[5,10,20]" ;; esac; }
job() {  # run bench w devices port mb vb
  local run=$1 b=$2 w=$3 dev=$4 port=$5 mb=$6 vb=$7
  local R=z_training/runs/$run CK=$PWD/z_training/runs/$run/latest.pt OUT=z_training/runs/$run/eval/intphys1_sliding__${b}_latest_w$w
  [[ -f $OUT/summary.json ]] && { say "있음: $run $b w$w"; return; }
  local KS=$("$VJEPA2_PY" -c "import yaml;pc=yaml.safe_load(open('$R/config.yaml'))['model']['predictor'];print(' '.join(f'model.predictor.{k}={pc[k]}' for k in ('kind','n_registers','ar_feedback','ctx_target','ctx_input','ctx_encoder','gating_residual','gating_anchor','gating_out_ln','ar_input','ar_target') if k in pc))")
  local ng=$(wc -w <<< "$dev")
  for try in 1 2 3; do
    say "$run / $b / w$w 시작 ($dev, max_batch $mb, video_batch $vb, 시도 $try)"
    SET="model.predictor_checkpoint=$CK $KS surprise.intphys1.frame_skips=$(skips $b) surprise.intphys1.window_sizes=[$w] model.window_size=$w surprise.intphys1.max_batch=$mb surprise.intphys1.video_batch=$vb" \
      DEVICES="$dev" GPUS=$ng EVAL_DDP_PORT=$port OUTDIR=$OUT TAG="${b}_${run}_w$w" bash z_research/scripts/run.sh intphys1_sliding "$b" vith > "$R/eval/logs/${b}_w$w.log" 2>&1
    [[ -f $OUT/summary.json ]] && { say "  완료 $run $b w$w"; return; }
    say "  !! 실패 $run $b w$w ($(grep -m1 -oE 'OutOfMemoryError|Error[^ ]*' "$R/eval/logs/${b}_w$w.log" || echo 로그))"
    mb=$((mb / 2)); vb=$(( vb > 1 ? vb / 2 : 1 ))
  done
}
L1=v4bc4_ar_vw_state_postft; P2=v4bc4_ar_vw_state_negimp_vanish_app02
mkdir -p z_training/runs/$L1/eval/logs z_training/runs/$P2/eval/logs
say "## 병렬 1: L1 GRASP 두 창"
job $L1 grasp_level2 16 "cuda:0 cuda:1 cuda:2 cuda:3" 29611 48 6 &
job $L1 grasp_level2 32 "cuda:4 cuda:5 cuda:6 cuda:7" 29621 48 6 &
# p 0.2 GRASP 창 32 (8 장) 가 끝나면 InfLevel 을 시작한다
until [[ -f z_training/runs/$P2/eval/intphys1_sliding__grasp_level2_latest_w32/summary.json ]] || ! pgrep -f "z_research/scripts/ru[n].sh intphys1_sliding grasp_level2 vith" > /dev/null; do sleep 30; done
say "## 병렬 2: InfLevel 12 개"
i=0
for run in $P2 $L1; do for s in continuity solidity gravity; do for w in 16 32; do
  pair=$(( i % 4 )); dev="cuda:$((2 * pair)) cuda:$((2 * pair + 1))"
  job $run inflevel_$s $w "$dev" $((29700 + i)) 72 8 &
  i=$((i + 1)); (( i % 4 == 0 )) && wait
done; done; done
wait
say "## 표"
/venv/vjepa/bin/python z_research/scripts/analysis/ar_full_table.py --json z_analysis/results/ar_full_table_20261109.json | tail -7 | tee -a "$LOG"
say "AR_FULL_FILL_DONE"
