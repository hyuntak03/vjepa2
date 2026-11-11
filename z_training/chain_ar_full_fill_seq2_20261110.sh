#!/bin/bash
# AR_full_tabl 남은 칸 병렬 채점 (2026-11-09 23:5x KST, 사용자 "GRASP 랑 InfLevel 빨리, GPU VRAM 남잖아").
#   chain_bench.sh (8 장 · 한 번에 하나) 를 대신한다. 같은 SET (chain_bench 와 같은 frame_skips · 창) 에 batch 만 줄여 GPU 를 나눠 쓴다.
#   ⚠️ 정정 (23:5x): 병렬은 오히려 느렸다 [측정] — L1 GRASP 4 장 · 배치 48 이 rank 당 0.08–0.10 편/초 (1024 편에 2.7–3.8 시간),
#      같이 돌던 p 0.2 GRASP 8 장도 느려짐. GPU 계산이 이미 차 있어 나눠 쓰기만 한다. → 8 장 · 원래 batch 로 하나씩 (InfLevel 먼저, 그다음 GRASP).
#   ⚠️ 00:3x: InfLevel 한 판은 2 분 중 절반이 모델 · 데이터 올리기라 GPU 가 논다 [측정: 전력 85–90 W / 490 W 인 GPU 가 다수] → InfLevel 만 2 장씩 4 개 동시에.
#      GRASP 는 계산이 꽉 차므로 8 장으로 하나씩, batch 만 키운다 (OOM 이면 반으로).
#   표 다시 계산 → AR_FULL_FILL_DONE (chain_ar_full_fill_20261109.log 에)
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
ALL="cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7"
say "## 남은 InfLevel 을 4 개씩 동시에 (지금 도는 채점 끝 기다림)"
while pgrep -f "z_research/scripts/ru[n].sh intphys1_sliding" > /dev/null; do sleep 10; done
i=0
for run in $P2 $L1; do for s in continuity solidity gravity; do for w in 16 32; do
  OUT=z_training/runs/$run/eval/intphys1_sliding__inflevel_${s}_latest_w$w
  [[ -f $OUT/summary.json ]] && continue
  pair=$(( i % 4 )); job $run inflevel_$s $w "cuda:$((2 * pair)) cuda:$((2 * pair + 1))" $((29800 + i)) 72 8 &
  i=$((i + 1)); (( i % 4 == 0 )) && wait
done; done; done
wait
say "## L1 GRASP (8 장, batch 216)"
for w in 16 32; do job $L1 grasp_level2 $w "$ALL" 29700 216 24; done
say "## 표"
/venv/vjepa/bin/python z_research/scripts/analysis/ar_full_table.py --json z_analysis/results/ar_full_table_20261109.json | tail -7 | tee -a "$LOG"
say "AR_FULL_FILL_DONE"
