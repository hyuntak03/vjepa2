#!/bin/bash
# v3 ablation 자동 채점 (2026-11-01) — 학습이 끝난 팔 (queue/<run>.done) 중 아직 채점 안 한 것을 모아 vll1 에 eval_v3.sh 잡 두 개로 낸다.
#   v3 시험 (GPU 4) · 외부 다섯 패널 (GPU 2). 낸 팔은 evaluated/<run> 으로 표시한다. 20 팔이 다 표시되면 끝난다.
#   띄우기 (세션과 분리): setsid nohup bash predictor_lab/v3_ablation/auto_eval.sh > predictor_lab/v3_ablation/logs/auto_eval.out 2>&1 &
#   (11-01 사용자: 채점은 vll1 에서. 학습은 vll6 · vll7)
set -uo pipefail
R=/data/hyuntak/project/2026/2027_cvpr/vjepa2
cd "$R"
D=$R/predictor_lab/v3_ablation; Q=$D/queue; E=$D/evaluated
mkdir -p "$E" "$D/logs"
NRUN=$(grep -v '^#' "$D/runs.tsv" | grep -c .)
while :; do
  new=()
  while IFS=$'\t' read -r run _; do
    [[ -z $run || $run == \#* ]] && continue
    [[ -f $Q/$run.done && ! -e $E/$run && -f $R/z_training/runs/$run/latest.pt ]] && new+=("$run")
  done < "$D/runs.tsv"
  if (( ${#new[@]} )); then
    sfx=b$(date +%H%M)
    PF=$D/logs/preds_${sfx}.txt                     # 쉼표 목록은 --export 에서 잘리므로 파일로 넘긴다
    for r in "${new[@]}"; do echo "$r=$R/z_training/runs/$r/latest.pt"; done > "$PF"
    j1=$(sbatch --parsable -p batch_vll -w vll1 --gres=gpu:4 -c 24 --mem=96G -t 0-04:00:00 -J v3eval_${sfx}_v3test -o "$D/logs/%j_${sfx}_v3test.out" \
         --export=ALL,PANELS=v3test,PREDS_FILE="$PF",OUTSFX=$sfx,GPUS=4 "$D/eval_v3.sh")
    j2=$(sbatch --parsable -p batch_vll -w vll1 --gres=gpu:2 -c 16 --mem=64G -t 0-04:00:00 -J v3eval_${sfx}_ext -o "$D/logs/%j_${sfx}_ext.out" \
         --export=ALL,PANELS="vob_late ip1uni ip2uni clevrer rubric",PREDS_FILE="$PF",OUTSFX=$sfx,GPUS=2 "$D/eval_v3.sh")
    for r in "${new[@]}"; do echo "$sfx $j1 $j2" > "$E/$r"; done
    echo "$(date '+%F %T') 배치 $sfx: ${new[*]} → $j1 $j2"
  fi
  ndone=$(ls "$E" | wc -l)
  (( ndone >= NRUN )) && { echo "$(date '+%F %T') 모든 팔 채점 제출 끝 ($ndone)"; break; }
  sleep 300
done
