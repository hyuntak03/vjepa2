#!/bin/bash
# v3 ablation 최종 채점 (2026-11-01) — 학습이 다 끝난 뒤 vll7 GPU 6 장에서 모든 팔 + release + VOBench PFT + 복사를 한 노드 · 한 통과로.
#   왜: 중간 채점은 vll1 (vjepa2) 과 vll7 (Blackwell, vjepa2_bw) 이 섞여 surprise 가 상대 ~2e-4 달라 짝 몇 개가 뒤집힌다.
#       최종 표는 한 노드 값으로 맞춘다 (table_v3.py 는 tag · 복사 모두 마지막 배치 값을 쓴다).
#   v3 시험 = GPU 0–3, 외부 다섯 패널 = GPU 4–5 를 동시에.
#   sbatch -p batch_vll -w vll7 --gres=gpu:6 -c 48 --mem=200G --dependency=afterany:<학습 잡들> \
#          -o predictor_lab/v3_ablation/logs/%j_final.out predictor_lab/v3_ablation/final_eval.sh
set -uo pipefail
R=/data/hyuntak/project/2026/2027_cvpr/vjepa2
D=$R/predictor_lab/v3_ablation
export PY=/data/hyuntak/anaconda3/envs/vjepa2_bw/bin/python
PF=$D/logs/preds_final.txt
{ echo "release=$R/z_training/runs/release_vith/latest.pt"; echo "vobftg=$R/z_training/runs/vobench_release_postft_g/latest.pt"
  grep -v '^#' "$D/runs.tsv" | cut -f1 | grep . | while read -r r; do
    if [[ -f $R/z_training/runs/$r/latest.pt && -f $D/queue/$r.done ]]; then echo "$r=$R/z_training/runs/$r/latest.pt"; else echo "빠짐: $r" >&2; fi
  done; } > "$PF"
echo "[final] $(hostname) $(date '+%F %T') predictor $(wc -l < "$PF") 개"
CUDA_VISIBLE_DEVICES=0,1,2,3 PANELS=v3test PREDS_FILE=$PF OUTSFX=final GPUS=4 bash "$D/eval_v3.sh" > "$D/logs/final_v3test.out" 2>&1 &
CUDA_VISIBLE_DEVICES=4,5 PANELS="vob_late ip1uni ip2uni clevrer rubric vomo" PREDS_FILE=$PF OUTSFX=final GPUS=2 bash "$D/eval_v3.sh" > "$D/logs/final_ext.out" 2>&1 &
wait
"$PY" "$D/table_v3.py"
echo FINAL_DONE
