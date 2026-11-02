#!/bin/bash
# CLEVRER 간격 시험 (2026-11-01): 같은 장면 · 같은 문맥 끝에서 샘플 간격만 0.08 → 0.16 초. 세 창 × predictor 셋 (+ 복사) 을 한 번 통과 채점기로.
#   창: s2_c16t32 (원 창, PNG 재현 확인) · s2_c8t16 (0.08 초) · s4_c8t16 (0.16 초). 프레임 = build_clevrer_interval.py
#   sbatch -p batch_vll --exclude=vll7 --gres=gpu:2 -c 16 --mem=64G -t 2:00:00 -o z_analysis/release/08_rubric_transfer/logs/%j_interval.out z_analysis/release/08_rubric_transfer/run_interval.sh
set -euo pipefail
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
R=$PWD/z_training/runs
PREDS="release=$R/release_vith/latest.pt,vobftg=$R/vobench_release_postft_g/latest.pt,v3_full_s0=$R/v3_full_s0/latest.pt"
for ARM in ${ARMS:-s2_c16t32 s2_c8t16 s4_c8t16}; do
  W=()
  [ "$ARM" != s2_c16t32 ] && W=(window.n_frames=16 window.context=8)
  echo "=== $ARM $(date)"
  $P z_research/scripts/analysis/transfer_gap_te.py --name "clevrer_interval_$ARM" --dataset vobench_v1 \
    --set data.root=$PWD/data_csv/clevrer_interval_$ARM data.index_csv=index.csv \
          data.frames_root=/data/hyuntak/project/2026/2027_cvpr/nfs_cache/clevrer_interval/$ARM \
          window.stride=1 window.frames_start=0 "${W[@]}" \
    --preds "$PREDS" --out $PWD/z_analysis/release/08_rubric_transfer/te/clevrer_$ARM --gpus 2 --batch 8 --workers 6 --threads 4
done
echo "=== done $(date)"
