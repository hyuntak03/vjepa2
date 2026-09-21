#!/bin/bash
# -----------------------------------------------------------------------------
# 학습 데이터 × 채점 bench 전이 행렬의 빈 칸을 채운다 (2026-09-19). 결과가 있는 칸은 건너뛴다.
#
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 GPUS=7 bash z_training/eval_matrix.sh
#   python z_training/harness/matrix.py            # 표로 모으기 -> z_training/runs/_matrix/MATRIX.md
#
# 행 (predictor, 마지막 epoch): 릴리즈 / v11_postft e10 / intphys1_postft e40 / predictor_v1_postft e15 / intphys2_postft e80
# 열 (bench 별 채점 protocol):
#   v11 test        surprise_c16t32 matched pair, v11_split_test (10,752 쌍)     z_training/eval.sh <run> v11_split_test
#   IntPhys1 dev    intphys1_sliding (Garrido 공식 sliding, 180 쌍)              z_training/eval.sh <run> intphys1_dev intphys1_sliding
#   IntPhys2 test   analysis/intphys2 growing-context sweep, Main 장면 split test 126 장면 (252 쌍)
#                   analysis/intphys2/configs/vjepa2_vith_intphys2_maintest_<run>_<ep>.yaml
#   Predictor_v1    holdout 예측 L1 (가능 영상뿐 — 짝 없음). z_research/scripts/analysis/predictor_holdout_l1.py (따로 돈다)
# -----------------------------------------------------------------------------
set -uo pipefail
PROJ=/data/hyuntak/project/2026/2027_cvpr/vjepa2; cd "$PROJ"
PY=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
G=${GPUS:-8}
LOG=z_training/runs/_matrix/eval_matrix.log; mkdir -p z_training/runs/_matrix
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

wma() {   # run ep dataset protocol
  local run=$1 ep=$2 ds=$3 proto=$4
  local out=z_training/runs/$run/eval/${proto}__${ds}_${ep}
  if [[ -f $out/summary.json ]]; then say "skip  $run $ep $ds ($proto) — 있음"; return; fi
  say "start $run $ep $ds ($proto)"
  CKPT=$ep.pt GPUS=$G bash z_training/eval.sh "$run" "$ds" "$proto" >> "$LOG" 2>&1 && say "done  $run $ep $ds" || say "FAIL  $run $ep $ds (exit $?)"
}
ip2() {   # run ep
  local run=$1 ep=$2 cfg=analysis/intphys2/configs/vjepa2_vith_intphys2_maintest_${1}_${2}.yaml
  [[ $run == intphys2_postft ]] && cfg=analysis/intphys2/configs/vjepa2_vith_intphys2_maintest_postft.yaml
  local tag; tag=$(grep -m1 "^tag:" "$cfg" | awk '{print $2}')
  if [[ -f z_exp/intphys2/$tag/summary.json ]]; then say "skip  $run $ep IntPhys2 MainTest — 있음"; return; fi
  say "start $run $ep IntPhys2 MainTest ($cfg)"
  $PY -m torch.distributed.run --standalone --nproc-per-node="$G" -m analysis.intphys2.eval --config "$cfg" > "z_exp/intphys2/$tag.log" 2>&1 \
    && $PY z_research/scripts/analysis/intphys2_column_best.py --run "z_exp/intphys2/$tag" > /dev/null && say "done  $run $ep IntPhys2" || say "FAIL  $run $ep IntPhys2 (exit $?)"
}

say "===== eval_matrix 시작 (GPUS=$G, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}) ====="
ip2 v11_postft e10
ip2 intphys1_postft e40
ip2 predictor_v1_postft e15
wma intphys2_postft e80 intphys1_dev intphys1_sliding
wma predictor_v1_postft e15 v11_split_test surprise_c16t32
wma intphys2_postft e80 v11_split_test surprise_c16t32
say "===== eval_matrix 끝 ====="
