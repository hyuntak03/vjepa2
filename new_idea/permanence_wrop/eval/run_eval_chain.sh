#!/usr/bin/env bash
# =============================================================================
# permanence_wrop — 판마다 학습이 끝나면 (e20.pt) 정식 채점, 셋 다 끝나면 WROP held-out L1 (2026-10-31)
#
#   tmux new-session -d -s permwrop_eval "bash new_idea/permanence_wrop/eval/run_eval_chain.sh"
#   ARMS="C D" GPU_IDS="4 5 6 7" bash new_idea/permanence_wrop/eval/run_eval_chain.sh
#
# 채점 세트 (A = 릴리즈 vith 는 이미 있다 — AUX_PERMANENCE_2026-10-31 과 같은 세트 · 같은 쌍):
#   VOBench held-out (split v2 test, 720 쌍) · real_bench (TCOW 실제 영상 40 쌍)   — cvpr/eval/testbed
#   IntPhys1 late (위반 직전이 문맥 끝, 180 쌍) · IntPhys2 violation-uni (500 쌍)   — cvpr/eval/intphys1_late
#   WROP held-out 예측 L1, 계열별 (과제당 20 표본)                                  — eval/wrop_heldout_l1.py
# 같은 쌍 비교 (릴리즈 대비 McNemar): z_research/scripts/analysis/compare_pairs_vs_ref.py
# =============================================================================
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."            # 레포 루트
PY=/venv/vjepa/bin/python
ARMS=${ARMS:-"B C D"}
export GPU_IDS=${GPU_IDS:-"4 5 6 7"}
OUT=new_idea/permanence_wrop/results; mkdir -p "$OUT" new_idea/permanence_wrop/logs
say() { echo "[$(TZ=Asia/Seoul date '+%F %T KST')] $*"; }

for arm in $ARMS; do
  run=z_training/runs/permanence_wrop_$arm; m=vast_permwrop_$arm
  say "대기: $run/e20.pt"
  until [[ -f $run/e20.pt ]]; do sleep 120; done
  sleep 60                                                  # latest.pt 쓰기 마무리
  say "채점 시작 $m"
  bash cvpr/eval/testbed/run.sh "vobench_v1_split_v2_test real_bench_quads" "$m" > new_idea/permanence_wrop/logs/eval_testbed_$arm.log 2>&1 || say "!! testbed 실패 $arm"
  DS=intphys1_late_uni bash cvpr/eval/intphys1_late/run.sh "$m" > new_idea/permanence_wrop/logs/eval_i1late_$arm.log 2>&1 || say "!! intphys1_late 실패 $arm"
  DS=intphys2_violation_uni bash cvpr/eval/intphys1_late/run.sh "$m" > new_idea/permanence_wrop/logs/eval_i2vuni_$arm.log 2>&1 || say "!! intphys2_vuni 실패 $arm"
  {
    echo "=== vobench_v1_split_v2_test"; $PY z_research/scripts/analysis/compare_pairs_vs_ref.py --ds vobench_v1_split_v2_test --ref vith --new $m --index data_csv/vobench_v1_split_v2/index_test.csv
    echo "=== real_bench_quads"; $PY z_research/scripts/analysis/compare_pairs_vs_ref.py --ds real_bench_quads --ref vith --new $m
    echo "=== intphys1_late_uni"; $PY z_research/scripts/analysis/compare_pairs_vs_ref.py --ds intphys1_late_uni --ref vith --new $m --task intphys1_late --group group
    echo "=== intphys2_violation_uni"; $PY z_research/scripts/analysis/compare_pairs_vs_ref.py --ds intphys2_violation_uni --ref vith --new $m --task intphys1_late --group block_type
  } > "$OUT/compare_vs_release_$arm.txt" 2>&1
  say "채점 끝 $m → $OUT/compare_vs_release_$arm.txt"
done

say "WROP held-out L1"
CUDA_VISIBLE_DEVICES=${GPU_IDS%% *} $PY new_idea/permanence_wrop/eval/wrop_heldout_l1.py --per-task 20 \
  --models release $(for arm in B C D; do [[ -f z_training/runs/permanence_wrop_$arm/e20.pt ]] && echo "$arm=z_training/runs/permanence_wrop_$arm/latest.pt"; done) \
  > new_idea/permanence_wrop/logs/eval_wrop_l1.log 2>&1 || say "!! WROP L1 실패"
say "전부 끝"
