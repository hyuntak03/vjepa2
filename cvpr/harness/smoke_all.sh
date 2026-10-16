#!/usr/bin/env bash
# =============================================================================
# cvpr/harness/smoke_all.sh — 다섯 엔진 경로의 **배관 점검** (1 GPU, 몇 클립씩, 10~20 분).
#
#   GPUS=1 bash cvpr/harness/smoke_all.sh                       # 지금 노드에서
#   GPUS=1 TIME=0-01:00:00 bash cvpr/harness/submit.sh cvpr/harness/smoke_all.sh   # sbatch
#   ONLY="testbed intphys1" bash cvpr/harness/smoke_all.sh      # 일부만
#
# 각 항목은 LIMIT/SMOKE 로 작게 돌리고 결과는 _smoke 접미사 폴더에 쓴다 (본 결과·토큰 캐시를 안 덮는다).
# 수치를 보는 시험이 아니다 — 끝까지 돌아서 summary.json 이 생기는지만 본다.
# =============================================================================
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); source "$HERE/../env.sh"
export GPUS=${GPUS:-1}
ONLY=${ONLY:-"testbed intphys1 intphys2 probing ek100"}
declare -A RC OUTD
run_one() {   # 이름, 기대 산출물, 명령...
  local name=$1 want=$2; shift 2
  echo; echo "################ smoke: $name  ($(date '+%H:%M:%S'))"; echo "+ $*"
  local t0=$SECONDS
  "$@"; RC[$name]=$?
  echo "################ $name rc=${RC[$name]}  ($((SECONDS - t0)) s)"
  OUTD[$name]=$(ls -dt "$want" 2>/dev/null | head -1)
}
for t in $ONLY; do case $t in
  testbed)  run_one testbed  "$CVPR_RESULTS/eval/testbed/v11_vith_smoke2/summary.json" \
              env LIMIT=2 bash "$CVPR_ROOT/eval/testbed/run.sh" v11 vith ;;
  intphys1) run_one intphys1 "$CVPR_RESULTS/eval/intphys1/intphys1_dev_vith_wgarrido_skip2_w32_smoke2/summary.json" \
              env LIMIT=2 WINDOWS=garrido_skip2_w32 bash "$CVPR_ROOT/eval/intphys1/run.sh" intphys1_dev vith ;;
  intphys2) run_one intphys2 "$CVPR_RESULTS/eval/intphys2/intphys2_main_vith_wip2_w16_smoke4/summary.json" \
              env LIMIT=4 WINDOWS=ip2_w16 bash "$CVPR_ROOT/eval/intphys2/run.sh" intphys2_main vith ;;
  # probing: LIMIT 으로 block 을 줄이면 조건별 head (auto_conditions) 의 학습셋이 비어 엔진이 죽는다 (의도된 검사) → pooled head 하나만
  probing)  run_one probing  "$CVPR_RESULTS/analysis/probing/attn_probe__v11_vith_smoke12/summary.json" \
              env LIMIT=12 SET="probing.fit_groups_sweep=[null] probing.optims.attn_30.num_epochs=1 probing.targets.color=null probing.targets.env=null" \
              bash "$CVPR_ROOT/analysis/probing/run.sh" v11 vith ;;
  ek100)    run_one ek100    "$CVPR_RESULTS/eval/ek100/action_anticipation_frozen/ek100_vith_smoke/val_metrics.jsonl" \
              env SMOKE=1 bash "$CVPR_ROOT/eval/ek100/run.sh" ek100 vith ;;
  *) echo "모르는 항목 $t" ;;
esac; done
echo; echo "================ smoke 요약 ($(date '+%H:%M:%S'), $(hostname)) ================"
fail=0
for t in $ONLY; do
  rc=${RC[$t]:-?}; out=${OUTD[$t]:-}
  if [[ $rc == 0 && -n $out ]]; then echo "  PASS  $t   $out"; else echo "  FAIL  $t   rc=$rc  산출물=${out:-없음}"; fail=1; fi
done
exit $fail
