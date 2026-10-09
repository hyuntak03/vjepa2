#!/usr/bin/env bash
# =============================================================================
# eval/intphys2 — IntPhys 2 프로토콜 (논문 D.3 격자). 엔진 analysis/intphys2/eval.py (torchrun).
#
#   GPUS=8 bash cvpr/eval/intphys2/run.sh                            # intphys2_main x vith x 창 {16,32,48}
#   GPUS=8 bash cvpr/eval/intphys2/run.sh intphys2_main "vith vjepa21g videomae2g"
#   WINDOWS="ip2_w48" GPUS=8 bash cvpr/eval/intphys2/run.sh intphys2_main vith_pft_intphys2_e80   # 학습한 predictor 로
#   DRYRUN=1 bash cvpr/eval/intphys2/run.sh
#   RESULTS_ROOT=legacy ...   # 옛 폴더 (z_research/Benchmarks/exp_results/intphys2/bench_intphys2_main_<model>_w<N>)
#
# 창: V-JEPA 계열 ip2_w16/w32/w48 (6 fps) · VideoMAEv2 는 창 16 고정 + fps 2/3/6 (ip2_vmae_fps2/3/6) — 논문 D.3.
# 배치 상한 (24 GB 기준, 수치 무관): 창 16/32/48 -> 16/12/8, vjepa21g (타깃 5632 차원) -> 12/8/6.
# =============================================================================
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); source "$HERE/../../env.sh"
DATASETS=${1:-${DATASETS:-intphys2_main}}
MODELS=${2:-${MODELS:-vith}}
win_for()   { [[ $1 == videomae2g ]] && echo "${WINDOWS:-ip2_vmae_fps6 ip2_vmae_fps3 ip2_vmae_fps2}" || echo "${WINDOWS:-ip2_w16 ip2_w32 ip2_w48}"; }
batch_for() {                                   # 모델/창 -> max_window_batch
  case "$1/$2" in
    vjepa21g/*16*) echo 12 ;; vjepa21g/*32*) echo 8 ;; vjepa21g/*48*) echo 6 ;;
    */*16*|*/*fps*) echo 16 ;; */*32*) echo 12 ;; */*48*) echo 8 ;; *) echo 8 ;;
  esac
}
fail=0
for d in $DATASETS; do for m in $MODELS; do for w in $(win_for "$m"); do
  S="${SET:-} surprise.max_window_batch=$(batch_for "$m" "$w")"
  extra=()
  if [[ ${RESULTS_ROOT:-} == legacy ]]; then        # 옛 run_grid.sh 이름
    n=${w##*_w}; [[ $w == *_w[0-9]* ]] || n=${w#ip2_}   # ip2_w16 -> 16 (옛 이름 _w16). ip2_vmae_fps6 -> vmae_fps6 (옛 run_grid.sh 는 videomae 를 w16 로만 돌려 대응 이름이 없다)
    extra=(OUTDIR="$CVPR_REPO/z_research/Benchmarks/exp_results/intphys2/bench_intphys2_main_${m}_w${n}")
  fi
  echo "### intphys2  $d / $m / $w  ($(date '+%H:%M:%S'))"
  env "${extra[@]}" WINDOW="$w" SET="$S" bash "$CVPR_ROOT/harness/launch.sh" "$HERE/protocol.yaml" "$d" "$m" \
    || { echo "!! 실패 $d/$m/$w"; fail=1; }
done; done; done
exit $fail
