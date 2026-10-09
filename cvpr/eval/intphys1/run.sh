#!/usr/bin/env bash
# =============================================================================
# eval/intphys1 — IntPhys 1 프로토콜 (Garrido A.8 sliding) 을 IntPhys1 · GRASP · InfLevel 에.
#
#   GPUS=8 bash cvpr/eval/intphys1/run.sh                                  # 전부: 5 벤치 x vith x 창 {16,32}
#   GPUS=8 bash cvpr/eval/intphys1/run.sh intphys1_dev vith                # 하나
#   GPUS=8 bash cvpr/eval/intphys1/run.sh "intphys1_dev grasp_level2" "vith vjepa21g videomae2g"
#   WINDOWS="garrido_w32" ...                                              # 창 하나만 (preset 이름, cvpr/registry/windows.yaml)
#   WINDOWS="garrido_skip2_w32" ...                                        # 88.89 칸만
#   GPUS=4 bash cvpr/eval/intphys1/run.sh intphys1_dev dinof_highres       # DINO-Foresight: 창 {5,8,16} (C=4 고정) 자동
#   COPY=1 GPUS=4 bash cvpr/eval/intphys1/run.sh intphys1_dev dinof_highres   # 복사 기준선 (이름에 _copy)
#   DRYRUN=1 bash cvpr/eval/intphys1/run.sh intphys1_dev vith              # 병합 config 만
#   RESULTS_ROOT=legacy ...   # 옛 폴더·이름 (z_research/Benchmarks/exp_results/intphys1_sliding__<bench>_<model>_w<N>) 그대로
#
# 규칙 (PROTOCOLS.md)
#   · 창(C+M) 마다 **실행을 나눈다** — 공식은 창마다 모델을 그 프레임 수로 짓는다. 한 프로세스에 두 창을 섞지 말 것
#     (VideoMAEv2 는 즉시 죽고, V-JEPA 는 RoPE 라 돌지만 공식과 다른 위치 인코딩이 된다).
#   · 벤치별 frame_skips 는 데이터셋 레지스트리 sliding_frame_skips (intphys1 [2,5,10] · grasp [5,10] 부분 탐색 · inflevel [5,10,20]).
#   · A.8 최고 칸은 z_research/scripts/analysis/garrido_rescore.py 가 창을 가로질러 고른다 (보고값은 descriptive).
#   · 지표는 쌍 비교 AvgSurprise 하나. 복사 기준선 Δ 를 같이 낸다 (README).
# 배치 상한 (VRAM): vjepa21g 는 타깃 5632 차원이라 MB21 (기본 8); videomae2g 창 32 는 MBVM (기본 8). 수치에 영향 없는 상한이다.
# =============================================================================
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); source "$HERE/../../env.sh"
BENCHES=${1:-${BENCHES:-"intphys1_dev grasp_level2 inflevel_continuity inflevel_solidity inflevel_gravity"}}
MODELS=${2:-${MODELS:-vith}}
MB21=${MB21:-8}; MBVM=${MBVM:-8}
# 모델별 기본 창: V-JEPA/VideoMAE 는 {16,32}, DINO-Foresight 는 C=4 가 되는 {5,8,16} (cvpr/registry/windows.yaml dinof_w*)
win_for() { case $1 in dinof_*) echo "${WINDOWS:-dinof_w5 dinof_w8 dinof_w16}" ;; *) echo "${WINDOWS:-garrido_w16 garrido_w32}" ;; esac; }
# COPY=1: 복사 기준선 (predictor 대신 마지막 문맥 특징). dinof 는 로더 플래그 model.dinof_copy=true — V-JEPA 계열은 README 의 copy 절차
copy_set() { case $1 in dinof_*) echo "model.dinof_copy=true" ;; *) echo "" ;; esac; }
fail=0
for b in $BENCHES; do for m in $MODELS; do for w in $(win_for "$m"); do
  S="${SET:-}"; SFX="${SUFFIX:-}"
  if [[ ${COPY:-0} == 1 ]]; then cs=$(copy_set "$m"); [[ -z $cs ]] && { [[ $w == $(win_for "$m" | cut -d" " -f1) ]] && { bash "$HERE/copy_baseline.sh" "$b" "$m" || { echo "!! 실패 copy $b/$m"; fail=1; }; }; continue; }; S="$S $cs"; SFX="${SFX}_copy"; fi   # V-JEPA 계열 → copy_baseline.sh (격자 세 칸을 한 번에 — 첫 창에서만 부른다)
  [[ $m == vjepa21g ]] && S="$S surprise.intphys1.max_batch=$MB21"
  [[ $m == videomae2g && $w == *32* ]] && S="$S surprise.intphys1.max_batch=$MBVM"
  extra=()
  if [[ ${RESULTS_ROOT:-} == legacy ]]; then         # 옛 run_all.sh 이름: <bench>_<model>_w<N>
    n=${w##*_w}; n=${n//[!0-9]/}
    extra=(TAG="${b}_${m}_w${n}${SFX}" OUTDIR="$CVPR_REPO/z_research/Benchmarks/exp_results/intphys1_sliding__${b}_${m}_w${n}${SFX}")
  fi
  echo "### intphys1  $b / $m / $w${SFX}  ($(date '+%H:%M:%S'))"
  env "${extra[@]}" WINDOW="$w" SET="$S" SUFFIX="$SFX" bash "$CVPR_ROOT/harness/launch.sh" "$HERE/protocol.yaml" "$b" "$m" \
    || { echo "!! 실패 $b/$m/$w"; fail=1; }
done; done; done
exit $fail
