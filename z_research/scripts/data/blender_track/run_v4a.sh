#!/bin/bash
# BlenderTrack v4a · v4b 렌더 (2026-11-05 · 11-06) — 분할 하나를 잠금으로 나눠 그린다 (+ 번호 목록 파일이 있으면 그 번호 먼저).
#   bash run_v4a.sh <root> [번호 목록 파일]                       v4a: CFG 기본 <root>/config.json · SPLIT train_fam (분할 크기 0 → 목록 번호만)
#   CFG=<root>/config_v4b.json SPLIT=probe bash run_v4a.sh <root> <root>/extra_static_probe.txt      v4b: 분할 전체 + 정지 단계 번호
#   GPU_LIST="0,...,7" (기본 8 장) · PER_GPU=8.  멈추기: kill -- -<PID>
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); source "$HERE/env.sh"
ROOT=${1:?root}; LIST=${2:-$ROOT/extra_train_fam.txt}
CFG=${CFG:-$ROOT/config.json}; SPLIT=${SPLIT:-train_fam}
GPU_LIST=${GPU_LIST:-0,1,2,3,4,5,6,7}; PER_GPU=${PER_GPU:-8}
IFS=, read -r -a GL <<<"$GPU_LIST"; NW=$(( ${#GL[@]} * PER_GPU ))
mkdir -p "$ROOT/logs"; rm -rf "$ROOT/locks"
EX=""; [[ -f $LIST ]] && EX="--extra_list $LIST"
echo "렌더 $(TZ=Asia/Seoul date '+%F %T KST') | root $ROOT | 설정 $CFG | 분할 $SPLIT | 목록 ${EX:-없음} | GPU [$GPU_LIST] × $PER_GPU = $NW"
for ((w = 0; w < NW; w++)); do
  g=${GL[$((w % ${#GL[@]}))]}
  CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=2 $BT_BLENDER_PY "$HERE/gen_render.py" --config "$CFG" --split "$SPLIT" \
    --worker $w --nworkers $NW --claim $EX > "$ROOT/logs/${SPLIT}_w$w.log" 2>&1 &
done
wait
n=$(ls "$ROOT"/clips/*.mp4 2>/dev/null | wc -l)
echo "V4A_RENDER_DONE $(TZ=Asia/Seoul date '+%F %T KST') | 분할 $SPLIT | 영상 (폴더 전체) $n | 오류 로그 $(grep -l Traceback "$ROOT"/logs/${SPLIT}_w*.log 2>/dev/null | wc -l)"
