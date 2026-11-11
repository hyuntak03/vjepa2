#!/bin/bash
# run_v4bc4x.sh 1 차 렌더 동안 GPU 마다 작업자 수를 지킨다 (2026-11-11). MPS 연쇄 종료가 같은 GPU 에서 거듭되면 그 GPU 의 작업자가 세 분할을 다 써서 일찍 끝난다 → 빈 자리를 새 작업자로 채운다.
#   남은 일 (clips + locks < 목표 − 3000) 일 때만 채우고, "1 차 끝" 이 찍히면 멈춘다 (2 차 · 정리와 겹치지 않게).
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); source "$HERE/env.sh"
R=$BT_ROOT/v4bc4x; RC=$BT_ROOT/v4cx_slide; PER=11; TARGET=${TARGET:-210000}; nid=100
one() { case $1 in
  train_fam)  echo "--config $R/config_v4bx.json --split train_fam --extra_list $R/extra_static_train_fam.txt" ;;
  rtrain_fam) echo "--config $R/config_v4bxr.json --split rtrain_fam" ;;
  slide)      echo "--config $RC/config_v4c.json --split slide --extra_list $RC/extra_slide.txt" ;; esac; }
spawn() { local g=$1 w=$2; local c0=$(( (w * 37) % 254 ))
  ( for s in train_fam rtrain_fam slide; do
      CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_render$g CUDA_MPS_LOG_DIRECTORY=/tmp/mps_render${g}_log OMP_NUM_THREADS=2 taskset -c $c0-$((c0 + 2)) \
        $BT_BLENDER_PY "$HERE/gen_render.py" $(one $s) --worker $w --nworkers 88 --claim >> $R/logs/render_k_w$w.log 2>&1
    done ) & }
while ! grep -q "1 차 끝" $R/logs/run_v4bc4x.out; do
  sleep 30
  done_n=$(( $(ls $R/clips | wc -l) + $(ls $RC/clips 2>/dev/null | wc -l) ))   # 08:2x 고침: 잠금은 설계 실패 번호에도 생겨서 세면 안 된다
  [ $done_n -gt $(( TARGET - 3000 )) ] && continue
  declare -A cnt=()
  for p in $(pgrep -f "gen_rend[e]r.py"); do g=$(tr '\0' '\n' < /proc/$p/environ 2>/dev/null | sed -n 's#^CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_render##p'); [ -n "$g" ] && cnt[$g]=$(( ${cnt[$g]:-0} + 1 )); done
  for g in 0 1 2 3 4 5 6 7; do
    miss=$(( PER - ${cnt[$g]:-0} ))
    if [ $miss -gt 0 ]; then echo "$(TZ=Asia/Seoul date +%T) GPU $g 작업자 ${cnt[$g]:-0} → $miss 개 채움"; for ((i = 0; i < miss; i++)); do spawn $g $nid; nid=$((nid + 1)); done; fi
  done
  unset cnt
done
echo "$(TZ=Asia/Seoul date +%T) 1 차 끝 — 채우기 멈춤"
