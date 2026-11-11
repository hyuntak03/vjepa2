#!/bin/bash
# MPS 렌더 감시 (2026-11-09 20:0x). 왜 [측정]: MPS 는 GPU 하나의 작업자들이 한 문맥을 나눠 써서, 작업자 하나가 CUDA illegal memory access (700) 로 죽으면
#   그 GPU 의 8 개가 같이 죽는다 (GPU 7, 1 차 렌더 중 8 개 동시 종료). 그래서 2 분마다 GPU 별 작업자 수를 보고, 0 이면 그 GPU 의 MPS 데몬을 다시 띄우고 작업자 8 개를 다시 붙인다.
#   run_v5_mps.sh 의 1 차가 끝나면 ("1 차 끝") 멈춘다 (2 차는 그 스크립트가 직접 돈다). 작업자 번호 · CPU 묶음은 run_v5_mps.sh 와 같다.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); source "$HERE/env.sh"
R=${R:-$BT_ROOT/v5id}
ts() { TZ=Asia/Seoul date '+%F %T KST'; }
n_on() { local g=$1 c=0; for p in $(pgrep -f "gen_rend[e]r.py"); do tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -qx "CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_render$g" && c=$((c + 1)); done; echo $c; }
run() { local g=$1 cfg=$2 split=$3 w=$4 nw=$5 ex=$6 slot=$7 tag=$8; local c0=$(( slot * 4 ))
  CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_render$g CUDA_MPS_LOG_DIRECTORY=/tmp/mps_render${g}_log OMP_NUM_THREADS=2 taskset -c $c0-$((c0 + 3)) \
    $BT_BLENDER_PY "$HERE/gen_render.py" --config $cfg --split $split --worker $w --nworkers $nw --claim $ex > $R/logs/${split}_${tag}$w.log 2>&1 & }
k=0
while ! grep -q "1 차 끝" $R/logs/run_v5_mps.out 2>/dev/null; do
  for g in 0 1 2 3 4 5 6 7; do
    if [[ $(n_on $g) -eq 0 ]]; then
      k=$((k + 1)); tag=w$k
      CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_render$g sh -c 'echo quit | nvidia-cuda-mps-control' 2>/dev/null; sleep 5
      rm -f /tmp/mps_render$g/control; mkdir -p /tmp/mps_render$g /tmp/mps_render${g}_log
      CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_render$g CUDA_MPS_LOG_DIRECTORY=/tmp/mps_render${g}_log CUDA_VISIBLE_DEVICES=$g nvidia-cuda-mps-control -d
      sleep 2
      for j in 0 1 2; do w=$((g + 8 * j)); run $g $R/config_v5id.json train_fam $w 24 "" $w $tag; run $g $R/config_v5idr.json rtrain_fam $w 24 "" $((24 + w)) $tag; done
      for j in 0 1; do w=$((g + 8 * j)); run $g $R/config_v5c.json slide $w 16 "--extra_list $R/extra_slide.txt" $((48 + w)) $tag; done
      echo "GPU $g 작업자 0 → MPS 다시 · 작업자 8 (로그 꼬리표 $tag) $(ts)"
      sleep 60
    fi
  done
  sleep 120
done
echo "감시 끝 $(ts)"
