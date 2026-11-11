#!/bin/bash
# v5id 렌더 한꺼번에 (2026-11-09 18:3x, 사용자 "너무 오래 걸리는데 다 동시에 띄우면 안돼?"): 평면 · 빗면 · 미끄러지는 가림막을 동시에.
#   왜 빨라지나 [측정]: 작업자 64 (GPU 당 8) 일 때 GPU 이용률 99 % 인데 전력은 145 / 490 W, 메모리 33 / 98 GB — 장 단위 고정 비용이 커서 GPU 가 덜 찬다.
#   ⚠️ 정정 (18:4x): GPU 당 16 (128 작업자) 은 처리량이 그대로 (분당 260 vs 270) 이고 Blender 프로세스마다 스레드 약 700 개라 컨테이너 pids 한도
#   (62,720) 에 닿아 작업자가 죽었다 (pthread_create 실패). → GPU 당 8 (평면 3 · 빗면 3 · 가림막 2) 로 세 부분을 나란히, 작업자마다 CPU 4 개로 묶어 (taskset) 스레드를 줄인다.
#   잠금 (--claim) 이라 나중에 작업자를 더 붙여도 된다 (잠금 폴더는 지우지 말 것).
#   R=<root> setsid nohup bash run_v5_all.sh > <root>/logs/run_v5_all.out 2>&1 &
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); source "$HERE/env.sh"
R=${R:-$BT_ROOT/v5id}; cd "$HERE"; mkdir -p "$R/logs" "$R/locks"
ts() { TZ=Asia/Seoul date '+%F %T KST'; }
GL=(0 1 2 3 4 5 6 7); SLOT=0
launch() {  # cfg split per_gpu extra tag
  local cfg=$1 split=$2 per=$3 ex=$4 tag=$5 nw=$(( 8 * $3 ))
  for ((w = 0; w < nw; w++)); do
    c0=$(( (SLOT % 64) * 4 )); SLOT=$((SLOT + 1))
    CUDA_VISIBLE_DEVICES=${GL[$((w % 8))]} OMP_NUM_THREADS=2 taskset -c $c0-$((c0 + 3)) $BT_BLENDER_PY "$HERE/gen_render.py" --config "$cfg" --split "$split" \
      --worker $w --nworkers $nw --claim $ex > "$R/logs/${split}_${tag}$w.log" 2>&1 &
  done
}
echo "렌더 한꺼번에 $(ts) | 평면 ${PF:=3} · 빗면 ${PR:=3} · 가림막 ${PS:=2} (GPU 당)"
launch $R/config_v5id.json train_fam $PF "" a
launch $R/config_v5idr.json rtrain_fam $PR "" a
launch $R/config_v5c.json slide $PS "--extra_list $R/extra_slide.txt" a
wait
echo "V5_RENDER_ALL_DONE $(ts) | 평면 $(ls $R/clips | grep -c '^train_fam') · 빗면 $(ls $R/clips | grep -c '^rtrain_fam') · 가림막 $(ls $R/clips | grep -c '^slide') | 오류 로그 $(grep -l Traceback $R/logs/*_a*.log 2>/dev/null | wc -l)"
$BT_PY index_v5id.py --root $R --name blender_track_v5id
echo "V5_ALL_DONE $(ts)"
