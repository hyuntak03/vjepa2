#!/bin/bash
# v5id 사슬 뒤 (2026-11-09): 미끄러지는 가림막 설계 (v5c_slide.py, CPU) → v5id 렌더 끝 (V5ID_RENDER_DONE) 기다림 → v5c 렌더 → 학습 목록 (index_v5id.py).
#   R=<root> setsid nohup bash run_v5c_after.sh > <root>/logs/run_v5c_after.out 2>&1 &
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); source "$HERE/env.sh"
R=${R:-$BT_ROOT/v5id}; P=$BT_PY; cd "$HERE"
ts() { TZ=Asia/Seoul date '+%F %T KST'; }
echo "## v5c 설계 $(ts)"
$P v5c_slide.py design --src $R --n ${V5C_N:-5000} | tail -1
echo "V5C_DESIGN_DONE $(ts)"
until grep -q V5ID_RENDER_DONE $R/logs/run_v5id.out 2>/dev/null; do
  if ! tmux has-session -t "=render_v5id" 2>/dev/null; then echo "⚠️ render_v5id 세션이 V5ID_RENDER_DONE 없이 끝났다 $(ts)"; exit 1; fi
  sleep 120; done
echo "## v5c 렌더 $(ts)"
CFG=$R/config_v5c.json SPLIT=slide bash run_v4a.sh $R $R/extra_slide.txt
echo "## 목록 $(ts)"
$P index_v5id.py --root $R --name blender_track_v5id
echo "V5_ALL_DONE $(ts)"
