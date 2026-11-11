#!/bin/bash
# BlenderTrack v5id 전체 사슬 (2026-11-09): 설계 (평면 · 빗면) → 고르기 → 렌더 (평면 → 빗면). 미끄러지는 가림막 (v5c) 과 목록은 따로.
#   사용자 11-09 "general data 0.2M, shape × color compositional + 정체 위반". 설계기 gen_design_v5id.py, 설정 configs/v5id.yaml · v5id_ramp.yaml.
#   BT_OUT=<root> setsid nohup bash run_v5id.sh > <root>/logs/run_v5id.out 2>&1 &
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); source "$HERE/env.sh"
export BT_OUT=${BT_OUT:-$BT_ROOT/v5id}
R=$BT_OUT; P=$BT_PY; cd "$HERE"; mkdir -p "$R/logs"
ts() { TZ=Asia/Seoul date '+%F %T KST'; }
export BT_V4_DESIGN=gen_design_v5id
for c in v5id.yaml v5id_ramp.yaml; do $P localize_cfg.py configs/$c --write 'config_{name}.yaml' >/dev/null; done
for n in v5id v5idr; do $P -c "import yaml,json,sys; json.dump(yaml.safe_load(open(sys.argv[1])), open(sys.argv[2],'w'), indent=1)" $R/config_$n.yaml $R/config_$n.json; done
echo "## 설계 $(ts)"
$P design_families_v4vc.py --config $R/config_v5id.yaml --split train_fam --procs ${DESIGN_PROCS:-220} > $R/logs/design_train_fam.log 2>&1
$P design_families_v4vc.py --config $R/config_v5id.yaml --split train_fam --select | tail -16
$P design_families_v4vc.py --config $R/config_v5idr.yaml --split rtrain_fam --procs ${DESIGN_PROCS:-220} > $R/logs/design_rtrain_fam.log 2>&1
$P design_families_v4vc.py --config $R/config_v5idr.yaml --split rtrain_fam --select | tail -14
echo "DESIGN_DONE $(ts)"
CFG=$R/config_v5id.json SPLIT=train_fam bash run_v4a.sh $R /nonexistent
CFG=$R/config_v5idr.json SPLIT=rtrain_fam bash run_v4a.sh $R /nonexistent
echo "V5ID_RENDER_DONE $(ts)"
