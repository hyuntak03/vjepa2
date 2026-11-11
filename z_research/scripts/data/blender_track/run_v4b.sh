#!/bin/bash
# BlenderTrack v4b (긴 가림, 2026-11-06) 전체 사슬: 설계 · 고르기 (평면 train_fam · 빗면 rtrain_fam) → 평면 trapdoor 정지 단계 → 렌더 → 학습 목록 · 짝 정보.
#   bash run_v4b.sh            (백그라운드로 돌리려면 setsid nohup bash run_v4b.sh > <root>/logs/v4b_full.out 2>&1 &)
#   설정 configs/v4b.yaml · v4b_ramp.yaml, 자리 BT_OUT (기본 <BT_ROOT>/v4b_long_hide), 목록 data_csv/blender_track_v4b
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); source "$HERE/env.sh"
export BT_OUT=${BT_OUT:-$BT_ROOT/v4b_long_hide}
R=$BT_OUT; P=$BT_PY; cd "$HERE"; mkdir -p "$R/logs"
ts() { TZ=Asia/Seoul date '+%F %T KST'; }
for c in v4b.yaml v4b_ramp.yaml; do $P localize_cfg.py configs/$c --write 'config_{name}.yaml' >/dev/null; done
for n in v4b v4br; do $P -c "import yaml,json,sys; json.dump(yaml.safe_load(open(sys.argv[1])), open(sys.argv[2],'w'), indent=1)" $R/config_$n.yaml $R/config_$n.json; done
echo "## 설계 $(ts)"
$P design_families_v4vc.py --config $R/config_v4b.yaml --split train_fam --procs ${DESIGN_PROCS:-200} > $R/logs/design_train_fam.log 2>&1
$P design_families_v4vc.py --config $R/config_v4b.yaml --split train_fam --select | tail -16
$P design_families_v4vc.py --config $R/config_v4br.yaml --split rtrain_fam --procs ${DESIGN_PROCS:-200} > $R/logs/design_rtrain_fam.log 2>&1
$P design_families_v4vc.py --config $R/config_v4br.yaml --split rtrain_fam --select | tail -14
$P v4a_static.py design_root --out $R --cfg $R/config_v4b.yaml --split train_fam
echo "DESIGN_DONE $(ts)"
CFG=$R/config_v4b.json SPLIT=train_fam bash run_v4a.sh $R $R/extra_static_train_fam.txt
CFG=$R/config_v4br.json SPLIT=rtrain_fam bash run_v4a.sh $R /nonexistent
echo "## 목록 $(ts)"
$P v4a_static.py index_root --out $R --cfg $R/config_v4b.yaml --split train_fam --cfg-ramp $R/config_v4br.yaml --split-ramp rtrain_fam --name blender_track_v4b
echo "V4B_DONE $(ts)"
