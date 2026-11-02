#!/bin/bash
# BlenderTrack v3t (v3 + trapdoor) 여러 노드 동시 렌더 (claim) → 네 팔 합성 (2026-11-01, 사용자: multi node 로 최대한 빠르게).
#   설계 · 고르기가 끝난 뒤:  bash z_research/scripts/data/blender_track/run_v3t_multi.sh
#   NODES 표 = "노드 GPU 작업자/GPU [의존 잡]" — 그 시각 빈 GPU 에 맞춘다 (vll7 은 최종 채점 잡이 끝난 뒤).
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
CFGY=$HERE/configs/v3t.yaml
O=$($P -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['out'])" "$CFGY")
$P -c "import yaml,json,sys; json.dump(yaml.safe_load(open(sys.argv[1])), open(sys.argv[2],'w'), indent=1)" "$CFGY" "$O/config.json"
cp "$CFGY" "$O/config.yaml"
rm -rf "$O/locks"
NODES=(${V3T_NODES:-"vll4 5 6" "vll3 4 6" "vll1 2 2" "vll7 6 7 afterany:237467"})
NWT=0; for row in "${NODES[@]}"; do read -r n g w _ <<<"$row"; NWT=$((NWT + g * w)); done
off=0; deps=""
for row in "${NODES[@]}"; do
  read -r n g w dep <<<"$row"
  j=$(sbatch --parsable -p batch_vll -w "$n" --gres=gpu:"$g" -c $((g * w * 2 + 4)) --mem=$((g * w * 4 + 16))G -t 0-06:00:00 -J v3t_render \
       ${dep:+--dependency=$dep} -o "$O/logs/%j_render_$n.out" \
       --export=ALL,CFG="$O/config.json",SPLITS="probe train test",NODE_I=0,N_NODES=1,GPUS="$g",PER_GPU="$w",THREADS=2,W_OFF="$off",NW_TOTAL="$NWT",CLAIM=1 \
       "$HERE/gen_render.sh")
  echo "  $n GPU $g × 작업자 $w (전역 $off–$((off + g * w - 1)) / $NWT) ${dep:+[$dep]} → $j"
  deps="$deps:$j"; off=$((off + g * w))
done
jc=$(sbatch --parsable -p batch_vll -w vll7 --dependency=afterany$deps -c 64 --mem=96G -t 0-02:00:00 -J v3t_compose -o "$O/logs/%j_compose.out" \
     --wrap "cd $HERE && $P compose_arms_v3.py --config $CFGY --split train --workers 64 && $P compose_arms_v3.py --config $CFGY --split test --workers 64 && echo V3T_DONE")
echo "  합성 → $jc"
