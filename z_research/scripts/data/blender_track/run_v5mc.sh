#!/bin/bash
# BlenderTrack v5_motion_control — 설계 (노드 셋 CPU 나눠) → 고르기 → 노드 셋 GPU 동시 렌더 (claim) → 묶기 (tar 친화 폴더) (2026-11-02, 사용자: vll3 · 6 · 7 다 써서)
#   bash z_research/scripts/data/blender_track/run_v5mc.sh
#   V5_DESIGN="노드 CPU|…" (설계 잡) · V5_NODES="노드 GPU 작업자/GPU|…" (렌더 잡) — 그 시각 빈 자원에 맞춘다
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
CFGY=$HERE/configs/v5mc.yaml
OUT=$($P -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['out'])" "$CFGY")
mkdir -p "$OUT/logs"
cp "$CFGY" "$OUT/config_v5mc.yaml"
SPL=$($P -c "import yaml,sys; print(' '.join(yaml.safe_load(open(sys.argv[1]))['splits']))" "$CFGY")
SPL_R=$(tr ' ' '\n' <<<"$SPL" | awk '/^p_/{print; next} {r=r" "$0} END{print r}' | xargs)     # 렌더 순서: 확인용 → 학습 → 시험
IFS="|" read -r -a DES <<<"${V5_DESIGN:-vll3 80|vll6 84|vll7 50}"
NP=${#DES[@]}; dj=""
for k in "${!DES[@]}"; do
  read -r n c <<<"${DES[$k]}"
  j=$(sbatch --parsable -p batch_vll -w "$n" -c "$c" --mem=$((c * 2 + 16))G -t 0-03:00:00 -J v5_design -o "$OUT/logs/%j_design_$n.out" \
       --wrap "cd $HERE && for s in $SPL; do $P design_families_v5mc.py --config $CFGY --split \$s --procs $((c - 2)) --part $k --nparts $NP || exit 1; done; echo DESIGN_PART_DONE")
  echo "  설계 $n (몫 $k / $NP, CPU $c) → $j"; dj="$dj:$j"
done
js=$(sbatch --parsable -p batch_vll -w "$(read -r n _ <<<"${DES[0]}"; echo "$n")" -c 4 --mem=16G -t 0-01:00:00 -J v5_select --dependency=afterok$dj \
     -o "$OUT/logs/%j_select.out" --wrap "cd $HERE && $P design_families_v5mc.py --config $CFGY --select-all && rm -rf $OUT/locks && echo SELECT_DONE")
echo "  고르기 → $js"
IFS="|" read -r -a NODES <<<"${V5_NODES:-vll3 7 6|vll6 6 7|vll7 2 12}"
NWT=0; for row in "${NODES[@]}"; do read -r n g w <<<"$row"; NWT=$((NWT + g * w)); done
off=0; deps=""
for row in "${NODES[@]}"; do
  read -r n g w <<<"$row"
  j=$(sbatch --parsable -p batch_vll -w "$n" --gres=gpu:"$g" -c $((g * w * 2 + 4)) --mem=$((g * w * 4 + 16))G -t 0-08:00:00 -J v5_render \
       --dependency=afterok:$js -o "$OUT/logs/%j_render_$n.out" \
       --export=ALL,CFG="$OUT/config_render.json",SPLITS="$SPL_R",NODE_I=0,N_NODES=1,GPUS="$g",PER_GPU="$w",THREADS=2,W_OFF="$off",NW_TOTAL="$NWT",CLAIM=1 \
       "$HERE/gen_render.sh")
  echo "  렌더 $n GPU $g × 작업자 $w (전역 $off–$((off + g * w - 1)) / $NWT) → $j"
  deps="$deps:$j"; off=$((off + g * w))
done
jc=$(sbatch --parsable -p batch_vll -w "${V5_PACK_NODE:-vll6}" --dependency=afterany$deps -c 64 --mem=120G -t 0-03:00:00 -J v5_pack -o "$OUT/logs/%j_pack.out" \
     --wrap "cd $HERE && $P compose_v5mc.py --config $CFGY --workers 60 && echo V5_DONE")
echo "  묶기 → $jc"
