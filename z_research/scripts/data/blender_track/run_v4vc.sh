#!/bin/bash
# BlenderTrack v4_velocity_control (v4vc) — 설계 (CPU) → 여러 노드 동시 렌더 (claim) → 팔 · 목록 (2026-11-02, 사용자: 멀티 노드 가능한 GPU 다 써서).
#   bash z_research/scripts/data/blender_track/run_v4vc.sh
#   NODES 표 = "노드 GPU 작업자/GPU" — 그 시각 빈 GPU 에 맞춘다 (V4VC_NODES 로 덮는다). 설계 잡이 끝나면 렌더 잡들이 함께 시작한다.
#   렌더 순서: probe → test → train_fam → train_ind (claim 이라 노드 수 · 늦게 뜬 잡과 무관하게 나눠 그린다)
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
CFGY=$HERE/configs/${V4VC_CFG:-v4vc.yaml}                       # 11-02 2 차: V4VC_CFG=v4vc_ramp.yaml (빗면)
O=$($P -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['out'])" "$CFGY")
mkdir -p "$O/logs"
NM=$($P -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['name'])" "$CFGY")
CJ=$O/config_$NM.json                                             # 평면 (v4vc) · 빗면 (v4vcr) 이 같은 폴더를 쓴다 → 설정 파일 이름을 나눈다
$P -c "import yaml,json,sys; json.dump(yaml.safe_load(open(sys.argv[1])), open(sys.argv[2],'w'), indent=1)" "$CFGY" "$CJ"
cp "$CFGY" "$O/config_$NM.yaml"
DN=${V4VC_DESIGN_NODE:-vll4}
SPL=$($P -c "import yaml,sys; print(' '.join(yaml.safe_load(open(sys.argv[1]))['splits']))" "$CFGY")   # 설정의 분할 순서 (확인용 probe 를 앞에)
SPL=$(tr ' ' '\n' <<<"$SPL" | awk '/probe/{print; next} {r=r" "$0} END{print r}' | xargs)
jd=$(sbatch --parsable -p batch_vll -w "$DN" -c 90 --mem=120G -t 0-02:00:00 -J v4vc_design -o "$O/logs/%j_design.out" \
     --wrap "cd $HERE && for s in $SPL; do $P design_families_v4vc.py --config $CFGY --split \$s --procs 88 && $P design_families_v4vc.py --config $CFGY --split \$s --select || exit 1; done; echo DESIGN_DONE")
echo "  설계 ($DN) → $jd"
rm -rf "$O/locks"
IFS="|" read -r -a NODES <<<"${V4VC_NODES:-vll4 8 5|vll7 2 6}"          # 줄 = "노드 GPU 작업자/GPU", 줄 사이는 |
NWT=0; for row in "${NODES[@]}"; do read -r n g w <<<"$row"; NWT=$((NWT + g * w)); done
off=0; deps=""
for row in "${NODES[@]}"; do
  read -r n g w <<<"$row"
  j=$(sbatch --parsable -p batch_vll -w "$n" --gres=gpu:"$g" -c $((g * w * 2 + 6)) --mem=$((g * w * 4 + 16))G -t 0-06:00:00 -J v4vc_render \
       --dependency=afterok:$jd -o "$O/logs/%j_render_$n.out" \
       --export=ALL,CFG="$CJ",SPLITS="$SPL",NODE_I=0,N_NODES=1,GPUS="$g",PER_GPU="$w",THREADS=2,W_OFF="$off",NW_TOTAL="$NWT",CLAIM=1 \
       "$HERE/gen_render.sh")
  echo "  렌더 $n GPU $g × 작업자 $w (전역 $off–$((off + g * w - 1)) / $NWT) → $j"
  deps="$deps:$j"; off=$((off + g * w))
done
jc=$(sbatch --parsable -p batch_vll -w "${V4VC_COMPOSE_NODE:-vll4}" --dependency=afterany$deps -c 64 --mem=96G -t 0-02:00:00 -J v4vc_compose -o "$O/logs/%j_compose.out" \
     --wrap "cd $HERE && $P compose_v4vc.py --config $CFGY --workers 60 && echo V4VC_DONE")
echo "  팔 · 목록 → $jc"
