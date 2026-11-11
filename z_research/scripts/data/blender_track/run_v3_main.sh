#!/bin/bash
# BlenderTrack v3 본 데이터 — 고르기 → 여러 노드 동시 렌더 (claim) → 네 팔 합성 · 검사 (2026-11-01).
#   설계 (design_blocks_v3.py) 가 끝난 뒤에 부른다:  bash z_research/scripts/data/blender_track/run_v3_main.sh
#   노드 · GPU · 작업자는 아래 NODES 표 (11-01 사용자: vll4 · 5 · 6 · 7 을 다 써서 최대한 빨리). 렌더 순서는 train → test.
# (11-02 Vast) 경로는 env.sh 하나에서, 설정은 out 을 이 머신 자리로 바꾼 사본 (localize_cfg.py) 으로 돈다. 클러스터에서는 예전과 같다.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/env.sh"
P=$BT_PY
command -v sbatch >/dev/null 2>&1 || { echo "!! sbatch 없음 — run_v3_main.sh 는 SLURM 전용 (로컬 판은 run_v5mc.sh · run_v4vc.sh)"; exit 1; }
CFGY=$("$P" "$HERE/localize_cfg.py" "$HERE/configs/v3.yaml" --write config.yaml)   # <out>/config.yaml (예전 cp 자리)
O=$($P -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['out'])" "$CFGY")
$P -c "import yaml,json,sys; json.dump(yaml.safe_load(open(sys.argv[1])), open(sys.argv[2],'w'), indent=1)" "$CFGY" "$O/config.json"
echo "### 1 고르기"
for sp in train test; do $P "$HERE/select_blocks_v3.py" --config "$CFGY" --split $sp | head -12; done
rm -rf "$O/locks"
echo "### 2 렌더 (claim, 노드마다 GPU 전부)"
# 노드 GPU 작업자/GPU 스레드
NODES=("vll4 8 4 2" "vll5 8 5 2" "vll6 8 5 2" "vll7 6 5 3")
NWT=0; for row in "${NODES[@]}"; do read -r n g w t <<<"$row"; NWT=$((NWT + g * w)); done
off=0; deps=""
for row in "${NODES[@]}"; do
  read -r n g w t <<<"$row"
  j=$(sbatch --parsable -p batch_vll -w "$n" --gres=gpu:"$g" -c $((g * w * t + 8)) --mem=$((g * w * 5 + 20))G -t 1-00:00:00 -J bt_v3_render \
       -o "$O/logs/%j_render_$n.out" --export=ALL,CFG="$O/config.json",SPLITS="train test",NODE_I=0,N_NODES=1,GPUS="$g",PER_GPU="$w",THREADS="$t",W_OFF="$off",NW_TOTAL="$NWT",CLAIM=1 \
       "$HERE/gen_render.sh")
  echo "  $n GPU $g × 작업자 $w (전역 $off–$((off + g * w - 1)) / $NWT) → $j"
  deps="$deps:$j"; off=$((off + g * w))
done
echo "### 3 네 팔 합성 · 검사 (렌더가 다 끝나면)"
jc=$(sbatch --parsable -p batch_vll -w vll7 --dependency=afterany$deps -c 64 --mem=96G -t 02:00:00 -J bt_v3_compose -o "$O/logs/%j_compose_main.out" \
     --wrap "cd $HERE && $P compose_arms_v3.py --config $CFGY --split train --workers 64 && $P compose_arms_v3.py --config $CFGY --split test --workers 64")
echo "  compose → $jc"
echo "진행: squeue -u \$USER | grep bt_v3   ·   ls $O/clips | wc -l"
