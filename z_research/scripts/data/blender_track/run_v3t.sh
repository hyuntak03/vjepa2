#!/bin/bash
# BlenderTrack v3t (v3 + trapdoor, 2026-11-01) 렌더 + 네 팔 합성 — vll7 GPU 6 장 한 노드 (11-01 사용자: vll7 써서).
#   설계 (design_blocks_v3.py · select_blocks_v3.py, configs/v3t.yaml) 가 끝난 뒤 sbatch 로:
#   sbatch -p batch_vll -w vll7 --gres=gpu:6 -c 100 --mem=200G --dependency=afterok:<설계 잡> -o <out>/logs/%j_render.out run_v3t.sh
#   probe (상황마다 6 후보, 확인용) → train → test 순서로 렌더하고, train · test 의 네 팔 (pos_a · pos_b · imp_ab · imp_ba) 을 만든다.
set -euo pipefail
HERE=/data/hyuntak/project/2026/2027_cvpr/vjepa2/z_research/scripts/data/blender_track
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
CFGY=$HERE/configs/v3t.yaml
O=$($P -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['out'])" "$CFGY")
$P -c "import yaml,json,sys; json.dump(yaml.safe_load(open(sys.argv[1])), open(sys.argv[2],'w'), indent=1)" "$CFGY" "$O/config.json"
cp "$CFGY" "$O/config.yaml"
rm -rf "$O/locks"
echo "### 렌더 $(date '+%F %T')"
CFG=$O/config.json SPLITS="probe train test" NODE_I=0 N_NODES=1 GPUS=${GPUS:-6} PER_GPU=${PER_GPU:-7} THREADS=2 CLAIM=1 bash "$HERE/gen_render.sh"
echo "### 네 팔 합성 $(date '+%F %T')"
cd "$HERE"
for sp in train test; do $P compose_arms_v3.py --config "$CFGY" --split $sp --workers 64; done
echo "V3T_DONE $(date '+%F %T')"
