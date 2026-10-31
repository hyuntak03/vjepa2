#!/bin/bash
# BlenderTrack 데이터 파이프라인 — 설정 파일 하나로 끝까지 자동 (2026-10-31).
#
#   bash z_research/scripts/data/blender_track/pipeline.sh z_research/scripts/data/blender_track/configs/v1.yaml
#   STAGE=render bash …/pipeline.sh <설정>      # 검사 · 스모크를 건너뛰고 본 렌더부터 (끝난 clip 은 건너뛰어 이어 한다)
#   STAGE=meta   bash …/pipeline.sh <설정>      # 메타만 다시
#   DRY=1        bash …/pipeline.sh <설정>      # 렌더 전 검사만 (GPU 0 장)
#
# 단계 (앞 단계가 실패하면 뒤는 안 돈다 — sbatch afterok 사슬)
#   1 check   렌더 전 검사 (CPU, 이 셸에서 바로): 운동 종류 비율 · late 창 · 경로 수 · 외우기 대 계산 참고값 → <out>/check.json
#   2 smoke   스모크 렌더 (첫 노드 GPU 1 장 · 작업자 2) → 3 verify: mp4 디코딩 · 라벨 · 투영 오차 · 확인 시트 · 본 렌더 예상 시간
#   4 render  본 렌더: 설정의 nodes 마다 job 하나 (GPU 전부 · 분할 train → val → ood 차례로, 작업자 번호는 노드를 가로질러 전역)
#   5 meta    학습 메타 (분할마다 data_csv/blender_track_<name>/<split>_meta.npz) + 검증 분할 확인 시트
# 결과: <out>/{clips/*.mp4, labels/*.npz, specs/*.json, check.json, smoke_verify.json, smoke_sheet.png, logs/}
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../../../.." && pwd)
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
CFG=$(realpath "${1:?사용법: bash pipeline.sh <설정.yaml>}")
read -r OUT NAME GPN PER THR NODES NGPUS CLAIMM <<<"$("$P" -c "
import yaml,sys; c=yaml.safe_load(open(sys.argv[1])); r=c['render']
g=r.get('node_gpus') or [r['gpus_per_node']]*len(r['nodes']); assert len(g)==len(r['nodes'])
print(c['out'], c['name'], r['gpus_per_node'], r['workers_per_gpu'], r['threads'], ','.join(r['nodes']), ','.join(map(str,g)), int(bool(r.get('claim'))))" "$CFG")"
IFS=, read -r -a NODE_ARR <<<"$NODES"
IFS=, read -r -a NGPU_ARR <<<"$NGPUS"                             # 노드별 GPU 수 (render.node_gpus, 없으면 gpus_per_node)
SEEDS=$("$P" -c "import yaml,sys; c=yaml.safe_load(open(sys.argv[1])); print(' '.join(f'{k}:{v[\"seed\"]}' for k,v in c['splits'].items() if k!='smoke'))" "$CFG")
mkdir -p "$OUT/logs"; cp "$CFG" "$OUT/config.yaml"
CFGJ=$OUT/config.json                                            # 렌더 잡 (blender env, yaml 모듈 없음) 은 json 을 읽는다
"$P" -c "import yaml,json,sys; json.dump(yaml.safe_load(open(sys.argv[1])), open(sys.argv[2],'w'), indent=1)" "$CFG" "$CFGJ"
STAGE=${STAGE:-check}
order() { case $1 in check) echo 1;; smoke) echo 2;; render) echo 4;; meta) echo 5;; esac; }
S=$(order "$STAGE")
echo "설정 $CFG | 출력 $OUT | 노드 ${NODE_ARR[*]} × GPU $GPN × 작업자 $PER | 시작 단계 $STAGE"

dep=""
if (( S <= 1 )); then
  echo "### 1 렌더 전 검사"
  "$P" "$HERE/gen_check.py" --config "$CFG" --out "$OUT/check.json" > "$OUT/logs/check.out" || { echo "!! 렌더 전 검사 실패 → $OUT/check.json"; exit 1; }
  "$P" -c "import json,sys; r=json.load(open(sys.argv[1])); print('  통과', r['gates'], '| 보폭별 late clip', {k:v['clips_with_late'] for k,v in r['late_by_stride'].items()})" "$OUT/check.json"
  [[ -n ${DRY:-} ]] && exit 0
fi
if (( S <= 2 )); then
  echo "### 2 스모크 렌더 → 3 검증"
  js=$(sbatch --parsable -p batch_vll -w "${NODE_ARR[0]}" --gres=gpu:1 -c $((2 * THR + 4)) --mem=32G -t 01:00:00 -J bt_smoke_$NAME \
       -o "$OUT/logs/%j_smoke.out" --export=ALL,CFG="$CFGJ",SPLITS=smoke,NODE_I=0,N_NODES=1,GPUS=1,PER_GPU=2,THREADS="$THR" "$HERE/gen_render.sh")
  jv=$(sbatch --parsable -p batch_vll -c 4 --mem=16G -t 00:30:00 --dependency=afterok:$js -J bt_verify_$NAME \
       -o "$OUT/logs/%j_verify.out" --wrap "$P $HERE/gen_verify.py --config $CFG --split smoke")
  echo "  smoke $js → verify $jv"
  dep="--dependency=afterok:$jv"
fi
rdeps=""
if (( S <= 4 )); then
  echo "### 4 본 렌더"
  n=${#NODE_ARR[@]}
  NWT=0; for g in "${NGPU_ARR[@]}"; do NWT=$((NWT + g * PER)); done
  CX=""; [[ $CLAIMM == 1 ]] && CX=",CLAIM=1"                      # claim 모드: 작업자가 남은 clip 을 잠금 파일로 나눠 가진다
  off=0
  for i in "${!NODE_ARR[@]}"; do
    g=${NGPU_ARR[$i]}
    jr=$(sbatch --parsable -p batch_vll -w "${NODE_ARR[$i]}" --gres=gpu:"$g" -c $((g * PER * THR + 8)) --mem=$((g * PER * 5 + 20))G -t 2-00:00:00 $dep \
         -J bt_render_${NAME}_$i -o "$OUT/logs/%j_render_$i.out" \
         --export=ALL,CFG="$CFGJ",SPLITS="train val ood",NODE_I="$i",N_NODES="$n",GPUS="$g",PER_GPU="$PER",THREADS="$THR",W_OFF="$off",NW_TOTAL="$NWT"$CX "$HERE/gen_render.sh")
    rdeps="$rdeps:$jr"; echo "  render ${NODE_ARR[$i]} (GPU $g, 작업자 $off–$((off + g * PER - 1)) / $NWT) → $jr"
    off=$((off + g * PER))
  done
fi
echo "### 5 학습 메타"
MCMD="cd $REPO"
for kv in $SEEDS; do sp=${kv%%:*}; sd=${kv##*:}
  MCMD="$MCMD && $P z_training/data/build_blender_track_index.py --root $OUT --prefix ${sp}${sd}_ --out data_csv/blender_track_${NAME}/${sp}_meta.npz"
done
MCMD="$MCMD && $P $HERE/gen_verify.py --config $CFG --split val"
jm=$(sbatch --parsable -p batch_vll -c 16 --mem=32G -t 01:00:00 ${rdeps:+--dependency=afterok$rdeps} -J bt_meta_$NAME \
     -o "$OUT/logs/%j_meta.out" --wrap "$MCMD")
echo "  meta → $jm"
echo "진행: squeue -u \$USER | grep bt_   ·   로그 $OUT/logs/   ·   스모크 시트 $OUT/smoke_sheet.png"
