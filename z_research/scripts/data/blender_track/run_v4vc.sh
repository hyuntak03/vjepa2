#!/bin/bash
# BlenderTrack v4_velocity_control (v4vc) — 설계 (CPU) → 여러 노드 동시 렌더 (claim) → 팔 · 목록 (2026-11-02, 사용자: 멀티 노드 가능한 GPU 다 써서).
#   bash z_research/scripts/data/blender_track/run_v4vc.sh
#   NODES 표 = "노드 GPU 작업자/GPU" — 그 시각 빈 GPU 에 맞춘다 (V4VC_NODES 로 덮는다). 설계 잡이 끝나면 렌더 잡들이 함께 시작한다.
#   렌더 순서: probe → test → train_fam → train_ind (claim 이라 노드 수 · 늦게 뜬 잡과 무관하게 나눠 그린다)
# (11-02 Vast) 경로는 env.sh 하나에서. sbatch 가 없으면 (또는 BT_LOCAL=1) 같은 단계를 이 머신 한 대에서 백그라운드로 돈다 — 아래 "로컬" 절.
#   팔 · 목록 (compose_v4vc.py) 은 BT_DATA_CSV/blender_track_<name>/ 에 쓴다 (Vast 기본 <BT_ROOT>/data_csv, 클러스터 기본 <레포>/data_csv).
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/env.sh"
P=$BT_PY
SRC=$HERE/configs/${V4VC_CFG:-v4vc.yaml}                       # 11-02 2 차: V4VC_CFG=v4vc_ramp.yaml (빗면)
if [[ -n ${DRY:-} ]]; then                                     # 아무것도 안 쓰고 바뀔 자리 · 실행 방식만 찍는다
  echo "자리 $BT_SITE | 파이썬 $P | 렌더 파이썬 $BT_BLENDER_PY | 목록 ${BT_DATA_CSV:-<레포>/data_csv} | sbatch $(command -v sbatch || echo '없음 → 로컬')${BT_LOCAL:+ (BT_LOCAL=1 → 로컬)}"
  "$P" "$HERE/localize_cfg.py" "$SRC" --show
  echo "렌더 여유 GPU (빈 메모리 ≥ ${MIN_FREE_GB:-$(( ${PER_GPU:-8} * 5 ))} GB, 학습과 겹쳐도 됨): $(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null | awk -F', *' -v m=$(( ${MIN_FREE_GB:-$(( ${PER_GPU:-8} * 5 ))} * 1024 )) '$2 >= m {print $1}' | paste -sd, -)"
  exit 0
fi
CFGY=$("$P" "$HERE/localize_cfg.py" "$SRC" --write 'config_{name}.yaml')   # <out>/config_<name>.yaml — out 을 이 머신 자리로 (클러스터는 원본 그대로 복사)
O=$($P -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['out'])" "$CFGY")
mkdir -p "$O/logs"
NM=$($P -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['name'])" "$CFGY")
CJ=$O/config_$NM.json                                             # 평면 (v4vc) · 빗면 (v4vcr) 이 같은 폴더를 쓴다 → 설정 파일 이름을 나눈다
$P -c "import yaml,json,sys; json.dump(yaml.safe_load(open(sys.argv[1])), open(sys.argv[2],'w'), indent=1)" "$CFGY" "$CJ"
SPL=$($P -c "import yaml,sys; print(' '.join(yaml.safe_load(open(sys.argv[1]))['splits']))" "$CFGY")   # 설정의 분할 순서 (확인용 probe 를 앞에)
SPL=$(tr ' ' '\n' <<<"$SPL" | awk '/probe/{print; next} {r=r" "$0} END{print r}' | xargs)

if command -v sbatch >/dev/null 2>&1 && [[ -z ${BT_LOCAL:-} ]]; then          # ---- 클러스터 (SLURM): 예전과 같다
DN=${V4VC_DESIGN_NODE:-vll4}
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
exit 0
fi

# ----------------------------------------------------------------------------- 로컬 (sbatch 없음 · BT_LOCAL=1, 11-02 Vast) — run_v5mc.sh 로컬 절과 같은 손잡이
#   GPU_LIST="0,1" (기본: 빈 메모리 ≥ MIN_FREE_GB = PER_GPU × 5 GB 인 GPU 전부) · PER_GPU=8 · THREADS=2 · DESIGN_PROCS (기본 nproc − 16) · PACK_WORKERS=48
#   STAGE=design|render|pack · FG=1 · DRY=1
#   멈추기: kill -- -<PID>
DESIGN_PROCS=${DESIGN_PROCS:-$(( $(nproc) - 16 ))}
PER_GPU=${PER_GPU:-8}; THREADS=${THREADS:-2}; PACK_WORKERS=${PACK_WORKERS:-48}; STAGE=${STAGE:-design}
MIN_FREE_GB=${MIN_FREE_GB:-$((PER_GPU * 5))}                   # 렌더 작업자 하나 ≈ GPU 메모리 4 GB (11-02 실측, OptiX 256²) — 학습과 겹쳐 써도 되니 "빈 GPU" 가 아니라 "빈 메모리" 로 고른다
if [[ -z ${GPU_LIST:-} ]]; then
  GPU_LIST=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', *' -v m=$((MIN_FREE_GB * 1024)) '$2 >= m {print $1}' | paste -sd, -)
fi
export GPU_LIST DESIGN_PROCS PER_GPU THREADS PACK_WORKERS STAGE
if [[ -z ${BT_LOCAL_RUN:-} ]]; then
  echo "로컬 $NM | out $O | 분할 $SPL | GPU [$GPU_LIST] × 작업자 $PER_GPU | 설계 프로세스 $DESIGN_PROCS | 시작 단계 $STAGE"
  [[ -n $GPU_LIST ]] || { echo "!! 빈 메모리 ${MIN_FREE_GB} GB 이상인 GPU 가 없다 — GPU_LIST · PER_GPU · MIN_FREE_GB 로 준다"; exit 1; }
  if [[ -n ${FG:-} ]]; then BT_LOCAL_RUN=1 exec bash "$HERE/run_v4vc.sh"; fi
  LOG=$O/logs/local_${NM}_$(date +%Y%m%d_%H%M%S).out
  BT_LOCAL_RUN=1 setsid nohup bash "$HERE/run_v4vc.sh" > "$LOG" 2>&1 < /dev/null &
  echo $! > "$O/logs/local_$NM.pid"
  echo "  PID $! (멈추기: kill -- -$!) · 로그 $LOG · 렌더 작업자 로그 $O/logs/<분할>_w<번호>.log"
  exit 0
fi

ts() { date '+%F %T'; }
case $STAGE in design) S=1;; render) S=3;; pack) S=4;; *) echo "!! STAGE=$STAGE"; exit 1;; esac
echo "### 로컬 $NM $(ts) | host $(hostname) | out $O | GPU [$GPU_LIST] × $PER_GPU | 설계 $DESIGN_PROCS | 시작 $STAGE"
cd "$HERE"
if (( S <= 1 )); then
  for s in $SPL; do
    echo "## 설계 · 고르기 $s $(ts)"
    $P design_families_v4vc.py --config "$CFGY" --split "$s" --procs "$DESIGN_PROCS"
    $P design_families_v4vc.py --config "$CFGY" --split "$s" --select
  done
  echo "DESIGN_DONE $(ts)"
fi
if (( S <= 3 )); then
  rm -rf "$O/locks"
  NG=$(tr ',' '\n' <<<"$GPU_LIST" | grep -c .)
  NW=$((NG * PER_GPU))
  echo "## 렌더 $(ts) | 순서 $SPL | GPU [$GPU_LIST] × $PER_GPU = 작업자 $NW"
  rc=0
  CFG="$CJ" SPLITS="$SPL" NODE_I=0 N_NODES=1 GPUS="$NG" PER_GPU="$PER_GPU" THREADS="$THREADS" W_OFF=0 NW_TOTAL="$NW" CLAIM=1 \
    GPU_LIST="$GPU_LIST" bash "$HERE/gen_render.sh" || rc=$?
  echo "RENDER_DONE rc=$rc $(ts)"                             # 팔 · 목록은 렌더가 실패해도 돈다 (sbatch afterany 와 같게)
fi
if (( S <= 4 )); then
  echo "## 팔 · 목록 $(ts)"
  $P compose_v4vc.py --config "$CFGY" --workers "$PACK_WORKERS"
  echo "V4VC_DONE $(ts)"
fi
