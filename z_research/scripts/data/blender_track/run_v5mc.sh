#!/bin/bash
# BlenderTrack v5_motion_control — 설계 (노드 셋 CPU 나눠) → 고르기 → 노드 셋 GPU 동시 렌더 (claim) → 묶기 (tar 친화 폴더) (2026-11-02, 사용자: vll3 · 6 · 7 다 써서)
#   bash z_research/scripts/data/blender_track/run_v5mc.sh
#   V5_DESIGN="노드 CPU|…" (설계 잡) · V5_NODES="노드 GPU 작업자/GPU|…" (렌더 잡) — 그 시각 빈 자원에 맞춘다
# (11-02 Vast) 경로는 env.sh 하나에서. sbatch 가 없으면 (또는 BT_LOCAL=1) 같은 단계를 이 머신 한 대에서 백그라운드로 돈다 — 아래 "로컬" 절.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/env.sh"
P=$BT_PY
if [[ -n ${DRY:-} ]]; then                                     # 아무것도 안 쓰고 바뀔 자리 · 실행 방식만 찍는다
  echo "자리 $BT_SITE | 파이썬 $P | 렌더 파이썬 $BT_BLENDER_PY | sbatch $(command -v sbatch || echo '없음 → 로컬')${BT_LOCAL:+ (BT_LOCAL=1 → 로컬)}"
  "$P" "$HERE/localize_cfg.py" "$HERE/configs/v5mc.yaml" --show
  echo "렌더 여유 GPU (빈 메모리 ≥ ${MIN_FREE_GB:-$(( ${PER_GPU:-8} * 5 ))} GB, 학습과 겹쳐도 됨): $(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null | awk -F', *' -v m=$(( ${MIN_FREE_GB:-$(( ${PER_GPU:-8} * 5 ))} * 1024 )) '$2 >= m {print $1}' | paste -sd, -)"
  exit 0
fi
CFGY=$("$P" "$HERE/localize_cfg.py" "$HERE/configs/v5mc.yaml" --write config_v5mc.yaml)   # <out>/config_v5mc.yaml — out · data_root 를 이 머신 자리로 (클러스터는 원본 그대로 복사)
OUT=$($P -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['out'])" "$CFGY")
mkdir -p "$OUT/logs"
SPL=$($P -c "import yaml,sys; print(' '.join(yaml.safe_load(open(sys.argv[1]))['splits']))" "$CFGY")
SPL_R=$(tr ' ' '\n' <<<"$SPL" | awk '/^p_/{print; next} {r=r" "$0} END{print r}' | xargs)     # 렌더 순서: 확인용 → 학습 → 시험

if command -v sbatch >/dev/null 2>&1 && [[ -z ${BT_LOCAL:-} ]]; then          # ---- 클러스터 (SLURM): 예전과 같다
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
exit 0
fi

# ----------------------------------------------------------------------------- 로컬 (sbatch 없음 · BT_LOCAL=1, 11-02 Vast)
#   bash run_v5mc.sh                  → 단계를 백그라운드 (setsid nohup) 로 띄우고 PID · 로그를 찍고 바로 끝난다. FG=1 이면 이 셸에서 (tmux 안 등)
#   GPU_LIST="0,1"    렌더 GPU (기본: 빈 메모리 ≥ MIN_FREE_GB = PER_GPU × 5 GB 인 GPU 전부 — 학습과 겹치면 렌더가 약 5 배 느려진다, SETUP_VAST 메모)
#   PER_GPU=8         GPU 마다 렌더 작업자 (하나 ≈ GPU 메모리 4 GB) · THREADS=2 작업자마다 CPU 스레드
#   DESIGN_PROCS=200  설계 프로세스 (기본 nproc − 16)                                PACK_WORKERS=48 묶기 프로세스
#   STAGE=design|select|render|pack   그 단계부터 (설계 · 렌더는 끝난 것을 건너뛰어 이어 한다)
#   SMOKE_FAMS=N      스모크: 고른 묶음 중 분할마다 N 개 (상황이 고루 퍼지게) 만 렌더한다 (gen_render_extra.sh 의 번호 목록으로)
#   DRY=1             아무것도 안 쓰고 자리 · 분할 · 빈 GPU 만 찍고 끝 (맨 위)  · 멈추기: kill -- -<PID> (세션 통째로)
#   BT_ROOT · BT_OUT · BT_DATA_ROOT · BT_CFG_ONLY · BT_CFG_SET  출력 자리 · 설정 사본 (env.sh · localize_cfg.py)
DESIGN_PROCS=${DESIGN_PROCS:-$(( $(nproc) - 16 ))}
PER_GPU=${PER_GPU:-8}; THREADS=${THREADS:-2}; PACK_WORKERS=${PACK_WORKERS:-48}; STAGE=${STAGE:-design}
MIN_FREE_GB=${MIN_FREE_GB:-$((PER_GPU * 5))}                   # 렌더 작업자 하나 ≈ GPU 메모리 4 GB (11-02 실측, OptiX 256²) — 학습과 겹쳐 써도 되니 "빈 GPU" 가 아니라 "빈 메모리" 로 고른다
if [[ -z ${GPU_LIST:-} ]]; then
  GPU_LIST=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', *' -v m=$((MIN_FREE_GB * 1024)) '$2 >= m {print $1}' | paste -sd, -)
fi
export GPU_LIST DESIGN_PROCS PER_GPU THREADS PACK_WORKERS STAGE
if [[ -z ${BT_LOCAL_RUN:-} ]]; then
  echo "로컬 v5mc | out $OUT | 분할 $SPL | GPU [$GPU_LIST] × 작업자 $PER_GPU | 설계 프로세스 $DESIGN_PROCS | 시작 단계 $STAGE${SMOKE_FAMS:+ | 스모크: 분할마다 묶음 $SMOKE_FAMS 개}"
  [[ -n $GPU_LIST ]] || { echo "!! 빈 메모리 ${MIN_FREE_GB} GB 이상인 GPU 가 없다 — GPU_LIST · PER_GPU · MIN_FREE_GB 로 준다"; exit 1; }
  if [[ -n ${FG:-} ]]; then BT_LOCAL_RUN=1 exec bash "$HERE/run_v5mc.sh"; fi
  LOG=$OUT/logs/local_$(date +%Y%m%d_%H%M%S).out
  BT_LOCAL_RUN=1 setsid nohup bash "$HERE/run_v5mc.sh" > "$LOG" 2>&1 < /dev/null &
  echo $! > "$OUT/logs/local.pid"
  echo "  PID $! (멈추기: kill -- -$!) · 로그 $LOG · 렌더 작업자 로그 $OUT/logs/<분할>_w<번호>.log"
  exit 0
fi

ts() { date '+%F %T'; }
case $STAGE in design) S=1;; select) S=2;; render) S=3;; pack) S=4;; *) echo "!! STAGE=$STAGE"; exit 1;; esac
echo "### 로컬 v5mc $(ts) | host $(hostname) | out $OUT | GPU [$GPU_LIST] × $PER_GPU | 설계 $DESIGN_PROCS | 시작 $STAGE"
cd "$HERE"
if (( S <= 1 )); then
  for s in $SPL; do
    echo "## 설계 $s $(ts)"
    $P design_families_v5mc.py --config "$CFGY" --split "$s" --procs "$DESIGN_PROCS"
  done
  echo "DESIGN_DONE $(ts)"
fi
if (( S <= 2 )); then
  echo "## 고르기 $(ts)"
  $P design_families_v5mc.py --config "$CFGY" --select-all
  echo "SELECT_DONE $(ts)"
fi
if (( S <= 3 )); then
  rm -rf "$OUT/locks"                                          # 한 머신에서는 앞 실행의 잠금이 남아 있으면 15 분 동안 그 clip 을 못 잡는다
  RCFG=$OUT/config_render.json; RS=gen_render.sh; EXTRA=""
  if [[ -n ${SMOKE_FAMS:-} ]]; then                            # 분할마다 고른 묶음 N 개 → 번호 목록 (n = 0 인 렌더 설정 사본 + 목록만 그린다)
    $P - "$OUT" "$SMOKE_FAMS" <<'PY'
import json, os, sys
import numpy as np
out, k = sys.argv[1], int(sys.argv[2])
cfg = json.load(open(f"{out}/config_render.json"))
os.makedirs(f"{out}/smoke_list", exist_ok=True)
for sp in cfg["splits"]:
    man = json.load(open(f"{out}/manifest_{sp}.json"))
    fams = sorted({m["family"] for m in man})
    pick = {fams[int(i)] for i in np.round(np.linspace(0, len(fams) - 1, min(k, len(fams))))} if fams else set()
    idx = [m["idx"] for m in man if m["family"] in pick]
    open(f"{out}/smoke_list/extra_{sp}.txt", "w").write(" ".join(map(str, idx)))
    cfg["splits"][sp]["n"] = 0
    print(f"  스모크 {sp}: 묶음 {sorted(pick)} · 영상 {len(idx)}")
json.dump(cfg, open(f"{out}/config_render_smoke.json", "w"), indent=1)
PY
    RCFG=$OUT/config_render_smoke.json; RS=gen_render_extra.sh; EXTRA=$OUT/smoke_list
  fi
  NG=$(tr ',' '\n' <<<"$GPU_LIST" | grep -c .)
  NW=$((NG * PER_GPU))
  echo "## 렌더 $(ts) | 순서 $SPL_R | GPU [$GPU_LIST] × $PER_GPU = 작업자 $NW"
  rc=0
  CFG="$RCFG" SPLITS="$SPL_R" NODE_I=0 N_NODES=1 GPUS="$NG" PER_GPU="$PER_GPU" THREADS="$THREADS" W_OFF=0 NW_TOTAL="$NW" CLAIM=1 \
    GPU_LIST="$GPU_LIST" EXTRA_DIR="$EXTRA" bash "$HERE/$RS" || rc=$?
  echo "RENDER_DONE rc=$rc $(ts)"                             # 묶기는 렌더가 실패해도 돈다 (sbatch afterany 와 같게 — 통과 못 한 묶음은 work/ 에 남는다)
fi
if (( S <= 4 )); then
  echo "## 묶기 $(ts)"
  $P compose_v5mc.py --config "$CFGY" --workers "$PACK_WORKERS"
  echo "V5_DONE $(ts)"
fi
