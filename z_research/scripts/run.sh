#!/bin/bash
# -----------------------------------------------------------------------------
# world_model_analysis 표준 진입점
#
#   bash z_research/scripts/run.sh <프로토콜> <데이터셋> [모델]
#
#   GPUS=4 bash z_research/scripts/run.sh surprise_c16t32  v8         vith
#   GPUS=4 bash z_research/scripts/run.sh attn_probe       v8         vith
#   GPUS=4 bash z_research/scripts/run.sh intphys1_sliding intphys1_dev
#   (모델 생략 시 vith)
#
# 세 조각이 합쳐져 최종 config 하나가 만들어진다 (resolve.py):
#   configs/protocols/<프로토콜>.yaml   프레임 배치·채점 규칙·probe 정의·dtype 관례
#   configs/protocols/datasets.md       경로·인덱스·컬럼 이름            <- 데이터 바꿀 때 여기만
#   configs/protocols/models.md         체크포인트·arch_name
# 충돌하면 **프로토콜이 이긴다**. 자세한 규칙은 resolve.py docstring.
#
#   bash z_research/scripts/run.sh --list           무엇이 있는지 본다
#   DRYRUN=1 bash ... run.sh attn_probe v8          병합 결과만 보고 끝낸다 (GPU 안 씀)
#
# ★ evals.main 을 직접 부르지 말 것. 이 스크립트만 하는 두 가지가 있다:
#     - DDP 포트 자동 탐색
#     - WMA_EXPECT_WS export (split-brain 가드)
#   init_distributed 는 bind 실패를 except Exception 으로 삼키고 조용히 world_size=1 로
#   폴백한다. 그러면 뒤에 뜬 job 의 rank0 만 폴백하고 나머지는 먼저 뜬 job 의 store 에
#   붙어 **두 job 이 섞인다.** 가드가 없으면 이게 조용히 일어난다.
#
# 환경변수
#   GPUS=N              --devices cuda:0..N-1 + WMA_EXPECT_WS        (기본 1)
#   TAG=, OUTDIR=       tag / output_dir 을 직접 지정 (기본은 resolve.py 가 짓는다)
#   LIMIT=N             limit 주입 + tag/output_dir 에 _smoke{N} 접미사
#                       (본 결과와 토큰 캐시를 안 덮는다)
#   SMOKE=1             shape 디버그
#   RECACHE=1           토큰 캐시 무시하고 재추출
#   BATCH_SIZE, DECODE_WORKERS    surprise.* 덮어씀
#   SET="a.b=1 c.d=null"  병합된 config 를 점 경로로 덮어쓴다 (공백으로 여러 개)
#                       실험 케이스마다 yaml 을 새로 만들지 않기 위한 장치
#                       예) SET="probing.optim=attn_50 probing.targets.shape=null"
#   EVAL_DDP_PORT       지정 시 자동 탐색 생략
#   EVAL_DDP_TIMEOUT_S  NCCL collective timeout (기본 7200).
#                       torch 기본 600s 로는 rank 부하 불균형에서 죽는다
#   WMA_BAR=on|off|auto 진행바 (auto = stderr 가 TTY 일 때만)
#   DRYRUN=1            병합·검사만 하고 끝낸다
# -----------------------------------------------------------------------------
set -euo pipefail
PY=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
PROJ=/data/hyuntak/project/2026/2027_cvpr/vjepa2
P=$PROJ/configs/protocols

if [[ "${1:-}" == "--list" || $# -eq 0 ]]; then
  echo "프로토콜:";  ls "$P"/*.yaml | xargs -n1 basename | sed 's/\.yaml$//;s/^/  /'
  echo "데이터셋:";  "$PY" - "$P/datasets.md" <<'L'
import sys
sys.path.insert(0, "/data/hyuntak/project/2026/2027_cvpr/vjepa2/z_research/scripts")
from resolve import parse_registry
for k, v in sorted(parse_registry(sys.argv[1], "root").items()):
    bad = v.get("available") is False
    print(f"  {k:<16}" + (f"  [사용 불가] {str(v.get('note',''))[:80]}" if bad else ""))
L
  echo "모델:";      grep '^## ' "$P/models.md" | sed 's/^## /  /'
  exit 0
fi

PROTOCOL=${1:?"사용법: run.sh <프로토콜> <데이터셋> [모델]   (목록: run.sh --list)"}
DATASET=${2:?"사용법: run.sh <프로토콜> <데이터셋> [모델]   (목록: run.sh --list)"}
MODEL=${3:-vith}

cd "$PROJ"; export PYTHONPATH="$PROJ:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export EVAL_DDP_TIMEOUT_S=${EVAL_DDP_TIMEOUT_S:-7200}

GPUS=${GPUS:-1}
DEVICES=$(for i in $(seq 0 $((GPUS-1))); do echo -n "cuda:$i "; done)
CFG=$(mktemp /tmp/wma_XXXXXX.yaml); trap 'rm -f "$CFG"' EXIT

echo "=================================================="
SET_ARGS=()
for kv in ${SET:-}; do SET_ARGS+=(--set "$kv"); done
"$PY" z_research/scripts/harness/resolve.py "$PROTOCOL" "$DATASET" "$MODEL" -o "$CFG" "${SET_ARGS[@]}"
echo "  devices  : $DEVICES  (world_size=$GPUS)"
echo "=================================================="

# LIMIT/SMOKE/RECACHE/BATCH_SIZE/DECODE_WORKERS 는 병합된 config 위에 덧씌운다
if [[ -n "${SMOKE:-}${LIMIT:-}${RECACHE:-}${BATCH_SIZE:-}${DECODE_WORKERS:-}" ]]; then
  "$PY" - "$CFG" "${SMOKE:-}" "${LIMIT:-}" "${RECACHE:-}" \
    "${BATCH_SIZE:-}" "${DECODE_WORKERS:-}" <<'PYEOF'
import sys, yaml
f, smoke, limit, recache, batch_size, decode_workers = sys.argv[1:7]
c = yaml.safe_load(open(f))
if smoke:   c["smoke"] = True
if limit:   c["limit"] = int(limit)
if recache: c["recache"] = True
if batch_size:     c.setdefault("surprise", {})["batch_size"] = int(batch_size)
if decode_workers: c.setdefault("surprise", {})["decode_workers"] = int(decode_workers)
if limit:                      # 본 결과와 토큰 캐시가 섞이지 않게 분리
    c["tag"] = f"{c['tag']}_smoke{limit}"
    c["output_dir"] = c["output_dir"] + f"_smoke{limit}"
yaml.safe_dump(c, open(f, "w"), allow_unicode=True, sort_keys=False)
print(f"  override : smoke={smoke} limit={limit} recache={recache} "
      f"batch_size={batch_size} decode_workers={decode_workers}")
print(f"  tag      : {c['tag']}\n  output   : {c['output_dir']}")
PYEOF
fi

if [[ -n "${DRYRUN:-}" ]]; then
  echo "--- DRYRUN: 병합된 config ---"; cat "$CFG"; exit 0
fi

if [[ -z "${EVAL_DDP_PORT:-}" ]]; then
  BASE=$(( 20000 + ( ${SLURM_JOB_ID:-$$} % 10000 ) ))
  EVAL_DDP_PORT=$("$PY" -c "
import socket, sys
for p in range($BASE, $BASE + 200):
    s = socket.socket()
    try:
        s.bind(('', p)); s.close(); print(p); sys.exit(0)
    except OSError:
        s.close()
sys.exit(1)
") || { echo 'ERROR: 빈 DDP 포트를 못 찾았다'; exit 1; }
fi
export EVAL_DDP_PORT WMA_EXPECT_WS=$GPUS
echo "  ddp port : $EVAL_DDP_PORT"

# 병합된 config 를 결과 옆에 남긴다 (summary.json 에도 전체가 들어가지만 사람이 읽기 편하게)
OUT=$("$PY" -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['output_dir'])" "$CFG")
mkdir -p "$OUT"; cp "$CFG" "$OUT/_resolved.yaml"

"$PY" -m evals.main --fname "$CFG" --devices $DEVICES
