#!/usr/bin/env bash
# =============================================================================
# vll7 학습 predictor 채점 — VOBench (late 포함) · VOBench gravity · IntPhys 1 · IntPhys 2 (2026-10-17)
#
#   GPUS=6 bash z_vll7_training/eval.sh ctxt_video_tgtvideo                 # 네 가지 전부
#   GPUS=6 bash z_vll7_training/eval.sh ctxt_video_tgtvideo "intphys1 vobench"   # 고르기
#   CKPT=e65.pt GPUS=6 bash z_vll7_training/eval.sh ctxt_video_tgtvideo     # 중간 체크포인트
#   DRYRUN=1 bash z_vll7_training/eval.sh ctxt_video_tgtvideo               # 병합 config 만 (GPU 0 장)
#
# 과제 이름: vobench | gravity | intphys1 | intphys2        (기본 = 넷 전부)
#
# ── vll7 에서 cvpr 하네스를 쓰기 위한 두 가지 ───────────────────────────────
#   1. CVPR_DATA2 shim — 레지스트리가 `${CVPR_DATA2}/local_datasets/world/Benchmark/{VOBench_v1,IntPhys2}` 를 보는데
#      이 노드의 실물은 `/local_datasets/world/{testbed/VOBench_v1,IntPhys2}` 다. 심볼릭 트리를 만들어 맞춘다 (아래 ensure_shim)
#   2. CVPR_PY — 기본값은 sm_120 을 모르는 env 다. vjepa2_bw (torch 2.12.1+cu129) 로 바꾼다
#
# ⚠️ 채점 타깃은 체크포인트 `arch.ctx_target` 이 정한다 (window | block). vast 는 이것을 채점기에 안 넘겨
#    block 학습판을 window 로 채점한 적이 있다 — IntPhys1 43.89 **무효** -> 고친 뒤 76.11
#    (`git show 73ae468:z_training/RESULTS_2026-10-16.md` §3). 우리 경로는 `analysis/predictors.ctx_target_of` 로 읽는다.
# ⚠️ VOBench **late** 는 따로 돌리는 게 아니라 `vobench` 결과를 조건별로 쪼갠 것이다
#    (`z_research/scripts/analysis/vobench_tables.py` 가 late/early/mid · 위반별 표를 낸다). 아래 요약이 그걸 부른다.
# =============================================================================
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/env.sh" >/dev/null
RUN=${1:?"사용법: GPUS=6 bash z_vll7_training/eval.sh <run> [과제들]"}
TASKS=${2:-"intphys1 vobench gravity intphys2"}
CKPT=${CKPT:-latest.pt}
D="$VLL7_RUNS/$RUN"
[[ -f "$D/$CKPT" ]] || { echo "체크포인트 없음: $D/$CKPT"; exit 1; }

ensure_shim() {   # ${CVPR_DATA2} 가 가리킬 심볼릭 트리 (vll7 전용). 이미 맞으면 아무것도 안 한다
  local s=/local_datasets/world/_cvpr_data2_vll7/local_datasets/world/Benchmark
  mkdir -p "$s" || return 1
  [[ -e "$s/VOBench_v1" ]] || ln -sfn /local_datasets/world/testbed/VOBench_v1 "$s/VOBench_v1"
  [[ -e "$s/IntPhys2"   ]] || ln -sfn /local_datasets/world/IntPhys2           "$s/IntPhys2"
}
ensure_shim || { echo "⚠️ shim 을 못 만들었다 — VOBench · IntPhys2 는 건너뛴다"; }

# cvpr/env.sh 를 먼저 읽어 CVPR_* 를 채운다 (CVPR_RESULTS 등). 그 뒤 vll7 전용 값으로 덮는다
# shellcheck source=/dev/null
source "$VJEPA2_ROOT/cvpr/env.sh"
export CVPR_DATA2=/local_datasets/world/_cvpr_data2_vll7
export CVPR_PY="$VJEPA2_PY"                       # sm_120 (Blackwell) 을 아는 env
export GPUS=${GPUS:-6}

# ── 모델 이름: 레지스트리에 있으면 그걸, 없으면 SET 으로 체크포인트를 꽂는다 ──
# 같은 **run 폴더**를 가리키는 레지스트리 항목을 찾는다 (파일 이름은 달라도 된다 — 다르면 SET 으로 덮는다)
MODEL=${MODEL:-}
CKPT_SET=""
REG_CK=""
if [[ -z "$MODEL" ]]; then
  read -r MODEL REG_CK < <("$VJEPA2_PY" "$HERE/_find_model.py" "$VJEPA2_ROOT/cvpr/registry/models.yaml" "$D") || true
fi
if [[ -z "${MODEL:-}" ]]; then
  echo "⚠️ cvpr/registry/models.yaml 에 $D 를 가리키는 항목이 없다."
  echo "   항목을 추가하거나 MODEL=<이름> 으로 줄 것 (예: vith_vll7_ctxt_tgtvideo)"
  exit 1
fi
if [[ -n "${REG_CK:-}" && "$REG_CK" != "$CKPT" ]]; then   # 레지스트리와 다른 체크포인트 -> SET 으로 덮고 결과 이름에 표시
  CKPT_SET="model.predictor_checkpoint=$D/$CKPT"
  export SUFFIX="${SUFFIX:-}_${CKPT%.pt}"
fi
echo "=================================================="
echo "  run      : $RUN   ($CKPT)"
echo "  model    : $MODEL"
echo "  과제     : $TASKS"
echo "  python   : $CVPR_PY"
echo "  CVPR_DATA2: $CVPR_DATA2  (vll7 shim)"
"$VJEPA2_PY" -c "
import torch, sys; d = torch.load('$D/$CKPT', map_location='cpu', weights_only=False)
a = d.get('arch') or {}
print(f\"  ckpt     : epoch {d.get('epoch')} step {d.get('step')} loss {d.get('loss'):.4f} | kind={a.get('kind')} ctx_target={a.get('ctx_target','window')}\")
"
echo "=================================================="

fail=0
for t in $TASKS; do
  echo; echo "########## $t  ($(date '+%H:%M:%S'))"
  case $t in
    intphys1) SET="${SET:-} $CKPT_SET" bash "$VJEPA2_ROOT/cvpr/eval/intphys1/run.sh" intphys1_dev       "$MODEL" || fail=1 ;;
    vobench)  SET="${SET:-} $CKPT_SET" bash "$VJEPA2_ROOT/cvpr/eval/testbed/run.sh"  vobench_v1         "$MODEL" || fail=1 ;;
    gravity)  SET="${SET:-} $CKPT_SET" bash "$VJEPA2_ROOT/cvpr/eval/testbed/run.sh"  vobench_v1_gravity "$MODEL" || fail=1 ;;
    intphys2) SET="${SET:-} $CKPT_SET" bash "$VJEPA2_ROOT/cvpr/eval/intphys2/run.sh" intphys2_main      "$MODEL" || fail=1 ;;
    *) echo "모르는 과제: $t  (vobench | gravity | intphys1 | intphys2)"; fail=1 ;;
  esac
done

# ── VOBench 표 (late · early · mid, 위반별) ─────────────────────────────────
if [[ "$TASKS" == *vobench* && ${DRYRUN:-0} != 1 ]]; then
  echo; echo "########## VOBench 표 (late 분해 포함)"
  "$VJEPA2_PY" "$VJEPA2_ROOT/z_research/scripts/analysis/vobench_tables.py" --model "$MODEL" 2>&1 | tail -40 || true
fi
echo; echo "끝 (exit $fail).  결과: $CVPR_RESULTS/eval/{testbed,intphys1,intphys2}/"
exit $fail
