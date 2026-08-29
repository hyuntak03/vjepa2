#!/bin/bash
#SBATCH --job-name=wma
#SBATCH --partition=batch_vll
#SBATCH -w vll5
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=45G
#SBATCH --time=0-08:00:00
#SBATCH --output=/data/hyuntak/project/2026/2027_cvpr/vjepa2/z_research/scripts/slurm_logs/%x_%j.out
#SBATCH --error=/data/hyuntak/project/2026/2027_cvpr/vjepa2/z_research/scripts/slurm_logs/%x_%j.err
# -----------------------------------------------------------------------------
# run.sh 를 SLURM 으로 제출한다. **이 파일 하나가 두 역할을 한다.**
#
#   sbatch ... z_research/scripts/sbatch.sh      SLURM 이 실행 -> run.sh 를 돈다
#   bash     ... z_research/scripts/sbatch.sh    사람이 실행   -> 스스로를 제출한다
#
# `SLURM_BATCH_SCRIPT` 유무로 갈린다 (SLURM_JOB_ID 는 salloc 안에서도 설정되므로 못 쓴다).
# 제출용 wrapper 를 따로 두지 않는다.
#
#   sbatch --job-name=ip1 --export=ALL,P=intphys1_sliding,D=intphys1_dev z_research/scripts/sbatch.sh
#   sbatch --job-name=prb --export=ALL,P=attn_probe,D=v8,M=vith         z_research/scripts/sbatch.sh
#
#   P=프로토콜 (필수) / D=데이터셋 (필수) / M=모델 (기본 vith) / GPUS=N (기본 4)
#   SET="a.b=1 c.d=null"   병합 config 를 점 경로로 덮어씀 (run.sh 와 같다)
#
# ── SPLIT: probing 을 target 별 job 으로 쪼개기 ───────────────────────────────
#
#   P=attn_probe D=v11 GPUS=2 SPLIT="shape color env" bash z_research/scripts/sbatch.sh
#
#   head 는 (sweep x target x run) 개고 **head 자체는 순차 학습**이다
#   (head 하나를 모든 rank 에 올리는 data-parallel). 그래서 한 job 에 GPU 를 몰아도
#   head 수만큼 시간이 든다. target 으로 쪼개면 그만큼 병렬이고 job 간 통신도 없다.
#   실측(v10, depth=1): 1 GPU 204.7s vs 8 GPU 418.7s — GPU 를 몰면 오히려 느리다.
#
# ⚠️ 캐시 경쟁 — job 이 동시에 시작하면 **같은 토큰 캐시 파일에 함께 쓴다** (잠금 없음).
#    그래서 prep job 하나가 먼저 캐시를 만들고 나머지는 --dependency=afterok 로 붙인다.
#
# ★ --gres 와 GPUS 를 반드시 맞출 것:
#     sbatch --gres=gpu:2 --export=ALL,P=attn_probe,D=v8,GPUS=2 z_research/scripts/sbatch.sh
#
# 데이터가 vll5 노드 로컬(/local_datasets)이라 -w vll5 로 고정한다.
#
# ⚠️ -w vll5 라서 이미 vll5 에서 돌고 있는 run 과 같은 노드에 떨어진다. DDP 포트는
#    run.sh 가 SLURM_JOB_ID 로 갈라 잡으므로 겹쳐 돌려도 된다 (직접 지정 불필요).
#    다만 GPU 는 --gres 로 별도 할당을 받아야 한다. bash 로 띄우면 남의 run 이 쓰는
#    GPU 를 그대로 나눠 쓰게 되어 OOM 이 난다.
#
# ⚠️ probing 의 batch_size 는 global 이라 DDP 가 rank 당 1/ws 로 나눈다.
#    surprise 의 batch_size 는 rank 당이라 effective = batch_size x world_size 다.
#    단일 GPU 결과와 비교하려면 surprise 쪽만 GPU 수로 나눌 것.
#
#   진행 상황:  watch -n 1 bash z_research/scripts/monitor.sh
# -----------------------------------------------------------------------------
set -euo pipefail
PROJECT=/data/hyuntak/project/2026/2027_cvpr/vjepa2

# ── 사람이 bash 로 직접 부른 경우: 제출만 하고 끝낸다 ────────────────────────
# ⚠️ SLURM_JOB_ID 로 가르면 안 된다 — 인터랙티브 salloc 안에서도 설정돼 있다.
#    SLURM_BATCH_SCRIPT 는 **sbatch 로 제출된 배치 job 에서만** 설정된다.
if [[ -z "${SLURM_BATCH_SCRIPT:-}" ]]; then
  cd "$PROJECT"
  : "${P:?P=<프로토콜> 필요}"; : "${D:?D=<데이터셋> 필요}"
  M=${M:-vith}; G=${GPUS:-4}; ME=z_research/scripts/sbatch.sh
  if [[ -z "${SPLIT:-}" ]]; then
    sbatch --job-name="${JOBNAME:-${P%%_*}_$D}" --gres=gpu:"$G" \
           --export=ALL,P="$P",D="$D",M="$M",GPUS="$G",SET="${SET:-}" "$ME"
    exit 0
  fi
  # prep: 캐시만 만든다 (head 1개). 첫 target 을 써서 targets 를 하나만 남긴다
  first=$(echo $SPLIT | awk '{print $1}')
  drop=""; for t in $SPLIT; do [[ $t == "$first" ]] || drop="$drop probing.targets.$t=null"; done
  PREP=$(sbatch --parsable --job-name="prep_$D" --gres=gpu:"$G" \
    --export=ALL,P="$P",D="$D",M="$M",GPUS="$G",SET="probing.fit_groups_sweep=[null] probing.runs=[{train:{model:predictor},eval:[self]}]$drop ${SET:-}" "$ME")
  echo "prep  $PREP   토큰 캐시 생성 + head 1개   (GPUS=$G)"
  for t in $SPLIT; do
    drop=""; for o in $SPLIT; do [[ $o == "$t" ]] || drop="$drop probing.targets.$o=null"; done
    ID=$(sbatch --parsable --job-name="prb_${D}_$t" --gres=gpu:"$G" --dependency=afterok:"$PREP" \
      --export=ALL,P="$P",D="$D",M="$M",GPUS="$G",SET="${drop# } ${SET:-}" "$ME")
    echo "  $t   $ID"
  done
  echo; echo "watch -n 1 bash z_research/scripts/monitor.sh"
  exit 0
fi

# ── 여기부터는 SLURM 이 실행하는 본체 ────────────────────────────────────────
P=${P:?"--export=ALL,P=<프로토콜>,D=<데이터셋> 필요  (목록: bash z_research/scripts/run.sh --list)"}
D=${D:?"--export=ALL,P=<프로토콜>,D=<데이터셋> 필요  (목록: bash z_research/scripts/run.sh --list)"}
M=${M:-vith}
source /data/hyuntak/anaconda3/bin/activate vjepa2
cd "$PROJECT"; export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"
mkdir -p "$PROJECT/z_research/scripts/slurm_logs"

echo "node $(hostname) | gres gpus=${SLURM_GPUS_ON_NODE:-?} | $P / $D / $M | GPUS=${GPUS:-4}"
GPUS=${GPUS:-4} SET="${SET:-}" bash z_research/scripts/run.sh "$P" "$D" "$M"
