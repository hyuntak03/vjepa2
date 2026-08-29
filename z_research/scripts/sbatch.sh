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
#   GSPLIT="static_visible,static_occlusion moving_visible,moving_occlusion" 을 더하면
#   조건까지 쪼갠다 (job 수 = |SPLIT| x |GSPLIT|). GPUS_PROBE 로 본 job 의 GPU 수를
#   따로 준다 (기본 1) — prep 만 GPUS 를 다 쓰고 본 job 은 1장씩이 가장 빠르다.
#
# ⚠️ 캐시 경쟁 — job 이 동시에 시작하면 **같은 토큰 캐시 파일에 함께 쓴다** (잠금 없음).
#    그래서 prep job 하나가 먼저 캐시를 만들고 나머지는 --dependency=afterok 로 붙인다.
#    **prep 은 runs 를 줄이면 안 된다** — base 를 다 안 만들면 뒤 job 이 전부 재추출한다.
#
# ⚠️ output_dir 충돌 — 모든 job 이 같은 경로를 계산하므로 job 마다 OUTDIR 을 따로 준다.
#    (TAG 는 공유 — 캐시를 나눠 뽑으면 안 된다.) 끝나면 merge_probe_runs.py 로 합친다.
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
  # ⚠️ salloc 안에서 제출하면 부모 job 의 SLURM_* 가 --export=ALL 로 딸려가 새 job 의
  #    노드/스텝 배치를 망친다 ("Requested node configuration is not available").
  unset SLURM_JOB_ID SLURM_JOBID SLURM_NODELIST SLURM_JOB_NODELIST \
        SLURM_NNODES SLURM_JOB_NUM_NODES SLURM_NTASKS SLURM_TASKS_PER_NODE \
        SLURM_GPUS_ON_NODE SLURM_JOB_GPUS SLURM_MEM_PER_NODE SLURM_CPUS_ON_NODE
  : "${P:?P=<프로토콜> 필요}"; : "${D:?D=<데이터셋> 필요}"
  M=${M:-vith}; G=${GPUS:-4}; ME=z_research/scripts/sbatch.sh
  if [[ -z "${SPLIT:-}" ]]; then
    sbatch --job-name="${JOBNAME:-${P%%_*}_$D}" --gres=gpu:"$G" \
           --export=ALL,P="$P",D="$D",M="$M",GPUS="$G",SET="${SET:-}" "$ME"
    exit 0
  fi
  # ── output_dir 충돌 방지 ──────────────────────────────────────────────────
  # ⚠️ 모든 job 이 같은 output_dir 을 계산한다 (resolve.py: results_root/<P>__<D>_<M>).
  #    그대로 두면 동시에 끝난 job 들이 summary.json / predictions.json 을 서로
  #    덮어써서 **마지막 하나만 남는다.** job 마다 OUTDIR 을 따로 준다.
  #    TAG 는 건드리지 않는다 — 토큰 캐시는 공유해야 한다.
  BASE=$(DRYRUN=1 bash z_research/scripts/run.sh "$P" "$D" "$M" 2>/dev/null \
         | awk '$1=="output_dir:"{print $2}' | tail -1)
  [[ -n "$BASE" ]] || { echo "output_dir 을 못 구했다 (DRYRUN 실패)"; exit 1; }
  echo "base  $BASE"

  # ── prep: 토큰 캐시를 만든다 ──────────────────────────────────────────────
  # ⚠️ **runs 를 줄이면 안 된다.** schema.expand 가 runs 에서 source 를 만들고
  #    extract() 가 그 source 의 base 만 캐시한다. predictor 만 남기면
  #    predictor.npy 하나만 쓰고 meta.json 에도 그것만 적힌다. 그러면 뒤따르는
  #    job 들이 ctx_masked/target 이 없다고 판단해(TokenCache.matches) **전부
  #    동시에 재추출**하고 같은 파일에 함께 쓴다 (잠금 없음). 캐시가 깨진다.
  #    싸게 만드는 건 epoch 로 한다 — head 자체는 버리는 값이다.
  # ⚠️ 사용자 SET 을 **앞**에 둔다. drop(=targets.X=null)이 뒤에 와야 한다 —
  #    순서가 반대면 SET 안의 probing.targets.shape.* 가 지워진 target 을 빈 dict 로
  #    되살려 column 키가 없는 채로 죽는다 (resolve.py 는 점 경로를 만들며 내려간다).
  drop=""; first=$(echo $SPLIT | awk '{print $1}')
  for t in $SPLIT; do [[ $t == "$first" ]] || drop="$drop probing.targets.$t=null"; done
  PREP=$(OUTDIR="$BASE/_prep" sbatch --parsable --job-name="prep_$D" --gres=gpu:"$G" \
    --export=ALL,P="$P",D="$D",M="$M",GPUS="$G",OUTDIR="$BASE/_prep",SET="${SET:-} probing.fit_groups_sweep=[null] probing.optims.attn_30.num_epochs=1$drop" "$ME")
  echo "prep  $PREP   토큰 캐시 전체(base 3종) + 버리는 head 3개 1ep   (GPUS=$G)"

  # ── 본 job: target x (조건 그룹) ──────────────────────────────────────────
  # GSPLIT="a,b,c d,e,f" 를 주면 조건까지 쪼갠다 (job 수 = |SPLIT| x |GSPLIT|).
  # head 는 job 안에서 **순차** 학습이라 job 을 늘리는 게 GPU 를 몰아주는 것보다 빠르다.
  # 조건으로 쪼개면 runs 는 그대로라 base 3종을 다 요구한다 -> 위 하자가 안 생긴다.
  PG=${GPUS_PROBE:-1}
  for t in $SPLIT; do
    drop=""; for o in $SPLIT; do [[ $o == "$t" ]] || drop="$drop probing.targets.$o=null"; done
    gi=0
    for gs in ${GSPLIT:-__all__}; do
      if [[ $gs == "__all__" ]]; then gset=""; sfx="$t"
      else gset="probing.fit_groups_sweep=[[${gs//,/],[}]]"; sfx="${t}_g${gi}"; fi
      ID=$(sbatch --parsable --job-name="prb_${D}_$sfx" --gres=gpu:"$PG" --dependency=afterok:"$PREP" \
        --export=ALL,P="$P",D="$D",M="$M",GPUS="$PG",OUTDIR="$BASE/$sfx",SET="${SET:-} ${drop# } $gset" "$ME")
      echo "  $sfx   $ID   (GPUS=$PG)  -> $BASE/$sfx"
      gi=$((gi+1))
    done
  done
  echo
  echo "합치기:  python z_research/scripts/analysis/merge_probe_runs.py $BASE"
  echo "진행:    watch -n 1 bash z_research/scripts/monitor.sh"
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
