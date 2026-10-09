#!/usr/bin/env bash
# =============================================================================
# cvpr/env.sh — 절대경로·노드·인터프리터가 사는 **유일한 자리**.
#
#   source cvpr/env.sh          (모든 cvpr/**/run.sh 가 맨 먼저 한다)
#
# 전부 `${VAR:-기본값}` 이라 바깥에서 먼저 export 하면 그 값이 이긴다.
# 레지스트리 yaml (cvpr/registry/*.yaml) 안의 `${CVPR_*}` 는 resolve.py 가 이 값으로 치환한다.
# 새 노드 · 새 계정 · 레포 이동 때 고치는 곳은 이 파일 하나다.
# =============================================================================
export CVPR_ROOT=${CVPR_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}     # cvpr/
export CVPR_REPO=${CVPR_REPO:-$(cd "$CVPR_ROOT/.." && pwd)}                       # 레포 루트 (evals/, analysis/, src/ 가 있는 곳)
export CVPR_PY=${CVPR_PY:-/data/hyuntak/anaconda3/envs/vjepa2/bin/python}        # 인터프리터
export CVPR_TORCHRUN=${CVPR_TORCHRUN:-$(dirname "$CVPR_PY")/torchrun}

# ── 데이터 루트 ──────────────────────────────────────────────────────────────
export CVPR_LOCAL=${CVPR_LOCAL:-/local_datasets}                 # 노드 로컬 (NVMe). IntPhysGen · IntPhys1 프레임
export CVPR_DATA2=${CVPR_DATA2:-/data2}                          # 노드 로컬 (xfs). 토큰 캐시 · 벤치마크 tar 풀어둔 곳 · EK100 영상
export CVPR_NFS=${CVPR_NFS:-/data/hyuntak/project/2026/2027_cvpr} # NFS. 형제 레포 (UnrealEngine, DINO-Foresight, VideoMAEv2, epic-kitchens annotations)
export CVPR_BENCH_SRC=${CVPR_BENCH_SRC:-/data/dataset/world/Benchmarks}   # 벤치마크 원본 tar / InfLevel 폴더
export CVPR_OTHER_CKPT=${CVPR_OTHER_CKPT:-/data/jongseo/project}          # 다른 사용자의 사전학습 체크포인트 (읽기 전용)

# ── 레포 안 자리 ─────────────────────────────────────────────────────────────
export CVPR_CKPT=${CVPR_CKPT:-$CVPR_REPO/checkpoint}             # 릴리즈 체크포인트 (HF 스냅샷)
export CVPR_DATA_CSV=${CVPR_DATA_CSV:-$CVPR_REPO/data_csv}        # index.csv 들
export CVPR_CACHE=${CVPR_CACHE:-$CVPR_LOCAL/world/world_analysis/cache}   # 토큰 캐시 (→ /data2 심볼릭). NFS 에 두지 말 것 (57 MB/s)
export CVPR_RESULTS=${CVPR_RESULTS:-$CVPR_ROOT/results}          # 새 코드 스페이스의 결과 루트. RESULTS_ROOT=legacy 면 옛 z_research/<셋>/exp_results 로 간다
export CVPR_LOGS=${CVPR_LOGS:-$CVPR_ROOT/logs}                   # sbatch 로그

# ── SLURM 기본값 (submit.sh) ────────────────────────────────────────────────
export CVPR_PARTITION=${CVPR_PARTITION:-batch_vll}
export CVPR_NODE=${CVPR_NODE:-vll5}                               # 데이터가 노드 로컬이라 기본 고정. NODE= 로 바꾼다
export CVPR_CPUS_PER_GPU=${CVPR_CPUS_PER_GPU:-12}
export CVPR_MEM_PER_GPU=${CVPR_MEM_PER_GPU:-45G}
export CVPR_TIME=${CVPR_TIME:-0-08:00:00}

# ── 실행 공통 ────────────────────────────────────────────────────────────────
export PYTHONPATH="$CVPR_REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export EVAL_DDP_TIMEOUT_S=${EVAL_DDP_TIMEOUT_S:-7200}            # NCCL timeout. torch 기본 600 s 는 rank 부하 불균형에서 죽는다
export PYTHONUNBUFFERED=1
