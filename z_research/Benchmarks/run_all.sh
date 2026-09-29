#!/usr/bin/env bash
# =============================================================================
# Garrido 프로토콜 벤치마크 실행기.  **설정의 정본은 `PROTOCOLS.md` 이고 이 파일이 그 구현이다.**
# 채점기는 기존 하네스(`z_research/scripts/run.sh`)이고, 모델은 `model.family` 로만 갈린다
# (`analysis/model_loaders.py`).
#
#   bash z_research/Benchmarks/run_all.sh                          # 안 된 칸 전부
#   BENCHES="intphys1_dev" MODELS="vith" bash ...                  # 골라서
#   WINDOWS="32" bash ...                                          # 창 하나만
#   watch -c -n 1 bash z_research/Benchmarks/monitor.sh
#
# 데이터가 어디 있어야 하는가
#   전부 ${BENCH_ROOT} 아래 (harness/paths.env). 자리는 configs/protocols/datasets.md 가 정본.
#   (옛 기계: inflevel=/data NFS, grasp=/data2 노드 로컬, intphys1_dev·v11=vll5 /local_datasets)
# =============================================================================
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../scripts/harness" && pwd)/env.sh"   # 경로 정본: harness/paths.env
REPO=$VJEPA2_ROOT
cd "$REPO"
L=$REPO/z_research/Benchmarks/exp_results/logs; mkdir -p "$L"

BENCHES=${BENCHES:-"intphys1_dev grasp_level2 inflevel_continuity inflevel_solidity inflevel_gravity"}
MODELS=${MODELS:-"vith vjepa21g videomae2g"}
GPUS=${GPUS:-8}
EXTRA_SET=${EXTRA_SET:-}
TAG_SUFFIX=${TAG_SUFFIX:-}
# 2.1-g 의 타깃 latent 는 5632 차원(ViT-H 1280 의 4.4 배)이라 배치 상한을 내려야 한다.
# max_batch 는 **상한**이라 너무 낮추면 그만큼 느려진다. 8 이 24GB 에서 도는 값이다.
MB21=${MB21:-8}

# --- A.8 탐색 공간 (PROTOCOLS.md §2) -----------------------------------------------
#     Window size = **C + M**.  Context lengths = [2,4,6,8,10] x (C+M)/16 은 프로토콜 yaml 의
#     `context_mult` 가 만든다.  프레임이 모자란 칸(`99//skip < window`)은 하네스가 뺀다.
set_bench() {
  case $1 in
    intphys1_dev)  echo "surprise.intphys1.frame_skips=[2,5,10]" ;;
    grasp_level2)  echo "surprise.intphys1.frame_skips=[2,5,10]" ;;
    inflevel_*)    echo "surprise.intphys1.frame_skips=[5,10,20]" ;;
    *)             echo "" ;;
  esac
}
proto_of() { [[ $1 == v11 ]] && echo surprise_c16t32 || echo intphys1_sliding; }
set_v11_model() { [[ $1 == videomae2g ]] && echo "data.n_frames=16 data.frames_stride=6 surprise.context_length=8 model.window_size=16" || echo ""; }

# --- 창은 **실행을 나눈다** ----------------------------------------------------------
#     공식 코드는 창마다 모델을 그 프레임 수로 짓는다
#     (`eval.py:149  init_model(frames_per_clip=eval_frames_per_clip)`).
#     우리 하네스는 프로세스당 한 번만 지으므로 한 실행에 두 창을 섞으면 안 된다:
#       · VideoMAEv2 는 **즉시 죽는다** — sinusoid 위치표가 생성 시점 고정이라 2048 vs 4096
#       · V-JEPA 는 RoPE 라 죽지는 않지만 **공식과 다른 위치 인코딩**으로 돈다
#     그래서 창마다 따로 돌리고 tag 에 `_w{N}` 을 붙인다.
#     A.8 최고값은 `garrido_rescore.py` 가 창을 가로질러 고른다.
WINDOWS=${WINDOWS:-"16 32"}

# 칸이 실패하면 그 랭크들이 GPU 메모리를 쥔 채 남아 다음 칸이 연쇄 OOM 난다.
# ⚠️ 2026-09-29 정정 — 예전 wait_gpu_free 는 GPU 메모리가 1.5GB 아래로 안 빠지면 **같은 사용자의 GPU
#    프로세스를 전부** kill -9 했다. 다른 벤치마크를 동시에 돌리면 그 랭크까지 죽인다 (IntPhys2 w48·w16 이
#    IntPhys1 w16→w32 전환 순간 SIGKILL 로 죽었다). 이제 칸마다 run.sh 를 **자기 프로세스 그룹**(setsid)
#    으로 띄우고, 끝나면 **그 그룹만** 치운다. 남의 job 은 건드리지 않는다.
run_cell() {   # run_cell <로그> <run.sh 인자...>   (환경변수는 호출자가 넘긴다)
  local log=$1; shift
  setsid bash z_research/scripts/run.sh "$@" > "$log" 2>&1 &
  local pid=$!
  wait "$pid"; local rc=$?
  pkill -9 -g "$pid" 2>/dev/null   # 이 칸이 남긴 랭크만 (pgid == setsid 로 띄운 pid)
  return $rc
}

for b in $BENCHES; do
  for m in $MODELS; do
    wins="$WINDOWS"; [[ $b == v11 ]] && wins="-"
    for w in $wins; do
      if [[ $w == "-" ]]; then
        tag="${b}_${m}${TAG_SUFFIX}"; S="$(set_v11_model "$m")"
      else
        tag="${b}_${m}_w${w}${TAG_SUFFIX}"
        S="$(set_bench "$b") surprise.intphys1.window_sizes=[$w] model.window_size=$w"
      fi
      out="$REPO/z_research/Benchmarks/exp_results/$(proto_of "$b")__${tag}"
      if [[ -f "$out/summary.json" ]]; then echo "-- $tag 이미 있음, 건너뜀"; continue; fi
      echo "=== $b / $m / w$w  ($(date '+%H:%M:%S'))"
      # 배치 상한(max_batch). 기본 48 은 ViT-H(latent 1280) 기준이다.
      #   2.1-g   : 타깃이 5632 차원(4.4 배) -> 창 무관하게 낮춘다
      #   VideoMAE: 창 32 에서 토큰 4096 + 픽셀 디코더라 48 이면 24 GiB 를 요청한다 (2026-09-21 OOM)
      [[ $m == vjepa21g ]] && S="$S surprise.intphys1.max_batch=$MB21"
      [[ $m == videomae2g && $w == 32 ]] && S="$S surprise.intphys1.max_batch=${MBVM:-8}"
      S="$S $EXTRA_SET"
      GPUS=$GPUS TAG="$tag" OUTDIR="$out" SET="$S" \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        run_cell "$L/${tag}.log" "$(proto_of "$b")" "$b" "$m"
      rc=$?
      if (( rc )); then echo "   !! 실패 (rc=$rc) — $L/${tag}.log"
      else grep -E "BEST\(avg\)|overall" "$L/${tag}.log" | tail -1 | sed 's/^/   /'; fi
    done
  done
done
echo "=== 끝 $(date '+%H:%M:%S')"
