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
#   inflevel_*   : /data (NFS) 직독        -> 아무 노드나
#   grasp_level2 : /data2 노드 로컬        -> stage.sh 돌린 노드만
#   intphys1_dev : /local_datasets 노드 로컬 -> vll5
#   v11          : /local_datasets          -> vll5  (P3, 이 세트 범위 밖이지만 실행은 된다)
# =============================================================================
set -uo pipefail
REPO=/data/hyuntak/project/2026/2027_cvpr/vjepa2
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

# 칸이 실패하면 그 랭크들이 GPU 메모리를 쥔 채 남아 다음 칸이 연쇄 OOM 난다. 빠질 때까지 기다린다.
wait_gpu_free() {
  local me; me=$(id -un)
  for i in $(seq 1 72); do
    local used; used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -rn | head -1)
    [[ ${used:-0} -lt 1500 ]] && return 0
    if (( i > 6 )); then
      for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
        [[ "$(ps -o user= -p "$p" 2>/dev/null | tr -d ' ')" == "$me" ]] && kill -9 "$p" 2>/dev/null   # 공용 노드다. 소유자 확인
      done
    fi
    sleep 5
  done
  echo "   !! GPU 메모리가 6 분 넘게 안 빠진다 — 그래도 진행"
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
      wait_gpu_free
      # 배치 상한(max_batch). 기본 48 은 ViT-H(latent 1280) 기준이다.
      #   2.1-g   : 타깃이 5632 차원(4.4 배) -> 창 무관하게 낮춘다
      #   VideoMAE: 창 32 에서 토큰 4096 + 픽셀 디코더라 48 이면 24 GiB 를 요청한다 (2026-09-21 OOM)
      [[ $m == vjepa21g ]] && S="$S surprise.intphys1.max_batch=$MB21"
      [[ $m == videomae2g && $w == 32 ]] && S="$S surprise.intphys1.max_batch=${MBVM:-8}"
      S="$S $EXTRA_SET"
      GPUS=$GPUS TAG="$tag" OUTDIR="$out" SET="$S" \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        bash z_research/scripts/run.sh "$(proto_of "$b")" "$b" "$m" > "$L/${tag}.log" 2>&1
      rc=$?
      if (( rc )); then echo "   !! 실패 (rc=$rc) — $L/${tag}.log"
      else grep -E "BEST\(avg\)|overall" "$L/${tag}.log" | tail -1 | sed 's/^/   /'; fi
    done
  done
done
echo "=== 끝 $(date '+%H:%M:%S')"
