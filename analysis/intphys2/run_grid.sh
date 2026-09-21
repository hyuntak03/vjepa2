#!/usr/bin/env bash
# =============================================================================
# IntPhys 2 — 논문 D.3 격자를 돈다. **설정의 정본은 `README.md` 이고 이 파일이 그 구현이다.**
#
#   GPUS=8 bash analysis/intphys2/run_grid.sh                    # 안 된 칸 전부
#   MODELS="vith" WINDOWS="16 32" bash analysis/intphys2/run_grid.sh
#
# 창은 `--set` 으로 준다 — **창마다 yaml 을 뜨지 않는다** (configs/README.md 규칙).
#   context 길이 C = 창 x {1/4, 3/8, 1/2, 5/8, 3/4, 7/8}
#   (공식 vjepa_2.yaml M=48 -> [12,18,24,30,36,42], videomaev2.yaml M=16 -> [4,6,8,10,12,14])
#
# 데이터는 노드 로컬 /data2/local_datasets/world/Benchmark/IntPhys2/Main (stage.sh 가 푼다).
# =============================================================================
set -uo pipefail
REPO=/data/hyuntak/project/2026/2027_cvpr/vjepa2
cd "$REPO"
PY=/data/hyuntak/anaconda3/envs/vjepa2/bin
OUT=$REPO/z_research/Benchmarks/exp_results/intphys2
L=$REPO/z_research/Benchmarks/exp_results/logs; mkdir -p "$L"

MODELS=${MODELS:-"vith vjepa21g videomae2g"}
# video_batch: 영상을 몇 개 쌓을지. **기본 1 (보고용)**.
#   >1 이면 배치 모양이 바뀌어 autocast fp16 reduction 순서가 달라지고, 실측 max|Δsurprise| = 7.3e-5 다.
#   IntPhys 2 는 쌍 마진 중앙값이 8e-4 인데 **쌍의 약 6% 가 그 잡음보다 작은 마진**으로 갈려
#   정확도가 ±0.5pt 흔들린다 (2026-09-22 실측). 탐색용으로만 올리고, 보고 수치는 1 로 낸다.
VIDEO_BATCH=${VIDEO_BATCH:-1}
GPUS=${GPUS:-8}
# video_batch: 영상을 몇 개 쌓을지 (2026-09-22 Garrido 하네스에서 이식).
#   연산 시간 약 1.5 배 이득. 다만 배치 모양이 바뀌면 autocast fp16 reduction 순서가 달라져
#   실측 max|Δsurprise| = 7.3e-5 다. 이게 문제가 되는지는 **그 벤치의 쌍 마진**에 달렸다:
#     IntPhys 1 (ViT-H 88.89%) 중앙 마진 5.4e-3 -> 잡음/마진 1.4%,  마진<잡음 0.6%   => 안전
#     IntPhys 2 (ViT-H ~54%)   중앙 마진 7.9e-4 -> 잡음/마진 9.2%,  마진<잡음 6.7%   => 위험
#   모델이 못 푸는 벤치일수록 마진이 작아 같은 잡음이 크게 보인다. **IntPhys 2 는 1 로 둔다.**
VIDEO_BATCH=${VIDEO_BATCH:-1}

# 논문 D.3
#   V-JEPA / V-JEPA 2 : "window sizes of 16, 32 or 48 frames with a fixed framerate of 6 fps"
#   VideoMAEv2        : "sincos embeddings ... keep the window size fixed at 16 and vary the
#                        framerate at 2, 3 and 6 fps"  -> 창이 아니라 fps 를 쓴다
win_for() { [[ $1 == videomae2g ]] && echo "16" || echo "${WINDOWS:-16 32 48}"; }

# window 당 토큰 = (w/2) x (256/16)^2.  배치 상한을 그에 반비례로 잡는다 (24GB 기준).
batch_for() {
  case "$1/$2" in
    vjepa21g/*) echo 2 ;;                      # 타깃 latent 5632 차원
    */16) echo 16 ;; */32) echo 12 ;; */48) echo 8 ;; *) echo 8 ;;
  esac
}

for m in $MODELS; do
  base=$REPO/analysis/intphys2/configs/bench_intphys2_main_${m}.yaml
  [[ -f $base ]] || { echo "!! config 없음: $base"; continue; }
  for w in $(win_for "$m"); do
    tag="bench_intphys2_main_${m}_w${w}"
    [[ -f "$OUT/$tag/summary.json" ]] && { echo "-- $tag 이미 있음, 건너뜀"; continue; }
    C=$($PY/python -c "print([int($w*f) for f in (0.25,0.375,0.5,0.625,0.75,0.875)])")
    echo "=== $m / w$w  C=$C  ($(date '+%H:%M:%S'))"
    PYTHONPATH=$REPO $PY/torchrun --nproc-per-node=$GPUS --master_port=$((24000 + RANDOM % 10000)) \
      -m analysis.intphys2.eval --config "$base" \
        --set tag=$tag \
        --set surprise.window_size=$w \
        --set "surprise.context_length_sweep=$C" \
        --set surprise.context_length=$((w/2)) \
        --set surprise.max_window_batch=$(batch_for "$m" "$w") \
        --set surprise.video_batch=$VIDEO_BATCH \
        --set surprise.video_batch=$VIDEO_BATCH \
      > "$L/${tag}.log" 2>&1
    rc=$?
    (( rc )) && echo "   !! 실패 (rc=$rc) — $L/${tag}.log" || echo "   완료 $(date '+%H:%M:%S')"
  done
done
echo "=== 끝 $(date '+%H:%M:%S')"
