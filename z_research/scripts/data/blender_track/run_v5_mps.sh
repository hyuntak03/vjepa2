#!/bin/bash
# v5id 렌더를 CUDA MPS 로 (2026-11-09 19:0x). 왜 [측정]: MPS 없이 GPU 당 작업자 8 개는 시분할이라 GPU 전력 150 / 490 W, GPU 당 분당 약 34 편.
#   GPU 7 하나에 MPS 를 켜자 같은 작업자 8 개로 2 분에 327 편 (기존 GPU 6 은 64 편) — 약 5 배. 영상 · 라벨 정상 (64 장, 밝기 · 고주파 통계 같음, proj_err 0).
#   GPU 마다 MPS 데몬을 따로 (파이프 /tmp/mps_render<g>) 띄워 렌더 작업자만 MPS 를 탄다 (다른 CUDA 프로세스는 영향 없음).
#   작업자 번호 · CPU 묶음은 run_v5_all.sh 와 같다 (평면 w → 칸 w, 빗면 → 24 + w, 가림막 → 48 + w, 칸마다 CPU 4 개).
#   GPUS="0 1 2 3 4 5 6" R=<root> bash run_v5_mps.sh   (GPU 7 은 시험 때 이미 MPS 작업자가 돈다)
#   끝: 모든 작업자 종료 → 잠금 지움 · 반쪽 파일 검사 → 한 번 더 (빠진 번호만) → 학습 목록 → MPS 데몬 끄기.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); source "$HERE/env.sh"
R=${R:-$BT_ROOT/v5id}; export R; mkdir -p "$R/logs" "$R/locks"
ts() { TZ=Asia/Seoul date '+%F %T KST'; }
mps_on() { local g=$1; export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_render$g CUDA_MPS_LOG_DIRECTORY=/tmp/mps_render${g}_log
  mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
  [[ -e $CUDA_MPS_PIPE_DIRECTORY/control ]] || CUDA_VISIBLE_DEVICES=$g nvidia-cuda-mps-control -d; }
run() { local g=$1 cfg=$2 split=$3 w=$4 nw=$5 ex=$6 slot=$7 tag=$8; local c0=$(( slot * 4 ))
  CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_render$g CUDA_MPS_LOG_DIRECTORY=/tmp/mps_render${g}_log OMP_NUM_THREADS=2 taskset -c $c0-$((c0 + 3)) \
    $BT_BLENDER_PY "$HERE/gen_render.py" --config $cfg --split $split --worker $w --nworkers $nw --claim $ex > $R/logs/${split}_${tag}$w.log 2>&1 & }
gpu_workers() { local g=$1 tag=$2
  for j in 0 1 2; do w=$((g + 8 * j)); run $g $R/config_v5id.json train_fam $w 24 "" $w $tag; run $g $R/config_v5idr.json rtrain_fam $w 24 "" $((24 + w)) $tag; done
  for j in 0 1; do w=$((g + 8 * j)); run $g $R/config_v5c.json slide $w 16 "--extra_list $R/extra_slide.txt" $((48 + w)) $tag; done; }
alive() { pgrep -fc "gen_rend[e]r.py"; }
echo "MPS 렌더 $(ts) | GPU ${GPUS:=0 1 2 3 4 5 6}"
for g in $GPUS; do mps_on $g; gpu_workers $g m; done
sleep 60
until [[ $(alive) -eq 0 ]]; do sleep 120; done
echo "1 차 끝 $(ts) | 평면 $(ls $R/clips | grep -c '^train_fam') · 빗면 $(ls $R/clips | grep -c '^rtrain_fam') · 가림막 $(ls $R/clips | grep -c '^slide')"
rm -rf $R/locks; mkdir -p $R/locks
$BT_PY - <<'PY'
import os, glob, numpy as np, decord
R = os.environ.get("R", "/workspace/dataset/training_data/blender_track/v5id"); bad = 0
for pre in ("train_fam", "rtrain_fam", "slide"):
    for c in glob.glob(f"{R}/clips/{pre}*.mp4"):
        l = c.replace("/clips/", "/labels/")[:-4] + ".npz"
        try:
            assert os.path.exists(l); np.load(l)["visible_n"]; assert len(decord.VideoReader(c)) == 64
        except Exception:
            for p in (c, l):
                if os.path.exists(p): os.remove(p)
            bad += 1
    for l in glob.glob(f"{R}/labels/{pre}*.npz"):
        if not os.path.exists(l.replace("/labels/", "/clips/")[:-4] + ".mp4"): os.remove(l); bad += 1
print("반쪽 · 깨진 파일 지움", bad)
PY
for g in 0 1 2 3 4 5 6 7; do mps_on $g; gpu_workers $g s; done
sleep 60
until [[ $(alive) -eq 0 ]]; do sleep 60; done
echo "V5_RENDER_ALL_DONE $(ts) | 평면 $(ls $R/clips | grep -c '^train_fam') · 빗면 $(ls $R/clips | grep -c '^rtrain_fam') · 가림막 $(ls $R/clips | grep -c '^slide')"
$BT_PY $HERE/index_v5id.py --root $R --name blender_track_v5id
echo "V5_ALL_DONE $(ts)"
for g in 0 1 2 3 4 5 6 7; do CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_render$g sh -c 'echo quit | nvidia-cuda-mps-control' 2>/dev/null; done
