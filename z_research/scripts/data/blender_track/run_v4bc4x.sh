#!/bin/bash
# v4bc4 0.2M 증축 (2026-11-11, 사용자 "v4bc4 distribution 그대로 · scale 만 · 0.2M · 최종 학습 데이터 1M · 최대한 빨리").
#   v4bc4 = v4b (평면 + 빗면, 긴 가림) + v4c (v4b 판 · 벽 장면의 가림막을 미끄러지는 판으로) × 4 가중. 여기서는 v4b 설정 그대로 묶음 × 17 (seed 270 · 280) → v4c 도 그 재료로 × 17.
#   0) 설계 (design_v4bx 스크립트, CPU) 가 끝나길 기다림  1) v4c 설계  2) MPS 렌더 1 차 (GPU 8 × 작업자 8, 작업자마다 세 분할을 돌아가며 --claim → 끝까지 GPU 가 놀지 않게)
#   3) 반쪽 · 깨진 파일 지우고 2 차 (빠진 번호만)  4) 목록: blender_track_v4bx · blender_track_v4cx_slide → blender_track_v4bc4x (v4cx × 4)  5) MPS 끄기
#   setsid nohup bash run_v4bc4x.sh > <root>/logs/run_v4bc4x.out 2>&1 &
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); source "$HERE/env.sh"
REPO=$(cd "$HERE/../../../.." && pwd)
R=$BT_ROOT/v4bc4x; RC=$BT_ROOT/v4cx_slide; export R RC
mkdir -p $R/logs $RC/logs
ts() { TZ=Asia/Seoul date '+%F %T KST'; }
until grep -q DESIGN_DONE $R/logs/design_all.out 2>/dev/null; do sleep 10; done
echo "설계 끝 $(ts)"
while [[ -f $R/WAIT_PROFILE ]]; do sleep 10; done                       # 학습 병목 측정 (GPU) 이 끝나길
cp $R/config_v4bx.yaml $R/config_v4b.yaml                                # v4c_slide.py 가 <src>/config_v4b.yaml 을 읽는다
[[ -s $RC/extra_slide.txt && -s $RC/config_v4c.json ]] || $BT_PY $HERE/v4c_slide.py design --src $R --out $RC --n 1000000 2>&1 | tail -2   # 다시 띄울 때 설계는 건너뜀
echo "V4C_DESIGN_DONE $(ts) | 평면 정지 단계 $(wc -w < $R/extra_static_train_fam.txt) · 가림막 $(wc -w < $RC/extra_slide.txt)"

# 2026-11-11 04:2x: 예전 렌더가 남긴 파이프 파일 때문에 데몬을 안 띄우고 넘어가 10 분 동안 0 편 (CUDA 컨텍스트 오류) → 늘 끄고 지우고 새로 띄운 뒤 확인
mps_on() { local g=$1; export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_render$g CUDA_MPS_LOG_DIRECTORY=/tmp/mps_render${g}_log
  [[ -e $CUDA_MPS_PIPE_DIRECTORY/control ]] && { echo quit | nvidia-cuda-mps-control 2>/dev/null; sleep 1; }
  rm -rf $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY; mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
  CUDA_VISIBLE_DEVICES=$g nvidia-cuda-mps-control -d; sleep 1
  [[ -e $CUDA_MPS_PIPE_DIRECTORY/control ]] || { echo "⚠️ GPU $g MPS 데몬 시작 실패"; exit 1; }; }
one() { local split=$1 w=$2
  case $split in
    train_fam)  echo "--config $R/config_v4bx.json --split train_fam --extra_list $R/extra_static_train_fam.txt" ;;
    rtrain_fam) echo "--config $R/config_v4bxr.json --split rtrain_fam" ;;
    slide)      echo "--config $RC/config_v4c.json --split slide --extra_list $RC/extra_slide.txt" ;;
  esac; }
PER_GPU=${PER_GPU:-11}; NW=$(( 8 * PER_GPU ))   # 04:3x: 8 → 11 (GPU 사용률 60 % 대 · CPU 53 % 놀고 · 스레드 36k/62.7k 측정 뒤)
slot() { local g=$1 w=$2 tag=$3; local c0=$(( (w * 256 / NW) % 254 )); local order
  case $(( w % 3 )) in 0) order="slide train_fam rtrain_fam" ;; 1) order="train_fam rtrain_fam slide" ;; 2) order="rtrain_fam slide train_fam" ;; esac
  ( for s in $order; do
      CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_render$g CUDA_MPS_LOG_DIRECTORY=/tmp/mps_render${g}_log OMP_NUM_THREADS=2 taskset -c $c0-$((c0 + 2)) \
        $BT_BLENDER_PY "$HERE/gen_render.py" $(one $s $w) --worker $w --nworkers $NW --claim >> $R/logs/render_${tag}_w$w.log 2>&1
    done ) & }
alive() { pgrep -fc "gen_rend[e]r.py"; }
count() { echo "평면 $(ls $R/clips 2>/dev/null | grep -c '^train_fam') · 빗면 $(ls $R/clips 2>/dev/null | grep -c '^rtrain_fam') · 가림막 $(ls $RC/clips 2>/dev/null | grep -c '^slide')"; }
render_pass() { local tag=$1
  for g in 0 1 2 3 4 5 6 7; do mps_on $g; for ((j = 0; j < PER_GPU; j++)); do slot $g $((g * PER_GPU + j)) $tag; done; done
  sleep 60
  while [[ -n "$(jobs -rp)" ]]; do sleep 120; echo "  $(ts) 작업자 $(alive) | $(count)"; done; wait; }
echo "MPS 렌더 1 차 $(ts)"
render_pass a
echo "1 차 끝 $(ts) | $(count)"
rm -rf $R/locks $RC/locks
$BT_PY - <<'PY'
import os, glob, numpy as np, decord
bad = 0
for root, pres in ((os.environ["R"], ("train_fam", "rtrain_fam")), (os.environ["RC"], ("slide",))):
    for pre in pres:
        for c in glob.glob(f"{root}/clips/{pre}*.mp4"):
            l = c.replace("/clips/", "/labels/")[:-4] + ".npz"
            try:
                assert os.path.exists(l); np.load(l)["visible_n"]; assert len(decord.VideoReader(c)) == 64
            except Exception:
                for p in (c, l):
                    if os.path.exists(p): os.remove(p)
                bad += 1
        for l in glob.glob(f"{root}/labels/{pre}*.npz"):
            if not os.path.exists(l.replace("/labels/", "/clips/")[:-4] + ".mp4"): os.remove(l); bad += 1
print("반쪽 · 깨진 파일 지움", bad)
PY
for g in 0 1 2 3 4 5 6 7; do CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_render$g sh -c 'echo quit | nvidia-cuda-mps-control' 2>/dev/null; done; sleep 5
echo "2 차 (빠진 번호만) $(ts)"
render_pass b
echo "RENDER_DONE $(ts) | $(count)"
cd $REPO
$BT_PY $HERE/v4a_static.py index_root --out $R --cfg $R/config_v4bx.yaml --split train_fam --cfg-ramp $R/config_v4bxr.yaml --split-ramp rtrain_fam --name blender_track_v4bx | tail -3
$BT_PY $HERE/v4c_slide.py index --out $RC --name blender_track_v4cx_slide | tail -3
$BT_PY $HERE/concat_negimp_lists.py --name blender_track_v4bc4x \
  --part v4bx:data_csv/blender_track_v4bx_negimp/families.json:1 --part v4cx:data_csv/blender_track_v4cx_slide_negimp/families.json:4 | tail -3
echo "V4BC4X_DONE $(ts)"
for g in 0 1 2 3 4 5 6 7; do CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_render$g sh -c 'echo quit | nvidia-cuda-mps-control' 2>/dev/null; done
