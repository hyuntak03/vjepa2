#!/usr/bin/env bash
# =============================================================================
# 영상을 **짧은 변 S(기본 320), 비율 유지** 로 재인코딩해 **별도 트리**에 쓴다 (원본 불변).
#
#   bash z_research/scripts/data/resize_videos.sh <src_root> <dst_root> <list.csv> [P=100] [S=320]
#     list.csv : "<abs mp4 path> <label>" (build_video_index.py 산출). src_root 아래 상대경로를 dst_root 에 그대로
#
# 왜 (2026-09-22)
#   · K400 이 720x1280 이라 clip당 디코드 0.6~1 s -> 8 GPU 학습에서 step 의 40% 가 데이터 대기
#   · x264 기본 keyint 250 이라 decord 가 요청 프레임 앞 키프레임부터 최대 8초치를 헛디코드 -> `-g 24`
#   · **재인코딩하면 헤더 길이 = 실제 스트림 길이** -> "헤더만 긴 서브클립에서 decord 무한 spin" 클래스가 사라짐
#   · 320 인 이유: SSv2 가 320, V-JEPA 2 사전학습도 HowTo 320p. RRC(scale 0.3~1.0) 가 자를 여유를 남긴다
# 크롭은 여기서 하지 않는다 — 로드 때 aug.random_resized_crop / square_crop 이 한다 (증강 보존).
# 이미 있는 출력은 건너뛴다 (재실행 안전). 실패는 <dst_root>/_failed.txt.
# =============================================================================
set -uo pipefail
SRC=${1:?src_root}; DST=${2:?dst_root}; LIST=${3:?list.csv}; P=${4:-100}; S=${5:-320}
export SRC DST S
mkdir -p "$DST"; : > "$DST/_failed.txt"
one() {
  f="$1"; rel="${f#$SRC/}"; out="$DST/$rel"
  [ -s "$out" ] && return 0
  mkdir -p "$(dirname "$out")"
  ffmpeg -y -nostdin -loglevel error -threads 1 -i "$f" \
    -vf "scale=w='if(gt(iw,ih),-2,$S)':h='if(gt(iw,ih),$S,-2)'" \
    -c:v libx264 -preset veryfast -crf 23 -g 24 -pix_fmt yuv420p -an -movflags +faststart "$out.tmp.mp4" \
    && mv "$out.tmp.mp4" "$out" || { rm -f "$out.tmp.mp4"; echo "$f" >> "$DST/_failed.txt"; }
}
export -f one
rev "$LIST" | cut -d' ' -f2- | rev | xargs -P "$P" -I{} bash -c 'one "$@"' _ {}
echo "done: $(find "$DST" -name '*.mp4' ! -name '*.tmp.mp4' | wc -l) files, failed $(wc -l < "$DST/_failed.txt")"
