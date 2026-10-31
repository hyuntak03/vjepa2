#!/usr/bin/env python3
"""IntPhys 1 late (균일 창) 프레임 폴더를 기록된 창 정의로 만든다 — dev PNG 를 가리키는 링크만 (복사 없음, 몇 초).

창 정의: z_pixel_decoder/runs/_frames/IntPhys1_uni/index.csv 의 `frames` 열 (영상마다 32 장, 0-based → scene_{f+1:03d}.png).
이 판을 처음 만든 규칙 그대로다 (cvpr/registry/datasets.yaml intphys1_late_uni 주석). 영상 목록은 data_csv/intphys1_late_uni 와 같은지 검사한다.
출력: $IP1L_FRAMES/<video_id>/{000000..000031}.png → $IP1_FRAMES/<block>/<quadruplet>/<run>/scene/scene_XXX.png
  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python; $P predictor_lab/setup_intphys1_late.py      (paths.env 를 읽는다)
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from table import ENV  # noqa: E402  (paths.env)

REPO = ENV["LAB_REPO"]
SRC, DST = ENV["IP1_FRAMES"], ENV["IP1L_FRAMES"]
rows = list(csv.DictReader(open(f"{REPO}/z_pixel_decoder/runs/_frames/IntPhys1_uni/index.csv")))
ref = {r["video_id"] for r in csv.DictReader(open(f"{ENV['IP1L_INDEX_DIR']}/index.csv"))}
assert {r["video_id"] for r in rows} == ref, "창 정의와 IntPhys 1 late index 의 영상이 다르다"
n = miss = 0
for r in rows:
    fr = [int(x) for x in r["frames"].split()]
    assert len(fr) == 32, r["video_id"]
    d = f"{DST}/{r['video_id']}"
    os.makedirs(d, exist_ok=True)
    for j, f in enumerate(fr):
        s = f"{SRC}/{r['block']}/{r['quadruplet']}/{r['run']}/scene/scene_{f + 1:03d}.png"
        if not os.path.exists(s):
            miss += 1; continue
        t = f"{d}/{j:06d}.png"
        if os.path.lexists(t):
            os.remove(t)
        os.symlink(s, t); n += 1
print(f"{len(rows)} 영상 · 링크 {n} · 원본 없음 {miss} -> {DST}")
sys.exit(1 if miss else 0)
