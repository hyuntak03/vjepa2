#!/usr/bin/env python3
"""VOBench GIF 를 [원본 영상 | release | AR (vast)] 세 칸으로 다시 붙인다 (2026-10-26, 사용자 지시: "raw video, release, ar 이렇게 gif 배치되도록").

입력 = render_future.py 가 모델별로 만든 GIF (<run>/<release|ar>/<set>/<clip>.gif, 각 [영상 | 그 모델 | encoder→decoder] 256 px 세 칸).
release GIF 의 첫 칸 (영상) · 둘째 칸 (release) 과 ar GIF 의 둘째 칸 (AR) 을 잘라 붙인다 — 다시 디코드하지 않는다.
두 GIF 의 클립 · 프레임 수가 같아야 한다 (같은 clips.py 선택, 32 장).

  python z_pixel_decoder/gif_raw_release_ar.py [--sets visible,late]
산출 <run>/gif_raw_release_ar/<set>/<clip>.gif  (768 × 256, 32 장, 원본 GIF 와 같은 간격)
"""
import argparse
import os

from PIL import Image, ImageSequence

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--run", default=os.path.join(HERE, "runs/vob_grav_i1_vitb_both"))
ap.add_argument("--sets", default="visible,late")
a = ap.parse_args()
W = 256

for s in a.sets.split(","):
    src_r, src_a = os.path.join(a.run, "release", s), os.path.join(a.run, "ar", s)
    out = os.path.join(a.run, "gif_raw_release_ar", s)
    os.makedirs(out, exist_ok=True)
    names = sorted(f for f in os.listdir(src_r) if f.endswith(".gif"))
    assert names == sorted(f for f in os.listdir(src_a) if f.endswith(".gif")), f"{s}: release · ar 클립이 다르다"
    for n in names:
        gr, ga = Image.open(os.path.join(src_r, n)), Image.open(os.path.join(src_a, n))
        fr = [f.convert("RGB") for f in ImageSequence.Iterator(gr)]
        fa = [f.convert("RGB") for f in ImageSequence.Iterator(ga)]
        assert len(fr) == len(fa), (n, len(fr), len(fa))
        frames = []
        for x, y in zip(fr, fa):
            im = Image.new("RGB", (3 * W, W))
            im.paste(x.crop((0, 0, W, W)), (0, 0))              # 원본 영상
            im.paste(x.crop((W, 0, 2 * W, W)), (W, 0))          # release
            im.paste(y.crop((W, 0, 2 * W, W)), (2 * W, 0))      # AR (vast)
            frames.append(im)
        frames[0].save(os.path.join(out, n), save_all=True, append_images=frames[1:], duration=gr.info.get("duration", 125), loop=0)
    print(f"{s}: {len(names)} 개 → {out}")
