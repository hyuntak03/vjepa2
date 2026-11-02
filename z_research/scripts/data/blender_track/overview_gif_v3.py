#!/usr/bin/env python3
"""BlenderTrack v3 확인용 — 렌더된 모든 상황을 한 GIF 로 (2026-11-01). 칸 128 px · target 윤곽 · 짧은 이름표 · 2 장마다 1 장 (10 fps).
  python overview_gif_v3.py --root <v3 out> --prefix preview90_ --out <out>/preview/overview_all.gif [--cols 9]
"""
import argparse, glob, json, os, subprocess, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from organize_preview_v3 import edge  # noqa: E402

AB = {"primitive": "P", "realistic": "R", "flat": "flat", "ramp": "ramp", "front": "front", "side": "side", "bird": "bird",
      "barrier": "bar", "object": "obj", "static": "st", "moving": "mv", "none": "none"}


def label(img, s):
    from PIL import Image, ImageDraw
    im = Image.fromarray(img); d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 6 * len(s) + 3, 11], fill=(0, 0, 0)); d.text((2, 0), s, fill=(255, 255, 255))
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--prefix", default="preview90_"); ap.add_argument("--out", required=True)
    ap.add_argument("--cols", type=int, default=9); ap.add_argument("--tile", type=int, default=128)
    a = ap.parse_args()
    import decord
    from PIL import Image
    tiles = []
    for f in sorted(glob.glob(f"{a.root}/specs/{a.prefix}*.json")):
        cid = os.path.basename(f)[:-5]
        sp = json.load(open(f)); c = sp["cell"]
        v = decord.VideoReader(f"{a.root}/clips/{cid}.mp4")[:].asnumpy()[::2]
        z = np.load(f"{a.root}/labels/{cid}.npz")
        if z["amodal"].shape[1] > 0 and "target" in [str(r) for r in z["roles"]]:
            am = np.unpackbits(z["amodal"][::2, 0], axis=-1).astype(bool); vi = np.unpackbits(z["visible"][::2, 0], axis=-1).astype(bool)
            v = v.copy()
            for t in range(len(v)):
                v[t][edge(am[t])] = (235, 104, 52); v[t][edge(vi[t])] = (42, 120, 214)
        rm = {"up": "^", "down": "v"}.get(c.get("ramp_mode") or "", "")
        s = f"{AB[c['objset']]} {AB[c['place']]}{rm} {AB[c['camera']]} {AB[c['occ_type']]}-{AB[c['occ_motion']]} t:{AB[c['target']]}"
        tiles.append([label(np.asarray(Image.fromarray(x).resize((a.tile, a.tile), Image.BILINEAR)), s) for x in v])
    n, C = len(tiles), a.cols
    R = (n + C - 1) // C
    T = len(tiles[0])
    tmp = a.out + ".tmp.mp4"
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{C * a.tile}x{R * a.tile}", "-r", "10", "-i", "-",
                          "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv444p", tmp], stdin=subprocess.PIPE)
    for t in range(T):
        canvas = np.zeros((R * a.tile, C * a.tile, 3), np.uint8)
        for k, tl in enumerate(tiles):
            r, cc = divmod(k, C)
            canvas[r * a.tile:(r + 1) * a.tile, cc * a.tile:(cc + 1) * a.tile] = tl[t]
        p.stdin.write(canvas.tobytes())
    p.stdin.close(); p.wait()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-vf", "split[a][b];[a]palettegen=max_colors=200[p];[b][p]paletteuse=dither=bayer:bayer_scale=4",
                    "-loop", "0", a.out], check=True)
    os.remove(tmp)
    print(a.out, n, "상황", f"{C}×{R}", f"{os.path.getsize(a.out) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
