#!/usr/bin/env python3
"""BlenderTrack 학습셋 미리보기 영상 — clip 여러 개를 격자로 한 mp4 에 (2026-11-01).

clip 마다: 원본 프레임 + target amodal 윤곽 (주황, 가려져도 물체가 있는 곳) · visible 윤곽 (파랑) · 왼쪽 위에 운동 종류 · 가림막 종류.
  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P z_research/scripts/data/blender_track/preview_mp4.py --meta data_csv/blender_track_v1/train_meta_noslide.npz --n 16 --seed 0 --out <out.mp4>
"""
import argparse
import json
import os
import subprocess

import numpy as np


def edge(m):
    """64×64 bool → 256×256 윤곽 bool."""
    M = np.kron(m, np.ones((4, 4), bool))
    inner = M.copy()
    inner[1:-1, 1:-1] = M[:-2, 1:-1] & M[2:, 1:-1] & M[1:-1, :-2] & M[1:-1, 2:] & M[1:-1, 1:-1]
    return M & ~inner


FONT = {}


def text(img, s, x=3, y=3):
    """작은 글자 없이 PIL 로 왼쪽 위에 영어 라벨."""
    from PIL import Image, ImageDraw
    im = Image.fromarray(img); d = ImageDraw.Draw(im)
    d.rectangle([x - 2, y - 2, x + 6 * len(s) + 2, y + 10], fill=(0, 0, 0))
    d.text((x, y), s, fill=(255, 255, 255))
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import decord
    m = np.load(a.meta)
    root, lab = str(m["root"]), str(m["labels_root"])
    spec_dir = os.path.join(os.path.dirname(root), "specs")
    rng = np.random.default_rng(a.seed)
    clips = [str(c) for c in rng.choice(m["clip"], a.n, replace=False)]
    g = int(np.ceil(np.sqrt(a.n)))
    vids, labs, tags = [], [], []
    for c in clips:
        vids.append(decord.VideoReader(f"{root}/{c}.mp4")[:].asnumpy())
        z = np.load(f"{lab}/{c}.npz")
        labs.append((np.unpackbits(z["amodal"][:, 0], axis=-1).astype(bool), np.unpackbits(z["visible"][:, 0], axis=-1).astype(bool)))
        s = json.load(open(f"{spec_dir}/{c}.json"))
        modes = sorted({p.get("mode", "static") for p in s["panels"] if p["role"] == "event"}) or ["none"]
        tags.append(f"{c} {s['objects'][0]['motion']['kind']} occ:{'+'.join(modes)}")
    T, H = vids[0].shape[0], vids[0].shape[1]
    fps = float(json.load(open(f"{spec_dir}/{clips[0]}.json"))["fps"])
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{g * H}x{g * H}", "-r", str(fps),
                          "-i", "-", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", a.out], stdin=subprocess.PIPE)
    for t in range(T):
        canvas = np.zeros((g * H, g * H, 3), np.uint8)
        for k, (v, (am, vi), tg) in enumerate(zip(vids, labs, tags)):
            f = v[t].copy()
            f[edge(am[t])] = (235, 104, 52)
            f[edge(vi[t])] = (42, 120, 214)
            f = text(f, tg)
            r, c = divmod(k, g)
            canvas[r * H:(r + 1) * H, c * H:(c + 1) * H] = f
        p.stdin.write(canvas.tobytes())
    p.stdin.close(); p.wait()
    print(a.out, clips)


if __name__ == "__main__":
    main()
