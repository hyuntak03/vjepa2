"""phys3d 예시 확인용 — 렌더 장 위에 라벨 (위치 · 속도 화살표 · 보임 · 사건) 을 그려 원본과 나란히 mp4 로 쓴다.

python z_training/tools/phys3d_overlay.py <생성기 --out 폴더>   → 같은 폴더에 <stem>_overlay.mp4
"""
import glob, json, os, subprocess, sys

import numpy as np
from PIL import Image, ImageDraw

root = sys.argv[1]
for js in sorted(glob.glob(os.path.join(root, "*.json"))):
    stem = js[:-5]
    meta = json.load(open(js))
    z = np.load(stem + "_states.npz")
    fr = sorted(glob.glob(stem + "_frames/f*.png"))
    T, N = z["pos"].shape[:2]
    ev = {}
    for e in meta["events"]:
        ev.setdefault(e["frame"], []).append(f'{e["kind"]}:{e["obj"]}')
    tmp = stem + "_ov"
    os.makedirs(tmp, exist_ok=True)
    for t in range(T):
        im = Image.open(fr[t]).convert("RGB")
        ov = im.copy()
        d = ImageDraw.Draw(ov)
        for i in range(N):
            x, y = z["pos"][t, i]
            vx, vy = z["vel"][t, i]
            hid = z["hidden"][t, i]
            col = (255, 40, 40) if hid else (20, 220, 60)
            r = 5
            d.ellipse([x - r, y - r, x + r, y + r], outline=col, width=2, fill=None if hid else col)
            d.line([x, y, x + vx * 0.25, y + vy * 0.25], fill=(255, 255, 0), width=2)
            ratio = z["vis_frac"][t, i] / max(z["inframe_frac"][t, i], 1e-6)
            d.text((x + 7, y - 7), f"{i} v{ratio:.2f}", fill=(0, 0, 0))
        d.text((4, 4), f"t={t}", fill=(0, 0, 0))
        if t in ev:
            d.text((4, 16), " ".join(ev[t]), fill=(200, 0, 0))
        W, H = im.size
        cat = Image.new("RGB", (2 * W, H))
        cat.paste(im, (0, 0)); cat.paste(ov, (W, 0))
        cat.save(os.path.join(tmp, f"{t:04d}.png"))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(meta["fps"] // 2), "-i", os.path.join(tmp, "%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", stem + "_overlay.mp4"], check=True)
    print(stem + "_overlay.mp4")
