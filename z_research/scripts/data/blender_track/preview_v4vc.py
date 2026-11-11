#!/usr/bin/env python3
"""v4vc 묶음 미리보기 (2026-11-02) — 같은 장면의 속도 단계 K 개 + 빈 장면을 한 줄에 나란히 (GIF · 모델이 읽는 간격 2 의 32 샘플).

칸 위에 "speed k (u px/sample)" , 문맥 끝 (샘플 15) 뒤에는 테두리를 빨갛게 (미래 구간). 플롯 라벨은 영어.
  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P preview_v4vc.py --split probe --out predictor_lab/results/v4vc/preview [--families 0 1 2]
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_design_v4vc as G  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "configs/v4vc.yaml"))
    ap.add_argument("--split", default="probe"); ap.add_argument("--out", required=True)
    ap.add_argument("--families", type=int, nargs="*")
    a = ap.parse_args()
    import decord
    cfg = G.load_cfg(a.config, a.split)
    out = cfg["out"]; sd = cfg["splits"][a.split]["seed"]
    os.makedirs(a.out, exist_ok=True)
    fams = a.families if a.families else range(int(cfg["splits"][a.split]["candidates"]))
    n = 0
    for f in fams:
        slots = G.render_slots(cfg, a.split, f)
        cids = [(lev, f"{a.split}{sd}_{i:06d}") for i, lev in slots]
        if not all(os.path.exists(f"{out}/clips/{c}.mp4") for _, c in cids):
            continue
        sp = json.load(open(f"{out}/specs/{cids[0][1]}.json"))
        V = [decord.VideoReader(f"{out}/clips/{c}.mp4")[:].asnumpy()[::2] for _, c in cids]
        frames = []
        for t in range(32):
            row = []
            for (lev, c), v in zip(cids, V):
                im = Image.fromarray(v[t]); d = ImageDraw.Draw(im)
                lab = "empty" if lev == "B" else f"speed {lev} ({sp['velocity']['u_px_per_sample'][lev]:.1f} px/s)"
                d.rectangle([0, 0, 255, 14], fill=(0, 0, 0)); d.text((3, 2), f"{lab}  t={t}", fill=(255, 255, 255))
                if t >= 16:
                    d.rectangle([0, 0, 255, 255], outline=(220, 40, 40), width=3)
                row.append(np.asarray(im))
            frames.append(Image.fromarray(np.concatenate(row, 1)))
        c = sp["cell"]
        name = f"{a.split}_f{f:04d}_{c['objset']}_{c['camera']}_{c['occ']}_{c['target_kind']}.gif"
        frames[0].save(f"{a.out}/{name}", save_all=True, append_images=frames[1:], duration=150, loop=0)
        n += 1
    print("GIF", n, "→", a.out)


if __name__ == "__main__":
    main()
