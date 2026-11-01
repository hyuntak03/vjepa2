"""S3 진단 — SAM2 후보가 왜 빠졌는지 한 장에: 클립마다 한 줄, 프레임 처음 · 가운데 · 끝에 후보 윤곽과 판정.

    python datagen/s3_diag.py --clips <dir>/ssv2 --out samples/diag_ssv2.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import unpack  # noqa: E402

COLORS = [(0x2A, 0x78, 0xD6), (0xEB, 0x68, 0x34), (0x1B, 0xAF, 0x7A), (255, 220, 0), (220, 0, 220), (0, 220, 220)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows = []
    for d in sorted(args.clips.iterdir()):
        if not (d / "object_cands.npz").exists():
            continue
        clean = np.load(d / "clean.npy")
        z = np.load(d / "object_cands.npz")
        C = unpack(z["cands_bits"], z["cands_shape"])
        obj = json.load(open(d / "object.json"))
        label = json.load(open(d / "clip.json"))["label"]
        M = len(clean)
        tiles = []
        for t in (0, M // 2, M - 1):
            im = clean[t].copy()
            for k, m in enumerate(C):
                cnts, _ = cv2.findContours(m[t].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(im, cnts, -1, COLORS[k % len(COLORS)], 2)
            cv2.putText(im, f"t={t}", (4, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(im)
        verdict = "; ".join(
            f"#{k} {c['reason']}" + (f" (move {c['rel_net_px']:.0f}px, area x{c['area_ratio']:.1f})" if "rel_net_px" in c else "")
            for k, c in enumerate(obj["candidates"]))
        rows.append((d.name, label, obj["status"], np.hstack(tiles), verdict))

    fig, axes = plt.subplots(len(rows), 1, figsize=(9, 2.55 * len(rows)))
    for ax, (name, label, status, img, verdict) in zip(np.atleast_1d(axes), rows):
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{name} [{status}] {label[:60]}\n{verdict}", fontsize=7, loc="left")
    fig.tight_layout()
    fig.savefig(args.out, dpi=80)
    print("->", args.out, len(rows), "clips")


if __name__ == "__main__":
    main()
