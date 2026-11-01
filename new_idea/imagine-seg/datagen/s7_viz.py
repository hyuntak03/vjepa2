"""S7 — 보기: 클립마다 GIF (2×2 패널 + 시간선) 와 프레임 띠 그림 (PNG), 그리고 index.html.

패널: 왼쪽 위 원본 / 오른쪽 위 합성 (모델 입력 = JEPA target) /
      왼쪽 아래 원본 + amodal mask (초록) / 오른쪽 아래 합성 + amodal 윤곽 (초록) + 가림막 윤곽 (주황)
시간선: 칸 하나 = 프레임 하나. 위 띠 = 문맥 (파랑) / 예측 (주황), 아래 띠 = 덮임 c(t) (진할수록 많이 덮임),
        흰 세로선 = 재등장 프레임, 테두리 = 현재 프레임.
GIF 는 보기 편하게 4 fps 로 느리게 재생한다 (데이터는 6 fps).

    python datagen/s7_viz.py --clips <dir>/ssv2 <dir>/k400 --tag v0 --out samples/v0
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import unpack  # noqa: E402

BLUE, ORANGE, GREEN = (0x2A, 0x78, 0xD6), (0xEB, 0x68, 0x34), (0x1B, 0xAF, 0x7A)


def contour(img, mask, color, thick=1):
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, cnts, -1, color, thick)


def label(img, text):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    img[:h + 9, :w + 8] = (img[:h + 9, :w + 8] * 0.3).astype(np.uint8)
    cv2.putText(img, text, (4, h + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)


def timeline(rep, t, width):
    M, T_ctx = rep["M"], rep["T_ctx"]
    cov = np.array(rep["coverage"])
    h = 44
    bar = np.full((h, width, 3), 30, np.uint8)
    cw = width / M
    for i in range(M):
        x0, x1 = int(i * cw) + 1, int((i + 1) * cw) - 1
        bar[4:14, x0:x1] = BLUE if i < T_ctx else ORANGE
        g = int(235 - 200 * cov[i])
        bar[16:34, x0:x1] = (g, g, g)
        if i == t:
            cv2.rectangle(bar, (x0 - 1, 2), (x1, 36), (255, 255, 0), 1)
    xr = int((rep["reappear"] + 0.5) * cw)
    cv2.line(bar, (xr, 14), (xr, 36), (255, 255, 255), 2)
    txt = f"t={t:02d}/{M}  {'ctx' if t < T_ctx else 'pred'}  cov={cov[t]:.2f}"
    cv2.putText(bar, txt, (4, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)
    return bar


def frames_2x2(clean, occl, A, O, rep):
    out = []
    for t in range(len(clean)):
        a = clean[t].copy()
        b = occl[t].copy()
        c = clean[t].copy()
        c[A[t]] = (0.5 * c[A[t]] + 0.5 * np.array(GREEN)).astype(np.uint8)
        d = occl[t].copy()
        contour(d, O[t], ORANGE)
        contour(d, A[t], GREEN)
        for img, txt in ((a, "clean"), (b, "occluded (input/target)"), (c, "amodal mask (GT)"),
                         (d, "amodal (green) occluder (orange)")):
            label(img, txt)
        grid = np.vstack([np.hstack([a, b]), np.hstack([c, d])])
        out.append(np.vstack([grid, timeline(rep, t, grid.shape[1])]))
    return out


def strip_png(path, clean, occl, A, O, rep, title):
    M = len(clean)
    step = 1 if M <= 16 else 2
    idx = list(range(0, M, step))
    small = lambda im: cv2.resize(im, (96, 96), interpolation=cv2.INTER_AREA)  # noqa: E731
    row_c, row_o = [], []
    for t in idx:
        c = clean[t].copy()
        contour(c, A[t], GREEN, 2)
        o = occl[t].copy()
        contour(o, A[t], GREEN, 2)
        contour(o, O[t], ORANGE, 2)
        row_c.append(small(c))
        row_o.append(small(o))
    fig, axes = plt.subplots(3, 1, figsize=(0.75 * len(idx) + 1, 5.6),
                             gridspec_kw=dict(height_ratios=[1, 1, 0.8]))
    for ax, row, name in ((axes[0], row_c, "clean + amodal"), (axes[1], row_o, "occluded")):
        ax.imshow(np.hstack(row))
        ax.set_xticks([96 * k + 48 for k in range(len(idx))], [str(t) for t in idx], fontsize=6)
        ax.set_yticks([])
        ax.set_ylabel(name, fontsize=8)
    cov = np.array(rep["coverage"])
    ax = axes[2]
    ax.axvspan(-0.5, rep["T_ctx"] - 0.5, color="#2a78d6", alpha=0.12, lw=0)
    ax.axvspan(rep["T_ctx"] - 0.5, M - 0.5, color="#eb6834", alpha=0.12, lw=0)
    ax.plot(range(M), cov, "-o", color="#2a78d6", ms=3, lw=1.2, label="coverage c(t)")
    ax.axhline(0.95, color="gray", lw=0.6, ls="--")
    ax.axvline(rep["reappear"], color="#1baf7a", lw=1.2, label="reappear")
    ax.text(rep["T_ctx"] / 2 - 0.5, 1.04, "context", ha="center", fontsize=7)
    ax.text((rep["T_ctx"] + M) / 2 - 0.5, 1.04, "prediction", ha="center", fontsize=7)
    ax.set_xlim(-0.5, M - 0.5)
    ax.set_ylim(-0.03, 1.12)
    ax.set_xlabel("frame (6 fps)", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=6, loc="center right")
    fig.suptitle(title, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=Path, nargs="+", required=True)
    ap.add_argument("--tag", default="v0")
    ap.add_argument("--clean_name", default="clean.npy")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max_per_source", type=int, default=10)
    ap.add_argument("--gif_fps", type=float, default=4.0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for root in args.clips:
        n = 0
        for d in sorted(root.iterdir()):
            jp = d / f"sample_{args.tag}.json"
            if n >= args.max_per_source or not jp.exists():
                continue
            rep = json.load(open(jp))
            if rep["status"] != "ok":
                continue
            clip = json.load(open(d / "clip.json"))
            clean = np.load(d / args.clean_name)
            z = np.load(d / f"sample_{args.tag}.npz")
            occl = z["occluded"]
            A = unpack(z["amodal_bits"], z["amodal_shape"])
            O = unpack(z["occluder_bits"], z["occluder_shape"])
            name = f"{d.name}_{args.tag}"
            imageio.mimsave(args.out / f"{name}.gif", frames_2x2(clean, occl, A, O, rep),
                            duration=1000 / args.gif_fps, loop=0)
            title = (f"{d.name} | {clip['label'][:70]} | {rep['mode']} occluder ({rep['occluder']['category']}) | "
                     f"full occlusion {rep['occ_start']}-{rep['occ_end']} | reappear {rep['reappear']}")
            strip_png(args.out / f"{name}.png", clean, occl, A, O, rep, title)
            rows.append((name, clip, rep))
            n += 1
        print(root.name, n)

    th = ("clip", "label", "mode", "occluder", "full occl. frames", "d_occ", "reappear",
          "ctx|pred", "occluder area", "object area", "object speed")
    body = []
    for name, clip, rep in rows:
        o, ob = rep["occluder"], rep["object"]
        cells = (clip["clip_id"], clip["label"], rep["mode"],
                 f"{o['category']} {o['wh_px'][0]}x{o['wh_px'][1]}px v={o['speed_px_per_frame']:g}",
                 f"{rep['occ_start']}-{rep['occ_end']}", rep["d_occ"], rep["reappear"],
                 f"{rep['T_ctx']}|{rep['M'] - rep['T_ctx']}", f"{o['area_max']:.2f}",
                 f"{ob['area_frac_mean']:.3f}", f"{ob['speed_px_per_frame']:.1f}px/f")
        body.append(f"<h3>{html.escape(name)}</h3><img src='{name}.gif'><br><img src='{name}.png' width=100%>"
                    "<table border=1 cellpadding=3><tr>" + "".join(f"<th>{h}</th>" for h in th) + "</tr><tr>"
                    + "".join(f"<td>{html.escape(str(c))}</td>" for c in cells) + "</tr></table><hr>")
    (args.out / "index.html").write_text(
        f"<html><meta charset='utf-8'><body style='font-family:sans-serif'><h2>imagine-seg samples ({args.tag})</h2>"
        "<p>GIF: top-left clean, top-right occluded (model input = JEPA target), bottom-left amodal mask, "
        "bottom-right contours (green amodal, orange occluder). Timeline: blue=context, orange=prediction, "
        f"gray=coverage (darker = more covered), white line = reappear. GIF plays at {args.gif_fps:g} fps "
        "(data is 6 fps).</p>" + "".join(body) + "</body></html>")
    print("->", args.out / "index.html")


if __name__ == "__main__":
    main()
