#!/usr/bin/env python3
"""BlenderTrack 렌더 확인 시트 — clip 마다 한 줄, 프레임 8 장 + target 라벨 겹침 (2026-10-31).

겹침: target amodal 칸 (가려져도 물체가 있는 곳) = 주황 윤곽, visible 칸 = 파랑 윤곽, 투영 중심 = 흰 점.
제목 줄: 운동 종류 · 가림 사건 수 · 프레임마다 visible/amodal 비율.
  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P z_research/scripts/figures/plot_blender_track_sheet.py --root <out> --n 12 --save <png>
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams["font.family"] = "DejaVu Sans"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--frames", default="0,25,50,75,100,125,150,179")
    ap.add_argument("--save", required=True)
    a = ap.parse_args()
    clips = sorted(d for d in glob.glob(f"{a.root}/clips/*") if os.path.exists(f"{d}/labels.npz"))[: a.n]
    fr = [int(x) for x in a.frames.split(",")]
    fig, axes = plt.subplots(len(clips), len(fr), figsize=(1.6 * len(fr), 1.75 * len(clips)), squeeze=False)
    for r, d in enumerate(clips):
        z = np.load(f"{d}/labels.npz")
        sp = json.load(open(f"{d}/spec.json"))
        am = np.unpackbits(z["amodal"][:, 0], axis=-1).astype(bool)
        vi = np.unpackbits(z["visible"][:, 0], axis=-1).astype(bool)
        kind = sp["objects"][0]["motion"]["kind"]
        nev = sum(p["role"] == "event" for p in sp["panels"])
        for c, f in enumerate(fr):
            ax = axes[r, c]
            ax.imshow(plt.imread(f"{d}/rgb/{f:04d}.png"))
            ext = (0, 256, 256, 0)
            if am[f].any():
                ax.contour(np.kron(am[f], np.ones((4, 4))), levels=[0.5], colors=["#eb6834"], linewidths=0.8, extent=ext, origin="upper")
            if vi[f].any():
                ax.contour(np.kron(vi[f], np.ones((4, 4))), levels=[0.5], colors=["#2a78d6"], linewidths=0.8, extent=ext, origin="upper")
            cu, cv = z["center_px"][f, 0]
            ax.plot([cu], [cv], ".", color="white", ms=3)
            n_a, n_v = int(z["amodal_n"][f, 0]), int(z["visible_n"][f, 0])
            ax.set_title(f"f{f} vis {n_v}/{n_a}", fontsize=6, pad=1)
            ax.set_xlim(0, 256); ax.set_ylim(256, 0)
            ax.set_xticks([]); ax.set_yticks([])
        axes[r, 0].set_ylabel(f"{os.path.basename(d)}\n{kind} ev{nev}", fontsize=6)
    plt.tight_layout(pad=0.2)
    os.makedirs(os.path.dirname(a.save), exist_ok=True)
    plt.savefig(a.save, dpi=130)
    print("saved", a.save, len(clips))


if __name__ == "__main__":
    main()
