"""AR p 에 물체가 어디 있나 그림 (2026-10-26) — vobench_ar_token_where.py 의 where.json.

(1) fig_kymo_<occ>: 운동 축 좌표 (마지막 관측 기준, 칸) × 걸음 의 평균 물체다움 점수. 윗줄 h · 아랫줄 p, 흰 점선 = 참 위치.
(2) fig_cats_<occ>: 걸음마다 p 블록의 범주 비율 (none / at truth / at last seen / on path / elsewhere).

  python z_research/scripts/figures/plot_ar_token_where.py [--name vast70]
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]
MOTS = ["flat_static", "flat_vel", "flat_acc", "flat_dec", "ramp_down", "ramp_up"]
TITLE = {"flat_static": "static", "flat_vel": "flat, constant", "flat_acc": "flat, speeding up",
         "flat_dec": "flat, slowing down", "ramp_down": "down ramp", "ramp_up": "up ramp"}
CAT_LAB = ["none", "at truth", "at last seen", "on path", "elsewhere"]
CAT_COL = ["#bbbbbb", "#1baf7a", "#eb6834", "#2a78d6", "#444444"]


def main(name):
    d = REPO / f"z_analysis/ar_token_where/{name}"
    r = json.load(open(d / "where.json")); E = np.array(r["ky_edges"]); fd = d / "figures"; fd.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42})
    for occ in ("not_late", "late"):
        fig, ax = plt.subplots(2, 6, figsize=(15, 5), sharex=True, sharey=True)
        allv = np.concatenate([np.array(r["groups"][f"{occ}/{m}"][f"ky_{t}"]).ravel() for m in MOTS for t in ("hA", "pA")])
        vmin, vmax = np.nanpercentile(allv, 2), np.nanpercentile(allv, 99.5)
        for j, m in enumerate(MOTS):
            g = r["groups"][f"{occ}/{m}"]
            for i, (tag, lab) in enumerate((("hA", "h (true future)"), ("pA", "p (predicted)"))):
                a = ax[i, j]
                im = a.imshow(np.array(g[f"ky_{tag}"]), origin="lower", aspect="auto", cmap="magma", vmin=vmin, vmax=vmax,
                              extent=[E[0], E[-1], -0.5, 7.5])
                a.plot(g["true_prog"], np.arange(8), color="w", ls="--", lw=1.2)
                a.axvline(0, color="w", lw=0.6, alpha=0.6)
                if i == 0:
                    a.set_title(f"{TITLE[m]} (n={g['n']})", fontsize=8)
                if j == 0:
                    a.set_ylabel(f"{lab}\nfuture step t")
                if i == 1:
                    a.set_xlabel("position along motion (cells, 0 = last seen)")
        fig.colorbar(im, ax=ax, label="mean objectness score", shrink=0.8)
        fig.savefig(fd / f"fig_kymo_{occ}.png", dpi=150, bbox_inches="tight"); fig.savefig(fd / f"fig_kymo_{occ}.pdf", bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(1, 6, figsize=(15, 2.8), sharey=True)
        for j, m in enumerate(MOTS):
            fr = np.array(r["groups"][f"{occ}/{m}"]["pA_frac"]); a = ax[j]; bot = np.zeros(8)
            for c in range(len(CAT_LAB)):
                a.bar(np.arange(8), fr[:, c], bottom=bot, color=CAT_COL[c], label=CAT_LAB[c], width=0.85); bot += fr[:, c]
            a.set_title(TITLE[m], fontsize=8); a.set_xlabel("future step t")
        ax[0].set_ylabel("fraction of clips (p)"); ax[-1].legend(fontsize=7, frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout(); fig.savefig(fd / f"fig_cats_{occ}.png", dpi=150); fig.savefig(fd / f"fig_cats_{occ}.pdf"); plt.close(fig)
    print(fd)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--name", default="vast70"); main(ap.parse_args().name)
