"""자 시간축 — release vs release + VOBench post-FT (held-out) 를 나란히 (2026-10-14).
(a) release · (b) release + FT. x = t0–t15, 파랑 = h (target encoder, 각 모델의 h 자) · 주황 = p (t8–t15, 각 모델의 p 자).
실선 = 정체 정답률, 점선 = '없음' 으로 고른 비율 (argmax). 회색 = 물체가 실제로 가려진 튜블릿. split v2 test block · primitive 가능 영상.

  python z_research/scripts/figures/plot_vobench_readout_compare.py [--occ late] [--motion moving|static|all]   # all = 정지 + 이동 합침
  → z_research/VOBench_v1/vjepa2_pft/readout_figures/fig_timeline_rel_vs_relft_<occ>_<motion>.{png,pdf}
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "z_research/scripts/analysis")); sys.path.insert(0, str(REPO / "z_research/scripts/figures"))
import vobench_ar_readout as V  # noqa: E402
import plot_vobench_ar_readout as PR  # noqa: E402

OUT = REPO / "z_research/VOBench_v1/vjepa2_pft/readout_figures"
MODELS = (("rel", "release V-JEPA 2"), ("relft", "+ VOBench post-FT (held-out)"))
LABEL = {"rel": "release V-JEPA 2", "relft": "+ VOBench post-FT (held-out)", "vast70": "vast AR-state ep70", "ar40": "Ariel AR ep40"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--occ", default="late"); ap.add_argument("--motion", default="moving", choices=["moving", "static", "all"])
    ap.add_argument("--models", default="rel relft", help="두 패널의 readout 이름 (vobench_<name>), 예: 'rel vast70'")
    a = ap.parse_args()
    models = [(n, LABEL.get(n, n)) for n in a.models.split()]
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from plot_v11_surprise import BLUE, ORANGE, MUTED, INK, frame
    test = {r["block_id"] for r in csv.DictReader(open(REPO / "data_csv/vobench_v1_split_v2/index_test.csv"))}
    allrows = V.rows_primitive; V.rows_primitive = lambda: [r for r in allrows() if r["block_id"] in test]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True); js = {}
    t = np.arange(16); tf = np.arange(8, 16)
    for pi, (ax, (name, title)) in enumerate(zip(axes, models)):
        Z = V.load(name); vid = {v: i for i, v in enumerate(Z["video_id"])}
        G = PR.timeline_stats(Z, vid)
        mots = ("static", "moving") if a.motion == "all" else (a.motion,)
        d = {k: [x for m in mots for x in G[(a.occ, m)][k]] for k in G[(a.occ, mots[0])]}
        frame(ax, (-3, 104), (0, 25, 50, 75, 100))
        for tt in t[np.mean(d["hid"], 0) > 0.5]:
            ax.axvspan(tt - 0.5, tt + 0.5, color="0.88", lw=0, zorder=0)
        ax.axvline(7.5, color=MUTED, lw=0.8, ls="--")
        curves = {"h_id": (t, BLUE, "-"), "h_none": (t, BLUE, ":"), "p_id": (tf, ORANGE, "-"), "p_none": (tf, ORANGE, ":")}
        js[name] = {"n": len(d["h_id"])}
        for k, (x, col, ls) in curves.items():
            y = 100 * np.mean(d[k], 0); js[name][k] = y.round(1).tolist()
            ax.plot(x, y, ls, color=col, lw=1.5, marker="o" if ls == "-" else None, ms=2.5)
        ax.set_xticks([0, 4, 8, 12, 15]); ax.set_xticklabels(["t0", "t4", "t8", "t12", "t15"], fontsize=7.5)
        ax.set_xlabel("tubelet (t0–t7 context · t8–t15 predicted)", fontsize=7.5)
        ax.set_title(f"{title}", fontsize=8.5, color=INK)
        if pi == 0: ax.set_ylabel("% of clips", fontsize=8.5)
        ax.text(0.5, -0.36, f"({'ab'[pi]})", transform=ax.transAxes, ha="center", va="top", fontsize=9.5, color=INK)
    h = [Line2D([], [], color=BLUE, ls="-", marker="o", ms=2.5, lw=1.5, label="h: identity acc."),
         Line2D([], [], color=BLUE, ls=":", lw=1.5, label="h: 'none' chosen"),
         Line2D([], [], color=ORANGE, ls="-", marker="o", ms=2.5, lw=1.5, label="p: identity acc."),
         Line2D([], [], color=ORANGE, ls=":", lw=1.5, label="p: 'none' chosen")]
    fig.legend(handles=h, loc="upper center", ncol=4, frameon=False, fontsize=7.2, bbox_to_anchor=(0.5, 1.04))
    fig.text(0.995, 1.0, f"{a.occ} · {a.motion} (n={js[models[0][0]]['n']} clips)", ha="right", va="bottom", fontsize=6.8, color=MUTED)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.84, bottom=0.27, wspace=0.08)
    OUT.mkdir(parents=True, exist_ok=True); stem = OUT / (f"fig_timeline_rel_vs_relft_{a.occ}_{a.motion}" if a.models == "rel relft" else f"fig_timeline_{'_vs_'.join(n for n, _ in models)}_{a.occ}_{a.motion}")
    for ext in (".png", ".pdf"): fig.savefig(stem.with_suffix(ext), dpi=300, bbox_inches="tight", facecolor="white")
    json.dump(js, open(stem.with_suffix(".json"), "w"), indent=1); print("saved →", stem)


if __name__ == "__main__":
    main()
