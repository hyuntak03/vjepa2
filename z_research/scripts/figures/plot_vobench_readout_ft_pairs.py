"""자 시간축 — post-FT 전후를 한 패널에 (2026-10-14). (a) release vs release + VOBench FT · (b) vast AR-state ep70 vs + VOBench FT.
x = t0–t15 (t8–t15 예측), 실선 = 정체 정답률, 점선 = '없음' 으로 고른 비율 (argmax). 각 모델의 p 는 그 모델의 p 자로 읽는다.
색: 파랑 = h (정답 쪽, FT 전 모델의 h 자 — encoder 가 frozen 이라 FT 전후 같다) · 회색 = FT 전 p · 주황 = FT 후 p. 회색 띠 = 물체가 실제로 가려진 튜블릿.
범위: split v2 test block · primitive · 물체 있는 가능 영상.

  python z_research/scripts/figures/plot_vobench_readout_ft_pairs.py [--occ late] [--motion all|moving|static]
  → z_research/VOBench_v1/vjepa2_pft/readout_figures/fig_timeline_ft_pairs_<occ>_<motion>.{png,pdf,json}
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
PANELS = (("rel", "relft", "release V-JEPA 2"), ("vast70", "vast70ft", "vast AR-state ep70"))
GREY, ORANGE, BLUE = "#8c8c8c", "#eb6834", "#2a78d6"


def stats(name, occ, motion):
    Z = V.load(name); vid = {v: i for i, v in enumerate(Z["video_id"])}
    G = PR.timeline_stats(Z, vid)
    mots = ("static", "moving") if motion == "all" else (motion,)
    d = {k: np.array([x for m in mots for x in G[(occ, m)][k]]) for k in G[(occ, mots[0])]}
    return {k: 100 * v.mean(0) for k, v in d.items() if k != "hid"} | {"hid": d["hid"].mean(0), "n": len(d["h_id"])}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--occ", default="late"); ap.add_argument("--motion", default="all", choices=["all", "moving", "static"])
    a = ap.parse_args()
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from plot_v11_surprise import MUTED, INK, frame
    test = {r["block_id"] for r in csv.DictReader(open(REPO / "data_csv/vobench_v1_split_v2/index_test.csv"))}
    allrows = V.rows_primitive; V.rows_primitive = lambda: [r for r in allrows() if r["block_id"] in test]
    t, tf = np.arange(16), np.arange(8, 16)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True); js = {}
    for pi, (ax, (base, ft, title)) in enumerate(zip(axes, PANELS)):
        B = stats(base, a.occ, a.motion); js[title] = {"base": base, "ft": ft}
        F = stats(ft, a.occ, a.motion) if list(V.out_dir(ft).glob("shard*.npz")) else None     # FT 판 읽기가 아직 없으면 빈 채로
        frame(ax, (-3, 104), (0, 25, 50, 75, 100))
        for tt in t[B["hid"] > 0.5]:
            ax.axvspan(tt - 0.5, tt + 0.5, color="0.88", lw=0, zorder=0)
        ax.axvline(7.5, color=MUTED, lw=0.8, ls="--")
        ax.plot(t, B["h_id"], "-", color=BLUE, lw=1.2, alpha=0.8, zorder=2)
        if F is None:
            ax.text(11.5, 60, "post-FT: pending", ha="center", fontsize=7.5, color=ORANGE)
        for S, col, key in ((B, GREY, "base"), (F, ORANGE, "ft")):
            if S is None: continue
            ax.plot(tf, S["p_id"], "-", color=col, lw=1.6, marker="o", ms=2.6, zorder=3)
            ax.plot(tf, S["p_none"], ":", color=col, lw=1.6, zorder=3)
            js[title][key + "_p_id"] = S["p_id"].round(1).tolist(); js[title][key + "_p_none"] = S["p_none"].round(1).tolist()
        js[title]["h_id"] = B["h_id"].round(1).tolist(); js[title]["n"] = int(B["n"])
        ax.set_xticks([0, 4, 8, 12, 15]); ax.set_xticklabels(["t0", "t4", "t8", "t12", "t15"], fontsize=7.5)
        ax.set_xlabel("tubelet (t0–t7 context · t8–t15 predicted)", fontsize=7.5)
        ax.set_title(f"{title} vs + VOBench post-FT", fontsize=8.5, color=INK)
        if pi == 0: ax.set_ylabel("% of clips", fontsize=8.5)
        ax.text(0.5, -0.36, f"({'ab'[pi]})", transform=ax.transAxes, ha="center", va="top", fontsize=9.5, color=INK)
    h = [Line2D([], [], color=BLUE, lw=1.2, label="h: identity acc."),
         Line2D([], [], color=GREY, lw=1.6, marker="o", ms=2.6, label="p before FT: identity acc."),
         Line2D([], [], color=ORANGE, lw=1.6, marker="o", ms=2.6, label="p after FT: identity acc."),
         Line2D([], [], color=INK, lw=1.6, ls=":", label="dotted: 'none' chosen")]
    fig.legend(handles=h, loc="upper center", ncol=4, frameon=False, fontsize=7.0, bbox_to_anchor=(0.5, 1.04))
    fig.text(0.995, 0.0, f"{a.occ} · {a.motion} (n={js[PANELS[0][2]]['n']} clips, split v2 test)", ha="right", va="bottom", fontsize=6.6, color=MUTED)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.84, bottom=0.27, wspace=0.08)
    OUT.mkdir(parents=True, exist_ok=True); stem = OUT / f"fig_timeline_ft_pairs_{a.occ}_{a.motion}"
    for ext in (".png", ".pdf"): fig.savefig(stem.with_suffix(ext), dpi=300, bbox_inches="tight", facecolor="white")
    json.dump(js, open(stem.with_suffix(".json"), "w"), indent=1); print("saved →", stem)


if __name__ == "__main__":
    main()
