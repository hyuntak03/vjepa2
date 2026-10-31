#!/usr/bin/env python3
"""transfer_gap Q3 R2 요약 그림 (2026-10-30, [X2]). 결과 JSON 만 읽는다.

(a) E32 · E32b — late 이동, t15 예측 자리가 참 자리 1 칸 안 (%) : 평가 세트 (v11 · VOBench) × 모델 (릴리즈 · v11 PFT · VOBench PFT)
(b) E29 — IntPhys 2 문맥 끝 오프셋 δ 별 VOBench PFT 이득 (permanence · continuity · solidity, permanence 는 95 % CI)
(c) E30 — IntPhys 1 sliding 창 분류별 이득 (skip 2, 창 32 · 16), 95 % CI
(d) E27 — 세트별 ε (PFT · PFT − 자기 공통 이동 · 릴리즈 + 학습 평균 공통 이동)

  /data/hyuntak/anaconda3/envs/vjepa2/bin/python z_research/scripts/figures/plot_transfer_gap_q3_r2.py
  → z_analysis/transfer_gap/results/q3_r2/fig/q3_r2_summary.{png,pdf}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
R = REPO / "z_analysis/transfer_gap/results/q3_r2"
OUT = R / "fig"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#222222", "#666666", "#dddddd"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})


def panel_label(ax, s):
    ax.text(0.5, -0.32, s, transform=ax.transAxes, ha="center", va="top", fontsize=9, color=INK)


def main():
    t = json.load(open(R / "E32_track.json"))["sets"]
    tv = json.load(open(R / "E32_track_v11pft.json"))["sets"]
    e29 = json.load(open(R / "E29_offset_sweep.json"))["panels"]
    e30 = json.load(open(R / "E30_slidingocc.json"))["combos"]
    e27 = json.load(open(R / "E27_offset.json"))["sets"]
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.6), gridspec_kw={"wspace": 0.45, "hspace": 0.75})
    axs = axs.ravel()

    # (a)
    ax = axs[0]
    sets = [("v11sub", "v11 late"), ("vob", "VOBench late")]
    w = 0.26
    for k, (lab, col, src, key) in enumerate((("release", C1, t, "release"), ("v11 PFT", C2, tv, "pft"), ("VOBench PFT", C3, t, "pft"))):
        vals = [100 * src[s]["occ=late"][key]["pred_loc_within1_of_true"]["t15"] for s, _ in sets]
        x = np.arange(len(sets)) + (k - 1) * w
        ax.bar(x, vals, w - 0.03, color=col, label=lab)
        for xi, v in zip(x, vals):
            ax.text(xi, v + 2, f"{v:.0f}", ha="center", va="bottom", fontsize=6.5, color=INK)
    ax.set_xticks(range(len(sets))); ax.set_xticklabels([s for _, s in sets])
    ax.set_ylim(0, 112); ax.set_ylabel("t15 object within 1 cell (%)")
    ax.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=3, handlelength=1)
    ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    panel_label(ax, "(a)")

    # (b)
    ax = axs[1]
    ds = [-60, -30, -20, -10, -5, 0]
    for prop, col in (("permanence", C1), ("continuity", C2), ("solidity", C3)):
        g = [e29[f"d{d}"][prop]["G"] for d in ds]
        ax.plot(ds, g, color=col, lw=2, marker="o", ms=4, label=prop)
        if prop == "permanence":
            lo = [e29[f"d{d}"][prop]["G_ci"][0] for d in ds]; hi = [e29[f"d{d}"][prop]["G_ci"][1] for d in ds]
            ax.fill_between(ds, lo, hi, color=col, alpha=0.15, lw=0)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("context end offset (frames @60fps)"); ax.set_ylabel("PFT − release (pt)")
    ax.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=3, handlelength=1)
    ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    panel_label(ax, "(b)")

    # (c)
    ax = axs[2]
    cats = [("occ_end|moving/occluded", "occ. end,\nmoving"), ("occ_end|static/occluded", "occ. end,\nstatic"),
            ("occ_future", "occ. in\nfuture"), ("none", "other")]
    for k, (combo, col, lab) in enumerate((("skip2_w32", C1, "window 32"), ("skip2_w16", C2, "window 16"))):
        y = np.arange(len(cats)) + (k - 0.5) * 0.36
        v = [e30[combo]["window_acc"][c]["G"] for c, _ in cats]
        lo = [e30[combo]["window_acc"][c]["G_ci"][0] for c, _ in cats]; hi = [e30[combo]["window_acc"][c]["G_ci"][1] for c, _ in cats]
        ax.barh(y, v, 0.32, color=col, label=lab)
        ax.errorbar(v, y, xerr=[np.array(v) - lo, np.array(hi) - v], fmt="none", ecolor=INK, elinewidth=0.8, capsize=1.5)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(range(len(cats))); ax.set_yticklabels([s for _, s in cats], fontsize=7); ax.invert_yaxis()
    ax.set_xlabel("PFT − release (pt, per window)")
    ax.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2, handlelength=1)
    ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    panel_label(ax, "(c)")

    # (d)
    ax = axs[3]
    sn = [("vob_test", "VOB"), ("v11sub", "v11"), ("intphys1_dev", "IP1"), ("grasp_level2_s5", "GRASP"), ("inflevel_continuity_s5", "InfL")]
    for k, (vk, col, lab) in enumerate((("V0_pft", C1, "PFT"), ("V1_pft-ci", C2, "PFT − own shift"), ("V3_rel+cbarVOBtrain", C3, "release + mean shift"))):
        x = np.arange(len(sn)) + (k - 1) * 0.26
        ax.bar(x, [e27[s]["eps"][vk] for s, _ in sn], 0.23, color=col, label=lab)
    ax.axhline(1.0, color=MUTED, lw=0.8)
    ax.set_xticks(range(len(sn))); ax.set_xticklabels([s for _, s in sn], fontsize=7)
    ax.set_ylim(0.78, 1.04); ax.set_ylabel("error ratio vs release")
    ax.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=3, handlelength=1)
    ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    panel_label(ax, "(d)")

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "q3_r2_summary.png", dpi=200, bbox_inches="tight")
    plt.rcParams["pdf.fonttype"] = 42
    fig.savefig(OUT / "q3_r2_summary.pdf", bbox_inches="tight")
    print("→", OUT / "q3_r2_summary.png")


if __name__ == "__main__":
    main()
