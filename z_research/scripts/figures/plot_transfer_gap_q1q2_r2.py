#!/usr/bin/env python3
"""[X1] R2 요약 그림 — E19 배율 곡선 · E20 위치 판독기 · E25 보간/층 교환 · E23 영역 교환 (2026-10-30).

입력: z_analysis/transfer_gap/results/q1q2_r2/{E19_scale,E20_patch,E25_stitch,E23_swapregion}.json
출력: z_analysis/transfer_gap/results/q1q2_r2/fig/q1q2_r2_summary.{png,pdf}
색: CLAUDE.md §8-4 의 검증된 slot 1–3 만 (#2a78d6 / #eb6834 / #1baf7a). 라벨은 영어, 서술 문장은 캡션 (문서) 으로.

  $P z_research/scripts/figures/plot_transfer_gap_q1q2_r2.py
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]
R = REPO / "z_analysis/transfer_gap/results/q1q2_r2"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED = "#222222", "#777777"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
                     "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42})


def main():
    e19 = json.load(open(R / "E19_scale.json")); e20 = json.load(open(R / "E20_patch.json"))
    e25 = json.load(open(R / "E25_stitch.json")); e23 = json.load(open(R / "E23_swapregion.json"))
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 4.6), gridspec_kw=dict(wspace=0.32, hspace=0.62))
    ax = axs.flatten()
    # (a) 배율 곡선
    ks = [0.5, 1, 2, 3, 4, 8]
    for o, col, mk in (("primitive", C1, "o"), ("realistic", C2, "s")):
        m = e19["objsets"][o]["S2"]["maps"]
        y = [m[f"P{k}"]["R"]["R"] for k in ks]; lo = [m[f"P{k}"]["R"]["lo"] for k in ks]; hi = [m[f"P{k}"]["R"]["hi"] for k in ks]
        ax[0].plot(ks, y, color=col, lw=2, marker=mk, ms=4, label=o)
        ax[0].fill_between(ks, lo, hi, color=col, alpha=0.15, lw=0)
    ax[0].set_xscale("log", base=2); ax[0].set_xticks(ks); ax[0].set_xticklabels(["0.5", "1", "2", "3", "4", "8"])
    ax[0].axhline(1, color=MUTED, lw=0.6, ls=":"); ax[0].axhline(0, color=MUTED, lw=0.6)
    ax[0].set_xlabel("contrast scale k (x gamma_t)"); ax[0].set_ylabel("recovery R (S2)"); ax[0].set_ylim(-0.2, 1.05)
    ax[0].legend(frameon=False, fontsize=7, loc="lower right")
    # (b) 위치 판독기 — 참 자리, 릴리즈 vs PFT
    groups = [("S1", "primitive"), ("S1", "realistic"), ("S2", "primitive"), ("S2", "realistic"), ("S4", "primitive"), ("S4", "realistic")]
    x = np.arange(len(groups)); w = 0.38
    for q, (nm, col, lab) in enumerate((("rel", C1, "release"), ("pft", C2, "PFT"))):
        v = [e20["objsets"][o][f"{s}|{nm}|Otrue|id"]["acc"]["mean"] for s, o in groups]
        ax[1].bar(x + (q - 0.5) * w, v, w - 0.04, color=col, label=lab)
    ax[1].set_xticks(x); ax[1].set_xticklabels([f"{s}\n{o[:4]}" for s, o in groups], fontsize=7)
    ax[1].set_ylabel("patch reader @ true place (%)"); ax[1].set_ylim(0, 100); ax[1].legend(frameon=False, fontsize=7, loc="upper right")
    # (c) 보간 · 층 교환: 존재 대비 크기 vs 사라짐 A
    fam = {"lam": (C1, "o", "interpolation"), "onlyP": (C2, "^", "only one group PFT"), "onlyR": (C3, "s", "only one group release")}
    for v, d in e25["variants"].items():
        key = "lam" if v.startswith("lam") else v.split("_")[0]
        col, mk, _ = fam[key]
        xx = d["both|perm|S4|rho_O"]["mean"]; yy = d["both|late_move|permanence|A"]["mean"]
        ax[2].scatter(xx, yy, color=col, marker=mk, s=28, edgecolor="white", linewidth=0.8, zorder=3)
        if v in ("lam0", "lam100", "onlyP_G1", "onlyR_G3", "lam75"):
            ax[2].annotate({"lam0": "release", "lam100": "PFT", "onlyP_G1": "G1 only", "onlyR_G3": "G3 release", "lam75": "lam .75"}[v],
                           (xx, yy), textcoords="offset points",
                           xytext={"lam0": (-6, 7), "lam100": (-4, -11), "onlyP_G1": (-10, -12), "onlyR_G3": (-50, -2), "lam75": (5, -3)}[v],
                           fontsize=6.5, color=INK)
    for key, (col, mk, lab) in fam.items():
        ax[2].scatter([], [], color=col, marker=mk, s=20, label=lab)
    ax[2].set_xlabel("existence contrast rho_O (late, reappear)"); ax[2].set_ylabel("late vanish A (%)"); ax[2].set_ylim(-16, 108)
    ax[2].legend(frameon=False, fontsize=6.5, loc="upper left", bbox_to_anchor=(0.0, 0.93))
    # (d) 영역 교환
    s = e23["summary"]["late_move_perm|both|A"]
    bars = [("release", "rel"), ("PFT\n@ box", "mixP_Otrue"), ("PFT\n@ path", "mixP_corridor"), ("PFT\n@ rest", "mixP_rest"), ("PFT", "pft")]
    vv = [s[k]["mean"] for _, k in bars]; lo = [s[k]["mean"] - s[k]["lo"] for _, k in bars]; hi = [s[k]["hi"] - s[k]["mean"] for _, k in bars]
    ax[3].bar(range(len(bars)), vv, 0.7, color=C1, yerr=[lo, hi], error_kw=dict(ecolor=MUTED, lw=0.8, capsize=2))
    for i, v in enumerate(vv):
        ax[3].text(i, v + hi[i] + 2, f"{v:.0f}", ha="center", fontsize=6.5, color=INK)
    ax[3].set_xticks(range(len(bars))); ax[3].set_xticklabels([b for b, _ in bars], fontsize=7)
    ax[3].set_ylabel("late vanish A (%)"); ax[3].set_ylim(0, 118)
    for i, a in enumerate(ax):
        a.text(0.5, -0.30, f"({'abcd'[i]})", transform=a.transAxes, ha="center", va="top", fontsize=8)
    out = R / "fig"; out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "q1q2_r2_summary.png", dpi=200, bbox_inches="tight"); fig.savefig(out / "q1q2_r2_summary.pdf", bbox_inches="tight")
    print("→", out / "q1q2_r2_summary.png")


if __name__ == "__main__":
    main()
