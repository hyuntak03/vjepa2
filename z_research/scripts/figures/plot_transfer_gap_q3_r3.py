#!/usr/bin/env python3
"""transfer_gap Q3 R3 요약 그림 (2026-10-30, [X2]). 결과 JSON 만 읽는다.

(a) E35 (i) — 잘라 옮기기 따라감 비 β (가로 = 운동 축) 를 미래 튜블릿별로: VOBench PFT · 처음 보는 물체 PFT. 1 = 장면을 따라 옮김, 0 = 화면 자리 고정.
    t8 은 late 의 미래 앞 샘플이 가려져 참 자리가 기본값이라 뺀다.
(b) E35 (ii) — 보폭 2 끌림 비 λ (모은 비) 를 튜블릿별로 (두 참 열 차 ≥ 1.5 인 block 만, 그 수가 3 이상인 튜블릿). 0 = 문맥 속력, 1 = 보폭 3 자리.
(c) E36 — seed 마다 Garrido 규칙 최고로 다시 고른 (PFT − 릴리즈), 원판과 같이. (복사 결과가 있으면 릴리즈 − 복사 도)
(d) E36 — 칸별 하락 (잡음 seed 평균 − 원판), 두 모델. 원판에서 규칙이 고른 칸은 빗금.

  /data/hyuntak/anaconda3/envs/vjepa2/bin/python z_research/scripts/figures/plot_transfer_gap_q3_r3.py
  → z_analysis/transfer_gap/results/q3/fig/q3_r3_summary.{png,pdf}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
R = REPO / "z_analysis/transfer_gap/results/q3"
OUT = R / "fig"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#222222", "#666666", "#dddddd"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})


def panel_label(ax, s):
    ax.text(0.5, -0.30, s, transform=ax.transAxes, ha="center", va="top", fontsize=9, color=INK)


def main():
    e35 = json.load(open(R / "E35_endprior.json"))
    e36 = json.load(open(R / "E36_garrido_noise.json"))
    cp = json.load(open(R / "E36_copy.json")) if (R / "E36_copy.json").exists() else None
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.4), gridspec_kw={"wspace": 0.42, "hspace": 0.72})
    axs = axs.ravel()
    models = (("vobpft", "VOBench PFT", C2), ("objho", "unseen-object PFT", C3))

    # (a) β by tubelet (x axis)
    ax = axs[0]
    b = e35["crop_follow_beta"]["x"]
    tubs = [f"t{q}" for q in range(9, 16)]
    for m, lab, col in models:
        v = [b[m]["beta_other_tubelets"][t]["beta"] for t in tubs]
        ax.plot(range(9, 16), v, "-o", color=col, lw=2, ms=4, label=lab)
    ax.axhline(1, color=MUTED, lw=0.8, ls="--"); ax.axhline(0, color=MUTED, lw=0.8, ls=":")
    ax.text(15.45, 1.03, "scene", fontsize=6.5, color=MUTED, va="bottom", ha="left"); ax.text(15.45, 0.03, "screen", fontsize=6.5, color=MUTED, va="bottom", ha="left")
    ax.set_xlim(8.7, 16.6); ax.set_ylim(-0.3, 1.4); ax.set_xticks(range(9, 16))
    ax.set_xlabel("future tubelet"); ax.set_ylabel("crop-follow ratio (horizontal)")
    ax.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=2, handlelength=1.5)
    ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    panel_label(ax, "(a)")

    # (b) λ by tubelet
    ax = axs[1]
    lq = e35["stride2_pull_lambda"]["lambda_pooled_by_tubelet"]
    for m, lab, col in models:
        xs, ys = [], []
        for q in range(9, 16):
            r = lq[f"t{q}"]
            if r["n_sep"] >= 3 and r.get(m) is not None and abs(r["true_s3_minus_s2_mean"]) >= 0.5:
                xs.append(q); ys.append(r[m])
        ax.plot(xs, ys, "-o", color=col, lw=2, ms=4, label=lab)
    ax.axhline(1, color=MUTED, lw=0.8, ls="--"); ax.axhline(0, color=MUTED, lw=0.8, ls=":")
    ax.text(15.45, 1.18, "stride-3", fontsize=6.5, color=MUTED, va="bottom", ha="left"); ax.text(15.45, 0.03, "context", fontsize=6.5, color=MUTED, va="bottom", ha="left")
    ax.set_xlim(8.7, 16.9); ax.set_ylim(-0.3, 1.4); ax.set_xticks(range(9, 16))
    ax.set_xlabel("future tubelet"); ax.set_ylabel("stride-2 pull ratio")
    ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    panel_label(ax, "(b)")

    # (c) D per seed
    ax = axs[2]
    pv = e36["per_version"]
    vers = ["orig"] + [v for v in ("src1", "src2", "src3", "src4") if v in pv]
    D = [pv[v]["D_best"] for v in vers]
    ax.plot(range(len(vers)), D, "o", color=C2, ms=6, label="PFT − release")
    if cp:
        rc = []
        for v in vers:
            key = "orig_te(ip1score_final)" if v == "orig" else v
            if key in cp["versions"]:
                rc.append((vers.index(v), pv[v]["release"]["acc"] - cp["versions"][key]["best_acc"]))
        if rc:
            ax.plot([x for x, _ in rc], [y for _, y in rc], "s", color=C1, ms=5, label="release − copy")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(range(len(vers))); ax.set_xticklabels(["orig"] + [f"seed {v[3:]}" for v in vers[1:]])
    ax.set_ylabel("Garrido-best difference (pt)")
    ax.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=2, handlelength=1)
    ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    panel_label(ax, "(c)")

    # (d) per-cell drop
    ax = axs[3]
    wc = e36["winners_curse"]
    cells = ("skip2_w16", "skip2_w32", "skip5_w16")
    w = 0.36
    for k, (m, lab, col) in enumerate((("release", "release", C1), ("vobpft", "VOBench PFT", C2))):
        x = np.arange(len(cells)) + (k - 0.5) * w
        v = [wc[m][c]["delta_mean"] for c in cells]
        bars = ax.bar(x, v, w - 0.04, color=col, label=lab)
        for c, bar in zip(cells, bars):
            if c == wc[m]["selected_orig"]:
                bar.set_hatch("////"); bar.set_edgecolor("white")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(range(len(cells))); ax.set_xticklabels(cells)
    ax.set_ylabel("noise − original (pt)")
    from matplotlib.patches import Patch
    hs = [Patch(color=C1, label="release"), Patch(color=C2, label="VOBench PFT"),
          Patch(facecolor="#bbbbbb", hatch="////", edgecolor="white", label="cell picked on original")]
    ax.legend(handles=hs, frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=3, handlelength=1)
    ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    panel_label(ax, "(d)")

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"q3_r3_summary.{ext}", dpi=200, bbox_inches="tight")
    print("→", OUT / "q3_r3_summary.png")


if __name__ == "__main__":
    main()
