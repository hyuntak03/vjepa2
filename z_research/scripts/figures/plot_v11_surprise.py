#!/usr/bin/env python3
"""v11 채점 그림 3종 — report.json 하나에서 전부 그린다.

  k         가림 길이 dose-response — 정확도 (k=0 은 가림막 자체가 없다)
  sens      같은 축의 sensitivity / bias 분해 — 편향을 걷어낸 탐지력
  rampflat  경사면(등가속도) vs 평지(등속) — 가림 유무별로
  occlusion 가림 유무 — 운동 팔별로

report.json 은 생성 시 summary.json 과 대조 검증된다(`verified`). 이 스크립트는
그 플래그를 확인하고, cells 에서 overall 을 다시 합산해 한 번 더 맞춘다.

  python z_research/scripts/figures/plot_v11_surprise.py \
      --report z_research/IntPhysGenV11/exp_results/report.json \
      --outdir z_research/IntPhysGenV11/figures/surprise
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, BAND = "#000000", "#3b3b3b", "#9a9a9a", "#efefef"
VIOL = ["vanish", "shape", "color"]
VLAB = {"vanish": "Object permanence", "shape": "Shape consistency", "color": "Colour consistency"}
VSHORT = {"vanish": "Permanence", "shape": "Shape", "color": "Colour"}
VCOL = {"vanish": BLUE, "shape": ORANGE, "color": GREEN}
PANEL = ["(a)", "(b)", "(c)"]

mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.2, "ytick.major.size": 2.2,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def tint(h, f):
    r, g, b = (int(h[i:i + 2], 16) for i in (1, 3, 5))
    return "#%02x%02x%02x" % tuple(round(c + (255 - c) * f) for c in (r, g, b))


def frame(ax, ylim=(0, 108), yt=(0, 25, 50, 75, 100)):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_axisbelow(True)                     # 격자는 데이터 뒤로
    ax.grid(axis="y", color=MUTED, alpha=0.30, lw=0.4)
    ax.set_ylim(*ylim); ax.set_yticks(yt)
    ax.tick_params(labelsize=8.5, colors=INK2, pad=2.0)
    # 축선·눈금은 **데이터 위로**. 막대가 y=0 에서 시작하고 음영이 축까지 닿아
    # 아래·왼쪽 spine 이 가려진다.
    for sp in ax.spines.values():
        sp.set_zorder(30)
    for ax_ in (ax.xaxis, ax.yaxis):
        ax_.set_zorder(30)
        for t in ax_.get_major_ticks():
            t.tick1line.set_zorder(30); t.tick2line.set_zorder(30)


def save(fig, out: Path, name):
    out.mkdir(parents=True, exist_ok=True)
    for ext in (".pdf", ".png"):
        fig.savefig((out / name).with_suffix(ext), dpi=400, bbox_inches="tight", facecolor="white")
    print(f"  [saved] {out / name}.pdf  + .png")


# ---------------------------------------------------------------- k dose
SPEC = {"acc":         ("pairwise acc. (%)", (0, 108), (0, 25, 50, 75, 100)),
        "sensitivity": ("sensitivity (pp)",  (-5, 58), (0, 25, 50)),
        "bias":        ("bias (pp)",         (-2, 58), (0, 25, 50))}


def _fig_k(R, out, metrics, name, subtitle=None, width=7.0):
    dose = {(d["sym_k"], d["violation"]): d for d in R["scoring"]["dose"]}
    ks = sorted({d["sym_k"] for d in R["scoring"]["dose"]}, key=int)
    fig, axes = plt.subplots(1, len(metrics), figsize=(width, 2.9), squeeze=False)
    axes = axes[0]
    for pi, (ax, metric) in enumerate(zip(axes, metrics)):
        lab, ylim, yt = SPEC[metric]
        frame(ax, ylim, yt)
        # k=0 은 가림막이 아예 없다 — 나머지와 성질이 다르므로 띄우고 음영으로 가른다.
        # 음영 = "가림막이 있는 구간". 축 아래 이탤릭 라벨로 명시한다 (다른 그림과 같은 관례).
        xs = {k: (0 if k == "0" else 1.6 + int(k) - 1) for k in ks}
        occ = [xs[k] for k in ks if k != "0"]
        ax.axvspan(min(occ) - 0.5, max(occ) + 0.5, color=BAND, lw=0, zorder=0)
        # 같은 x 에서 값이 가까우면 라벨이 겹친다 — 순위대로 띄운다
        at = {}
        for k in ks:
            vals = sorted(((dose[(k, v)][metric], v) for v in VIOL if (k, v) in dose))
            for rank, (_, v) in enumerate(vals):
                at[(k, v)] = rank
        for v in VIOL:
            y = [dose[(k, v)][metric] for k in ks if (k, v) in dose]
            x = [xs[k] for k in ks if (k, v) in dose]
            off = [at[(k, v)] for k in ks if (k, v) in dose]
            ax.plot(x[1:], y[1:], "-o", color=VCOL[v], ms=5, lw=1.4, zorder=3,
                    mec="white", mew=0.8, label=VLAB[v])
            ax.plot(x[:1], y[:1], "o", color=VCOL[v], ms=6, zorder=3, mec="white", mew=0.8)
            ax.plot(x[:2], y[:2], ":", color=tint(VCOL[v], 0.45), lw=1.0, zorder=2)
            step = 4.2 if metric == "acc" else 2.6
            for xx, yy, oo in zip(x, y, off):
                ax.annotate(f"{yy:.0f}", (xx, yy + 1.6 + step * oo), ha="center", va="bottom",
                            fontsize=6.6, color=VCOL[v], zorder=5)
        if metric == "acc":
            ax.axhline(50, color=MUTED, lw=0.8, ls=(0, (2.5, 2.5)), zorder=1)
            ax.annotate("chance", (-0.5, 51.5), ha="left", va="bottom",
                        fontsize=6.4, color=MUTED, zorder=1)
        else:
            ax.axhline(0, color=MUTED, lw=0.8, ls=(0, (2.5, 2.5)), zorder=1)
        ax.set_xticks(list(xs.values()))
        ax.set_xticklabels(["0" if k == "0" else k for k in ks], fontsize=8.4, color=INK)
        ax.set_xlim(-0.55, max(xs.values()) + 0.55)
        ax.tick_params(axis="x", length=0, pad=3)
        # 음영이 무엇인지 축 아래에 밝힌다
        ax.annotate("no occluder", (xs["0"], -0.115), xycoords=("data", "axes fraction"),
                    ha="center", va="top", fontsize=7.6, color=INK2, style="italic")
        ax.annotate("occluded — hidden frames per side (k)",
                    (float(np.mean(occ)), -0.115), xycoords=("data", "axes fraction"),
                    ha="center", va="top", fontsize=7.6, color=INK2, style="italic")
        ax.set_ylabel(lab, fontsize=9.5)
        if len(metrics) > 1:
            ax.text(0.5, -0.215, PANEL[pi], transform=ax.transAxes, ha="center", va="top",
                    fontsize=9.5, color=INK)
    h = [Line2D([], [], color=VCOL[v], marker="o", ms=5, lw=1.4, mec="white",
                mew=0.8, label=VLAB[v]) for v in VIOL]
    # 범례와 부제가 겹치지 않게 두 줄로 쌓는다. 범례가 위, 부제가 아래.
    yanc = 1.135 if subtitle else 1.015
    fig.legend(handles=h, loc="upper center", ncol=3, frameon=False, fontsize=8.6,
               handlelength=1.6, handletextpad=0.5, columnspacing=1.6, bbox_to_anchor=(0.5, yanc))
    if subtitle:
        fig.text(0.5, 1.035, subtitle, ha="center", va="top", fontsize=7.6,
                 color=INK2, style="italic")
    fig.subplots_adjust(left=0.085 if len(metrics) > 1 else 0.105, right=0.995,
                        top=0.925, bottom=0.205 if len(metrics) > 1 else 0.185, wspace=0.26)
    save(fig, out, name)


def fig_k(R, out):
    """정확도만. 편향과 탐지가 섞여 있는 값이라 sensitivity 와 따로 둔다."""
    _fig_k(R, out, ["acc"], "fig_k_dose", width=4.2)


def fig_sens(R, out):
    """탐지력과 편향의 분해.

    한 쌍의 두 방향(A->B, B->A)은 같은 두 후보를 역할만 바꿔 쓴다. 그래서 고정된
    편향만 작동하면 두 방향의 합이 100 이 되고 sensitivity 가 0 이 된다.
    실측(vanish k=4): 빈->나타남 95.2% / 물체->사라짐 13.7% -> acc 54.5 지만 sensitivity 4.5.
    """
    _fig_k(R, out, ["sensitivity", "bias"], "fig_k_sensitivity",
           subtitle=r"per shape/colour pair with both directions:   "
                    r"sensitivity $=\frac{a_{A\to B}+a_{B\to A}}{2}-50$  (bias removed)"
                    r"      bias $=\frac{|a_{A\to B}-a_{B\to A}|}{2}$")


# ------------------------------------------------------------ ramp vs flat
def fig_rampflat(R, out):
    C = {(c["condition"], c["violation_type"]): c for c in R["scoring"]["cells"]}
    groups = [("moving_visible", "moving_visible_flat", "no occluder"),
              ("moving_occlusion", "moving_occlusion_flat", "occluded")]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    W = 0.32
    for pi, (ax, (ramp, flat, title)) in enumerate(zip(axes, groups)):
        frame(ax, (0, 124), (0, 25, 50, 75, 100))
        for gi, v in enumerate(VIOL):
            for bi, (cond, lab, f) in enumerate([(ramp, "ramp", 0.46), (flat, "flat", 0.10)]):
                y = C[(cond, v)]["acc"]
                x = gi + (bi - 0.5) * W
                ax.bar(x, y, width=W * 0.88, color=tint(VCOL[v], f),
                       edgecolor=tint(VCOL[v], max(0, f - 0.28)), lw=0.7, zorder=3)
                ax.annotate(f"{y:.1f}", (x, y + 1.6), ha="center", va="bottom",
                            fontsize=7.2, color=INK, zorder=5)
                ax.annotate(lab, (x, 2.5), ha="center", va="bottom", fontsize=6.2,
                            color="white" if y > 20 else INK2, rotation=90, zorder=6)
            d = C[(flat, v)]["acc"] - C[(ramp, v)]["acc"]
            ax.annotate(f"{d:+.1f}", (gi, 113), ha="center", va="bottom", fontsize=7.6,
                        color=INK if abs(d) > 3 else MUTED, zorder=5,
                        fontweight="bold" if abs(d) > 3 else "normal")
        ax.axhline(50, color=MUTED, lw=0.8, ls=(0, (2.5, 2.5)), zorder=4)
        ax.set_xticks(range(3)); ax.set_xticklabels([VSHORT[v] for v in VIOL], fontsize=8.4)
        ax.set_xlim(-0.5 - W, 2.5 + W); ax.tick_params(axis="x", length=0, pad=3)
        ax.set_title(f"Moving, {title}", fontsize=9.5, color=INK, pad=4)
        if pi == 0:
            ax.set_ylabel("pairwise acc. (%)", fontsize=9.5)
        ax.text(0.5, -0.115, PANEL[pi], transform=ax.transAxes, ha="center", va="top",
                fontsize=9.5, color=INK)
    fig.text(0.5, 1.00, "ramp = constant acceleration · flat = constant velocity   |   "
                        "numbers above each pair: flat $-$ ramp (pp)",
             ha="center", va="top", fontsize=7.8, color=INK2, style="italic")
    fig.subplots_adjust(left=0.075, right=0.995, top=0.885, bottom=0.125, wspace=0.10)
    save(fig, out, "fig_ramp_vs_flat")


# ----------------------------------------------------------- occlusion
def fig_occlusion(R, out):
    C = {(c["condition"], c["violation_type"]): c for c in R["scoring"]["cells"]}
    arms = [("static_visible", "static_occlusion", "Static"),
            ("moving_visible_flat", "moving_occlusion_flat", "Moving (flat)"),
            ("moving_visible", "moving_occlusion", "Moving (ramp)")]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.9))
    W = 0.32
    for pi, (ax, (vis, occ, title)) in enumerate(zip(axes, arms)):
        frame(ax, (0, 124), (0, 25, 50, 75, 100))
        for gi, v in enumerate(VIOL):
            for bi, (cond, lab, f) in enumerate([(vis, "visible", 0.46), (occ, "occluded", 0.10)]):
                y = C[(cond, v)]["acc"]
                x = gi + (bi - 0.5) * W
                ax.bar(x, y, width=W * 0.88, color=tint(VCOL[v], f),
                       edgecolor=tint(VCOL[v], max(0, f - 0.28)), lw=0.7, zorder=3)
                ax.annotate(f"{y:.0f}", (x, y + 1.6), ha="center", va="bottom",
                            fontsize=6.8, color=INK, zorder=5)
                # 범례 대신 막대에 직접 단다 — 범례 스와치는 그림에 없는 색이 된다
                ax.annotate(lab, (x, 2.5), ha="center", va="bottom", fontsize=6.2,
                            color="white" if y > 20 else INK2, rotation=90, zorder=6)
            d = C[(occ, v)]["acc"] - C[(vis, v)]["acc"]
            ax.annotate(f"{d:+.0f}", (gi, 113), ha="center", va="bottom",
                        fontsize=7.6, color=INK, zorder=5, fontweight="bold")
        ax.axhline(50, color=MUTED, lw=0.8, ls=(0, (2.5, 2.5)), zorder=4)
        ax.set_xticks(range(3))
        ax.set_xticklabels([VSHORT[v] for v in VIOL], fontsize=8.0)
        ax.set_xlim(-0.5 - W, 2.5 + W); ax.tick_params(axis="x", length=0, pad=3)
        ax.set_title(title, fontsize=9.5, color=INK, pad=4)
        if pi:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("pairwise acc. (%)", fontsize=9.5)
        ax.text(0.5, -0.115, PANEL[pi], transform=ax.transAxes, ha="center", va="top",
                fontsize=9.5, color=INK)
    # 범례 없음 — x 축이 위반을, 막대 안 글자가 가림 유무를, 위 숫자가 낙차를 말한다
    fig.text(0.5, 1.00, "numbers above each pair: occluded $-$ visible (pp)",
             ha="center", va="top", fontsize=7.8, color=INK2, style="italic")
    fig.subplots_adjust(left=0.075, right=0.995, top=0.885, bottom=0.125, wspace=0.09)
    save(fig, out, "fig_occlusion")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--which", nargs="*", default=["k", "sens", "rampflat", "occlusion"])
    a = ap.parse_args()
    R = json.loads(a.report.read_text())

    # 자체 검증: report 의 verified 플래그 + cells 재합산
    ov = R["scoring"]["overall"][0]
    tot = sum(c["n"] for c in R["scoring"]["cells"])
    wa = sum(c["acc"] * c["n"] for c in R["scoring"]["cells"]) / tot
    assert ov["verified"], "report.json 이 summary.json 과 대조 검증되지 않았다"
    assert abs(wa - ov["overall"]) < 0.02, f"cells 재합산 {wa:.4f} != overall {ov['overall']:.4f}"
    print(f"  검증 OK — {ov['dataset']} overall {ov['overall']:.2f}%  n={ov['n_pair']}  "
          f"(cells 재합산 {wa:.2f}%)")

    for w in a.which:
        {"k": fig_k, "sens": fig_sens, "rampflat": fig_rampflat,
         "occlusion": fig_occlusion}[w](R, a.outdir)


if __name__ == "__main__":
    main()
