#!/usr/bin/env python3
"""Object permanence 의 방향 분해 — 왜 moving+occluded 가 정확히 50% 인가.

vanish 블록의 문맥일치 쌍은 두 방향으로 갈린다:
    object → vanishes   pos_obj   vs imp_vanish   (물체가 있다가 사라진다)
    empty  → appears    pos_empty vs imp_appear   (빈 화면에 물체가 생긴다)
두 방향의 표본 수가 같으므로 보고되는 값은 정확히 둘의 평균이다.
가림에서 이 둘이 0% / 100% 로 갈리면 평균은 신호와 무관하게 50% 가 된다.

  python z_research/scripts/figures/plot_vanish_direction.py \
      --result-dir z_research/IntPhysGenV10/exp_results/surprise_c16t32__v10_vith \
      --index data_csv/intphysgen_v10/index_probe.csv \
      --output z_research/IntPhysGenV10/figures/fig2_vanish_direction.pdf
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

CONDS = ["static_visible", "moving_visible", "moving_occlusion", "static_occlusion"]
CLABEL = {"static_visible": "Static\nno occluder", "moving_visible": "Moving\nno occluder",
          "moving_occlusion": "Moving\noccluded", "static_occlusion": "Static\noccluded"}
# (possible role, impossible role, 표기)
DIRS = [("pos_obj", "imp_vanish", "object $\\rightarrow$ vanishes"),
        ("pos_empty", "imp_appear", "empty $\\rightarrow$ appears")]

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED = "#000000", "#3b3b3b", "#9a9a9a"
FILL, EDGE_D = 0.30, 0.28

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.2, "ytick.major.size": 2.2,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def tint(hex_color: str, f: float) -> str:
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return "#%02x%02x%02x" % tuple(round(c + (255 - c) * f) for c in (r, g, b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", type=Path, required=True)
    ap.add_argument("--index", type=Path, required=True, help="role 컬럼이 있어야 한다")
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()

    S = json.loads((a.result_dir / "per_block.json").read_text())["per_video_surprise"]
    rows = list(csv.DictReader(a.index.open()))
    grp = collections.defaultdict(list)
    for r in rows:
        grp[(r["block_id"], r["pair_id"])].append(r)

    hit = collections.defaultdict(list)
    for v in grp.values():
        pos = [x for x in v if x["plausible"] == "1"][0]
        imp = [x for x in v if x["plausible"] != "1"][0]
        if imp["violation_type"] != "vanish":
            continue
        sp, si = S[pos["video_id"]], S[imp["video_id"]]
        hit[(pos["condition"], pos["role"])].append(
            1.0 if si > sp else (0.5 if si == sp else 0.0))

    acc = {k: 100 * float(np.mean(v)) for k, v in hit.items()}
    n = {k: len(v) for k, v in hit.items()}

    fig, ax = plt.subplots(figsize=(7.0, 2.95))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=MUTED, alpha=0.30, lw=0.4)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(labelsize=9, colors=INK2, pad=2.0)
    ax.set_ylabel("pairwise acc. (\\%)".replace("\\", ""), fontsize=10)

    W, PAD = 0.30, 0.06
    HALF = 1.0 * W          # 막대 2개: 중심에서 ±0.5W, 반폭 0.45W -> 0.95W
    for gi, c in enumerate(CONDS):
        pooled = []
        for di, (pr, ir, _) in enumerate(DIRS):
            base = BLUE if di == 0 else ORANGE
            x = gi + (di - 0.5) * W
            y = acc[(c, pr)]
            pooled += hit[(c, pr)]
            ax.bar(x, y, width=W * 0.90, color=tint(base, FILL),
                   edgecolor=tint(base, max(0.0, FILL - EDGE_D)), lw=0.7, zorder=3)
            ax.annotate(f"{y:.1f}", (x, y + 1.8), ha="center", va="bottom",
                        fontsize=8.2, color=INK, zorder=5)
            if y in (0.0, 100.0):        # 완전 붕괴/완전 정답은 개수를 병기한다
                ax.annotate(f"{int(round(y/100*n[(c,pr)]))}/{n[(c,pr)]}",
                            (x, max(y, 0) + 7.2), ha="center", va="bottom",
                            fontsize=7.2, color=INK2, zorder=5)
        # 보고되는 값 = 두 방향의 평균
        m = 100 * float(np.mean(pooled))
        ax.plot([gi - 0.98 * W, gi + 0.98 * W], [m, m], color=INK, lw=1.3,
                ls=(0, (1.2, 1.2)), zorder=6)
        ax.annotate(f"{m:.1f}", (gi + 1.02 * W, m), fontsize=8.2, color=INK,
                    ha="left", va="center", zorder=6)
        if gi:
            ax.axvline(gi - 0.5, color=MUTED, lw=0.5, alpha=0.35, zorder=1)

    ax.axhline(50, color=MUTED, lw=0.8, ls=(0, (2.5, 2.5)), zorder=4)
    ax.set_xticks(range(len(CONDS)))
    ax.set_xticklabels([CLABEL[c] for c in CONDS], fontsize=10, linespacing=1.35)
    ax.set_xlim(-HALF - PAD - 0.06, len(CONDS) - 1 + HALF + PAD + 0.16)
    ax.tick_params(axis="x", length=0, pad=5)

    handles = [Patch(facecolor=tint(BLUE if i == 0 else ORANGE, FILL),
                     edgecolor=tint(BLUE if i == 0 else ORANGE, FILL - EDGE_D), lw=0.7,
                     label=d[2]) for i, d in enumerate(DIRS)]
    handles += [Line2D([], [], color=INK, lw=1.3, ls=(0, (1.2, 1.2)), label="reported (mean)"),
                Line2D([], [], color=MUTED, lw=1.1, ls=(0, (2.5, 2.5)), label="chance (50\\%)".replace("\\", ""))]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=9,
               handlelength=1.6, handletextpad=0.5, columnspacing=1.5,
               bbox_to_anchor=(0.5, 1.015))
    fig.subplots_adjust(left=0.068, right=0.998, top=0.885, bottom=0.155)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".pdf", ".png"):
        fig.savefig(a.output.with_suffix(ext), dpi=400, bbox_inches="tight", facecolor="white")
    print(f"[saved] {a.output.with_suffix('.pdf')}  +  .png\n")

    print(f"{'condition':<18}{'object→vanishes':>18}{'empty→appears':>18}{'mean':>10}")
    for c in CONDS:
        pooled = hit[(c, "pos_obj")] + hit[(c, "pos_empty")]
        print(f"{c:<18}{acc[(c,'pos_obj')]:>13.2f} (n{n[(c,'pos_obj')]})"
              f"{acc[(c,'pos_empty')]:>13.2f} (n{n[(c,'pos_empty')]})"
              f"{100*float(np.mean(pooled)):>10.2f}")


if __name__ == "__main__":
    main()
