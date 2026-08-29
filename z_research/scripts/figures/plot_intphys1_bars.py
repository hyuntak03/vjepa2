#!/usr/bin/env python3
"""Figure 1 (IntPhys1) — principle x (motion x occlusion) pairwise accuracy.

IntPhysGen 판(plot_violation_bars.py)과 같은 그림을 IntPhys1 dev 에 대해 그린다.
다른 점은 라벨 출처뿐이다:

  - IntPhysGen : index.csv 에 condition / violation_type 이 들어 있다
  - IntPhys1   : index.csv 에는 block_type(O1/O2/O3) 밖에 없다. motion/occlusion 은
                 원본 dev 의 task.txt 에서만 나오는데 그 파일은 이미 사라졌고,
                 파생물인 IntPhys1_dev_by_scene/pairs.csv 에 label_motion(이동/정지) ·
                 label_vis(가려짐/눈앞) 로 살아남았다. 그래서 그걸 조인한다.

⚠️ 셀 크기가 고르지 않다 — moving 20쌍 / static 10쌍. static 은 1쌍이 10%p 다.
   그래서 막대마다 n 을 함께 찍는다.

  python z_research/scripts/figures/plot_intphys1_bars.py \
      --result-dir z_exp/world_model_analysis/results/intphys1_vith \
      --pairs /local_datasets/world/world_analysis/IntPhys1_dev_by_scene/pairs.csv \
      --combo skip2_w32/avg \
      --output z_research/IntPhys/figures/fig1_intphys1_bars.pdf
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
# pairs.csv 의 한국어 라벨 -> 조건 키
MOTION = {"정지": "static", "이동": "moving"}
VIS = {"눈앞": "visible", "가려짐": "occlusion"}

PRIN = ["O1_object_permanence", "O2_shape_constancy", "O3_spatiotemporal_continuity"]
PLABEL = {"O1_object_permanence": "Object permanence",
          "O2_shape_constancy": "Shape constancy",
          "O3_spatiotemporal_continuity": "Spatio-temporal continuity"}

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED = "#000000", "#3b3b3b", "#9a9a9a"


def tint(hex_color: str, f: float) -> str:
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return "#%02x%02x%02x" % tuple(round(c + (255 - c) * f) for c in (r, g, b))


FILL_LIGHT, FILL_DARK, EDGE_D = 0.46, 0.16, 0.28
BAR = {"static_visible":   (BLUE,   FILL_DARK,  "Static + visible"),
       "moving_visible":   (BLUE,   FILL_LIGHT, "Moving + visible"),
       "moving_occlusion": (ORANGE, FILL_LIGHT, "Moving + occluded"),
       "static_occlusion": (ORANGE, FILL_DARK,  "Static + occluded")}

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.2, "ytick.major.size": 2.2,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def score(result_dir: Path, pairs_csv: Path, combo: str):
    """pairs.csv 의 (pos, imp) 를 per_video_surprise 로 다시 채점한다."""
    S = json.loads((result_dir / "per_block.json").read_text())["per_video_surprise"]
    out = []
    for r in csv.DictReader(pairs_csv.open()):
        sp, si = S[r["pos"]], S[r["imp"]]
        sp = sp[combo] if isinstance(sp, dict) else sp
        si = si[combo] if isinstance(si, dict) else si
        cond = f'{MOTION[r["label_motion"]]}_{VIS[r["label_vis"]]}'
        out.append((cond, r["principle"], 1.0 if si > sp else (0.5 if si == sp else 0.0)))

    got = float(np.mean([a for _, _, a in out]))
    sur = json.loads((result_dir / "summary.json").read_text())["surprise"]
    want = (sur[combo] if combo in sur else sur)["overall"]["block_pairwise"]
    if abs(got - want) > 1e-9:
        raise ValueError(f"재계산 {got:.6f} != summary.json {want:.6f}")
    print(f"  검증 OK — 재계산 overall {got*100:.4f}% == summary.json[{combo}] "
          f"{want*100:.4f}%  (n={len(out)})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", type=Path, required=True)
    ap.add_argument("--pairs", type=Path,
                    default=Path("/local_datasets/world/world_analysis/"
                                 "IntPhys1_dev_by_scene/pairs.csv"))
    ap.add_argument("--combo", default="skip2_w32/avg")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--title", default=None)
    a = ap.parse_args()

    pairs = score(a.result_dir, a.pairs, a.combo)
    acc, n = {}, {}
    for c in CONDS:
        for p in PRIN:
            s = [x for cc, pp, x in pairs if cc == c and pp == p]
            acc[(c, p)], n[(c, p)] = 100 * float(np.mean(s)), len(s)
    overall = 100 * float(np.mean([x for _, _, x in pairs]))

    fig, ax = plt.subplots(figsize=(7.0, 2.85))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=MUTED, alpha=0.30, lw=0.4)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(labelsize=9, colors=INK2, pad=2.0)
    ax.set_ylabel("pairwise acc. (%)", fontsize=10)

    W, PAD = 0.21, 0.055
    HALF = 1.95 * W
    for gi, p in enumerate(PRIN):
        for bi, c in enumerate(CONDS):
            base, f, _ = BAR[c]
            x = gi + (bi - 1.5) * W
            y = acc[(c, p)]
            ax.bar(x, y, width=W * 0.90, color=tint(base, f),
                   edgecolor=tint(base, max(0.0, f - EDGE_D)), lw=0.7, zorder=3)
            ax.annotate(f"{y:.1f}", (x, y + 1.6), ha="center", va="bottom",
                        fontsize=7.8, color=INK, zorder=5)
            # 셀 크기가 20 / 10 으로 다르다 — 숨기지 않는다
            ax.annotate(f"n{n[(c, p)]}", (x, y + 6.4), ha="center", va="bottom",
                        fontsize=6.0, color=MUTED, zorder=5)
        if gi:
            ax.axvline(gi - 0.5, color=MUTED, lw=0.5, alpha=0.35, zorder=1)

    ax.axhline(50, color=MUTED, lw=0.8, ls=(0, (2.5, 2.5)), zorder=4)

    ax.set_xticks(range(len(PRIN)))
    ax.set_xticklabels([PLABEL[p] for p in PRIN], fontsize=10.5)
    ax.set_xlim(-HALF - PAD, len(PRIN) - 1 + HALF + PAD)
    ax.tick_params(axis="x", length=0, pad=4)

    handles = [Patch(facecolor=tint(BAR[c][0], BAR[c][1]),
                     edgecolor=tint(BAR[c][0], max(0.0, BAR[c][1] - EDGE_D)), lw=0.7,
                     label=BAR[c][2]) for c in CONDS]
    handles.append(Line2D([], [], color=MUTED, lw=1.1, ls=(0, (2.5, 2.5)),
                          label="chance (50%)"))
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, fontsize=9,
               handlelength=1.5, handletextpad=0.5, columnspacing=1.35,
               bbox_to_anchor=(0.5, 1.015))

    if a.title:
        fig.suptitle(a.title, fontsize=11.5, y=1.10, x=0.02, ha="left", color=INK)
    fig.subplots_adjust(left=0.068, right=0.998, top=0.885, bottom=0.115)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".pdf", ".png"):
        fig.savefig(a.output.with_suffix(ext), dpi=400, bbox_inches="tight", facecolor="white")
    print(f"[saved] {a.output.with_suffix('.pdf')}  +  .png")

    head = {"O1_object_permanence": "O1 perm", "O2_shape_constancy": "O2 shape",
            "O3_spatiotemporal_continuity": "O3 cont"}
    print(f"\n{'condition':<20}" + "".join(f"{head[p]:>12}" for p in PRIN) + f"{'all':>10}")
    for c in CONDS:
        s = [x for cc, _, x in pairs if cc == c]
        print(f"{c:<20}"
              + "".join(f"{acc[(c,p)]:>7.2f} (n{n[(c,p)]:>2})"[-12:] for p in PRIN)
              + f"{100*float(np.mean(s)):>10.2f}")
    print(f"{'all':<20}" + "".join(
        f"{100*float(np.mean([x for _,pp,x in pairs if pp==p])):>12.2f}" for p in PRIN)
        + f"{overall:>10.2f}")
    print(f"\ntotal {len(pairs)} matched pairs | combo {a.combo}")


if __name__ == "__main__":
    main()
