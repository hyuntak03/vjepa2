#!/usr/bin/env python3
"""IntPhys 1 dev 가림 통계 그림 — 입력은 `occlusion_intphys1.py` 산출물만 (다시 계산하지 않는다).

  fig_visibility_raster.png   행 = 가림 조건 쌍 90 개의 possible 영상 (조건 · 사건 종류별 묶음), 열 = raw 프레임 0–99.
                              사건 물체가 보임 / 가림 / 화면 밖 (등장 전·퇴장 뒤 · 물체 없는 영상), 점 = d
  fig_gap_stats.png           (a) 사건 가림 길이 (초) (b) 가려진 동안 이동 거리 (물체 폭) 대 길이
                              (c) 이동+가림 쌍의 V-JEPA 2 정답 (skip2_w32) 대 skip 2 격자에서 가려진 프레임 수 (+ 사건 전 미관측 쌍)
그림 값은 summary.json 과 대조한다 (assert).

  python z_research/scripts/figures/plot_occlusion_intphys1.py
"""
from __future__ import annotations
import collections, csv, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[3]
D = ROOT / "z_research/OcclusionStats/intphys1"
FIG = D / "figures"
BLUE, ORANGE, GREEN, GREY, INK = "#2a78d6", "#eb6834", "#1baf7a", "#e3e3e3", "#222222"
ORDER = ["moving/occluded", "static/occluded", "moving/visible", "static/visible"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})


def load():
    S = json.loads((D / "videos.json").read_text())
    summ = json.loads((D / "summary.json").read_text())
    gaps = list(csv.DictReader((D / "gaps.csv").open()))
    pairs = list(csv.DictReader((D / "pairs.csv").open()))
    return S, summ, gaps, pairs


def raster(S, pairs):
    rows, labels, bounds, ds = [], [], [], []
    for g in ORDER[:2]:
        for kind_name, kinds in (("event gap", ("gap_contains_d", "gap_ends_just_before_d")), ("not seen before d", ("not_seen_before_d",))):
            ps = sorted((p for p in pairs if p["group"] == g and p["event_kind"] in kinds), key=lambda p: (p["pos"][:2], int(p["d"])))
            if not ps:
                continue
            bounds.append((f"{g}\n{kind_name}", len(rows), len(ps)))
            for p in ps:
                vis = np.array([int(c) for c in S[p["pos"]]["visible"]])
                r = np.zeros(100, int)                           # 0 = 화면 밖 (등장 전·퇴장 뒤 · 물체 없는 영상)
                idx = np.nonzero(vis)[0]
                if len(idx):
                    r[idx[0]:idx[-1] + 1] = 2                     # 2 = 가림 (보임 사이의 안 보임)
                    r[vis == 1] = 1                               # 1 = 보임
                rows.append(r); labels.append(p["pos"]); ds.append(int(p["d"]))
    M = np.array(rows)
    fig, ax = plt.subplots(figsize=(7.0, 6.4))
    ax.imshow(M, aspect="auto", interpolation="nearest", cmap=ListedColormap([GREY, BLUE, ORANGE]), vmin=0, vmax=2,
              extent=(-0.5, 99.5, len(rows) - 0.5, -0.5))
    for i, d in enumerate(ds):
        ax.plot(d, i, marker="o", ms=2.2, color=INK, lw=0)
    for g, s, n in bounds:
        ax.axhline(s - 0.5, color="white", lw=2.5)
        ax.text(-2, s + n / 2 - 0.5, f"{g}\n(n={n} pairs)", ha="right", va="center", fontsize=7)
    ax.set_yticks([]); ax.set_xlim(-0.5, 99.5)
    sec = ax.secondary_xaxis("top", functions=(lambda f: f / 15.0, lambda s: s * 15.0))
    sec.set_xlabel("time (s, 15 fps)")
    ax.set_xlabel("raw frame index (0-based)")
    for sp in ("left", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(handles=[Patch(color=BLUE, label="event object visible (possible video of the pair)"), Patch(color=ORANGE, label="occluded (hidden between visible spans)"),
                       Patch(color=GREY, label="not in view (before entry / after exit / no object)"),
                       plt.Line2D([], [], color=INK, marker="o", lw=0, ms=3, label="first pos/imp pixel difference d")],
              loc="upper center", bbox_to_anchor=(0.45, -0.07), ncol=2, frameon=False)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig_visibility_raster.png", dpi=220, bbox_inches="tight"); plt.close(fig)


def gap_stats(summ, gaps, pairs):
    ev = [g for g in gaps if g["event_gap"] == "1"]
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.5), gridspec_kw={"width_ratios": [1.2, 1.2, 1.0], "wspace": 0.6})
    rng = np.random.default_rng(0)
    # (a) 길이
    ax = axs[0]
    for i, (g, col) in enumerate((("moving/occluded", ORANGE), ("static/occluded", BLUE))):
        x = np.array([float(e["seconds"]) for e in ev if e["group"] == g])
        assert abs(np.median(x) - summ["by_group"][g]["event_gap_seconds"]["median"]) < 0.01      # summary 는 소수 둘째 자리 반올림
        ax.scatter(x, i + rng.uniform(-0.18, 0.18, len(x)), s=12, color=col, edgecolor="white", lw=0.4, zorder=3)
    nm = {g: sum(1 for e in ev if e["group"] == g) for g in ("moving/occluded", "static/occluded")}
    ax.set_yticks([0, 1]); ax.set_yticklabels([f"moving/\noccluded\n({nm['moving/occluded']} pairs)", f"static/\noccluded\n({nm['static/occluded']} pairs)"], fontsize=6.8)
    ax.set_ylim(-0.6, 1.6); ax.set_xlim(0, 7)
    ax.set_xlabel("event occlusion (s)")
    ax.grid(axis="x", color="#dddddd", lw=0.5); ax.set_axisbelow(True)
    # (b) 이동 거리 대 길이
    ax = axs[1]
    for g, col in (("moving/occluded", ORANGE), ("static/occluded", BLUE)):
        x = [float(e["seconds"]) for e in ev if e["group"] == g]
        y = [float(e["disp_widths"]) for e in ev if e["group"] == g]
        ax.scatter(x, y, s=12, color=col, edgecolor="white", lw=0.4, zorder=3, label=g)
    ax.set_xlabel("event occlusion (s)"); ax.set_ylabel("displacement while hidden\n(object widths)")
    ax.set_xlim(0, 7); ax.set_ylim(0, 10)
    ax.legend(frameon=False, fontsize=6.5, loc="upper right")
    ax.grid(color="#dddddd", lw=0.5); ax.set_axisbelow(True)
    # (c) 정답 대 가려진 skip2 프레임 수 (이동+가림)
    ax = axs[2]
    tab = summ["by_group"]["moving/occluded"]["acc_by_event_gap_skip2_frames"]
    ks = sorted((k for k in tab if k.isdigit()), key=int) + ["no_event_gap"]
    acc = [tab[k][0] for k in ks]; n = [tab[k][1] for k in ks]
    ax.bar(range(len(ks)), acc, width=0.6, color=ORANGE, zorder=3)
    for i, (a, m) in enumerate(zip(acc, n)):
        ax.text(i, a + 2, f"{a:.0f}\nn={m}", ha="center", va="bottom", fontsize=6.3)
    ax.axhline(50, color=INK, lw=0.8, ls=":")
    ax.set_xticks(range(len(ks))); ax.set_xticklabels([k if k.isdigit() else "not seen\nbefore d" for k in ks], fontsize=7)
    ax.set_xlabel("hidden skip-2 frames"); ax.set_ylabel("V-JEPA 2 pair acc. (%)")
    ax.set_ylim(0, 118); ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis="y", color="#dddddd", lw=0.5); ax.set_axisbelow(True)
    for a in axs:
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    for a, t in zip(axs, "abc"):
        a.text(0.5, -0.42, f"({t})", transform=a.transAxes, ha="center", va="top", fontsize=8)
    fig.savefig(FIG / "fig_gap_stats.png", dpi=220, bbox_inches="tight"); plt.close(fig)


def main():
    S, summ, gaps, pairs = load()
    raster(S, pairs)
    gap_stats(summ, gaps, pairs)
    print("→", FIG)


if __name__ == "__main__":
    main()
