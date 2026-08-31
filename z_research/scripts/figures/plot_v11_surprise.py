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
import argparse, collections, json
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, BAND = "#000000", "#3b3b3b", "#9a9a9a", "#efefef"
VIOL = ["vanish", "shape", "color"]
VLAB = {"vanish": "Object permanence", "shape": "Shape consistency", "color": "Colour consistency"}
VSHORT = {"vanish": "Permanence", "shape": "Shape", "color": "Colour"}
VCOL = {"vanish": BLUE, "shape": ORANGE, "color": GREEN}
PANEL = [f"({c})" for c in "abcdefghi"]   # 3패널 그림은 앞 셋만 쓴다
# 정확도용 발산 컬러맵 — 50(chance) 이 흰색. 아래가 주황(놓침), 위가 파랑(잡음).
CMAP_ACC = LinearSegmentedColormap.from_list(
    "acc", [(0.0, ORANGE), (0.5, "#ffffff"), (1.0, BLUE)])

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


def stack_labels(vals, pad=1.6, sep=4.2, ymax=104.5):
    """같은 x 에 모인 값들의 라벨 위치를 정한다 -> {i: (y, va)}.

    아래에서 위로 쌓되, 천장을 넘는 것은 **자기 점 아래로** 내려서 또 쌓는다.
    (vanish k=0 은 세 팔이 전부 100 이라 위로만 쌓으면 두 개가 축 밖으로 나간다.)
    fig_k_dose 와 fig_k_dose_arm 이 같은 함수를 써야 나란히 놨을 때 읽는 법이 같다.
    """
    def clear(y):
        # 라벨이 **다른 계열의 마커** 위에 얹히면 가려진다 (k=0 에서 84 와 82 가 그랬다).
        # 어떤 점과도 최소 간격을 두도록 위로 민다.
        # 글자는 anchor 에서 **위로** 자란다 (va="bottom"). 그 높이까지 감안하지 않으면
        # 라벨 윗부분이 위쪽 마커에 걸린다 (k=0 의 94 / 90 이 그랬다).
        while any(abs(y - v) < pad * 1.25 or -0.2 < v - y < pad * 2.4 for v in vals):
            y += 0.5
        return y

    out, up, dn = {}, -1e9, 1e9
    for i in sorted(range(len(vals)), key=lambda j: vals[j]):
        y = clear(max(vals[i] + pad, up + sep))
        if y > ymax:
            y = min(vals[i] - pad, dn - sep)
            out[i] = (y, "top"); dn = y
        else:
            out[i] = (y, "bottom"); up = y
    return out


# 그림을 논문 beat 순서대로 폴더에 나눈다 (PAPER_STORY_2026-08-31.md).
# 여기 없는 이름은 outdir 최상위에 떨어진다 — 새 그림이 눈에 띄라고 일부러 그렇게 뒀다.
FIGDIR = {
    "fig_occlusion":            "01_condition",      # beat 1-2  가림 유무
    "fig_ramp_vs_flat":         "01_condition",      # beat 2    등속 vs 등가속
    "fig_k_dose":               "02_occlusion_k",    # beat 2-4  k 평균
    "fig_k_dose_arm":           "02_occlusion_k",    # beat 2-4  k x 운동
    "fig_k_sensitivity":        "02_occlusion_k",
    "fig_vanish_direction":     "03_direction",      # beat 4    100 / 0
    "fig_direction_split":      "03_direction",
    "fig_direction_gap":        "03_direction",
    "fig_predictor_vote":       "04_object_order",   # beat 4    무엇을 만드나 (2지선다, 원자료)
    "fig_make_prob":            "_superseded",       # 7지선다 변환 — 최저 상대 하나가 지배한다
    "fig_make_prob_grid_shape": "_superseded",
    "fig_make_prob_grid_color": "_superseded",
    "fig_vote_grid_shape":      "04_object_order",
    "fig_vote_grid_color":      "04_object_order",
    "fig_keep_matrix":          "_superseded",       # 04 의 "지켜냈나" 판. 틀이 뒤집혔다
    "fig_keep_matrix_visible":  "_superseded",
    "fig_keep_dose":            "_superseded",
}


def save(fig, out: Path, name):
    out = out / FIGDIR.get(name, "")
    out.mkdir(parents=True, exist_ok=True)
    for ext in (".pdf", ".png"):
        fig.savefig((out / name).with_suffix(ext), dpi=400, bbox_inches="tight", facecolor="white")
    print(f"  [saved] {out / name}.pdf  + .png")


# ---------------------------------------------------------------- k dose
# ⚠️ fig_k_dose 와 fig_k_dose_arm 은 **나란히 놓고 읽는 짝**이다. 축·색·라벨 관례를
#    반드시 같이 유지할 것. acc 는 두 그림 모두 (40,108) / (50,75,100) 이다
#    (실측 최저 48.2 라 0 부터 그리면 해상도만 버린다).
# fig_k_dose_arm 의 판형. fig_k_dose 는 **이 그림의 패널 하나와 같은 축 크기**로 그린다
# (둘을 나란히 놓았을 때 (a) 와 크기가 같아야 한다). 아래 값을 고치면 둘 다 따라온다.
ARMFIG = dict(w=7.6, h=2.9, left=0.075, right=0.995, top=0.905, bottom=0.215,
              wspace=0.20, ncol=3)


# 막대 그림 두 장(fig_occlusion 3패널 / fig_ramp_vs_flat 2패널)은 **나란히 놓고 읽는다**.
# 패널 수가 달라도 **막대 폭이 같아야** 비교가 된다 -> 패널 폭을 고정하고 figsize 를 역산한다.
OCCFIG = dict(w=7.0, h=2.9, left=0.075, right=0.995, wspace=0.09, ncol=3)


def matched_bars(n):
    """OCCFIG 의 패널 폭을 그대로 쓰는 n 패널 figure 의 (figsize, 여백)."""
    A = OCCFIG
    pw = A["w"] * (A["right"] - A["left"]) / (A["ncol"] + (A["ncol"] - 1) * A["wspace"])
    lm, rm = A["w"] * A["left"], A["w"] * (1 - A["right"])
    W = lm + pw * (n + (n - 1) * A["wspace"]) + rm
    return (W, A["h"]), dict(left=lm / W, right=1 - rm / W, wspace=A["wspace"])


def matched_single(scale=1.0):
    """ARMFIG 패널 하나를 기준으로 한 단일 패널 figure 의 (figsize, 여백).

    scale=1.0 이면 축 크기가 (a) 패널과 정확히 같다. 다만 범례·축 아래 이탤릭이
    2.06in 안에 안 들어가서 실제로는 조금 넓혀 쓴다 (fig_k 는 1.5).
    """
    A = ARMFIG
    usable = A["w"] * (A["right"] - A["left"])
    aw = usable / (A["ncol"] + (A["ncol"] - 1) * A["wspace"]) * scale   # 패널 폭(inch)
    lm, rm = A["w"] * A["left"], A["w"] * (1 - A["right"])          # 좌우 여백(inch)
    W = lm + aw + rm
    return (W, A["h"]), dict(left=lm / W, right=1 - rm / W,
                             top=A["top"], bottom=A["bottom"])


SPEC = {"acc":         ("pairwise acc. (%)", (40, 108), (50, 75, 100)),
        "sensitivity": ("sensitivity (pp)",  (-5, 58), (0, 25, 50)),
        "bias":        ("bias (pp)",         (-2, 58), (0, 25, 50))}


def _fig_k(R, out, metrics, name, subtitle=None, width=7.0, box=None, short_lg=False):
    dose = {(d["sym_k"], d["violation"]): d for d in R["scoring"]["dose"]}
    ks = sorted({d["sym_k"] for d in R["scoring"]["dose"]}, key=int)
    figsize = box[0] if box else (width, 2.9)
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize, squeeze=False)
    axes = axes[0]
    for pi, (ax, metric) in enumerate(zip(axes, metrics)):
        lab, ylim, yt = SPEC[metric]
        frame(ax, ylim, yt)
        # k=0 은 가림막이 아예 없다 — 나머지와 성질이 다르므로 띄우고 음영으로 가른다.
        # 음영 = "가림막이 있는 구간". 축 아래 이탤릭 라벨로 명시한다 (다른 그림과 같은 관례).
        xs = {k: (0 if k == "0" else 1.6 + int(k) - 1) for k in ks}
        occ = [xs[k] for k in ks if k != "0"]
        ax.axvspan(min(occ) - 0.5, max(occ) + 0.5, color=BAND, lw=0, zorder=0)
        # 같은 x 에서 값이 가까우면 라벨이 겹친다 — 공통 헬퍼로 쌓는다
        step = 4.2 if metric == "acc" else 2.6
        cap = (ylim[1] - 3.5) if metric == "acc" else 1e9
        at = {}
        for k in ks:
            vv = [v for v in VIOL if (k, v) in dose]
            pos = stack_labels([dose[(k, v)][metric] for v in vv], pad=2.1,
                               sep=step, ymax=cap)
            for i, v in enumerate(vv):
                at[(k, v)] = pos[i]
        for v in VIOL:
            y = [dose[(k, v)][metric] for k in ks if (k, v) in dose]
            x = [xs[k] for k in ks if (k, v) in dose]
            ax.plot(x[1:], y[1:], "-o", color=VCOL[v], ms=6.0, lw=1.4, zorder=3,
                    mec="white", mew=0.9, label=VLAB[v])
            ax.plot(x[:1], y[:1], "o", color=VCOL[v], ms=6.8, zorder=3, mec="white", mew=0.9)
            ax.plot(x[:2], y[:2], ":", color=tint(VCOL[v], 0.45), lw=1.0, zorder=2)
            for kk, xx in zip([k for k in ks if (k, v) in dose], x):
                ly, lva = at[(kk, v)]
                ax.annotate(f"{dose[(kk, v)][metric]:.0f}", (xx, ly), ha="center", va=lva,
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
        fs = 7.6 if len(metrics) > 1 else 7.0
        ax.annotate("no occluder", (xs["0"], -0.115), xycoords=("data", "axes fraction"),
                    ha="center", va="top", fontsize=fs, color=INK2, style="italic")
        ax.annotate("occluded — hidden frames (k)",
                    (float(np.mean(occ)), -0.115), xycoords=("data", "axes fraction"),
                    ha="center", va="top", fontsize=fs, color=INK2, style="italic")
        ax.set_ylabel(lab, fontsize=9.5)
        if len(metrics) > 1:
            ax.text(0.5, -0.215, PANEL[pi], transform=ax.transAxes, ha="center", va="top",
                    fontsize=9.5, color=INK)
    # 좁은 판형에서는 짧은 라벨로 **가로 한 줄**을 유지한다 (세로로 쌓으면 그림이 눌린다).
    # 전체 이름은 fig_k_dose_arm 의 패널 제목이 갖고 있다.
    LB = VSHORT if short_lg else VLAB
    h = [Line2D([], [], color=VCOL[v], marker="o", ms=6.0, lw=1.4, mec="white",
                mew=0.9, label=LB[v]) for v in VIOL]
    # 범례와 부제가 겹치지 않게 두 줄로 쌓는다. 범례가 위, 부제가 아래.
    # ⚠️ loc="upper center" 로 두면 범례가 **아래로** 자라 축을 덮는다 (좁은 판형에서
    #    Colour consistency 줄이 100 라벨 위로 들어왔다). 위로 자라게 한다.
    #    앵커는 **축 위쪽 여백에서** 잡는다 — 1.005 로 고정하면 판형마다 간격이 달라진다.
    top = box[1]["top"] if box else 0.925
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False,
               fontsize=8.0 if short_lg else 8.6, handlelength=1.5, handletextpad=0.45,
               columnspacing=1.1 if short_lg else 1.6,
               bbox_to_anchor=(0.5, top + (0.075 if subtitle else 0.022)))
    if subtitle:
        fig.text(0.5, top + 0.022, subtitle, ha="center", va="bottom", fontsize=7.6,
                 color=INK2, style="italic")
    if box:
        fig.subplots_adjust(**box[1])
    else:
        fig.subplots_adjust(left=0.085, right=0.995, top=0.925, bottom=0.205, wspace=0.26)
    save(fig, out, name)


def fig_k(R, out):
    """정확도만. 편향과 탐지가 섞여 있는 값이라 sensitivity 와 따로 둔다."""
    # 축 크기를 fig_k_dose_arm 의 패널 하나와 똑같이 맞춘다 (나란히 놓고 읽는 짝이다).
    # 폭이 좁아 범례는 한 열로 쌓는다.
    _fig_k(R, out, ["acc"], "fig_k_dose", box=matched_single(1.5), short_lg=True)


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
    box = matched_bars(2)
    fig, axes = plt.subplots(1, 2, figsize=box[0])
    W = 0.32
    for pi, (ax, (ramp, flat, title)) in enumerate(zip(axes, groups)):
        frame(ax, (0, 124), (0, 25, 50, 75, 100))
        for gi, v in enumerate(VIOL):
            for bi, (cond, lab, f) in enumerate([(ramp, "ramp", 0.46), (flat, "flat", 0.10)]):
                y = C[(cond, v)]["acc"]
                x = gi + (bi - 0.5) * W
                ax.bar(x, y, width=W * 0.88, color=tint(VCOL[v], f),
                       edgecolor=tint(VCOL[v], max(0, f - 0.28)), lw=0.7, zorder=3)
                ax.annotate(f"{y:.0f}", (x, y + 1.6), ha="center", va="bottom",
                            fontsize=6.8, color=INK, zorder=5)
                ax.annotate(lab, (x, 2.5), ha="center", va="bottom", fontsize=6.2,
                            color="white" if y > 20 else INK2, rotation=90, zorder=6)
            d = C[(flat, v)]["acc"] - C[(ramp, v)]["acc"]
            ax.annotate(f"{d:+.1f}", (gi, 113), ha="center", va="bottom", fontsize=7.6,
                        color=INK if abs(d) > 3 else MUTED, zorder=5,
                        fontweight="bold" if abs(d) > 3 else "normal")
        ax.axhline(50, color=MUTED, lw=0.8, ls=(0, (2.5, 2.5)), zorder=4)
        ax.set_xticks(range(3))
        ax.set_xticklabels([VSHORT[v] for v in VIOL], fontsize=8.0)
        ax.set_xlim(-0.5 - W, 2.5 + W); ax.tick_params(axis="x", length=0, pad=3)
        ax.set_title(f"Moving, {title}", fontsize=9.5, color=INK, pad=4)
        if pi:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("pairwise acc. (%)", fontsize=9.5)
        ax.text(0.5, -0.115, PANEL[pi], transform=ax.transAxes, ha="center", va="top",
                fontsize=9.5, color=INK)
    # 정의(ramp = 등가속, flat = 등속)는 캡션이 받는다 — 그림에 서술을 넣지 않는다 (§8-3)
    fig.text(0.5, 1.00, "numbers above each pair: flat $-$ ramp (pp)",
             ha="center", va="top", fontsize=7.8, color=INK2, style="italic")
    fig.subplots_adjust(top=0.885, bottom=0.125, **box[1])
    save(fig, out, "fig_ramp_vs_flat")


# ----------------------------------------------------------- occlusion
def fig_occlusion(R, out):
    C = {(c["condition"], c["violation_type"]): c for c in R["scoring"]["cells"]}
    arms = [("static_visible", "static_occlusion", "Static"),
            ("moving_visible_flat", "moving_occlusion_flat", "Moving (flat)"),
            ("moving_visible", "moving_occlusion", "Moving (ramp)")]
    box = matched_bars(3)
    fig, axes = plt.subplots(1, 3, figsize=box[0])
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
            ax.annotate(f"{d:+.1f}", (gi, 113), ha="center", va="bottom", fontsize=7.6,
                        color=INK if abs(d) > 3 else MUTED, zorder=5,
                        fontweight="bold" if abs(d) > 3 else "normal")
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
    fig.subplots_adjust(top=0.885, bottom=0.125, **box[1])
    save(fig, out, "fig_occlusion")


# ------------------------------------------------- k dose, 운동 팔별
# fig_occlusion 과 **패널 축을 맞춘다** — 둘 다 패널 = 운동 팔, 색 = 위반이다.
# (한 번은 패널을 위반으로 잡았는데, fig_occlusion 옆에 놓으면 축이 뒤바뀌어 읽히고
#  등속 vs 등가속 비교가 같은 패널 안의 마커 구분으로 묻혔다. 지금은 (b) vs (c) 다.)
# fig_k_dose 는 이 그림을 팔에 대해 합친 것이라, 셋이 같은 규칙으로 읽힌다.
ARMS_K = [("static_visible", "static_occlusion", "Static"),
          ("moving_visible_flat", "moving_occlusion_flat", "Moving (flat)"),
          ("moving_visible", "moving_occlusion", "Moving (ramp)")]


def fig_k_arm(R, out, width=7.6):
    by = {(c["condition"], c["violation_type"], c["sym_k"]): c for c in R["scoring"]["cells"]}
    ks = ["0", "1", "2", "3", "4"]
    xs = {k: (0 if k == "0" else 1.6 + int(k) - 1) for k in ks}
    occx = [xs[k] for k in ks if k != "0"]
    lab, ylim, yt = SPEC["acc"]
    fig, axes = plt.subplots(1, 3, figsize=(width, 2.9))
    for pi, (ax, (vis, occ, title)) in enumerate(zip(axes, ARMS_K)):
        frame(ax, ylim, yt)
        ax.axvspan(min(occx) - 0.5, max(occx) + 0.5, color=BAND, lw=0, zorder=0)
        ax.axhline(50, color=MUTED, lw=0.8, ls=(0, (2.5, 2.5)), zorder=1)
        Y = {v: [by[(vis if k == "0" else occ, v, k)]["acc"] for k in ks] for v in VIOL}
        at = []
        for i in range(len(ks)):
            pos = stack_labels([Y[v][i] for v in VIOL], pad=2.1, ymax=ylim[1] - 3.5)
            at.append({v: pos[j] for j, v in enumerate(VIOL)})
        for v in VIOL:
            y, x = Y[v], [xs[k] for k in ks]
            ax.plot(x[1:], y[1:], "-o", color=VCOL[v], ms=6.0, lw=1.4, zorder=3,
                    mec="white", mew=0.9)
            ax.plot(x[:1], y[:1], "o", color=VCOL[v], ms=6.8, zorder=3,
                    mec="white", mew=0.9)
            ax.plot(x[:2], y[:2], ":", color=tint(VCOL[v], 0.45), lw=1.0, zorder=2)
            for i, xx in enumerate(x):
                ly, lva = at[i][v]
                ax.annotate(f"{y[i]:.0f}", (xx, ly), ha="center", va=lva,
                            fontsize=6.6, color=VCOL[v], zorder=5)
        ax.annotate("chance", (-0.5, 51.5), ha="left", va="bottom", fontsize=6.4,
                    color=MUTED, zorder=1)
        ax.set_xticks(list(xs.values())); ax.set_xticklabels(ks, fontsize=8.4, color=INK)
        ax.set_xlim(-0.55, max(xs.values()) + 0.55)
        ax.tick_params(axis="x", length=0, pad=3)
        ax.annotate("no occl.", (xs["0"], -0.115), xycoords=("data", "axes fraction"),
                    ha="center", va="top", fontsize=7.0, color=INK2, style="italic")
        ax.annotate("occluded — hidden frames (k)", (float(np.mean(occx)), -0.115),
                    xycoords=("data", "axes fraction"), ha="center", va="top",
                    fontsize=7.0, color=INK2, style="italic")
        ax.set_title(title, fontsize=9.5, color=INK, pad=3.5)
        if pi == 0:
            ax.set_ylabel(lab, fontsize=9.5)
        ax.text(0.5, -0.235, PANEL[pi], transform=ax.transAxes, ha="center", va="top",
                fontsize=9.5, color=INK)
    h = [Line2D([], [], color=VCOL[v], marker="o", ms=6.0, lw=1.4, mec="white", mew=0.9,
                label=VSHORT[v]) for v in VIOL]
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, fontsize=8.6,
               handlelength=1.5, handletextpad=0.45, columnspacing=1.6,
               bbox_to_anchor=(0.5, 0.995))   # 패널 제목 위로. 0.927 이면 제목을 덮는다
    fig.subplots_adjust(left=0.075, right=0.995, top=0.905, bottom=0.215, wspace=0.20)
    save(fig, out, "fig_k_dose_arm")
    print("  pairwise acc — 팔별")
    for vis, occ, title in ARMS_K:
        for v in VIOL:
            y = [by[(vis if k == "0" else occ, v, k)]["acc"] for k in ks]
            print(f"    {title:14s} {v:7s} " + "  ".join(f"k{k}={x:6.2f}" for k, x in zip(ks, y)))


# --------------------------------------------- vanish 의 두 방향
# "가림이 시작되면 한 방향만 맞춘다" 를 **원 데이터로** 보인다.
# sensitivity/bias 분해는 같은 사실을 말하지만 독자가 산수를 해야 나온다.
#
# ★ vanish 는 방향이 **설계상 둘뿐**이라 사후 정의가 아니다:
#     object -> empty   pos_obj   vs imp_vanish   (있다가 사라진다)
#     empty  -> object  pos_empty vs imp_appear   (빈 화면에 생긴다)
#   두 방향의 표본이 같으므로 보고되는 정확도는 정확히 둘의 평균이다.
#   shape/colour 는 21/28 쌍이라 "방향" 이 선험적으로 주어지지 않는다 — 여기 못 넣는다.
def fig_vanish_direction(R, out, width=7.6):
    by = {(c["condition"], c["sym_k"]): c["pairs"][0]
          for c in R["scoring"]["cells"] if c["violation_type"] == "vanish"}
    ks = ["0", "1", "2", "3", "4"]
    xs = {k: (0 if k == "0" else 1.6 + int(k) - 1) for k in ks}
    occx = [xs[k] for k in ks if k != "0"]
    DIRS = [("object $\\to$ empty", "rev", tint(BLUE, 0.00), "o"),
            ("empty $\\to$ object", "fwd", tint(BLUE, 0.52), "^")]
    fig, axes = plt.subplots(1, 3, figsize=(width, 2.9))
    for pi, (ax, (vis, occ, title)) in enumerate(zip(axes, ARMS_K)):
        frame(ax, (-6, 112), (0, 25, 50, 75, 100))
        ax.axvspan(min(occx) - 0.5, max(occx) + 0.5, color=BAND, lw=0, zorder=0)
        Y = {key: [by[(vis if k == "0" else occ, k)][key] for k in ks]
             for _, key, _, _ in DIRS}
        mean = [(a + b) / 2 for a, b in zip(Y["fwd"], Y["rev"])]
        x = [xs[k] for k in ks]
        # 보고되는 정확도 = 둘의 평균. 이게 50 에 앉는 것이 요점이다
        ax.plot(x[1:], mean[1:], "-", color=MUTED, lw=2.6, alpha=0.45, zorder=2)
        ax.annotate("mean = reported acc.", (x[-1], mean[-1] - 4), ha="right", va="top",
                    fontsize=6.2, color=MUTED, style="italic", zorder=5)
        for lab, key, col, mk in DIRS:
            y = Y[key]
            ax.plot(x[1:], y[1:], "-", marker=mk, color=col, ms=6.0, lw=1.4, zorder=3,
                    mec="white", mew=0.9, label=lab)
            ax.plot(x[:1], y[:1], marker=mk, color=col, ms=6.8, ls="none", zorder=3,
                    mec="white", mew=0.9)
            ax.plot(x[:2], y[:2], ":", color=tint(col, 0.45), lw=1.0, zorder=2)
        at = [stack_labels([Y[k2][i] for _, k2, _, _ in DIRS], pad=2.1, ymax=108.0)
              for i in range(len(ks))]
        for di, (lab, key, col, mk) in enumerate(DIRS):
            for i, xx in enumerate(x):
                ly, lva = at[i][di]
                ax.annotate(f"{Y[key][i]:.0f}", (xx, ly), ha="center", va=lva,
                            fontsize=6.6, color=col, zorder=5)
        ax.axhline(50, color=MUTED, lw=0.8, ls=(0, (2.5, 2.5)), zorder=1)
        ax.set_xticks(list(xs.values())); ax.set_xticklabels(ks, fontsize=8.4, color=INK)
        ax.set_xlim(-0.55, max(xs.values()) + 0.55)
        ax.tick_params(axis="x", length=0, pad=3)
        ax.annotate("no occl.", (xs["0"], -0.115), xycoords=("data", "axes fraction"),
                    ha="center", va="top", fontsize=7.0, color=INK2, style="italic")
        ax.annotate("occluded — hidden frames (k)", (float(np.mean(occx)), -0.115),
                    xycoords=("data", "axes fraction"), ha="center", va="top",
                    fontsize=7.0, color=INK2, style="italic")
        ax.set_title(title, fontsize=9.5, color=INK, pad=3.5)
        if pi == 0:
            ax.set_ylabel("pairwise acc. (%)", fontsize=9.5)
        ax.text(0.5, -0.235, PANEL[pi], transform=ax.transAxes, ha="center", va="top",
                fontsize=9.5, color=INK)
    h = [Line2D([], [], color=c, marker=mk, ms=6.0, lw=1.4, mec="white", mew=0.9, label=l)
         for l, _, c, mk in DIRS]
    fig.legend(handles=h, loc="lower center", ncol=2, frameon=False, fontsize=8.6,
               handlelength=1.5, handletextpad=0.45, columnspacing=2.0,
               bbox_to_anchor=(0.5, 0.995))
    fig.subplots_adjust(left=0.075, right=0.995, top=0.905, bottom=0.215, wspace=0.20)
    save(fig, out, "fig_vanish_direction")
    print("  vanish 방향 분해")
    for vis, occ, title in ARMS_K:
        for k in ks:
            d = by[(vis if k == "0" else occ, k)]
            print(f"    {title:14s} k={k}  empty→object {d['fwd']:6.1f}   "
                  f"object→empty {d['rev']:6.1f}   평균 {(d['fwd']+d['rev'])/2:6.1f}")


# ---------------------- 방향 비대칭: 예시 한 쌍 -> 물체 순서
# 주장은 두 줄이다:
#     문맥 cube,  불가능한 미래 torus     -> 100% 잡아냄
#     문맥 torus, 불가능한 미래 cube      ->   0% 잡아냄
# 같은 두 물체, 같은 위반(splice 에서 모양이 바뀜), 순서만 뒤집었는데 100 과 0 이다.
# (a) 가 그 한 쌍이고, (b)(c) 가 "그게 물체마다 정해져 있다" 로 일반화한다.
#
# 놓친다 = |p - h(그 미래)| < |p - h(가능한 미래)| 이므로, 못 잡는 물체일수록
# predictor 출력이 그쪽에 가깝다. 그래서 (b)(c) 의 왼쪽이 "기울어 있는 쪽" 이다.
#
# ⚠️ 네 번 갈아엎었다: (낮은,높은) 산점도 / 쌍별 dumbbell / 물체별 막대만 /
#    7x7 행렬. 앞의 셋은 유도량을 눈으로 재야 했고, 행렬은 거울칸을 대각선 건너
#    비교하는 네 단계를 요구해 안 읽혔다. 예시 하나로 읽는 법을 가르치고 나서
#    일반화하는 것이 유일하게 통했다.
#
# ⚠️ "predictor 가 cube 를 생성한다" 는 못 한다. p 가 전역 평균 근처라
#    (|p-mu| 15 vs |h-mu| 31, v10) cube 가 평균에 가까운 것일 수 있다 — E2 가 답한다.
def fig_direction_split(R, out, width=8.8, arm="moving_visible", cond="occluded",
                        example=("cube", "torus")):
    cells = collections.defaultdict(list)
    for c in R["scoring"]["cells"]:
        cells[(c["condition"], c["violation_type"])].append(c)
    vis, occ, mname = next(a for a in ARMS_K if a[0] == arm)
    use = occ if cond == "occluded" else vis

    def directed(c0, viol):
        """{(pre, future): acc}"""
        agg = collections.defaultdict(lambda: [0, 0, 0, 0])
        for cell in cells[(c0, viol)]:
            for p in cell["pairs"]:
                a = agg[(p["a"], p["b"])]
                a[0] += p["fwd"] * p["n_fwd"]; a[1] += p["n_fwd"]
                a[2] += p["rev"] * p["n_rev"]; a[3] += p["n_rev"]
        D = {}
        for (a, b), v in agg.items():
            D[(a, b)] = v[0] / v[1]; D[(b, a)] = v[2] / v[3]
        return D

    def per_future(D):
        hit = collections.defaultdict(list)
        for (pre, fut), a in D.items():
            hit[fut].append(a)
        return {k: float(np.mean(v)) for k, v in hit.items()}

    fig, axes = plt.subplots(1, 3, figsize=(width, 3.4),
                             gridspec_kw=dict(width_ratios=[2.4, 7, 8]))

    # --- (a) 예시 한 쌍
    ax = axes[0]
    D = directed(use, "shape"); A, B = example
    vals = [(f"future = {B}", D[(A, B)]), (f"future = {A}", D[(B, A)])]
    frame(ax, (0, 118), (0, 25, 50, 75, 100))
    ax.axhline(50, color=MUTED, lw=0.9, ls=(0, (2.5, 2.5)), zorder=4)
    for xi, (lab, v) in enumerate(vals):
        ax.bar(xi, v, width=0.55, color=VCOL["shape"] if v > 50 else tint(VCOL["shape"], 0.5),
               edgecolor=tint(VCOL["shape"], 0.0), lw=0.8, zorder=3)
        ax.annotate(f"{v:.0f}%", (xi, v + 2.5), ha="center", va="bottom", fontsize=11,
                    color=INK, zorder=5, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"{A}\n$\\to$ {B}", f"{B}\n$\\to$ {A}"], fontsize=7.6,
                       color=INK, linespacing=1.25)
    ax.set_xlim(-0.7, 1.7); ax.tick_params(axis="x", length=0, pad=3)
    ax.set_ylabel("violations caught (%)", fontsize=9.0)
    ax.set_title("one pair, both ways", fontsize=9.2, color=INK, pad=20)
    ax.annotate("same two objects,\nonly the order swapped", (0.5, 1.012),
                xycoords="axes fraction", ha="center", va="bottom", fontsize=6.9,
                color=INK2, style="italic", linespacing=1.2)

    # --- (b)(c) 물체별
    for ci, (viol, vl) in enumerate((("shape", "Shape"), ("color", "Colour")), start=1):
        ax = axes[ci]
        Pv = per_future(directed(vis, viol)); Po = per_future(directed(use, viol))
        order = sorted(Po, key=Po.get)
        frame(ax, (0, 118), (0, 25, 50, 75, 100))
        ax.axhline(50, color=MUTED, lw=0.9, ls=(0, (2.5, 2.5)), zorder=4)
        W = 0.36
        for gi, k in enumerate(order):
            for bi, (v, f) in enumerate(((Pv[k], 0.55), (Po[k], 0.0))):
                x = gi + (bi - 0.5) * W
                ax.bar(x, v, width=W * 0.88, color=tint(VCOL[viol], f),
                       edgecolor=tint(VCOL[viol], max(0, f - 0.28)), lw=0.7, zorder=3)
                ax.annotate(f"{v:.0f}", (x, v + 1.8), ha="center", va="bottom",
                            fontsize=6.2, color=INK, zorder=5)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, fontsize=6.8, rotation=35, ha="right", color=INK2)
        ax.set_xlim(-0.5 - W, len(order) - 0.5 + W)
        ax.tick_params(axis="x", length=0, pad=2)
        ax.set_yticklabels([])
        ax.set_title(vl, fontsize=9.2, color=INK, pad=20)
        ax.annotate("caught when THIS object is the impossible future",
                    (0.5, 1.012), xycoords="axes fraction", ha="center", va="bottom",
                    fontsize=6.9, color=INK2, style="italic")
        ax.annotate("$\\longleftarrow$  the predictor leans this way", (0.02, 0.955),
                    xycoords="axes fraction", ha="left", va="top", fontsize=7.2,
                    color=INK2, style="italic")
    h = [Patch(facecolor=tint(INK2, 0.55), edgecolor=tint(INK2, 0.27), lw=0.7,
               label="no occluder"),
         Patch(facecolor=INK2, edgecolor=INK, lw=0.7, label="occluded")]
    fig.legend(handles=h, loc="lower center", ncol=2, frameon=False, fontsize=8.4,
               handlelength=1.3, handletextpad=0.5, columnspacing=2.0,
               bbox_to_anchor=(0.5, 0.985))
    fig.subplots_adjust(left=0.075, right=0.995, top=0.735, bottom=0.185, wspace=0.10)
    fig.canvas.draw(); rend = fig.canvas.get_renderer(); inv = fig.transFigure.inverted()
    for ax, lab in zip(axes, PANEL):
        bb = ax.get_tightbbox(rend).transformed(inv)
        fig.text(bb.x0 + bb.width / 2, bb.y0 - 0.02, lab, ha="center", va="top",
                 fontsize=9.5, color=INK)
    fig.text(0.5, -0.055, f"[{mname}, {cond}]  a missed violation means the predictor "
             "output is closer to the impossible future than to the possible one.",
             ha="center", va="top", fontsize=7.4, color=INK2, style="italic")
    save(fig, out, "fig_direction_split")
    print(f"  예시 {A}->{B} {D[(A,B)]:.1f}%   {B}->{A} {D[(B,A)]:.1f}%")
    for viol, vl in (("shape", "Shape"), ("color", "Colour")):
        Po = per_future(directed(use, viol))
        print(f"    {vl:7s} " + "  ".join(f"{k} {Po[k]:.0f}" for k in sorted(Po, key=Po.get)))


# ---------------------- 두 방향 간격을 k x 운동 타입으로
# fig_direction_split 이 보인 비대칭을 숫자 하나로 접어 dose-response 로 본다.
#     간격 = mean |acc(A->B) - acc(B->A)|      (= report 의 bias x 2)
# 0 이면 두 방향이 같고, 100 이면 한 방향만 맞힌다.
# ⚠️ 산술 천장이 있다: 그 셀의 평균 정확도가 m 이면 간격 <= 2*min(m, 100-m).
#    탐지가 살아 있으면 간격이 클 수가 없다. 원값만 비교하지 말 것.
def fig_direction_gap(R, out, width=7.6):
    by = {(c["condition"], c["violation_type"], c["sym_k"]): c for c in R["scoring"]["cells"]}
    ks = ["0", "1", "2", "3", "4"]
    xs = {k: (0 if k == "0" else 1.6 + int(k) - 1) for k in ks}
    occx = [xs[k] for k in ks if k != "0"]
    fig, axes = plt.subplots(1, 3, figsize=(width, 2.9))
    for pi, (ax, (vis, occ, title)) in enumerate(zip(axes, ARMS_K)):
        frame(ax, (-4, 112), (0, 25, 50, 75, 100))
        ax.axvspan(min(occx) - 0.5, max(occx) + 0.5, color=BAND, lw=0, zorder=0)
        Y = {v: [2 * by[(vis if k == "0" else occ, v, k)]["bias"] for k in ks] for v in VIOL}
        at = []
        for i2 in range(len(ks)):
            pos = stack_labels([Y[v][i2] for v in VIOL], pad=2.1, ymax=104.0)
            at.append({v: pos[j2] for j2, v in enumerate(VIOL)})
        for v in VIOL:
            y = Y[v]; x = [xs[k] for k in ks]
            ax.plot(x[1:], y[1:], "-o", color=VCOL[v], ms=6.0, lw=1.4, zorder=3,
                    mec="white", mew=0.9)
            ax.plot(x[:1], y[:1], "o", color=VCOL[v], ms=6.8, zorder=3, mec="white", mew=0.9)
            ax.plot(x[:2], y[:2], ":", color=tint(VCOL[v], 0.45), lw=1.0, zorder=2)
            for i2, xx in enumerate(x):
                ly, lva = at[i2][v]
                ax.annotate(f"{y[i2]:.0f}", (xx, ly), ha="center", va=lva, fontsize=6.6,
                            color=VCOL[v], zorder=5)
        ax.set_xticks(list(xs.values())); ax.set_xticklabels(ks, fontsize=8.4, color=INK)
        ax.set_xlim(-0.55, max(xs.values()) + 0.55)
        ax.tick_params(axis="x", length=0, pad=3)
        ax.annotate("no occl.", (xs["0"], -0.115), xycoords=("data", "axes fraction"),
                    ha="center", va="top", fontsize=7.0, color=INK2, style="italic")
        ax.annotate("occluded — hidden frames (k)", (float(np.mean(occx)), -0.115),
                    xycoords=("data", "axes fraction"), ha="center", va="top",
                    fontsize=7.0, color=INK2, style="italic")
        ax.set_title(title, fontsize=9.5, color=INK, pad=3.5)
        if pi == 0:
            ax.set_ylabel("gap between the two\ndirections (pp)", fontsize=8.8,
                          linespacing=1.3)
        ax.text(0.5, -0.235, PANEL[pi], transform=ax.transAxes, ha="center", va="top",
                fontsize=9.5, color=INK)
    h = [Line2D([], [], color=VCOL[v], marker="o", ms=6.0, lw=1.4, mec="white", mew=0.9,
                label=VSHORT[v]) for v in VIOL]
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, fontsize=8.6,
               handlelength=1.5, handletextpad=0.45, columnspacing=1.6,
               bbox_to_anchor=(0.5, 0.995))
    fig.subplots_adjust(left=0.095, right=0.995, top=0.905, bottom=0.215, wspace=0.20)
    save(fig, out, "fig_direction_gap")
    print("  두 방향 간격 = mean |acc(A->B) - acc(B->A)|")
    for vis, occ, title in ARMS_K:
        for v in VIOL:
            y = [2 * by[(vis if k == "0" else occ, v, k)]["bias"] for k in ks]
            print(f"    {title:14s} {v:7s} " + "  ".join(f"k{k}={x:5.1f}" for k, x in zip(ks, y)))


# ---------------------- 문맥 x 미래 행렬, 대각선 = "그 물체를 지켜내는가"
# 행 = splice 앞(문맥) 물체, 열 = 불가능한 미래 물체, 칸 = acc(행 -> 열).
#
# ⚠️ 대각선에는 **측정값이 없다** — 쌍은 항상 서로 다른 두 물체라 A->A 시행이 없다.
#    대신 **행 평균**을 넣는다: 문맥이 A 일 때 다른 여섯 후보를 상대로 A 를 지켜낸 비율.
#    측정이 아니라 요약이므로 굵은 테두리로 구분하고 안내 문구에 밝힌다.
#
# 행을 지켜냄 순으로 정렬하면 위에서 아래로 색이 빠진다:
#   cylinder 89.6  cube 86.5  ...  torus 3.1   (shape, ramp+OCC)
# = predictor 는 cylinder/cube 는 유지하고 torus 는 전혀 유지하지 못한다.
def fig_keep_matrix(R, out, width=9.0, arm="moving_visible", cond="occluded",
                    name="fig_keep_matrix"):
    cells = collections.defaultdict(list)
    for c in R["scoring"]["cells"]:
        cells[(c["condition"], c["violation_type"])].append(c)
    vis, occ, mname = next(a for a in ARMS_K if a[0] == arm)
    use = occ if cond == "occluded" else vis

    def directed(viol):
        agg = collections.defaultdict(lambda: [0, 0, 0, 0])
        for cell in cells[(use, viol)]:
            for p in cell["pairs"]:
                a = agg[(p["a"], p["b"])]
                a[0] += p["fwd"] * p["n_fwd"]; a[1] += p["n_fwd"]
                a[2] += p["rev"] * p["n_rev"]; a[3] += p["n_rev"]
        D = {}
        for (a, b), v in agg.items():
            D[(a, b)] = v[0] / v[1]; D[(b, a)] = v[2] / v[3]
        return D

    panels = [("vanish", "Object permanence"), ("shape", "Shape"), ("color", "Colour")]
    fig, axes = plt.subplots(1, 3, figsize=(width, 3.6),
                             gridspec_kw=dict(width_ratios=[2, 7, 8]))
    for ci, (viol, vl) in enumerate(panels):
        ax = axes[ci]
        D = directed(viol)
        objs = sorted({x for k in D for x in k})
        keep = {o: float(np.mean([D[(o, x)] for x in objs if x != o])) for o in objs}
        nm = sorted(objs, key=lambda o: -keep[o])
        K = len(nm)
        M = np.zeros((K, K))
        for i, a in enumerate(nm):
            for j, b in enumerate(nm):
                M[i, j] = keep[a] if i == j else D[(a, b)]
        ax.imshow(M, cmap=CMAP_ACC, vmin=0, vmax=100, aspect="equal")
        for i in range(K):
            for j in range(K):
                v = M[i, j]
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        fontsize=6.4 if i != j else 7.2,
                        fontweight="normal" if i != j else "bold",
                        color="white" if abs(v - 50) > 34 else INK2)
                if i == j:                       # 대각선은 요약값이므로 테두리로 구분
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                               edgecolor=INK, lw=1.4, zorder=6))
        ax.set_xticks(range(K)); ax.set_yticks(range(K))
        ax.set_xticklabels(nm, rotation=45, ha="right", fontsize=6.4, color=INK2)
        ax.set_yticklabels(nm, fontsize=6.4, color=INK2)
        ax.tick_params(length=0, pad=1.5)
        for sp in ax.spines.values():
            sp.set_color(MUTED); sp.set_linewidth(0.5); sp.set_zorder(30)
        ax.set_title(vl, fontsize=9.2, color=INK, pad=6)
        if ci == 0:
            ax.set_ylabel("context object (before the splice)", fontsize=8.2, color=INK,
                          labelpad=2)
    fig.subplots_adjust(left=0.085, right=0.995, top=0.855, bottom=0.285, wspace=0.16)
    cax = fig.add_axes([0.050, 0.125, 0.160, 0.026])
    cax.imshow(np.linspace(0, 1, 256)[None, :], cmap=CMAP_ACC, aspect="auto")
    cax.set_yticks([]); cax.set_xticks([0, 127.5, 255])
    cax.set_xticklabels(["0", "50", "100"], fontsize=6.6, color=INK2)
    cax.tick_params(length=0, pad=1.5)
    for sp in cax.spines.values():
        sp.set_color(MUTED); sp.set_linewidth(0.5)
    cax.annotate("always missed", (0, 1.45), xycoords="axes fraction", ha="left",
                 va="bottom", fontsize=6.4, color=ORANGE, style="italic")
    cax.annotate("chance", (0.5, 1.45), xycoords="axes fraction", ha="center",
                 va="bottom", fontsize=6.4, color=MUTED, style="italic")
    cax.annotate("always caught", (1, 1.45), xycoords="axes fraction", ha="right",
                 va="bottom", fontsize=6.4, color=BLUE, style="italic")
    cax.annotate("violations caught (%)", (0.5, -2.6), xycoords="axes fraction",
                 ha="center", va="top", fontsize=6.6, color=INK2)
    fig.canvas.draw(); rend = fig.canvas.get_renderer(); inv = fig.transFigure.inverted()
    for ax, lab in zip(axes, PANEL):
        bb = ax.get_tightbbox(rend).transformed(inv)
        fig.text(bb.x0 + bb.width / 2, bb.y0 - 0.02, lab, ha="center", va="top",
                 fontsize=9.5, color=INK)
    fig.text(0.5, -0.05,
             "ROW = object before the splice.   COLUMN = object in the impossible future."
             "   cell = violations caught (%).\n"
             "BOXED DIAGONAL: A|A is the POSSIBLE clip — the reference every pair in that "
             "row is scored against,\nso it has no accuracy of its own. the box shows the "
             "row mean: how often the score keeps that object.\n"
             f"dark row = the model holds it, pale row = the model loses it.   "
             f"[{mname}, {cond}]",
             ha="center", va="top", fontsize=7.4, color=INK2, style="italic",
             linespacing=1.6)
    save(fig, out, name)
    for viol, vl in panels:
        D = directed(viol); objs = sorted({x for k in D for x in k})
        keep = {o: float(np.mean([D[(o, x)] for x in objs if x != o])) for o in objs}
        print(f"    {vl:18s} " + "  ".join(f"{o} {keep[o]:.0f}"
                                           for o in sorted(objs, key=lambda o: -keep[o])))


# ---------------------- "그 물체를 지켜냄" 을 물체 x k x 운동 타입으로
# fig_keep_matrix 의 대각선(행 평균)만 뽑아 조건축을 편다. 행렬 18장 대신 6패널이다.
#   x = 물체 (모든 패널에서 **같은 순서** — ramp+OCC 기준. 그래야 패널끼리 비교된다)
#   선 = k (0 = 가림막 자체가 없음, 1~4 = 가림)
# 가림이 없으면 평평하고, 가림이 생기면 부챗살처럼 벌어진다. k 선끼리는 거의 겹친다.
def fig_keep_dose(R, out, width=9.2):
    cells = collections.defaultdict(list)
    for c in R["scoring"]["cells"]:
        cells[(c["condition"], c["violation_type"], c["sym_k"])].append(c)

    def keep(cond, viol, k=None):
        agg = collections.defaultdict(lambda: [0, 0, 0, 0])
        for (cd, v, kk), cs in cells.items():
            if cd != cond or v != viol or (k is not None and kk != k):
                continue
            for cell in cs:
                for p in cell["pairs"]:
                    a = agg[(p["a"], p["b"])]
                    a[0] += p["fwd"] * p["n_fwd"]; a[1] += p["n_fwd"]
                    a[2] += p["rev"] * p["n_rev"]; a[3] += p["n_rev"]
        D = {}
        for (a, b), v in agg.items():
            D[(a, b)] = v[0] / v[1]; D[(b, a)] = v[2] / v[3]
        objs = sorted({x for kk2 in D for x in kk2})
        return {o: float(np.mean([D[(o, x)] for x in objs if x != o])) for o in objs}

    rows = [("shape", "Shape"), ("color", "Colour")]
    fig, axes = plt.subplots(2, 3, figsize=(width, 5.2), squeeze=False)
    for ri, (viol, vl) in enumerate(rows):
        ref = keep("moving_occlusion", viol)
        order = sorted(ref, key=lambda o: -ref[o])        # 모든 패널 공통 순서
        for ci, (vis, occ, title) in enumerate(ARMS_K):
            ax = axes[ri][ci]
            frame(ax, (-5, 112), (0, 25, 50, 75, 100))
            ax.axhline(50, color=MUTED, lw=0.8, ls=(0, (2.5, 2.5)), zorder=1)
            x = range(len(order))
            K0 = keep(vis, viol)
            ax.plot(x, [K0[o] for o in order], "--s", color=INK2, lw=1.5, ms=4.6,
                    zorder=5, mec="white", mew=0.8)
            for kk, f in zip("1234", (0.60, 0.42, 0.24, 0.0)):
                Kk = keep(occ, viol, kk)
                ax.plot(x, [Kk[o] for o in order], "-o", color=tint(VCOL[viol], f),
                        lw=1.2, ms=3.6, zorder=3, mec="white", mew=0.6)
            ax.set_xticks(list(x))
            ax.set_xticklabels(order, fontsize=6.4, rotation=35, ha="right", color=INK2)
            ax.set_xlim(-0.5, len(order) - 0.5)
            ax.tick_params(axis="x", length=0, pad=2)
            ax.set_title(f"{vl}  —  {title}", fontsize=8.8, color=INK, pad=3.0)
            if ci == 0:
                ax.set_ylabel("keeps this object (%)", fontsize=8.2, color=INK, labelpad=2)
            else:
                ax.set_yticklabels([])
    h = ([Line2D([], [], color=INK2, marker="s", ls="--", lw=1.5, ms=4.6, mec="white",
                 mew=0.8, label="no occluder")]
         + [Line2D([], [], color=tint(INK2, f), marker="o", ls="-", lw=1.2, ms=3.6,
                   mec="white", mew=0.6, label=f"occluded  k={kk}")
            for kk, f in zip("1234", (0.60, 0.42, 0.24, 0.0))])
    fig.subplots_adjust(top=0.86, bottom=0.115, wspace=0.10, hspace=0.55)
    fig.legend(handles=h, loc="lower center", ncol=5, frameon=False, fontsize=8.2,
               handlelength=1.7, handletextpad=0.5, columnspacing=1.5,
               bbox_to_anchor=(0.5, 0.985))
    fig.canvas.draw(); rend = fig.canvas.get_renderer(); inv = fig.transFigure.inverted()
    for ax, lab in zip([a for r in axes for a in r], PANEL):
        bb = ax.get_tightbbox(rend).transformed(inv)
        fig.text(bb.x0 + bb.width / 2, bb.y0 - 0.014, lab, ha="center", va="top",
                 fontsize=9.0, color=INK)
    fig.text(0.5, 0.012, "x order is fixed across panels (ramp+occluded ranking) so the "
             "panels can be compared. y = row mean of the context x future matrix.",
             ha="center", va="top", fontsize=7.2, color=MUTED, style="italic")
    save(fig, out, "fig_keep_dose")
    print("  지켜냄 — 물체 x k x 운동")
    for viol, vl in rows:
        ref = keep("moving_occlusion", viol)
        order = sorted(ref, key=lambda o: -ref[o])
        print(f"    {vl}  순서 {' > '.join(order)}")
        for vis, occ, title in ARMS_K:
            K0 = keep(vis, viol)
            print(f"      {title:14s} 가림없음 " + " ".join(f"{K0[o]:5.0f}" for o in order))
            for kk in "1234":
                Kk = keep(occ, viol, kk)
                print(f"      {'':14s} k={kk}      " + " ".join(f"{Kk[o]:5.0f}" for o in order))



def _sided_with(cells, cond, viol, k=None):
    """-> (M, objs).  모든 칸 = **predictor 가 열 물체 쪽으로 간 비율(%)**.
    비대각 M[A,B] = 문맥 A, 도전자 B 경기에서 B 편을 든 비율 (= 100 - acc(A→B)).
    대각   M[A,A] = 그 행 도전자 전체 평균으로 **자기 자신에 남은** 비율 (= mean acc).

    의미가 한 방향으로 통일된다 — "출력이 이 물체 쪽으로 갔다".
    그래서 **완벽한 predictor 는 대각만 진하다.** 비대각이 진하면 그쪽으로 샌 것이다.

    ⚠️ 세 번 갈아엎었다.
       (1) 행 합 100% 정규화 — 비대각 상한이 100/(K-1)=16.7 이라 "16.7" 이 정확도 20% 로 오독됨.
       (2) 모든 칸을 acc(지킴)로 — 완벽한 predictor 면 판 전체가 진해져서 "무엇을 만드나" 가 안 보임.
       (3) 지금: 모든 칸이 "열 물체 쪽으로 갔나". 대각만 진한 것이 좋은 상태다.
    """
    ag = collections.defaultdict(lambda: [0, 0, 0, 0])
    for (cd, v, kk), cs in cells.items():
        if cd != cond or v != viol or (k is not None and kk != k):
            continue
        for cell in cs:
            for q in cell["pairs"]:
                x = ag[(q["a"], q["b"])]
                x[0] += q["fwd"] * q["n_fwd"]; x[1] += q["n_fwd"]
                x[2] += q["rev"] * q["n_rev"]; x[3] += q["n_rev"]
    D = {}
    for (a, b), v in ag.items():
        D[(a, b)] = (v[0] / v[1], v[1]); D[(b, a)] = (v[2] / v[3], v[3])
    objs = sorted({x for kk2 in D for x in kk2})
    ix = {o: i for i, o in enumerate(objs)}
    M = np.zeros((len(objs), len(objs)))
    for a in objs:
        tot = sum(D[(a, b)][1] for b in objs if b != a)
        M[ix[a], ix[a]] = 100 * sum(D[(a, b)][0] / 100 * D[(a, b)][1]
                                    for b in objs if b != a) / tot
        for b in objs:
            if b != a:
                M[ix[a], ix[b]] = 100 - D[(a, b)][0]
    return M, objs


# ------------- 문맥이 A 일 때 predictor 는 누구 편을 들었나
# 한 시행 = matched pair 하나. 문맥이 A 이고 도전자가 B 일 때
#     |p - h(A)| < |p - h(B)|  이면 A 에 표를 준 것,  아니면 B 에 준 것이다.
# A 의 모든 시행(도전자 6~7명 x 각 n)을 모아 **행 합 100%** 로 정규화한다:
#     대각   = 도전자 전체 평균으로 A 를 지킨 비율
#     (A,B) = **A 대 B 경기에서** B 쪽으로 간 비율 (= 100 - acc(A→B))
#    모든 칸이 같은 0~100 정확도 척도다. 100 이면 그 경기 전패.
#
# ⚠️ "predictor 가 B 를 예측했다" 로 읽으면 안 된다. 각 경기는 A 와 B **둘만** 후보로
#    주므로 B 표는 "B 가 후보였을 때" 조건부다. 실제로 torus 행은 여섯 도전자에게
#    거의 균등하게(14.6~16.7) 샌다 — 특정 물체로 가는 것이 아니라 자기에게서 멀어진다.
#
# ⚠️ block 밖의 후보로 7-way retrieval 을 하면 안 된다. 실측(등가속+가림, shape):
#       같은 block 다른 모양 0.5438 / 다른 block 같은 모양 0.7602  (신호/잡음 0.715)
#    h 를 질의로 쓴 대조군조차 프로토타입 retrieval 40.3% (chance 14.3) 다.
#    -> z_research/scripts/analysis/retrieval_confusion.py 최상단. 다시 시도하지 말 것.
def fig_predictor_vote(R, out, width=9.4, arm="moving_visible", cond="occluded"):
    cells = collections.defaultdict(list)
    for c in R["scoring"]["cells"]:
        cells[(c["condition"], c["violation_type"], c["sym_k"])].append(c)
    vis, occ, mname = next(a for a in ARMS_K if a[0] == arm)
    use = occ if cond == "occluded" else vis

    def votes(viol):
        M, objs = _sided_with(cells, use, viol)
        order = sorted(range(len(objs)), key=lambda i2: -M[i2, i2])
        return M[np.ix_(order, order)], [objs[i2] for i2 in order]

    panels = [("shape", "Shape"), ("color", "Colour")]
    fig, axes = plt.subplots(1, 2, figsize=(width, 4.3),
                             gridspec_kw=dict(width_ratios=[7, 8]))
    for ci, (viol, vl) in enumerate(panels):
        ax = axes[ci]
        M, nm = votes(viol)
        K = len(nm)
        # 모든 칸이 "이 물체 쪽으로 갔다" 로 의미가 같으므로 단색이 맞다.
        cm = LinearSegmentedColormap.from_list("v", ["#ffffff", tint(VCOL[viol], 0.0)])
        ax.imshow(M, cmap=cm, vmin=0, vmax=100, aspect="equal")
        for a2 in range(K):
            for b2 in range(K):
                v = M[a2, b2]
                ax.text(b2, a2, f"{v:.0f}",
                        ha="center", va="center",
                        fontsize=7.0 if a2 == b2 else 6.1,
                        fontweight="bold" if a2 == b2 else "normal",
                        color="white" if v > 62 else INK2, zorder=5)
                if a2 == b2:
                    ax.add_patch(plt.Rectangle((b2 - .5, a2 - .5), 1, 1, fill=False,
                                               edgecolor=INK, lw=1.4, zorder=6))
                elif v > 99.5:           # 그 경기 전패
                    ax.add_patch(plt.Rectangle((b2 - .5, a2 - .5), 1, 1, fill=False,
                                               edgecolor=INK, lw=0.9, ls=(0, (2, 1.5)),
                                               zorder=6))
        ax.set_xticks(range(K)); ax.set_yticks(range(K))
        ax.set_xticklabels(nm, rotation=45, ha="right", fontsize=6.6, color=INK2)
        ax.set_yticklabels(nm, fontsize=6.6, color=INK2)
        ax.tick_params(length=0, pad=1.5)
        for sp in ax.spines.values():
            sp.set_color(MUTED); sp.set_linewidth(0.5); sp.set_zorder(30)
        ax.set_title(vl, fontsize=9.5, color=INK, pad=6)
        if ci == 0:
            ax.set_ylabel("context object", fontsize=8.4, color=INK, labelpad=2)
        ax.set_xlabel("which object the output went to", fontsize=8.4, color=INK,
                      labelpad=1.5)
    fig.subplots_adjust(left=0.085, right=0.995, top=0.90, bottom=0.175, wspace=0.20)
    fig.canvas.draw(); rend = fig.canvas.get_renderer(); inv = fig.transFigure.inverted()
    for ax, lab in zip(axes, PANEL):
        bb = ax.get_tightbbox(rend).transformed(inv)
        fig.text(bb.x0 + bb.width / 2, bb.y0 - 0.02, lab, ha="center", va="top",
                 fontsize=9.5, color=INK)
    # 서술 문장은 그림에 넣지 않는다 — 캡션으로 뺀다 (CLAUDE.md §8-3)
    save(fig, out, "fig_predictor_vote")
    for viol, vl in panels:
        M, nm = votes(viol); K = len(nm)
        print(f"    {vl}")
        for i2, o in enumerate(nm):
            off = np.delete(M[i2], i2)
            print(f"      {o:10s} 지킴 {M[i2,i2]:5.1f}   전패한 상대 "
                  f"{int((off > 99.5).sum())}/{K-1}   최대 이탈 {off.max():5.1f}")


# ------------- 같은 투표 행렬을 운동 타입 x k 로 (small multiples)
# ⚠️ (조건, k) 로 쪼개면 비대각 칸이 **방향당 n=4** 다 (shape). 개별 칸을 읽으면 안 된다.
#    그래서 숫자를 빼고 색만 쓴다 — **대각선이 어떻게 옅어지는가** 만 읽는 그림이다.
#    대각선은 도전자 6~7명을 합치므로 n=24~28 (SE ~10pp) 로 그나마 읽을 수 있다.
# ⚠️ 행·열 순서를 **모든 패널에서 고정**한다 (등가속+가림 순위). 그래야 패널끼리 비교된다.
def fig_vote_grid(R, out, target="shape", width=8.6):
    cells = collections.defaultdict(list)
    for c in R["scoring"]["cells"]:
        cells[(c["condition"], c["violation_type"], c["sym_k"])].append(c)

    def votes(cond, viol, k=None):
        return _sided_with(cells, cond, viol, k)

    Mref, objs = votes("moving_occlusion", target)
    order = sorted(range(len(objs)), key=lambda i2: -Mref[i2, i2])
    nm = [objs[i2] for i2 in order]                      # 모든 패널 공통 순서
    K = len(nm)
    cm = LinearSegmentedColormap.from_list("v", ["#ffffff", tint(VCOL[target], 0.0)])
    ks = ["0", "1", "2", "3", "4"]
    fig, axes = plt.subplots(3, 5, figsize=(width, 6.3), squeeze=False)
    for ri, (vis, occ, title) in enumerate(ARMS_K):
        for ci, k in enumerate(ks):
            ax = axes[ri][ci]
            M, o2 = votes(vis if k == "0" else occ, target, None if k == "0" else k)
            M = M[np.ix_(order, order)]
            ax.imshow(M, cmap=cm, vmin=0, vmax=100, aspect="equal")
            for d2 in range(K):
                ax.add_patch(plt.Rectangle((d2 - .5, d2 - .5), 1, 1, fill=False,
                                           edgecolor=INK, lw=0.8, zorder=6))
                # 대각선은 칸마다 값을 적는다 (평균 하나로 뭉치지 않는다)
                ax.text(d2, d2, f"{M[d2, d2]:.0f}", ha="center", va="center",
                        fontsize=4.8, color="white" if M[d2, d2] > 62 else INK2,
                        zorder=7)
            ax.set_xticks(range(K)); ax.set_yticks(range(K))
            ax.set_xticklabels(nm if ri == 2 else [], rotation=45, ha="right",
                               fontsize=5.6, color=INK2)
            ax.set_yticklabels(nm if ci == 0 else [], fontsize=5.6, color=INK2)
            ax.tick_params(length=0, pad=1.2)
            for sp in ax.spines.values():
                sp.set_color(MUTED); sp.set_linewidth(0.4)
            if ri == 0:
                ax.set_title("no occluder" if k == "0" else f"k={k}", fontsize=8.4,
                             color=INK, pad=4)
            if ci == 0:
                ax.set_ylabel(title, fontsize=8.4, color=INK, labelpad=10)
    fig.subplots_adjust(left=0.115, right=0.905, top=0.945, bottom=0.105,
                        wspace=0.10, hspace=0.10)
    cax = fig.add_axes([0.925, 0.105, 0.018, 0.840])      # 색막대는 맨 오른쪽
    cax.imshow(np.linspace(1, 0, 256)[:, None], cmap=cm, aspect="auto")
    cax.set_xticks([]); cax.yaxis.tick_right()
    cax.set_yticks([0, 127.5, 255])
    cax.set_yticklabels(["100", "50", "0"], fontsize=6.6, color=INK2)
    cax.tick_params(length=0, pad=2)
    for sp in cax.spines.values():
        sp.set_color(MUTED); sp.set_linewidth(0.5)
    cax.set_ylabel("output went to this object (%)", fontsize=7.0, color=INK2, labelpad=3)
    cax.yaxis.set_label_position("right")
    # 서술 문장은 그림에 넣지 않는다 — 캡션으로 뺀다 (CLAUDE.md §8-3)
    save(fig, out, f"fig_vote_grid_{target}")
    print(f"  투표 행렬 대각선 평균 — {target}")
    for vis, occ, title in ARMS_K:
        row = []
        for k in ks:
            M, _ = votes(vis if k == "0" else occ, target, None if k == "0" else k)
            row.append(np.trace(M) / K)
        print(f"    {title:14s} " + "  ".join(f"k{k}={v:5.1f}" for k, v in zip(ks, row)))


# ------------- "문맥이 A 일 때 predictor 가 각 물체를 만들 확률"
# ⚠️ **2026-08-31 물러남 (_superseded).** 수식은 맞지만 요약치로 못 쓴다.
#    s = (1-p)/p 가 p 가 작을 때 폭발해서 **최저 상대 하나가 행 전체를 지배한다.**
#    실측(등가속·비가림, shape): cube 는 짝평균 83.3 인데 cylinder 에게만 6% 라
#    7지선다 P 가 8.7 로 내려앉는다. capsule 69.8 -> 2.8, pyramid 64.6 -> 9.4.
#    스무딩 탓이 아니다 (Jeffreys vs add-1 차이 0.1~6.0pt).
#    그리고 k 별 판은 비대각이 **방향당 n=4** 인데 대각이 그 6칸 전부의 함수라
#    "개별 칸을 읽지 말라" 던 잡음을 대각선이 통째로 물려받는다.
#    IIA(무관한 대안의 독립) 가정도 행 안에서 검정 불가능하다.
#    -> 논문 그림은 fig_predictor_vote (2지선다 원자료) 를 쓴다.
# 한 경기는 2지선다다 — 문맥 A, 후보 {A, B}. 거기서 나오는 것은 P(A | {A,B}) 뿐이라
# 7지선다 확률이 그대로 나오지 않는다. Luce 선택 공리를 쓰면 후보마다 척도 s 가 있고
#     P(A | {A,B}) = s_A / (s_A + s_B)
# 이므로 s_A = 1 로 두면 s_B = (1 - p) / p 로 **정확히** 풀린다 (행마다 미지수 6, 식 6).
#     P(X | 문맥 A) = s_X / sum_Y s_Y      <- 행 합 100%, chance = 100/K
#
# ⚠️ 이건 새 주장이 아니라 **같은 수의 재표현**이다. 행이 정확히 결정되므로
#    행 안에서는 적합도 검정이 불가능하다. Luce 가정(무관한 대안의 독립)이 들어간다.
#    가정의 검정은 "행이 달라도 s 가 같은가" 이고 그게 beat 4 의 1차원 순서다.
# ⚠️ 앞서 기각된 '행 합 100% 정규화'와 다르다. 그건 원 횟수를 그냥 나눠서 대각 상한이
#    1/K 로 눌렸다. 여기선 전승이면 대각이 100 으로 간다 (s_B -> 0).
# ⚠️ 스무딩은 Jeffreys (x+0.5)/(n+1). n 이 작을 때 p=0/1 이 무한대로 튀는 것을 막는다.
def _make_prob(cells, cond, viol, k=None):
    ag = collections.defaultdict(lambda: [0.0, 0, 0.0, 0])
    for (cd, v, kk), cs in cells.items():
        if cd != cond or v != viol or (k is not None and kk != k):
            continue
        for cell in cs:
            for q in cell["pairs"]:
                x = ag[(q["a"], q["b"])]
                x[0] += q["fwd"] / 100 * q["n_fwd"]; x[1] += q["n_fwd"]
                x[2] += q["rev"] / 100 * q["n_rev"]; x[3] += q["n_rev"]
    P = {}
    for (a, b), v in ag.items():
        P[(a, b)] = (v[0] + 0.5) / (v[1] + 1)          # 문맥 a 가 자기를 지킨 비율
        P[(b, a)] = (v[2] + 0.5) / (v[3] + 1)
    objs = sorted({x for kk2 in P for x in kk2}); ix = {o: i for i, o in enumerate(objs)}
    M = np.zeros((len(objs), len(objs)))
    for a in objs:
        s = {a: 1.0}
        for b in objs:
            if b != a:
                pa = min(max(P[(a, b)], 1e-6), 1 - 1e-6)
                s[b] = (1 - pa) / pa
        tot = sum(s.values())
        for b in objs:
            M[ix[a], ix[b]] = 100 * s[b] / tot
    return M, objs


def fig_make_prob(R, out, width=9.4, arm="moving_visible", cond="occluded"):
    cells = collections.defaultdict(list)
    for c in R["scoring"]["cells"]:
        cells[(c["condition"], c["violation_type"], c["sym_k"])].append(c)
    vis, occ, mname = next(a for a in ARMS_K if a[0] == arm)
    use = occ if cond == "occluded" else vis
    panels = [("shape", "Shape"), ("color", "Colour")]
    fig, axes = plt.subplots(1, 2, figsize=(width, 4.3),
                             gridspec_kw=dict(width_ratios=[7, 8]))
    for ci, (viol, vl) in enumerate(panels):
        ax = axes[ci]
        M0, objs = _make_prob(cells, use, viol)
        order = sorted(range(len(objs)), key=lambda i2: -M0[i2, i2])
        M = M0[np.ix_(order, order)]; nm = [objs[i2] for i2 in order]; K = len(nm)
        ch = 100 / K
        cm = LinearSegmentedColormap.from_list("v", ["#ffffff", tint(VCOL[viol], 0.0)])
        ax.imshow(M, cmap=cm, vmin=0, vmax=100, aspect="equal")
        for a2 in range(K):
            for b2 in range(K):
                v = M[a2, b2]
                ax.text(b2, a2, f"{v:.0f}", ha="center", va="center",
                        fontsize=7.0 if a2 == b2 else 6.1,
                        fontweight="bold" if a2 == b2 else "normal",
                        color="white" if v > 62 else INK2, zorder=5)
            ax.add_patch(plt.Rectangle((a2 - .5, a2 - .5), 1, 1, fill=False,
                                       edgecolor=INK, lw=1.4, zorder=6))
        ax.set_xticks(range(K)); ax.set_yticks(range(K))
        ax.set_xticklabels(nm, rotation=45, ha="right", fontsize=6.6, color=INK2)
        ax.set_yticklabels(nm, fontsize=6.6, color=INK2)
        ax.tick_params(length=0, pad=1.5)
        for sp in ax.spines.values():
            sp.set_color(MUTED); sp.set_linewidth(0.5); sp.set_zorder(30)
        ax.set_title(f"{vl}    (chance {ch:.1f})", fontsize=9.5, color=INK, pad=6)
        if ci == 0:
            ax.set_ylabel("context object", fontsize=8.4, color=INK, labelpad=2)
        ax.set_xlabel("P(predictor makes this object)  %", fontsize=8.4, color=INK,
                      labelpad=1.5)
    fig.subplots_adjust(left=0.085, right=0.995, top=0.90, bottom=0.175, wspace=0.20)
    fig.canvas.draw(); rend = fig.canvas.get_renderer(); inv = fig.transFigure.inverted()
    for ax, lab in zip(axes, PANEL):
        bb = ax.get_tightbbox(rend).transformed(inv)
        fig.text(bb.x0 + bb.width / 2, bb.y0 - 0.02, lab, ha="center", va="top",
                 fontsize=9.5, color=INK)
    save(fig, out, "fig_make_prob")
    for viol, vl in panels:
        M0, objs = _make_prob(cells, use, viol)
        order = sorted(range(len(objs)), key=lambda i2: -M0[i2, i2])
        print(f"    {vl:8s} P(자기 자신) — " + "  ".join(
            f"{objs[i2]} {M0[i2, i2]:.0f}" for i2 in order))


def fig_make_prob_grid(R, out, target="shape", width=8.6):
    cells = collections.defaultdict(list)
    for c in R["scoring"]["cells"]:
        cells[(c["condition"], c["violation_type"], c["sym_k"])].append(c)
    Mref, objs = _make_prob(cells, "moving_occlusion", target)
    order = sorted(range(len(objs)), key=lambda i2: -Mref[i2, i2])
    nm = [objs[i2] for i2 in order]; K = len(nm)
    cm = LinearSegmentedColormap.from_list("v", ["#ffffff", tint(VCOL[target], 0.0)])
    ks = ["0", "1", "2", "3", "4"]
    fig, axes = plt.subplots(3, 5, figsize=(width, 6.3), squeeze=False)
    for ri, (vis, occ, title) in enumerate(ARMS_K):
        for ci, k in enumerate(ks):
            ax = axes[ri][ci]
            M, _ = _make_prob(cells, vis if k == "0" else occ, target,
                              None if k == "0" else k)
            M = M[np.ix_(order, order)]
            ax.imshow(M, cmap=cm, vmin=0, vmax=100, aspect="equal")
            for d2 in range(K):
                ax.add_patch(plt.Rectangle((d2 - .5, d2 - .5), 1, 1, fill=False,
                                           edgecolor=INK, lw=0.8, zorder=6))
                ax.text(d2, d2, f"{M[d2, d2]:.0f}", ha="center", va="center",
                        fontsize=4.8, color="white" if M[d2, d2] > 62 else INK2, zorder=7)
            ax.set_xticks(range(K)); ax.set_yticks(range(K))
            ax.set_xticklabels(nm if ri == 2 else [], rotation=45, ha="right",
                               fontsize=5.6, color=INK2)
            ax.set_yticklabels(nm if ci == 0 else [], fontsize=5.6, color=INK2)
            ax.tick_params(length=0, pad=1.2)
            for sp in ax.spines.values():
                sp.set_color(MUTED); sp.set_linewidth(0.4)
            if ri == 0:
                ax.set_title("no occluder" if k == "0" else f"k={k}", fontsize=8.4,
                             color=INK, pad=4)
            if ci == 0:
                ax.set_ylabel(title, fontsize=8.4, color=INK, labelpad=10)
    fig.subplots_adjust(left=0.115, right=0.905, top=0.945, bottom=0.105,
                        wspace=0.10, hspace=0.10)
    cax = fig.add_axes([0.925, 0.105, 0.018, 0.840])
    cax.imshow(np.linspace(1, 0, 256)[:, None], cmap=cm, aspect="auto")
    cax.set_xticks([]); cax.yaxis.tick_right()
    ch = 100 / K
    cax.set_yticks([0, 255 * (1 - ch / 100), 255])
    cax.set_yticklabels(["100", f"{ch:.0f} (chance)", "0"], fontsize=6.6, color=INK2)
    cax.tick_params(length=0, pad=2)
    for sp in cax.spines.values():
        sp.set_color(MUTED); sp.set_linewidth(0.5)
    cax.set_ylabel("P(predictor makes this object)  %", fontsize=7.0, color=INK2, labelpad=3)
    cax.yaxis.set_label_position("right")
    save(fig, out, f"fig_make_prob_grid_{target}")
    print(f"  P(자기 자신) 대각 평균 — {target}")
    for vis, occ, title in ARMS_K:
        row = []
        for k in ks:
            M, _ = _make_prob(cells, vis if k == "0" else occ, target,
                              None if k == "0" else k)
            row.append(f"k{k}={np.mean(np.diag(M)):5.1f}")
        print(f"    {title:14s} " + "  ".join(row))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--which", nargs="*", default=["k", "karm", "sens", "rampflat", "occlusion", "vandir", "dirsplit", "dirgap", "keep", "keepvis", "keepdose", "vote", "votegrid", "prob", "probgrid"])
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
        {"k": fig_k, "karm": fig_k_arm, "sens": fig_sens, "rampflat": fig_rampflat,
         "occlusion": fig_occlusion, "vandir": fig_vanish_direction,
         "dirsplit": fig_direction_split, "dirgap": fig_direction_gap, "keep": fig_keep_matrix,
         "keepvis": lambda R2, o2: fig_keep_matrix(R2, o2, cond="visible",
                                                   name="fig_keep_matrix_visible"),
         "keepdose": fig_keep_dose, "vote": fig_predictor_vote,
         "prob": fig_make_prob,
         "probgrid": lambda R2, o2: [fig_make_prob_grid(R2, o2, t)
                                     for t in ("shape", "color")],
         "votegrid": lambda R2, o2: [fig_vote_grid(R2, o2, t)
                                     for t in ("shape", "color")]}[w](R, a.outdir)


if __name__ == "__main__":
    main()
