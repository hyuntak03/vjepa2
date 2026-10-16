#!/usr/bin/env python3
"""VOBench_v1 채점 그림 (2026-10-10). 입력 = 채점 실행의 per_block.json + index (vobench_tables.py 의 load · stat 를 그대로 쓴다).

  **primitive · realistic 은 따로 그린다** — 모든 그림이 행 2 (위 primitive, 아래 realistic) × 열 3 (위반).
  fig_timing          x = 가림 시점 (visible · early · mid · late), 막대 = 정확도 [block 부트스트랩 CI], 점 = A · B 방향
  fig_k               x = k (0 = 가림 없음, 1–3), 선 = 가림 시점 (early · mid · late)
  fig_late_motion     late 만: x = 운동 6 종, 막대 = 정확도, 점 = A · B 방향
  fig_motion_{visible,early,mid}   같은 꼴, 가림 시점만 바꿈 (2026-10-11)
  fig_permanence_context  permanence 만, 가능한 미래 둘로 쪼갬: 물체 문맥 (있는데→있는, 불가능 = 사라짐) vs 빈 문맥 (없는데→없는, 불가능 = 나타남).
                      행 = objset, 열 = (가림 시점별 · late × 운동)

스타일 = plot_v11_surprise.py (rcParams · 팔레트 · frame · PANEL). 그린 수치는 tables.json 과 대조해 다르면 죽는다.

  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P z_research/scripts/figures/plot_vobench.py --model vjepa2           # → z_research/VOBench_v1/vjepa2/surprise/figures/
  $P z_research/scripts/figures/plot_vobench.py --model dino_foresight   # → z_research/VOBench_v1/dino_foresight/surprise/figures/
  (먼저 vobench_tables.py --model <같은 이름> 으로 <모델>/surprise/tables.json 을 만든다 — 대조용)
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "analysis"))
from plot_v11_surprise import BLUE, ORANGE, GREEN, INK, INK2, MUTED, BAND, PANEL, frame, tint, stack_labels  # noqa: E402  (rcParams 도 같이)
import vobench_tables as VT  # noqa: E402

OUT = VT.ROOT / "vjepa2" / "surprise" / "figures"   # main() 이 --model 로 바꾼다 → z_research/VOBench_v1/<모델>/surprise/figures/
VIOL = VT.VIOL
VLAB = {"permanence": "Object permanence", "shape": "Shape consistency", "color": "Colour consistency"}
VCOL = {"permanence": BLUE, "shape": ORANGE, "color": GREEN}
OCC = VT.OCC
OLAB = {"visible": "visible", "early": "early", "mid": "mid", "late": "late"}
MOT = VT.MOT
MLAB = {"flat_static": "static", "flat_vel": "const.\nvel.", "flat_acc": "accel.", "flat_dec": "decel.",
        "ramp_down": "ramp\ndown", "ramp_up": "ramp\nup"}
# 방향 표식: A = (pos_a, imp_ab), B = (pos_b, imp_ba). permanence 에서 A 의 위반 = 물체→빈, B = 빈→물체 (캡션에 적는다)
DIRS = [("A", "o", 0.00), ("B", "^", 0.45)]          # 세 번째 = 검정 쪽으로 섞는 비율 (막대 tint 위에서 보이게 둘 다 진하게)


def shade(h, f):
    r, g, b = (int(h[i:i + 2], 16) for i in (1, 3, 5))
    return "#%02x%02x%02x" % tuple(round(c * (1 - f)) for c in (r, g, b))


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in (".pdf", ".png"):
        fig.savefig((OUT / name).with_suffix(ext), dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [saved] {OUT / name}.pdf  + .png")


def bars_with_dirs(ax, xs, stats, col, W=0.62, lab_y=True):
    """막대 = 정확도 + CI, 그 위에 A · B 방향 점. 막대 위 숫자 = 정확도."""
    for x, s in zip(xs, stats):
        ax.bar(x, s["acc"], width=W, color=tint(col, 0.55), edgecolor=tint(col, 0.25), lw=0.7, zorder=3)
        ax.errorbar(x, s["acc"], yerr=[[s["acc"] - s["lo"]], [s["hi"] - s["acc"]]], fmt="none",
                    ecolor=INK2, elinewidth=0.8, capsize=2.0, capthick=0.8, zorder=4)
        for d, mk, f in DIRS:
            ax.plot(x + (-0.14 if d == "A" else 0.14), s[d], marker=mk, ls="none", ms=5.0,
                    color=shade(col, f), mec="white", mew=0.8, zorder=6, clip_on=False)
        if lab_y:                                                   # 정확도는 막대 안 바닥에 — 방향 점과 안 겹친다
            ax.annotate(f"{s['acc']:.0f}", (x, 6.0), ha="center", va="bottom", fontsize=6.8, color=INK, zorder=7)


def dir_legend(fig, y=1.0):
    h = [Patch(facecolor=tint(INK2, 0.55), edgecolor=tint(INK2, 0.25), lw=0.7, label="pairwise acc. (95% CI)")]
    h += [Line2D([], [], marker=mk, ls="none", ms=5.5, color=shade("#7a7a7a", f), mec="white", mew=0.8,
                 label=f"direction {d}") for d, mk, f in DIRS]
    fig.legend(handles=h, loc="upper center", ncol=3, frameon=False, fontsize=7.8, handletextpad=0.4,
               columnspacing=1.8, bbox_to_anchor=(0.5, y))


OBJSETS = ("primitive", "realistic")
OTITLE = {"primitive": "Primitive", "realistic": "Realistic"}


def bar_panels(axes, P, filt, xkey, xvals, xlab, title, panel0, band=None, W=0.62, xfs=8.0):
    """한 행 (위반 3 패널): 막대 + A · B 점. 반환 {위반: {x값: stat}}."""
    js = {}
    for pi, (ax, v) in enumerate(zip(axes, VIOL)):
        frame(ax, (-3, 112), (0, 25, 50, 75, 100))
        if band is not None:
            ax.axvspan(band - 0.5, band + 0.5, color=BAND, lw=0, zorder=0)
        st = [VT.stat([p for p in P if filt(p) and p["viol"] == v and p[xkey] == x]) for x in xvals]
        js[v] = dict(zip(xvals, st))
        bars_with_dirs(ax, range(len(xvals)), st, VCOL[v], W=W)
        ax.axhline(50, color=MUTED, lw=0.8, ls=(0, (2.5, 2.5)), zorder=2)
        ax.set_xticks(range(len(xvals))); ax.set_xticklabels([xlab[x] for x in xvals], fontsize=xfs, linespacing=0.95)
        ax.set_xlim(-0.6, len(xvals) - 0.4); ax.tick_params(axis="x", length=0, pad=3)
        ax.set_title(title(v), fontsize=9.0, color=INK, pad=4)
        if pi:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("pairwise acc. (%)", fontsize=9.0)
        ax.text(0.5, -0.17 if xfs >= 8 else -0.27, PANEL[panel0 + pi], transform=ax.transAxes, ha="center", va="top",
                fontsize=9.5, color=INK)
    return js


def fig_timing(P):
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 5.1))
    js = {}
    for ri, o in enumerate(OBJSETS):
        js[o] = bar_panels(axes[ri], P, lambda p, o=o: p["objset"] == o, "occ", OCC, OLAB,
                           lambda v, o=o: f"{OTITLE[o]}: {VLAB[v]}", 3 * ri, band=3)
    dir_legend(fig, 1.01)
    fig.subplots_adjust(left=0.075, right=0.995, top=0.925, bottom=0.085, wspace=0.09, hspace=0.42)
    save(fig, "fig_timing")
    return js


def fig_motion(P, occ="late"):
    """가림 시점 하나 (visible · early · mid · late) 만: x = 운동 6 종. late 판의 이름은 옛 그대로 fig_late_motion."""
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 5.5))
    js = {}
    for ri, o in enumerate(OBJSETS):
        js[o] = bar_panels(axes[ri], P, lambda p, o=o: p["objset"] == o and p["occ"] == occ, "motion", MOT, MLAB,
                           lambda v, o=o: f"{OTITLE[o]}: {VLAB[v]} ({occ})", 3 * ri, W=0.66, xfs=6.8)
    dir_legend(fig, 1.01)
    fig.subplots_adjust(left=0.075, right=0.995, top=0.93, bottom=0.11, wspace=0.09, hspace=0.55)
    save(fig, "fig_late_motion" if occ == "late" else f"fig_motion_{occ}")
    return js


def fig_late_motion(P):
    return fig_motion(P, "late")


def fig_k(P):
    TIM = [("early", "s", 0.55), ("mid", "D", 0.28), ("late", "o", 0.00)]
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 5.2), sharey=True)
    js = {}
    for ri, o in enumerate(OBJSETS):
        js[o] = {}
        for pi, (ax, v) in enumerate(zip(axes[ri], VIOL)):
            frame(ax, (-3, 112), (0, 25, 50, 75, 100))
            ax.axvspan(0.5, 3.5, color=BAND, lw=0, zorder=0)
            sel = lambda p, t, k=None: p["objset"] == o and p["viol"] == v and p["occ"] == t and (k is None or p["k"] == k)
            v0 = VT.stat([p for p in P if sel(p, "visible")])
            js[o][v] = {"visible": v0}
            for t, mk, f in TIM:
                st = [VT.stat([p for p in P if sel(p, t, k)]) for k in ("1", "2", "3")]
                js[o][v][t] = dict(zip(("1", "2", "3"), st))
                col = tint(VCOL[v], f) if f else VCOL[v]
                ax.plot([1, 2, 3], [s["acc"] for s in st], "-", marker=mk, color=col, ms=5.0, lw=1.3, mec="white", mew=0.8, zorder=4)
                ax.plot([0, 1], [v0["acc"], st[0]["acc"]], ":", color=tint(col, 0.35), lw=1.0, zorder=3)
            ax.plot(0, v0["acc"], marker="*", color=INK2, ms=8, mec="white", mew=0.6, ls="none", zorder=5)
            ax.axhline(50, color=MUTED, lw=0.8, ls=(0, (2.5, 2.5)), zorder=2)
            ax.set_xticks(range(4)); ax.set_xticklabels(["0", "1", "2", "3"], fontsize=8.0)
            ax.set_xlim(-0.45, 3.45); ax.tick_params(axis="x", length=0, pad=3)
            ax.set_xlabel("k   (0 = no occluder)", fontsize=8.0, color=INK)
            ax.set_title(f"{OTITLE[o]}: {VLAB[v]}", fontsize=9.0, color=INK, pad=4)
            if pi == 0:
                ax.set_ylabel("pairwise acc. (%)", fontsize=9.0)
            ax.text(0.5, -0.33, PANEL[3 * ri + pi], transform=ax.transAxes, ha="center", va="top", fontsize=9.5, color=INK)
    h = [Line2D([], [], color=tint(INK2, f) if f else INK2, marker=mk, ms=5.0, lw=1.3, mec="white", mew=0.8, label=t)
         for t, mk, f in TIM]
    h.append(Line2D([], [], color=INK2, marker="*", ms=8, ls="none", mec="white", label="visible"))
    fig.legend(handles=h, loc="upper center", ncol=4, frameon=False, fontsize=7.8, handletextpad=0.4,
               columnspacing=1.8, bbox_to_anchor=(0.5, 1.01))
    fig.subplots_adjust(left=0.075, right=0.995, top=0.925, bottom=0.11, wspace=0.09, hspace=0.62)
    save(fig, "fig_k")
    return js


CTX = [("A", "object $\\to$ object  (impossible: vanishes)", BLUE, 0.00),
       ("B", "empty $\\to$ empty  (impossible: appears)", BLUE, 0.62)]


def fig_permanence_context(P):
    """permanence 의 두 쌍을 따로: A = 물체 문맥 (pos_a vs imp_ab), B = 빈 문맥 (pos_b vs imp_ba). 막대 = 정확도 + CI."""
    W = 0.36
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0), gridspec_kw=dict(width_ratios=[4, 6]))
    js = {}
    for ri, o in enumerate(OBJSETS):
        js[o] = {}
        for ci, (xkey, xvals, xlab, filt) in enumerate([
                ("occ", OCC, OLAB, lambda p: True),
                ("motion", MOT, MLAB, lambda p: p["occ"] == "late")]):
            ax = axes[ri, ci]
            frame(ax, (0, 112), (0, 25, 50, 75, 100))
            if ci == 0:
                ax.axvspan(2.5, 3.5, color=BAND, lw=0, zorder=0)
            js[o][xkey] = {}
            for di, (d, lab, col, f) in enumerate(CTX):
                st = [VT.stat([p for p in P if p["viol"] == "permanence" and p["objset"] == o and filt(p)
                               and p[xkey] == x and p["dir"] == d]) for x in xvals]
                js[o][xkey][d] = dict(zip(xvals, st))
                for xi, s_ in enumerate(st):
                    x = xi + (di - 0.5) * W
                    ax.bar(x, s_["acc"], width=W * 0.9, color=tint(col, f), edgecolor=tint(col, max(0, f - 0.3)) if f else shade(col, 0.25),
                           lw=0.7, zorder=3)
                    ax.errorbar(x, s_["acc"], yerr=[[s_["acc"] - s_["lo"]], [s_["hi"] - s_["acc"]]], fmt="none",
                                ecolor=INK2, elinewidth=0.7, capsize=1.6, capthick=0.7, zorder=4)
                    ax.annotate(f"{s_['acc']:.0f}", (x, s_["hi"] + 1.5), ha="center", va="bottom", fontsize=6.0, color=INK, zorder=5)
            ax.axhline(50, color=MUTED, lw=0.8, ls=(0, (2.5, 2.5)), zorder=2)
            ax.set_xticks(range(len(xvals))); ax.set_xticklabels([xlab[x] for x in xvals], fontsize=8.0 if ci == 0 else 6.8, linespacing=0.95)
            ax.set_xlim(-0.6, len(xvals) - 0.4); ax.tick_params(axis="x", length=0, pad=3)
            ax.set_title(f"{OTITLE[o]}: " + ("by occluder timing" if ci == 0 else "late occlusion, by motion"), fontsize=9.0, color=INK, pad=4)
            if ci == 0:
                ax.set_ylabel("pairwise acc. (%)", fontsize=9.0)
            else:
                ax.set_yticklabels([])
            ax.text(0.5, -0.27, PANEL[2 * ri + ci], transform=ax.transAxes, ha="center", va="top", fontsize=9.5, color=INK)
    h = [Patch(facecolor=tint(col, f), edgecolor=tint(col, max(0, f - 0.3)) if f else shade(col, 0.25), lw=0.7, label=lab)
         for _, lab, col, f in CTX]
    fig.legend(handles=h, loc="upper center", ncol=2, frameon=False, fontsize=7.8, handletextpad=0.5,
               columnspacing=2.2, bbox_to_anchor=(0.5, 1.01))
    fig.subplots_adjust(left=0.075, right=0.995, top=0.915, bottom=0.10, wspace=0.06, hspace=0.55)
    save(fig, "fig_permanence_context")
    return js


def check(js, tab):
    """그린 수치 == tables.json (같은 load · stat, CI 는 RNG 순서가 달라 정확도 · A · B 만)."""
    n = 0
    eq = lambda a, b: all(abs(a[k] - b[k]) < 1e-9 for k in ("acc", "A", "B"))
    for o in OBJSETS:
        for v in VIOL:
            for x in OCC:
                assert eq(js["timing"][o][v][x], tab[o]["viol_x_occ"][v][x]), (o, v, x); n += 1
            for m in MOT:
                assert eq(js["late_motion"][o][v][m], tab[o]["motion_x_viol_late"][m][v]), (o, v, m); n += 1
            for t in ("early", "mid", "late"):
                for k in ("1", "2", "3"):
                    assert eq(js["k"][o][v][t][k], tab[o][f"viol_x_k_{t}"][v][k]), (o, v, t, k); n += 1
    for o in OBJSETS:                      # permanence 두 쌍 == tables 의 A · B 방향
        for x in OCC:
            for d in "AB":
                assert abs(js["perm_ctx"][o]["occ"][d][x]["acc"] - tab[o]["viol_x_occ"]["permanence"][x][d]) < 1e-9, (o, x, d); n += 1
        for m in MOT:
            for d in "AB":
                assert abs(js["perm_ctx"][o]["motion"][d][m]["acc"] - tab[o]["motion_x_viol_late"][m]["permanence"][d]) < 1e-9, (o, m, d); n += 1
    print(f"  tables.json 대조 {n} 칸 일치")


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vjepa2", choices=list(VT.MODELS))
    a = ap.parse_args()
    OUT = VT.ROOT / a.model / "surprise" / "figures"
    P = VT.load(VT.MODELS[a.model][1])
    js = {"timing": fig_timing(P), "k": fig_k(P), "late_motion": fig_late_motion(P), "perm_ctx": fig_permanence_context(P),
          **{f"motion_{oc}": fig_motion(P, oc) for oc in ("visible", "early", "mid")}}
    check(js, json.load(open(VT.ROOT / a.model / "surprise" / "tables.json")))


if __name__ == "__main__":
    main()
