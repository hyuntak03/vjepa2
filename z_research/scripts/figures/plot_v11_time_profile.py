#!/usr/bin/env python3
"""예측 구간의 **시간칸(tubelet) 별** 채점 신호 프로파일.

입력은 `token_time_profile.py` 가 만든 npz (전수 21,504 pair).
토큰마다 d = |p−h(imp)| − |p−h(pos)|. 채점은 `mean_token d > 0` 이므로
시간칸마다 그 내역을 보면 **언제 벌고 언제 까먹는지**가 나온다.

시간칸 t_j = 미래 프레임 raw 48+6j, 51+6j (tubelet 2장씩, 8칸).

색 규칙은 `plot_v11_timing.py` 와 같다 — **한 그림에 범주형 척도는 하나만**.
timing 은 4단계라 `vis` 만 중립 회색을 쓴다 (팔레트가 3색뿐).

  python z_research/scripts/figures/plot_v11_time_profile.py \
    --npz z_research/IntPhysGenV11_occlusion_timing_ablation/exp_results/token_time_profile.npz \
    --outdir z_research/IntPhysGenV11_occlusion_timing_ablation/figures/05_time_profile
"""
from __future__ import annotations
import argparse, colorsys
from pathlib import Path
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, BAND = "#000000", "#3b3b3b", "#9a9a9a", "#f2f2f2"
PANEL = [f"({c})" for c in "abcdefghi"]
VIOL = ["vanish", "shape", "color"]
VLAB = {"vanish": "Object permanence", "shape": "Shape consistency", "color": "Colour consistency"}
MOT = ["static", "flat", "ramp"]
MLAB = {"static": "Static", "flat": "Moving (flat)", "ramp": "Moving (ramp)"}
TIM = ["vis", "early", "mid", "late"]
TLAB = {"vis": "no occluder", "early": "early", "mid": "mid", "late": "late"}
TCOL = {"vis": INK2, "early": BLUE, "mid": GREEN, "late": ORANGE}
T = [f"t{j+1}" for j in range(8)]
mpl.rcParams.update({"font.family": "serif",
                     "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "xtick.major.size": 2.2,
                     "ytick.major.size": 2.2, "pdf.fonttype": 42, "ps.fonttype": 42})
LGF = 8.5


def _pastel(h, lift=0.26, sat=1.06):
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (1, 3, 5))
    hh, ll, ss = colorsys.rgb_to_hls(r, g, b)
    return "#%02x%02x%02x" % tuple(
        round(255 * c) for c in colorsys.hls_to_rgb(hh, ll + (1 - ll) * lift, min(1.0, ss * sat)))


def nice_ylim(vals, pad=0.16):
    """데이터에 맞춰 y 범위를 잡는다. 고정값으로 두면 어떤 패널은 위쪽이 텅 빈다."""
    v = np.concatenate([np.asarray(x).ravel() for x in vals if len(np.asarray(x))])
    lo, hi = float(v.min()), float(v.max())
    if hi <= 0: hi = 0.0
    if lo >= 0: lo = 0.0
    r = max(hi - lo, 1e-9)
    return lo - r * pad, hi + r * (pad + 0.10)


def legend(ax, items, ncol, anchor):
    """범례를 **명시 핸들**로 만든다. 데이터가 없는 패널에서 label 을 못 달아
    항목이 통째로 빠지는 일을 막는다 (vanish/vis 에는 wrong 이 0 개다)."""
    h = [plt.Line2D([], [], marker="o", ms=5.2, lw=1.8, color=_pastel(c), mec="white",
                    mew=0.7, label=l) for l, c in items]
    ax.legend(handles=h, frameon=False, fontsize=LGF, ncol=ncol, loc="lower center",
              bbox_to_anchor=anchor, handlelength=1.3, columnspacing=1.5)


def frame(ax, ylim, zero=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_axisbelow(True); ax.grid(axis="y", color=MUTED, alpha=0.30, lw=0.4)
    if zero:
        ax.axhline(0, color=INK2, lw=0.8, ls="--", zorder=2)
    ax.set_ylim(*ylim); ax.tick_params(labelsize=7.2, colors=INK2)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)


def panel_labels(fig, axes, pad=0.018):
    fig.canvas.draw(); rend = fig.canvas.get_renderer(); inv = fig.transFigure.inverted()
    for ax, lab in zip(np.ravel(axes), PANEL):
        bb = ax.get_position(); bot = bb.y0
        parts = list(ax.get_xticklabels()) + ([ax.xaxis.label] if ax.get_xlabel() else [])
        for t in parts:
            try:
                bot = min(bot, t.get_window_extent(rend).transformed(inv).y0)
            except Exception:
                pass
        fig.text(bb.x0 + bb.width / 2, bot - pad, lab, ha="center", va="top",
                 fontsize=9.0, color=INK)


def save(fig, out: Path, name):
    out.mkdir(parents=True, exist_ok=True)
    for ext in (".pdf", ".png"):
        fig.savefig((out / name).with_suffix(ext), dpi=400, bbox_inches="tight",
                    facecolor="white")
    print(f"  [saved] {name}")
    plt.close(fig)


def load(npz):
    Z = np.load(npz, allow_pickle=True)
    K = Z["keys"]
    return dict(viol=K[:, 0], mot=K[:, 1], tim=K[:, 2], k=K[:, 3], ok=K[:, 4],
                net=Z["net"], pos=Z["pos"], neg=Z["neg"], fp=Z["fpos"])


def line(ax, y, c, lab=None):
    ax.plot(range(8), y, marker="o", ms=4.6, lw=1.8, color=_pastel(c), mec="white",
            mew=0.7, label=lab, zorder=4)


# ── 1. net(t) × 타이밍  (정답 / 오답) ──────────────────────────────────────────
def fig_net(D, out):
    """주장 — 신호는 **t1 에 몰려 있고** 시간이 갈수록 사라진다. late 는 t1 마저 죽는다.
    정답·오답을 행으로 갈라 본다 (오답은 거의 모든 칸이 음수다)."""
    fig, axes = plt.subplots(2, 3, figsize=(7.6, 4.7), sharex=True, squeeze=False)
    for ri, o in enumerate(("correct", "wrong")):
        for ci, v in enumerate(VIOL):
            ax = axes[ri][ci]; ys = []
            for t in TIM:
                m = (D["viol"] == v) & (D["tim"] == t) & (D["ok"] == o)
                if m.sum() == 0:
                    continue
                y = D["net"][m].mean(0) * 1000; ys.append(y)
                line(ax, y, TCOL[t])
            if not ys:
                ax.text(0.5, 0.5, "no wrong cases\n(100% correct)", ha="center", va="center",
                        transform=ax.transAxes, fontsize=7.6, color=MUTED, style="italic")
                ys = [np.zeros(8)]
            frame(ax, nice_ylim(ys))
            ax.set_xticks(range(8)); ax.set_xticklabels(T, fontsize=7.0)
            if ri == 0:
                ax.set_title(VLAB[v], fontsize=9.2, color=INK, pad=4)
            if ri == 1:
                ax.set_xlabel("tubelet  (t1 = raw 48–51)", fontsize=8.0, color=INK)
            if ci == 0:
                ax.set_ylabel(f"{o}\nnet signal  mean$_{{token}}$ d  (×1000)",
                              fontsize=8.2, color=BLUE if o == "correct" else ORANGE)
    legend(axes[0][0], [(TLAB[t], TCOL[t]) for t in TIM], 4, (1.70, 1.11))
    fig.subplots_adjust(left=0.105, right=0.995, top=0.885, bottom=0.115, wspace=0.30,
                        hspace=0.16)
    save(fig, out, "fig_net_by_timing")


# ── 2. correct vs wrong 직접 비교 ──────────────────────────────────────────────
def fig_ok(D, out):
    """주장 — **t1 은 틀린 쌍에서도 대체로 맞다.** 실패는 t2 이후에서 쌓인다."""
    rows = ["vis", "late"]
    fig, axes = plt.subplots(2, 3, figsize=(7.6, 4.7), sharex=True, squeeze=False)
    for ri, t in enumerate(rows):
        for ci, v in enumerate(VIOL):
            ax = axes[ri][ci]; ys = []
            for o, c in (("correct", BLUE), ("wrong", ORANGE)):
                m = (D["viol"] == v) & (D["tim"] == t) & (D["ok"] == o)
                if m.sum() == 0:
                    continue
                y = D["net"][m].mean(0) * 1000; ys.append(y); line(ax, y, c)
                ax.annotate(f"n={m.sum()}", (7, y[7]), textcoords="offset points",
                            xytext=(4, 0), ha="left", va="center", fontsize=6.0,
                            color=c, annotation_clip=False)
            frame(ax, nice_ylim(ys))
            ax.set_xticks(range(8)); ax.set_xticklabels(T, fontsize=7.0)
            if ri == 0:
                ax.set_title(VLAB[v], fontsize=9.2, color=INK, pad=4)
            if ri == 1:
                ax.set_xlabel("tubelet", fontsize=8.0, color=INK)
            if ci == 0:
                ax.set_ylabel(f"{TLAB[t]}\nnet signal (×1000)", fontsize=8.2, color=INK)
    legend(axes[0][0], [("correct", BLUE), ("wrong", ORANGE)], 2, (1.70, 1.11))
    fig.subplots_adjust(left=0.105, right=0.975, top=0.885, bottom=0.115, wspace=0.32,
                        hspace=0.16)
    save(fig, out, "fig_correct_vs_wrong")


# ── 3. + / − 상쇄  (정답 / 오답) ───────────────────────────────────────────────
def fig_posneg(D, out):
    """주장 — 버는 쪽과 까먹는 쪽이 **거의 같은 크기**다. net 은 그 잔여물이다.
    ⚠️ 타이밍은 접었다 (4단계 전부 합산)."""
    fig, axes = plt.subplots(2, 3, figsize=(7.6, 4.7), sharex=True, squeeze=False)
    for ri, o in enumerate(("correct", "wrong")):
        for ci, v in enumerate(VIOL):
            ax = axes[ri][ci]
            m = (D["viol"] == v) & (D["ok"] == o)
            if m.sum() == 0:
                ax.text(0.5, 0.5, "no wrong cases", ha="center", va="center",
                        transform=ax.transAxes, fontsize=7.6, color=MUTED, style="italic")
                frame(ax, (0, 1), zero=False); ax.set_xticks(range(8))
                ax.set_xticklabels(T, fontsize=7.0); continue
            P, N = D["pos"][m].mean(0) * 1000, D["neg"][m].mean(0) * 1000
            line(ax, P, BLUE); line(ax, N, ORANGE)
            frame(ax, nice_ylim([P, N]), zero=False)
            ax.annotate(f"|net| = {abs(D['net'][m].mean())*1000:.2f}   (n={m.sum()})",
                        (0.03, 0.06), xycoords="axes fraction", fontsize=6.6, color=INK2)
            ax.set_xticks(range(8)); ax.set_xticklabels(T, fontsize=7.0)
            if ri == 0:
                ax.set_title(VLAB[v], fontsize=9.2, color=INK, pad=4)
            if ri == 1:
                ax.set_xlabel("tubelet", fontsize=8.0, color=INK)
            if ci == 0:
                ax.set_ylabel(f"{o}\nmean |d| per side (×1000)", fontsize=8.2,
                              color=BLUE if o == "correct" else ORANGE)
    legend(axes[0][0], [("mean of  d > 0", BLUE), ("mean of  |d|  where d < 0", ORANGE)],
           2, (1.70, 1.11))
    fig.subplots_adjust(left=0.105, right=0.995, top=0.885, bottom=0.115, wspace=0.30,
                        hspace=0.16)
    save(fig, out, "fig_pos_vs_neg")


# ── 4. 운동별  (정답 / 오답) ───────────────────────────────────────────────────
def fig_motion(D, out, viol="vanish"):
    """주장 — **정지는 평평하고 이동은 t1 에 몰린다.** 신호의 시간 구조가 운동에 달렸다."""
    fig, axes = plt.subplots(2, 3, figsize=(7.6, 4.7), sharex=True, squeeze=False)
    for ri, o in enumerate(("correct", "wrong")):
        for ci, m_ in enumerate(MOT):
            ax = axes[ri][ci]; ys = []
            for t in TIM:
                m = (D["viol"] == viol) & (D["tim"] == t) & (D["mot"] == m_) & (D["ok"] == o)
                if m.sum() == 0:
                    continue
                y = D["net"][m].mean(0) * 1000; ys.append(y); line(ax, y, TCOL[t])
            if not ys:
                ax.text(0.5, 0.5, "no wrong cases", ha="center", va="center",
                        transform=ax.transAxes, fontsize=7.6, color=MUTED, style="italic")
                ys = [np.zeros(8)]
            frame(ax, nice_ylim(ys))
            ax.set_xticks(range(8)); ax.set_xticklabels(T, fontsize=7.0)
            if ri == 0:
                ax.set_title(MLAB[m_], fontsize=9.2, color=INK, pad=4)
            if ri == 1:
                ax.set_xlabel("tubelet", fontsize=8.0, color=INK)
            if ci == 0:
                ax.set_ylabel(f"{o}\n{VLAB[viol]}  net (×1000)", fontsize=8.2,
                              color=BLUE if o == "correct" else ORANGE)
    legend(axes[0][0], [(TLAB[t], TCOL[t]) for t in TIM], 4, (1.70, 1.11))
    fig.subplots_adjust(left=0.105, right=0.995, top=0.885, bottom=0.115, wspace=0.30,
                        hspace=0.16)
    save(fig, out, f"fig_motion_{viol}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    a = ap.parse_args()
    D = load(a.npz)
    print(f"  pair {len(D['viol'])}")
    fig_net(D, a.outdir)
    fig_ok(D, a.outdir)
    fig_posneg(D, a.outdir)
    for v in ("vanish", "shape"):
        fig_motion(D, a.outdir, v)


if __name__ == "__main__":
    main()
