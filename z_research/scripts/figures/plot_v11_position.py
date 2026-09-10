#!/usr/bin/env python3
"""v11_full — predictor 미래 8슬롯에서 읽은 물체 위치·속도를 가림 k / 타이밍 축으로.

입력  z_research/v11_roll_out/exp_results/position_regression.json  (`v11_position.py`)
출력  z_research/v11_roll_out/figures/position/

그림 하나 = 주장 하나. 선 = h(파랑, 미래를 본 인코더 = 천장) / p(주황, predictor). 나머지 축은 패널·x.
  fig_pos_by_k        위치 오차(cm)      x = k (0 = 가림 없음, 1~4 = late)   패널 = static / flat / ramp
  fig_vel_by_k        읽은 속도(cm/s)    같은 축, 점선 = 정답 속도
  fig_pos_by_timing   위치 오차(cm)      x = visible / early / mid / late (k 합침)
  fig_vel_by_timing   읽은 속도(cm/s)    같은 축
  fig_slot_err        슬롯별 위치 오차   x = 미래 튜블릿 0..7, 실선 = 가림 없음, 파선 = late k=4

⚠️ static 은 클립 안에서 위치가 안 변해 "클립내부 R²" 가 정의되지 않는다. 그래서 y 는 전부 **절대 오차(cm)** 다.
⚠️ late k 의 미래 첫 k 프레임은 가려져 있다 — 슬롯 0 (k≥1), 슬롯 1 (k≥3) 의 라벨은 화면에 없는 위치다.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
JSON = ROOT / "z_research/v11_roll_out/exp_results/position_regression.json"
OUT = ROOT / "z_research/v11_roll_out/figures/position"     # main() 이 --fit-by 로 하위 폴더를 정한다
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#000000", "#3b3b3b", "#9a9a9a"
SRC = {"h": (BLUE, "h  target encoder (saw the future)"), "p": (ORANGE, "p  predictor (did not)")}
MOT = ["static", "flat", "ramp"]
MLAB = {"static": "Static (v = 0)", "flat": "Moving (flat, 260 cm/s)", "ramp": "Moving (ramp, 116→404 cm/s)"}
TIM = ["visible", "early", "mid", "late"]
TLAB = {"visible": "no\noccluder", "early": "early", "mid": "mid", "late": "late"}
LGF = 8.5
mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "pdf.fonttype": 42, "figure.dpi": 120})


def frame(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_axisbelow(True); ax.grid(axis="y", color=MUTED, alpha=0.30, lw=0.4)
    ax.tick_params(labelsize=7.4, colors=INK2)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in (".pdf", ".png"):
        fig.savefig((OUT / name).with_suffix(ext), dpi=400, bbox_inches="tight", facecolor="white")
    print(f"  [saved] position/{name}"); plt.close(fig)


def panel_labels(fig, axes, pad=0.02):
    fig.canvas.draw(); rend = fig.canvas.get_renderer(); inv = fig.transFigure.inverted()
    for ax, lab in zip(np.ravel(axes), "abcdefg"):
        bb = ax.get_position(); bot = bb.y0
        for t in list(ax.get_xticklabels()) + ([ax.xaxis.label] if ax.get_xlabel() else []):
            try:
                bot = min(bot, t.get_window_extent(rend).transformed(inv).y0)
            except Exception:
                pass
        fig.text(bb.x0 + bb.width / 2, bot - pad, f"({lab})", ha="center", va="top", fontsize=9, color=INK)


def label_pts(ax, xs, ys, col, fmt, dy):
    for x, y in zip(xs, ys):
        if np.isfinite(y):
            ax.text(x, y + dy, fmt % y, ha="center", va="bottom", fontsize=6.4, color=col, zorder=6)


def cell(R, s, mo, tm, k):
    return R[s]["by_cell"].get(f"{mo}|{tm}|{k}")


def pooled(R, s, mo, tm, field):
    """k 를 합친 값 — n 가중 평균 (라벨 없는 셀은 건너뛴다)."""
    cs = [cell(R, s, mo, tm, k) for k in range(5)]
    cs = [c for c in cs if c]
    n = sum(c["n"] for c in cs)
    return sum(c[field] * c["n"] for c in cs) / n if n else np.nan


def fig_by_k(R, field, ylab, name, true_field=None, fmt="%.0f"):
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.1), sharey=(field != "b_hat_cm_s"))
    for ax, mo in zip(axes, MOT):
        frame(ax)
        ks = list(range(5))
        for s in ("h", "p"):
            ys = [(cell(R, s, mo, "visible", 0) if k == 0 else cell(R, s, mo, "late", k)) for k in ks]
            ys = [c[field] if c else np.nan for c in ys]
            ax.plot(ks, ys, marker="o", ms=5, lw=1.8, color=SRC[s][0], mec="white", mew=0.7,
                    label=SRC[s][1] if mo == MOT[0] else None, zorder=4)
        if true_field:
            c0 = cell(R, "h", mo, "visible", 0)
            ax.axhline(c0[true_field], ls="--", color=INK2, lw=0.9, zorder=2,
                       label="true (label)" if mo == MOT[0] else None)
        ax.set_xticks(ks); ax.set_xticklabels(["0"] + [str(k) for k in ks[1:]], fontsize=7.4)
        ax.set_xlabel("k   (0 = no occluder, 1–4 = occluded late)", fontsize=8, color=INK)
        ax.set_title(MLAB[mo], fontsize=8.8, color=INK, pad=4)
    # 라벨은 축 범위가 정해진 뒤에
    for ax, mo in zip(axes, MOT):
        lo, hi = ax.get_ylim(); dy = (hi - lo) * 0.03
        for s in ("h", "p"):
            ys = [(cell(R, s, mo, "visible", 0) if k == 0 else cell(R, s, mo, "late", k)) for k in range(5)]
            label_pts(ax, range(5), [c[field] if c else np.nan for c in ys], SRC[s][0], fmt, dy)
        ax.set_ylim(lo, hi + (hi - lo) * 0.12)
    axes[0].set_ylabel(ylab, fontsize=8.4, color=INK)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=LGF, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0), handlelength=1.4)
    fig.subplots_adjust(left=0.085, right=0.995, top=0.80, bottom=0.25, wspace=0.22)
    panel_labels(fig, axes); save(fig, name)


def fig_by_timing(R, field, ylab, name, true_field=None, fmt="%.0f"):
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.1), sharey=(field != "b_hat_cm_s"))
    for ax, mo in zip(axes, MOT):
        frame(ax)
        for s in ("h", "p"):
            ys = [pooled(R, s, mo, tm, field) for tm in TIM]
            ax.plot(range(4), ys, marker="o", ms=5, lw=1.8, color=SRC[s][0], mec="white", mew=0.7,
                    label=SRC[s][1] if mo == MOT[0] else None, zorder=4)
        if true_field:
            ax.axhline(cell(R, "h", mo, "visible", 0)[true_field], ls="--", color=INK2, lw=0.9, zorder=2,
                       label="true (label)" if mo == MOT[0] else None)
        ax.set_xticks(range(4)); ax.set_xticklabels([TLAB[t] for t in TIM], fontsize=7.4)
        ax.set_xlabel("occlusion timing  (k pooled)", fontsize=8, color=INK)
        ax.set_title(MLAB[mo], fontsize=8.8, color=INK, pad=4)
    for ax, mo in zip(axes, MOT):
        lo, hi = ax.get_ylim(); dy = (hi - lo) * 0.03
        for s in ("h", "p"):
            label_pts(ax, range(4), [pooled(R, s, mo, tm, field) for tm in TIM], SRC[s][0], fmt, dy)
        ax.set_ylim(lo, hi + (hi - lo) * 0.12)
    axes[0].set_ylabel(ylab, fontsize=8.4, color=INK)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=LGF, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0), handlelength=1.4)
    fig.subplots_adjust(left=0.085, right=0.995, top=0.80, bottom=0.25, wspace=0.22)
    panel_labels(fig, axes); save(fig, name)


def fig_slot(R, name="fig_slot_err"):
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.1), sharey=True)
    for ax, mo in zip(axes, MOT):
        frame(ax)
        for s in ("h", "p"):
            for tm, k, ls, tag in (("visible", 0, "-", "no occluder"), ("late", 4, "--", "occluded late, k=4")):
                c = cell(R, s, mo, tm, k)
                if not c:
                    continue
                ax.plot(range(8), c["mae_cm_by_slot"], ls=ls, marker="o", ms=3.6, lw=1.6, color=SRC[s][0],
                        mec="white", mew=0.6, label=f"{s}  {tag}" if mo == MOT[0] else None, zorder=4)
        ax.axvspan(-0.5, 1.5, color=MUTED, alpha=0.10, lw=0)
        ax.text(0.5, 0.97, "hidden at k=4", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6.4, color=INK2)
        ax.set_xticks(range(8)); ax.set_xlabel("future tubelet", fontsize=8, color=INK)
        ax.set_title(MLAB[mo], fontsize=8.8, color=INK, pad=4)
    axes[0].set_ylabel("position error  (cm)", fontsize=8.4, color=INK)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=LGF, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.0), handlelength=1.8)
    fig.subplots_adjust(left=0.085, right=0.995, top=0.76, bottom=0.25, wspace=0.12)
    panel_labels(fig, axes); save(fig, name)


def fig_summary(R, name="fig_summary"):
    """한 장 요약 — 위 줄: 위치 오차, 아래 줄: 읽은 속도. x = 타이밍(k 합침), 패널 = 운동, 선 = h / p (self readout)."""
    fig, axes = plt.subplots(2, 3, figsize=(7.6, 5.0), sharex=True)
    for j, mo in enumerate(MOT):
        for i, (field, ylab, tf) in enumerate((("mae_cm", "position error  (cm)", None),
                                              ("b_hat_cm_s", "read-out velocity  (cm/s)", "b_true_cm_s"))):
            ax = axes[i, j]; frame(ax)
            for s_ in ("h", "p"):
                ys = [pooled(R, s_, mo, tm, field) for tm in TIM]
                ax.plot(range(4), ys, marker="o", ms=5, lw=1.8, color=SRC[s_][0], mec="white", mew=0.7,
                        label=SRC[s_][1] if (i == 0 and j == 0) else None, zorder=4)
            if tf:
                ax.axhline(cell(R, "h", mo, "visible", 0)[tf], ls="--", color=INK2, lw=0.9, zorder=2,
                           label="true (label)" if j == 0 else None)
            lo, hi = ax.get_ylim(); dy = (hi - lo) * 0.03
            for s_ in ("h", "p"):
                label_pts(ax, range(4), [pooled(R, s_, mo, tm, field) for tm in TIM], SRC[s_][0], "%.0f", dy)
            ax.set_ylim(lo, hi + (hi - lo) * 0.14)
            if i == 0:
                ax.set_title(MLAB[mo], fontsize=8.8, color=INK, pad=4)
            if j == 0:
                ax.set_ylabel(ylab, fontsize=8.4, color=INK)
            if i == 1:
                ax.set_xticks(range(4)); ax.set_xticklabels([TLAB[t] for t in TIM], fontsize=7.4)
                ax.set_xlabel("occlusion timing  (k pooled)", fontsize=8, color=INK)
    if MOT[0] == "static":
        axes[1, 0].set_ylim(-8, 8)
    h, l = axes[0, 0].get_legend_handles_labels(); h2, l2 = axes[1, 0].get_legend_handles_labels()
    fig.legend(h + h2, l + l2, frameon=False, fontsize=LGF, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0), handlelength=1.4)
    fig.subplots_adjust(left=0.09, right=0.995, top=0.88, bottom=0.14, wspace=0.24, hspace=0.12)
    panel_labels(fig, axes[1], pad=0.02); save(fig, name)


COND = [("static", "visible"), ("static", "early"), ("static", "mid"), ("static", "late"),
        ("flat", "visible"), ("flat", "early"), ("flat", "mid"), ("flat", "late"),
        ("ramp", "visible"), ("ramp", "early"), ("ramp", "mid"), ("ramp", "late")]
CLAB = {"visible": "vis", "early": "early", "mid": "mid", "late": "late"}


def fig_performance(R, name="fig_performance"):
    """잘 하나 못 하나 — 조건 12개 막대, 축은 위아래 하나씩.
    (a) 위치 R². static 은 클립 안에서 위치가 안 변해 클립내부 R² 가 없다 → 클립 간 R² (위치 5개), 빗금.
    (b) 속도 오차 = |읽은 속도 − 정답| (cm/s). static 정답 0 / flat 260 / ramp 332 (창 평균)."""
    fig, axes = plt.subplots(2, 1, figsize=(7.6, 5.2), sharex=True)
    x = np.arange(len(COND)); wdt = 0.36
    ax = axes[0]; frame(ax)
    for j, s_ in enumerate(("h", "p")):
        vals, hatch = [], []
        for mo, tm in COND:
            v = pooled(R, s_, mo, tm, "r2_pos_within_clip") if mo != "static" else pooled(R, s_, mo, tm, "r2_pos_across_clip")
            vals.append(100 * v); hatch.append("//" if mo == "static" else "")
        xs = x + (j - 0.5) * wdt
        bars = ax.bar(xs, vals, wdt, color=SRC[s_][0], alpha=0.85, label=SRC[s_][1], zorder=4)
        for b_, hch in zip(bars, hatch):
            b_.set_hatch(hch)
        for xx, v in zip(xs, vals):
            ax.text(xx, v + 0.8, f"{v:.0f}", ha="center", va="bottom", fontsize=6.2, color=SRC[s_][0])
    ax.set_ylim(80, 104); ax.set_yticks((80, 90, 100)); ax.set_ylabel("position R²  (%)", fontsize=8.4, color=INK)
    ax.axhline(100, color=INK2, lw=0.8, ls="--", zorder=2)
    for j, mo in enumerate(MOT):
        ax.text(4 * j + 1.5, 103.2, MLAB[mo], ha="center", va="bottom", fontsize=8.4, color=INK)
    ax = axes[1]; frame(ax)
    for j, s_ in enumerate(("h", "p")):
        vals = [abs(pooled(R, s_, mo, tm, "b_hat_cm_s") - pooled(R, s_, mo, tm, "b_true_cm_s")) for mo, tm in COND]
        xs = x + (j - 0.5) * wdt
        ax.bar(xs, vals, wdt, color=SRC[s_][0], alpha=0.85, zorder=4)
        for xx, v in zip(xs, vals):
            ax.text(xx, v + 0.4, f"{v:.1f}" if v < 10 else f"{v:.0f}", ha="center", va="bottom", fontsize=6.2, color=SRC[s_][0])
    ax.set_ylim(0, 24); ax.set_yticks((0, 5, 10, 15, 20))
    ax.set_ylabel("velocity error  |read-out − true|  (cm/s)", fontsize=8.4, color=INK)
    for j, (mo, tv) in enumerate((("static", "true 0"), ("flat", "true 260"), ("ramp", "true 332 (mean over window)"))):
        ax.text(4 * j + 1.5, 22.3, tv, ha="center", va="top", fontsize=7.2, color=INK2)
    for a_ in axes:
        for b0 in (3.5, 7.5):
            a_.axvline(b0, color=MUTED, lw=0.6)
    axes[1].set_xticks(x); axes[1].set_xticklabels([CLAB[t] for _, t in COND], fontsize=7.4)
    axes[1].set_xlabel("condition   (early / mid / late = occluded, k pooled)", fontsize=8, color=INK)
    h, l = axes[0].get_legend_handles_labels()
    import matplotlib.patches as mp
    h.append(mp.Patch(facecolor="white", edgecolor=INK2, hatch="//")); l.append("static: R² across clips (5 positions)")
    fig.legend(h, l, frameon=False, fontsize=LGF, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0), handlelength=1.4)
    fig.subplots_adjust(left=0.09, right=0.995, top=0.86, bottom=0.12, hspace=0.12)
    panel_labels(fig, axes, pad=0.012); save(fig, name)


def main():
    global OUT
    ap = argparse.ArgumentParser(); ap.add_argument("--json", type=Path, default=JSON)
    ap.add_argument("--fit-by", default="condition", choices=["condition", "motion", "all"],
                    help="readout 을 어느 단위로 학습한 결과를 그릴까. condition 이 주, all 은 time-index 배제 대조군")
    a = ap.parse_args()
    J = json.loads(a.json.read_text())
    R = J["results"][a.fit_by]
    if a.fit_by != "condition":
        OUT = OUT / f"fit_{a.fit_by}"
    print(f"  fit_by = {a.fit_by}  ->  {OUT}")
    fig_performance(R)
    fig_summary(R)
    fig_by_k(R, "mae_cm", "position error  (cm, mean over 8 tubelets)", "fig_pos_by_k")
    fig_by_k(R, "b_hat_cm_s", "read-out velocity  (cm/s)", "fig_vel_by_k", true_field="b_true_cm_s")
    fig_by_timing(R, "mae_cm", "position error  (cm, mean over 8 tubelets)", "fig_pos_by_timing")
    fig_by_timing(R, "b_hat_cm_s", "read-out velocity  (cm/s)", "fig_vel_by_timing", true_field="b_true_cm_s")
    fig_slot(R)


if __name__ == "__main__":
    main()
