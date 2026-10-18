#!/usr/bin/env python3
"""anchor 이식 — 프레임별 그림 (2026-10-13, 사용자 피드백 "잔상 그림도 어렵다 → 프레임별로").

한 예시 clip, 같은 화면 영역 (고정 crop) 을 칸마다 보여 준다.
  열 1–4 = 입력 프레임 10 · 12 · 14 · 15 (14 · 15 = 마지막 문맥 블록 = 조작 대상, 빨간 테두리. 'block 6 only' 행은 12 · 13 이 조작 대상)
  열 5–8 = 미래 블록 k = 2 · 4 · 6 · 8 — predictor 는 픽셀을 그리지 않으므로 배경 위에
           흰 점선 원 = 그 시점 공의 진짜 자리, 색 점 = 세 predictor 의 미래를 위치 자로 읽은 자리 (자가 '없음' 이라 하면 안 찍음)
  모든 칸의 흰 세로선 = 마지막 문맥 블록에서 공의 원래 자리 (기준선)
  오른쪽 막대 = 165 clip 전체: k4 · k8 에 공이 원래 방향으로 몇 칸 갔나 (중앙값; 회색 = 진짜)
입력 = predictor_anchor_test.py 산출물. 라벨 영어.

  python z_research/scripts/figures/plot_anchor_frames.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "z_research/scripts/analysis")); sys.path.insert(0, str(REPO / "z_research/scripts/figures"))
import predictor_anchor_test as AT  # noqa: E402
import predictor_bottleneck as PB  # noqa: E402
from plot_anchor_transplant import A, FIG, MODELS, load  # noqa: E402

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "pdf.fonttype": 42, "axes.spines.top": True, "axes.spines.right": True})
ROWS = [("true", "original"), ("stay", "last block:\nstop"), ("back", "last block:\nreverse"), ("erase", "last block:\nobject removed"),
        ("back6", "block 6 only:\nreverse")]
C, K, CELL, SC = 8, 8, 18.0, 256 / 288
IN_FR = (10, 12, 14, 15)
FUT_K = (2, 4, 6, 8)


def main():
    D = {m: load(m) for m, _, _ in MODELS}
    rows = PB.select(1600, 1200)[1]
    R0 = D["ar40"]["row"]
    T = np.stack([PB.truth(rows, int(i))[0] for i in R0])
    v = T[:, C - 1] - T[:, C - 2]; vh = v / np.linalg.norm(v, axis=1)[:, None]
    along = lambda xy, ref: ((xy - ref[:, None]) * vh[:, None]).sum(-1) / CELL
    S = {m: {c: along(D[m][f"{c}_xy"], T[:, C - 1]) for c, _ in ROWS} for m in D}
    PN = {m: {c: D[m][f"{c}_pn"] for c, _ in ROWS} for m in D}
    fut = along(T[:, C:], T[:, C - 1])
    keep = (vh[:, 0] > 0.95) & np.all([PN[m][c][:, :4].max(1) < 0.5 for m in D for c in ("true", "stay", "back")], axis=0)
    cand = np.nonzero(keep)[0]
    dev = [sum(np.abs(S[m][c][i] - np.median(S[m][c], 0)).mean() for m in D for c in ("true", "stay", "back")) for i in cand]
    ex = int(cand[int(np.argmin(dev))]); r = rows[int(R0[ex])]
    print("example clip", r["video_id"])

    from importlib.util import module_from_spec, spec_from_file_location
    from evals.world_model_analysis.data import MEAN, STD, WMADataset
    spec = spec_from_file_location("ir", REPO / "z_research/scripts/analysis/rollout2_identity_readout.py")
    ir = module_from_spec(spec); spec.loader.exec_module(ir)
    ds = WMADataset(dict(data=ir.rp.DATA, model=dict(ir.rp.MODEL), features={"cache_dir": "/tmp"}, surprise={}))
    x0 = ds.clip(int(R0[ex]))
    to_img = lambda x: (x * STD + MEAN).clamp(0, 1).permute(1, 2, 3, 0).numpy()
    bg = np.median(to_img(x0), 0)
    Tb = T[ex] * SC
    rad = float(r["obj_apparent_px"]) / 2 * SC
    xa, xb = int(max(Tb[C - 3, 0] - 2.2 * rad, 0)), int(min(Tb[-1, 0] + 2.2 * rad, 256))
    yc = Tb[:, 1].mean(); ya, yb = int(max(yc - 2.2 * rad, 0)), int(min(yc + 1.8 * rad, 256))
    x_last = Tb[C - 1, 0]

    ncol = len(IN_FR) + len(FUT_K)
    fig = plt.figure(figsize=(7.2, 0.62 * len(ROWS) + 0.9))
    gs = fig.add_gridspec(len(ROWS), ncol + 2, width_ratios=[1] * ncol + [0.12, 0.9], wspace=0.05, hspace=0.12,
                          left=0.11, right=0.99, top=0.80, bottom=0.12)
    for ri, (c, lab) in enumerate(ROWS):
        fr = to_img(AT.manipulate(x0, r, c))
        manip = (12, 13) if c == "back6" else (14, 15)
        for j, f in enumerate(IN_FR):
            ax = fig.add_subplot(gs[ri, j]); ax.imshow(fr[f], extent=(0, 256, 256, 0)); ax.set_xlim(xa, xb); ax.set_ylim(yb, ya)
            ax.axvline(x_last, color="w", lw=0.7, ls=":")
            red = c != "true" and f in manip
            for sp in ax.spines.values(): sp.set_edgecolor("#d62728" if red else "0.75"); sp.set_linewidth(1.8 if red else 0.5)
            ax.set_xticks([]); ax.set_yticks([])
            if ri == 0: ax.set_title(f"input\nframe {f}", fontsize=6.5)
            if j == 0: ax.set_ylabel(lab, fontsize=6.5, rotation=0, ha="right", va="center", labelpad=4)
        for j, k in enumerate(FUT_K):
            ax = fig.add_subplot(gs[ri, len(IN_FR) + j]); ax.imshow(0.6 * bg + 0.4, extent=(0, 256, 256, 0)); ax.set_xlim(xa, xb); ax.set_ylim(yb, ya)
            ax.axvline(x_last, color="w", lw=0.7, ls=":")
            ax.add_patch(Circle(Tb[C - 1 + k], rad, fill=False, ec="w", lw=1.0, ls="--"))
            for mi, (m, _, col) in enumerate(MODELS):
                if PN[m][c][ex, k - 1] < 0.5:
                    xy = D[m][f"{c}_xy"][ex, k - 1] * SC
                    ax.plot(xy[0], yc + (mi - 1) * 0.55 * rad, "o", color=col, ms=4.2, mec="w", mew=0.4)
            for sp in ax.spines.values(): sp.set_edgecolor("0.75"); sp.set_linewidth(0.5)
            ax.set_xticks([]); ax.set_yticks([])
            if ri == 0: ax.set_title(f"future\nblock k={k}", fontsize=6.5)
        # 전체 막대
        ax = fig.add_subplot(gs[ri, ncol + 1])
        xs = np.arange(2); w = 0.2
        ax.bar(xs - 1.5 * w, [np.median(fut[:, 3]), np.median(fut[:, 7])], w, color="0.6")
        for mi, (m, _, col) in enumerate(MODELS):
            ax.bar(xs + (mi - 0.5) * w, [np.median(S[m][c][:, 3]), np.median(S[m][c][:, 7])], w, color=col)
        ax.axhline(0, color="0.3", lw=0.6); ax.set_ylim(-3, 5); ax.set_xticks(xs); ax.set_xticklabels(["k=4", "k=8"] if ri == len(ROWS) - 1 else ["", ""])
        ax.tick_params(labelsize=6); ax.spines[["top", "right"]].set_visible(False)
        if ri == 0: ax.set_title(f"{len(R0)} clips, median:\ncells moved", fontsize=6.5)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    h = [Line2D([], [], color="w", marker="o", mfc="none", mec="0.3", ls="--", ms=7, label="true object position")]
    h += [Line2D([], [], color=col, marker="o", ls="", ms=5, label=f"{mlab}: decoded position") for _, mlab, col in MODELS]
    h += [Line2D([], [], color="0.5", ls=":", label="object position at last context block"), Patch(color="0.6", label="bars: true"),
          Patch(fc="none", ec="#d62728", lw=1.5, label="manipulated frames")]
    fig.legend(handles=h, loc="upper center", ncol=3, frameon=False, fontsize=6, bbox_to_anchor=(0.5, 1.0))
    out = FIG / "fig_anchor_frames"
    fig.savefig(f"{out}.png", dpi=230); fig.savefig(f"{out}.pdf"); print("saved", out)


if __name__ == "__main__":
    main()
