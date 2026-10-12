#!/usr/bin/env python3
"""VOBench_v1 gravity arc — 논문용 "고른 미래" 그림 (2026-10-11, 사용자 예시 판형).

패널 셋 = 문맥 끝 상태 (오르는 중 apex66 · 꼭대기 apex48 · 내려가는 중 apex30). 막대 = 미래 9 개 중 p 가 가장 가까운 비율
(argmin L1, block 부트스트랩 95 % CI, 위 숫자 = %), 점선 = chance 11 %. 막대 아래 = 그 미래의 궤적 (대표 block 하나,
label.json 물체 중심, 모델 입력 샘플 stride 3: 회색 = 문맥, 파랑 = 가능, 주황 = 불가능, 점선 = 문맥 끝).
수치는 vobench_gravity_tables.py 의 tables.json (arc_pick) 그대로 — 다시 세지 않는다. objset 마다 한 장 (primitive · realistic 따로).

  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P z_research/scripts/figures/plot_vobench_gravity_choice.py [--model vjepa2|ariel_ar_ep40]   # → z_research/VOBench_v1/<model>/gravity/figures/fig_gravity_choice{,_ball}_<objset>.{pdf,png}
"""
from __future__ import annotations
import csv, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

REPO = Path(__file__).resolve().parents[3]
_ap = __import__("argparse").ArgumentParser()
_ap.add_argument("--model", default="vjepa2", choices=["vjepa2", "ariel_ar_ep40"])
MODEL = _ap.parse_args().model
D = REPO / "z_research/VOBench_v1" / MODEL / "gravity"
FIG = D / "figures"
INDEX = REPO / "data_csv/vobench_v1_gravity/index.csv"
FR = Path("/data2/local_datasets/world/Benchmark/VOBench_v1/gravity")
C_POS, C_IMP, C_CTX = "#2a78d6", "#eb6834", "0.62"
ARC = ["pos", "imp_rise", "imp_float", "imp_line", "imp_stop", "imp_stop2", "imp_stop4", "imp_stop8", "imp_stop12"]
NAME = ["arc", "rise", "float", "line", "stop", "stop\n+2f", "stop\n+4f", "stop\n+8f", "stop\n+12f"]   # stopK = splice 뒤 K 프레임 진짜 운동 후 멈춤
LET = ["P", "R", "F", "L", "S", "S2", "S4", "S8", "S12"]
CELLS = [("apex66", "Context: rising"), ("apex48", "Context ends at the apex"), ("apex30", "Context: descending")]
CTX = list(range(0, 46, 3)); FUT = list(range(48, 94, 3))
plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman", "DejaVu Serif"], "font.size": 8,
                     "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})


def representative(objset):
    """셀마다 대표 block 하나 — v11 배경, 이름순 첫 block 의 label.json."""
    rows = [r for r in csv.DictReader(INDEX.open()) if r["objset"] == objset and r["law"] == "arc" and r["background"] == "v11"]
    out = {}
    for cell, _ in CELLS:
        r = sorted((x for x in rows if x["cell"] == cell), key=lambda x: x["source_block"])[0]
        L = json.loads((FR / Path(r["file_name"]).parent / "label.json").read_text())["labels"]
        out[cell] = ({k: np.array([[e["u"], e["v"]] for e in L[k]]) for k in L}, r["source_block"])
    return out


def draw(objset, key, tag, ylab):
    js = json.loads((D / "tables.json").read_text())
    rep = representative(objset)
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 3.35), sharey=True)
    x = np.arange(len(ARC))
    for p, (ax, (cell, title)) in enumerate(zip(axs, CELLS)):
        cis = [js["arc_pick"][f"c16t32|{key}|{objset} · {cell}"][a] for a in ARC]
        m = np.array([100 * c[0] for c in cis]); lo = np.array([100 * c[1] for c in cis]); hi = np.array([100 * c[2] for c in cis])
        cols = [C_POS if a == "pos" else C_IMP for a in ARC]
        ax.bar(x, np.maximum(m, 0.4), 0.72, color=cols, alpha=0.88, edgecolor=cols, lw=0.8, zorder=2)
        ax.errorbar(x, m, yerr=[m - lo, hi - m], fmt="none", ecolor="0.2", elinewidth=0.8, capsize=2, zorder=3)
        for i in range(len(ARC)):
            ax.text(x[i], hi[i] + 2.5, f"{m[i]:.0f}", ha="center", va="bottom", fontsize=7)
        ax.axhline(100 / len(ARC), ls="--", lw=0.9, color="0.2", zorder=1)
        ax.set_ylim(0, 112); ax.set_yticks([0, 25, 50, 75, 100]); ax.set_xlim(-0.6, len(ARC) - 0.4)
        ax.grid(axis="y", color="0.88", lw=0.6, zorder=0); ax.set_axisbelow(True)
        ax.set_xticks([]); ax.spines["bottom"].set_color("0.5"); ax.spines["left"].set_color("0.5")
        ax.set_title(title, fontsize=9)
        if p == 0:
            ax.set_ylabel(ylab)
        # 궤적 썸네일 (막대마다)
        U, _ = rep[cell]
        allp = np.concatenate([U[k][CTX + FUT] for k in LET])
        (x0, y0), (x1, y1) = allp.min(0), allp.max(0)
        pad = 0.08 * max(x1 - x0, y1 - y0)
        xe = 0.5 * (U["P"][45, 0] + U["P"][48, 0])
        for i, k in enumerate(LET):
            ins = ax.inset_axes([x[i] - 0.47, -0.30, 0.94, 0.25], transform=ax.get_xaxis_transform())
            ins.scatter(U[k][CTX, 0], U[k][CTX, 1], s=1.2, color=C_CTX, lw=0)
            ins.scatter(U[k][FUT, 0], U[k][FUT, 1], s=1.6, color=cols[i], lw=0)
            if k.startswith("S"):                                                   # 멈춘 자리 강조
                ins.scatter(*U[k][FUT[-1]], s=9, color=cols[i], lw=0)
            ins.axvline(xe, ls=":", lw=0.6, color="0.45")
            ins.set_xlim(x0 - pad, x1 + pad); ins.set_ylim(y1 + pad, y0 - pad)     # 영상 좌표 (v 아래로) → 위아래 뒤집어 그린다
            ins.set_xticks([]); ins.set_yticks([])
            for s_ in ins.spines.values():
                s_.set_color("0.78"); s_.set_lw(0.6)
            ax.text(x[i], -0.33, NAME[i], transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6.5, color=cols[i],
                    fontweight="bold" if ARC[i] == "pos" else "normal", linespacing=0.95)
        ax.text(0.5, -0.50, f"({'abc'[p]})", transform=ax.transAxes, ha="center", va="top", fontsize=9)
    h = [Line2D([], [], ls="", marker="o", ms=4, color=C_CTX, label="context (seen)"),
         Line2D([], [], ls="", marker="o", ms=4, color=C_POS, label="possible future"),
         Line2D([], [], ls="", marker="o", ms=4, color=C_IMP, label="impossible future"),
         Line2D([], [], ls=":", color="0.45", label="end of context"),
         Line2D([], [], ls="--", color="0.2", label="chance (1/9)")]
    fig.legend(handles=h, loc="lower center", ncol=5, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(left=0.10, right=0.995, top=0.92, bottom=0.37, wspace=0.08)
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig_gravity_choice{tag}_{objset}.{ext}", dpi=250)
    plt.close(fig)
    print("→", FIG / f"fig_gravity_choice{tag}_{objset}.pdf", " 대표 block:", {c: b for c, (_, b) in rep.items()})


def main():
    for o in ("primitive", "realistic"):
        draw(o, "pick_full", "", "chosen future (%)\n(lowest L1, all future tokens)")
        draw(o, "pick_ball", "_ball", "chosen future (%)\n(lowest L1, ball tokens only)")


if __name__ == "__main__":
    main()
