#!/usr/bin/env python3
"""VOBench_v1 gravity 그림 2 장 (2026-10-10). 입력 = vobench_gravity_tables.py 의 tables.json (수치를 다시 세지 않는다).

  fig_arc_choice   표준 창 c16t32: 미래 9 개 중 p 가 가장 가까운 미래 비율 — 모델 (full) · 모델 (공 칸만) · 복사. 행 = objset, 열 = 셀 (오르는 중 → 꼭대기 → 내려가는 중)
  fig_arc_sliding  sliding (stride 2): x = 문맥 끝 − 꼭대기 (프레임), y = 고른 멈춤 후보의 D (= 멈춘 프레임 − 문맥 끝).
                   회색 가로 눈금 = 그 창에 있는 멈춤 후보의 D, 점 크기 = 고른 block 수. 열 = objset
                   (2026-10-11 재렌더판은 sliding 없이 표준 창 셀 셋 — 점이 셀마다 하나 열이다)

  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P z_research/scripts/figures/plot_vobench_gravity.py     # → z_research/VOBench_v1/vjepa2/gravity/figures/
"""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]
_ap = __import__("argparse").ArgumentParser()
_ap.add_argument("--model", default="vjepa2", choices=["vjepa2", "ariel_ar_ep40"])
MODEL = _ap.parse_args().model
MNAME = {"vjepa2": "V-JEPA 2", "ariel_ar_ep40": "Ariel AR"}[MODEL]
D = REPO / "z_research/VOBench_v1" / MODEL / "gravity"
FIG = D / "figures"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
ARC = ["pos", "imp_rise", "imp_float", "imp_line", "imp_stop", "imp_stop2", "imp_stop4", "imp_stop8", "imp_stop12"]
XL = ["arc\n(true)", "rise", "float", "line", "stop", "stop2", "stop4", "stop8", "stop12"]
FREEZE = {"imp_stop": 45, "imp_stop2": 50, "imp_stop4": 52, "imp_stop8": 56, "imp_stop12": 60}
OBJ = ["primitive", "realistic"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("→", FIG / f"{name}.pdf")


def main():
    js = json.loads((D / "tables.json").read_text())
    W = js["windows"]
    # ── fig_arc_choice ──
    E = W["c16t32"]["frames"]["E"]
    ACELLS = sorted({k.split("|")[2].split(" · ")[1] for k in js["arc_pick"] if " · apex" in k}, key=lambda c: -int(c[4:]))
    phase = lambda rel: "rising" if rel < -3 else "apex" if rel <= 3 else "descending"
    fig, axs = plt.subplots(2, len(ACELLS), figsize=(7.0, 4.4), sharey=True, squeeze=False)
    x = np.arange(len(ARC)); bw = 0.27
    lab = iter("abcdefgh")
    for r, o in enumerate(OBJ):
        for c, cell in enumerate(ACELLS):
            ax = axs[r, c]
            key = f"c16t32|{{}}|{o} · {cell}"
            for k, (src, col, name) in enumerate((("pick_full", C1, f"{MNAME} (all tokens)"), ("pick_ball", C3, f"{MNAME} (ball tokens)"),
                                                  ("cpick_full", C2, "copy last context"))):
                v = [100 * js["arc_pick"][key.format(src)][a][0] for a in ARC]
                ax.bar(x + (k - 1) * bw, v, bw * 0.92, color=col, label=name)
            ax.axhline(100 / 9, ls="--", lw=0.8, color="0.5")
            ax.set_xticks(x, XL, fontsize=6, rotation=90)
            ax.set_ylim(0, 100)
            if c == 0:
                ax.set_ylabel(f"{o}\nclosest future (% of 12 blocks)")
            rel = E - int(cell[4:])
            ax.set_title(f"{phase(rel)}: apex f{cell[4:]}, context ends f{E} ({rel:+d})", fontsize=7)
            ax.text(0.5, -0.42, f"({next(lab)})", transform=ax.transAxes, ha="center")
    fig.tight_layout(h_pad=2.2, rect=(0, 0, 1, 0.94))
    fig.legend(*axs[0, 0].get_legend_handles_labels(), frameon=False, fontsize=7, loc="upper center", ncol=3)
    save(fig, "fig_arc_choice")

    # ── fig_arc_sliding ── (sliding 창이 있을 때만 — 2026-10-11 재렌더판은 표준 창 하나라 그리지 않는다)
    if not any(k.split("|")[1] != "c16t32" for k in js["freeze_delay"]):
        return
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    for r, o in enumerate(OBJ):
        ax = axs[r]
        for k, v in js["freeze_delay"].items():
            ob, w, cell = k.split("|")
            if ob != o:
                continue
            std = w == "c16t32"
            xx = v["rel"] + (0.6 if std else 0) + (0.25 if cell == "apex42" else -0.25)
            for a, f in FREEZE.items():
                ax.plot([xx - 0.35, xx + 0.35], [f - v["E"]] * 2, color="0.8", lw=1, zorder=1)
            for Dv, n in Counter(v["D_picked"]).items():
                ax.scatter(xx, Dv, s=10 + 12 * n, color=C2 if std else C1, zorder=3, edgecolor="white", lw=0.5)
            if v["n_non_freeze"]:
                ax.text(xx, -4, f"{v['n_non_freeze']}✕", ha="center", fontsize=6, color="0.3")
        ax.axvline(0, ls=":", lw=0.8, color="0.5")
        ax.set_xlabel("context end − apex (frames)   ← rising | descending →")
        if r == 0:
            ax.set_ylabel("chosen future: true motion followed\nbefore freezing, D (frames)")
        ax.set_title(o, fontsize=8)
        ax.set_ylim(-6, 32)
        ax.text(0.5, -0.40, f"({'ab'[r]})", transform=ax.transAxes, ha="center")
    if any(k.split("|")[1] != "c16t32" for k in js["freeze_delay"]):
        axs[1].scatter([], [], s=40, color=C1, label="stride 2 (sliding)")
    axs[1].scatter([], [], s=40, color=C2, label="stride 3 (standard c16t32)")
    axs[1].plot([], [], color="0.8", label="freeze candidates in window")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.legend(*axs[1].get_legend_handles_labels(), frameon=False, fontsize=7, loc="upper center", ncol=3)
    save(fig, "fig_arc_sliding")


if __name__ == "__main__":
    main()
