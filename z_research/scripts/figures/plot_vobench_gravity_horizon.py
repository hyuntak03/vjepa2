#!/usr/bin/env python3
"""VOBench_v1 gravity arc — 예측 길이 M 별 "공이 진짜 운동을 따르나" 곡선 (2026-10-11, 2 차: 보기 쉽게).

objset 마다 한 장, 패널 = 문맥 끝 상태 (오르는 중 · 꼭대기 · 내려가는 중). x = 예측 샘플 순번 (문맥 끝에서 몇 번째, 3 차),
y = p 가 정지 (f45) 보다 진짜 운동 (arc) 에 가까운 block % (옅은 띠 = block 부트스트랩 95 % CI).
선 = 예측 길이 16 · 8 · 4 · 2 샘플 (선이 끝나는 곳 = 그 창의 끝). 2 차의 절대 프레임축 · 모양 다른 표지는 사용자: "보기 어렵다". 입력 = vobench_gravity_horizon.py 의 tables.json.
1 차 (2×3, 선 4 개 + 오차 막대) 는 보기 어려워 figures/_superseded/ 로 내렸다.

  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P z_research/scripts/figures/plot_vobench_gravity_horizon.py
     → z_research/VOBench_v1/vjepa2/gravity_horizon/figures/fig_horizon_ball_<objset> (주) · fig_horizon_all_<objset> (전체 토큰)
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]
D = REPO / "z_research/VOBench_v1/vjepa2/gravity_horizon"
COL = {16: "0.1", 8: "#2a78d6", 4: "#eb6834", 2: "#1baf7a"}
CELLS = [("apex66", "Context: rising"), ("apex48", "Context ends at the apex"), ("apex30", "Context: descending")]
plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman", "DejaVu Serif"], "font.size": 8,
                     "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})


def draw(js, o, tag, name, ylab):
    Ms = sorted({int(k.split("|")[3]) for k in js}, reverse=True)
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.5), sharey=True)
    for p, (ax, (cell, title)) in enumerate(zip(axs, CELLS)):
        ax.axhline(50, ls="--", lw=0.6, color="0.7", zorder=0)
        for k, M in enumerate(Ms):
            T = M // 2
            v = 100 * np.array([js[f"{tag}|{o}|{cell}|{M}|{t}"] for t in range(T)])
            x = np.array([2 * t + 1.5 for t in range(T)]) + (k - 1.5) * 0.12          # 튜블릿 t = 예측 샘플 2t+1, 2t+2 (겹침 방지로 살짝 비킴)
            ax.fill_between(x, v[:, 1], v[:, 2], color=COL[M], alpha=0.10, lw=0, zorder=1)
            ax.plot(x, v[:, 0], color=COL[M], lw=1.6, marker="o", ms=3.2, zorder=3 + k, label=f"predict {M} samples")
        ax.set_xlim(0.5, 16.5); ax.set_ylim(-4, 104)
        ax.set_xticks([2, 4, 8, 12, 16]); ax.set_yticks([0, 50, 100])
        ax.set_title(title, fontsize=9, pad=9)
        if p == 1:
            ax.set_xlabel("prediction step (samples after the context ends)")
        if p == 0:
            ax.set_ylabel(ylab)
        ax.text(0.5, -0.36, f"({'abc'[p]})", transform=ax.transAxes, ha="center", va="top", fontsize=9)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(left=0.085, right=0.99, top=0.86, bottom=0.36, wspace=0.08)
    out = D / "figures"; out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}_{o}.{ext}", dpi=250)
    plt.close(fig)
    print("→", out / f"{name}_{o}.pdf")


def main():
    js = json.loads((D / "tables.json").read_text())["move"]
    for o in ("primitive", "realistic"):
        draw(js, o, "move_ball", "fig_horizon_ball", "ball follows true motion\n(% blocks, ball tokens)")
        draw(js, o, "move_all", "fig_horizon_all", "follows true motion\n(% blocks, all tokens)")


if __name__ == "__main__":
    main()
