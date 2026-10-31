#!/usr/bin/env python3
"""transfer_gap R4 [X2] 그림 — 진단 PFT 의 전이 시험판 주 지표 + 끝 자리 지표 (2026-10-30).
(같은 이름 plot_transfer_gap_r4.py 는 [X1] 의 그림이다 — 이 파일과 다르다.)

입력: z_analysis/transfer_gap/results/r4_x2/R4_transfer.json (transfer_gap_r4_report.py) ·
      z_analysis/transfer_gap/results/r4_x2/E35r4_endprior.json (transfer_gap_r4_endprior.py report)
출력: z_analysis/transfer_gap/results/r4_x2/fig/r4_x2_summary.{png,pdf}
  (a)–(e) 판마다 주 칸의 쌍 정확도 (점 = 값, 세로 막대 = 장면 bootstrap 95 % CI, 점선 = 우연 50, 회색 띠 = 복사 기준선 CI)
  (f) 보폭 2 끌림 비 λ (t15; 0 = 문맥 속력, 1 = 보폭 3 자리) · (g) 가로 따라감 비 β (t15; 1 = 장면을 따름, 0 = 화면 고정)
  (h) 판마다 가능 영상 평균 surprise 의 비 (T1 · T2 ÷ T0 s0), 장면 bootstrap CI. 릴리즈는 (f)(g) 에서 뺐다 (지도에 물체 자리 없음)
  라벨은 영어 (한글 폰트 없음). 색 = 검증 팔레트 slot 1–3 (T0 파랑 · T1 주황 · T2 초록), 참조 모델은 회색.

  /data/hyuntak/anaconda3/envs/vjepa2/bin/python z_research/scripts/figures/plot_transfer_gap_r4_x2.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
RES = REPO / "z_analysis/transfer_gap/results/r4_x2"
OUT = RES / "fig"
MODELS = [("release", "release", "#8a8f98", "o"), ("vobftg", "orig PFT", "#5f6670", "o"),
          ("t0s0", "T0 s0", "#2a78d6", "o"), ("t0s1", "T0 s1", "#2a78d6", "s"),
          ("t1", "T1 end-scatter", "#eb6834", "D"), ("t2", "T2 gen-mix", "#1baf7a", "^")]
PANELS = [("vob_late", "perm_move_A", "VOBench late (in-dist)\nvanish A, moving"),
          ("ip1uni", "move_occ", "IntPhys 1 late uni\nmoving + occluded"),
          ("ip2uni", "permanence", "IntPhys 2 late uni\npermanence"),
          ("v11late", "move_A", "v11 late occlusion\nvanish A, moving"),
          ("v11real", "move_A", "v11 realistic\nvanish A, moving")]


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    R = json.load(open(RES / "R4_transfer.json"))
    E = json.load(open(RES / "E35r4_endprior.json"))
    fig = plt.figure(figsize=(7.0, 5.4))
    gs = fig.add_gridspec(2, 5, height_ratios=[1, 0.9], hspace=1.45, wspace=0.45)
    labels = "abcdefgh"
    for i, (pn, cell, title) in enumerate(PANELS):
        ax = fig.add_subplot(gs[0, i])
        c = R.get(pn, {}).get("cells", {}).get(cell)
        ax.set_title(title, fontsize=6.5)
        ax.axhline(50, ls=":", lw=0.8, color="#9aa0a6")
        if c:
            lo, hi = c["acc_ci"]["copy"]
            ax.axhspan(lo, hi, color="#d9dce0", alpha=0.7, lw=0)
            for j, (t, lab, col, mk) in enumerate(MODELS):
                if t not in c["acc"]:
                    continue
                v, (l, h) = c["acc"][t], c["acc_ci"][t]
                ax.plot([j, j], [l, h], color=col, lw=1.8, solid_capstyle="round")
                ax.plot(j, v, mk, color=col, ms=5, mec="white", mew=0.8)
            ax.set_xlabel(f"n = {c['n_pair']} pairs", fontsize=6, color="#5f6670")
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels([m[1] for m in MODELS], rotation=70, ha="right", fontsize=5.5)
        ax.set_ylim(-3, 103)
        if i == 0:
            ax.set_ylabel("pair accuracy (%)")
        ax.text(0.5, -0.97, f"({labels[i]})", transform=ax.transAxes, ha="center", fontsize=8)
    M2 = [m for m in MODELS if m[0] != "release"]          # 릴리즈는 물체 자리가 지도에 없다 (유효 0–10 %) → λ · β 를 그리지 않는다
    for k, (key, title) in enumerate([("stride2_pull_lambda", "stride-2 pull \u03bb (t15)\n0 = context speed, 1 = stride-3 spot"),
                                      ("crop_follow_beta", "horizontal follow \u03b2 (t15)\n1 = follows scene, 0 = screen-fixed")]):
        ax = fig.add_subplot(gs[1, 2 * k: 2 * k + 2])
        ax.set_title(title, fontsize=6.5)
        src = E.get(key, {})
        if key == "crop_follow_beta":
            src = src.get("x", {})
        for j, (t, lab, col, mk) in enumerate(M2):
            o = src.get(t if t != "vobftg" else "vobpft")
            if not isinstance(o, dict):
                continue
            v = o.get("lambda_median") if key == "stride2_pull_lambda" else o.get("beta")
            ci = o.get("ci95")
            if ci:
                ax.plot([j, j], ci, color=col, lw=1.8, solid_capstyle="round")
            ax.plot(j, v, mk, color=col, ms=5, mec="white", mew=0.8)
        ax.axhline(0, ls=":", lw=0.8, color="#9aa0a6"); ax.axhline(1, ls=":", lw=0.8, color="#9aa0a6")
        ax.set_ylim(-0.5, 1.5)
        ax.set_xticks(range(len(M2)))
        ax.set_xticklabels([m[1] for m in M2], rotation=30, ha="right", fontsize=6)
        ax.text(0.5, -0.62, f"({labels[len(PANELS) + k]})", transform=ax.transAxes, ha="center", fontsize=8)
    # (h) 가능 영상 예측 오차 비 ε = X / T0 s0 (판마다)
    ax = fig.add_subplot(gs[1, 4])
    ax.set_title("possible-video error ratio\nvs T0 s0 (<1 = lower error)", fontsize=6.5)
    short = ["VOB", "IP1", "IP2", "v11", "v11r"]
    for j, (pn, _, _) in enumerate(PANELS):
        e = R.get(pn, {}).get("eps", {}).get("ratio", {})
        for dx, (t, col, mk) in zip((-0.15, 0.15), (("t1", "#eb6834", "D"), ("t2", "#1baf7a", "^"))):
            o = e.get(f"{t}/t0s0")
            if o:
                ax.plot([j + dx, j + dx], o["ci"], color=col, lw=1.8)
                ax.plot(j + dx, o["v"], mk, color=col, ms=4.5, mec="white", mew=0.6, label=("T1" if t == "t1" else "T2") if j == 0 else None)
    ax.axhline(1, ls=":", lw=0.8, color="#9aa0a6")
    ax.set_xticks(range(len(PANELS))); ax.set_xticklabels(short, fontsize=6, rotation=45, ha="right")
    ax.legend(frameon=False, fontsize=6, loc="upper right")
    ax.text(0.5, -0.62, f"({labels[7]})", transform=ax.transAxes, ha="center", fontsize=8)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"r4_x2_summary.{ext}", dpi=200, bbox_inches="tight")
    print("→", OUT / "r4_x2_summary.png")


if __name__ == "__main__":
    main()
