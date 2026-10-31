#!/usr/bin/env python3
"""transfer_gap Q3 요약 그림 (2026-10-30, 실험 B [X2]). 수치는 전부 results/q3/*.json 에서 읽는다 (손으로 옮긴 값 없음).

(a) late · 이동 · 사라짐 A (물체 → 빈) 정확도: 릴리즈 vs 릴리즈+PFT — 분포 안 · 처음 보는 물체 · 가까운 세트 (v11, k4 = 학습에 없던 가림 길이) · 변환
(b) 유지율 r = G(T)/G(기준) 와 95 % CI (전체 · late) — 사전 규칙선 0.5 · 0.8
(c) 외부 벤치 정확도: 릴리즈 vs 릴리즈+PFT

  python z_research/scripts/figures/plot_transfer_gap_q3.py   → z_analysis/transfer_gap/results/q3/fig/q3_summary.{png,pdf}
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
Q = REPO / "z_analysis/transfer_gap/results/q3"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})

e10 = json.load(open(Q / "E10_objholdout.json"))["cells"]
e11 = json.load(open(Q / "E11_neardomain.json"))["sets"]
e12 = json.load(open(Q / "E12_transforms.json"))["transforms"]
e9 = json.load(open(Q / "E9_existing.json"))["matrix"]

rows = [("VOBench test\n(orig, te)", None, None)]
A = []
t0 = e12["T0"]["late_moving_perm_A"]["ref_same_blocks"]
A.append(("VOBench test", t0["rel"], t0["pft"]))
x = e10["all|unseen_any|late_moving_perm_A"]; A.append(("unseen objects", x["acc_b"], x["acc_a"]))
for s, lab in (("v11sub", "v11"), ("v11_realistic", "v11-real")):
    for g, kl in (("occ_moving_A_k1to3", "k1-3"), ("occ_moving_A_k4", "k4 (new)")):
        y = e11[s]["groups"][g]; A.append((f"{lab} {kl}", y["release"], y["release_pft"]))
for T, lab in (("T1", "T1 stride 2"), ("T2", "T2 reversed"), ("T5", "T5 window shift"), ("T4", "T4 hue +180")):
    y = e12[T]["late_moving_perm_A"]; A.append((lab, y["acc_b"], y["acc_a"]))

fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.6), gridspec_kw={"width_ratios": [2.3, 1.3, 1.2]})
xs = np.arange(len(A)); w = 0.38
ax[0].bar(xs - w / 2, [a[1] for a in A], w, color=C1, label="release")
ax[0].bar(xs + w / 2, [a[2] for a in A], w, color=C2, label="release + PFT")
ax[0].set_xticks(xs); ax[0].set_xticklabels([a[0] for a in A], rotation=60, ha="right", fontsize=6)
ax[0].set_ylabel("late moving vanish (A) acc. (%)"); ax[0].set_ylim(0, 105); ax[0].axhline(50, color="0.6", lw=0.6, ls=":")
ax[0].legend(frameon=False, fontsize=6, loc="upper left", bbox_to_anchor=(0.0, 1.18), ncol=2)

Ts = [("T0", "T0 writer"), ("T1", "T1 stride 2"), ("T2", "T2 reversed"), ("T5", "T5 window"), ("T3_x0y0", "T3 (0,0)"),
      ("T3_x64y0", "T3 (64,0)"), ("T3_x0y64", "T3 (0,64)"), ("T3_x64y64", "T3 (64,64)"), ("T4", "T4 hue")]
for j, (cell, col, lab) in enumerate((("all", C1, "all pairs"), ("late", C2, "late"))):
    for i, (T, _) in enumerate(Ts):
        r = (e12.get(T, {}).get(cell) or {}).get("ratio")
        if not r or r.get("r") is None:
            continue
        y = i + (j - 0.5) * 0.3
        ax[1].errorbar(r["r"], y, xerr=[[r["r"] - r["r_ci"][0]], [r["r_ci"][1] - r["r"]]], fmt="o", ms=3, color=col, lw=1,
                       label=lab if i == 1 else None)
ax[1].set_yticks(range(len(Ts))); ax[1].set_yticklabels([t[1] for t in Ts], fontsize=6); ax[1].invert_yaxis()
for v in (0.5, 0.8, 1.0):
    ax[1].axvline(v, color="0.6", lw=0.6, ls=":" if v != 1 else "-")
ax[1].set_xlim(0, 1.6); ax[1].set_xlabel("retention r = G(T) / G(ref)")
ax[1].legend(frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2)

B = [("IntPhys 1", e9["intphys1"]["vith"]["acc"], e9["intphys1"]["vith_release_vobftg"]["acc"]),
     ("GRASP", e9["grasp"]["vith"]["acc"], e9["grasp"]["vith_release_vobftg"]["acc"]),
     ("InfLevel cont.", e9["inflevel_continuity"]["vith"]["acc"], e9["inflevel_continuity"]["vith_release_vobftg"]["acc"]),
     ("IntPhys 2", e9["intphys2"]["vith"]["acc"], e9["intphys2"]["vith_release_vobftg"]["acc"])]
xb = np.arange(len(B))
ax[2].bar(xb - w / 2, [b[1] for b in B], w, color=C1)
ax[2].bar(xb + w / 2, [b[2] for b in B], w, color=C2)
ax[2].set_xticks(xb); ax[2].set_xticklabels([b[0] for b in B], rotation=60, ha="right", fontsize=6)
ax[2].set_ylabel("pair accuracy (%)"); ax[2].set_ylim(40, 100); ax[2].axhline(50, color="0.6", lw=0.6, ls=":")
for i, lab in enumerate("abc"):
    ax[i].text(0.5, -0.66 if i != 1 else -0.42, f"({lab})", transform=ax[i].transAxes, ha="center", fontsize=8)
fig.subplots_adjust(left=0.07, right=0.99, bottom=0.36, top=0.88, wspace=0.55)
out = Q / "fig"; out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "q3_summary.png", dpi=200); fig.savefig(out / "q3_summary.pdf")
print("→", out / "q3_summary.png")
