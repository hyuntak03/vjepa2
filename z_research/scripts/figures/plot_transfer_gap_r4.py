#!/usr/bin/env python3
"""transfer_gap R4 [X1] 요약 그림 (2026-10-30).

(a) E37 × E39: 다시 나타날 자리 query → 과거 물체 자리 attention (균등 대비, 층 L5–L11) vs 같은 판의 late 사라짐 A 이득 (릴리즈 대비, te 경로).
(b) E38: VOBench PFT 에서 미래 → 과거 물체 자리 간선을 막았을 때 참 자리 정체 판독 (test block).
(c) E38: 사라짐 A · 존재 대비 최대 자리 1 칸 안 — 릴리즈 (깨끗 · 키우기 β4 · 대조) · PFT (깨끗 · 막기 · 대조).
입력: z_analysis/transfer_gap/results/r4_x1/{E37_attn,E39_matrix,E38_ko}.json
출력: z_analysis/transfer_gap/results/r4_x1/fig/r4_summary.{png,pdf}
재현: /data/hyuntak/anaconda3/envs/vjepa2/bin/python z_research/scripts/figures/plot_transfer_gap_r4.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]
R = REPO / "z_analysis/transfer_gap/results/r4_x1"
C1, C2, C3, GR = "#2a78d6", "#eb6834", "#1baf7a", "#8a8f98"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#555", "axes.labelcolor": "#222", "xtick.color": "#444", "ytick.color": "#444", "pdf.fonttype": 42})
NAME = {"release": "rel", "vobftg": "VOB", "v11pft": "v11", "ip1late_f0": "IP1L", "ip2e80": "IP2", "ip2pair": "IP2p"}
# 직접 라벨 자리 (점 기준 오프셋, pt) — 겹침을 피하려고 손으로 둔다
OFF = {("vobench", "release"): (-4, -10), ("vobench", "v11pft"): (2, -11), ("vobench", "ip2pair"): (-12, 5), ("vobench", "ip2e80"): (-2, -12),
       ("vobench", "ip1late_f0"): (3, 6), ("vobench", "vobftg"): (-6, -15), ("v11", "release"): (-14, 5), ("v11", "ip1late_f0"): (5, -3),
       ("v11", "ip2pair"): (5, -3), ("v11", "ip2e80"): (5, -3), ("v11", "vobftg"): (-22, -4), ("v11", "v11pft"): (-16, 3)}

E37 = json.load(open(R / "E37_attn.json")); E39 = json.load(open(R / "E39_matrix.json")); E38 = json.load(open(R / "E38_ko.json"))
fig, ax = plt.subplots(1, 3, figsize=(7.0, 3.0), gridspec_kw={"width_ratios": [1.25, 1.1, 1.0]})

# (a)
a = ax[0]
for setn, col, c in (("vobench", "vob", C1), ("v11", "v11", C2)):
    sub = E37["sets"][setn]["late_test"]
    for m, v in sub.items():
        if m not in NAME: continue
        x = v["band_ring"]["obj_over_uniform"]
        r = E39["cols"][col]["rows"].get(m)
        if r is None: continue
        g = r.get("G_A", [0.0, 0.0, 0.0])
        a.errorbar(x, g[0], yerr=[[g[0] - g[1]], [g[2] - g[0]]], fmt="o", ms=4.5, color=c, mec="white", mew=0.8, elinewidth=0.8, capsize=0, zorder=3)
        if m != "release":   # 릴리즈는 정의상 이득 0 — 캡션에 적는다
            a.annotate(NAME[m], (x, g[0]), xytext=OFF.get((setn, m), (4, 3)), textcoords="offset points", fontsize=6, color="#333")
a.set_xscale("log"); a.set_xlim(0.4, 11); a.set_xticks([0.5, 1, 2, 4, 8]); a.set_xticklabels(["0.5", "1", "2", "4", "8"])
from matplotlib.ticker import NullFormatter
a.xaxis.set_minor_formatter(NullFormatter())
a.axvline(1, color=GR, lw=0.6, ls=":"); a.axhline(0, color=GR, lw=0.6)
a.set_xlabel("attention to past object box / uniform\n(reappearance query, layers 5-11)"); a.set_ylabel("vanish-A gain over release (pt)")
a.plot([], [], "o", color=C1, label="VOBench late (test)"); a.plot([], [], "o", color=C2, label="v11 occluded moving")
a.legend(frameon=False, fontsize=6, loc="upper left", markerscale=0.8, handletextpad=0.2); a.set_ylim(-15, 110)

# (b)
b = ax[1]
tasks = ["primitive_shape", "primitive_color", "realistic_shape", "realistic_color"]
lab = ["prim\nshape", "prim\ncolor", "real\nshape", "real\ncolor"]
rows = [("release__clean", "release", GR), ("vobftg__clean", "PFT", C1), ("vobftg__ko_obj", "PFT, cut to object", C2), ("vobftg__ko_ctl", "PFT, cut to control", C3)]
w = 0.19
for k, (key, nm, c) in enumerate(rows):
    vals = [E38["readout"][t][key][0] for t in tasks]
    lo = [E38["readout"][t][key][0] - E38["readout"][t][key][1] for t in tasks]; hi = [E38["readout"][t][key][2] - E38["readout"][t][key][0] for t in tasks]
    xs = np.arange(len(tasks)) + (k - 1.5) * (w + 0.02)
    b.bar(xs, vals, w, color=c, label=nm, yerr=[lo, hi], error_kw=dict(elinewidth=0.6, ecolor="#444"))
for i, t in enumerate(tasks):
    b.plot([i - 0.45, i + 0.45], [E38["readout"][t]["chance"]] * 2, color="#222", lw=0.6, ls=":")
b.set_xticks(range(len(tasks))); b.set_xticklabels(lab, fontsize=7); b.set_ylabel("identity readout at true spot (%)"); b.set_ylim(0, 100)
b.legend(frameon=False, fontsize=5.5, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, handlelength=1.0, columnspacing=0.8)

# (c)
c_ = ax[2]
conds = [("release__clean", "rel"), ("release__st_obj_b4", "rel\nsteer"), ("release__st_ctl_b4", "rel\nst-ctl"),
         ("vobftg__clean", "PFT"), ("vobftg__ko_obj", "PFT\ncut"), ("vobftg__ko_ctl", "PFT\ncut-ctl")]
P = E38["perm"]; xs = np.arange(len(conds))
c_.bar(xs - 0.2, [P[k]["A"][0] for k, _ in conds], 0.38, color=C1, label="vanish A")
c_.bar(xs + 0.2, [P[k]["within1"][0] for k, _ in conds], 0.38, color=C2, label="presence peak at true spot")
c_.set_xticks(xs); c_.set_xticklabels([n for _, n in conds], fontsize=6); c_.set_ylabel("VOBench late (%)"); c_.set_ylim(0, 100)
c_.legend(frameon=False, fontsize=5.5, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=1, handlelength=1.0)
fig.tight_layout(rect=[0, 0.06, 1, 1])
for i, axx in enumerate(ax):
    pos = axx.get_position()
    fig.text(pos.x0 + pos.width / 2, 0.01, f"({'abc'[i]})", ha="center", fontsize=8)
(R / "fig").mkdir(exist_ok=True)
fig.savefig(R / "fig/r4_summary.png", dpi=200, bbox_inches="tight"); fig.savefig(R / "fig/r4_summary.pdf", bbox_inches="tight")
print(R / "fig/r4_summary.png")
