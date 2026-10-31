#!/usr/bin/env python3
"""VOBench k 2·3 (전 타이밍) — 16 장 문맥 encoder 블록 하나에서 물체 모양 · 색이 읽히나: (a) 물체 보이는 블록 (b) 가린 블록 (2026-10-29, 사용자 요청).

판독기는 보임 · 가림 따로 학습 (vobench_attnprobe_release.py probe_vis --subset k23). 각 판독기는 자기 종류의 블록에서만 시험.
  /venv/vjepa/bin/python z_research/scripts/figures/plot_vobench_ctx_vis_k23.py
입력: z_analysis/results/vobench_attnprobe_release/ctx_vis_k23/z_{shape,color}.json · 출력: z_analysis/figures/vobench_ctx_vis_k23.{png,pdf}
"""
import json, os
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
RES = os.path.join(REPO, "z_analysis/results/vobench_attnprobe_release/ctx_vis_k23")
M = json.load(open("/workspace/auto_research_cache/vobench_attnprobe_release/meta.json"))
R = {t: json.load(open(os.path.join(RES, f"z_{t}.json"))) for t in ("shape", "color")}
CONDS = [f"{o} k{k}" for o in ("early", "mid", "late") for k in (2, 3)]
cond = lambda m: f"{m['occlusion']} k{m['k']}"


def acc(tgt, cat, key, c):
    rs = [r for r in R[tgt] if r["cat"] == cat and (c == "All" or cond(M[r["i"]]) == c)]
    return 100 * float(np.mean([r[key] == r["y"] for r in rs])), len(rs)


plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
COL = {"shape": "#2a78d6", "color": "#eb6834"}
CH = {"shape": 100 / 6, "color": 100 / 8}
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
xs = np.arange(len(CONDS) + 1); w = 0.38
for ax, (cat, key, lab) in zip(axes, (("visible", "vis", "(a) object visible blocks"), ("hidden", "hid", "(b) object hidden blocks"))):
    for j, tgt in enumerate(("shape", "color")):
        v = [acc(tgt, cat, key, c)[0] for c in CONDS + ["All"]]
        b = ax.bar(xs + (j - 0.5) * w, v, w - 0.04, color=COL[tgt], label=tgt, zorder=3)
        ax.text(xs[-1] + (j - 0.5) * w, v[-1] + 1.5, f"{v[-1]:.0f}", ha="center", va="bottom", fontsize=8, color="#333")
        ax.axhline(CH[tgt], color=COL[tgt], lw=1, ls=(0, (3, 3)), alpha=0.8, zorder=2)
    n = acc("shape", cat, key, "All")[1]
    ax.set_xticks(xs); ax.set_xticklabels([c.replace(" ", "\n") for c in CONDS] + ["All"])
    ax.set_ylim(0, 105); ax.grid(axis="y", color="#e6e6e6", lw=0.6, zorder=0)
    ax.set_xlabel(f"{lab}  (n = {n})", labelpad=6)
axes[0].set_ylabel("probe accuracy (%)")
axes[1].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.42, 1.04), ncol=2, fontsize=8)
axes[1].text(xs[-1] + 0.62, CH["shape"], "chance", fontsize=7, color="#666", va="center", ha="left")
fig.tight_layout()
OD = os.path.join(REPO, "z_analysis/figures"); os.makedirs(OD, exist_ok=True)
plt.rcParams["pdf.fonttype"] = 42
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OD, f"vobench_ctx_vis_k23.{ext}"), dpi=200, bbox_inches="tight")
print(os.path.join(OD, "vobench_ctx_vis_k23.png"))
