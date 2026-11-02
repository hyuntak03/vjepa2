#!/usr/bin/env python3
"""VOBench late · 움직이는 물체 사라짐 정답률 — 평면 (등속 · 가속 · 감속 18 쌍) / 빗면 (내려감 · 올라감 12 쌍), v3 · v4 학습 팔 (2026-11-02).
판정 = table_v3 (미래 8 블록 평균 L1, 사라짐 > 가능 이면 정답, 같으면 0.5). 값은 채점 산출물에서 다시 셈. 플롯 라벨은 영어.
  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python; $P predictor_lab/v4vc/plot_vob_moving_vanish.py → predictor_lab/results/v4vc/figures/vob_moving_vanish_flat_ramp.png
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "predictor_lab/v3_ablation"))
import table_v3 as T  # noqa: E402

ROWS = [("copy (no prediction)", "copy", "ref"), ("release (before training)", "release", "ref"),
        ("v3 mix (1,600 scenes)", "v3_mix1600_s0", "v3"), ("v3 all (7,136 scenes)", "v3_full_s0", "v3"), ("v3 trapdoor only", "v3t_only_s0", "v3"),
        ("v4 flat, 2k steps", "v4vc_fam_s0", "v4"), ("v4 flat, 6k steps", "v4vc_fam6k_s0", "v4"), ("v4 flat + ramp, 4k steps", "v4vc_all_s0", "v4"),
        ("VOBench-trained (in-domain)", "vobftg", "ref")]
GROUPS = [("flat", ("flat_vel", "flat_acc", "flat_dec")), ("ramp", ("ramp_down", "ramp_up"))]
S, P = T.panel_scores("vob_late")
res = {}
for lab, arm, _ in ROWS:
    for g, mos in GROUPS:
        sel = [(a, b) for a, b, r in P if r["violation_type"] == "permanence" and r["pair_id"] == "A" and r["motion"] in mos]
        j = T._judge(S[arm], sel)
        res[(arm, g)] = (100 * j.mean(), j.sum(), len(sel))
json.dump({f"{a}|{g}": v for (a, g), v in res.items()}, open(REPO / "predictor_lab/results/v4vc/figures/vob_moving_vanish_flat_ramp.json", "w"), indent=1, default=float)

C1, C2 = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, ax = plt.subplots(figsize=(8.2, 5.6), dpi=200)
fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
n = len(ROWS); y = np.arange(n)[::-1] * 1.0; h = 0.36
for i, (lab, arm, grp) in enumerate(ROWS):
    for k, (g, _) in enumerate(GROUPS):
        acc, kk, nn = res[(arm, g)]
        yy = y[i] + (h / 2 + 0.02 if k == 0 else -h / 2 - 0.02)
        ax.barh(yy, max(acc, 0.6), height=h, color=C1 if k == 0 else C2, alpha=1.0 if grp != "ref" else 0.45, zorder=3, linewidth=0)
        ax.text(max(acc, 0.6) + 1.2, yy, f"{acc:.0f}%  ({kk:g}/{nn})", va="center", ha="left", fontsize=8.5, color=INK2, zorder=4)
ax.axvline(50, color=INK2, lw=1, ls=(0, (3, 3)), zorder=2)
ax.text(50.8, y[-1] - 0.75, "chance 50%", fontsize=8, color=INK2, va="bottom")
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in ROWS])
for t, (_, _, grp) in zip(ax.get_yticklabels(), ROWS):
    t.set_color(INK if grp != "ref" else INK2)
    if grp == "v4": t.set_fontweight("bold")
for yb in (y[1] - 0.5, y[4] - 0.5, y[7] - 0.5):
    ax.axhline(yb, color=GRID, lw=1, zorder=1)
ax.set_xlim(0, 118); ax.set_xticks([0, 25, 50, 75, 100])
ax.xaxis.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID); ax.tick_params(axis="y", length=0); ax.tick_params(axis="x", colors=INK2)
ax.set_xlabel("accuracy (%): impossible (object vanished) judged more surprising", color=INK2, fontsize=9)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=C1, label="flat floor: const. / accel. / decel. (18 pairs)"), Patch(color=C2, label="ramp: down / up (12 pairs)")],
          loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False, fontsize=8.5, borderaxespad=0.3, handlelength=1.2)
fig.suptitle("VOBench late occlusion, moving object vanishes behind the occluder", x=0.02, ha="left", fontsize=11.5, fontweight="bold", color=INK)
fig.text(0.02, 0.928, "Same test pairs for every model. Faded rows are references. One seed per training run.", fontsize=8.5, color=INK2, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.9))
out = REPO / "predictor_lab/results/v4vc/figures/vob_moving_vanish_flat_ramp.png"
fig.savefig(out, facecolor=fig.get_facecolor()); print(out)
for (a, g), v in res.items(): print(a, g, v)
