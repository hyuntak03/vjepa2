#!/usr/bin/env python3
"""VOBench late — release vs AR, 움직임 종류별 쌍 정확도, 위반 종류별 패널 (2026-10-23, 사용자 지시).

  /venv/vjepa/bin/python z_research/scripts/figures/plot_vobench_late_release_vs_ar.py [--out <png>]

x = motion (flat_static · flat_vel · flat_acc · flat_dec · ramp_down · ramp_up), y = 쌍 정확도 (%, chance 50), 막대 = 모델.
(a) object permanence · (b) shape constancy · (c) color constancy. primitive · realistic 을 합친다.
쌍 = 문맥이 같은 matched 2 쌍 (pos_a ↔ imp_ab, pos_b ↔ imp_ba). 칸마다 n = 24 쌍 (1 쌍 = 4.2 pt).
결과 = late 만 따로 돌린 실행 (cvpr/eval/testbed/run.sh vobench_v1_late "vith vast_ar_vw_state")."""
import argparse, collections, csv, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
R = os.path.join(REPO, "cvpr/results/eval/testbed")
MODELS = [("V-JEPA 2 (release predictor)", "vobench_v1_late_vith"), ("AR predictor", "vobench_v1_late_vast_ar_vw_state")]
MOTIONS = ["flat_static", "flat_vel", "flat_acc", "flat_dec", "ramp_down", "ramp_up"]
VIOL = [("permanence", "(a) object permanence"), ("shape", "(b) shape constancy"), ("color", "(c) color constancy")]
COLORS = ["#7cc4fa", "#ffa48a"]   # 밝은 파스텔 (사용자 지시 2026-10-23: "너무 탁해") — 테두리는 같은 계열의 진한 색
EDGES = ["#3d9be0", "#f0764f"]

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.join(REPO, "z_research/VOBench_v1/figures/late_release_vs_ar.png"))
a = ap.parse_args()

blocks = collections.defaultdict(dict)
for r in csv.DictReader(open(os.path.join(REPO, "data_csv/vobench_v1/index.csv"))):
    if r["occlusion"] == "late":
        blocks[r["block_id"]][r["variant"]] = r
acc, tot = {}, {}
for name, d in MODELS:
    pv = json.load(open(os.path.join(R, d, "per_block.json")))["per_video_surprise"]
    cnt = collections.defaultdict(lambda: [0, 0])
    for v in blocks.values():
        for p, i in (("pos_a", "imp_ab"), ("pos_b", "imp_ba")):
            ok = pv[v[p]["video_id"]] < pv[v[i]["video_id"]]
            for k in ((v[p]["violation"], v[p]["motion"]), "all"):
                cnt[k][0] += ok; cnt[k][1] += 1
    acc[name] = {k: (100 * c / n, n) for k, (c, n) in cnt.items()}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), sharey=True)
w = 0.38
xs = np.arange(len(MOTIONS))
for ax, (viol, lab) in zip(axes, VIOL):
    for j, (name, _) in enumerate(MODELS):
        vals = [acc[name][(viol, m)][0] for m in MOTIONS]
        bars = ax.bar(xs + (j - 0.5) * w, vals, w, color=COLORS[j], edgecolor=EDGES[j], linewidth=0.8, label=name)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}", ha="center", va="bottom", fontsize=7)
    ax.axhline(50, color="k", lw=0.8, ls="--")
    ax.set_xticks(xs, MOTIONS, rotation=20)
    ax.set_ylim(0, 108)
    ax.set_xlabel(lab, fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
axes[0].set_ylabel("pairwise accuracy (%)")
fig.tight_layout(rect=(0, 0, 1, 0.92))
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=len(MODELS), frameon=False, bbox_to_anchor=(0.5, 1.0), fontsize=10)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
fig.savefig(a.out, dpi=160, bbox_inches="tight")
print(a.out)
print(f"칸당 쌍 {acc[MODELS[0][0]][(VIOL[0][0], MOTIONS[0])][1]}")
for name, _ in MODELS:
    print(f"\n[{name}] 전체 {acc[name]['all'][0]:.1f}")
    for viol, _ in VIOL:
        print(f"  {viol:11s}" + " ".join(f"{acc[name][(viol, m)][0]:6.1f}" for m in MOTIONS))
