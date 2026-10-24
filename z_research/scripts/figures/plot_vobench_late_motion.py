#!/usr/bin/env python3
"""VOBench late — 움직임 종류별 쌍 정확도, AR 변형 비교 막대그래프 (2026-10-21, 사용자 지시).

  /venv/vjepa/bin/python z_research/scripts/figures/plot_vobench_late_motion.py [--out <png 경로>]

x = motion (flat_static · flat_vel · flat_acc · flat_dec · ramp_down · ramp_up), y = 쌍 정확도 (%, chance 50), 막대 = 모델.
(a) primitive · (b) realistic. 쌍 = 문맥이 같은 matched 2 쌍 (pos_a ↔ imp_ab, pos_b ↔ imp_ba) — per_block.json 의 블록 acc 216 개를 그대로 재현한다.
영상 surprise 는 각 실행의 per_block.json 의 per_video_surprise (전체 VOBench 실행이면 late 영상만 쓴다).
칸마다 n = 36 쌍 (위반 3 × 물체 · 배경 · k 조합) — 1 쌍 = 2.8 pt.
⚠️ AR · AR + op · video context 는 VOBench 전체 실행에서 late 만 골라 쓴다 — late 만 따로 돌린 시트 값 (AR 73.6 / 77.3) 과 0.5 pt 다르다 (전체 실행 74.1 / 77.8)."""
import argparse, collections, csv, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
R = os.path.join(REPO, "z_research/VOBench_v1/exp_results")
MODELS = [   # (그림 이름, 결과 폴더) — 시트 Intuitive Physics 의 행 이름을 따른다
    ("AR", "surprise_c16t32__vobench_v1_vast_ar_vw_state"),
    ("AR + object permanence", "surprise_c16t32__vobench_v1_vast_ar_vw_state_op"),
    ("AR w/ residual gating", "surprise_c16t32__vobench_v1_late_vast_ar_vw_state_gres"),
    ("AR + phys2d + sup.", "surprise_c16t32__vobench_v1_late_vast_ar_vw_state_phys2d_single"),
    ("AR + phys3d + sup. (e68)", "surprise_c16t32__vobench_v1_late_vast_ar_vw_state_phys3d_event_e68"),
    ("AR, video context", "surprise_c16t32__vobench_v1_vast_ar_vw_state_ctxt_video"),
]
MOTIONS = ["flat_static", "flat_vel", "flat_acc", "flat_dec", "ramp_down", "ramp_up"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#8a5cc2", "#c23b5a", "#7a7a7a"]

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.join(REPO, "z_research/VOBench_v1/figures/late_motion_ar_variants.png"))
a = ap.parse_args()

idx = [r for r in csv.DictReader(open(os.path.join(REPO, "data_csv/vobench_v1/index.csv"))) if r["occlusion"] == "late"]
blocks = collections.defaultdict(dict)
for r in idx:
    blocks[r["block_id"]][r["variant"]] = r
acc = {}   # (모델, objset, motion) → (맞은 쌍, 쌍)
for name, d in MODELS:
    pv = json.load(open(os.path.join(R, d, "per_block.json")))["per_video_surprise"]
    cnt = collections.defaultdict(lambda: [0, 0])
    for b, v in blocks.items():
        for p, i in (("pos_a", "imp_ab"), ("pos_b", "imp_ba")):
            k = (v[p]["objset"], v[p]["motion"])
            cnt[k][0] += pv[v[p]["video_id"]] < pv[v[i]["video_id"]]
            cnt[k][1] += 1
    for k, (c, n) in cnt.items():
        acc[(name, *k)] = (100 * c / n, n)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
w = 0.8 / len(MODELS)
xs = np.arange(len(MOTIONS))
for ax, objset, lab in zip(axes, ("primitive", "realistic"), ("(a) primitive", "(b) realistic")):
    for j, (name, _) in enumerate(MODELS):
        vals = [acc[(name, objset, m)][0] for m in MOTIONS]
        ax.bar(xs + (j - (len(MODELS) - 1) / 2) * w, vals, w, color=COLORS[j], label=name)
    ax.axhline(50, color="k", lw=0.8, ls="--")
    ax.set_xticks(xs, MOTIONS)
    ax.set_ylim(0, 105)
    ax.set_xlabel(lab, fontsize=10)
    ax.grid(axis="y", alpha=0.3)
axes[0].set_ylabel("pair accuracy (%)")
axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
fig.tight_layout()
os.makedirs(os.path.dirname(a.out), exist_ok=True)
fig.savefig(a.out, dpi=160, bbox_inches="tight")
print(a.out)
n = acc[(MODELS[0][0], "primitive", MOTIONS[0])][1]
print(f"칸당 쌍 {n}")
for objset in ("primitive", "realistic"):
    print(f"\n[{objset}] " + " ".join(f"{m:>10s}" for m in MOTIONS))
    for name, _ in MODELS:
        print(f"  {name:26s}" + " ".join(f"{acc[(name, objset, m)][0]:10.1f}" for m in MOTIONS))
