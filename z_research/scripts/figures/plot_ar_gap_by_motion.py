"""블록 AR 의 VOBench 끝 가림 surprise gap — 가로 = 운동 종류, 세로 = gap, 위반 종류 3 개를 패널 3 개로 (2026-10-26).
사용자: "AR 의 surprise score gap 을 x 축을 motion type, y 축을 gap 이라 해서 figure. violation type 3 개니까 한 파일로".

입력: cvpr/results/eval/testbed/vobench_v1_late_vast_ar_vw_state/per_block.json (기준 AR natural_ar_vw_state, surprise_c16t32 채점)
쌍 = (block_id, pair_id) 안의 가능 1 · 불가능 1 (matched). gap = surprise(불가능) − surprise(가능), > 0 이면 맞힘.
영속 (permanence) 은 obj_b = empty 라 짝 A = 사라짐 (가능에 물체 있음 → 불가능에서 사라짐), 짝 B = 나타남 — 점 색을 나눈다.
점 = 쌍 하나 (운동 종류마다 12 block × 2 쌍 = 24), 막대 = 평균 ± 표준편차, 위 숫자 = 정확도 (gap > 0 비율).

  /venv/vjepa/bin/python z_research/scripts/figures/plot_ar_gap_by_motion.py
출력: z_analysis/figures/ar_vobench_late_gap_by_motion.png · z_analysis/results/ar_vobench_late_gap_by_motion.json
"""
import csv, json
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/workspace/world_model/vjepa2"
PV = json.load(open(f"{REPO}/cvpr/results/eval/testbed/vobench_v1_late_vast_ar_vw_state/per_block.json"))["per_video_surprise"]
get = lambda v: float(v["all"] if isinstance(v, dict) else v)
MOT = ["flat_static", "flat_vel", "flat_acc", "flat_dec", "ramp_down", "ramp_up"]
MOT_LAB = ["static", "const. vel.", "accel.", "decel.", "ramp down", "ramp up"]
VIO = ["permanence", "shape", "color"]
pairs = defaultdict(dict)
for r in csv.DictReader(open(f"{REPO}/data_csv/vobench_v1/index.csv")):
    if r["occlusion"] != "late" or r["video_id"] not in PV:
        continue
    k = (r["block_id"], r["pair_id"])
    pairs[k][int(r["plausible"])] = get(PV[r["video_id"]])
    pairs[k]["meta"] = (r["violation"], r["motion"], r["pair_id"], r.get("obj_b", ""))
data = defaultdict(list)                                                     # (위반, 운동) → [(gap, 방향)]
for k, p in pairs.items():
    if 0 in p and 1 in p:
        vio, mot, pid, objb = p["meta"]
        d = ("vanish" if pid == "A" else "appear") if (vio == "permanence" and objb == "empty") else "pair"   # 짝 A = pos_a (물체) vs imp_ab (사라짐), 짝 B = pos_b (빈) vs imp_ba (나타남)
        data[(vio, mot)].append((p[0] - p[1], d))
summary = {f"{v}|{m}": {"n": len(data[(v, m)]), "mean_gap": float(np.mean([g for g, _ in data[(v, m)]])), "std_gap": float(np.std([g for g, _ in data[(v, m)]])),
                       "acc": float(np.mean([g > 0 for g, _ in data[(v, m)]]) * 100)} for v in VIO for m in MOT}
json.dump(summary, open(f"{REPO}/z_analysis/results/ar_vobench_late_gap_by_motion.json", "w"), indent=1)

plt.rcParams["font.family"] = "DejaVu Sans"
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)
rng = np.random.default_rng(0)
COL = {"vanish": "#eb6834", "appear": "#2a78d6", "pair": "#7a7a7a"}
for ax, v in zip(axes, VIO):
    for i, m in enumerate(MOT):
        G = data[(v, m)]
        for g, d in G:
            ax.scatter(i + rng.uniform(-0.18, 0.18), g, s=12, color=COL[d], alpha=0.6, lw=0)
        gs = np.array([g for g, _ in G])
        ax.errorbar(i + 0.3, gs.mean(), yerr=gs.std(), fmt="o", color="k", capsize=4, ms=5, lw=1.5)
        ax.text(i, 1.0, f"{summary[f'{v}|{m}']['acc']:.0f}%", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8)
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_xticks(range(len(MOT))); ax.set_xticklabels(MOT_LAB, fontsize=9, rotation=20)
    ax.set_title(v, fontsize=11, pad=16); ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("surprise gap  (impossible − possible)")
h = [plt.Line2D([], [], marker="o", ls="", color=COL[k], label=lab) for k, lab in (("vanish", "permanence: vanish"), ("appear", "permanence: appear"), ("pair", "shape / color pair"))]
h.append(plt.Line2D([], [], marker="o", ls="-", color="k", label="mean ± std"))
for ax, lab in zip(axes, ("(a)", "(b)", "(c)")):
    ax.set_xlabel(lab, fontsize=11)   # 2026-10-26 사용자: "겹치는 것들 모두 해결" — 패널 표시는 x 축 이름 아래 줄로 (자동 배치)
fig.suptitle("Surprise gap, VOBench late occlusion: AR predictor output vs encoder targets, 1280-d tokens (mean L1 over 256 × 1280 per block)", fontsize=12, y=0.995)   # 2026-10-26 사용자: 상단 가운데 제목
fig.tight_layout(rect=(0, 0.08, 1, 0.95)); fig.legend(handles=h, loc="lower center", ncol=4, fontsize=9, frameon=False)
fig.savefig(f"{REPO}/z_analysis/figures/ar_vobench_late_gap_by_motion.png", dpi=170, bbox_inches="tight")
for v in VIO:
    print(v, {m: (round(summary[f'{v}|{m}']['mean_gap'], 4), round(summary[f'{v}|{m}']['acc'])) for m in MOT})
