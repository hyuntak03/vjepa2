"""teacher forcing 의 마지막 문맥 자리 (블록 7) 은닉 — 층별 attentive probe (2026-10-27). 기존 그림 (predictor_info_by_layer.png) 은 그대로 두고 새로 그림.
사용자: "teacher forcing 하는 context 자리에서도 그려 줄래? 마지막 context 자리만" → "기존 그림에 그리지 말고 새로 그려".
입력: z_analysis/results/attnprobe384/tfctx7_{own,next}_L{0..13}.json (attnprobe384_lastctx.py). 실선 = 블록 7 자기 라벨, 점선 = 다음 블록 (8) 라벨.
  ⚠️ 첫 판 (tfctx7, train 1,280 / 1,322) 은 표본이 적었다 → 2026-10-27 train 8,000 클립 재실행 (tfctx7_full, train 3,658 / 3,799 · val 159 / 169) 이 정본.
     첫 판 그림은 figures/_superseded/predictor_info_by_layer_tf_last_context_n1280.png.
  /venv/vjepa/bin/python z_research/scripts/figures/plot_predictor_info_by_layer_tf_last_context.py [--prefix tfctx7_full]
출력: z_analysis/figures/predictor_info_by_layer_tf_last_context.png
"""
import argparse, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ap = argparse.ArgumentParser(); ap.add_argument("--prefix", default="tfctx7_full"); ap.add_argument("--out", default="predictor_info_by_layer_tf_last_context.png"); a = ap.parse_args()
R = "/workspace/world_model/vjepa2/z_analysis"
LAYERS = ["input"] + [str(i) for i in range(1, 13)] + ["norm"]; x = np.arange(14)
J = {lab: [json.load(open(f"{R}/results/attnprobe384/{a.prefix}_{lab}_L{l}.json"))["val_all"] for l in range(14)] for lab in ("own", "next")}
lab = np.load("/workspace/auto_research_cache/attnprobe384/val_lab.npy")
maj = np.bincount((lab[:, 0] * 8 + lab[:, 1]).astype(int)).max() / len(lab) * 100
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
for ax, (key, ylab, ylim) in zip(axes, (("shape_x_color", "shape × color accuracy (%)", (0, 100)), ("velocity_r2", "velocity R²", (-1.1, 1.0)))):
    for lb, nm, ls, c in (("own", "last context position (blk 7), label = blk 7 (its own input block)", "-", "#2a78d6"),
                          ("next", "last context position (blk 7), label = blk 8 (the block it predicts)", "--", "#eb6834")):
        ax.plot(x, [J[lb][l][key] for l in range(14)], ls, marker="o", color=c, lw=2, ms=4, label=nm)
    if key == "shape_x_color":
        ax.axhline(maj, color="black", ls=":", lw=1.0, label="majority class")
    else:
        ax.axhline(0, color="black", ls=":", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(LAYERS, fontsize=9); ax.set_ylim(*ylim); ax.set_ylabel(ylab)
for ax, pl in zip(axes, ("(a)", "(b)")):
    ax.set_xlabel("predictor layer (teacher forcing)\n" + pl)
fig.suptitle("Attentive probe on AR predictor hidden tokens at the last context position, 384-d per layer (256 × 384)", fontsize=12, y=0.995)
fig.tight_layout(rect=(0, 0.16, 1, 0.95))
hs, ls_ = [], []
for ax in axes:
    for h_, l_ in zip(*ax.get_legend_handles_labels()):
        if l_ not in ls_:
            hs.append(h_); ls_.append(l_)
fig.legend(hs, ls_, loc="lower center", ncol=2, fontsize=9, frameon=False)
fig.savefig(f"{R}/figures/{a.out}", dpi=170, bbox_inches="tight")
print("saved")
