#!/usr/bin/env python3
"""ledge / wall pos·imp probe (rollout2_probe_pos_imp.py) 의 confusion matrix 를 test 입력 4종 × 시나리오 2 로 한 장에.

입력: exp_results/probe_pos_imp/probe_pos_imp.json (acc_on_possible / acc_on_impossible 과 n_test 로 칸 수를 복원; 각 클래스 n_test/2).
출력: figures/probe_pos_imp/fig_confusion_all_<train_ctx>.png/.pdf  (train_ctx = h_ctx, z)
  python z_research/scripts/figures/plot_probe_pos_imp_confusion.py
"""
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2/z_research/RollOutV2")
J = ROOT / "exp_results/probe_pos_imp/probe_pos_imp.json"; FIG = ROOT / "figures/probe_pos_imp"
TESTS = ("[h_ctx ; h_fut]", "[z ; h_fut]", "[z ; p]", "[h_ctx ; p]")
TTL = {"[h_ctx ; h_fut]": "[h ctx ; h future]\n(train distribution)", "[z ; h_fut]": "[z ctx ; h future]", "[z ; p]": "[z ctx ; predictor p]", "[h_ctx ; p]": "[h ctx ; predictor p]"}
LAB = {"ledge": ["falls\n(possible)", "floats\n(impossible)"], "wall": ["stops\n(possible)", "passes\n(impossible)"]}
plt.rcParams.update({"font.family": "Nimbus Roman", "pdf.fonttype": 42, "font.size": 8})


def main():
    R = json.load(open(J)); FIG.mkdir(parents=True, exist_ok=True)
    for ctx in ("h_ctx", "z"):
        fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.9))
        for i, scn in enumerate(("ledge", "wall")):
            S = R["scenarios"][scn]; n = S["n_test"] // 2; runs = S["runs"][f"train_ctx={ctx}"]
            for j, name in enumerate(TESTS):
                r = runs[name]; ax = axes[i, j]
                cm = np.array([[r["acc_on_possible"] * n, (1 - r["acc_on_possible"]) * n], [(1 - r["acc_on_impossible"]) * n, r["acc_on_impossible"] * n]]).round().astype(int)
                ax.imshow(cm, cmap="Blues", vmin=0, vmax=n)
                for a in range(2):
                    for b in range(2):
                        ax.text(b, a, f"{cm[a, b]}\n({100 * cm[a, b] / n:.0f}%)", ha="center", va="center", fontsize=7.5, color="white" if cm[a, b] > n / 2 else "black")
                ax.set_xticks([0, 1]); ax.set_xticklabels(["possible", "impossible"], fontsize=6.5); ax.set_yticks([0, 1]); ax.set_yticklabels(LAB[scn] if j == 0 else ["", ""], fontsize=7)
                if i == 0:
                    ax.set_title(TTL[name], fontsize=8)
                if i == 1:
                    ax.set_xlabel("probe output", fontsize=7)
                ax.text(1.02, 0.5, f"acc {100 * r['acc']:.0f}%\nP(imp) {r['mean_P_impossible_on_pos_clips']:.2f}" if name in ("[z ; p]",) else f"acc {100 * r['acc']:.0f}%",
                        transform=ax.transAxes, fontsize=6.5, va="center", ha="left")
            axes[i, 0].set_ylabel(f"{scn}\ntrue clip", fontsize=8)
        fig.suptitle(f"pos/imp probe trained on target encoder h (32 frames), context source = {ctx}; test inputs below.  [z ; p] shares p across the pair → acc 50% by construction", fontsize=7.5)
        fig.tight_layout(rect=(0, 0, 0.97, 0.95))
        fig.savefig(FIG / f"fig_confusion_all_{ctx}.png", dpi=200); fig.savefig(FIG / f"fig_confusion_all_{ctx}.pdf"); plt.close(fig)
    print(f"→ {FIG}/fig_confusion_all_{{h_ctx,z}}.png")


if __name__ == "__main__":
    main()
