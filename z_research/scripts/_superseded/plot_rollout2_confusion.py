#!/usr/bin/env python3
"""RollOut_v2 실험 2 — possible/impossible probe 의 confusion matrix 두 장.
  fig_confusion_h.png    입력 = target encoder 32 프레임 (학습 분포)
  fig_confusion_zp.png   입력 = [context encoder 16 프레임 ; predictor 최종 출력 16 프레임]
패널 = ledge / wall. probe 는 h 32 프레임으로 학습 (probe_pos_imp.json, train_ctx=h_ctx).
"""
import json
from pathlib import Path
import matplotlib as mpl, matplotlib.pyplot as plt, numpy as np

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
R = json.loads((ROOT / "z_research/RollOutV2/exp_results/probe_pos_imp.json").read_text())
OUT = ROOT / "z_research/RollOutV2/figures/probe"; OUT.mkdir(parents=True, exist_ok=True)
mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "pdf.fonttype": 42})
LAB = {"ledge": ("falls", "floats"), "wall": ("stops", "passes")}

for tag, key, title in (("h", "[h_ctx ; h_fut]", "input: target encoder, 32 frames"), ("zp", "[z ; p]", "input: context encoder 16 frames ; predictor output 16 frames")):
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.1))
    for ax, scn in zip(axes, ("ledge", "wall")):
        r = R["scenarios"][scn]["runs"]["train_ctx=h_ctx"][key]; n = R["scenarios"][scn]["n_test"] // 2
        tp = round(r["acc_on_possible"] * n); ti = round(r["acc_on_impossible"] * n)
        M = np.array([[tp, n - tp], [n - ti, ti]])                      # 행 = 정답 (possible, impossible), 열 = 예측
        ax.imshow(M, cmap="Blues", vmin=0, vmax=n)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{M[i, j]}\n({100*M[i, j]/n:.0f}%)", ha="center", va="center", fontsize=9, color="white" if M[i, j] > n / 2 else "black")
        pos, imp = LAB[scn]
        ax.set_xticks([0, 1]); ax.set_xticklabels([f"possible\n({pos})", f"impossible\n({imp})"], fontsize=8)
        ax.set_yticks([0, 1]); ax.set_yticklabels([f"possible\n({pos})", f"impossible\n({imp})"], fontsize=8)
        ax.set_xlabel("predicted", fontsize=9); ax.set_ylabel("true", fontsize=9)
        ax.set_title(f"{scn}   acc {100*r['acc']:.0f}%", fontsize=9.5)
    fig.suptitle(title, fontsize=9.5, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_confusion_{tag}.{ext}", dpi=300)
    plt.close(fig); print(f"  [saved] probe/fig_confusion_{tag}.png")
