#!/usr/bin/env python3
"""SSv2 시간 궤적 모양 그림 (2026-10-15) — z_research/scripts/analysis/ssv2_straightness.py 의 per_video.npz 를 읽는다.

영상마다 그 영상의 16 시점 z_t 를 PCA 한 앞 3 축 좌표 (공간마다 따로) 를 그린다. 색 = 시간 (어두움 → 밝음).
  fig_traj_2d.png : 영상 8 개 × 공간 4 개 (pixel · enc_mean · tgt_mean · enc_patch), PC1–PC2, 패널 제목에 PC1/PC2 분산 비율 · θ · R²_line
  fig_traj_3d.png : 같은 것을 PC1–PC2–PC3 3D 로
  fig_summary.png : 영상 200 개 분포 — θ · R²_line · R²_quad · PC1 분산 비율, 실제 순서 vs 섞은 순서
  python z_research/scripts/figures/plot_ssv2_straightness.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]
D = REPO / "z_research/VOBench_v1/vjepa2/ssv2_straightness"
SP = [("pixel", "pixel (64x64 gray)"), ("enc_mean", "context enc · spatial mean"), ("tgt_mean", "target enc · spatial mean"), ("enc_patch", "context enc · all patches")]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})


def main():
    z = np.load(D / "per_video.npz", allow_pickle=True); n = len(z["paths"]); pick = np.linspace(0, n - 1, 8).round().astype(int)
    cm = plt.get_cmap("viridis"); T = z["pixel_pca"].shape[1]; col = cm(np.linspace(0, 1, T))
    for dim in (2, 3):
        fig = plt.figure(figsize=(7.0 * 1.6, 7.0 * 1.6 * 8 / 4 / 1.25))
        for r, v in enumerate(pick):
            for c, (k, lab) in enumerate(SP):
                P = z[f"{k}_pca"][v]; ev = z[f"{k}_evr"][v]
                ax = fig.add_subplot(8, 4, r * 4 + c + 1, projection="3d" if dim == 3 else None)
                if dim == 2:
                    ax.plot(P[:, 0], P[:, 1], color="0.7", lw=0.8, zorder=1); ax.scatter(P[:, 0], P[:, 1], c=col, s=14, zorder=2)
                    ax.set_aspect("equal", adjustable="datalim")
                else:
                    ax.plot(P[:, 0], P[:, 1], P[:, 2], color="0.7", lw=0.8); ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=col, s=12, depthshade=False)
                    ax.view_init(elev=22, azim=-55)
                ax.set_xticks([]); ax.set_yticks([])
                if dim == 3: ax.set_zticks([])
                th = z[f"{k}_real_theta"][v]; rl = z[f"{k}_real_r2_line"][v]
                ax.set_title(f"{lab if r == 0 else ''}\nPC1 {100*ev[0]:.0f}% PC2 {100*ev[1]:.0f}% | θ {th:.0f}° R²line {rl:.2f}" if r == 0 else
                             f"PC1 {100*ev[0]:.0f}% PC2 {100*ev[1]:.0f}% | θ {th:.0f}° R²line {rl:.2f}", fontsize=6.5)
                if c == 0: ax.text2D(-0.15, 0.5, Path(str(z["paths"][v])).stem, transform=ax.transAxes, rotation=90, va="center", fontsize=7) if dim == 3 else ax.set_ylabel(Path(str(z["paths"][v])).stem, fontsize=7)
        fig.tight_layout(); fig.savefig(D / f"fig_traj_{dim}d.png", dpi=130); plt.close(fig)
    fig, axs = plt.subplots(1, 4, figsize=(7.0 * 1.6, 2.6))
    for ax, (key, lab) in zip(axs, [("theta", "mean turning angle θ (deg)"), ("r2_line", "R² line"), ("r2_quad", "R² quadratic"), ("evr", "PC1 variance ratio")]):
        for i, (k, nm) in enumerate(SP):
            if key == "evr":
                vals = [z[f"{k}_evr"][:, 0]]; labs = ["real"]
            else:
                vals = [z[f"{k}_real_{key}"], z[f"{k}_shuffle_{key}"]]; labs = ["real", "shuffled"]
            for j, (v, l) in enumerate(zip(vals, labs)):
                pos = i * 3 + j
                ax.boxplot(v, positions=[pos], widths=0.8, showfliers=False, patch_artist=True,
                           boxprops=dict(facecolor=["#2a78d6", "#eb6834"][j], alpha=0.6), medianprops=dict(color="k"))
        ax.set_xticks([i * 3 + 0.5 for i in range(4)]); ax.set_xticklabels([s[0] for s in SP], rotation=20); ax.set_title(lab)
    axs[0].legend(handles=[plt.Rectangle((0, 0), 1, 1, color="#2a78d6", alpha=0.6), plt.Rectangle((0, 0), 1, 1, color="#eb6834", alpha=0.6)], labels=["real order", "shuffled order"], fontsize=7)
    fig.tight_layout(); fig.savefig(D / "fig_summary.png", dpi=150); plt.close(fig)
    print("saved", D)


if __name__ == "__main__":
    main()
