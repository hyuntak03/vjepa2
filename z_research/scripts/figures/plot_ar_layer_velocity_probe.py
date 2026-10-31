"""AR predictor 층별 속도 probe 그림 (2026-10-26) — vobench_ar_layer_velocity_probe.py analyze 의 probe.json 을 그린다.

(1) fig_speed_traj: 걸음 공유 probe 로 읽은 속력 vs 걸음, 운동 4 종 × (문맥 끝 가림 아님 / 문맥 끝 가림). 참 · p · h · hc.
(2) fig_layer_t: (층, 걸음) 속력 R² (움직이는 clip, test 절반) 히트맵.

  python z_research/scripts/figures/plot_ar_layer_velocity_probe.py [--name vast70]
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
MOTS = ["flat_acc", "flat_dec", "ramp_down", "ramp_up"]
TITLE = {"flat_acc": "flat, speeding up", "flat_dec": "flat, slowing down", "ramp_down": "down ramp (speeding up)",
         "ramp_up": "up ramp (slowing down)"}


def main(name):
    d = REPO / f"z_analysis/ar_layer_velocity_probe/{name}"
    r = json.load(open(d / "probe.json"))
    fd = d / "figures"; fd.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42})
    t = np.arange(8)
    fig, ax = plt.subplots(2, 4, figsize=(11, 5.2), sharex=True)
    for i, occ in enumerate(["not_late", "late"]):
        for j, m in enumerate(MOTS):
            a = ax[i, j]; tr = r["shared"]["p"]["traj"][occ][m]
            a.plot(t, tr["true"], color="0.3", ls="--", lw=1.5, label="true")
            a.plot(t, tr["read"], color=C1, marker="o", ms=3, label="p (predicted future)")
            a.plot(t, r["shared"]["h(천장)"]["traj"][occ][m]["read"], color=C2, marker="s", ms=3, label="h (true future, encoder)")
            a.plot(t, r["shared"]["hc(복사)"]["traj"][occ][m]["read"], color=C3, lw=1.2, label="last context block (copy)")
            a.set_title(f"{TITLE[m]}\n{'context end visible' if occ == 'not_late' else 'context end occluded'} (n={tr['n']})", fontsize=8)
            if j == 0:
                a.set_ylabel("decoded speed (cm/s)")
            if i == 1:
                a.set_xlabel("future step t")
    ax[0, 0].legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(fd / "fig_speed_traj.png", dpi=150); fig.savefig(fd / "fig_speed_traj.pdf"); plt.close(fig)

    reps = [f"L{l}" for l in range(14)] + ["p", "h(천장)", "hc(복사)"]
    lab = [f"L{l}" for l in range(14)] + ["p", "h (true)", "copy"]
    M = np.array([[r["per_t"][k][str(s)]["speed_r2_moving"] for s in range(8)] for k in reps])
    fig, a = plt.subplots(figsize=(5, 5.5))
    im = a.imshow(M, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    for y in range(M.shape[0]):
        for x in range(8):
            a.text(x, y, f"{M[y, x]:.2f}", ha="center", va="center", fontsize=6, color="w" if M[y, x] < 0.6 else "k")
    a.set_xticks(range(8)); a.set_xticklabels([f"t{s}" for s in range(8)]); a.set_yticks(range(len(lab))); a.set_yticklabels(lab)
    a.set_xlabel("future step"); fig.colorbar(im, ax=a, label="speed R² (moving, held-out)")
    fig.tight_layout(); fig.savefig(fd / "fig_layer_t.png", dpi=150); fig.savefig(fd / "fig_layer_t.pdf"); plt.close(fig)
    print(fd)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--name", default="vast70"); main(ap.parse_args().name)
