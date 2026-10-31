"""h 로 배운 위치 자를 AR p 에 건 결과 그림 (2026-10-26) — vobench_ar_token_transfer.py analyze 의 transfer.json.

윗줄: 운동 축 위 진행 (칸) vs 걸음 — 참 · h (자 검사) · p 기대값 · p argmax · p 본 맞춤 (파라미터 없음).
아랫줄: 자의 최대 질량 — h vs p (균등 = 1/256).

  python z_research/scripts/figures/plot_ar_token_transfer.py [--name vast70]
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
MOTS = ["flat_vel", "flat_acc", "flat_dec", "ramp_down", "ramp_up"]
TITLE = {"flat_vel": "flat, constant", "flat_acc": "flat, speeding up", "flat_dec": "flat, slowing down",
         "ramp_down": "down ramp", "ramp_up": "up ramp"}


def main(name):
    d = REPO / f"z_analysis/ar_token_transfer/{name}"
    r = json.load(open(d / "transfer.json"))["groups"]
    fd = d / "figures"; fd.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42})
    t = np.arange(8)
    for occ in ("not_late", "late"):
        fig, ax = plt.subplots(2, 5, figsize=(13, 5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
        for j, m in enumerate(MOTS):
            g = r[f"{occ}/{m}"]; a = ax[0, j]
            a.plot(t, g["true_prog"], color="0.3", ls="--", lw=1.5, label="true")
            a.plot(t, g["h_prog"], color=C2, marker="s", ms=3, label="h (true future, readout check)")
            a.plot(t, g["p_prog"], color=C1, marker="o", ms=3, label="p, readout expectation")
            a.plot(t, g["pa_prog"], color=C1, ls=":", marker="^", ms=3, label="p, readout argmax")
            a.plot(t, g["pt_prog"], color=C3, marker="d", ms=3, label="p, template match (no params)")
            a.axhline(0, color="0.7", lw=0.8)
            a.set_title(f"{TITLE[m]} (n={g['n']})", fontsize=8)
            b = ax[1, j]
            b.plot(t, g["h_mass"], color=C2, marker="s", ms=3, label="h")
            b.plot(t, g["p_mass"], color=C1, marker="o", ms=3, label="p")
            b.set_ylim(0, 0.35); b.set_xlabel("future step t")
        ax[0, 0].set_ylabel("progress along motion (cells)"); ax[1, 0].set_ylabel("readout peak mass")
        ax[0, 0].legend(fontsize=6.5, frameon=False); ax[1, 0].legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(fd / f"fig_progress_{occ}.png", dpi=150); fig.savefig(fd / f"fig_progress_{occ}.pdf"); plt.close(fig)
    print(fd)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--name", default="vast70"); main(ap.parse_args().name)
