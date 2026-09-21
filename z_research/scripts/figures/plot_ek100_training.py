#!/usr/bin/env python3
"""EK100 probe 학습 곡선 — 규약별 arm 을 겹쳐 그린다 (수렴 여지 판단용).

입력: <run>/log_r0.csv (epoch 별 train/val, nan = 그 epoch 에 val 안 함) · <run>/val_metrics.jsonl (exact val, head 중 max)
출력: z_research/anticipation/EK100/figures/training/fig_ek100_training.{png,pdf}
  (a) action mean-class recall@5 — train (선) 과 val (표식)
  (b) val verb / noun recall

수치는 실행 때마다 산출물에서 다시 읽고 stdout 에 epoch 별 증분까지 찍는다 (수렴 판단).
⚠️ 두 arm 은 **다른 과제**다 (released = 문맥이 action 끝 1 s 전, paper = 시작 1 s 전). 같은 축에 두는 것은 학습 곡선의 모양 비교용이다.

  python z_research/scripts/figures/plot_ek100_training.py
  python z_research/scripts/figures/plot_ek100_training.py --runs ek100_vith_official256_paper_grid8
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
EXP = ROOT / "z_research/anticipation/EK100/exp_results/action_anticipation_frozen"
FIG = ROOT / "z_research/anticipation/EK100/figures/training"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
RUNS = [("ek100_vith_official256_released_grid8", "released anchor (official code)", BLUE),
        ("ek100_vith_official256_paper_grid8", "paper anchor (our branch)", ORANGE)]

fam = [f.name for f in matplotlib.font_manager.fontManager.ttflist]
plt.rcParams.update({"font.family": "Nimbus Roman" if "Nimbus Roman" in fam else "DejaVu Sans",
                     "font.size": 8, "pdf.fonttype": 42, "axes.linewidth": 0.6})


def load(tag):
    rows = list(csv.DictReader((EXP / tag / "log_r0.csv").open()))
    ep = np.array([int(r["epoch"]) for r in rows])
    tr = np.array([float(r["train-recall"]) for r in rows])
    val = {"action": {}, "verb": {}, "noun": {}}
    p = EXP / tag / "val_metrics.jsonl"
    if p.is_file():
        for line in p.read_text().strip().split("\n"):
            r = json.loads(line)
            if r.get("val_only"):
                continue
            for k in val:
                val[k][int(r["epoch"])] = r[k]["recall"]
    return ep, tr, val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=[t for t, _, _ in RUNS])
    a = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    sel = [(t, lab, c) for t, lab, c in RUNS if t in a.runs]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    for tag, lab, col in sel:
        ep, tr, val = load(tag)
        print(f"== {tag}  ({len(ep)} epoch 기록)")
        ve = sorted(val["action"])
        prev = None
        for e in ve:
            d = "" if prev is None else f"  (+{val['action'][e] - val['action'][prev]:.2f})"
            print(f"   epoch {e:2d}  val action {val['action'][e]:5.2f}{d}   verb {val['verb'][e]:5.2f}  noun {val['noun'][e]:5.2f}")
            prev = e
        print(f"   train action recall: epoch 1 {tr[0]:.2f} → {ep[-1]} {tr[-1]:.2f}")

        axes[0].plot(ep, tr, color=col, lw=1.4, label=f"{lab} — train")
        if ve:
            axes[0].plot(ve, [val["action"][e] for e in ve], color=col, lw=1.4, ls="--",
                         marker="o", ms=4, mfc="white", label=f"{lab} — val")
        for k, ls, mk in (("verb", "-", "o"), ("noun", "--", "s")):
            if ve:
                axes[1].plot(ve, [val[k][e] for e in ve], color=col, lw=1.4, ls=ls, marker=mk, ms=4,
                             mfc="white", label=f"{lab.split(' (')[0]} — {k}")
    axes[0].set_title("action mean-class recall@5", fontsize=9)
    axes[1].set_title("val verb / noun mean-class recall@5", fontsize=9)
    for ax, lb in zip(axes, ["(a)", "(b)"]):
        ax.set_xlabel("epoch")
        ax.set_ylabel("%")
        ax.set_xlim(0, 21)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=6.2, frameon=False, loc="upper left")
        ax.text(0.5, -0.40, lb, transform=ax.transAxes, ha="center", fontsize=9)
    fig.text(0.5, -0.06,
             "Two arms are different tasks: released = context ends 1 s before the action ends, paper = 1 s before it starts. "
             "Overlaid to compare the shape of the curve, not the level. Validation every epoch (released) or every 5 (paper); cosine schedule over 20 epochs.",
             ha="center", fontsize=7)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"fig_ek100_training.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"→ {FIG}/fig_ek100_training.png (+pdf)")


if __name__ == "__main__":
    main()
