#!/usr/bin/env python3
"""실험 6 그림 — 문맥 encoder 의 좌/우 운동 방향 (정방향 학습 → 역재생 시험).

입력: z_research/context_encoder_analysis/exp_results/encoder_temporal_dynamics/{ssv2,ntu}/results.json (+ summary.json)
출력 (figures/encoder_temporal_dynamics/):
  fig_direction_confusion.{png,pdf}  (a)~(d) 2x2 혼동행렬 4 개 — 데이터셋 2 x {정방향, 역재생}. attentive probe
  fig_direction_summary.{png,pdf}    probe·데이터셋별 정방향 정확도 / 역재생 정확도 (뒤집힌 라벨) / 뒤집힘 비율

수치는 실행 때마다 results.json 에서 다시 읽는다 (문서 표와 대조용으로 stdout 에도 찍는다).
라벨은 영어 (한글 폰트 없음, CLAUDE.md §8-4). 색은 검증된 팔레트 slot 1~3 만.

  python z_research/scripts/figures/plot_ctxenc_direction.py
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
RES = ROOT / "z_research/context_encoder_analysis/exp_results/encoder_temporal_dynamics"
FIG = ROOT / "z_research/context_encoder_analysis/figures/encoder_temporal_dynamics"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
SETS = [("ssv2", "ssv2_VP"), ("ntu", "NTU")]
CLASSES = ["left", "right"]

fam = [f.name for f in matplotlib.font_manager.fontManager.ttflist]
plt.rcParams.update({
    "font.family": "Nimbus Roman" if "Nimbus Roman" in fam else "DejaVu Sans",
    "font.size": 8, "pdf.fonttype": 42, "axes.linewidth": 0.6,
})
CMAP = LinearSegmentedColormap.from_list("b", ["#ffffff", BLUE])


def load():
    out = {}
    for k, _ in SETS:
        out[k] = json.loads((RES / k / "results.json").read_text())
    return out


def panel_conf(ax, m, title, sub):
    m = np.asarray(m, float)
    frac = m / np.maximum(m.sum(1, keepdims=True), 1)
    ax.imshow(frac, cmap=CMAP, vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{int(m[i, j])}\n{100 * frac[i, j]:.0f}%", ha="center", va="center",
                    fontsize=8, color="white" if frac[i, j] > 0.55 else "#222222")
    ax.set_xticks([0, 1], CLASSES)
    ax.set_yticks([0, 1], CLASSES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true direction")
    ax.set_title(f"{title}\n{sub}", fontsize=8)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)


def fig_confusion(rep):
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.25))
    labs = "(a) (b) (c) (d)".split()
    k = 0
    for name, nice in SETS:
        r = rep[name]["probes"]["attentive"]
        for which, key, acc in (("forward", "confusion_fwd", r["fwd_test_acc"]),
                                ("reversed", "confusion_rev", r["rev_test_acc_flipped_label"])):
            panel_conf(axes[k], r[key], f"{nice} — {which}", f"accuracy {100 * acc:.1f}%")
            axes[k].text(0.5, -0.42, labs[k], transform=axes[k].transAxes, ha="center", fontsize=9)
            k += 1
    fig.text(0.5, -0.02,
             "Attentive probe trained on forward clips only; ssv2_VP = Something-Something direction split, NTU = NTU direction benchmark. "
             "In the reversed panels the true label is the flipped one, because a clip labelled 'left' played backwards moves right. "
             "Cell text: count and row share.",
             ha="center", fontsize=7)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"fig_direction_confusion.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_summary(rep):
    rows = [(f"{n}\n{p}", rep[n]["probes"][p]) for n, _ in SETS for p in ("attentive", "tiny")]
    x = np.arange(len(rows))
    w = 0.27
    series = [("forward test accuracy", "fwd_test_acc", BLUE),
              ("reversed test accuracy (flipped label)", "rev_test_acc_flipped_label", GREEN),
              ("prediction flip rate (no labels)", "flip_rate", ORANGE)]
    fig, ax = plt.subplots(figsize=(7.0, 2.4))
    for j, (lab, key, col) in enumerate(series):
        v = [100 * r[key] for _, r in rows]
        ax.bar(x + (j - 1) * w, v, w * 0.94, color=col, label=lab)
        for xi, vi in zip(x + (j - 1) * w, v):
            ax.text(xi, vi + 1.5, f"{vi:.1f}", ha="center", fontsize=6.5, color=col)
    ax.axhline(50, color="k", lw=0.6, ls=":")
    ax.text(len(rows) - 0.45, 51.5, "chance", fontsize=6.5, ha="right")
    ax.set_xticks(x, [r for r, _ in rows], fontsize=7.5)
    ax.set_ylim(0, 108)
    ax.set_ylabel("%")
    ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.20))
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.5, 0.005,
             "Blue and orange are the independent measurements; green is derived from them (a forward error that does not flip counts as correct there). "
             "A probe whose answer depends on temporal order flips when the clip is played backwards (orange near 100); one reading order-free scene cues would not (orange near 0).",
             ha="center", fontsize=7)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"fig_direction_summary.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    rep = load()
    print("results.json 에서 다시 읽은 값:")
    for n, _ in SETS:
        for p, r in rep[n]["probes"].items():
            print(f"  {n:5s} {p:9s} fwd {100*r['fwd_test_acc']:5.1f}  rev(flipped) {100*r['rev_test_acc_flipped_label']:5.1f}"
                  f"  flip {100*r['flip_rate']:5.1f}  conf_fwd {r['confusion_fwd']}  conf_rev {r['confusion_rev']}")
    fig_confusion(rep)
    fig_summary(rep)
    print(f"→ {FIG}/fig_direction_confusion.png, fig_direction_summary.png (+pdf)")


if __name__ == "__main__":
    main()
