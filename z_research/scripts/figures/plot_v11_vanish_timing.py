#!/usr/bin/env python3
"""v11 전체 (flat / ramp / static × late / early / mid × k) 위치 readout 을 한 표·한 그림으로 모은다.

입력: `plot_v11_vanish_readout.py` 가 저장한 figures/<train>/v11_vanish/<rep>/<motion>[_early|_mid]/k<k>/readout.npz
      (pred (n, n_tub, 2) px, truth (n, 16, 2) px 튜블릿 평균, last (n, 2) = 튜블릿 7 진실).
지표 (미래 8 튜블릿, 진실은 pos_a):
  ratio    = 슬롯 7 에서 (읽기 − 마지막 관측) / (진실 − 마지막 관측), 진행 방향 x.  static 은 정의 안 됨 (진실 변위 0)
  L2       = |읽기 − 진실| 8 슬롯 평균 (px; 1 칸 = 18 px)
  L2_last  = |읽기 − 마지막 관측 위치| 8 슬롯 평균 (px)  — 얼마나 옮겼나
  alpha    = 슬롯 7 에서 가림막 기준 물체 쪽 비율 (1 = 물체, 0 = 가림막). 가림막 x = 가려진 구간 한가운데 샘플의 물체 x (물체가 그 뒤에 있으므로).
             moving × k≥1 만.
출력: figures/<train>/v11_vanish/timing/{timing_summary.json, timing_summary.md, fig_timing.png/.pdf}

  ROLLOUT2_TRAIN=v5 python z_research/scripts/figures/plot_v11_vanish_timing.py
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
sys.path.insert(0, str(ROOT / "z_research/scripts/analysis"))
import rollout2_test_readout as rt                                 # noqa: E402
FIG = rt.FIG_ROOT / "v11_vanish"; OUT = FIG / "timing"
META = Path("/local_datasets/world/world_analysis/IntPhysGen_v11/metadata.csv")
REPS = ("p", "z"); MOTIONS = ("flat", "ramp", "static"); TIMINGS = ("late", "early", "mid"); KS = ("0", "1", "2", "3", "4")
COL = {"p": "#eb6834", "z": "#1baf7a"}; LS = {"late": "-", "mid": "--", "early": ":"}; CELL = 18.0
plt.rcParams.update({"font.family": "Nimbus Roman", "pdf.fonttype": 42, "font.size": 8})


def load(rep, motion, timing, K):
    d = FIG / rep / (motion if timing == "late" else f"{motion}_{timing}") / f"k{K}" / "readout.npz"
    return np.load(d, allow_pickle=True) if d.exists() else None


def stats(z, meta, motion, K):
    pred, truth, last, blocks = z["pred"][:, -8:], z["truth"], z["last"], z["block"]
    fut = truth[:, 8:]; n = len(blocks)
    l2 = float(np.linalg.norm(pred - fut, axis=-1).mean()); l2_last = float(np.linalg.norm(pred - last[:, None], axis=-1).mean())
    rec = {"n": n, "l2": l2, "l2_last": l2_last, "ratio": None, "alpha": None, "bias_x": None}
    if motion == "static":
        return rec
    sgn = np.sign(fut[:, 7, 0] - last[:, 0]); dt = (fut[:, :, 0] - last[:, None, 0]) * sgn[:, None]; dp = (pred[:, :, 0] - last[:, None, 0]) * sgn[:, None]
    rec["ratio"] = float(dp[:, 7].mean() / dt[:, 7].mean()); rec["bias_x"] = [float(v) for v in (dp - dt).mean(0)]
    if K != "0":
        occ = np.empty(n)
        for i, b in enumerate(blocks):
            m = meta[b]; s0, s1 = int(m["hidden_start"]) // 3, int(m["hidden_end"]) // 3      # raw frame → 32-sample index (stride 3)
            occ[i] = rt.arr(m["object_px_x_by_sample"])[(s0 + s1) // 2]
        rec["alpha"] = float(((pred[:, 7, 0] - occ) / (fut[:, 7, 0] - occ)).mean())
    return rec


def main():
    idx = ROOT / "data_csv/intphysgen_v11_full/index_probe.csv"
    blk2vid = {r["block_id"]: r["video_id"] for r in csv.DictReader(idx.open()) if r["variant"] == "pos_a"}          # block_id (숫자) → pos_a video_id
    meta_v = {r["name"]: r for r in csv.DictReader(META.open())}; meta = {b: meta_v[v] for b, v in blk2vid.items() if v in meta_v}
    res = {}
    for rep in REPS:
        for motion in MOTIONS:
            for timing in TIMINGS:
                for K in KS:
                    if K == "0" and timing != "late":
                        continue
                    z = load(rep, motion, timing, K)
                    if z is None:
                        print(f"missing {rep} {motion} {timing} k{K}"); continue
                    res[f"{rep}/{motion}/{timing}/k{K}"] = stats(z, meta, motion, K)
    OUT.mkdir(parents=True, exist_ok=True); json.dump(res, open(OUT / "timing_summary.json", "w"), indent=1)

    g = lambda rep, m, t, K, key: res.get(f"{rep}/{m}/{t}/k{K}", {}).get(key)
    f = lambda v, d=2: "–" if v is None else f"{v:.{d}f}"
    md = ["# v11 전체 위치 readout — timing × k (학습셋 v5 자, 가능 clip pos_a, 미래 8 tubelet)", "",
          "late = 문맥 끝 k 장 + 미래 앞 k 장 가림 (v11 본체), early = raw 12~ k 장 (문맥 안에서만), mid = raw 21~ k 장 (문맥 안에서만). early/mid 는 **미래에서 물체가 다 보인다**.",
          "k=0 = 가림막 없음 (n 224), k≥1 각 n 56. 1 칸 = 18 px. ratio = 슬롯 7 진행 비율 (읽은 변위 / 진실 변위). alpha = 슬롯 7 에서 가림막→물체 비율 (1 물체, 0 가림막).", ""]
    for key, title, d in (("ratio", "슬롯 7 진행 비율 (moving 만)", 2), ("l2", "L2 진실 (px, 8 슬롯 평균)", 1), ("l2_last", "L2 마지막 관측 위치 (px) — 얼마나 옮겼나", 1), ("alpha", "alpha (가림막 0 → 물체 1, 슬롯 7, moving k≥1)", 2)):
        md += [f"## {title}", "", "| rep | motion | timing | k=0 | k=1 | k=2 | k=3 | k=4 |", "|---|---|---|---|---|---|---|---|"]
        for rep in REPS:
            for m in MOTIONS:
                if key in ("ratio", "alpha") and m == "static":
                    continue
                for t in TIMINGS:
                    md.append(f"| {rep} | {m} | {t} | " + " | ".join(f(g(rep, m, t, K, key), d) for K in KS) + " |")
        md.append("")
    md += ["## 슬롯별 x 편향 (읽기 − 진실, 진행 방향, px), k=4", "", "| rep | motion | timing | " + " | ".join(f"s{i}" for i in range(8)) + " |", "|---|---|---|" + "---|" * 8]
    for rep in REPS:
        for m in ("flat", "ramp"):
            for t in TIMINGS:
                b = g(rep, m, t, "4", "bias_x")
                if b:
                    md.append(f"| {rep} | {m} | {t} | " + " | ".join(f"{v:+.0f}" for v in b) + " |")
    (OUT / "timing_summary.md").write_text("\n".join(md) + "\n"); print("\n".join(md))

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.7)); x = np.arange(5)
    for ax, m in zip(axes, MOTIONS):
        key = "l2" if m == "static" else "ratio"
        for rep in REPS:
            for t in TIMINGS:
                y = [g(rep, m, t, K, key) for K in KS]
                if t != "late":
                    y[0] = g(rep, m, "late", "0", key)                                     # k=0 은 공통 (가림막 없음)
                y = [np.nan if v is None else (v / CELL if key == "l2" else v) for v in y]
                ax.plot(x, y, LS[t], color=COL[rep], lw=1.4, marker="o", ms=3, label=f"{rep}, {t}")
        ax.set_xticks(x); ax.set_xticklabels(["0\n(visible)", "1", "2", "3", "4"]); ax.set_xlabel("hidden frames k")
        ax.set_title({"flat": "flat, constant velocity", "ramp": "ramp, constant acceleration", "static": "static"}[m], fontsize=8)
        ax.set_ylabel("read / true displacement (slot 7)" if key == "ratio" else "L2 to true position (cells)")
        if key == "ratio":
            ax.set_ylim(0, 1); ax.axhline(1, color="k", lw=0.5, ls=":")
        else:
            ax.set_ylim(0, None); ax.axhline(1, color="k", lw=0.5, ls=":")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=6, frameon=False, ncol=2, loc="center right", bbox_to_anchor=(1.0, 0.5))
    for i, ax in enumerate(axes):
        ax.text(0.5, -0.5, f"({'abc'[i]})", transform=ax.transAxes, ha="center", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fig_timing.png", dpi=200, bbox_inches="tight"); fig.savefig(OUT / "fig_timing.pdf", bbox_inches="tight"); print(f"→ {OUT}")


if __name__ == "__main__":
    main()
