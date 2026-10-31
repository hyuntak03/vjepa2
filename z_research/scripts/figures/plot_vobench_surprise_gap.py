#!/usr/bin/env python3
"""VOBench — surprise 차이 (gap = S(불가능) − S(가능), matched 쌍) 를 보임 vs 문맥 끝 가림 (late) 로, 정지 · 이동으로 나눠 (2026-10-25, 사용자 지시:
"visible 일 때랑 occ last 일 때랑 surprise score gap 차이 좀 보여줄래? moving / static 이거 나눠서").

gap > 0 이면 정답 (불가능이 더 놀랍다). 쌍 = block 안 문맥 일치 (pos_a, imp_ab) · (pos_b, imp_ba) — index 의 pair_id (A / B).
정지 = motion flat_static, 이동 = 나머지 다섯 (flat_vel · flat_acc · flat_dec · ramp_down · ramp_up).
모델 = release (V-JEPA 2 ViT-H), 학습 안 한 모델이라 VOBench 전체 (보임 144 쌍 · late 432 쌍). 결과 cvpr/results/eval/testbed/vobench_v1_vith/per_block.json.
상대 gap = gap / S(가능) 도 같이 낸다 (조건마다 surprise 크기가 달라서).

  python z_research/scripts/figures/plot_vobench_surprise_gap.py [--model vith]
산출 cvpr/results/eval/testbed/figures/vobench_surprise_gap_<model>.png + .json
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]
TB = REPO / "cvpr/results/eval/testbed"
OCC = [("visible", "visible", "#2a78d6"), ("late", "occluded at context end (late)", "#eb6834")]      # dataviz 범주 slot 1 · 2
VIOL = ["all", "permanence", "shape", "color"]
MOT = ["static", "moving"]


def gaps(model):
    """[(occlusion, motion, violation, objset, gap, S_pos)]"""
    S = json.load(open(TB / f"vobench_v1_{model}/per_block.json"))["per_video_surprise"]
    sc = lambda v: S[v] if isinstance(S[v], (int, float)) else S[v].get("all", list(S[v].values())[0])
    pairs = defaultdict(dict)
    for r in csv.DictReader(open(REPO / "data_csv/vobench_v1/index.csv")):
        pairs[(r["block_id"], r["pair_id"])][r["plausible"]] = r
    out = []
    for p in pairs.values():
        a, b = p["1"], p["0"]
        if a["video_id"] not in S:
            continue
        out.append((a["occlusion"], "static" if a["motion"] == "flat_static" else "moving", a["violation"], a["objset"],
                    sc(b["video_id"]) - sc(a["video_id"]), sc(a["video_id"])))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="vith"); a = ap.parse_args()
    G = gaps(a.model)
    out = TB / "figures"; out.mkdir(parents=True, exist_ok=True)
    table = {}
    fig, axes = plt.subplots(1, len(VIOL), figsize=(12, 3.9), sharey=True)
    for vi, v in enumerate(VIOL):
        ax = axes[vi]
        for oi, (occ, olab, col) in enumerate(OCC):
            data, pos = [], []
            for mi, m in enumerate(MOT):
                g = np.array([x[4] for x in G if x[0] == occ and x[1] == m and (v == "all" or x[2] == v)])
                rel = np.array([x[4] / x[5] for x in G if x[0] == occ and x[1] == m and (v == "all" or x[2] == v)])
                table[f"{v}/{m}/{occ}"] = {"n": len(g), "acc": round(100 * float(np.mean(g > 0)), 1), "median_gap": round(float(np.median(g)), 4),
                                           "mean_gap": round(float(g.mean()), 4), "median_rel_gap_pct": round(100 * float(np.median(rel)), 2)}
                data.append(g); pos.append(mi + (oi - 0.5) * 0.36)
            bp = ax.boxplot(data, positions=pos, widths=0.3, patch_artist=True, showfliers=True, medianprops=dict(color="black", lw=1.2),
                            flierprops=dict(marker=".", ms=3, mfc=col, mec=col, alpha=0.6), whiskerprops=dict(color="#555555", lw=0.8),
                            capprops=dict(color="#555555", lw=0.8), boxprops=dict(lw=0.6, edgecolor="#555555"))
            for b in bp["boxes"]:
                b.set_facecolor(col); b.set_alpha(0.85)
            if vi == 0:
                bp["boxes"][0].set_label(olab)
        ax.axhline(0, color="#888888", lw=0.8, ls="--")
        ax.set_xticks(range(len(MOT)))
        ax.set_xticklabels([f"{m}\nn={table[f'{v}/{m}/visible']['n']} / {table[f'{v}/{m}/late']['n']}" for m in MOT], fontsize=8.5)
        ax.set_title(v if v != "all" else "all violations", fontsize=10)
        ax.grid(axis="y", color="#e5e5e5", lw=0.6); ax.set_axisbelow(True)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        if vi == 0:
            ax.set_ylabel("surprise gap  S(impossible) − S(possible)")
    fig.legend(loc="upper center", ncol=2, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out / f"vobench_surprise_gap_{a.model}.png", dpi=170)
    json.dump({"model": a.model, "gap": "S(impossible) - S(possible), matched pairs; acc = gap > 0; n = visible / late", "cells": table},
              open(out / f"vobench_surprise_gap_{a.model}.json", "w"), indent=1, ensure_ascii=False)
    for k, x in table.items():
        print(f"{k:28s} n {x['n']:4d}  acc {x['acc']:5.1f}  median gap {x['median_gap']:+.4f}  mean {x['mean_gap']:+.4f}  rel {x['median_rel_gap_pct']:+.2f}%")
    print("→", out / f"vobench_surprise_gap_{a.model}.png")


if __name__ == "__main__":
    main()
