#!/usr/bin/env python3
"""IntPhys 2 late occ (labels_final) — 카메라 움직임 (Moving / Fixed) 마다 그림 한 장, x = 위반 종류, 막대 = release · AR (vast) (2026-10-25, 사용자 지시:
"camera moving / camera static 이렇게 이미지 파일 따로 만들고 … 한 이미지당 violation type 별로 acc … ar(vast), release").

채점 = 시트 IntPhys2 late occ 열과 같다 (cvpr/eval/intphys2_late/collect.py): 쌍 = 같은 장면의 (k 가능, k 불가능), 창 하나 (문맥 16 = 위반 시작 앞 uniform),
  S(imp) > S(pos) 이면 정답 (동점 0.5). 두 모델 모두 IntPhys 2 로 학습하지 않아 500 쌍 전부.
카메라 라벨 = IntPhys 2 metadata 의 Camera 열 (index 의 Camera; 쌍의 두 영상이 같은지 확인한다). 물체 운동 라벨은 IntPhys 2 에 없다.

  python z_research/scripts/figures/plot_intphys2_late_camera.py
산출 cvpr/results/eval/intphys2_late/figures/intphys2_late_final_camera_{moving,static}.png + intphys2_late_final_camera.json
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]
DS = "intphys2_late_final"
RES = REPO / "cvpr/results/eval/intphys2_late"
OUT = RES / "figures"
MODELS = [("release (V-JEPA 2 ViT-H)", "vith", "#2a78d6"), ("AR (vast)", "vith_vast_ar_state_ep70", "#eb6834")]   # dataviz 범주 slot 1 · 2
VIOL = ["permanence", "immutability", "continuity", "solidity"]
CAMS = [("Moving", "moving"), ("Fixed", "static")]                   # metadata 값 → 그림 · 파일 이름 (사용자: camera moving / camera static)


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def outcomes(model):
    """[(위반 종류, 카메라, 1 / 0.5 / 0)]"""
    pairs = defaultdict(dict)
    for r in csv.DictReader(open(REPO / f"data_csv/{DS}/index.csv")):
        pairs[(r["block_id"], r["pair_id"])][r["plausible"]] = r
    S = json.load(open(RES / f"{DS}_{model}/per_block.json"))["per_video_surprise"]
    sc = lambda v: S[v] if isinstance(S[v], (int, float)) else list(S[v].values())[0]
    out = []
    for p in pairs.values():
        assert p["1"]["Camera"] == p["0"]["Camera"], p["1"]["video_id"]
        a, b = sc(p["1"]["video_id"]), sc(p["0"]["video_id"])
        out.append((p["1"]["condition"], p["1"]["Camera"], 1.0 if b > a else 0.5 if b == a else 0.0))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    R = {n: outcomes(m) for n, m, _ in MODELS}
    table, counts = {}, {}
    for n, r in R.items():
        allv = [c for _, _, c in r]
        print(f"{n:26s} 전체 {100 * np.mean(allv):.1f} (n {len(allv)})")     # 시트 IntPhys2 late occ 열과 같은지 (release 52.4 · AR 52.0)
    for cam, tag in CAMS:
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        w = 0.8 / len(MODELS)
        ns = []
        for si, (n, _, col) in enumerate(MODELS):
            ys, lo, hi = [], [], []
            for v in VIOL:
                cs = [c for vv, cc, c in R[n] if vv == v and cc == cam]
                k = sum(cs); ys.append(100 * k / len(cs)); a, b = wilson(k, len(cs)); lo.append(ys[-1] - a); hi.append(b - ys[-1])
                table.setdefault(n, {})[f"{cam}/{v}"] = {"acc": round(ys[-1], 1), "n": len(cs)}
                if si == 0:
                    ns.append(len(cs))
            pos = np.arange(len(VIOL)) + (si - (len(MODELS) - 1) / 2) * w
            allc = [c for _, cc, c in R[n] if cc == cam]
            ax.bar(pos, ys, w * 0.92, color=col, label=f"{n}  ·  all {100 * np.mean(allc):.1f}", edgecolor="white", linewidth=0.6)
            ax.errorbar(pos, ys, yerr=[lo, hi], fmt="none", ecolor="#555555", elinewidth=0.7, capsize=2)
            for xx, yy in zip(pos, ys):
                ax.text(xx, 2, f"{yy:.0f}", ha="center", va="bottom", fontsize=7.5, color="white")
        allc = {n: [c for _, cc, c in R[n] if cc == cam] for n in R}
        for n in R:
            table[n][f"{cam}/all"] = {"acc": round(100 * np.mean(allc[n]), 1), "n": len(allc[n])}
        ax.axhline(50, color="#888888", lw=0.8, ls="--")
        tot = {v: sum(1 for vv, _, _ in R[MODELS[0][0]] if vv == v) for v in VIOL}                      # 위반 종류별 전체 쌍 수 (두 카메라 합)
        ax.set_xticks(np.arange(len(VIOL))); ax.set_xticklabels([f"{v}\n{k} / {tot[v]} pairs" for v, k in zip(VIOL, ns)], fontsize=8.5)
        ax.set_ylim(0, 105); ax.set_ylabel("pairwise accuracy (%)"); ax.grid(axis="y", color="#e5e5e5", lw=0.6); ax.set_axisbelow(True)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        N = len(R[MODELS[0][0]])
        ax.set_title(f"IntPhys 2 late occ  ·  camera {tag}:  {sum(ns)} of {N} pairs ({100 * sum(ns) / N:.1f}%)", fontsize=10)
        counts[tag] = {"pairs": sum(ns), "of": N, "share": round(100 * sum(ns) / N, 1), "per_violation": {v: {"pairs": k, "of": tot[v]} for v, k in zip(VIOL, ns)}}
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8.5, frameon=False)
        fig.tight_layout()
        fig.savefig(OUT / f"{DS}_camera_{tag}.png", dpi=170)
        plt.close(fig)
        print("→", OUT / f"{DS}_camera_{tag}.png")
    print("카메라별 쌍 수:", json.dumps(counts, ensure_ascii=False))
    json.dump({"dataset": DS, "counts": counts, "cells": table, "rule": "matched pair, S(imp) > S(pos) correct, tie 0.5; Wilson 95% error bars; camera = IntPhys2 metadata Camera"},
              open(OUT / f"{DS}_camera.json", "w"), indent=1, ensure_ascii=False)
    for n in table:
        print(f"{n:26s} " + " · ".join(f"{k} {v['acc']} ({v['n']})" for k, v in table[n].items()))


if __name__ == "__main__":
    main()
