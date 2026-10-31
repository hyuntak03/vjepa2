#!/usr/bin/env python3
"""IntPhys 1 dev — 위반 종류 (O1 영속 · O2 모양 · O3 연속) 마다 x = 운동 종류 (정지 · 이동), y = 쌍 정확도, 막대 = 모델 (2026-10-25, 사용자 지시:
"IntPhys1 물체 움직이고 안움직이는거 다 labeling 되어 있잖아. violation type 별로 그래프 … x축을 motion type, y축을 acc").

채점 = 시트 IntPhys 1 열과 같다 (sheet_row_report.intphys1): Garrido A.8, 창 w16 · w32 (DINO-F 는 w5 · w8 · w16) 를 합쳐
  Filtered (시작점마다 C 최솟값) → AvgSurprise → 쌍 정답 = S(pos) < S(imp) (strict, 동점 오답) → property macro 최고 combo.
  그 combo 하나의 쌍 결과를 운동 라벨로 나눈다 (운동마다 combo 를 다시 고르지 않는다).
운동 라벨 = z_research/previous/OcclusionStats/intphys1/videos.json 의 group (가능 영상마다 moving|static / occluded|visible).
  쌍 수: 위반 종류마다 정지 20 · 이동 40 (가림 · 보임 반반). 정지 칸은 한 쌍이 5 pt 다 → 막대에 Wilson 95 % 구간.
복사 기준선 = V-JEPA 2 ViT-H 의 마지막 문맥 latent 복사, 같은 skip2_w32 칸 (TrainingEffects/ip1score/ip1score_final_scores.json 'copy|std|skip2_w32').
모델 = release (V-JEPA 2 ViT-H 릴리즈 predictor) · AR (vast AR-state ep70) 둘 (사용자 10-25: "일단 AR(vast) 랑 release 만").
  다른 모델은 MODELS 에 (이름, 폴더 glob, 색) 을 더하면 된다. 복사 기준선은 그림에 넣지 않고 json 에 칸 값만 남긴다.

  python z_research/scripts/figures/plot_intphys1_motion_models.py
  python z_research/scripts/figures/plot_intphys1_motion_models.py --bench late_uni      (2026-10-25)
    IntPhys1 late occ (시트 열 = 이 세트의 가림 90 쌍) 의 창 하나 채점 — 위반 시작 d 앞을 문맥으로 uniform (cvpr/eval/intphys1_late, build --uniform-d).
    이 세트는 보임 쌍도 같이 채점돼 있다 (180 쌍) → 정지 · 이동 × 보임 · 가림 네 칸. 쌍 결과 = per_block.json 의 영상 surprise, S(imp) > S(pos) 이면 정답 (동점 0.5).
    운동 라벨은 같은 영상이라 위와 같은 videos.json (index 의 group 열과 같은지 확인한다).
    산출 cvpr/results/eval/intphys1_late/figures/intphys1_late_uni_motion_by_violation_{visible,occluded}.png + .json (복사 기준선 copy_z · copy_h 는 json 에만)
산출 cvpr/results/eval/intphys1/figures/intphys1_motion_by_violation{,_occ}.png + .json (칸 값 · n · 고른 combo)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "z_research/scripts/analysis"))
import garrido_rescore as G  # noqa: E402

IP, BM, OLD = REPO / "cvpr/results/eval/intphys1", REPO / "z_research/previous/Benchmarks/exp_results", REPO / "z_research/previous/IntPhys/exp_results"
LAB = json.load(open(REPO / "z_research/previous/OcclusionStats/intphys1/videos.json"))          # 가능 영상 → group
COPY = json.load(open(REPO / "z_research/previous/TrainingEffects/ip1score/ip1score_final_scores.json"))
OUT = IP / "figures"

# (표시 이름, 결과 폴더들, 색) — dataviz 기본 범주 팔레트 slot 1 · 2
MODELS = [("release (V-JEPA 2 ViT-H)", lambda: sorted(IP.glob("intphys1_dev_vith_wgarrido_w[0-9]*")), "#2a78d6"),
          ("AR (vast)", lambda: sorted(IP.glob("intphys1_dev_vith_vast_ar_state_ep70_wgarrido_w[0-9]*")), "#eb6834")]
CPN = "copy (V-JEPA 2 last context latent)"
PRIN = [("O1", "O1 permanence"), ("O2", "O2 shape constancy"), ("O3", "O3 spatio-temporal continuity")]


def outcomes(dirs):
    """시트와 같은 combo 선택 → 그 combo 의 쌍마다 (가능 영상, 정답 0/1)"""
    dirs = [Path(d) for d in dirs]
    assert dirs and all((d / "per_window.json").exists() for d in dirs), dirs
    combos, src = {}, {}
    for d in dirs:
        c = G.rescore(d)["combos"]
        combos.update(c); src.update({k: d for k in c})
    best = G.select(combos, "intphys")
    d = src[best["combo"]]
    _, meta = G.load_index(d)
    vids, _, _, arr = G.score_table(json.load(open(d / "per_window.json"))["windows"], best["combo"])
    S = {v: float(np.nanmean(np.nanmin(arr[i], axis=1))) for i, v in enumerate(vids)}           # Filtered → AvgSurprise
    pairs = defaultdict(dict)
    for v, s in S.items():
        r = meta[v]; pairs[G.pair_key(r)][r["plausible"]] = (v, s)
    res = [(p["1"][0], int(p["1"][1] < p["0"][1])) for p in pairs.values()]
    return best, res


LATE = REPO / "cvpr/results/eval/intphys1_late"
LATE_MODELS = [("release (V-JEPA 2 ViT-H)", "vith", "#2a78d6"), ("AR (vast)", "vith_vast_ar_state_ep70", "#eb6834")]


def late_outcomes(ds, model, copy_key=None):
    """창 하나 채점의 쌍 결과 — [(가능 영상, 1 / 0.5 / 0)]. copy_key 를 주면 copy_baseline.json 의 그 복사로"""
    pairs = defaultdict(dict)
    for r in csv.DictReader(open(REPO / f"data_csv/{ds}/index.csv")):
        pairs[(r["block_id"], r["pair_id"])][r["plausible"]] = r
    d = LATE / f"{ds}_{model}"
    S = json.load(open(d / "copy_baseline.json"))["per_video"][copy_key] if copy_key else json.load(open(d / "per_block.json"))["per_video_surprise"]
    sc = lambda v: S[v] if isinstance(S[v], (int, float)) else list(S[v].values())[0]
    res = []
    for p in pairs.values():
        pv, iv = p["1"]["video_id"], p["0"]["video_id"]
        assert LAB[pv]["group"] == p["1"]["group"], (pv, LAB[pv]["group"], p["1"]["group"])     # 운동 라벨이 같은지
        a, b = sc(pv), sc(iv)
        res.append((pv, 1.0 if b > a else 0.5 if b == a else 0.0))
    return res


def cells(res, key):
    acc = defaultdict(list)
    for v, c in res:
        g = LAB[v]
        acc[(g["principle"], key(g["group"]))].append(c)
    return acc


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def main_late(ds="intphys1_late_uni"):
    out = LATE / "figures"; out.mkdir(parents=True, exist_ok=True)
    R = {n: late_outcomes(ds, m) for n, m, _ in LATE_MODELS}
    for k in ("copy_z", "copy_h"):
        R[f"{k} (release)"] = late_outcomes(ds, "vith", k)
    occ = {n: round(100 * np.mean([c for v, c in r if LAB[v]["group"].endswith("occluded")]), 1) for n, r in R.items()}
    print("가림 90 쌍 (시트 IntPhys1 late occ 열):", occ)
    xs = ["static", "moving"]                                        # 2026-10-25 사용자: 보임 · 가림은 그림을 따로, x 축은 운동만
    table = {}
    for occ_ in ("visible", "occluded"):
        key = lambda g, o=occ_: g.split("/")[0] if g.endswith(o) else None
        fig, axes = plt.subplots(1, 3, figsize=(9, 3.8), sharey=True)
        w = 0.8 / len(LATE_MODELS)
        for pi, (p, ptitle) in enumerate(PRIN):
            ax = axes[pi]
            for si, (n, _, col) in enumerate(LATE_MODELS):
                cc = cells(R[n], key)
                ys, lo, hi = [], [], []
                for x in xs:
                    v = cc[(p, x)]; k = sum(v)
                    ys.append(100 * k / len(v)); a, b = wilson(k, len(v)); lo.append(ys[-1] - a); hi.append(b - ys[-1])
                pos = np.arange(len(xs)) + (si - (len(LATE_MODELS) - 1) / 2) * w
                ax.bar(pos, ys, w * 0.92, color=col, label=n if pi == 0 else None, edgecolor="white", linewidth=0.6)
                ax.errorbar(pos, ys, yerr=[lo, hi], fmt="none", ecolor="#555555", elinewidth=0.7, capsize=2)
                for xx, yy in zip(pos, ys):
                    ax.text(xx, 2, f"{yy:.0f}", ha="center", va="bottom", fontsize=7, color="white")
            ax.axhline(50, color="#888888", lw=0.8, ls="--")
            ax.set_xticks(np.arange(len(xs)))
            ax.set_xticklabels([f"{x}\n(n={len(cells(R[LATE_MODELS[0][0]], key)[(p, x)])})" for x in xs], fontsize=8)
            ax.set_ylim(0, 105); ax.grid(axis="y", color="#e5e5e5", lw=0.6); ax.set_axisbelow(True)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
            ax.set_title(ptitle, fontsize=10)
            if pi == 0:
                ax.set_ylabel("pairwise accuracy (%)")
        fig.legend(loc="upper center", ncol=2, fontsize=8.5, frameon=False)
        fig.text(0.01, 0.97, f"object {occ_}", fontsize=10, fontweight="bold", va="top")
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(out / f"{ds}_motion_by_violation_{occ_}.png", dpi=170)
        plt.close(fig)
        for n, r in R.items():
            for p, _ in PRIN:
                for x in xs:
                    v = cells(r, key)[(p, x)]
                    table.setdefault(n, {})[f"{p} {x}/{occ_}"] = {"acc": round(100 * np.mean(v), 1), "n": len(v)}
        print("→", out / f"{ds}_motion_by_violation_{occ_}.png")
    json.dump({"dataset": ds, "occluded_90": occ, "cells": table, "labels": "z_research/previous/OcclusionStats/intphys1/videos.json",
               "protocol": "single window: context 16 = uniform up to violation start d, future 16 after; matched pairs; tie 0.5; Wilson 95% error bars"},
              open(out / f"{ds}_motion_by_violation.json", "w"), indent=1, ensure_ascii=False)
    for n in R:
        print(f"{n:26s} " + " · ".join(f"{k} {v['acc']}" for k, v in table[n].items()))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    R, rec = {}, {}
    for name, dirs, _ in MODELS:
        best, res = outcomes(dirs())
        R[name] = res
        mac = {p: 100 * np.mean([c for v, c in res if LAB[v]["principle"] == p]) for p, _ in PRIN}
        assert abs(np.mean(list(mac.values())) - best["accuracy"]) < 1e-6, (name, mac, best["accuracy"])   # 시트 값과 같은 셈인지
        rec[name] = {"combo": best["combo"], "macro": round(best["accuracy"], 2), "per_principle": {p: round(v, 1) for p, v in mac.items()}}
        print(f"{name:26s} {best['combo']:10s} macro {best['accuracy']:.1f} · " + " · ".join(f"{p} {v:.1f}" for p, v in mac.items()))
    ck = "copy|std|skip2_w32"
    R[CPN] = [(pp[0], int(o == 1)) for pp, o in zip(COPY["pairs"]["skip2_w32"], COPY["outcomes"][ck])]

    for tag, key, xs, xlab in [("", lambda g: g.split("/")[0], ["static", "moving"], ["static", "moving"]),
                               ("_occ", lambda g: g, ["static/visible", "static/occluded", "moving/visible", "moving/occluded"],
                                ["static\nvisible", "static\noccluded", "moving\nvisible", "moving\noccluded"])]:
        fig, axes = plt.subplots(1, 3, figsize=(11 if tag else 9, 3.8), sharey=True)
        table = {n: {f"{p} {x}": {"acc": round(100 * np.mean(cells(R[n], key)[(p, x)]), 1), "n": len(cells(R[n], key)[(p, x)])}
                     for p, _ in PRIN for x in xs} for n in R}
        w = 0.8 / len(MODELS)
        for pi, (p, ptitle) in enumerate(PRIN):
            ax = axes[pi]
            for si, (n, _, col) in enumerate(MODELS):
                cc = cells(R[n], key)
                ys, lo, hi = [], [], []
                for x in xs:
                    v = cc[(p, x)]; k = sum(v)
                    ys.append(100 * k / len(v)); a, b = wilson(k, len(v)); lo.append(ys[-1] - a); hi.append(b - ys[-1])
                pos = np.arange(len(xs)) + (si - (len(MODELS) - 1) / 2) * w
                ax.bar(pos, ys, w * 0.92, color=col, label=n if pi == 0 else None, edgecolor="white", linewidth=0.6)
                ax.errorbar(pos, ys, yerr=[lo, hi], fmt="none", ecolor="#555555", elinewidth=0.7, capsize=2)
                for xx, yy in zip(pos, ys):
                    ax.text(xx, 2, f"{yy:.0f}", ha="center", va="bottom", fontsize=7, color="white")
            ax.axhline(50, color="#888888", lw=0.8, ls="--")
            ax.set_xticks(np.arange(len(xs)))
            ax.set_xticklabels([f"{l}\n(n={len(cells(R[MODELS[0][0]], key)[(p, x)])})" for l, x in zip(xlab, xs)], fontsize=8)
            ax.set_ylim(0, 105); ax.grid(axis="y", color="#e5e5e5", lw=0.6); ax.set_axisbelow(True)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
            ax.set_title(ptitle, fontsize=10)
            if pi == 0:
                ax.set_ylabel("pairwise accuracy (%)")
        fig.legend(loc="upper center", ncol=2, fontsize=8.5, frameon=False)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(OUT / f"intphys1_motion_by_violation{tag}.png", dpi=170)
        json.dump({"models": rec, "cells": table, "labels": "z_research/previous/OcclusionStats/intphys1/videos.json", "copy": ck,
                   "protocol": "Garrido A.8, best combo per model (sheet rule), Filtered, matched pairs, Wilson 95% error bars, chance 50"},
                  open(OUT / f"intphys1_motion_by_violation{tag}.json", "w"), indent=1, ensure_ascii=False)
        print("→", OUT / f"intphys1_motion_by_violation{tag}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="sliding", choices=["sliding", "late_uni"])
    main_late() if ap.parse_args().bench == "late_uni" else main()
