"""VOBench surprise 를 실제 물체 자리 토큰만으로 다시 채점 (보조 분석, 2026-10-13). 주 결론은 전체 토큰 (CLAUDE.md · 메모리 '채점 해석은 전체 토큰').

입력 = 채점 실행의 token_surprise.npz (미래 8 튜블릿 × 16×16 토큰별 L1). 다시 돌리지 않는다.
물체 자리 = label.json 의 중심 (u, v) ± 1.1 × (ext_u, ext_v) (반폭 · 반높이, 구르는 물체는 반지름) 상자와 겹치는 32 px 칸.
  튜블릿 t = 샘플 2t, 2t+1 (원본 프레임 3·(2t), 3·(2t+1)) 두 장의 상자 합집합.
  쌍의 두 영상에 같은 토큰을 쓴다 = 그 block 의 A arm · B arm 상자 합집합 (빈 arm 은 없음). 가려진 튜블릿도 실제 자리 (가림막 뒤) 그대로.
점수 = 선택된 (튜블릿, 토큰) 전부의 평균 L1. 쌍 정확도 (동점 0.5) 를 전체 토큰 점수와 나란히.

  python z_research/scripts/analysis/vobench_object_region_score.py [--margin 1.1]
  → z_research/VOBench_v1/vjepa2/surprise_object_region/{compare.md, compare.json}
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
EX = REPO / "z_research/VOBench_v1/exp_results"
FR = Path("/data2/local_datasets/world/Benchmark/VOBench_v1")
INDEX = REPO / "data_csv/vobench_v1/index.csv"
OUT = REPO / "z_research/VOBench_v1/vjepa2/surprise_object_region"
RUNS = {"release": EX / "surprise_c16t32__vobench_v1_vith", "ar_ep40": EX / "surprise_c16t32__vobench_v1_vith_ariel_ar_ep40"}
VIOL, OCC = ("permanence", "shape", "color"), ("visible", "early", "mid", "late")
G = 16; CELL = 32.0


def region(labels, margin):
    """labels = [arm 의 label 리스트 …] → (8, 16, 16) bool. 칸 [c·32, (c+1)·32) 가 상자와 겹치면 True."""
    m = np.zeros((8, G, G), bool)
    lo = np.arange(G) * CELL; hi = lo + CELL
    for L in labels:
        for j in range(8):
            for s in (16 + 2 * j, 17 + 2 * j):
                e = L[3 * s]
                eu, ev = margin * e.get("ext_u", e["r_px"]), margin * e.get("ext_v", e["r_px"])
                cu = (hi > e["u"] - eu) & (lo < e["u"] + eu); cv = (hi > e["v"] - ev) & (lo < e["v"] + ev)
                m[j] |= cv[:, None] & cu[None, :]
    return m


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--margin", type=float, default=1.1); a = ap.parse_args()
    E = {}
    for name, run in RUNS.items():
        z = np.load(run / "token_surprise.npz"); E[name] = dict(zip(z["video_id"], z["e"]))
    rows = list(csv.DictReader(INDEX.open())); by = defaultdict(dict)
    for r in rows: by[r["block_id"]][r["variant"]] = r
    P = []; frac = []
    for b, v in by.items():
        r = v["pos_a"]; bdir = FR / Path(r["file_name"]).parent
        lab = json.load(open(bdir / "label.json"))["labels"]
        arms = [lab[k] for k in ("A", "B") if lab.get(k) and lab[k][0] is not None]
        R = region(arms, a.margin); frac.append(R.mean())
        for d, p, q in (("A", "pos_a", "imp_ab"), ("B", "pos_b", "imp_ba")):
            row = dict(objset=r["objset"], viol=r["violation"], occ=r["occlusion"], mot=r["motion"], dir=d)
            for m in E:
                ep, ei = E[m][v[p]["video_id"]], E[m][v[q]["video_id"]]
                for tag, sp, si in (("all", ep.mean(), ei.mean()), ("obj", ep[R].mean(), ei[R].mean())):
                    row[f"{m}|{tag}"] = 1.0 if si > sp else 0.5 if si == sp else 0.0
            P.append(row)
    cols = [f"{m}|{t}" for m in E for t in ("all", "obj")]
    acc = lambda ps, c: 100 * np.mean([p[c] for p in ps]) if ps else float("nan")
    J = {"margin": a.margin, "region_frac_mean": float(np.mean(frac)), "cells": {}}
    head = "| 칸 | 쌍 | " + " | ".join(f"{m} 전체 → 물체 자리" for m in E) + " |"
    md = [f"# VOBench — 물체 자리 토큰만으로 채점 (보조, 상자 ×{a.margin}, 2026-10-13)", "",
          f"물체 자리 = 튜블릿마다 실제 물체 상자 (중심 ± {a.margin} × 반폭 · 반높이) 와 겹치는 토큰, A · B arm 합집합. 평균 토큰 비율 {100 * np.mean(frac):.1f} % (256 중).",
          "칸 = 쌍 정확도 %, '전체 → 물체 자리'. 원 채점 (전체 토큰) 이 주 결과다.", ""]

    def table(title, keys, f):
        nonlocal md
        md += [f"## {title}", "", head, "|---|---:|" + "---:|" * len(E)]
        for k in keys:
            ps = [p for p in P if f(p, k)]
            J["cells"][f"{title}/{k}"] = {c: acc(ps, c) for c in cols} | {"n": len(ps)}
            md.append(f"| {k} | {len(ps)} | " + " | ".join(f"{acc(ps, m + '|all'):.1f} → {acc(ps, m + '|obj'):.1f}" for m in E) + " |")
        md.append("")
    for o in ("primitive", "realistic"):
        table(f"{o} · overall", ["all"], lambda p, k, o=o: p["objset"] == o)
        table(f"{o} · 위반 × 가림", [f"{v}/{oc}" for v in VIOL for oc in OCC], lambda p, k, o=o: p["objset"] == o and f"{p['viol']}/{p['occ']}" == k)
        table(f"{o} · late permanence 방향", ["A", "B"], lambda p, k, o=o: p["objset"] == o and p["occ"] == "late" and p["viol"] == "permanence" and p["dir"] == k)
        table(f"{o} · late 모양+색 · 정지 / 이동", ["static", "moving"],
              lambda p, k, o=o: p["objset"] == o and p["occ"] == "late" and p["viol"] != "permanence" and (p["mot"] == "flat_static") == (k == "static"))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "compare.md").write_text("\n".join(md) + "\n"); json.dump(J, open(OUT / "compare.json", "w"), indent=1)
    print("\n".join(md))


if __name__ == "__main__":
    main()
