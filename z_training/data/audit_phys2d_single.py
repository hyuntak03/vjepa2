#!/usr/bin/env python3
"""phys2d_single 편향 점검 (2026-10-19) — 시간 · 정체 · 시나리오. 출력은 markdown.

  /venv/vjepa/bin/python z_training/data/audit_phys2d_single.py --root <out>/train [--C 16 --K 16 --tubelet 2]

1. 시간: P(완전히 가려짐 | 영상 장 t) — t 구간별, 그리고 uniform.tsv 창 (문맥 C + 예측 K) 안 블록 자리별.
   블록 단위 상태 head 가 "몇 번째 블록이면 가려져 있다" 를 외울 수 있는지 본다. 사건 행 (boundary.tsv) 창도 같이 — 대조용.
2. 정체: 48 정체별 영상 수 · 가림 비율 · 평균 속력 · 평균 위치. 정체 하나로 이것들을 맞힐 수 있으면 편향이다 (최대-최소 폭과 표준편차).
3. 시나리오별 가림 비율 · 가림 구간 수 · 첫 장/끝 장 가림 비율 · 사건 행 수.
"""
import argparse
import csv
import os
from collections import defaultdict

import numpy as np

NID = 48


def load(root):
    man = list(csv.DictReader(open(os.path.join(root, "manifest.tsv")), delimiter="\t"))
    clips = {}
    for m in man:
        p = os.path.join(root, "clips", m["name"], f"{m['name']}_{int(m['k']):05d}")
        z = np.load(p + "_states.npz")
        clips[os.path.abspath(p + ".mp4")] = dict(m=m, hidden=z["hidden"][:, 0], pos=z["pos"][:, 0], vel=z["vel"][:, 0],
                                                  seen=z["seen"][:, 0], ident=int(z["identity"][0]))
    return clips


def window_rates(rows, clips, C, K, tub):
    """창 = [b − C, b + K). 블록 = tubelet 장 묶음, 블록 라벨 = 마지막 장 (state_head 와 같다)."""
    nb = (C + K) // tub
    hid, n = np.zeros(nb), 0
    for r in rows:
        b, c0 = int(r["boundary_frame"]), int(r["ctx_start"])
        c = clips.get(r["path"])
        if c is None or b - C < c0 or b + K > len(c["hidden"]):
            continue
        h = c["hidden"][b - C:b + K][tub - 1::tub]
        hid += h
        n += 1
    return hid / max(n, 1), n


def fmt_row(v):
    return " | ".join(f"{100 * x:.1f}" for x in v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--C", type=int, default=16)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--tubelet", type=int, default=2)
    a = ap.parse_args()
    clips = load(a.root)
    print(f"# phys2d_single 편향 점검 — `{a.root}`\n\n영상 {len(clips):,}\n")

    # 1. 시간
    Tm = max(len(c["hidden"]) for c in clips.values())
    hs, ns = np.zeros(Tm), np.zeros(Tm)
    for c in clips.values():
        T = len(c["hidden"])
        hs[:T] += c["hidden"]
        ns[:T] += 1
    edges = list(range(0, Tm, 8))
    print("## 1. 시간\n\n### 영상 장 t 별 P(가려짐) (%)\n")
    print("| t | " + " | ".join(f"{e}–{e + 7}" for e in edges) + " |")
    print("|---|" + "---:|" * len(edges))
    print("| P | " + " | ".join(f"{100 * hs[e:e + 8].sum() / max(ns[e:e + 8].sum(), 1):.1f}" for e in edges) + " |")
    print("| 영상 수 | " + " | ".join(f"{int(ns[e])}" for e in edges) + " |\n")
    tot = hs.sum() / ns.sum()
    print(f"전체 장 중 가려짐 {100 * tot:.1f} %\n")
    for name in ("uniform.tsv", "boundary.tsv"):
        f = os.path.join(a.root, name)
        if not os.path.exists(f):
            continue
        rows = list(csv.DictReader(open(f), delimiter="\t"))
        v, n = window_rates(rows, clips, a.C, a.K, a.tubelet)
        nb = len(v)
        print(f"### `{name}` 창 (C{a.C} K{a.K}, 블록 {nb}) 블록 자리별 P(가려짐) (%) — 창 {n:,}\n")
        print("| 블록 | " + " | ".join(("c" if j < a.C // a.tubelet else "p") + str(j) for j in range(nb)) + " |")
        print("|---|" + "---:|" * nb)
        print("| P | " + fmt_row(v) + " |")
        print(f"\n폭 (최대 − 최소) {100 * (v.max() - v.min()):.1f} pt, 평균 {100 * v.mean():.1f} %\n")

    # 2. 정체
    by = defaultdict(list)
    for c in clips.values():
        by[c["ident"]].append(c)
    stat = []
    for i in range(NID):
        cs = by.get(i, [])
        if not cs:
            stat.append((i, 0, np.nan, np.nan, np.nan, np.nan))
            continue
        hf = np.mean([c["hidden"].mean() for c in cs])
        sp = np.mean([np.nanmean(np.linalg.norm(c["vel"], axis=-1)) for c in cs])
        x = np.mean([np.nanmean(c["pos"][:, 0]) for c in cs])
        y = np.mean([np.nanmean(c["pos"][:, 1]) for c in cs])
        stat.append((i, len(cs), hf, sp, x, y))
    s = np.array(stat, float)
    print("## 2. 정체 (48)\n")
    print("| 지표 | 최소 | 최대 | 평균 | 표준편차 |\n|---|---:|---:|---:|---:|")
    for j, lab in [(1, "영상 수"), (2, "가림 비율 %"), (3, "평균 속력 px/장"), (4, "평균 x px"), (5, "평균 y px")]:
        col = s[:, j] * (100 if j == 2 else 1)
        print(f"| {lab} | {np.nanmin(col):.1f} | {np.nanmax(col):.1f} | {np.nanmean(col):.1f} | {np.nanstd(col):.1f} |")
    shapes = ("circle", "square", "rectangle", "triangle", "pentagon", "hexagon")
    print("\n모양별 (색 8 개 평균):\n\n| 모양 | 영상 | 가림 % | 속력 |\n|---|---:|---:|---:|")
    for k, sh in enumerate(shapes):
        r = s[k * 8:(k + 1) * 8]
        print(f"| {sh} | {int(np.nansum(r[:, 1]))} | {100 * np.nanmean(r[:, 2]):.1f} | {np.nanmean(r[:, 3]):.2f} |")

    # 3. 시나리오
    sc = defaultdict(list)
    for c in clips.values():
        sc[(c["m"]["category"], c["m"]["name"])].append(c)
    print("\n## 3. 시나리오\n\n| 범주 | 시나리오 | 영상 | 길이 | 가림 % | 가림 구간 | 첫 장 가림 % | 끝 장 가림 % | 가림 0 % | 사건 행 | seed 재시도 % |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for (cat, name), cs in sorted(sc.items()):
        m = [c["m"] for c in cs]
        print(f"| {cat} | {name} | {len(cs)} | {np.mean([len(c['hidden']) for c in cs]):.0f} | "
              f"{100 * np.mean([c['hidden'].mean() for c in cs]):.1f} | {np.mean([int(x['n_hidden_spans']) for x in m]):.2f} | "
              f"{100 * np.mean([c['hidden'][0] for c in cs]):.0f} | {100 * np.mean([c['hidden'][-1] for c in cs]):.0f} | "
              f"{100 * np.mean([not c['hidden'].any() for c in cs]):.0f} | {np.mean([int(x['n_rows']) for x in m]):.1f} | "
              f"{100 * np.mean([int(x.get('seed_retry', 0)) > 0 for x in m]):.0f} |")
    spans = np.array([int(c["m"]["n_hidden_spans"]) for c in clips.values()])
    print("\n가림 구간 수 분포: " + ", ".join(f"{k}: {100 * (spans == k).mean():.0f} %" for k in range(int(spans.max()) + 1)))


if __name__ == "__main__":
    main()
