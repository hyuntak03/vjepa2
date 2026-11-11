#!/usr/bin/env python3
"""BlenderTrack v3 장면 고르기 — 설계에 성공한 후보 중 상황마다 요인 분포가 가장 고른 blocks_per_situation 개 (2026-11-01).

왜: 요인 (물건 · 색 · 크기 · 속도 · 방향 · 가림 길이 · 물체 수 · 실내외 · 바닥 · 가림 하위 종류 · 빗면 방향 · 측면 부호) 은 후보 요인표에서 정확히 고르지만,
    설계 실패가 특정 조합에 몰리면 성공한 것만 쓰면 치우친다 (예: 비치볼). 그래서 후보를 넉넉히 (2.5 배) 설계하고 여기서 고른다.
방법: 탐욕 — 매 단계, 넣었을 때 요인마다 (수준별 개수 − 고른 목표) 제곱합이 가장 작아지는 후보를 넣는다.
안 고른 장면은 렌더 번호 두 개 (A · B) 를 design/*.fail ("unselected") 로 표시해 렌더가 건너뛴다 (spec 은 남긴다).
보고: <out>/select_<split>.json — 상황마다 성공 후보 수 · 고른 수 · 요인마다 수준별 개수와 최대 치우침 (고른 것 / 고르게였다면).
  $P select_blocks_v3.py --config configs/v3.yaml --split train
"""
import argparse
import collections
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_design_v3 as G  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--split", required=True)
    a = ap.parse_args()
    cfg = G.load_cfg(a.config, a.split)
    out, sp = cfg["out"], cfg["splits"][a.split]
    sd, N, NC = sp["seed"], int(sp["blocks_per_situation"]), int(sp["candidates"])
    S = G.block_situations(cfg)
    NS = len(S)
    rep = {}
    total_sel = 0
    for si, sit in enumerate(S):
        levels = G.factor_levels(sit, cfg)
        cands = []
        for k in range(NC):
            b = k * NS + si
            f = f"{out}/design/{a.split}{sd}_{2 * b:06d}.json"
            if os.path.exists(f):
                cands.append((b, json.load(open(f))["factors"]))
        cnt = {fn: collections.Counter() for fn in levels}
        chosen, left = [], list(range(len(cands)))
        target = {fn: N / len(lv) for fn, lv in levels.items()}
        while len(chosen) < N and left:
            best, bs = None, None
            for j in left:
                F = cands[j][1]
                sc = sum((cnt[fn][str(F[fn])] + 1 - target[fn]) ** 2 - (cnt[fn][str(F[fn])] - target[fn]) ** 2 for fn in levels)
                if bs is None or sc < bs:
                    best, bs = j, sc
            chosen.append(best); left.remove(best)
            for fn in levels:
                cnt[fn][str(cands[best][1][fn])] += 1
        sel = {cands[j][0] for j in chosen}
        for b, _ in cands:
            if b not in sel:
                for i in (2 * b, 2 * b + 1):
                    open(f"{out}/design/{a.split}{sd}_{i:06d}.fail", "w").write("unselected (select_blocks_v3)")
        total_sel += len(sel)
        dev = {fn: round(max(cnt[fn][str(v)] for v in lv) / max(1e-9, len(chosen) / len(lv)), 2) for fn, lv in levels.items()}
        rep[G.situation_key(sit)] = {"candidates": NC, "succeeded": len(cands), "selected": len(chosen), "max_over_uniform": dev,
                                     "counts": {fn: dict(c) for fn, c in cnt.items()}}
    json.dump(rep, open(f"{out}/select_{a.split}.json", "w"), indent=1, ensure_ascii=False)
    short = {k: v for k, v in rep.items() if v["selected"] < N}
    print(f"{a.split}: 상황 {NS} · 고른 장면 {total_sel} (목표 {N * NS}) · 모자란 상황 {len(short)}")
    for k, v in sorted(short.items(), key=lambda x: x[1]["selected"]):
        print(f"   {v['selected']}/{N} (성공 후보 {v['succeeded']}/{NC})  {k}")
    worst = sorted(((max(v["max_over_uniform"].values()), k, v["max_over_uniform"]) for k, v in rep.items()), reverse=True)[:5]
    for w, k, d in worst:
        print(f"   가장 치우친 요인 {w:.2f} × 고른 값  {k}  {d}")


if __name__ == "__main__":
    main()
