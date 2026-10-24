#!/usr/bin/env python3
"""VOBench_v1 을 train / test 로 나눈다 (predictor post-FT 용).

  python z_training/data/build_vobench_split_index.py                 # v2 (기본): 장면 묶음 단위 → data_csv/vobench_v1_split_v2/
  python z_training/data/build_vobench_split_index.py --version v1    # v1: block 단위 (⚠️ 누수 — 기록용) → data_csv/vobench_v1_split/
  python z_training/data/build_vobench_split_index.py --version all   # 전부 train (가능 1,440 clip, test 없음) → data_csv/vobench_v1_allpos/ — 상한선용

⚠️ v1 (2026-10-12, block 단위 · 셀 층화) 은 **누수가 있다** (2026-10-14 발견): VOBench 는 한 위반 block 의 가능 arm 을
   다른 위반 block 에서 같은 설정으로 다시 쓴다 (예: permanence 의 pos_a 와 shape 의 pos_a 가 같은 물체 · 배경 · 운동 · 가림 · 궤적).
   v1 의 test 가능 영상 720 중 52 개가 train 영상과 (objset · 운동 · 배경 · 가림 · k · 물체 · 궤적) 이 같고 렌더 노이즈만 다르다.
v2 = 장면 묶음 단위: 가능 arm 의 설정 키 (objset, motion, background, occlusion, k, 물체) 를 하나라도 공유하는 block 을
   union-find 로 묶고, 묶음째로 train / test 에 보낸다 — train 과 test 사이에 같은 키가 하나도 없음을 검사한다.
   묶음 배정은 셀 (objset × violation × occlusion × motion) 별 block 수가 반반에 가깝도록 탐욕 배정 (seed 0 순서).
train = train 쪽 **가능 변이만** (pos_a · pos_b), test = test 쪽 4 변이 전부 (matched 쌍 채점용).
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "data_csv/vobench_v1/index.csv"


def arm_key(r):
    obj = r["obj_a"] if r["variant"] == "pos_a" else r["obj_b"]
    return (r["objset"], r["motion"], r["background"], r["occlusion"], r["k"], obj)


def split_v1(rows, frac, seed):
    cells = defaultdict(set)
    for r in rows:
        cells[(r["objset"], r["violation"], r["occlusion"], r["motion"])].add(r["block_id"])
    rng = np.random.default_rng(seed); train = set()
    for k in sorted(cells):
        b = sorted(cells[k]); rng.shuffle(b); train |= set(b[: int(round(len(b) * frac))])
    return train


def split_v2(rows, frac, seed):
    by = defaultdict(dict)
    for r in rows: by[r["block_id"]][r["variant"]] = r
    parent = {b: b for b in by}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    owner = {}
    for b, v in by.items():
        for p in ("pos_a", "pos_b"):
            k = arm_key(v[p])
            if k[-1] == "empty":
                continue
            if k in owner: parent[find(b)] = find(owner[k])
            else: owner[k] = b
    groups = defaultdict(list)
    for b in by: groups[find(b)].append(b)
    cell = lambda b: (by[b]["pos_a"]["objset"], by[b]["pos_a"]["violation"], by[b]["pos_a"]["occlusion"], by[b]["pos_a"]["motion"])
    G = sorted(groups.values(), key=lambda g: sorted(g)[0]); rng = np.random.default_rng(seed); rng.shuffle(G)
    G.sort(key=len, reverse=True)                                   # 큰 묶음 먼저 (탐욕 배정이 덜 치우친다)
    tot = defaultdict(int)
    for b in by: tot[cell(b)] += 1
    cnt = {"train": defaultdict(int), "test": defaultdict(int)}; train = set()
    for g in G:
        cs = [cell(b) for b in g]
        gap = sum(frac * tot[c] - cnt["train"][c] for c in cs) - sum((1 - frac) * tot[c] - cnt["test"][c] for c in cs)
        side = "train" if gap > 0 or (gap == 0 and rng.random() < frac) else "test"
        for b, c in zip(g, cs): cnt[side][c] += 1
        if side == "train": train |= set(g)
    print(f"장면 묶음 {len(G)} 개 (block 1 개짜리 {sum(len(g) == 1 for g in G)}, 가장 큰 {max(map(len, G))} block)")
    return train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v2", choices=["v1", "v2", "all"])
    ap.add_argument("--train-frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rows = list(csv.DictReader(SRC.open()))
    train = ({r["block_id"] for r in rows} if a.version == "all" else
             (split_v1 if a.version == "v1" else split_v2)(rows, a.train_frac, a.seed))
    tr = [r for r in rows if r["block_id"] in train and r["variant"] in ("pos_a", "pos_b")]
    te = [r for r in rows if r["block_id"] not in train]
    # 검사: train · test 가 공유하는 가능 arm 설정 키
    trk = {arm_key(r) for r in tr if arm_key(r)[-1] != "empty"}
    shared = sum(arm_key(r) in trk for r in te if r["variant"] in ("pos_a", "pos_b") and arm_key(r)[-1] != "empty")
    cells = defaultdict(lambda: [0, 0])
    for r in rows:
        if r["variant"] == "pos_a":
            cells[(r["objset"], r["violation"], r["occlusion"])][r["block_id"] in train] += 1
    worst = max(abs(c[1] / (c[0] + c[1]) - a.train_frac) for c in cells.values())
    out = REPO / {"v1": "data_csv/vobench_v1_split", "v2": "data_csv/vobench_v1_split_v2", "all": "data_csv/vobench_v1_allpos"}[a.version]
    out.mkdir(parents=True, exist_ok=True)
    for name, rs in (("index_train.csv", tr), ("index_test.csv", te)):
        with open(out / name, "w", newline="") as f:
            w = csv.DictWriter(f, rows[0].keys()); w.writeheader(); w.writerows(rs)
    nb = len({r["block_id"] for r in rows})
    print(f"[{a.version}] block {nb} → train {len(train)} · test {nb - len(train)} | train clip (가능만) {len(tr)} · test clip {len(te)}")
    print(f"  train·test 공유 가능 arm 설정 키: {shared} (0 이어야 누수 없음) | (objset × 위반 × 가림) 셀 train 비율 최대 편차 {worst:.2f} → {out}")


if __name__ == "__main__":
    main()
