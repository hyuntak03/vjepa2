#!/usr/bin/env python3
"""VOBench_v1 gravity 를 train / test 로 나누고, VOBench split v2 train 과 합친 학습 index 를 만든다 (2026-10-15, 사용자: PFT 에 gravity 가능 clip 포함).

  python z_training/data/build_vobench_gravity_split_index.py
  → data_csv/vobench_v1_gravity_split/index_{train,test}.csv   (train = 가능 'pos' 만, test = 4 ~ 10 미래 전부 — cross 채점용)
  → data_csv/vobench_v1_split_v2g/index_train.csv              (VOBench split v2 train 720 + gravity train pos — 한 데이터셋이라 크기 비례로 섞인다)

묶음 단위 = (objset, law, background, obj). arc 의 apex66 · 48 · 30 은 같은 물체 · 배경이면 같은 포물선을 시간만 밀어 자른 것이라
(apex66 의 미래 f66 이후 = apex48 의 미래) 한 묶음으로 묶어야 test 미래가 train 에 안 나온다. ledge 도 같은 물체 · 배경의 6 셀을 같이 보낸다.
같은 셀 (같은 z · v 또는 apex) 의 궤적은 다른 물체 · 배경으로 train 에 나온다 — VOBench split v2 가 같은 운동을 다른 물체로 양쪽에 두는 것과 같은 정책.
묶음 배정은 셀 (objset × law × cell) 별 block 수가 반반에 가깝도록 탐욕 배정 (seed 0, build_vobench_split_index.py v2 와 같은 방식).
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "data_csv/vobench_v1_gravity/index.csv"
VOB_TRAIN = REPO / "data_csv/vobench_v1_split_v2/index_train.csv"
OUT_G = REPO / "data_csv/vobench_v1_gravity_split"
OUT_C = REPO / "data_csv/vobench_v1_split_v2g"


def main(frac=0.5, seed=0):
    rows = list(csv.DictReader(SRC.open()))
    by = defaultdict(list)
    for r in rows: by[r["block_id"]].append(r)
    head = {b: next(r for r in rs if r["variant"] == "pos") for b, rs in by.items()}
    groups = defaultdict(list)
    for b, h in head.items(): groups[(h["objset"], h["law"], h["background"], h["obj"])].append(b)
    cell = lambda b: (head[b]["objset"], head[b]["law"], head[b]["cell"])
    G = sorted(groups.values(), key=lambda g: sorted(g, key=int)[0]); rng = np.random.default_rng(seed); rng.shuffle(G)
    G.sort(key=len, reverse=True)
    tot = defaultdict(int)
    for b in by: tot[cell(b)] += 1
    cnt = {"train": defaultdict(int), "test": defaultdict(int)}; train = set()
    for g in G:
        cs = [cell(b) for b in g]
        gap = sum(frac * tot[c] - cnt["train"][c] for c in cs) - sum((1 - frac) * tot[c] - cnt["test"][c] for c in cs)
        side = "train" if gap > 0 or (gap == 0 and rng.random() < frac) else "test"
        for c in cs: cnt[side][c] += 1
        if side == "train": train |= set(g)
    tr = [r for r in rows if r["block_id"] in train and r["plausible"] == "1"]
    te = [r for r in rows if r["block_id"] not in train]
    # 검사: 묶음 키 공유 0, train 은 가능만
    k = lambda r: (r["objset"], r["law"], r["background"], r["obj"])
    shared = len({k(r) for r in tr} & {k(r) for r in te})
    assert shared == 0 and all(r["variant"] == "pos" for r in tr)
    OUT_G.mkdir(parents=True, exist_ok=True)
    for name, rs in (("index_train.csv", tr), ("index_test.csv", te)):
        with open(OUT_G / name, "w", newline="") as f:
            w = csv.DictWriter(f, rows[0].keys()); w.writeheader(); w.writerows(rs)
    print(f"[gravity] 묶음 {len(G)} 개 · block {len(by)} → train {len(train)} · test {len(by) - len(train)} | train clip (pos) {len(tr)} · test clip {len(te)} | 공유 묶음 키 {shared}")
    for c in sorted(tot):
        print(f"  {'/'.join(c):28s} train {cnt['train'][c]:2d} · test {cnt['test'][c]:2d}")
    # 합친 학습 index — gravity 프레임은 VOBench_v1/gravity 아래라 file_name 앞에 'gravity/' 를 붙인다
    vob = list(csv.DictReader(VOB_TRAIN.open()))
    cols = list(dict.fromkeys(list(vob[0].keys()) + list(rows[0].keys()) + ["source"]))
    comb = [{**r, "source": "vobench_split2_train"} for r in vob] + \
           [{**r, "file_name": "gravity/" + r["file_name"], "block_id": "g" + r["block_id"], "source": "gravity_split_train"} for r in tr]
    OUT_C.mkdir(parents=True, exist_ok=True)
    with open(OUT_C / "index_train.csv", "w", newline="") as f:
        w = csv.DictWriter(f, cols, restval=""); w.writeheader(); w.writerows(comb)
    print(f"[합침] VOBench split v2 train {len(vob)} + gravity train {len(tr)} = {len(comb)} clip → {OUT_C / 'index_train.csv'}")


if __name__ == "__main__":
    main()
