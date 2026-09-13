#!/usr/bin/env python3
"""IntPhysGen v11 을 **block 단위** train / test 로 나눈다 (predictor 학습용).

  python z_training/data/build_v11_split_index.py                     # v11_full(12조건) 50/50, seed 0
  python z_training/data/build_v11_split_index.py --source v11 --train-frac 0.5 --seed 0
  python z_training/data/build_v11_split_index.py --holdout-shape torus,cone --holdout-color red

출력 (data_csv/intphysgen_v11_split/, csv 는 .gitignore 라 untracked 가 정상)
  index.csv        전 행 + `split` (train/test) + `holdout` (0/1) 컬럼. index_probe 의 라벨 컬럼도 그대로 둔다
  index_train.csv  split=train **이면서 plausible=1** 인 행만  -> 학습 (configs/training/datasets.md ## v11_split_train)
  index_test.csv   split=test 인 행 전부(가능+불가능, block 온전)   -> 채점 (configs/protocols/datasets.md ## v11_split_test)
  split_report.json 셀별 block 수와 검증 결과

왜 block 단위인가 — block 안 4 클립은 문맥을 2x2 로 공유한다 (CLAUDE.md §1-5). 클립 단위로 나누면
test 의 possible 미래가 train 에 들어가 matched-pair 점수가 학습셋 점수가 된다.
층화: (condition, violation_type, sym_k) 셀마다 block 을 섞어 train_frac 만큼 train.
holdout: 지정한 shape/color 가 pre 나 post 에 등장하는 block 은 전부 test 로 보낸다 (안 본 외형 일반화 검사).
  그 block 은 index.csv 의 holdout=1 이고, index_test.csv 에도 들어간다. holdout 이 있으면 index_test_holdout.csv 를 따로 낸다.
검증: block 4행이 같은 split, pair_id 마다 가능1+불가능1, train 에 불가능 0, test block 이 train 과 안 겹침.
"""
import argparse
import csv
import json
import os
import random
from collections import Counter, defaultdict

ROOT = "/data/hyuntak/project/2026/2027_cvpr/vjepa2"
ap = argparse.ArgumentParser()
ap.add_argument("--source", default="v11_full", choices=["v11_full", "v11"])
ap.add_argument("--train-frac", type=float, default=0.5)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--holdout-shape", default="", help="콤마 구분. pre/post 어느 쪽이든 등장하면 test")
ap.add_argument("--holdout-color", default="")
ap.add_argument("--out", default=f"{ROOT}/data_csv/intphysgen_v11_split")
a = ap.parse_args()

src = f"{ROOT}/data_csv/intphysgen_{a.source}/index_probe.csv"    # sym_k / shape / color 가 여기만 있다
with open(src, newline="", encoding="utf-8") as f:
    rd = csv.DictReader(f)
    cols = [c for c in rd.fieldnames if c != "probe_type"]
    rows = list(rd)
hs = {s for s in a.holdout_shape.split(",") if s}
hc = {c for c in a.holdout_color.split(",") if c}

blocks = defaultdict(list)
for r in rows:
    blocks[r["block_id"]].append(r)
strata = {}
holdout = set()
for b, rs in blocks.items():
    keys = {(r["condition"], r["violation_type"], r["sym_k"]) for r in rs}
    assert len(keys) == 1, f"block {b}: strata 가 섞였다 {keys}"
    strata[b] = next(iter(keys))
    if any(r["shape_pre"] in hs or r["shape_post"] in hs or r["color_pre"] in hc or r["color_post"] in hc for r in rs):
        holdout.add(b)

by_cell = defaultdict(list)
for b, k in strata.items():
    if b not in holdout:
        by_cell[k].append(b)
train = set()
for k in sorted(by_cell):
    bs = sorted(by_cell[k], key=int)
    random.Random(a.seed).shuffle(bs)
    train |= set(bs[: int(round(len(bs) * a.train_frac))])

def split_of(b):
    return "train" if b in train else "test"

os.makedirs(a.out, exist_ok=True)
out_cols = cols + ["split", "holdout"]
n = Counter()
with open(f"{a.out}/index.csv", "w", newline="") as f_all, \
     open(f"{a.out}/index_train.csv", "w", newline="") as f_tr, \
     open(f"{a.out}/index_test.csv", "w", newline="") as f_te, \
     open(f"{a.out}/index_test_holdout.csv", "w", newline="") as f_ho:
    W = {k: csv.DictWriter(fh, fieldnames=out_cols, extrasaction="ignore") for k, fh in
         (("all", f_all), ("train", f_tr), ("test", f_te), ("ho", f_ho))}
    for w in W.values():
        w.writeheader()
    for r in rows:                                     # 원본 순서 유지
        b = r["block_id"]
        r = dict(r, split=split_of(b), holdout=str(int(b in holdout)))
        W["all"].writerow(r)
        if r["split"] == "train" and r["plausible"] == "1":
            W["train"].writerow(r); n["train_pos"] += 1
        elif r["split"] == "train":
            n["train_imp_dropped"] += 1
        else:
            W["test"].writerow(r); n["test"] += 1
            if b in holdout:
                W["ho"].writerow(r); n["test_holdout"] += 1
if not holdout:
    os.remove(f"{a.out}/index_test_holdout.csv")

# ── 검증 ──────────────────────────────────────────────────────────────────────
tr = list(csv.DictReader(open(f"{a.out}/index_train.csv")))
te = list(csv.DictReader(open(f"{a.out}/index_test.csv")))
assert all(r["plausible"] == "1" for r in tr), "train 에 불가능 변이가 있다"
assert not ({r["block_id"] for r in tr} & {r["block_id"] for r in te}), "train/test block 겹침"
teb = defaultdict(list)
for r in te:
    teb[r["block_id"]].append(r)
for b, rs in teb.items():
    assert len(rs) == 4, f"test block {b} 가 {len(rs)}행 (4 여야 한다)"
    for pid, prs in defaultdict(list, {p: [x for x in rs if x["pair_id"] == p] for p in {x["pair_id"] for x in rs}}).items():
        assert sum(x["plausible"] == "1" for x in prs) == 1 and sum(x["plausible"] != "1" for x in prs) == 1, f"test block {b} pair {pid} 가 1+1 이 아니다"
cell_tab = {}
for k in sorted(by_cell):
    bs = by_cell[k]
    cell_tab["/".join(k)] = {"train": sum(b in train for b in bs), "test": sum(b not in train for b in bs)}
rep = {"source": src, "train_frac": a.train_frac, "seed": a.seed, "holdout_shape": sorted(hs), "holdout_color": sorted(hc),
       "n_blocks": len(blocks), "n_blocks_train": len(train), "n_blocks_test": len(blocks) - len(train),
       "n_blocks_holdout": len(holdout), "rows": dict(n), "cells": cell_tab, "verified": True}
json.dump(rep, open(f"{a.out}/split_report.json", "w"), indent=1, ensure_ascii=False)
print(f"blocks {len(blocks)} -> train {len(train)} / test {len(blocks)-len(train)} (holdout {len(holdout)})")
print(f"rows: train possible {n['train_pos']} (불가능 {n['train_imp_dropped']} 버림) | test {n['test']} (= {n['test']//2} matched pair)"
      + (f" | test_holdout {n['test_holdout']}" if holdout else ""))
print(f"cells {len(cell_tab)}, 검증 통과 -> {a.out}/split_report.json")
