#!/usr/bin/env python3
"""VOBench_v2 학습 · 평가 분할 — 궤적 묶음 단위 (2026-10-31, z_analysis/release binding 학습).

VOBench 의 궤적 · 가림막은 (objset, background, motion, occlusion, k) 로만 정해진다 (위반 종류 · replica · 물체와 무관,
VOBench_v2/README.md). 그래서 그 다섯을 공유하는 장면을 한 묶음으로 두고 묶음째 train / test 에 배정한다 — test 와 공유 궤적 0
(VOBench_v1 split v2 와 같은 원칙). (objset, occlusion, k) 층마다 묶음 (배경 2 × 운동 6 = 12, visible 은 k 0) 을 seed 0 으로
섞어 절반을 train 으로.

출력: data_csv/vobench_v2_split/index_train.csv  — train 묶음의 **가능 변이** (학습용)
      data_csv/vobench_v2_split/index_test.csv   — test 묶음의 **네 변이 전부** (채점용 matched 짝)
      data_csv/vobench_v2_split/split.json        — 묶음 목록 · 개수
  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python; $P z_training/data/build_vobench_v2_split_index.py
"""
import csv, json, os
from collections import defaultdict
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
SRC = os.path.join(REPO, "data_csv/vobench_v2/index.csv")
OUT = os.path.join(REPO, "data_csv/vobench_v2_split")


def group_of(r):
    return "|".join(r[k] for k in ("objset", "background", "motion", "occlusion", "k"))


def main():
    rows = list(csv.DictReader(open(SRC)))
    strata = defaultdict(set)
    for r in rows:
        strata[(r["objset"], r["occlusion"], r["k"])].add(group_of(r))
    train_g = set()
    for key in sorted(strata):
        g = sorted(strata[key]); np.random.default_rng(0).shuffle(g)
        train_g |= set(g[: len(g) // 2])
    tr = [r for r in rows if group_of(r) in train_g and r["plausible"] == "1"]
    te = [r for r in rows if group_of(r) not in train_g]
    os.makedirs(OUT, exist_ok=True)
    for name, rr in (("index_train.csv", tr), ("index_test.csv", te)):
        with open(os.path.join(OUT, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rr)
    ng = len({group_of(r) for r in rows})
    info = dict(rule="묶음 = (objset, background, motion, occlusion, k), (objset, occlusion, k) 층마다 seed 0 절반 → train",
                n_groups=ng, n_train_groups=len(train_g), n_train_clips=len(tr), n_test_clips=len(te),
                n_test_blocks=len({r["block_id"] for r in te}), train_groups=sorted(train_g))
    json.dump(info, open(os.path.join(OUT, "split.json"), "w"), indent=1, ensure_ascii=False)
    print({k: v for k, v in info.items() if k != "train_groups"})


if __name__ == "__main__":
    main()
