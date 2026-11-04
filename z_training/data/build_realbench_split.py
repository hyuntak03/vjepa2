#!/usr/bin/env python3
"""real world permanence (95 영상 × 4) 를 영상 단위로 학습 / 시험 절반으로 나눈다 (2026-11-02, 사용자: "real bench 로 PFT · held out").

나누기: 출처 (RubricOffice 20 · MAGICONTROL 10 · PerceptionTest 50 · HOOT 2 · PhysicsIQ 13) 마다 영상을 seed 0 으로 섞어 앞 절반 (내림) = 학습.
  한 영상의 네 변이 (가능 있음 · 가능 없음 · 사라짐 · 나타남) 는 같은 쪽에 간다 — 문맥이 같으므로 (CLAUDE.md §1-5 block 단위 split).
학습 목록: 학습 쪽 영상의 **가능 영상 두 개만** (possible_present · possible_absent). 불가능 영상은 학습에 안 쓴다.
  창 = 채점과 같은 프레임 번호 (index 의 frames 열, 문맥 [0, t−1] 균일 16 · 미래 [t, n−1] 균일 16).
출력 (data_csv/real_world_permanence_uniform/):
  index_train.csv · index_test.csv (채점 index 와 같은 열, 쪽마다) · train_possible_mp4list.csv ("<mp4 절대경로>\\t<번호 32 개>") · split.json
  /venv/vjepa/bin/python z_training/data/build_realbench_split.py
"""
import csv
import json
import os
import random
from collections import defaultdict

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
D = f"{REPO}/data_csv/real_world_permanence_uniform"
SRC = "/workspace/dataset/benchmarks/real_world_permanence"


def main(seed=0):
    rows = list(csv.DictReader(open(f"{D}/index.csv")))
    by_src = defaultdict(set)
    for r in rows:
        by_src[r["source"]].add(r["source_block"])
    rng = random.Random(seed)
    train = set()
    for s in sorted(by_src):
        v = sorted(by_src[s]); rng.shuffle(v)
        train |= set(v[: len(v) // 2])
    tr = [r for r in rows if r["source_block"] in train]
    te = [r for r in rows if r["source_block"] not in train]
    for name, part in (("index_train.csv", tr), ("index_test.csv", te)):
        with open(f"{D}/{name}", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(part)
    pos = [r for r in tr if r["plausible"] == "1"]
    for r in pos:
        assert os.path.isfile(f"{SRC}/{r['file']}"), r["file"]
        assert len(r["frames"].split()) == 32, r["video_id"]
    with open(f"{D}/train_possible_mp4list.csv", "w") as f:
        f.writelines(f"{SRC}/{r['file']}\t{r['frames']}\n" for r in pos)
    cnt = {s: {"train": len([b for b in by_src[s] if b in train]), "test": len([b for b in by_src[s] if b not in train])} for s in sorted(by_src)}
    info = {"seed": seed, "videos_train": len(train), "videos_test": len(by_src_all := set().union(*by_src.values())) - len(train),
            "train_possible_clips": len(pos), "test_pairs": len(te) // 2, "per_source": cnt}
    json.dump(info, open(f"{D}/split.json", "w"), indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
