#!/usr/bin/env python3
"""IntPhys 1 late occ (uniform-d) 창으로 릴리즈 predictor post-FT — dev 를 block 단위 2-fold 로 (2026-10-21, 사용자: "dev set 으로 그냥 학습해").

창 = 평가와 똑같다: data_csv/intphys1_late_uni (cvpr/eval/intphys1_late/build.py --uniform-d) 의 32 장 —
  문맥 16 장 = [0, d−1] 균일 · 미래 16 장 = [d, 99] 균일 (d = 쌍의 위반 시점). 학습은 그 32 장을 stride 1 로 그대로 읽는다.
split = block (block · quadruplet = 4 영상 = 2 쌍) 단위, (block_type × group) 마다 반반 (seed 0) → fold 0 / 1.
  fold k 학습 = fold k block 의 가능 영상 (학습기가 plausible == 1 만 고른다) · fold k 채점 = 반대 fold block 의 쌍 전부.
  두 fold 의 채점을 합치면 180 쌍 (가림 90) 전부가 held-out 점수를 한 번씩 받는다.
⚠️ IntPhys 1 dev 는 평가 세트다 — 이 학습 뒤 점수는 각 fold 의 held-out block 으로만 읽는다 (dev 전체 · IntPhys 1 sliding 칸은 오염).
프레임: --copy 면 /local_datasets/…/IntPhys1_late_uni 의 링크를 풀어 data_gen/IntPhys1_late_uni (NFS) 로 복사 (다른 노드에서 학습 · 채점하려고).

  python z_training/data/build_intphys1_late_uni_cv.py [--copy]
  → data_csv/intphys1_late_uni_cv/{fold0,fold1}_{train,test}.csv · split.json"""
import argparse, collections, csv, json, os, random, shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "data_csv/intphys1_late_uni/index.csv"
OUT = REPO / "data_csv/intphys1_late_uni_cv"
FR_SRC = Path("/local_datasets/world/world_analysis/IntPhys1_late_uni")
FR_DST = REPO / "data_gen/IntPhys1_late_uni"

ap = argparse.ArgumentParser(); ap.add_argument("--copy", action="store_true"); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
rows = list(csv.DictReader(open(SRC)))
blk = collections.OrderedDict()
for r in rows:
    blk.setdefault(r["block_id"], []).append(r)
strat = collections.defaultdict(list)
for b, rs in blk.items():
    key = (rs[0]["block_type"], rs[0]["group"])
    assert all((r["block_type"], r["group"]) == key for r in rs), f"block {b} 안에서 층이 갈린다"
    strat[key].append(b)
rng = random.Random(a.seed); fold = {}
for i, (key, bs) in enumerate(sorted(strat.items())):
    bs = sorted(bs, key=int); rng.shuffle(bs)
    h = len(bs) // 2 + (len(bs) % 2) * (i % 2)                    # 홀수 층은 fold 를 번갈아 하나 더
    for j, b in enumerate(bs):
        fold[b] = 0 if j < h else 1
OUT.mkdir(parents=True, exist_ok=True)
cols = list(rows[0].keys())
summ = {}
for k in (0, 1):
    tr = [r for r in rows if fold[r["block_id"]] == k]
    te = [r for r in rows if fold[r["block_id"]] != k]
    for name, rs in (("train", tr), ("test", te)):
        with open(OUT / f"fold{k}_{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rs)
    occ = lambda rs: sum(r["plausible"] == "1" and r["group"].endswith("occluded") for r in rs)
    summ[f"fold{k}"] = {"train_blocks": len({r['block_id'] for r in tr}), "train_possible_videos": sum(r["plausible"] == "1" for r in tr),
                        "test_blocks": len({r['block_id'] for r in te}), "test_pairs": sum(r["plausible"] == "1" for r in te), "test_pairs_occluded": occ(te),
                        "by_stratum": {f"{bt} {g}": [sum(fold[b] == k for b in bs), sum(fold[b] != k for b in bs)] for (bt, g), bs in sorted(strat.items())}}
json.dump({"source": str(SRC.relative_to(REPO)), "seed": a.seed, "fold_of_block": fold, **summ}, open(OUT / "split.json", "w"), indent=1)
print(json.dumps(summ, indent=1))
if a.copy:
    n = 0
    for r in rows:
        d = FR_DST / r["video_id"]; d.mkdir(parents=True, exist_ok=True)
        for t in range(32):
            s = FR_SRC / r["video_id"] / f"{t:06d}.png"
            if not (d / s.name).exists():
                shutil.copyfile(os.path.realpath(s), d / s.name); n += 1
    print(f"복사 {n} 장 → {FR_DST}")
