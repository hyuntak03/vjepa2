#!/usr/bin/env python3
"""VOBench_v1 (+ gravity) 를 **처음 보는 물체** 로 나눈다 — 모양 · 색 · 메시를 train 에서 통째로 뺀다 (2026-10-21, 사용자:
"처음 보는 물체 · 모양 · 색에도 되는지 보고 싶다").

held-out 속성 (train 에 한 번도 안 나온다):
  primitive  모양 torus · cone (6 중 2)   ·   색 cyan · purple (8 중 2)
  realistic  메시 basketball_8d17 · scooter_32f1 (6 중 2)   ·   칠한 색 42D4F4 (cyan) · 911EB4 (purple)
배정 = split v2 와 같은 장면 묶음 (가능 arm 설정 키 (objset, 운동, 배경, 가림, k, 물체) 를 공유하는 block 을 union-find) 위에서,
  묶음 안 block 의 물체 (obj_a · obj_b) 중 하나라도 held-out 속성이면 묶음째 test, 아니면 train.
  → train 에 held-out 속성 0 (검사) · train 과 test 의 가능 arm 설정 키 공유 0 (검사).
gravity: 묶음 (objset, law, 배경, obj) 에서 obj 가 held-out 이면 test, 아니면 train (build_vobench_gravity_split_index.py 와 같은 묶음 키).
test 쌍의 분류 (index 열 holdout): shape (모양만 새것) · color (색만) · mesh (메시만) · both (둘 이상) · seen (held-out 묶음에 딸려 온 본 물체만의 block).
train = 가능 변이만 (pos_a · pos_b / gravity pos), test = 변이 전부.

  python z_training/data/build_vobench_objholdout_index.py [--copy]
  → data_csv/vobench_v1_objholdout/{index_train.csv (VOBench + gravity 가능, 학습용), index_test.csv, gravity_test.csv, split.json}
  --copy: train 영상 프레임 (원본 96 장 전부) 을 data_gen/VOBench_v1_objholdout (NFS) 로 — 다른 노드에서 학습하려고"""
import argparse, collections, csv, json, os, shutil, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "z_training/data"))
from build_vobench_split_index import arm_key                                   # noqa: E402  (split v2 와 같은 설정 키)

SRC = REPO / "data_csv/vobench_v1/index.csv"
SRC_G = REPO / "data_csv/vobench_v1_gravity/index.csv"
OUT = REPO / "data_csv/vobench_v1_objholdout"
ROOT = Path("/data2/local_datasets/world/Benchmark/VOBench_v1")
DST = REPO / "data_gen/VOBench_v1_objholdout"
H_SHAPE = {"torus", "cone", "basketball_8d17", "scooter_32f1"}
H_COLOR = {"cyan", "purple", "42D4F4", "911EB4"}


def tags(obj):
    if obj in ("", "empty"):
        return set()
    shape, color = obj.split("/")
    t = set()
    if shape in H_SHAPE: t.add("mesh" if "_" in shape else "shape")
    if color in H_COLOR: t.add("color")
    return t


def kind(objs):
    t = set().union(*[tags(o) for o in objs])
    return "seen" if not t else next(iter(t)) if len(t) == 1 else "both"


ap = argparse.ArgumentParser(); ap.add_argument("--copy", action="store_true"); a = ap.parse_args()
rows = list(csv.DictReader(open(SRC)))
by = collections.defaultdict(dict)
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
        if k[-1] == "empty": continue
        if k in owner: parent[find(b)] = find(owner[k])
        else: owner[k] = b
groups = collections.defaultdict(list)
for b in by: groups[find(b)].append(b)
objs = lambda b: {by[b]["pos_a"]["obj_a"], by[b]["pos_a"]["obj_b"]}
test = set()
for g in groups.values():
    if any(tags(o) for b in g for o in objs(b)): test |= set(g)
train_rows = [r for r in rows if r["block_id"] not in test and r["plausible"] == "1"]
test_rows = [dict(r, holdout=kind(objs(r["block_id"]))) for r in rows if r["block_id"] in test]
assert not any(tags(r["obj_a"]) or tags(r["obj_b"]) for r in train_rows), "train 에 held-out 속성"
ktr = {arm_key(r) for r in train_rows}; kte = {arm_key(r) for r in test_rows if r["variant"] in ("pos_a", "pos_b")}
assert not (ktr & kte) - {k for k in ktr if k[-1] == "empty"}, "train · test 가 가능 arm 설정을 공유"

G = list(csv.DictReader(open(SRC_G)))
gtest = {r["block_id"] for r in G if tags(r["obj"])}
g_train = [r for r in G if r["block_id"] not in gtest and r["plausible"] == "1"]
g_test = [dict(r, holdout=kind({r["obj"]})) for r in G if r["block_id"] in gtest]
assert not any(tags(r["obj"]) for r in g_train)

OUT.mkdir(parents=True, exist_ok=True)
gcols = ["video_id", "file_name", "block_id", "variant", "plausible", "objset", "obj_or_pair", "set"]
with open(OUT / "index_train.csv", "w", newline="") as f:                       # 학습기는 file_name · plausible 만 쓴다 — 두 세트를 한 파일로
    w = csv.writer(f); w.writerow(["file_name", "plausible", "set", "objset", "block_id"])
    for r in train_rows: w.writerow([r["file_name"], "1", "vobench", r["objset"], r["block_id"]])
    for r in g_train: w.writerow(["gravity/" + r["file_name"], "1", "gravity", r["objset"], "g" + r["block_id"]])
for name, rs in (("index_test.csv", test_rows), ("gravity_test.csv", g_test)):
    with open(OUT / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rs[0].keys())); w.writeheader(); w.writerows(rs)
cnt = lambda rs, key: dict(collections.Counter(key(r) for r in rs))
summ = {"heldout_shape_mesh": sorted(H_SHAPE), "heldout_color": sorted(H_COLOR),
        "vobench": {"train_blocks": len({r['block_id'] for r in train_rows}), "train_clips": len(train_rows), "test_blocks": len(test),
                    "test_pairs_by_holdout": cnt([r for r in test_rows if r["plausible"] == "1"], lambda r: r["holdout"]),
                    "test_pairs_late_by_holdout": cnt([r for r in test_rows if r["plausible"] == "1" and r["occlusion"] == "late"], lambda r: r["objset"] + " " + r["holdout"])},
        "gravity": {"train_clips": len(g_train), "test_blocks": len(gtest), "test_blocks_by_holdout": cnt([r for r in g_test if r["variant"] == "pos"], lambda r: r["objset"] + " " + r["holdout"])}}
json.dump(summ, open(OUT / "split.json", "w"), indent=1)
print(json.dumps(summ, indent=1))
if a.copy:
    n = 0
    for r in train_rows + [dict(file_name="gravity/" + g["file_name"]) for g in g_train]:
        s, d = ROOT / r["file_name"], DST / r["file_name"]
        d.mkdir(parents=True, exist_ok=True)
        for p in sorted(s.glob("*.png")):
            if not (d / p.name).exists(): shutil.copyfile(p, d / p.name); n += 1
    print(f"복사 {n} 장 → {DST}")
