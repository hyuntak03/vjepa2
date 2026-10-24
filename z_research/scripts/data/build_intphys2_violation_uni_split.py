#!/usr/bin/env python3
"""IntPhys2 violation-uni (위반 시작 라벨로 문맥 16 · 미래 16) 의 장면 단위 반반 split (2026-10-22, 사용자: release · AR 을 이 규칙으로 IntPhys2 post-FT).

  /venv/vjepa/bin/python z_research/scripts/data/build_intphys2_violation_uni_split.py

장면 (condition, SceneIndex) 단위로 반반 (seed 0, condition 층화) — 같은 장면의 두 pair 는 같은 쪽에 간다 (배경 · 물체가 같다).
쓰는 파일 (data_csv/intphys2_violation_uni/):
  index_train.csv     train 장면의 전 행 (학습 로더가 plausible == 1 만 쓴다). file_name = video_id (PNG 폴더 이름)
  index_test.csv      test 장면의 전 행 — 채점 (PNG)
  index_mp4_test.csv  같은 test 행, mp4 직독판 (index_mp4.csv 에서)"""
import collections, csv, os, random
D = "data_csv/intphys2_violation_uni"
rows = list(csv.DictReader(open(f"{D}/index.csv")))
mp4 = {r["video_id"]: r for r in csv.DictReader(open(f"{D}/index_mp4.csv"))}
by = collections.defaultdict(set)
for r in rows:
    by[r["block_type"]].add(r["scene"])
train = set()
for c in sorted(by):
    s = sorted(by[c]); random.Random(0).shuffle(s)
    train |= {(c, x) for x in s[: len(s) // 2]}
tr = [dict(r, file_name=r["video_id"]) for r in rows if (r["block_type"], r["scene"]) in train]
te = [r for r in rows if (r["block_type"], r["scene"]) not in train]
def w(name, rs):
    with open(f"{D}/{name}", "w", newline="") as f:
        x = csv.DictWriter(f, fieldnames=list(rs[0])); x.writeheader(); x.writerows(rs)
w("index_train.csv", tr); w("index_test.csv", te); w("index_mp4_test.csv", [mp4[r["video_id"]] for r in te])
cnt = lambda rs: dict(collections.Counter(r["block_type"] for r in rs if r["variant"] == "pos"))
print(f"train 장면 {len(train)} · 가능 영상 {sum(r['variant'] == 'pos' for r in tr)} {cnt(tr)}")
print(f"test 쌍 {sum(r['variant'] == 'pos' for r in te)} {cnt(te)}")
