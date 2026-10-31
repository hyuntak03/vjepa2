#!/usr/bin/env python3
"""CLEVRER 영속성 v1 채점표 — 릴리즈 · VOBench PFT (vobftg) · 복사 기준선, 쌍 정답률 (2026-10-31, z_analysis/release/07_clevrer_perm).

입력: z_analysis/release/07_clevrer_perm/te/ (transfer_gap_te 한 번 통과: l1 (n,K,8) · copy (n,8) · meta)
쌍: 같은 장면의 A = present ↔ vanish (사라짐) · B = absent ↔ appear (생겨남). 판정 = s(불가능) > s(가능), 같으면 0.5. s = 미래 8 블록 평균 L1.
나눔: 전체 · 운동 (static · behind · overtake · cross) · 가림막 종류 (object · screen) · 창 기준 (hidden · reveal). CI = 장면 재표집 2,000 (95 %).
  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python; $P z_research/scripts/analysis/clevrer_perm_table.py → results/clevrer_table.json
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "z_research/scripts/analysis"))
import transfer_gap_r4_matrix as MX  # noqa: E402

BASE = REPO / "z_analysis/release/07_clevrer_perm"


def main():
    d = BASE / "te"
    m, l1, cp = MX.load(d)
    idx = {r["video_id"]: r for r in csv.DictReader(open(REPO / "data_csv/clevrer_perm_v1/index.csv"))}
    by = defaultdict(dict)
    for i, vid in enumerate(m["video_id"]):
        r = idx[vid]
        by[(r["source_block"], r["pair_id"])]["pos" if r["plausible"] == "1" else "imp"] = i
    P = []
    for (sid, pid), v in by.items():
        r = idx[m["video_id"][v["pos"]]]
        P.append(dict(pos=v["pos"], imp=v["imp"], scene=sid, d=pid, motion=r["motion"], occ=r["occluder_kind"], anchor=r["occlusion"]))
    tags = list(m["pred_tags"])
    J = {t: MX.judge(l1[:, k], P) for k, t in enumerate(tags)}
    J["copy"] = MX.judge(cp, P)
    sc = np.array([p["scene"] for p in P])
    subsets = {"all": lambda p: True}
    for key in ("motion", "occ", "anchor"):
        for val in sorted({p[key] for p in P}):
            subsets[f"{key}={val}"] = (lambda k, v: (lambda p: p[k] == v))(key, val)
    out = {}
    print(f"{'부분집합':22s} {'방향':4s} {'n':>4s} | " + " ".join(f"{t:>8s}" for t in tags + ["copy"]) + " | vobftg−release · release−copy · vobftg−copy")
    for sname, f in subsets.items():
        for di in ("A", "B", "AB"):
            msk = np.array([f(p) and (di == "AB" or p["d"] == di) for p in P])
            if not msk.any():
                continue
            r = {"n": int(msk.sum())}
            for t in tags + ["copy"]:
                r[t] = MX.boot(J[t][msk], sc[msk])
            r["vobftg-release"] = MX.boot(J["vobftg"][msk], sc[msk], J["release"][msk])
            r["release-copy"] = MX.boot(J["release"][msk], sc[msk], J["copy"][msk])
            r["vobftg-copy"] = MX.boot(J["vobftg"][msk], sc[msk], J["copy"][msk])
            out[f"{sname}|{di}"] = r
            fmt = lambda x: f"{x[0]:+.1f} [{x[1]:+.0f},{x[2]:+.0f}]"
            print(f"{sname:22s} {di:4s} {r['n']:4d} | " + " ".join(f"{r[t][0]:8.1f}" for t in tags + ["copy"])
                  + f" | {fmt(r['vobftg-release'])} · {fmt(r['release-copy'])} · {fmt(r['vobftg-copy'])}")
    (BASE / "results").mkdir(parents=True, exist_ok=True)
    json.dump(out, open(BASE / "results/clevrer_table.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
