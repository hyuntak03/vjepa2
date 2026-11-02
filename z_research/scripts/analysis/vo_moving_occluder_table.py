#!/usr/bin/env python3
"""VO_moving_occluder 채점 결과를 분해한다 (2026-11-01).

입력 (산출물에서 다시 계산한다 — summary.json 의 숫자를 옮겨 적지 않는다):
  - 모델마다 cvpr/results/eval/testbed/vo_moving_occluder_<model>/per_block.json 의 per_video_surprise
  - 복사 기준선 cvpr/results/eval/testbed/copy/vo_moving_occluder_vith/{meta.json, copy.npy}
  - 인덱스 data_csv/vo_moving_occluder/index.csv (objset · violation · k · motion · pair_id)

채점 = matched pair: 같은 block 안 문맥이 같은 (가능, 불가능) 쌍. 정답 = S(불가능) > S(가능), 동점 0.5. chance 50 %.
  A 방향 = (pos_a, imp_ab): 문맥에 A 가 보였다 → 미래가 B 로 바뀐다 (permanence 면 물체 → 빈 장면 = 사라짐)
  B 방향 = (pos_b, imp_ba): 문맥에 B → 미래가 A (permanence 면 빈 장면 → 물체 = 생겨남)
CI = block 단위 bootstrap 95 % (2000 회). Δ = 모델 − 복사, 같은 쌍 위에서.

  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P z_research/scripts/analysis/vo_moving_occluder_table.py vith vith_release_vobftg vith_vast_ar_state_ep70
"""
from __future__ import annotations
import csv, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
RES = REPO / "cvpr/results/eval/testbed"
DS = "vo_moving_occluder"
INDEX = REPO / "data_csv" / DS / "index.csv"
COPY = RES / "copy" / f"{DS}_vith"
N_BOOT, SEED = 2000, 0


def load_index():
    rows = list(csv.DictReader(INDEX.open()))
    pairs = defaultdict(dict)
    for r in rows:
        pairs[(r["block_id"], r["pair_id"])][r["plausible"]] = r
    out = []
    for (b, pid), d in sorted(pairs.items()):
        p, n = d["1"], d["0"]
        out.append(dict(block=b, pair=pid, pos=p["video_id"], imp=n["video_id"], objset=p["objset"], violation=p["violation"],
                        k=p["k"], motion=p["motion"], background=p["background"]))
    return out


def hits(pairs, S):
    s = np.array([[S[x["pos"]], S[x["imp"]]] for x in pairs])
    return np.where(s[:, 1] > s[:, 0], 1.0, np.where(s[:, 1] == s[:, 0], 0.5, 0.0))


def boot_ci(h, blocks):
    ub, inv = np.unique(blocks, return_inverse=True)
    sums, cnt = np.bincount(inv, weights=h), np.bincount(inv)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(ub), (N_BOOT, len(ub)))
    v = sums[idx].sum(1) / cnt[idx].sum(1) * 100
    return np.percentile(v, [2.5, 97.5])


def main():
    models = sys.argv[1:] or ["vith", "vith_release_vobftg", "vith_vast_ar_state_ep70"]
    pairs = load_index()
    blocks = np.array([x["block"] for x in pairs])
    H = {}
    for m in models:
        f = RES / f"{DS}_{m}" / "per_block.json"
        if not f.exists():
            print(f"!! 없음: {f}"); continue
        S = json.load(f.open())["per_video_surprise"]
        miss = [x for x in pairs if x["pos"] not in S or x["imp"] not in S]
        assert not miss, f"{m}: surprise 없는 쌍 {len(miss)}"
        H[m] = hits(pairs, S)
    if (COPY / "copy.npy").exists():
        meta = json.load((COPY / "meta.json").open())
        done = np.load(COPY / "done.npy").astype(bool)
        assert done.all(), f"복사 기준선 미완료 clip {(~done).sum()}"
        c = np.load(COPY / "copy.npy").mean(1)
        H["copy"] = hits(pairs, dict(zip(meta["video_id"], c.tolist())))
    else:
        print(f"!! 복사 기준선 없음: {COPY}")

    groups = [("전체", lambda x: True)]
    for o in ("primitive", "realistic"):
        groups.append((o, lambda x, o=o: x["objset"] == o))
    for o in ("primitive", "realistic"):
        for v in ("permanence", "shape", "color"):
            groups.append((f"{o}/{v}", lambda x, o=o, v=v: x["objset"] == o and x["violation"] == v))
    for v in ("permanence", "shape", "color"):
        for d in ("A", "B"):
            groups.append((f"{v} {d}", lambda x, v=v, d=d: x["violation"] == v and x["pair"] == d))
    for k in ("1", "2", "3"):
        groups.append((f"k={k}", lambda x, k=k: x["k"] == k))
    for mo in sorted({x["motion"] for x in pairs}):
        groups.append((mo, lambda x, mo=mo: x["motion"] == mo))

    cols = list(H)
    print(f"{'group':<24}{'n':>5}  " + "".join(f"{c:>26}" for c in cols))
    for name, fn in groups:
        sel = np.array([fn(x) for x in pairs])
        cells = []
        for c in cols:
            h = H[c][sel]
            lo, hi = boot_ci(h, blocks[sel])
            cells.append(f"{h.mean() * 100:6.1f} [{lo:5.1f},{hi:5.1f}]")
        print(f"{name:<24}{sel.sum():>5}  " + "".join(f"{s:>26}" for s in cells))
    if "copy" in H:
        print("\nΔ = 모델 − 복사 (pt, 같은 쌍), block bootstrap 95 % CI")
        for name, fn in groups[:9]:
            sel = np.array([fn(x) for x in pairs])
            row = []
            for c in cols:
                if c == "copy":
                    continue
                d = (H[c] - H["copy"])[sel]
                lo, hi = boot_ci(d, blocks[sel])   # 차의 평균도 같은 식 (합/개수)
                row.append(f"{c}: {d.mean() * 100:+5.1f} [{lo:+5.1f},{hi:+5.1f}]")
            print(f"  {name:<22} " + " | ".join(row))


if __name__ == "__main__":
    main()
