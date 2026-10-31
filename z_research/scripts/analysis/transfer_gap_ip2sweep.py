#!/usr/bin/env python3
"""transfer_gap Q3 · E29 — IntPhys 2 문맥 끝 위치 스윕: 문맥 끝을 사람 라벨 위반 시작 d 의 δ 장 앞에 둘 때 VOBench PFT 이득 곡선. 2026-10-30, [X2] R2.

왜: IntPhys 2 문맥 끝 판에서 VOBench PFT 의 permanence 이득이 라벨 −10 판 +9.2 / 라벨 판 −0.8 이었다 (E28 재확인). 위반이 미래 앞쪽일 때만
    이득이 생기는가 (H3-W, δ ∈ [−20, −5] 에서만 +5 이상) 아니면 −10 판의 +9 는 창 잡음인가 (모든 δ 에서 ±5 안).
정의
  판 δ: d' = d + δ (d = 사람 라벨, 60 fps). 문맥 16 장 = round(linspace(0, d' − 1, 16)), 미래 16 장 = round(linspace(d', n − 1, 16)) (build.py --d-offset δ).
  δ ∈ {−60, −30, −20, −10, −5, 0}. −10 · 0 은 기존 판 (intphys2_late_uni_m10 · _uni), 나머지는 이번에 새로 만든 판.
  채점은 E27 의 한 번 통과 루프 (encoder 한 번 + 릴리즈 · VOBench PFT + 복사 copy_z; 고정 창 c16t32, matched) 로 모든 δ 를 같은 수치 경로에서.
  G(δ, 성질) = acc(PFT) − acc(릴리즈) 같은 쌍. CI = 장면 bootstrap 4,000. 이웃 δ 쌍 판정 일치율 (같은 모델).
  ⚠️ IntPhys 2 는 가능 · 불가능 문맥이 화소로 같지 않다 (build.py 가 context_integrity.json 에 차이를 적는다).
출력: 배열 /data2/local_datasets/world/world_analysis/cache/transfer_gap/E29/<판>/ · 결과 z_analysis/transfer_gap/results/q3_r2/E29_offset_sweep.json

  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P z_research/scripts/analysis/transfer_gap_ip2sweep.py score --deltas -60,-30,-20,-10,-5,0     (GPU 1, 판당 약 3 분)
  $P z_research/scripts/analysis/transfer_gap_ip2sweep.py report
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "z_research/scripts/analysis"))
sys.path.insert(0, str(REPO))
import numpy as np  # noqa: E402

import transfer_gap_offset as TO  # noqa: E402

C = Path("/data2/local_datasets/world/world_analysis/cache/transfer_gap")
OUT = REPO / "z_analysis/transfer_gap/results/q3_r2/E29_offset_sweep.json"


def panel(d: int):
    if d == 0:
        return ("intphys2_late_uni", [], 0, "external")
    if d == -10:
        return ("intphys2_late_uni_m10", [], 0, "external")
    return ("intphys2_late_uni", [f"data.root={REPO}/data_csv/transfer_gap_ip2late_d{d}", f"data.frames_root={C}/ip2late_d{d}"], 0, "external")


def score(deltas):
    TO.ARR = C / "E29"
    TO.ARR.mkdir(parents=True, exist_ok=True)
    cbar = np.load(C / "E27/cbar_vob_train.npy")
    for d in deltas:
        name = f"d{d}"
        if (TO.ARR / name / "meta.json").exists():
            print("skip", name); continue
        TO.SETS[name] = panel(d)
        TO.gpu_loop(name, cbar)


def report():
    import csv
    rng = np.random.default_rng(0)
    res = {"definition": __doc__.split("정의")[1].split("출력:")[0].strip(), "panels": {}, "neighbors": {}}
    J = {}
    for d in (-60, -30, -20, -10, -5, 0):
        od = C / "E29" / f"d{d}"
        if not (od / "meta.json").exists():
            continue
        meta = json.load(open(od / "meta.json"))
        s = np.load(od / "l1.npy").mean(2); cp = np.load(od / "copy.npy").mean(1)
        root = REPO / "data_csv" / (panel(d)[0] if d in (0, -10) else f"transfer_gap_ip2late_d{d}")
        prin = {r["video_id"]: r["condition"] for r in csv.DictReader(open(root / "index.csv"))}
        ix = {v: i for i, v in enumerate(meta["video_id"])}
        grp = defaultdict(dict)
        for i, (b, p, pl) in enumerate(zip(meta["block_id"], meta["pair_id"], meta["plausible"])):
            grp[(b, p)]["pos" if pl == "1" else "imp"] = i
        pairs = [(k, g["pos"], g["imp"]) for k, g in grp.items() if "pos" in g and "imp" in g]
        jd = {}
        for name, sv in (("release", s[:, 0]), ("vob_pft", s[:, 1]), ("copy_z", cp)):
            jd[name] = {k: (1.0 if sv[im] > sv[ip] else 0.5 if sv[im] == sv[ip] else 0.0) for k, ip, im in pairs}
        J[d] = jd
        props = {k: prin[meta["video_id"][ip]] for k, ip, im in pairs}
        P = {"n_pairs": len(pairs), "eps_pft": round(float(s[[g for _, g, _ in pairs], 1].mean() / s[[g for _, g, _ in pairs], 0].mean()), 4)}
        for cell in ["all"] + sorted(set(props.values())):
            keys = [k for k, _, _ in pairs if cell == "all" or props[k] == cell]
            row = {m: round(100 * float(np.mean([jd[m][k] for k in keys])), 2) for m in jd}
            diff = np.array([jd["vob_pft"][k] - jd["release"][k] for k in keys]); scn = np.array([k[0] for k in keys])
            u = np.unique(scn); idx = {x: np.flatnonzero(scn == x) for x in u}
            bs = [diff[np.concatenate([idx[u[q]] for q in rng.choice(len(u), len(u))])].mean() for _ in range(4000)]
            row.update({"n": len(keys), "G": round(100 * float(diff.mean()), 2), "G_ci": [round(100 * float(np.percentile(bs, q)), 2) for q in (2.5, 97.5)],
                        "release_minus_copy": round(row["release"] - row["copy_z"], 2)})
            P[cell] = row
        res["panels"][f"d{d}"] = P
        print(d, {c: (v["G"], v["G_ci"]) for c, v in P.items() if isinstance(v, dict)}, flush=True)
    ds = sorted(J)
    for a, b in zip(ds[:-1], ds[1:]):
        common = set(J[a]["release"]) & set(J[b]["release"])
        res["neighbors"][f"d{a}|d{b}"] = {m: round(100 * float(np.mean([J[a][m][k] == J[b][m][k] for k in common])), 1) for m in ("release", "vob_pft", "copy_z")} | {"n": len(common)}
    # 판정
    v = {}
    for prop in ("permanence", "continuity"):
        g = {d: res["panels"][d][prop]["G"] for d in res["panels"] if prop in res["panels"][d]}
        inside = [g[f"d{x}"] for x in (-20, -10, -5) if f"d{x}" in g]
        outside = [g[f"d{x}"] for x in (-60, -30, 0) if f"d{x}" in g]
        v[prop] = {"G_by_delta": g, "H3-W": bool(inside and min(inside) >= 5 and outside and max(abs(x) for x in outside) < 5),
                   "all_within_5": bool(all(abs(x) < 5 for x in g.values()))}
    res["verdict_inputs"] = v
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(res["neighbors"], ensure_ascii=False)); print(json.dumps(v, ensure_ascii=False)); print("→", OUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["score", "report"])
    ap.add_argument("--deltas", default="-60,-30,-20,-10,-5,0")
    a = ap.parse_args()
    score([int(x) for x in a.deltas.split(",")]) if a.part == "score" else report()


if __name__ == "__main__":
    main()
