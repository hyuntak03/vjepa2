#!/usr/bin/env python3
"""v3 시험 사라짐 (A) 짝이 왜 100 % 가 안 되나 (2026-11-01) — 틀린 짝의 장면 특징을 라벨로 나눠 본다.

짝 A = (pos_a: target 이 가림 뒤에서 다시 나옴, imp_ab: 같은 문맥, 미래에 target 없음). 정답 = s(imp_ab) > s(pos_a),
s = 미래 8 블록 평균 L1 (표준 surprise). 특징은 pos_a 의 target 라벨 (64×64 칸, 프레임 64 장) 에서:
  다시 보임 지연 = 프레임 32 (문맥 끝 다음) 부터 target 이 처음 절반 넘게 보이는 프레임까지 (프레임 4 장 = 미래 블록 1 개)
  미래 보이는 양 = 프레임 32–62 (짝수, 모델이 보는 16 장) 의 보이는 칸 수 평균 (64×64 = 4,096 칸 중)
  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python; $P predictor_lab/v3_ablation/analyze_vanish.py [--tag v3_full_s0]
"""
import argparse
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
TE = REPO / "predictor_lab/results/v3_ablation/te"


def load(tag):
    for d in sorted(TE.glob("v3test_*"), key=lambda d: (d / "meta.json").stat().st_mtime, reverse=True):
        if not (d / "done.npy").exists() or not np.load(d / "done.npy").all():
            continue
        m = json.load(open(d / "meta.json"))
        if tag in m["pred_tags"] or tag == "copy":
            l1 = np.load(d / "copy.npy") if tag == "copy" else np.load(d / "l1.npy")[:, m["pred_tags"].index(tag)]
            return m, l1, d.name
    raise SystemExit(f"{tag} 결과 없음")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="v3_full_s0"); ap.add_argument("--also", nargs="*", default=["release", "copy"])
    a = ap.parse_args()
    idx = {r["video_id"]: r for r in csv.DictReader(open(REPO / "data_csv/blender_track_v3/voe_test/index.csv"))}
    res = {}
    for tag in [a.tag] + a.also:
        m, l1, src = load(tag)
        pos = {v: i for i, v in enumerate(m["video_id"])}
        rows = []
        for v, r in idx.items():
            if r["variant"] != "pos_a":
                continue
            ip, ii = pos[v], pos[v.replace("_pos_a", "_imp_ab")]
            lab = np.load(Path(r["file"]).with_suffix(".npz"))
            vis, am = lab["visible_n"].astype(float), lab["amodal_n"].astype(float)
            frac = np.where(am > 0, vis / np.maximum(am, 1), 0.0)
            fut = np.arange(32, 63, 2)
            seen = np.where(frac[32:] > 0.5)[0]
            rows.append(dict(r=r, ok=float(l1[ii].mean() > l1[ip].mean()), d=l1[ii] - l1[ip],
                             delay=int(seen[0]) if len(seen) else 99, vis=float(vis[fut].mean()), size=float(np.median(am[am > 0])) if (am > 0).any() else 0.0,
                             out=bool(am[fut].max() == 0)))
        res[tag] = (rows, src)
    rows, src = res[a.tag]
    print(f"== {a.tag} ({src}) 사라짐 A 짝 {len(rows)} 개, 정답률 {100 * np.mean([x['ok'] for x in rows]):.1f} %  |  비교: " +
          " · ".join(f"{t} {100 * np.mean([x['ok'] for x in res[t][0]]):.1f}" for t in a.also))

    def table(name, key, bins=None):
        g = defaultdict(list)
        for k, x in enumerate(rows):
            v = key(x)
            if bins is not None:
                v = next(lab for lo, hi, lab in bins if lo <= v < hi)
            g[v].append(k)
        print(f"\n[{name}]  칸: n · {a.tag} · " + " · ".join(a.also))
        for v in (sorted(g) if bins is None else [b[2] for b in bins if b[2] in g]):
            ks = g[v]
            print(f"  {str(v):26s} {len(ks):5d} · {100 * np.mean([rows[k]['ok'] for k in ks]):5.1f} · " +
                  " · ".join(f"{100 * np.mean([res[t][0][k]['ok'] for k in ks]):5.1f}" for t in a.also))

    table("다시 보임 지연 (문맥 끝 뒤 프레임; 4 장 = 미래 블록 1)", lambda x: x["delay"],
          [(0, 4, "0–3 (블록 0)"), (4, 8, "4–7 (블록 1)"), (8, 12, "8–11 (블록 2)"), (12, 20, "12–19 (블록 3–4)"), (20, 32, "20–31 (블록 5–7)"), (32, 100, "미래 안에 안 보임")])
    table("미래 평균 보이는 칸 (4,096 칸 중)", lambda x: x["vis"],
          [(0, 10, "<10"), (10, 25, "10–25"), (25, 50, "25–50"), (50, 100, "50–100"), (100, 1e9, "≥100")])
    table("target 크기 (가림 없는 칸 수 중앙)", lambda x: x["size"], [(0, 40, "<40"), (40, 70, "40–70"), (70, 110, "70–110"), (110, 1e9, "≥110")])
    for f in ("camera", "occ_type", "occ_motion", "target_mode", "place", "f_hide", "f_count", "objset", "occ_sub"):
        table(f, lambda x, f=f: x["r"][f])
    # 블록별 차이 (imp − pos): 틀린 짝은 어느 블록에서 차이가 안 나나
    D = np.array([x["d"] for x in rows]); ok = np.array([x["ok"] for x in rows]) > 0
    print("\n[미래 블록별 L1 차 (imp − pos) 평균, ×1e3]  블록 0 … 7")
    print("  맞힌 짝 ", " ".join(f"{v * 1e3:6.2f}" for v in D[ok].mean(0)))
    print("  틀린 짝 ", " ".join(f"{v * 1e3:6.2f}" for v in D[~ok].mean(0)))


if __name__ == "__main__":
    main()
