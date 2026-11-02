#!/usr/bin/env python3
"""BlenderTrack v2 test 도메인 안 점검 — 집계 (2026-11-01). blender_track_v2_indomain.py 의 shard*.npz + data_csv/blender_track_v2/test_cells.csv.

표 (전부 미래 8 블록의 오차 e = mean_d |p − LN(h)|, 낮을수록 좋다):
  전체          = 모든 토큰 평균 (표준 채점 기준)
  target 자리   = amodal 점유 비율 가중 (가려져도 물체가 있는 곳)
  가려진 동안   = target 이 가려진 (보이는 비율 < 0.1) 미래 블록의 target 자리 — 영속 (안 보이는 것) 의 도메인 안 척도
  보이는 동안   = target 이 보이는 (보이는 비율 ≥ 0.5) 미래 블록의 target 자리
  복사          = 마지막 문맥 블록을 미래에 그대로 (예측 없이)
  줄인 몫 (%)   = (릴리즈 − 그 predictor) / 릴리즈 × 100, clip 마다 차의 평균에 대한 95 % 구간 (clip 부트스트랩)
  $P z_research/scripts/analysis/blender_track_v2_indomain_table.py --out predictor_lab/results/v2_indomain
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
AXES = ["timing", "form", "form_sub", "motion", "camera", "count"]


def ci(d, rng, B=2000):
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return np.nan, np.nan, np.nan
    bs = rng.choice(d, (B, len(d))).mean(1)
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    Z = [np.load(f) for f in sorted(glob.glob(f"{a.out}/shard*.npz"))]
    models = [str(m) for m in Z[0]["models"]]
    clip = np.concatenate([z["clip"] for z in Z])
    e_all = np.concatenate([z["e_all"] for z in Z], 1)           # (M, N, 8)
    e_obj = np.concatenate([z["e_obj"] for z in Z], 1)
    c_all = np.concatenate([z["c_all"] for z in Z])               # (N, 8)
    c_obj = np.concatenate([z["c_obj"] for z in Z])
    hid = np.concatenate([z["hid"] for z in Z])                   # (N, 8) 1 − 보이는 비율
    occ = np.concatenate([z["occ_sum"] for z in Z])
    cells = {r["clip"]: r for r in csv.DictReader(open(f"{REPO}/data_csv/blender_track_v2/test_cells.csv"))}
    rows = [cells[str(c)] for c in clip]
    N = len(clip)
    has = occ > 0
    m_hid, m_vis = has & (hid >= 0.9), has & (hid <= 0.5)

    def masked_mean(x, m):                                        # (…, N, 8) · (N, 8) → (…, N) clip 마다 (해당 블록 없으면 nan)
        s = (np.where(m, x, 0)).sum(-1); n = m.sum(-1)
        return np.where(n > 0, s / np.maximum(n, 1), np.nan)
    met = {"전체": (e_all.mean(-1), c_all.mean(-1)),
           "target 자리": (masked_mean(e_obj, has), masked_mean(c_obj, has)),
           "가려진 동안 target 자리": (masked_mean(e_obj, m_hid), masked_mean(c_obj, m_hid)),
           "보이는 동안 target 자리": (masked_mean(e_obj, m_vis), masked_mean(c_obj, m_vis))}
    rng = np.random.default_rng(0)
    rel = models.index("release")
    out, md = {"n_clips": int(N), "models": models}, []
    md.append(f"# BlenderTrack v2 test 도메인 안 오차 (clip {N:,}, 미래 8 블록, e = mean|p − LN(h)|, 낮을수록 좋음)\n")
    for name, (E, C) in met.items():
        md.append(f"\n## {name}\n")
        md.append("| 묶음 | n | 복사 | " + " | ".join(models) + " | " + " | ".join(f"{m} 줄인 몫 % [95 %]" for m in models if m != "release") + " |")
        md.append("|---|---:|---:|" + "---:|" * len(models) + "---:|" * (len(models) - 1))
        groups = [("전체", np.ones(N, bool))]
        for ax in AXES:
            for lv in sorted({r[ax] for r in rows}):
                groups.append((f"{ax}={lv}", np.array([r[ax] == lv for r in rows])))
        out[name] = {}
        for gname, g in groups:
            ok = g & np.isfinite(C)
            if ok.sum() == 0:
                continue
            vals = [np.nanmean(E[i][ok]) for i in range(len(models))]
            imp = []
            for i, m in enumerate(models):
                if m == "release":
                    continue
                d = (E[rel][ok] - E[i][ok]) / np.nanmean(E[rel][ok]) * 100
                imp.append(ci(d, rng))
            out[name][gname] = {"n": int(ok.sum()), "copy": float(np.nanmean(C[ok])), **{m: float(v) for m, v in zip(models, vals)},
                                **{f"{m}_imp": [float(x) for x in im] for m, im in zip([m for m in models if m != "release"], imp)}}
            md.append(f"| {gname} | {int(ok.sum())} | {np.nanmean(C[ok]):.4f} | " + " | ".join(f"{v:.4f}" for v in vals) + " | "
                      + " | ".join(f"{x[0]:+.1f} [{x[1]:+.1f}, {x[2]:+.1f}]" for x in imp) + " |")
    # 칸 (324) 별: 줄인 몫 분포
    md.append("\n## 칸별 (타이밍 × 가림 형태 × 운동 × 카메라) 줄인 몫 % — 전체 토큰\n")
    keys = sorted({(r["timing"], r["form"], r["motion"], r["camera"]) for r in rows})
    K = np.array([(r["timing"], r["form"], r["motion"], r["camera"]) for r in rows])
    cellres = {}
    for m in models:
        if m == "release":
            continue
        i = models.index(m)
        v = []
        for k in keys:
            g = np.all(K == np.array(k), 1)
            v.append((np.mean(e_all.mean(-1)[rel][g]) - np.mean(e_all.mean(-1)[i][g])) / np.mean(e_all.mean(-1)[rel][g]) * 100)
        v = np.array(v); cellres[m] = v
        lo = [("|".join(keys[j]), round(float(v[j]), 1)) for j in np.argsort(v)[:5]]
        hi = [("|".join(keys[j]), round(float(v[j]), 1)) for j in np.argsort(v)[-5:][::-1]]
        md.append(f"- **{m}**: 칸 중앙값 {np.median(v):+.1f} % · 최소 {v.min():+.1f} · 최대 {v.max():+.1f} · 나빠진 칸 {int((v < 0).sum())} / {len(v)}")
        md.append(f"  - 가장 적게: {lo}")
        md.append(f"  - 가장 많이: {hi}")
        out.setdefault("cells", {})[m] = {"|".join(k): float(x) for k, x in zip(keys, v)}
    open(f"{a.out}/table.md", "w").write("\n".join(md) + "\n")
    json.dump(out, open(f"{a.out}/table.json", "w"), indent=1, ensure_ascii=False)
    print("\n".join(md))


if __name__ == "__main__":
    main()
