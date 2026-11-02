#!/usr/bin/env python3
"""VOBench_v1 late 블록의 장면 통계 (2026-11-01) — v2 late 학습셋과 같은 기준으로 비교하려고.
label.json 의 프레임별 물체 중심 (u, v) · 반지름 r_px · 범위 ext_u/v · 가려진 비율 (512 px) → 256 px 기준으로 환산.
  지름 = 2·ext/2 (가로 · 세로 중 큰 쪽, 256 px) · 속도 = 문맥 끝 (raw 30–45) 중심 이동 px/s (256 px) · 가림 길이 = 2k 샘플
  python vobench_late_scene_stats.py --root /local_datasets/world/testbed/VOBench_v1 --out <json>   (vll7: 데이터가 그 노드 로컬)
"""
import argparse, csv, json, os
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
rows = [r for r in csv.DictReader(open(f"{a.root}/blocks.csv")) if r["occlusion"] == "late"]
out = []
for r in rows:
    d = json.load(open(f"{a.root}/{r['path']}/label.json"))
    L = d["labels"]["A"] if d["labels"]["A"] else d["labels"]["B"]
    u = np.array([x["u"] for x in L]) / 2; v = np.array([x["v"] for x in L]) / 2
    eu = np.array([x["ext_u"] for x in L]); ev = np.array([x["ext_v"] for x in L])
    sp = np.hypot(np.diff(u), np.diff(v)) * 30.0
    out.append({"block": r["block_id"], "objset": r["objset"], "motion": r["motion"], "k": int(r["k"]), "violation": r["violation"],
                "background": r["background"], "diam_px": float(np.median(np.maximum(eu, ev)[:48])),
                "speed_ctx_end_px_s": float(np.median(sp[30:45])), "speed_max_px_s": float(sp.max()),
                "hidden_samples": 2 * int(r["k"]), "cam_pitch": d["block"]["camera"]["pitch"], "cam_z_cm": d["block"]["camera"]["loc"][2],
                "occluder": d["block"]["params"]["occluder"].get("moving")})
json.dump(out, open(a.out, "w"), indent=1)
print(len(out), "blocks")
