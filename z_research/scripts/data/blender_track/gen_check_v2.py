#!/usr/bin/env python3
"""BlenderTrack v2 렌더 전 검사 — 칸마다 설계를 뽑아 운동 물리 · 가림 타이밍 · 분포를 본다. 하나라도 어기면 종료 코드 1 (2026-11-01).

  · 설계: 칸 324 × checks.n_per_cell (train 분할의 칸 번호 그대로: 번호 = 칸 + 324 k)
  · 관문: 설계 실패 비율 ≤ checks.max_design_fail_rate (칸별 실패 > 20 % 는 thin_cells 로 보고) · 물리 위반 0 (경사로 쐐기 침범 포함) · 가리는 물체 화면 크기 비 ≤ 상한 × 1.1 (gen_design_v2.physics_check: 바닥 · 경사로 · 물체끼리 · 소품 · 운동 꼴 · 에너지)
          · 칸마다 개수 같음 · 고정 가림막 하위 종류마다 5 % 이상 · 재시도 평균 ≤ checks.max_attempt_mean
  · 보고: 칸 · 축별 재시도 · 하위 종류 · 가림 구간 (샘플) · target 화면 속력 · 크기 · 방해 물체 수
  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P gen_check_v2.py --config configs/v2.yaml --out <out>/check.json
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import design as D0  # noqa: E402
import gen_design_v2 as G  # noqa: E402

CFG = None


def _init(path):
    global CFG
    CFG = G.load_cfg(path, "train")


def one(i):
    c, nd = G.cell_of(CFG, i)
    t0 = time.time()
    try:
        s = G.sample_clip(CFG, "train", i, max_tries=1500, streams=10)
    except RuntimeError as e:
        return {"i": i, "cell": c, "fail": str(e)}
    bad = G.physics_check(s)
    o = s["objects"][0]
    T, fps, g = s["T"], s["fps"], s["g"]
    cam = D0.Cam(s["cam"]["h"], s["cam"]["pitch"], s["cam"]["lens"], size=s["size"])
    pos, _, R = G.track(o, T, fps, g)
    ctr = G.center_path(o, pos, R)
    u, v, d = cam.to_px(ctr)
    sp_px = float(np.median(np.hypot(np.diff(u), np.diff(v))) * fps)
    hs = G.obj_half(o)
    size_px = float(np.median(cam.f * 2 * np.sqrt(max(hs[0], hs[1]) * hs[2]) / np.maximum(d, 1e-3)))
    return {"i": i, "cell": c, "count": int(nd), "n_distractor": s["cell"]["n_distractor"], "sub": s["cell"]["form_sub"],
            "attempt": s["attempt"], "stream": s["stream"], "event": s["event_samples"], "bad": bad, "speed_px_s": sp_px,
            "size_px": size_px, "sec": time.time() - t0,
            "occ_ratio": next((float(x["screen_ratio"]) for x in s["objects"] if x.get("role") == "occluder"), None)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n", type=int, default=0, help="칸마다 설계 수 (기본 checks.n_per_cell)")
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    cfg = G.load_cfg(a.config, "train")
    nc = len(G.cells(cfg))
    n = a.n or cfg["checks"]["n_per_cell"]
    t0 = time.time()
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.config,)) as ex:
        R = list(ex.map(one, range(nc * n), chunksize=4))
    fails = [r for r in R if "fail" in r]
    ok = [r for r in R if "fail" not in r]
    bad = [r for r in ok if r["bad"]]
    by = lambda key: collections.Counter(key(r) for r in ok)  # noqa: E731
    att = collections.defaultdict(list)
    for r in ok:
        for ax in ("timing", "form", "motion", "camera"):
            att[f"{ax}={r['cell'][ax]}"].append(r["attempt"] + 1500 * r["stream"])
    subs = collections.defaultdict(collections.Counter)
    for r in ok:
        subs[r["cell"]["form"]][r["sub"].split("_")[0] if r["cell"]["form"] != "fixed" else r["sub"]] += 1
    ev = collections.defaultdict(list)
    for r in ok:
        ev[r["cell"]["timing"]].append(r["event"])
    res = {"config": os.path.abspath(a.config), "n_cells": nc, "n_per_cell": n, "n_designs": len(R), "design_s": round(time.time() - t0, 1),
           "design_sec_per_clip_median": round(float(np.median([r["sec"] for r in ok])), 2) if ok else None,
           "design_fail": [{"i": r["i"], "cell": r["cell"]} for r in fails],
           "physics_bad": [{"i": r["i"], "cell": r["cell"], "bad": r["bad"]} for r in bad][:50], "n_physics_bad": len(bad),
           "attempt_mean_by_axis": {k: round(float(np.mean(v)), 1) for k, v in sorted(att.items())},
           "form_sub": {k: dict(v) for k, v in subs.items()},
           "count_levels": dict(by(lambda r: r["count"])), "n_distractor": dict(by(lambda r: r["n_distractor"])),
           "event_samples": {k: {"start_median": float(np.median([e[0] for e in v])), "end_median": float(np.median([e[1] for e in v])),
                                 "start_range": [int(min(e[0] for e in v)), int(max(e[0] for e in v))], "end_range": [int(min(e[1] for e in v)), int(max(e[1] for e in v))]}
                             for k, v in ev.items()},
           "target_speed_px_s_by_motion": {m: round(float(np.median([r["speed_px_s"] for r in ok if r["cell"]["motion"] == m])), 1) for m in cfg["cells"]["motion"]},
           "target_size_px_median": round(float(np.median([r["size_px"] for r in ok])), 1)}
    orr = np.array([r["occ_ratio"] for r in ok if r.get("occ_ratio") is not None])
    res["object_occluder_screen_ratio"] = ({"n": int(len(orr)), "median": round(float(np.median(orr)), 2), "p90": round(float(np.percentile(orr, 90)), 2),
                                            "max": round(float(orr.max()), 2), "frac_gt_2": round(float((orr > 2).mean()), 3)} if len(orr) else None)
    fx = subs.get("fixed", collections.Counter())
    tot = sum(fx.values()) or 1
    K = cfg["checks"]
    # 칸마다 설계 실패 비율 (기하로 어려운 칸은 얇아진다 — 관문이 아니라 보고)
    fc = collections.Counter((r["cell"]["timing"], r["cell"]["form"], r["cell"]["motion"], r["cell"]["camera"]) for r in fails)
    res["thin_cells"] = {"|".join(k): round(v / n, 2) for k, v in sorted(fc.items(), key=lambda x: -x[1]) if v / n > 0.2}
    res["design_fail_rate"] = round(len(fails) / max(len(R), 1), 4)
    gates = {"design_fail_rate_ok": len(fails) / max(len(R), 1) <= K.get("max_design_fail_rate", 0.0),
             "no_physics_bad": len(bad) == 0,
             "all_cells_present": len({(r["cell"]["timing"], r["cell"]["form"], r["cell"]["motion"], r["cell"]["camera"]) for r in ok}) == nc,
             "fixed_sub_each_ge_5pct": all(v / tot >= 0.05 for v in fx.values()),
             "attempt_mean_ok": max(np.mean(v) for v in att.values()) <= K["max_attempt_mean"],
             "occluder_screen_ratio_ok": len(orr) == 0 or float(orr.max()) <= cfg["occluders"]["object_screen_ratio"][1] * 1.1}
    gates = {k: bool(v) for k, v in gates.items()}
    res["gates"] = gates
    res["pass"] = all(gates.values())
    print(json.dumps(res, indent=1, ensure_ascii=False, default=lambda x: x.item() if hasattr(x, "item") else str(x)))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    sys.exit(0 if res["pass"] else 1)


if __name__ == "__main__":
    main()
