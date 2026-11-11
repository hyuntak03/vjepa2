#!/usr/bin/env python3
"""BlenderTrack v2 렌더 뒤 검사 — 분할 하나의 렌더 결과가 쓸 만한지. 관문을 어기면 종료 코드 1 (2026-11-01).

  · mp4 가 끝까지 읽힌다 (64 장 · 256²) · 라벨 키 · 길이
  · 물리 (설계 그대로 다시 계산): gen_design_v2.physics_check — 바닥 · 경사로 · 물체끼리 · 소품 · 운동 꼴 · 에너지
  · Blender 안에서 잰 회전 중심 어긋남 · 구가 바닥 아래로 들어간 깊이 (작업자 첫 clip 들, 로그) · 카메라 투영 오차
  · 가림 타이밍 (렌더 라벨로): 샘플마다 target 가려진 비율 = 1 − visible / amodal 칸 → 칸의 타이밍 (early · mid · late) 정의를 지키나
  · 칸마다 개수 (설계 실패로 모자란 칸)
산출: <out>/<split>_verify.json · <out>/<split>_sheet.png · data_csv/blender_track_<name>/<split>_cells.csv (clip · 축 · 렌더 타이밍 통과)
      + <split>_meta_ok.npz (렌더 타이밍을 통과한 clip 만, <split>_meta.npz 가 있으면)
  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P gen_verify_v2.py --config configs/v2.yaml --split smoke
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_design_v2 as G  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
CFG = None


def _init(path, split):
    global CFG
    CFG = G.load_cfg(path, split)


def one(lab_path):
    out = CFG["out"]
    cid = os.path.basename(lab_path)[:-4]
    r = {"cid": cid, "err": []}
    try:
        import decord
        vr = decord.VideoReader(f"{out}/clips/{cid}.mp4")
        T = int(CFG["frames"])
        if len(vr) != T:
            r["err"].append(f"프레임 {len(vr)} != {T}")
        fr = vr.get_batch([0, T // 2, T - 1]).asnumpy()
        if fr.shape[1:] != (CFG["size"], CFG["size"], 3):
            r["err"].append(f"크기 {fr.shape}")
        z = np.load(lab_path)
        if z["center_px"].shape[0] != T or z["amodal"].shape[0] != T:
            r["err"].append("라벨 길이")
        spec = json.load(open(f"{out}/specs/{cid}.json"))
        r["cell"] = spec["cell"]
        r["bad"] = G.physics_check(spec)
        am, vi = z["amodal_n"][:, 0].astype(float), z["visible_n"][:, 0].astype(float)
        bb = z["bbox_px"][:, 0]
        S = CFG["size"]
        w = np.maximum(bb[:, 2] - bb[:, 0], 1e-6); h = np.maximum(bb[:, 3] - bb[:, 1], 1e-6)
        inw = np.clip(np.minimum(bb[:, 2], S) - np.maximum(bb[:, 0], 0), 0, None) * np.clip(np.minimum(bb[:, 3], S) - np.maximum(bb[:, 1], 0), 0, None)
        off = 1.0 - inw / (w * h)
        hid = np.where(am > 0, 1.0 - vi / np.maximum(am, 1), 0.0)
        ok, ev = G.timing_ok(spec["cell"]["timing"], hid[::G.STRIDE], off[::G.STRIDE], CFG["timing"])
        r["timing_ok"], r["event_render"], r["event_design"] = bool(ok), ev, spec["event_samples"]
    except Exception as e:  # noqa: BLE001
        r["err"].append(f"{type(e).__name__} {e}")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="smoke")
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    cfg = G.load_cfg(a.config, a.split)
    out, sp = cfg["out"], cfg["splits"][a.split]
    pre = f"{a.split}{sp['seed']}_"
    labs = sorted(glob.glob(f"{out}/labels/{pre}*.npz"))
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.config, a.split)) as ex:
        R = list(ex.map(one, labs, chunksize=16))
    res = {"split": a.split, "expected": sp["n"], "found": len(labs)}
    errs = [f"{r['cid']}: {e}" for r in R for e in r["err"]]
    bad = [(r["cid"], r["bad"]) for r in R if r.get("bad")]
    tok = [r for r in R if "timing_ok" in r]
    res["n_decode_or_label_err"] = len(errs); res["errors_head"] = errs[:20]
    res["n_physics_bad"] = len(bad); res["physics_bad_head"] = bad[:20]
    res["render_timing_pass"] = round(float(np.mean([r["timing_ok"] for r in tok])), 4) if tok else None
    by = collections.defaultdict(list)
    for r in tok:
        for ax in ("timing", "form", "motion", "camera"):
            by[f"{ax}={r['cell'][ax]}"].append(r["timing_ok"])
    res["render_timing_pass_by_axis"] = {k: round(float(np.mean(v)), 3) for k, v in sorted(by.items())}
    agree = [r for r in tok if r["timing_ok"] and r["event_render"]]
    res["event_start_shift_render_minus_design"] = dict(collections.Counter(int(r["event_render"][0] - r["event_design"][0]) for r in agree))
    # 칸마다 개수
    cnt = collections.Counter((r["cell"]["timing"], r["cell"]["form"], r["cell"]["motion"], r["cell"]["camera"]) for r in tok)
    want = sp["n"] // len(G.cells(cfg))
    res["cells_found"] = len(cnt); res["cell_count_min"] = min(cnt.values()) if cnt else 0; res["cell_count_expected"] = want
    df = f"{out}/logs/design_fail_{a.split}.txt"
    res["design_fail"] = sum(1 for _ in open(df)) if os.path.exists(df) else 0
    # Blender 물리 · 투영 (작업자 첫 clip 들)
    pv, pe = [], []
    for lg in glob.glob(f"{out}/logs/{a.split}_w*.log"):
        for line in open(lg, errors="ignore"):
            m = re.search(r"pivot_err ([0-9.]+)m sink ([0-9.]+)", line)
            if m:
                pv.append((float(m.group(1)), float(m.group(2))))
            m = re.search(r"proj_err ([0-9.]+)px", line)
            if m:
                pe.append(float(m.group(1)))
    res["blender_pivot_err_max_m"] = max((x[0] for x in pv), default=None)
    res["blender_sphere_sink_max_frac"] = max((x[1] for x in pv), default=None)
    res["proj_err_px_max"] = max(pe, default=None)
    # 칸 표 · 렌더 타이밍을 통과한 목록
    dcsv = f"{REPO}/data_csv/blender_track_{cfg['name']}"
    os.makedirs(dcsv, exist_ok=True)
    with open(f"{dcsv}/{a.split}_cells.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip", "timing", "form", "form_sub", "motion", "place", "camera", "count", "n_distractor", "event_s_design", "event_e_design",
                    "event_s_render", "event_e_render", "render_timing_ok", "physics_ok"])
        for r in tok:
            c = r["cell"]; er = r["event_render"] or (None, None)
            w.writerow([r["cid"], c["timing"], c["form"], c["form_sub"], c["motion"], c["motion"].split("_")[0], c["camera"], c["count"], c["n_distractor"],
                        r["event_design"][0], r["event_design"][1], er[0], er[1], int(r["timing_ok"]), int(not r.get("bad"))])
    keep = {r["cid"] for r in tok if r["timing_ok"] and not r.get("bad") and not r["err"]}
    mp = f"{dcsv}/{a.split}_meta.npz"
    if os.path.exists(mp):
        m = np.load(mp); d = {k: m[k] for k in m.files}
        sel = np.array([str(c) in keep for c in d["clip"]])
        np.savez(f"{dcsv}/{a.split}_meta_ok.npz", **{k: (v[sel] if hasattr(v, "shape") and v.ndim >= 1 and v.shape[0] == len(sel) else v) for k, v in d.items()})
        res["meta_ok_clips"] = int(sel.sum())
    # 확인 시트
    try:
        import decord
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        rng = np.random.default_rng(0)
        sel = [labs[i] for i in rng.choice(len(labs), min(12, len(labs)), replace=False)]
        T, S = int(cfg["frames"]), int(cfg["size"])
        fi = np.linspace(0, T - 2, 8).astype(int) // 2 * 2
        fig, ax = plt.subplots(len(sel), len(fi), figsize=(1.6 * len(fi), 1.8 * len(sel)), squeeze=False)
        for i, f in enumerate(sel):
            cid = os.path.basename(f)[:-4]
            z = np.load(f); sp_ = json.load(open(f"{out}/specs/{cid}.json"))
            frames = decord.VideoReader(f"{out}/clips/{cid}.mp4").get_batch(list(fi)).asnumpy()
            am = np.unpackbits(z["amodal"][:, 0], axis=-1).astype(bool); vi = np.unpackbits(z["visible"][:, 0], axis=-1).astype(bool)
            for j, t in enumerate(fi):
                a_ = ax[i, j]; a_.imshow(frames[j]); ext = (0, S, S, 0)
                if am[t].any():
                    a_.contour(np.kron(am[t], np.ones((S // 64, S // 64))), levels=[0.5], colors=["#eb6834"], linewidths=0.8, extent=ext, origin="upper")
                if vi[t].any():
                    a_.contour(np.kron(vi[t], np.ones((S // 64, S // 64))), levels=[0.5], colors=["#2a78d6"], linewidths=0.8, extent=ext, origin="upper")
                a_.set_title(f"s{t // 2}", fontsize=6, pad=1); a_.set_xlim(0, S); a_.set_ylim(S, 0); a_.set_xticks([]); a_.set_yticks([])
            c = sp_["cell"]
            ax[i, 0].set_ylabel(f"{c['timing']} {c['form']}\n{c['motion']}\n{c['camera']} n{c['n_distractor']}", fontsize=5)
        plt.tight_layout(pad=0.2)
        res["sheet"] = f"{out}/{a.split}_sheet.png"
        plt.savefig(res["sheet"], dpi=110)
    except Exception as e:  # noqa: BLE001
        res["sheet_error"] = f"{type(e).__name__} {e}"
    gates = {"decode_label_ok": len(errs) == 0, "physics_ok": len(bad) == 0,
             "blender_ok": (res["blender_pivot_err_max_m"] is None or (res["blender_pivot_err_max_m"] <= 1e-3 and res["blender_sphere_sink_max_frac"] <= 0.02)),
             "proj_ok": res["proj_err_px_max"] is None or res["proj_err_px_max"] <= 0.01,
             "render_timing_ok": (res["render_timing_pass"] or 0) >= cfg["checks"]["render_timing_pass_min"],
             "all_cells": res["cells_found"] == len(G.cells(cfg)) if a.split != "smoke" or len(labs) >= len(G.cells(cfg)) else True}
    gates = {k: bool(v) for k, v in gates.items()}
    res["gates"] = gates
    res["pass"] = all(gates.values())
    print(json.dumps(res, indent=1, ensure_ascii=False, default=lambda x: x.item() if hasattr(x, "item") else str(x)))
    json.dump(res, open(f"{out}/{a.split}_verify.json", "w"), indent=1, ensure_ascii=False, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    sys.exit(0 if res["pass"] else 1)


if __name__ == "__main__":
    main()
