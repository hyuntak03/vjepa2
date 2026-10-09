#!/usr/bin/env python3
"""IntPhys 2 위반 시점 (v2, 국소 셀 통계) 검증 시트 — 무작위 쌍 N 개의 s_loc 곡선 + 문턱 + 검출 시점, 그리고 시점 −1 / 0 / +1 s 의 두 영상 프레임 (2026-09-30).
    python intphys2_onset_validate.py --json <onset_main_v2.json> --out <png> [--n 8 --seed 0]     (데이터 노드, CPU)
읽는 법: 곡선의 계단이 검출선 (빨강) 과 맞고, 프레임 띠에서 −1 s 는 두 영상이 같고 +1 s 는 달라야 한다. 아니면 그 쌍의 시점은 틀린 것이다.
"""
import argparse, json, os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt


def frames_at(path, secs, size=128):
    import decord
    vr = decord.VideoReader(path, width=size, height=size, num_threads=1); fps = vr.get_avg_fps()
    idx = [int(min(max(s, 0) * fps, len(vr) - 1)) for s in secs]; return vr.get_batch(idx).asnumpy()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--json", required=True); ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=8); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--conditions", default="")
    a = ap.parse_args(); J = json.load(open(a.json)); rows = [r for r in J["rows"] if "error" not in r]
    if a.conditions: rows = [r for r in rows if r["condition"] in a.conditions.split(",")]
    rng = np.random.default_rng(a.seed); pick = [rows[i] for i in rng.choice(len(rows), min(a.n, len(rows)), replace=False)]
    fig, axes = plt.subplots(len(pick), 2, figsize=(13, 2.6 * len(pick)), gridspec_kw=dict(width_ratios=[1.6, 1]))
    for (axc, axf), r in zip(np.atleast_2d(axes), pick):
        s = np.array(r["s_loc_curve"]); fps = r["fps"]; t = np.arange(len(s)) / fps; o = r.get("onset_loc_sec")
        axc.plot(t, s, lw=0.8, color="#2a78d6"); axc.axhline(r["loc_thr"], color="#eb6834", lw=0.8, ls="--")
        if o is not None: axc.axvline(o, color="red", lw=1.2)
        axc.set_title(f"scene {r['scene']} pair {r['pair']} {r['condition']} ({r['difficulty']}, occ={r['occluder']})  onset={o if o is None else round(o, 2)} s  pre={r['loc_pre']:.1f}", fontsize=8)
        axc.set_ylabel("local cell diff"); axc.set_xlabel("s")
        if o is None: o = len(s) / fps / 2
        secs = [o - 1, o, o + 1]
        try:
            P = frames_at(r["pos"], secs); I = frames_at(r["imp"], secs)
            strip = np.concatenate([np.concatenate(list(P), 1), np.concatenate(list(I), 1)], 0); axf.imshow(strip)
            axf.set_title("top possible / bottom impossible @ onset-1s, onset, onset+1s", fontsize=7)
        except Exception as e: axf.text(0.1, 0.5, str(e)[:60], fontsize=7)
        axf.axis("off")
    plt.tight_layout(); os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); plt.savefig(a.out, dpi=90); print("saved", a.out, [(r["scene"], r["pair"], r.get("onset_loc_sec")) for r in pick])


if __name__ == "__main__":
    main()
