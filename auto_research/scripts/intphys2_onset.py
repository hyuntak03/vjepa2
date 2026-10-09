#!/usr/bin/env python3
"""IntPhys 2 위반 시점 — 쌍 (possible / impossible) 영상이 픽셀 단위로 갈라지는 첫 프레임 (2026-09-30).

⚠️ v2 (2026-09-30 저녁) — 기본 방법을 **국소 셀 통계**로 바꿨다. 이전 두 방법 (전역 평균차 절대 문턱 2.0 · 앞 15 프레임 MAD 상대 문턱) 은
장면 6 개를 눈으로 대조했을 때 4 개에서 틀렸다: 쌍은 위반 전에도 픽셀 단위로 같지 않고 (카메라 이동 · 구름 · 그림자 → 전역 평균차 1–30),
작은 물체 (공 지름 ~3 px @128) 는 전역 평균을 못 움직인다. 지금 방법:
  128×128 RGB 절대차 → 8×8 셀 (16×16 격자) 평균 → s_t = max(셀) − median(셀)   (전역 표류 제거한 국소 최대)
  15 프레임 median filter → 기준 = 앞 1 초의 중앙값 med · MAD → 문턱 = med + max(3·MAD, 12, 0.5·med) → 0.5 초 연속 초과하는 첫 프레임
  (검증: 장면 1 · 107 · 112 · 116 · 123 몽타주와 일치, 133 은 카메라 이동 장면이라 ±1 s. 이전 필드 onset_frame/onset_rel_* 도 남겨 둔다)

metadata 에 위반 시점이 없다. 같은 SceneIndex 의 `k_Possible` / `k_Impossible` (k = 1, 2) 은 위반 전까지 같은 장면이므로
프레임별 평균 절대차 (64×64 회색, 0–255) 가 문턱 (기본 2.0) 을 처음 넘는 프레임을 위반 시점으로 잡는다. 두 쌍 (1, 2) 을 따로 낸다.
    python intphys2_onset.py --root /data2/local_datasets/world/IntPhys2 --split Main --out <json>   (CPU, 데이터 노드)
출력: 쌍마다 onset_frame · n_frames · fps · onset_sec · frac (onset / length) · 그 뒤 남는 초. 요약: condition 별 중앙값 · 분위수.
"""
import argparse, json, os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter


def frames_rgb(path, size=128):
    import decord
    vr = decord.VideoReader(path, width=size, height=size, num_threads=1)
    return vr.get_batch(list(range(len(vr)))).asnumpy().astype(np.float32), float(vr.get_avg_fps())      # (T, 128, 128, 3)


def local_stat(A, B, cell=8):
    T = min(len(A), len(B)); D = np.abs(A[:T] - B[:T]).mean(-1); n = D.shape[1] // cell
    C = D.reshape(T, n, cell, n, cell).mean((2, 4)).reshape(T, -1)
    return C.max(1) - np.median(C, 1), D.mean((1, 2))


def local_onset(s, fps, dmin=12.0, hold=0.5, smooth=15):
    sm = median_filter(s, size=smooth); pre = sm[: int(fps)]; med = float(np.median(pre)); mad = float(np.median(np.abs(pre - med))) + 1e-6
    thr = med + max(3 * mad, dmin, 0.5 * med); ab = sm > thr; H = int(hold * fps)
    for t in range(len(sm) - H):
        if ab[t : t + H].all(): return t, thr, med
    return -1, thr, med


def onset(pa, pb, thr):
    A, fa = frames_rgb(pa); B, fb = frames_rgb(pb)
    T = min(len(A), len(B)); s_loc, d = local_stat(A, B)
    o_loc, thr_loc, pre_loc = local_onset(s_loc, fa)
    idx = np.where(d > thr)[0]
    o = int(idx[0]) if len(idx) else -1
    # 2026-09-30: 쌍이 처음부터 조금 다를 수 있어 (렌더 잡음 · 카메라) 절대 문턱 대신 **상대 문턱**도 낸다:
    #   기준 = 앞 15 프레임의 중앙값 · MAD, onset_rel = d 가 기준 + max(6·MAD, 1.0) 을 넘고 그 뒤 5 프레임 연속 유지하는 첫 프레임
    base = d[:15]; med = float(np.median(base)); mad = float(np.median(np.abs(base - med))) + 1e-6; thr_rel = med + max(6 * mad, 1.0)
    above = d > thr_rel; o_rel = -1
    for t in range(T - 5):
        if above[t:t + 5].all(): o_rel = t; break
    return dict(onset_frame=o, n_frames=int(T), fps=fa, onset_sec=(o / fa if o >= 0 else None), frac=(o / T if o >= 0 else None),
                sec_after=((T - o) / fa if o >= 0 else None), max_diff_before=float(d[:o].max()) if o > 0 else 0.0, diff_at=float(d[o]) if o >= 0 else float(d.max()),
                base_med=med, base_mad=mad, thr_rel=float(thr_rel), onset_rel_frame=o_rel, onset_rel_sec=(o_rel / fa if o_rel >= 0 else None),
                sec_after_rel=((T - o_rel) / fa if o_rel >= 0 else None), d_curve=[round(float(v), 3) for v in d],
                onset_loc_frame=o_loc, onset_loc_sec=(o_loc / fa if o_loc >= 0 else None), sec_after_loc=((T - o_loc) / fa if o_loc >= 0 else None),
                loc_thr=float(thr_loc), loc_pre=float(pre_loc), loc_post_max=float(np.max(s_loc)), s_loc_curve=[round(float(v), 2) for v in s_loc])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="/data2/local_datasets/world/IntPhys2"); ap.add_argument("--split", default="Main")
    ap.add_argument("--out", required=True); ap.add_argument("--thr", type=float, default=2.0); ap.add_argument("--workers", type=int, default=16); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    df = pd.read_csv(os.path.join(a.root, a.split, "metadata.csv")); vd = os.path.join(a.root, a.split, "Videos")
    def path(r):
        fn = str(r["file_name"]); fn = fn if fn.endswith(".mp4") else fn + ".mp4"
        if os.path.isabs(fn): return fn
        return os.path.join(a.root, a.split, fn) if fn.startswith("Videos/") else os.path.join(vd, fn)   # metadata 의 file_name 이 이미 Videos/<hash>.mp4
    jobs = []
    for sc, g in df.groupby("SceneIndex"):
        for k in ("1", "2"):
            p = g[g["type"] == f"{k}_Possible"]; i = g[g["type"] == f"{k}_Impossible"]
            if len(p) == 1 and len(i) == 1:
                jobs.append(dict(scene=int(sc), pair=int(k), condition=str(p.iloc[0]["condition"]), occluder=str(p.iloc[0].get("occluder", "")), difficulty=str(p.iloc[0].get("Difficulty", "")),
                                 pos=path(p.iloc[0]), imp=path(i.iloc[0])))
    if a.limit: jobs = jobs[: a.limit]
    print("pairs", len(jobs), flush=True)
    def run(j):
        try: return {**j, **onset(j["pos"], j["imp"], a.thr)}
        except Exception as e: return {**j, "error": str(e)[:120]}
    with ThreadPoolExecutor(a.workers) as ex: rows = list(ex.map(run, jobs))
    ok = [r for r in rows if "error" not in r and (r["onset_frame"] >= 0 or r.get("onset_loc_frame", -1) >= 0)]
    summ = {}
    for cond in sorted({r["condition"] for r in ok}):
        rs = [r for r in ok if r["condition"] == cond]; rl_all = rs; rs = [r for r in rs if r["onset_frame"] >= 0] or rs[:1]; s_ = np.array([r["onset_sec"] or 0 for r in rs]); f_ = np.array([r["frac"] or 0 for r in rs]); after = np.array([r["sec_after"] or 0 for r in rs]); L = np.array([r["n_frames"] / r["fps"] for r in rs])
        rr = [r for r in rs if r.get("onset_rel_frame", -1) >= 0]; sr = np.array([r["onset_rel_sec"] for r in rr]); ar = np.array([r["sec_after_rel"] for r in rr])
        rl = [r for r in rl_all if r.get("onset_loc_frame", -1) >= 0]; sl = np.array([r["onset_loc_sec"] for r in rl]); al = np.array([r["sec_after_loc"] for r in rl])
        summ[cond] = dict(n_loc=len(rl), loc_onset_sec_median=float(np.median(sl)) if len(rl) else None, loc_onset_q25_q75=[float(np.percentile(sl, 25)), float(np.percentile(sl, 75))] if len(rl) else None,
                          loc_sec_after_median=float(np.median(al)) if len(rl) else None, loc_share_before_3s=float((sl < 3).mean()) if len(rl) else None,
                          loc_share_before_5s=float((sl < 5).mean()) if len(rl) else None, loc_share_after_7s=float((sl > 7).mean()) if len(rl) else None,
                          n=len(rs), onset_sec_median=float(np.median(s_)), onset_sec_q25_q75=[float(np.percentile(s_, 25)), float(np.percentile(s_, 75))], onset_frac_median=float(np.median(f_)),
                          sec_after_median=float(np.median(after)), length_sec_median=float(np.median(L)), share_onset_before_3s=float((s_ < 3).mean()), share_after_lt_1s=float((after < 1).mean()),
                          base_med_median=float(np.median([r["base_med"] for r in rs])), n_rel=len(rr),
                          rel_onset_sec_median=float(np.median(sr)) if len(rr) else None, rel_onset_q25_q75=[float(np.percentile(sr, 25)), float(np.percentile(sr, 75))] if len(rr) else None,
                          rel_sec_after_median=float(np.median(ar)) if len(rr) else None, rel_share_onset_before_3s=float((sr < 3).mean()) if len(rr) else None,
                          rel_share_onset_after_7s=float((sr > 7).mean()) if len(rr) else None)
    json.dump(dict(thr=a.thr, n_pairs=len(jobs), n_ok=len(ok), n_error=sum("error" in r for r in rows), n_no_onset=sum(("error" not in r) and r["onset_frame"] < 0 for r in rows), summary=summ, rows=rows), open(a.out, "w"), indent=1)
    print(json.dumps(dict(n_pairs=len(jobs), n_ok=len(ok), summary=summ), indent=1))


if __name__ == "__main__":
    main()
