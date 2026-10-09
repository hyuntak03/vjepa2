#!/usr/bin/env python3
"""IntPhys 2 사중항 안에서 **어느 둘이 위반 전까지 같은 영상인가** 를 실측한다 (2026-09-30).

k_Possible / k_Impossible 로 짝지으면 처음부터 다른 쌍이 많았다 (intphys2_onset.py). dataset.py 머리말: "한 쌍의 possible 이 다른 쌍의 impossible 이 된다".
그래서 4 영상의 6 조합 전부에 대해 프레임별 절대차 (128×128 RGB, 0–255 평균) 를 재고, 앞 1 초 평균차가 가장 작은 조합과 그 조합의 갈림 시점 (앞 15 프레임 기준 + max(6·MAD, 1.0) 을 5 프레임 연속 초과) 을 낸다.
    python intphys2_pairing_probe.py --root ... --split Main --out <json> [--limit 40]
"""
import argparse, itertools, json, os
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd
STAT_Q = 0.0


def load(path, size=128):
    import decord
    vr = decord.VideoReader(path, width=size, height=size, num_threads=1)
    return vr.get_batch(list(range(len(vr)))).asnumpy().astype(np.float32), float(vr.get_avg_fps())


def onset_of(d):
    base = d[:15]; med = float(np.median(base)); mad = float(np.median(np.abs(base - med))) + 1e-6; thr = med + max(6 * mad, 1.0)
    ab = d > thr
    for t in range(len(d) - 5):
        if ab[t:t + 5].all(): return t, thr
    return -1, thr


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="/data2/local_datasets/world/IntPhys2"); ap.add_argument("--split", default="Main")
    ap.add_argument("--out", required=True); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--workers", type=int, default=12); ap.add_argument("--stat-q", type=float, default=0.0, help="0 = 평균차, 99 = 픽셀 차의 99 백분위 (국소 사건용)")
    a = ap.parse_args(); global STAT_Q; STAT_Q = a.stat_q
    df = pd.read_csv(os.path.join(a.root, a.split, "metadata.csv"))
    def path(fn):
        fn = fn if fn.endswith(".mp4") else fn + ".mp4"; return os.path.join(a.root, a.split, fn) if fn.startswith("Videos/") else os.path.join(a.root, a.split, "Videos", fn)
    scenes = []
    for sc, g in df.groupby("SceneIndex"):
        m = {str(r["type"]): path(str(r["file_name"])) for _, r in g.iterrows()}
        if all(t in m for t in ("1_Possible", "1_Impossible", "2_Possible", "2_Impossible")):
            scenes.append(dict(scene=int(sc), condition=str(g.iloc[0]["condition"]), paths=m))
    if a.limit: scenes = scenes[: a.limit]
    print("scenes", len(scenes), flush=True)
    def run(s):
        try:
            V = {t: load(p) for t, p in s["paths"].items()}; fps = next(iter(V.values()))[1]
            out = {}
            for t1, t2 in itertools.combinations(sorted(V), 2):
                A, B = V[t1][0], V[t2][0]; T = min(len(A), len(B)); D = np.abs(A[:T] - B[:T]).mean(-1)          # (T, H, W)
                d = np.percentile(D.reshape(T, -1), STAT_Q, axis=1) if STAT_Q else D.mean((1, 2))                # 국소 (상위 q%) 또는 평균
                o, thr = onset_of(d); n1 = int(round(fps))
                out[f"{t1}|{t2}"] = dict(first1s=float(d[:n1].mean()), onset_frame=o, onset_sec=(o / fps if o >= 0 else None), T=int(T), thr=thr,
                                         d_half=[round(float(v), 2) for v in d[:: max(1, int(round(fps / 2)))]])
            best = min(out, key=lambda k: out[k]["first1s"])
            return dict(scene=s["scene"], condition=s["condition"], fps=fps, best_pair=best, best_first1s=out[best]["first1s"], best_onset_sec=out[best]["onset_sec"], combos=out)
        except Exception as e:
            return dict(scene=s["scene"], condition=s["condition"], error=str(e)[:120])
    with ThreadPoolExecutor(a.workers) as ex: rows = list(ex.map(run, scenes))
    json.dump(rows, open(a.out, "w"), indent=1)
    ok = [r for r in rows if "error" not in r]
    from collections import Counter
    print("best pair 분포", Counter(r["best_pair"] for r in ok))
    for c in sorted({r["condition"] for r in ok}):
        rs = [r for r in ok if r["condition"] == c]; f = np.array([r["best_first1s"] for r in rs]); o = np.array([r["best_onset_sec"] for r in rs if r["best_onset_sec"] is not None])
        print(f"{c:12s} n {len(rs)}  best 앞1초 차 med {np.median(f):.2f} (q75 {np.percentile(f,75):.2f})  onset med {np.median(o) if len(o) else float('nan'):.2f} s  n_onset {len(o)}  <3s {(o<3).mean() if len(o) else float('nan'):.2f}")


if __name__ == "__main__":
    main()
