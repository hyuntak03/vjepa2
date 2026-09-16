#!/usr/bin/env python3
"""EK100 256x256 재인코딩본이 **dataloader 가 실제로 읽는 프레임**에서 원본과 같은 순간인지 픽셀로 대조한다.

`build_ek100_resized.py --verify-only` 는 헤더(해상도·프레임 수·fps)만 본다. 여기서는 학습·평가와 똑같이
decord 로 연다 (`epickitchens.decode_videos_to_clips` 와 같은 인덱스 산술: time_source=timestamp, 논문 ap, at=1s).

clip 하나마다
  1. 원본(1080p)을 decord 로 [첫 인덱스-2, 끝 인덱스+2] 연속 구간 디코드 -> 짧은변 256 + 가운데 256 crop (cv2 INTER_AREA)
  2. 재인코딩본을 decord 로 같은 32 인덱스 디코드
  3. 32 장 각각에서 재인코딩본 k 번째와 원본 k+d (d = -2..2) 의 평균 절대차(회색조)를 재서 **d=0 이 최소인지** 본다.
     판정 불가로 따로 세는 것: 움직임이 작은 프레임 (원본 k-1 과 k+1 의 차 < 3 회색값) / 1·2 등의 차가 0.5 미만인 동률.
     동률은 **원본에 이웃 프레임이 거의 같은 프레임이 섞인 비디오**에서 생긴다 (예: P13_06 은 움직이는 프레임의 43% 가
     한쪽 이웃과 거의 같다). 재인코딩과 무관하다 (2026-09-14 ffmpeg 순차 디코드로 확인: P07_101 은 2,867/2,869 가 d=0).
     동률 프레임은 **동률 안에 d=0 이 있는지**를 따로 센다 (tie_without_0). 원본이 2장씩 같은 비디오에서 1장 밀려 있으면
     동률의 절반이 (-2,-1) 처럼 d=0 을 빼고 생긴다. 밀림이 없으면 동률은 늘 (-1,0) 또는 (0,+1) 이다.

출력: 비디오별 한 줄 + 합계. 결과 json 은 --out.

  python z_research/scripts/data/verify_ek100_resized_frames.py --jobs 12 --out /tmp/ek100_frames.json
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

SRC = Path("/data/dataset/EPIC-KITCHENS")
DST = Path("/data2/local_datasets/EPIC-KITCHENS_resized")
ANNOT = Path("/data/hyuntak/project/2026/2027_cvpr/epic-kitchens-100-annotations")
FPC, FPS, AT = 32, 8, 1.0
REPAIRED = ["P29_01", "P29_05", "P30_05", "P30_08"]


def ts(s):
    h, m, x = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(x)


def to256(frames):
    import cv2

    out = []
    for f in frames:
        h, w = f.shape[:2]
        s = 256 / min(h, w)
        nh, nw = max(256, round(h * s)), max(256, round(w * s))
        g = cv2.resize(f, (nw, nh), interpolation=cv2.INTER_AREA)
        i, j = (nh - 256) // 2, (nw - 256) // 2
        out.append(g[i:i + 256, j:j + 256])
    return np.stack(out)


def gray(x):
    return x.astype(np.float32).mean(-1)


def check_video(vid, segs):
    from decord import VideoReader, cpu

    pid = vid.split("_")[0]
    rs, rd = [], []
    try:
        vs = VideoReader(str(SRC / pid / "videos" / f"{vid}.MP4"), num_threads=4, ctx=cpu(0))
        vd = VideoReader(str(DST / pid / "videos" / f"{vid}.MP4"), num_threads=2, ctx=cpu(0))
    except Exception as e:  # noqa: BLE001
        return dict(video=vid, error=f"open: {e}")
    fs, fd = vs.get_avg_fps(), vd.get_avg_fps()
    res = dict(video=vid, fps_src=fs, fps_dst=fd, len_src=len(vs), len_dst=len(vd), clips=[])
    for st in segs:
        af_d = int(round(st * fd) - int(AT * fd))
        fstp = int(fd / FPS)
        idx = np.arange(af_d - FPC * fstp, af_d, fstp).astype(np.int64)
        idx[idx < 0] = 0
        lo, hi = max(int(idx.min()) - 2, 0), min(int(idx.max()) + 2, len(vs) - 1, len(vd) - 1)
        try:
            src = to256(vs.get_batch(list(range(lo, hi + 1))).asnumpy())
            dst = vd.get_batch(list(idx)).asnumpy()
        except Exception as e:  # noqa: BLE001
            res["clips"].append(dict(start_ts=st, error=f"decode: {str(e)[:120]}"))
            continue
        gs, gd = gray(src), gray(dst)
        best, still, tie, tie_no0, mad0 = [], 0, 0, 0, []
        for k, i in enumerate(idx):
            c = int(i) - lo
            if c - 2 < 0 or c + 2 >= len(gs):
                continue
            motion = np.abs(gs[c + 1] - gs[c - 1]).mean()
            errs = [np.abs(gd[k] - gs[c + d]).mean() for d in (-2, -1, 0, 1, 2)]
            mad0.append(errs[2])
            if motion < 3.0:
                still += 1
                continue
            srt = sorted(errs)
            if srt[1] - srt[0] < 0.5:
                tie += 1
                tie_no0 += int(errs[2] - srt[0] >= 0.5)
                continue
            best.append(int(np.argmin(errs)) - 2)
        best = np.array(best)
        res["clips"].append(dict(start_ts=st, n_judged=int(len(best)), n_still=still, n_tie=tie, n_tie_without_0=tie_no0,
                                 shift_hist={int(d): int((best == d).sum()) for d in (-2, -1, 0, 1, 2)},
                                 mad_at_0=float(np.mean(mad0)) if mad0 else None))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-video", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    ap.add_argument("--videos", nargs="*", help="이 비디오만 (기본: 복구 4개 + fps 그룹별 표본 21개)")
    a = ap.parse_args()
    rng = random.Random(a.seed)

    fps_of = {}
    for f in (DST / "_meta").glob("*.json"):
        from fractions import Fraction
        fps_of[f.stem] = round(float(Fraction(json.loads(f.read_text())["src"]["avg_fps"])), 2)
    segs = {}
    for split in ("train", "validation"):
        for r in csv.DictReader(open(ANNOT / f"EPIC_100_{split}.csv")):
            segs.setdefault(r["video_id"], []).append(ts(r["start_timestamp"]))
    groups = {}
    for v in segs:
        groups.setdefault(fps_of[v], []).append(v)
    pick = list(REPAIRED)
    for fps, n in ((59.94, 8), (50.0, 8), (29.97, 2), (47.95, 2), (90.0, 1)):
        cand = sorted(v for v in groups.get(fps, []) if v not in pick)
        pick += rng.sample(cand, min(n, len(cand)))
    if a.videos:
        pick = a.videos
    jobs = {}
    with ProcessPoolExecutor(a.jobs) as ex:
        for v in pick:
            s = sorted(x for x in segs[v] if x > 6.0)
            jobs[ex.submit(check_video, v, rng.sample(s, min(a.per_video, len(s))))] = v
        out = []
        for fu in as_completed(jobs):
            r = fu.result()
            out.append(r)
            if "error" in r:
                print(f"{r['video']:8s} ERROR {r['error']}", flush=True)
                continue
            h = {d: sum(c.get("shift_hist", {}).get(d, 0) for c in r["clips"]) for d in (-2, -1, 0, 1, 2)}
            errs = [c["error"] for c in r["clips"] if "error" in c]
            mad = [c["mad_at_0"] for c in r["clips"] if c.get("mad_at_0") is not None]
            ties = sum(c.get("n_tie", 0) for c in r["clips"]); tno0 = sum(c.get("n_tie_without_0", 0) for c in r["clips"]); still = sum(c.get("n_still", 0) for c in r["clips"])
            print(f"{r['video']:8s} fps {r['fps_src']:.3f}->{r['fps_dst']:.3f} len {r['len_src']}->{r['len_dst']}  "
                  f"shift d=-2..2 {[h[d] for d in (-2, -1, 0, 1, 2)]}  tie {ties} (d=0 빠진 동률 {tno0}) still {still}  MAD@0 {np.mean(mad) if mad else float('nan'):.2f}"
                  + (f"  decode errors {len(errs)}: {errs[0]}" if errs else ""), flush=True)
    tot = {d: sum(c.get("shift_hist", {}).get(d, 0) for r in out for c in r.get("clips", [])) for d in (-2, -1, 0, 1, 2)}
    t_tie = sum(c.get("n_tie", 0) for r in out for c in r.get("clips", []))
    t_no0 = sum(c.get("n_tie_without_0", 0) for r in out for c in r.get("clips", []))
    print(f"\n합계 videos {len(out)}  판정 프레임 d=-2..2: {[tot[d] for d in (-2, -1, 0, 1, 2)]}  동률 {t_tie} (그중 d=0 이 빠진 것 {t_no0})")
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
