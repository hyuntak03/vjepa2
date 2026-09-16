#!/usr/bin/env python3
"""EK100 원본 vs 256x256 재인코딩본 — 파일을 직접 다시 읽어 대조한다 (`_meta/*.json` 기록을 믿지 않는다).

A. 전체 700 개, ffprobe 헤더: 해상도 · nb_frames · r_frame_rate · avg_frame_rate · duration
B. 표본 (--sample 개, val 비디오 중 fps 그룹별 + 복구 4 개), dataloader 와 같은 decord 로:
   len(vr) · get_avg_fps() 를 두 파일에서, 그리고 같은 프레임 8 장의 색 대조
   (원본 1080p -> 짧은변 256 가운데 crop 을 cv2 INTER_AREA 로 만든 것 vs 재인코딩본 decord 출력).
   채널별 평균 차 (재인코딩 − 원본) 와 차의 std. 색 변환 행렬 (BT.709 vs BT.601) 이 어긋나면
   채널 평균이 한쪽으로 치우친다 (특히 R·B).

⚠️ 1080p decord 는 worker 하나가 수십 GB 를 쓸 수 있다 (2026-09-14 EK100 원본 eval 이 그 이유로 OOM).
   --jobs 를 작게, decord 스레드 2 로 둔다.

  python z_research/scripts/data/verify_ek100_resized_headers.py --jobs 16 --sample 12 --sample-jobs 3
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

import numpy as np

SRC = Path("/data/dataset/EPIC-KITCHENS")
DST = Path("/data2/local_datasets/EPIC-KITCHENS_resized")
ANNOT = Path("/data/hyuntak/project/2026/2027_cvpr/epic-kitchens-100-annotations")
REPAIRED = ["P29_01", "P29_05", "P30_05", "P30_08"]


def path(root, v):
    return root / v.split("_")[0] / "videos" / f"{v}.MP4"


def probe(p):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,nb_frames,r_frame_rate,avg_frame_rate,duration,color_space,color_transfer,color_primaries,pix_fmt",
                          "-of", "json", str(p)], capture_output=True, text=True).stdout
    s = json.loads(out or "{}").get("streams", [{}])
    return s[0] if s else {}


def header_pair(v):
    return v, probe(path(SRC, v)), probe(path(DST, v))


def to256(f):
    import cv2
    h, w = f.shape[:2]; s = 256 / min(h, w)
    nh, nw = max(256, round(h * s)), max(256, round(w * s))
    g = cv2.resize(f, (nw, nh), interpolation=cv2.INTER_AREA)
    i, j = (nh - 256) // 2, (nw - 256) // 2
    return g[i:i + 256, j:j + 256]


def decord_pair(v):
    from decord import VideoReader, cpu
    try:
        a = VideoReader(str(path(SRC, v)), num_threads=2, ctx=cpu(0))
        b = VideoReader(str(path(DST, v)), num_threads=2, ctx=cpu(0))
    except Exception as e:  # noqa: BLE001
        return dict(video=v, error=str(e)[:120])
    n = min(len(a), len(b)); idx = [int(x) for x in np.linspace(n * 0.1, n * 0.9, 8)]
    res = dict(video=v, len_src=len(a), len_dst=len(b), fps_src=a.get_avg_fps(), fps_dst=b.get_avg_fps())
    try:
        fa = np.stack([to256(a.get_batch([i]).asnumpy()[0]) for i in idx]).astype(np.float32)
        fb = b.get_batch(idx).asnumpy().astype(np.float32)
        d = fb - fa
        res.update(mean_src=fa.reshape(-1, 3).mean(0).tolist(), mean_dst=fb.reshape(-1, 3).mean(0).tolist(),
                   diff_mean=d.reshape(-1, 3).mean(0).tolist(), diff_std=d.reshape(-1, 3).std(0).tolist(),
                   mad=float(np.abs(d).mean()))
    except Exception as e:  # noqa: BLE001
        res["decode_error"] = str(e)[:120]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--sample", type=int, default=12)
    ap.add_argument("--sample-jobs", type=int, default=3)
    ap.add_argument("--skip-headers", action="store_true")
    a = ap.parse_args()
    vids = sorted(p.stem for p in DST.glob("P*/videos/*.MP4"))
    print(f"재인코딩본 파일 {len(vids)} 개")

    if not a.skip_headers:
        with ThreadPoolExecutor(a.jobs) as ex:
            rows = list(ex.map(header_pair, vids))
        bad, groups = [], {}
        for v, s, d in rows:
            fr = lambda x: float(Fraction(x)) if x and x != "0/0" else float("nan")
            prob = []
            if (int(d.get("width", 0)), int(d.get("height", 0))) != (256, 256):
                prob.append(f"dst 해상도 {d.get('width')}x{d.get('height')}")
            if s.get("nb_frames") != d.get("nb_frames"):
                prob.append(f"nb_frames {s.get('nb_frames')} -> {d.get('nb_frames')}")
            if abs(fr(s.get("r_frame_rate")) - fr(d.get("r_frame_rate"))) > 1e-3:
                prob.append(f"r_fps {s.get('r_frame_rate')} -> {d.get('r_frame_rate')}")
            if abs(fr(s.get("avg_frame_rate")) - fr(d.get("avg_frame_rate"))) > 0.01:
                prob.append(f"avg_fps {fr(s.get('avg_frame_rate')):.4f} -> {fr(d.get('avg_frame_rate')):.4f}")
            if abs(float(s.get("duration", 0)) - float(d.get("duration", 0))) > 0.1:
                prob.append(f"duration {s.get('duration')} -> {d.get('duration')}")
            key = (round(fr(s.get("r_frame_rate")), 2), s.get("color_space"), s.get("pix_fmt"), d.get("color_space"), d.get("pix_fmt"))
            groups[key] = groups.get(key, 0) + 1
            if prob:
                bad.append((v, prob))
        print("\n[A] ffprobe 헤더 대조 (700)")
        print("  (원본 fps, 원본 color_space, 원본 pix_fmt, 재인코딩 color_space, 재인코딩 pix_fmt) -> 개수")
        for k, c in sorted(groups.items(), key=lambda x: -x[1]):
            print(f"    {k}: {c}")
        print(f"  불일치 파일 {len(bad)} 개")
        for v, p in bad[:20]:
            print(f"    {v}: {'; '.join(p)}")

    fps_of = {}
    for f in (DST / "_meta").glob("*.json"):
        fps_of[f.stem] = round(float(Fraction(json.loads(f.read_text())["src"]["avg_fps"])), 2)
    val = sorted({r["video_id"] for r in csv.DictReader(open(ANNOT / "EPIC_100_validation.csv"))})
    pick = [v for v in REPAIRED]
    rng = np.random.RandomState(0)
    for g in (59.94, 50.0, 29.97, 47.95, 90.0):
        cand = [v for v in val if fps_of.get(v) == g and v not in pick]
        pick += list(rng.choice(cand, min(len(cand), 3 if g in (59.94, 50.0) else 1), replace=False))
    pick = pick[: max(a.sample, len(REPAIRED))]
    print(f"\n[B] decord 대조 + 색 (표본 {len(pick)} 개: {pick})")
    with ProcessPoolExecutor(a.sample_jobs) as ex:
        for r in ex.map(decord_pair, pick):
            if "error" in r:
                print(f"  {r['video']}: 열기 실패 {r['error']}"); continue
            line = (f"  {r['video']:8s} len {r['len_src']} -> {r['len_dst']} ({'같음' if r['len_src'] == r['len_dst'] else '다름'})  "
                    f"fps {r['fps_src']:.3f} -> {r['fps_dst']:.3f}")
            if "diff_mean" in r:
                dm, ds = r["diff_mean"], r["diff_std"]
                line += f"  | RGB 평균 차 {dm[0]:+.2f} {dm[1]:+.2f} {dm[2]:+.2f}  차 std {ds[0]:.1f} {ds[1]:.1f} {ds[2]:.1f}  MAD {r['mad']:.2f}"
            if "decode_error" in r:
                line += f"  | 디코드 실패 {r['decode_error']}"
            print(line, flush=True)


if __name__ == "__main__":
    main()
