#!/usr/bin/env python3
"""EK100 256p 미러 → 256×256 · gop 32 재인코딩 (2026-10-14, vast).

  python z_research/scripts/data/build_ek100_256sq.py --jobs 96
  python z_research/scripts/data/build_ek100_256sq.py --only P09_07          # 한 개 시험

왜: 이 기계의 EK100 은 HF 미러 (짧은 변 256, libx264 ultrafast crf 28, keyframe 약 250 장 간격) 다.
    clip 하나 (8 fps × 32 장 = 원본 59.94 fps 로 약 240 장) 를 꺼낼 때 keyframe 부터 최대 약 500 장을 디코딩해
    probe 학습이 디코딩에 묶인다 (8 GPU · rank 당 batch 64 에서 iter 의 약 20 %).
무엇:
  * 기하 = main 의 공식 val 기하 (`build_ek100_val_official.py`): 짧은 변 292 → 가운데 256 crop.
    ⚠️ main 은 원본 1080p 에서 만들었고 여기는 256p 미러에서 키운다 (292/256 = 1.14 배 업스케일) — 같은 기하, 더 흐린 원본
  * 프레임 = 원본과 1:1 (`-fps_mode passthrough`), fps 태그 유지 → 엔진의 시각 → 프레임 번호 계산이 그대로다
  * libx264 crf 15, yuv420p, keyframe 32 장 간격 (`-g 32 -keyint_min 32 -sc_threshold 0`)
  * 저장 직후 decord 로 프레임 수 · fps 를 원본과 대조해 `_meta/<video>.json` 에 기록 (어긋나면 실패로 남긴다)
출력: <dst>/<pid>/videos/<vid>.MP4 (엔진 file_format 0)
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="/workspace/dataset/benchmarks/EK100/videos_256p/videos")
ap.add_argument("--dst", default="/workspace/world_model/local_datasets/EPIC-KITCHENS_256sq")
ap.add_argument("--jobs", type=int, default=96)
ap.add_argument("--only", default=None)
a = ap.parse_args()


def info(path):
    import decord
    vr = decord.VideoReader(path, num_threads=1)
    return len(vr), float(vr.get_avg_fps()), tuple(vr[0].shape)


def one(src):
    vid = os.path.basename(src)[:-4]
    pid = vid.split("_")[0]
    out = os.path.join(a.dst, pid, "videos", vid + ".MP4")
    meta = os.path.join(a.dst, "_meta", vid + ".json")
    if os.path.exists(meta) and json.load(open(meta)).get("ok"):
        return vid, "skip"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp.MP4"
    t0 = time.time()
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-threads", "2", "-i", src,
           "-vf", "scale=-2:292:flags=bilinear,crop=256:256", "-fps_mode", "passthrough",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "15", "-pix_fmt", "yuv420p",
           "-g", "32", "-keyint_min", "32", "-sc_threshold", "0", "-an", tmp]
    r = subprocess.run(cmd, capture_output=True, text=True)
    rec = {"video": vid, "src": src, "cmd": " ".join(cmd), "sec": None, "ok": False}
    if r.returncode != 0:
        rec["err"] = r.stderr[-2000:]
    else:
        try:
            n0, f0, _ = info(src)
            n1, f1, shp = info(tmp)
            rec.update(src_frames=n0, out_frames=n1, src_fps=f0, out_fps=f1, shape=list(shp))
            rec["ok"] = (n0 == n1) and abs(f0 - f1) < 0.01 and tuple(shp[:2]) == (256, 256)
            if rec["ok"]:
                os.replace(tmp, out)
        except Exception as e:
            rec["err"] = repr(e)
    rec["sec"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(meta), exist_ok=True)
    json.dump(rec, open(meta, "w"), indent=1)
    if not rec["ok"] and os.path.exists(tmp):
        os.remove(tmp)
    return vid, "ok" if rec["ok"] else "FAIL"


srcs = sorted(glob.glob(os.path.join(a.src, "*.MP4")))
if a.only:
    srcs = [s for s in srcs if os.path.basename(s)[:-4] == a.only]
print(f"{len(srcs)} 개 → {a.dst} (jobs {a.jobs})", flush=True)
t0, n, bad = time.time(), 0, []
with ProcessPoolExecutor(a.jobs) as ex:
    for f in as_completed([ex.submit(one, s) for s in srcs]):
        vid, st = f.result()
        n += 1
        if st == "FAIL":
            bad.append(vid)
        if n % 50 == 0 or st == "FAIL":
            print(f"  {n}/{len(srcs)} {time.time() - t0:.0f}s  {vid} {st}", flush=True)
print(f"끝 {n} 개 {time.time() - t0:.0f}s  실패 {len(bad)}: {bad[:20]}")
sys.exit(1 if bad else 0)
