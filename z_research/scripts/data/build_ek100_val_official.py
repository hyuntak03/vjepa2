#!/usr/bin/env python3
"""EK100 **validation 138 개**를 공식 V-JEPA 2 val 입력과 같은 256x256 으로 저장한다 (test set 이 아니다).

공식 val 변환 (`evals/action_anticipation_frozen/dataloader.py` eval_transform, center_crop) 을 그대로 따른다:
  decord RGB 프레임 -> `src/datasets/utils/video/functional.py resize_clip`: 짧은변 292, 긴변 int(292*긴변/짧은변),
  cv2.INTER_LINEAR -> `transforms.CenterCrop(256)`: x1 = int(round((W-256)/2)), y1 = int(round((H-256)/2))
  1920x1080 이면 519x292 로 줄이고 [18:274, 132:388].
이렇게 만든 256x256 을 쓸 때 config 는 `spatial_mode: short_side` (256x256 입력에서 항등) 로 둔다.

기존 `EPIC-KITCHENS_resized` (짧은변 256 -> 256 crop, 색 태그 없음) 와 다른 점:
  1. 기하: 공식과 같은 1.14 배 확대 화면
  2. 색: RGB -> YUV 를 BT.709 행렬 (정밀 반올림) 로 명시, bt709 / tv range 태그, **yuv444p (크로마 축소 없음)**.
     2026-09-14 실측 (같은 32 장을 인코딩 -> decord 로 되읽기): 4:2:0 은 무손실이어도 RGB 평균이 −0.9 / −1.4 / −0.9
     로 치우쳤고 (256 해상도에서 색을 절반으로 줄였다 되살리는 과정), BT.601 태그로 바꿔도 같았다. 4:4:4 는 +0.2 / +0.1 / +0.4.
     crf 12 기준 프레임당 4.1 KB (59.94 fps P01_11) ~ 9.8 KB (P09_07), val 전체 11~26 GB 추정
  3. 프레임: decord 인덱스 그대로 (공식 dataloader 가 읽는 k 번째 = 출력 k 번째). decord 가 깨진 패킷에서
     읽지 못한 프레임은 바로 앞 프레임으로 채우고 _meta 에 기록한다

저장 직후 출력 파일을 decord 로 다시 열어 (a) 프레임 수 = 원본 decord 길이, (b) fps, (c) 표본 8 장이
인코더에 넣은 RGB 와 같은지 (채널별 평균 차, MAD) 를 검사하고 _meta/<video>.json 에 남긴다.

  python z_research/scripts/data/build_ek100_val_official.py --jobs 8
  python z_research/scripts/data/build_ek100_val_official.py --videos P01_11 --jobs 1     # 시험
  python z_research/scripts/data/build_ek100_val_official.py --split train --crf 15 --color-tol 1.0 --jobs 40   # train 495 (2026-09-14)

train 도 같은 기하 (짧은변 292 -> 가운데 256) 로 저장한다. 공식 train 은 원본 16:9 전체에서 random resized crop 을 하므로
이 데이터로 학습하면 crop 이 가운데 256 안에서만 일어난다 (val 입력은 공식과 같다). train 은 crf 15 (P02_13 에서 crf 12 의 63% 크기,
MAD 2.7 / crf 18 은 MAD 3.26 로 검사 탈락). ffmpeg 디코딩에서 프레임 수가 decord 와 다르면 자동으로 decord 로 다시 만든다.
SLURM: z_research/scripts/data/sbatch_build_ek100_official.sh
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

SRC = Path("/data/dataset/EPIC-KITCHENS")
DST = Path("/data2/local_datasets/EPIC-KITCHENS_resized")   # 2026-09-15: epic_test 에서 이름 변경 (옛 짧은변 256 데이터 자리)
ANNOT = Path("/data/hyuntak/project/2026/2027_cvpr/epic-kitchens-100-annotations")
SHORT, CROP = 292, 256          # int(256 * 256 / 224) = 292 (dataloader VideoTransform)


def vpath(root, v):
    return root / v.split("_")[0] / "videos" / f"{v}.MP4"


def official_geometry(h, w):
    """resize_clip + CenterCrop 과 같은 산술."""
    if w < h:
        ow, oh = SHORT, int(SHORT * h / w)
    else:
        oh, ow = SHORT, int(SHORT * w / h)
    x1, y1 = int(round((ow - CROP) / 2.0)), int(round((oh - CROP) / 2.0))
    return ow, oh, x1, y1


def r_fps_of(p):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate",
                          "-of", "csv=p=0", str(p)], capture_output=True, text=True).stdout.strip()
    return out or "60000/1001"


def ffmpeg_frames(src, h, w):
    """ffmpeg rgb24 파이프로 전 프레임을 순서대로 낸다 (-vsync 0). 2026-09-14 P03_26 전 667 장 수, 표본 7 장이
    decord get_batch 와 비트 단위 동일 (ffmpeg 4.2.7). decord 는 1080p 에서 worker 당 수십 GB 를 써서 병렬을 못 늘린다."""
    p = subprocess.Popen(["ffmpeg", "-nostdin", "-v", "error", "-threads", "2", "-i", str(src), "-an", "-vsync", "0",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE, bufsize=0)
    sz = h * w * 3
    try:
        while True:
            buf = bytearray(sz); mv = memoryview(buf); got = 0
            while got < sz:
                r = p.stdout.readinto(mv[got:])
                if not r:
                    break
                got += r
            if got < sz:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    finally:
        p.stdout.close(); p.kill(); p.wait()


def build(v, dst_root, crf, gop, chunk, threads, decoder="decord", color_tol=0.5, min_free_gb=0.0):
    import cv2
    from decord import VideoReader, cpu

    src, dst = vpath(SRC, v), vpath(dst_root, v)
    meta_path = dst_root / "_meta" / f"{v}.json"
    if meta_path.exists() and json.loads(meta_path.read_text()).get("ok") and dst.exists():
        return dict(video=v, ok=True, skipped=True)
    free_gb = shutil.disk_usage(dst_root).free / 2**30      # /data2 는 다른 사용자도 쓴다 (2026-09-14 한때 24 MB)
    if free_gb < min_free_gb:
        raise RuntimeError(f"디스크 여유 {free_gb:.1f} GB < {min_free_gb} GB — 시작하지 않음")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.MP4")
    t0 = time.time()
    vr = VideoReader(str(src), num_threads=1 if decoder == "ffmpeg" else threads, ctx=cpu(0))
    n, fps = len(vr), float(vr.get_avg_fps())
    h, w = vr[0].shape[:2]
    if decoder == "ffmpeg":
        del vr                                   # 길이·fps·크기만 decord 로 (공식 dataloader 가 보는 값)
    ow, oh, x1, y1 = official_geometry(h, w)
    r_fps = r_fps_of(src)
    num = int(r_fps.split("/")[0])
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{CROP}x{CROP}", "-framerate", r_fps,
           "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-color_range", "pc", "-i", "-",
           "-an", "-vf", "scale=in_color_matrix=bt709:out_color_matrix=bt709:in_range=pc:out_range=tv:flags=accurate_rnd+full_chroma_int+bitexact,format=yuv444p",
           "-c:v", "libx264", "-threads", "1", "-preset", "veryfast", "-crf", str(crf), "-g", str(gop),
           "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-color_range", "tv",
           "-video_track_timescale", str(num), "-movflags", "+faststart", str(tmp)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    probe_idx = [int(x) for x in np.linspace(0, n - 1, 8)]
    probes, filled, last = {}, [], None

    def decord_frames():
        for a in range(0, n, chunk):
            ids = list(range(a, min(n, a + chunk)))
            try:
                frames = list(vr.get_batch(ids).asnumpy())
            except Exception:               # 깨진 패킷: 한 장씩, 못 읽으면 앞 프레임
                frames = []
                for i in ids:
                    try:
                        frames.append(vr.get_batch([i]).asnumpy()[0])
                    except Exception:
                        frames.append(None)
            yield from zip(ids, frames)

    # ffmpeg 은 못 읽은 프레임을 건너뛰므로 수가 decord 와 다르면 아래 frames 검사에서 FAIL -> --decoder decord 로 다시
    frame_iter = enumerate(ffmpeg_frames(src, h, w)) if decoder == "ffmpeg" else decord_frames()
    try:
        if True:
            for i, f in frame_iter:
                if f is None:
                    out = last
                    filled.append(i)
                else:
                    r = cv2.resize(f, (ow, oh), interpolation=cv2.INTER_LINEAR)
                    out = np.ascontiguousarray(r[y1:y1 + CROP, x1:x1 + CROP])
                if out is None:
                    raise RuntimeError(f"{v}: 첫 프레임부터 읽기 실패")
                if i in probe_idx:
                    probes[i] = out.copy()
                proc.stdin.write(out.tobytes())
                last = out
        proc.stdin.close()
        err = proc.stderr.read().decode(errors="ignore")
        rc = proc.wait()
    except Exception:
        proc.kill()
        raise
    if rc != 0:
        raise RuntimeError(f"{v}: ffmpeg rc={rc} {err[-400:]}")

    # -- 되읽기 검증
    ov = VideoReader(str(tmp), num_threads=threads, ctx=cpu(0))
    got = ov.get_batch(sorted(probes)).asnumpy().astype(np.float32)
    ref = np.stack([probes[i] for i in sorted(probes)]).astype(np.float32)
    d = got - ref
    res = dict(video=v, src=str(src), src_hw=[h, w], resize=[ow, oh], crop_xy=[x1, y1], r_fps=r_fps,
               frames_src_decord=n, frames_dst=len(ov), fps_src=fps, fps_dst=float(ov.get_avg_fps()),
               filled=len(filled), filled_first=filled[:20],
               color_diff_mean_rgb=[round(float(x), 3) for x in d.reshape(-1, 3).mean(0)],
               color_mad=round(float(np.abs(d).mean()), 3), crf=crf, gop=gop, sec=round(time.time() - t0, 1))
    probs = []
    if len(ov) != n:
        probs.append(f"frames {len(ov)} != src decord {n}")
    if abs(res["fps_dst"] - fps) > 0.01:
        probs.append(f"fps {res['fps_dst']:.4f} != {fps:.4f}")
    res["color_tol"] = color_tol
    if max(abs(x) for x in res["color_diff_mean_rgb"]) > color_tol or res["color_mad"] > 3.0:
        probs.append(f"color diff {res['color_diff_mean_rgb']} mad {res['color_mad']}")
    res["problems"], res["ok"] = probs, not probs
    if decoder == "ffmpeg" and len(ov) != n:     # 깨진 패킷: ffmpeg 이 프레임을 건너뛰었다 -> decord (앞 프레임 채움) 로 다시
        del ov
        tmp.unlink(missing_ok=True)
        return build(v, dst_root, crf, gop, chunk, threads, "decord", color_tol, min_free_gb)
    if res["ok"]:
        os.replace(tmp, dst)
    (dst_root / "_meta").mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(res, indent=1))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dst", default=str(DST))
    ap.add_argument("--videos", nargs="*")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--threads", type=int, default=2, help="decord 스레드 (1080p 는 스레드가 많을수록 메모리를 많이 쓴다)")
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--crf", type=int, default=12)
    ap.add_argument("--gop", type=int, default=32)
    ap.add_argument("--decoder", choices=["ffmpeg", "decord"], default="ffmpeg",
                    help="ffmpeg: rgb24 파이프 (decord 와 비트 동일, 메모리 작음) / decord: 깨진 프레임을 앞 프레임으로 채워야 할 때")
    ap.add_argument("--color-tol", type=float, default=0.5,
                    help="채널 평균 차 허용치. 2026-09-14 P18_08 은 crf 12 / 8 모두 B +0.60 / +0.56 (압축이 아니라 tv range 클리핑) -> 1.0 으로 통과시켰다")
    ap.add_argument("--split", choices=["val", "train", "both"], default="val",
                    help="val 138 / train 495 / 둘 다. 같은 --dst 에 같은 레이아웃으로 쌓인다 (train·val 비디오는 겹치지 않는다)")
    ap.add_argument("--min-free-gb", type=float, default=8.0, help="디스크 여유가 이보다 적으면 새 비디오를 시작하지 않는다")
    a = ap.parse_args()
    dst_root = Path(a.dst); dst_root.mkdir(parents=True, exist_ok=True)
    csvs = {"val": ["EPIC_100_validation.csv"], "train": ["EPIC_100_train.csv"],
            "both": ["EPIC_100_train.csv", "EPIC_100_validation.csv"]}[a.split]
    vids = a.videos or sorted({r["video_id"] for f in csvs for r in csv.DictReader(open(ANNOT / f))})
    info_path = dst_root / "_info.json"
    runs = json.loads(info_path.read_text()).get("runs", []) if info_path.exists() else []
    runs.append(dict(split=a.split, n_videos=len(vids), crf=a.crf, color_tol=a.color_tol, decoder=a.decoder,
                     created=time.strftime("%Y-%m-%d %H:%M:%S"), host=os.uname().nodename))
    info_path.write_text(json.dumps(dict(
        what="EPIC-KITCHENS-100 videos (validation 138 + train 495 as built; see runs) in the official V-JEPA 2 val geometry — not the test set",
        runs=runs,
        spatial="official V-JEPA 2 val transform: decord RGB -> cv2.INTER_LINEAR short side 292 -> CenterCrop 256 (round((W-256)/2))",
        color="rgb24 -> yuv444p (no chroma subsampling) with BT.709 matrix (accurate_rnd), tv range, bt709 tags", frames="decord index k -> output frame k; undecodable frames = previous frame",
        use_with="spatial_mode: short_side (identity on 256x256)", layout="P??/videos/P??_??.MP4 (file_format 0)",
        crf=a.crf, gop=a.gop, created=time.strftime("%Y-%m-%d %H:%M:%S"), host=os.uname().nodename,
        script="z_research/scripts/data/build_ek100_val_official.py"), indent=1))
    t0, done, bad = time.time(), 0, []
    with ProcessPoolExecutor(a.jobs) as ex:
        futs = {ex.submit(build, v, dst_root, a.crf, a.gop, a.chunk, a.threads, a.decoder, a.color_tol, a.min_free_gb): v for v in vids}
        for fu in as_completed(futs):
            v = futs[fu]; done += 1
            try:
                r = fu.result()
            except Exception as e:  # noqa: BLE001
                bad.append(v); print(f"[{done}/{len(vids)} {(time.time()-t0)/60:.1f}m] FAIL {v}: {str(e)[:200]}", flush=True); continue
            if r.get("skipped"):
                print(f"[{done}/{len(vids)}] skip {v} (이미 검증됨)", flush=True); continue
            if not r["ok"]:
                bad.append(v)
            print(f"[{done}/{len(vids)} {(time.time()-t0)/60:.1f}m] {'ok  ' if r['ok'] else 'FAIL'} {v} frames {r['frames_dst']}/{r['frames_src_decord']} "
                  f"fps {r['fps_dst']:.2f} filled {r['filled']} color {r['color_diff_mean_rgb']} mad {r['color_mad']} {r['sec']}s"
                  + (f"  {r['problems']}" if r["problems"] else ""), flush=True)
    print(f"끝: ok {len(vids) - len(bad)} / {len(vids)}  실패 {bad}  ({(time.time()-t0)/60:.1f} 분)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
