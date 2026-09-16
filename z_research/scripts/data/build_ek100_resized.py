#!/usr/bin/env python3
"""EPIC-KITCHENS-100 비디오를 **짧은변 256 -> 가운데 256x256** 으로 줄여 새 루트에 저장한다.

⚠️ 기록 (2026-09-13) — 프레임 수 불일치
  - 처음 판은 `-vsync 0` 만 썼고 P29_01 / P29_05 에서 2장씩 빠졌다. 원본 타임스탬프 흔들림(실제 평균 fps 59.9386
    vs 명목 60000/1001) 때문에 반올림으로 버려진다고 **추정**하고 `setpts=N*den/(num*TB)` + timescale 을 넣었다.
  - ❌ **그 추정은 틀렸다.** setpts 판에서도 P29_01 -2 / P29_05 -2 / P30_08 -16 / P30_05 -1 이 그대로 나왔다 (700 중 4).
    확인한 것: P30_08 은 패킷 67,260 = 헤더, 크기 0·중복 pts·dts 역전 없음, 키프레임으로 시작, GOP 1,120 개 전부 60,
    유일한 pts 틈(1,062 s, 2.1 간격) 부근 5 초를 따로 디코드하면 300/300 장. **ffmpeg 가 어디서 빠뜨리는지는 미확인.**
  - ❌ decord 재처리도 실패했다 — 4개 모두 `Error sending packet` 로 멈췄고, 로그에 `Invalid NAL unit size`,
    `Error splitting the input into NAL units`, `error while decoding MB` 가 나왔다. **원본에 디코드 불가 패킷이 있다**
    — **공식 배포본 자체가 그렇다**: 2026-09-14 `ek100_downloader/epic_video_downloader_verify.py` 로 4개 모두 공식 md5.csv 와 일치 확인 (P30_08 은 md5sum 으로도 bd32d6e5... 일치). 다시 받아도 같은 파일이다. 공식 dataloader(decord)도 그 구간을 지나는 clip 은
    예외로 건너뛴다.
  - ✅ 최종 처리 (2026-09-13): `--retry-failed --decoder ffmpeg-fill`. ffmpeg 로 디코드하며 프레임 pts 로 원본 패킷 번호를 찾고,
    디코드 안 된 번호는 바로 앞 프레임을 복제한다 -> 출력 k 번째 = 원본 k 번째, 프레임 수 = 패킷 수.
    채운 수 P29_01 2 (31855) / P29_05 2 (88285) / P30_05 1 (18071) / P30_08 16 (9곳, 3460~26533) = 이전 부족분과 정확히 일치,
    매칭 실패 0. 채운 자리 앞뒤·영상 끝(ffmpeg 디코드 원본과 비교) 모두 움직이는 프레임 밀림 0.
    깨끗한 파일에서 이 모드는 일반 모드와 픽셀 차 0.0.
  - 프레임 수가 **맞은** 파일은 ffmpeg 판을 그대로 쓴다. setpts + `-vsync 0` 에는 프레임을 복제하는 경로가 없어서
    "하나 빠지고 하나 복제돼 수만 맞는" 경우는 생기지 않는다.
  - setpts 는 해가 없어 남겨 둔다.
  - 크기: 한때 짧은변 292 no-crop 으로 바꿨다가 사용자 결정으로 **256x256 crop 으로 되돌렸다**.

⚠️ 이 데이터를 쓸 때 config 는 `spatial_mode: short_side` 여야 한다 (256x256 입력에서 항등 변환).
   `center_crop` 은 짧은변을 292 로 **다시 키운 뒤** 256 을 떼므로 1.14배 추가 확대 + 업샘플 흐림이 생긴다.
   train 의 random_resized_crop 은 원본 전체 화면이 아니라 이 256x256 안에서만 고른다 (공식과 다름).

annotation 의 `start_frame` / `stop_frame` 은 **원본 비디오의 프레임 번호**다. 그래서
- 프레임을 하나도 버리거나 복제하지 않는다 (위 2)
- 변환 뒤 **프레임 수를 원본 헤더(nb_frames)와 대조**하고, 어긋나면 그 파일을 실패로 남긴다
- 폴더 구조를 원본과 같게 둔다 (`P01/videos/P01_01.MP4`) -> config 는 `base_path` 만 바꾸면 된다
- train/val 에 쓰이는 비디오(633개)를 먼저, test(67개)를 나중에 처리한다

⚠️ 남는 차이:
  - 공간: 공식 val 은 짧은변 292 -> 가운데 256 (세로 12.3% 추가로 버림), 여기는 짧은변 256 -> 가운데 256
  - 축소 커널: ffmpeg swscale `bilinear` vs 공식 cv2 `INTER_LINEAR`(1080p 에서 anti-alias 없이). 픽셀이 조금 다르다
  - H.264 재인코딩(crf 18) 손실

루트에 `_info.json` (처리 방법) 과 `_meta/<video_id>.json` (파일별 대조 결과) 를 남긴다.
중단돼도 다시 돌리면 검증을 통과한 파일은 건너뛴다.

    python z_research/scripts/data/build_ek100_resized.py --dry-run
    python z_research/scripts/data/build_ek100_resized.py --videos P29_01 --dst <scratch> --jobs 1 --threads 32   # 시험
    python z_research/scripts/data/build_ek100_resized.py                        # 전부 (train/val 먼저, 큰 파일부터)
    python z_research/scripts/data/build_ek100_resized.py --verify-only          # 대조만 다시
    python z_research/scripts/data/build_ek100_resized.py --retry-failed --decoder decord --jobs 8   # 프레임 수 불일치 파일만 decord 로
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path

SRC = Path("/data/dataset/EPIC-KITCHENS")
DST = Path("/data2/local_datasets/EPIC-KITCHENS_resized")
SIZE = 256
ANNOT = Path("/data/hyuntak/project/2026/2027_cvpr/epic-kitchens-100-annotations")


def ffprobe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name,width,height,avg_frame_rate,r_frame_rate,nb_frames,duration",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    return dict(codec=s.get("codec_name"), w=int(s["width"]), h=int(s["height"]),
                avg_fps=s.get("avg_frame_rate"), r_fps=s.get("r_frame_rate"),
                nb_frames=int(s["nb_frames"]) if s.get("nb_frames", "N/A") != "N/A" else None,
                duration=float(s["duration"]) if s.get("duration") else None)


def decord_info(path):
    from decord import VideoReader, cpu
    vr = VideoReader(str(path), num_threads=1, ctx=cpu(0))
    return len(vr), float(vr.get_avg_fps())


def vf(size, crop=None):
    """짧은변 -> size (반대 변은 비율 유지, 짝수). crop 이 양수면 가운데 crop x crop, 0 이면 자르지 않는다."""
    crop = size if crop is None else crop
    f = f"scale=w='if(gt(iw,ih),-2,{size})':h='if(gt(iw,ih),{size},-2)':flags=bilinear"
    return f + (f",crop={crop}:{crop}" if crop else "")


def encode(src, dst_tmp, size, crf, gop, threads, crop=None, r_fps="60000/1001"):
    num, den = (int(x) for x in r_fps.split("/")) if "/" in r_fps else (int(float(r_fps)), 1)
    # 디코드 순서 N 번째 프레임에 N/fps 초를 새로 매긴다 (원본 타임스탬프 흔들림 무시). timescale=num 이면
    # pts = N*den 이 정수라 반올림으로 버려지는 프레임이 없다.
    retime = f"setpts=N*{den}/({num}*TB)"
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
           "-threads", str(threads), "-i", str(src),
           "-map", "0:v:0", "-an", "-sn", "-dn",
           "-vf", retime + "," + vf(size, crop), "-vsync", "0",
           "-video_track_timescale", str(num),
           "-c:v", "libx264", "-threads", "1", "-preset", "veryfast", "-crf", str(crf), "-g", str(gop),
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst_tmp)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg rc={r.returncode}: {r.stderr[-800:]}")
    # loglevel error 라 stderr 에는 오류만 남는다. 0 이 아니면 프레임 수가 맞아도 화면이 깨졌을 수 있다
    # (P29_01 등: 깨진 패킷 뒤 프레임이 다음 키프레임까지 번짐·블록 깨짐으로 디코드됐다)
    errs = [l for l in r.stderr.splitlines() if l.strip()]
    return " ".join(cmd), errs


def encode_decord(src, dst_tmp, size, crf, gop, threads, crop=None, r_fps="60000/1001", chunk=256):
    """decord 로 원본을 읽어 (공식 dataloader 와 같은 프레임 인덱스) cv2 로 줄이고 ffmpeg 에 raw 로 넘겨 인코딩.

    ffmpeg 디코더가 일부 패킷을 프레임으로 내놓지 못하는 파일(P29_01 -2, P30_08 -16 ...)에서도
    출력 k 번째 = decord k 번째 가 되도록 한다. 축소는 cv2 INTER_AREA (swscale bilinear 축소와 같은 anti-alias 성격).
    """
    import cv2
    import numpy as np
    from decord import VideoReader, cpu

    crop = size if crop is None else crop
    vr = VideoReader(str(src), num_threads=threads, ctx=cpu(0))
    n = len(vr)
    h, w = vr[0].shape[:2]
    scale = size / min(h, w)
    nw, nh = int(round(w * scale / 2) * 2), int(round(h * scale / 2) * 2)
    nw, nh = (max(nw, size), max(nh, size))
    ow, oh = (crop, crop) if crop else (nw, nh)
    x0, y0 = (nw - ow) // 2, (nh - oh) // 2
    num = int(r_fps.split("/")[0]) if "/" in r_fps else int(float(r_fps))
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{ow}x{oh}", "-framerate", r_fps, "-i", "-",
           "-an", "-c:v", "libx264", "-threads", "1", "-preset", "veryfast", "-crf", str(crf), "-g", str(gop),
           "-pix_fmt", "yuv420p", "-video_track_timescale", str(num), "-movflags", "+faststart", str(dst_tmp)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for a in range(0, n, chunk):
            batch = vr.get_batch(list(range(a, min(n, a + chunk)))).asnumpy()
            for f in batch:
                r = cv2.resize(f, (nw, nh), interpolation=cv2.INTER_AREA)
                proc.stdin.write(np.ascontiguousarray(r[y0:y0 + oh, x0:x0 + ow]).tobytes())
        proc.stdin.close()
        err = proc.stderr.read().decode(errors="ignore")
        rc = proc.wait()
    except Exception:
        proc.kill()
        raise
    if rc != 0:
        raise RuntimeError(f"ffmpeg(raw) rc={rc}: {err[-800:]}")
    return f"decord({n} frames) -> cv2 INTER_AREA {nw}x{nh} crop {ow}x{oh} -> " + " ".join(cmd)


def packet_pts(src):
    """원본 비디오 패킷의 pts 를 재생 순서로. EK100 원본은 pts == dts 순서라 패킷 순서 = 프레임 순서다."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts",
                          "-of", "csv=p=0", str(src)], capture_output=True, text=True, check=True).stdout
    return sorted(int(x) for x in out.split() if x.strip().lstrip("-").isdigit())


def encode_fill(src, dst_tmp, size, crf, gop, crop=None, r_fps="60000/1001", w=None, h=None):
    """깨진 패킷이 있는 원본용. ffmpeg 로 디코드하며 각 프레임의 pts 로 **원본 패킷 번호 k** 를 찾고,
    디코드되지 않은 번호는 바로 앞 프레임을 복제해 채운다 -> 출력 k 번째 = 원본 k 번째, 프레임 수 = 원본 패킷 수.

    P29_01 / P29_05 / P30_05 / P30_08 은 H.264 패킷이 깨져 있다 (`Invalid NAL unit size`,
    `Error splitting the input into NAL units`). ffmpeg 는 그 패킷을 건너뛰고 (-2 / -2 / -1 / -16 장),
    decord 는 `Error sending packet` 으로 멈춘다 (공식 dataloader 에서도 그 구간 clip 은 예외로 건너뛰어진다).
    """
    import bisect
    import re
    import threading
    import queue

    crop = size if crop is None else crop
    pts_list = packet_pts(src)
    N = len(pts_list)
    idx_of = {v: i for i, v in enumerate(pts_list)}
    step = sorted(b - a for a, b in zip(pts_list, pts_list[1:]))[max(0, (N - 1) // 2)] if N > 1 else 1
    if crop:
        ow = oh = crop
    else:
        sc = size / min(w, h)
        ow, oh = int(round(w * sc / 2) * 2), int(round(h * sc / 2) * 2)
    fbytes = ow * oh * 3 // 2  # yuv420p
    num = int(r_fps.split("/")[0]) if "/" in r_fps else int(float(r_fps))

    dec_cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "info", "-threads", "1", "-copyts",
               "-i", str(src), "-map", "0:v:0", "-an", "-sn", "-dn",
               "-vf", "showinfo," + vf(size, crop), "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"]
    enc_log = open(str(dst_tmp) + ".enc.log", "wb")
    enc_cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{ow}x{oh}", "-framerate", r_fps, "-i", "-",
               "-an", "-c:v", "libx264", "-threads", "1", "-preset", "veryfast", "-crf", str(crf), "-g", str(gop),
               "-pix_fmt", "yuv420p", "-video_track_timescale", str(num), "-movflags", "+faststart", str(dst_tmp)]
    dec = subprocess.Popen(dec_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    enc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE, stderr=enc_log)

    q = queue.Queue()
    errs = []
    pat = re.compile(rb"n:\s*(\d+)\s+pts:\s*(-?\d+)")

    def reader():
        for line in dec.stderr:
            if b"showinfo" in line:
                m = pat.search(line)
                if m:
                    q.put(int(m.group(2)))
                    continue
            low = line.lower()
            if b"error" in low or b"invalid" in low:
                errs.append(line.decode(errors="ignore").strip())
        q.put(None)

    th = threading.Thread(target=reader, daemon=True)
    th.start()

    def read_exact(n):
        b = bytearray()
        while len(b) < n:
            c = dec.stdout.read(n - len(b))
            if not c:
                break
            b += c
        return bytes(b)

    next_idx, last, filled, extra, decoded = 0, None, [], 0, 0
    try:
        while True:
            fr = read_exact(fbytes)
            if len(fr) < fbytes:
                break
            pv = q.get(timeout=900)
            if pv is None:
                raise RuntimeError("showinfo 가 프레임보다 먼저 끝났다")
            decoded += 1
            k = idx_of.get(pv)
            if k is None:
                j = bisect.bisect_left(pts_list, pv)
                cand = [c for c in (j - 1, j) if 0 <= c < N]
                k = min(cand, key=lambda c: abs(pts_list[c] - pv)) if cand else None
                if k is None or abs(pts_list[k] - pv) > step / 2:
                    extra += 1
                    continue
            if k < next_idx:
                extra += 1
                continue
            gap = k - next_idx
            if gap:
                filler = last if last is not None else fr
                for _ in range(gap):
                    enc.stdin.write(filler)
                filled.append([next_idx, gap])
            enc.stdin.write(fr)
            last, next_idx = fr, k + 1
        if next_idx < N and last is not None:
            for _ in range(N - next_idx):
                enc.stdin.write(last)
            filled.append([next_idx, N - next_idx])
            next_idx = N
        enc.stdin.close()
    except Exception:
        dec.kill(); enc.kill()
        raise
    rc_enc, rc_dec = enc.wait(), dec.wait()
    th.join(timeout=60)
    enc_log.close()
    if rc_enc != 0:
        raise RuntimeError(f"encoder rc={rc_enc}: {open(str(dst_tmp) + '.enc.log', errors='ignore').read()[-800:]}")
    return dict(packets=N, decoded=decoded, written=next_idx, filled=filled, n_filled=sum(g for _, g in filled),
                unmatched=extra, decoder_rc=rc_dec, decoder_errors=len(errs), decoder_error_examples=errs[:8],
                cmd=" ".join(dec_cmd) + "  |  " + " ".join(enc_cmd))


def verify(src, dst, src_probe=None, size=SIZE, decord_src=False, crop=None):
    """원본과 결과의 프레임 수·fps·해상도를 대조. decord_src=True 면 원본도 decord 로 연다 (NFS 에서 느리다)."""
    sp = src_probe or ffprobe(src)
    dp = ffprobe(dst)
    d_len, d_fps = decord_info(dst)
    res = dict(src=sp, dst=dp, dst_decord_frames=d_len, dst_decord_fps=d_fps)
    ref_frames = sp["nb_frames"]
    if decord_src:
        s_len, s_fps = decord_info(src)
        res.update(src_decord_frames=s_len, src_decord_fps=s_fps)
        ref_frames = s_len
    problems = []
    crop = size if crop is None else crop
    if crop and (dp["w"], dp["h"]) != (crop, crop):
        problems.append(f"해상도 {dp['w']}x{dp['h']} != {crop}x{crop}")
    if not crop and min(dp["w"], dp["h"]) != size:
        problems.append(f"짧은변 {min(dp['w'], dp['h'])} != {size}")
    if ref_frames is not None and d_len != ref_frames:
        problems.append(f"프레임 수 {d_len} != 원본 {ref_frames}")
    if sp["avg_fps"] and abs(float(Fraction(sp["avg_fps"])) - d_fps) > 1e-3 * max(1.0, d_fps):
        problems.append(f"fps {d_fps:.4f} != 원본 {float(Fraction(sp['avg_fps'])):.4f}")
    res["ok"] = not problems
    res["problems"] = problems
    return res


def process(vid, src_root, dst_root, size, crf, gop, threads, force, decord_src, crop=None, decoder="ffmpeg"):
    pid = vid.split("_")[0]
    src = src_root / pid / "videos" / f"{vid}.MP4"
    dst = dst_root / pid / "videos" / f"{vid}.MP4"
    meta = dst_root / "_meta" / f"{vid}.json"
    if not force and dst.exists() and meta.exists():
        try:
            if json.loads(meta.read_text()).get("ok"):
                return vid, "skip", None
        except Exception:
            pass
    dst.parent.mkdir(parents=True, exist_ok=True)
    meta.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.stem + ".tmp.MP4")
    t0 = time.time()
    try:
        sp = ffprobe(src)
        fill = None
        dec_errs = None
        if decoder == "decord":
            cmd = encode_decord(src, tmp, size, crf, gop, threads, crop, sp.get("r_fps") or "60000/1001")
            decord_src = True  # decord 경로는 원본 decord 프레임 수와 대조한다
        elif decoder == "ffmpeg-fill":
            fill = encode_fill(src, tmp, size, crf, gop, crop, sp.get("r_fps") or "60000/1001", sp["w"], sp["h"])
            cmd = fill.pop("cmd")
            decord_src = False  # 원본은 decord 가 깨진 패킷에서 멈춘다 -> 헤더 nb_frames 와 대조
        else:
            cmd, dec_errs = encode(src, tmp, size, crf, gop, threads, crop, sp.get("r_fps") or "60000/1001")
        t_enc = time.time() - t0
        v = verify(src, tmp, sp, size, decord_src, crop)
        if fill is not None:
            v["fill"] = fill
            if fill["written"] != fill["packets"]:
                v["ok"] = False
                v["problems"] = v.get("problems", []) + [f"written {fill['written']} != packets {fill['packets']}"]
        if dec_errs is not None:
            v["decode_errors"] = len(dec_errs)
            v["decode_error_examples"] = dec_errs[:8]
        v.update(video_id=vid, decoder=decoder, cmd=cmd, encode_sec=round(t_enc, 1),
                 src_bytes=src.stat().st_size, dst_bytes=tmp.stat().st_size)
        if v["ok"]:
            os.replace(tmp, dst)
        meta.write_text(json.dumps(v, ensure_ascii=False, indent=1))
        return vid, ("ok" if v["ok"] else "FAIL"), v
    except Exception as e:
        meta.write_text(json.dumps(dict(video_id=vid, ok=False, problems=[repr(e)]), ensure_ascii=False))
        return vid, "ERROR", dict(problems=[repr(e)])
    finally:
        if tmp.exists() and not dst.exists():
            pass  # 실패한 tmp 는 조사용으로 남긴다


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--dst", default=str(DST))
    ap.add_argument("--size", type=int, default=SIZE, help="짧은변 길이")
    ap.add_argument("--crop", type=int, default=None, help="가운데 정사각 crop 크기. 기본 = size, 0 = 자르지 않음")
    ap.add_argument("--num-shards", type=int, default=1, help="여러 노드로 나눌 때 전체 조각 수")
    ap.add_argument("--shard", type=int, default=0, help="이 프로세스가 맡을 조각 번호 (0-based)")
    ap.add_argument("--crf", type=int, default=18, help="libx264 CRF (낮을수록 고화질)")
    ap.add_argument("--gop", type=int, default=32, help="keyframe 간격 (decord 임의 접근 속도)")
    # 실측 (P17_01, 1080p H.264): 디코드 스레드 2개 = CPU 157.5s / wall 81.8s, 1개 = CPU 80.1s / wall 80.1s.
    # 스레드를 늘려도 wall 이 안 줄고 CPU 만 두 배 쓴다 -> 스레드 1 개짜리를 코어 수만큼 띄우는 게 처리량이 최대다.
    ap.add_argument("--jobs", type=int, default=60, help="동시에 도는 ffmpeg 수 (≈ 할당 코어 수)")
    ap.add_argument("--threads", type=int, default=1, help="ffmpeg 하나당 디코드 스레드 (1 이 CPU 당 처리량 최대)")
    ap.add_argument("--videos", nargs="*", help="특정 video_id 만")
    ap.add_argument("--force", action="store_true", help="이미 검증된 파일도 다시 만든다")
    ap.add_argument("--decord-src", action="store_true", help="원본 프레임 수도 decord 로 센다 (느림, 시험용)")
    ap.add_argument("--decoder", choices=["ffmpeg", "decord", "ffmpeg-fill"], default="ffmpeg",
                    help="ffmpeg (빠름) / decord (공식 리더, 깨진 패킷에서 멈춤) / "
                         "ffmpeg-fill (pts 로 원본 패킷 번호를 맞추고 디코드 안 된 번호는 앞 프레임 복제 — 깨진 원본용)")
    ap.add_argument("--retry-failed", action="store_true",
                    help="_meta 에 실패로 남은 비디오만 다시 (보통 --decoder decord 와 함께)")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    src_root, dst_root = Path(a.src), Path(a.dst)
    vids = a.videos or sorted(p.stem for p in src_root.glob("P*/videos/*.MP4"))
    if a.retry_failed:
        failed = []
        for f in (dst_root / "_meta").glob("*.json"):
            try:
                if not json.loads(f.read_text()).get("ok"):
                    failed.append(f.stem)
            except Exception:
                failed.append(f.stem)
        vids = [v for v in vids if v in set(failed)]
    size_of = {v: (src_root / v.split("_")[0] / "videos" / f"{v}.MP4").stat().st_size for v in vids}
    # 큰 파일부터 시작한다 — 시간 제한에 걸려 끊겨도 가장 오래 걸리는 것들이 먼저 끝나 있게.
    # 중단 후 다시 돌리면 검증된 파일은 건너뛴다.
    used = set()
    for f in ("EPIC_100_train.csv", "EPIC_100_validation.csv"):
        try:
            import csv as _csv
            with open(ANNOT / f) as fh:
                used |= {r["video_id"] for r in _csv.DictReader(fh)}
        except FileNotFoundError:
            pass
    # train/val 비디오 먼저, 그 안에서 큰 파일부터 (test 는 맨 뒤)
    vids = sorted(vids, key=lambda v: (v not in used, -size_of[v]))
    if a.num_shards > 1:
        # 용량이 고르게 나뉘도록 큰 것부터 가장 가벼운 조각에 넣는다 (결정적)
        loads = [0] * a.num_shards; owner = {}
        for v in vids:
            k = min(range(a.num_shards), key=lambda j: loads[j]); owner[v] = k; loads[k] += size_of[v]
        vids = [v for v in vids if owner[v] == a.shard]
    total = sum(size_of[v] for v in vids)
    free = shutil.disk_usage(dst_root.parent if not dst_root.exists() else dst_root).free
    n_used = sum(v in used for v in vids)
    print(f"src   {src_root}  ({len(vids)} videos, {total/1e9:.0f} GB; train/val {n_used} 먼저, 나머지 {len(vids)-n_used})")
    print(f"dst   {dst_root}  (여유 {free/1e9:.0f} GB)   host {os.uname().nodename}")
    crop = a.size if a.crop is None else a.crop
    print(f"shard {a.shard}/{a.num_shards}")
    print(f"처리  short side -> {a.size}, " + (f"center crop {crop}x{crop}" if crop else "crop 없음 (비율 유지)") + ", fps/프레임 수 원본 그대로, "
          f"libx264 crf {a.crf} gop {a.gop}, jobs {a.jobs} x threads {a.threads}")
    print(f"vf    {vf(a.size, crop)}")
    if a.dry_run:
        return

    dst_root.mkdir(parents=True, exist_ok=True)
    info = dst_root / "_info.json"
    if not info.exists():
        info.write_text(json.dumps(dict(
            source=str(src_root), size=a.size, crop=crop,
            spatial=(f"short side -> {a.size}, center crop {crop}x{crop}" if crop else f"short side -> {a.size}, no crop (aspect kept)"),
            resize_kernel="ffmpeg swscale bilinear",
            fps="nominal r_frame_rate; timestamps re-assigned as N/fps (setpts) so decode-order frame k -> output frame k",
            frames="unchanged (verified per file against source nb_frames)",
            codec=f"libx264 crf {a.crf} gop {a.gop} yuv420p", layout="P??/videos/P??_??.MP4 (file_format 0)",
            use_with=("spatial_mode: short_side (identity on 256x256). center_crop would re-zoom 1.14x; "
                      "train random_resized_crop only sees this center 256x256"),
            host=os.uname().nodename, created=time.strftime("%Y-%m-%d %H:%M:%S"),
            script="z_research/scripts/data/build_ek100_resized.py"), indent=1, ensure_ascii=False))

    if a.verify_only:
        bad = 0
        for v in vids:
            pid = v.split("_")[0]
            r = verify(src_root / pid / "videos" / f"{v}.MP4", dst_root / pid / "videos" / f"{v}.MP4",
                       size=a.size, decord_src=a.decord_src, crop=crop)
            bad += not r["ok"]
            print(f"{'ok  ' if r['ok'] else 'FAIL'} {v} {r['problems']}")
        print(f"검증 실패 {bad}/{len(vids)}")
        sys.exit(1 if bad else 0)

    t0 = time.time(); done = 0; counts = {}
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        futs = [ex.submit(process, v, src_root, dst_root, a.size, a.crf, a.gop, a.threads,
                          a.force or a.retry_failed, a.decord_src, crop, a.decoder)
                for v in vids]
        for f in as_completed(futs):
            vid, st, res = f.result()
            done += 1; counts[st] = counts.get(st, 0) + 1
            extra = ""
            if res and st in ("ok", "FAIL"):
                extra = (f"frames {res['dst_decord_frames']} fps {res['dst_decord_fps']:.2f} "
                         f"{res['src_bytes']/1e9:.2f}GB->{res['dst_bytes']/1e6:.0f}MB {res['encode_sec']}s")
            if res and res.get("problems"):
                extra += f"  {res['problems']}"
            el = time.time() - t0
            print(f"[{done}/{len(vids)} {el/60:6.1f}m] {st:5} {vid} {extra}", flush=True)
    print(f"끝: {counts}  ({(time.time()-t0)/60:.1f} 분)")
    sys.exit(1 if counts.get("FAIL") or counts.get("ERROR") else 0)


if __name__ == "__main__":
    main()
