#!/usr/bin/env python
"""fps 4 인접 쌍의 optical flow 를 미리 뽑아 memmap 으로 저장한다 (2026-10-22, 사용자 요청).

왜: AR predictor 출력에 **움직임 보조 손실** (aux head) 을 걸려면 블록마다 flow 라벨이 필요하다.
    상태 라벨 (phys2d_sup) 과 달리 flow 는 **관측만으로 얻어진다** — "관측만으로 학습" 주장을 깨지 않는다
    (CLAUDE.md §12 Q6). 라벨 자체는 RAFT 의 추정치이므로 진실이 아니다 — 아래 ⚠️ 를 볼 것.

무엇을: 영상 v 의 fps 4 프레임 j → j+1 (0.25 s) flow 를 RAFT 로 구해 G×G 격자로 평균 풀링.
    * 좌표: **원본 프레임 비율** (u = du_px / W_orig, v = dv_px / H_orig). 해상도 · 자르기와 무관해
      학습 때 ClipTransform 의 crop box · hflip 만 적용하면 된다.
      (RAFT 를 (R, R) 정사각으로 리사이즈해 돌리므로 du_R / R = du_orig / W_orig 가 정확히 성립한다.)
    * 유효도: 전후 일관성 (forward-backward) 검사. |f + f_bwd∘warp| <= a·(|f| + |f_bwd∘warp|) + b 인 픽셀만 유효.
      칸마다 유효 픽셀 비율을 uint8 (0~255) 로 같이 저장하고, flow 는 **유효 픽셀만 평균**한다.
      K400 은 컷 전환이 흔하고 0.25 s 는 큰 변위라 이 검사가 없으면 라벨의 상당수가 쓰레기다.

출력 (dataset 마다 한 벌):
    flow.f16   (P, 2, G, G) float16   normalized flow (u, v)
    valid.u8   (P, G, G)   uint8      유효 픽셀 비율 × 255
    index.tsv  path, n4, off, npair   — 영상 v 의 쌍 j 는 전역 행 off(v) + j
    meta.json  G, R, 모델, 일관성 파라미터, 생성 시각

⚠️ 라벨의 한계 (쓰기 전에 읽을 것)
  1. fps 4 = 0.25 s 간격이다. RAFT 는 인접 장 (1/30 s) 용으로 학습됐다 — 큰 변위에서 정확도가 떨어진다.
     전후 일관성 검사가 틀린 자리를 **지우지만**, 지워진 자리는 라벨이 없는 것이지 "움직임 0" 이 아니다.
  2. K400 · SSv2 의 flow 는 **카메라 운동이 지배**한다. 물체 운동만 보려면 `--subtract-global` (칸 중앙값 빼기) 을 쓴다.
     둘은 다른 라벨이다 — 섞어서 쓰지 말 것.
  3. 가려진 물체에는 flow 가 없다. 이 라벨은 **영속 (permanence) 을 가르치지 않는다.**

쓰는 곳: app/vjepa_frozen/var_window.py (라벨 읽기) + app/vjepa_frozen/flow_head.py (보조 손실).
실행 예는 파일 끝 `## 재현`.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------- flow 유틸
def warp(x, flow):
    """x (B, C, H, W) 를 flow (B, 2, H, W) (픽셀 단위, x[..] 가 가리키는 자리) 로 샘플."""
    B, _, H, W = flow.shape
    gy, gx = torch.meshgrid(torch.arange(H, device=flow.device, dtype=flow.dtype),
                            torch.arange(W, device=flow.device, dtype=flow.dtype), indexing="ij")
    px = (gx[None] + flow[:, 0]) / max(W - 1, 1) * 2 - 1
    py = (gy[None] + flow[:, 1]) / max(H - 1, 1) * 2 - 1
    return F.grid_sample(x, torch.stack([px, py], -1), mode="bilinear", padding_mode="border", align_corners=True)


def fb_valid(f_fwd, f_bwd, alpha=0.05, beta=0.5):
    """전후 일관성 마스크 (B, 1, H, W) float {0, 1}."""
    f_bwd_w = warp(f_bwd, f_fwd)
    d2 = ((f_fwd + f_bwd_w) ** 2).sum(1, keepdim=True)
    lim = alpha * ((f_fwd ** 2).sum(1, keepdim=True) + (f_bwd_w ** 2).sum(1, keepdim=True)) + beta ** 2
    return (d2 <= lim).to(f_fwd.dtype)


def pool(flow, valid, G):
    """flow (B, 2, R, R) 픽셀 단위 · valid (B, 1, R, R) → (B, 2, G, G) 유효 가중 평균, (B, 1, G, G) 유효 비율."""
    vg = F.avg_pool2d(valid, valid.shape[-1] // G)
    fg = F.avg_pool2d(flow * valid, valid.shape[-1] // G) / vg.clamp_min(1e-6)
    return fg, vg


# ---------------------------------------------------------------- 본체
def build_model(arch, device):
    from torchvision.models.optical_flow import Raft_Large_Weights, Raft_Small_Weights, raft_large, raft_small
    if arch == "large":
        m = raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2, progress=False)
    elif arch == "small":
        m = raft_small(weights=Raft_Small_Weights.C_T_V2, progress=False)
    else:
        raise ValueError(arch)
    return m.to(device).eval().requires_grad_(False)


DECODE_SWAP = None     # (from, to) 경로 치환 — 색인은 원래 경로를 쓰고 디코드만 다른 사본에서
FUSE_FB = False        # 2026-10-27: --fuse-fb
DECODE_SIZE = 0        # > 0 이면 decord 가 디코드하면서 바로 (size, size) 로 리사이즈한다


def read_fps4(path, n_frames, fstp, fps=0, n_t=0):
    """영상의 fps 4 프레임 전부를 **uint8 원본 크기** (n4, H, W, 3) 로. 리사이즈는 GPU 에서 한다.
    ⚠️ 2026-10-22 정정: 예전 판은 여기서 (n4, 3, 256, 256) float32 (영상당 34 MB) 를 만들었다.
       앞서 읽는 큐와 합쳐져 프로세스마다 RSS 50 GB 를 넘겼다 (실측). uint8 원본은 영상당 ~18 MB 이고
       GPU 리사이즈는 공짜다 — 아래 `to_gpu` 가 한다."""
    import decord
    q = path.replace(*DECODE_SWAP) if DECODE_SWAP else path
    kw = {"width": DECODE_SIZE, "height": DECODE_SIZE} if DECODE_SIZE > 0 else {}
    vr = decord.VideoReader(q, num_threads=1, **kw)
    n = min(len(vr), int(n_frames))
    if int(fstp) < 0:                                   # 2026-10-27 시간 기준 표 (build_time_fps_table.py, fstp = -T):
        T_ = -int(fstp)                                 #   장 k = 가장 가까운 원본 프레임 round(k / T · fps) — var_window._read 의 snap 판과 같은 격자
        idx = [min(n - 1, int(round(k / T_ * fps))) for k in range(int(n_t))]
    else:
        idx = list(range(0, n, int(fstp)))
    if len(idx) < 2:
        raise ValueError("프레임 2장 미만")
    a = vr.get_batch(idx).asnumpy()                                         # (n4, H, W, 3) uint8
    return a, (a.shape[1], a.shape[2])


def to_gpu(a, res, device):
    """uint8 (n4, H, W, 3) → (n4, 3, res, res) float [-1, 1] on device."""
    x = torch.from_numpy(a).to(device, non_blocking=True).permute(0, 3, 1, 2).float()
    x = F.interpolate(x, size=(res, res), mode="bilinear", align_corners=False)
    return x.div_(127.5).sub_(1.0)


@torch.no_grad()
def flow_for_clip(model, x, G, iters, consistency, device, chunk, alpha, beta):
    """x (n4, 3, R, R) [-1,1] → (n4-1, 2, G, G) float32 normalized, (n4-1, G, G) float32 valid."""
    R = x.shape[-1]
    outs_f, outs_v = [], []
    for s in range(0, x.shape[0] - 1, chunk):
        a = x[s:s + chunk]
        b = x[s + 1:s + 1 + chunk]
        a, b = a[:b.shape[0]], b
        with torch.autocast("cuda", torch.bfloat16):
            if consistency == "fb" and FUSE_FB:                              # 2026-10-27: 정 · 역방향을 batch 하나로 (같은 계산, 큰 batch)
                n = a.shape[0]
                ff = model(torch.cat([a, b]), torch.cat([b, a]), num_flow_updates=iters)[-1].float()
                f, fb = ff[:n], ff[n:]
                v = fb_valid(f, fb, alpha, beta)
            else:
                f = model(a, b, num_flow_updates=iters)[-1].float()          # (n, 2, R, R) 픽셀
                if consistency == "fb":
                    fb = model(b, a, num_flow_updates=iters)[-1].float()
                    v = fb_valid(f, fb, alpha, beta)
                else:
                    v = torch.ones_like(f[:, :1])
        fg, vg = pool(f, v, G)
        outs_f.append((fg / R).cpu())                                        # /R = /W_orig (정사각 리사이즈라 성립)
        outs_v.append(vg[:, 0].cpu())
    return torch.cat(outs_f), torch.cat(outs_v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True, help="build_fps4_table.py 의 tsv (path, fps, n_frames, fstp, n4)")
    ap.add_argument("--out", required=True, help="출력 디렉토리")
    ap.add_argument("--grid", type=int, default=32, help="저장 격자 G (학습 때 16×16 로 자르고 줄인다)")
    ap.add_argument("--res", type=int, default=256, help="RAFT 입력 한 변 (8 의 배수)")
    ap.add_argument("--arch", default="large", choices=["large", "small"])
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--consistency", default="fb", choices=["fb", "none"])
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--chunk", type=int, default=24, help="RAFT 한 번에 넣는 쌍 수")
    ap.add_argument("--decode-workers", type=int, default=12)
    ap.add_argument("--shard", type=int, default=0, help="이 shard 번호")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--fuse-fb", action="store_true", help="정 · 역방향 RAFT 를 batch 하나로 (2026-10-27)")
    ap.add_argument("--compile", action="store_true", help="torch.compile (2026-10-27)")
    ap.add_argument("--limit", type=int, default=0, help="앞 N 편만 (벤치마크용)")
    # 2026-10-22: 이미 돌던 shard 를 **버리지 않고** 남은 꼬리만 여러 프로세스로 나눠 끝내려고 더했다.
    #   같은 출력 dir 에 memmap 을 r+ 로 열고 **자기 구간 행만** 쓴다 (구간이 겹치지 않으면 안전하다).
    #   --append 는 index.tsv · meta.json 을 다시 쓰지 않는다 (행 배치가 그대로여야 하므로 shard 규격도 같아야 한다).
    ap.add_argument("--start", type=int, default=0, help="이 shard 목록의 [start, end) 만")
    ap.add_argument("--end", type=int, default=0, help="0 이면 끝까지")
    ap.add_argument("--append", action="store_true", help="기존 출력에 이어 쓴다 (memmap r+, 색인 유지)")
    # 2026-10-22: 디코드만 저해상 사본에서 — 실측 320p 2.46 편/s → 256p + 내부 리사이즈 3.82 편/s (1.55x).
    #   색인에는 **원래 경로**를 쓴다 (학습 로더가 320p 경로를 본다). RAFT 가 어차피 256x256 로 돌리므로 손실이 없다.
    #   ⚠️ 이미 뽑아 둔 부분은 320p + GPU bilinear 로 만들어졌다 — 리샘플 필터가 다른 이음매가 생긴다
    #      (라벨은 RAFT 추정치 + 유효 마스크라 그 차이는 라벨 잡음보다 훨씬 작다. 기록해 둔다)
    ap.add_argument("--decode-swap", default="", help="'from:to' 경로 치환 (디코드 전용)")
    ap.add_argument("--decode-size", type=int, default=0, help="decord 가 디코드하며 바로 (N, N) 로 리사이즈")
    ap.add_argument("--bench", action="store_true", help="쓰지 않고 처리량만 잰다")
    args = ap.parse_args()

    global DECODE_SWAP, DECODE_SIZE, FUSE_FB
    FUSE_FB = bool(args.fuse_fb)
    if args.decode_swap:
        DECODE_SWAP = tuple(args.decode_swap.split(":", 1))
    DECODE_SIZE = int(args.decode_size)

    rows = []
    for ln in open(args.table, encoding="utf-8"):
        p, fps, n, fstp, n4 = ln.rstrip("\n").split("\t")
        if int(n4) >= 2:
            rows.append((p, int(n), int(fstp), int(n4), int(fps)))
    if args.limit:
        rows = rows[:args.limit]
    rows = rows[args.shard::args.num_shards]
    npair = [r[3] - 1 for r in rows]
    off = np.concatenate([[0], np.cumsum(npair)]).astype(np.int64)
    P, G = int(off[-1]), args.grid
    nbytes = P * (2 * G * G * 2 + G * G)
    print(f"영상 {len(rows):,} 편 · 쌍 {P:,} 개 · G {G} → {nbytes / 2**30:.1f} GiB", flush=True)

    lo_i = max(0, int(args.start))
    hi_i = len(rows) if args.end <= 0 else min(len(rows), int(args.end))
    if lo_i or hi_i != len(rows):
        print(f"  구간 [{lo_i:,}, {hi_i:,}) 만 처리 ({hi_i - lo_i:,} 편)", flush=True)

    model = build_model(args.arch, args.device)
    if args.compile:                                                         # 2026-10-27
        model = torch.compile(model, dynamic=True)
    if not args.bench:
        os.makedirs(args.out, exist_ok=True)
        if args.append:
            mf = np.load(f"{args.out}/flow.f16", mmap_mode="r+")
            mv = np.load(f"{args.out}/valid.u8", mmap_mode="r+")
            if mf.shape != (P, 2, G, G):
                raise ValueError(f"--append 인데 기존 모양이 다르다 {mf.shape} != {(P, 2, G, G)} — shard 규격이 같아야 한다")
        else:
            mf = np.lib.format.open_memmap(f"{args.out}/flow.f16", mode="w+", dtype=np.float16, shape=(P, 2, G, G))
            mv = np.lib.format.open_memmap(f"{args.out}/valid.u8", mode="w+", dtype=np.uint8, shape=(P, G, G))
    if not args.bench and not args.append:
        with open(f"{args.out}/index.tsv", "w", encoding="utf-8") as fh:
            for i, (p, n, fstp, n4, _fps) in enumerate(rows):
                fh.write(f"{p}\t{n4}\t{off[i]}\t{n4 - 1}\n")
        json.dump({"grid": G, "res": args.res, "arch": args.arch, "iters": args.iters,
                   "consistency": args.consistency, "alpha": args.alpha, "beta": args.beta,
                   "n_videos": len(rows), "n_pairs": P, "table": args.table,
                   "units": "flow normalized by original frame size (u=du/W_orig, v=dv/H_orig); valid = consistent-pixel fraction x255",
                   "created": datetime.now().isoformat(timespec="seconds")},
                  open(f"{args.out}/meta.json", "w"), indent=2)

    # ⚠️ ThreadPoolExecutor.map 은 **모든** 작업을 즉시 제출해 결과를 무한정 앞서 쌓는다 (2026-10-22 실측 RSS 50 GB+).
    #    앞서 읽기를 ahead 개로 묶는다 — 디코드가 GPU 보다 빠르면 큐가 차고 producer 가 멈춘다.
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor
    pool_ex = ThreadPoolExecutor(args.decode_workers)
    ahead = max(args.decode_workers * 3, 8)
    futs, nxt = deque(), lo_i
    while nxt < min(lo_i + ahead, hi_i):
        futs.append((nxt, pool_ex.submit(_safe_read, rows[nxt]))); nxt += 1
    t0, done, pairs, bad = time.time(), 0, 0, 0
    n_todo = hi_i - lo_i
    while futs:
        i, fu = futs.popleft()
        if nxt < hi_i:
            futs.append((nxt, pool_ex.submit(_safe_read, rows[nxt]))); nxt += 1
        got = fu.result()
        if got is None:
            bad += 1
            continue
        a, (H, W) = got
        x = to_gpu(a, args.res, args.device)
        del a
        fg, vg = flow_for_clip(model, x, G, args.iters, args.consistency, args.device, args.chunk, args.alpha, args.beta)
        if not args.bench:
            s, e = off[i], off[i] + fg.shape[0]
            mf[s:e] = fg.numpy().astype(np.float16)
            mv[s:e] = (vg.clamp(0, 1) * 255).round().numpy().astype(np.uint8)
        done += 1; pairs += fg.shape[0]
        if done % 20 == 0 or i == hi_i - 1:
            dt = time.time() - t0
            print(f"  {done:,}/{n_todo:,} 편 · {pairs:,} 쌍 · {dt:.0f}s → "
                  f"{pairs / dt:.0f} 쌍/s · {done / dt:.1f} 편/s · 실패 {bad}"
                  + (f" · 남은 {(n_todo - done) / (done / dt) / 3600:.1f} h" if done else ""), flush=True)
    print(f"끝: {done:,} 편 · {pairs:,} 쌍 · {time.time() - t0:.0f}s · 실패 {bad}")


def _safe_read(r):
    try:
        return read_fps4(r[0], r[1], r[2], r[4], r[3])
    except Exception:
        return None


if __name__ == "__main__":
    main()

# ## 재현
#   python z_training/data/build_flow_cache.py --table data_csv/vll7_k400_320/fps4_table.tsv \
#          --out /data/hyuntak/flow_cache/vll7_k400_320 --grid 32 --device cuda:0 --num-shards 4 --shard 0
#   (--bench --limit 60 으로 먼저 처리량만 잰다)
#   fps 10 시간 기준 (2026-10-27): --table data_csv/vll7_k400_320/fps10t_table.tsv --out /data/hyuntak/flow_cache/vll7_k400_320_fps10t/s<i>
#     (표 = z_training/data/build_time_fps_table.py). 학습 로더는 snap_start 로 시작을 0.1 초 격자에 맞춘다
