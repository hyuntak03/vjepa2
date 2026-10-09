#!/usr/bin/env python3
"""EK100 encoder 충분성 — frozen encoder 의 문맥 끝 토큰이 ego-motion (카메라 운동) 을 담는가 (2026-09-29, RETHINK §1-b 미확정 칸).

    python e_ek_egomotion.py --source ek100|ssv2|pan --out <dir> [--gpus N] [--limit K]

clip 마다 저장:
  z_last  (256, 1280) fp16 : online encoder(문맥 16 장) 의 마지막 튜블릿 토큰 (predictor 가 실제로 먹는 것)
  h_last  (256, 1280) fp16 : target_encoder(문맥 16 장) 의 마지막 튜블릿 토큰 (LN)
  flow    (256, 2)   fp32 : 칸별 문맥 끝 flow (프레임 12→15 평균, px/프레임) — Farneback, 칸 중심 표본
  gflow   (2,)       fp32 : 전역 (중앙값) flow = ego-motion 대리
  fut     (8, 256, 2) fp32: 미래 슬롯 i 의 칸 s 가 마지막 관측에서 어디서 왔나 (역추적 변위, px) — e_scene_reach_nat.src_maps 와 같은 flow
판독은 로그인 노드에서 (`e_ek_egomotion_analyze.py`): ridge z_last (토큰 평균 · 토큰별) → gflow / flow, clip split 5-fold R²; 팬 (v 알려짐) 이 양성 대조.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arlib  # noqa: E402
from h4_single_query_v3 import idx_range  # noqa: E402
import e_scene_reach as ES  # noqa: E402
import e_scene_reach_nat as EN  # noqa: E402
from e_flowtrack import to_gray  # noqa: E402

S_TOK, D, NF, TC, G, PX = 256, 1280, 32, 8, 16, 16


def flows(x):
    import cv2
    g = to_gray(x.cpu())
    ff = [cv2.calcOpticalFlowFarneback(g[t], g[t + 1], None, 0.5, 3, 15, 3, 5, 1.2, 0) for t in (12, 13, 14)]
    vf = np.mean(ff, 0)                                                        # (H, W, 2)
    cx = (np.arange(S_TOK) % G) * PX + PX / 2; cy = (np.arange(S_TOK) // G) * PX + PX / 2
    ctr = np.stack([cx, cy], 1)
    per = EN._sample(vf, ctr)                                                  # (256, 2)
    gl = np.median(vf.reshape(-1, 2), 0)
    bf = {t: cv2.calcOpticalFlowFarneback(g[t], g[t - 1], None, 0.5, 3, 15, 3, 5, 1.2, 0) for t in range(16, NF)}
    fut = np.zeros((8, S_TOK, 2), np.float32)
    for i in range(8):
        p = ctr.copy()
        for t in range(16 + 2 * i, 15, -1):
            p = p + EN._sample(bf[t], p)
        fut[i] = p - ctr
    return per.astype(np.float32), gl.astype(np.float32), fut


def _worker(rank, world, a, items):
    dev = torch.device(f"cuda:{rank}"); torch.cuda.set_device(dev)
    cfg = arlib.v3_cfg(16, 16); os.environ.pop("ARLIB_PRED_CKPT", None)
    bundle = arlib.build_bundle(cfg, dev, window=NF); ac = arlib.autocast_ctx(cfg)
    ci = idx_range(0, TC, 1, dev)
    O = {nm: np.lib.format.open_memmap(Path(a.out) / f"{nm}.npy", mode="r+") for nm in ("z_last", "h_last", "flow", "gflow", "fut")}
    mine = list(range(rank, len(items), world)); t0 = time.time()
    for n_done, row in enumerate(mine):
        it = items[row]
        x0 = ES.load_pan(it) if a.source == "pan" else EN.load_item(a.source, it)[0]
        if a.source == "pan":
            per = np.tile(np.array([it["dir"] * it["v"], 0.0], np.float32), (S_TOK, 1)); gl = per[0].copy()
            fut = np.zeros((8, S_TOK, 2), np.float32)
            for i in range(8): fut[i, :, 0] = -it["dir"] * it["v"] * 2 * (i + 1)
        else:
            per, gl, fut = flows(x0)
        O["flow"][row] = per; O["gflow"][row] = gl; O["fut"][row] = fut
        x = x0[None].to(dev, bundle.dtype)
        with torch.inference_mode(), ac():
            z = bundle.context_encoder(x, masks=[ci]).float().reshape(8, S_TOK, D)[7]
            h = F.layer_norm(bundle.target_encoder(x[:, :, :16]), (D,)).float().reshape(8, S_TOK, D)[7]
        O["z_last"][row] = z.half().cpu().numpy(); O["h_last"][row] = h.half().cpu().numpy()
        if rank == 0 and n_done % 20 == 0:
            el = time.time() - t0; print(f"[rank0] {n_done+1}/{len(mine)} {el/60:.1f} 분", flush=True)
    for v_ in O.values(): v_.flush()
    print(f"[rank{rank}] 끝", flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--source", choices=("ek100", "ssv2", "pan"), required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--gpus", type=int, default=torch.cuda.device_count()); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--n", type=int, default=800)
    a = ap.parse_args(); Path(a.out).mkdir(parents=True, exist_ok=True)
    if a.source == "pan": items = ES.pan_items(a.limit or a.n)
    elif a.source == "ssv2":
        import e_scene_ar as EA; items = EA.ssv2_items(a.limit)
    else: items = EN.items_for(a.source, a.limit)
    n = len(items)
    for nm, sh, dt in (("z_last", (n, S_TOK, D), np.float16), ("h_last", (n, S_TOK, D), np.float16), ("flow", (n, S_TOK, 2), np.float32), ("gflow", (n, 2), np.float32), ("fut", (n, 8, S_TOK, 2), np.float32)):
        m = np.lib.format.open_memmap(Path(a.out) / f"{nm}.npy", mode="w+", dtype=dt, shape=sh); m[:] = np.nan; del m
    json.dump(dict(n=n, source=a.source, items=items), open(Path(a.out) / "meta.json", "w"))
    print("clips", n, flush=True); torch.set_num_threads(4)
    mp.spawn(_worker, args=(a.gpus, a, items), nprocs=a.gpus, join=True); print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
