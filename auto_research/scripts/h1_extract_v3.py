#!/usr/bin/env python3
"""H1 추출 — 가속은 입력 (z) 에 있는가: 문맥 encoder 32 층 · predictor 문맥 토큰 12 층 · target encoder (문맥만) 천장.

RollOut_v3, 창 = 문맥 [split−C, split),  C ∈ {8, 16, 32} (가속을 재려면 튜블릿 ≥ 3), split 기본 32.
clip 마다 문맥 마지막 3 튜블릿 (T−2, T−1, T) 의 **진실 물체 칸 3×3 창 평균** 을 층마다 저장:
  enc   (n, 3, 32, 1280) fp16  — 문맥 encoder (online) 블록 l 출력 (norm 전 residual) 을 affine-free LN 한 것
  encF  (n, 3, 1280)           — 문맥 encoder 최종 출력 (encoder.norm) = predictor 입력 z  → LN
  tgt   (n, 3, 1280)           — target encoder 에 **문맥 프레임만** 넣은 최종 출력 → LN  (공정한 천장: 미래를 안 본다)
  pctx  (n, 3, 12, 384) fp16   — predictor 블록 l 출력 중 문맥 토큰 (같은 칸) → LN
predictor 는 미래 P = 8 프레임 질의로 돌린다 (문맥 토큰 쪽 값은 미래 질의 수에 약하게 의존 — 기록만).

  sbatch 로 (h1_extract_v3.sbatch) — GPU 4 장이면 충분
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arlib  # noqa: E402
from h23_extract_v3 import ClipDS, collate, arr, win_idx  # noqa: E402

G, S_TOK, D, CELL = 16, 256, 1280, 18.0
OUT_DEFAULT = "/data2/local_datasets/world/world_analysis/cache/auto_research/v3_h1"
CS = (8, 16, 32)


def last3_cells(raw, split):
    x, y = arr(raw["px_x_by_sample"]), arr(raw["px_y_by_sample"])
    out = []
    for k in (3, 2, 1):                                 # T−2, T−1, T
        f0 = split - 2 * k
        cx = (x[f0] + x[f0 + 1]) / 2; cy = (y[f0] + y[f0 + 1]) / 2
        out.append(np.clip(np.floor(np.array([cx, cy]) / CELL).astype(int), 0, G - 1))
    return out


def _worker(rank, world, a, n):
    dev = torch.device(f"cuda:{rank}"); torch.cuda.set_device(dev)
    ds64 = arlib.WMADataset(dict(arlib.v3_cfg(0, 64, split=0)))
    bundle = arlib.build_bundle(arlib.v3_cfg(4, 60), dev, window=64)
    recs = ds64.records
    idx = list(range(rank, n, world))
    loader = torch.utils.data.DataLoader(ClipDS(ds64, idx), batch_size=8, num_workers=a.workers, collate_fn=collate)
    mm = {C: {k: np.lib.format.open_memmap(Path(a.out) / f"{k}_C{C}.npy", mode="r+") for k in ("enc", "encF", "tgt", "pctx")}
          for C in CS}
    t0 = time.time(); done = 0
    for ids, clips in loader:
        B = len(ids)
        for C in CS:
            cfg = arlib.v3_cfg(C, 8, a.split)
            x = clips[:, :, a.split - C:a.split + 8]
            Tc = C // 2
            o = arlib.forward(bundle, x, cfg, ctx_frames=C, enc_layers=range(32), pred_layers=range(12), want_h=False)
            with torch.inference_mode(), arlib.autocast_ctx(cfg)():
                ht = bundle.target_encoder(x[:, :, :C].to(dev, bundle.dtype))
            E = np.zeros((B, 3, 32, D), np.float16); EF = np.zeros((B, 3, D), np.float32)
            TG = np.zeros((B, 3, D), np.float32); PQ = np.zeros((B, 3, 12, 384), np.float16)
            ln = lambda v: F.layer_norm(v.float(), (v.size(-1),))
            for b, i in enumerate(ids):
                for k, (cx, cy) in enumerate(last3_cells(recs[i].raw, a.split)):
                    tt = Tc - 3 + k
                    w = torch.tensor([tt * S_TOK + p for p in win_idx(cx, cy)], device=dev)
                    E[b, k] = torch.stack([ln(o["z_layers"][l][b, w]).mean(0) for l in range(32)]).half().cpu().numpy()
                    EF[b, k] = ln(o["z"][b, w]).mean(0).cpu().numpy(); TG[b, k] = ln(ht[b, w]).mean(0).cpu().numpy()
                    PQ[b, k] = torch.stack([ln(o["q_layers"][l][b, w]).mean(0) for l in range(12)]).half().cpu().numpy()
            ib = np.asarray(ids)
            mm[C]["enc"][ib] = E; mm[C]["encF"][ib] = EF; mm[C]["tgt"][ib] = TG; mm[C]["pctx"][ib] = PQ
            del o, ht
        done += B
        if rank == 0 and (done // B) % 10 == 0:
            el = time.time() - t0
            print(f"[rank0] {done}/{len(idx)}  {el/60:.1f} 분  남은 {el/done*(len(idx)-done)/60:.1f} 분", flush=True)
    for d in mm.values():
        for v in d.values():
            v.flush()
    print(f"[rank{rank}] 끝", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--split", type=int, default=32)
    ap.add_argument("--gpus", type=int, default=torch.cuda.device_count())
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    ds = arlib.WMADataset(dict(arlib.v3_cfg(0, 64, split=0)))
    n = a.limit or len(ds)
    for C in CS:
        for k, sh, dt in (("enc", (n, 3, 32, D), np.float16), ("encF", (n, 3, D), np.float32),
                          ("tgt", (n, 3, D), np.float32), ("pctx", (n, 3, 12, 384), np.float16)):
            m = np.lib.format.open_memmap(Path(a.out) / f"{k}_C{C}.npy", mode="w+", dtype=dt, shape=sh); del m
    json.dump(dict(n=n, split=a.split, Cs=CS, video_ids=[r.video_id for r in ds.records[:n]],
                   scenario=[r.raw["scenario"] for r in ds.records[:n]], plausible=[r.plausible for r in ds.records[:n]]),
              open(Path(a.out) / "meta.json", "w"))
    mp.spawn(_worker, args=(a.gpus, a, n), nprocs=a.gpus, join=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
