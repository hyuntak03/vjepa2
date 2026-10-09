#!/usr/bin/env python3
"""H8 — 먼 미래의 물체를 지우는 것은 무엇이고 어느 층인가 (attention knockout, frozen). RollOut_v3.

H2: 뒤층 (L7–11) 을 지나며 먼 슬롯의 물체다움이 사라진다. H4: 미래를 한꺼번에 질의하면 (미래 토큰끼리 섞임) 단일 질의보다 빨리 사라진다.
→ 가설: **뒤층의 미래 튜블릿 사이 attention (mask→mask, 다른 튜블릿)** 이 먼 물체를 공통 미래 쪽으로 지운다.
예측: mask→mask:diff_tub@L7-11 을 끊으면 먼 슬롯 물체다움이 오른다. 대조: 같은 간선을 L0-6 에서 끊기, mask→ctx 를 끊기.

창: 문맥 [16, 32) (C16) + 미래 [32, 64) (P32 = 16 튜블릿). 명세 (keep_self=True, SDPA bool mask, True = attend):
  clean_null                        모든 층 전부 허용 (bool mask 경로 기준선)
  mm_L7-11 · mm_L0-6 · mm_all       mask→mask
  mmx_all · mmx_L7-11               mask→mask 중 **다른 튜블릿** 끼리만 (같은 튜블릿 안 공간 상호작용은 남김)
  mc_L7-11 · mc_L0-6                mask→ctx
저장 (n, spec, 16, 9): loc_tru(y,x,max,med) · loc_app(y,x,max,med) · l1  (H4 와 같은 읽기).
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
from h23_extract_v3 import ClipDS, collate  # noqa: E402
from h4_single_query_v3 import cellxy, idx_range, locate  # noqa: E402
from analysis.attention.knockout.runner import clean_forward, resume  # noqa: E402

G, S_TOK, D, SPLIT, C, K = 16, 256, 1280, 32, 16, 16
TC = C // 2
N = (TC + K) * S_TOK
SPECS = ["clean", "clean_null", "mm_L7-11", "mm_L0-6", "mm_all", "mmx_all", "mmx_L7-11", "mc_L7-11", "mc_L0-6"]
OUT_DEFAULT = "/data2/local_datasets/world/world_analysis/cache/auto_research/v3_h8"


def build_masks(dev):
    ii = torch.arange(N, device=dev); t = ii // S_TOK; fut = t >= TC
    mm = fut[:, None] & fut[None, :]
    mmx = mm & (t[:, None] != t[None, :])
    mc = fut[:, None] & ~fut[None, :]
    def keep(cut):
        k = ~cut; k.fill_diagonal_(True); return k
    full = torch.ones(N, N, dtype=torch.bool, device=dev)
    L = lambda a, b: list(range(a, b + 1))
    return {"clean_null": {l: full for l in range(12)},
            "mm_L7-11": {l: keep(mm) for l in L(7, 11)}, "mm_L0-6": {l: keep(mm) for l in L(0, 6)}, "mm_all": {l: keep(mm) for l in range(12)},
            "mmx_all": {l: keep(mmx) for l in range(12)}, "mmx_L7-11": {l: keep(mmx) for l in L(7, 11)},
            "mc_L7-11": {l: keep(mc) for l in L(7, 11)}, "mc_L0-6": {l: keep(mc) for l in L(0, 6)}}


def _worker(rank, world, a, sel):
    dev = torch.device(f"cuda:{rank}"); torch.cuda.set_device(dev)
    ds64 = arlib.WMADataset(dict(arlib.v3_cfg(0, 64, split=0)))
    bundle = arlib.build_bundle(arlib.v3_cfg(4, 60), dev, window=64)
    pred = bundle.predictor; recs = ds64.records
    masks = build_masks(dev)
    mine = [sel[i] for i in range(rank, len(sel), world)]
    pos_of = {g: k for k, g in enumerate(sel)}
    loader = torch.utils.data.DataLoader(ClipDS(ds64, mine), batch_size=2, num_workers=a.workers, collate_fn=collate)
    mm = np.lib.format.open_memmap(Path(a.out) / "h8.npy", mode="r+")
    ac = arlib.autocast_ctx(arlib.v3_cfg(16, 16)); t0 = time.time(); done = 0
    for ids, clips in loader:
        B = len(ids)
        x = clips[:, :, SPLIT - C:SPLIT + 2 * K].to(dev, bundle.dtype)
        with torch.inference_mode(), ac():
            h = F.layer_norm(bundle.target_encoder(x), (D,)).float().reshape(B, TC + K, S_TOK, D)
            ci = idx_range(0, TC, B, dev); ti = idx_range(TC, K, B, dev)
            z = bundle.context_encoder(x, masks=[ci])
            p0, prefix, rmasks, n_ctx = clean_forward(pred, z, ci, ti, 0, keep_prefix=True)
            outs = {"clean": p0.float()}
            for sp, lm in masks.items():
                l0 = min(lm); am = [lm.get(l) for l in range(12)]
                outs[sp] = resume(pred, prefix[l0], l0, rmasks, n_ctx, attn_masks=am).float()
        last = [cellxy(recs[i].raw, SPLIT - 2) for i in ids]
        app = torch.stack([h[b, TC - 1, last[b][1] * G + last[b][0]] for b in range(B)])
        res = np.zeros((B, len(SPECS), K, 9), np.float32)
        for k in range(K):
            tru = [cellxy(recs[i].raw, SPLIT + 2 * k) for i in ids]
            tt = torch.stack([h[b, TC + k, tru[b][1] * G + tru[b][0]] for b in range(B)])
            for si, sp in enumerate(SPECS):
                P = outs[sp].reshape(B, K, S_TOK, D)[:, k]
                res[:, si, k, 0:4] = locate(P, tt).cpu().numpy(); res[:, si, k, 4:8] = locate(P, app).cpu().numpy()
                res[:, si, k, 8] = (P - h[:, TC + k]).abs().mean((-1, -2)).cpu().numpy()
        mm[np.array([pos_of[i] for i in ids])] = res
        done += B
        if rank == 0 and (done // B) % 25 == 0:
            el = time.time() - t0
            print(f"[rank0] {done}/{len(mine)}  {el/60:.1f} 분  남은 {el/done*(len(mine)-done)/60:.1f} 분  "
                  f"clean_null−clean 상대 {((outs['clean_null']-outs['clean']).abs().mean()/outs['clean'].abs().mean()).item():.4f}", flush=True)
    mm.flush(); print(f"[rank{rank}] 끝", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--gpus", type=int, default=torch.cuda.device_count())
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    ds = arlib.WMADataset(dict(arlib.v3_cfg(0, 64, split=0)))
    free = ("flat_v", "flat_a", "flat_d", "ramp_a", "ramp_d", "arc")
    sel = [i for i, r in enumerate(ds.records) if r.plausible == "1" and r.raw["scenario"] in free]
    if a.limit:
        sel = sel[:: max(1, len(sel) // a.limit)][: a.limit]
    m = np.lib.format.open_memmap(Path(a.out) / "h8.npy", mode="w+", dtype=np.float32, shape=(len(sel), len(SPECS), K, 9)); m[:] = np.nan; del m
    json.dump(dict(n=len(sel), specs=SPECS, C=C, K=K, split=SPLIT, video_ids=[ds.records[i].video_id for i in sel]),
              open(Path(a.out) / "meta.json", "w"))
    mp.spawn(_worker, args=(a.gpus, a, sel), nprocs=a.gpus, join=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
