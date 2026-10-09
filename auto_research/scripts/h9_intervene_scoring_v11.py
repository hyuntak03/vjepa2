#!/usr/bin/env python3
"""H9 — H8 이 짚은 병목 (초기층 미래↔미래 섞임) 을 frozen 상태에서 풀면 v11 채점이 달라지나.

H4: 미래를 한 튜블릿씩 질의하면 먼 물체가 더 오래 남는다. H8: L0–6 의 미래↔미래 attention 을 끊으면 먼 물체가 더 오래 남는다 (대신 자리 틀어짐).
질문: 그 개입이 **채점** (matched pair surprise) 을 가림·움직임 조건에서 올리나 — 병목이 벤치마크 점수와 이어지나 (use case: frozen 개입).
표준 프로토콜 (C16/P16 stride 3, fp32 + autocast fp16, mask_index 0), v11_full block 1/4 표본 (H6f 와 같은 표본).
변형 (슬롯 j = 0..7 마다 mean|p_j − LN(h_full)_j|):
  clean      표준 (clean_forward)
  mm_L0-6    L0–6 에서 미래 → 미래 차단 (keep_self)
  mmx_all    전 층에서 미래 → 다른 튜블릿 미래 차단
  single     미래 튜블릿마다 따로 질의 (256 mask 토큰)
저장 (n, 4, 8).
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
from h7_extract_v11 import cfg_v11  # noqa: E402
from h4_single_query_v3 import idx_range  # noqa: E402
from analysis.attention.knockout.runner import clean_forward, resume  # noqa: E402

S_TOK, D, TC, K = 256, 1280, 8, 8
N = (TC + K) * S_TOK
VARS = ["clean", "mm_L0-6", "mmx_all", "single"]
OUT_DEFAULT = "/data2/local_datasets/world/world_analysis/cache/auto_research/v11_h9"


def masks(dev):
    ii = torch.arange(N, device=dev); t = ii // S_TOK; fut = t >= TC
    mm = fut[:, None] & fut[None, :]; mmx = mm & (t[:, None] != t[None, :])
    def keep(c):
        k = ~c; k.fill_diagonal_(True); return k
    a, b = keep(mm), keep(mmx)
    return {"mm_L0-6": [a if l <= 6 else None for l in range(12)], "mmx_all": [b] * 12}


def _worker(rank, world, a, sel):
    dev = torch.device(f"cuda:{rank}"); torch.cuda.set_device(dev)
    cfg = cfg_v11(); ds = arlib.WMADataset(cfg); bundle = arlib.build_bundle(cfg, dev, window=32); pred = bundle.predictor
    M = masks(dev)
    mine = [sel[i] for i in range(rank, len(sel), world)]; pos_of = {g: k for k, g in enumerate(sel)}
    loader = torch.utils.data.DataLoader(ClipDS(ds, mine), batch_size=4, num_workers=a.workers, collate_fn=collate)
    mm = np.lib.format.open_memmap(Path(a.out) / "l1.npy", mode="r+")
    ac = arlib.autocast_ctx(cfg); t0 = time.time(); done = 0
    for ids, clips in loader:
        B = len(ids); x = clips.to(dev, bundle.dtype)
        with torch.inference_mode(), ac():
            ci = idx_range(0, TC, B, dev); ti = idx_range(TC, K, B, dev)
            z = bundle.context_encoder(x, masks=[ci])
            hf = F.layer_norm(bundle.target_encoder(x), (D,)).float().reshape(B, 16, S_TOK, D)[:, TC:]
            p0, prefix, rm, n_ctx = clean_forward(pred, z, ci, ti, 0, keep_prefix=True)
            P = {"clean": p0.float()}
            for v, am in M.items():
                l0 = min(l for l in range(12) if am[l] is not None)
                P[v] = resume(pred, prefix[l0], l0, rm, n_ctx, attn_masks=am).float()
            P["single"] = torch.cat([pred(z, ci, idx_range(TC + j, 1, B, dev), mask_index=0).float() for j in range(K)], 1)
            r = torch.stack([(P[v].reshape(B, K, S_TOK, D) - hf).abs().mean((-1, -2)) for v in VARS], 1)
        mm[np.array([pos_of[i] for i in ids])] = r.cpu().numpy()
        done += B
        if rank == 0 and (done // B) % 100 == 0:
            el = time.time() - t0
            print(f"[rank0] {done}/{len(mine)}  {el/60:.1f} 분  남은 {el/done*(len(mine)-done)/60:.1f} 분", flush=True)
    mm.flush(); print(f"[rank{rank}] 끝", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DEFAULT); ap.add_argument("--gpus", type=int, default=torch.cuda.device_count())
    ap.add_argument("--workers", type=int, default=6); ap.add_argument("--every", type=int, default=4); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(); Path(a.out).mkdir(parents=True, exist_ok=True)
    ds = arlib.WMADataset(cfg_v11())
    sel = [i for i, r in enumerate(ds.records) if int(r.raw["block_id"]) % a.every == 0]
    if a.limit: sel = sel[: a.limit]
    m = np.lib.format.open_memmap(Path(a.out) / "l1.npy", mode="w+", dtype=np.float32, shape=(len(sel), len(VARS), K)); m[:] = np.nan; del m
    keys = ("video_id", "block_id", "pair_id", "plausible", "occ_timing", "motion", "violation_type", "variant", "sym_k", "condition")
    meta = {k: [ds.records[i].raw[k] for i in sel] for k in keys}; meta.update(n=len(sel), vars=VARS)
    json.dump(meta, open(Path(a.out) / "meta.json", "w"))
    mp.spawn(_worker, args=(a.gpus, a, sel), nprocs=a.gpus, join=True); print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
