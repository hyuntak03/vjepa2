#!/usr/bin/env python3
"""H6f — v11_full 에 IntPhys 1 과 같은 대조: 예측 없는 복사 기준선 · 인과 표적 (튜블릿 단위).

표준 프로토콜 (C16 / P16 stride 3, fp32 + autocast fp16, mask_index 0). 미래 튜블릿 j = 0..7 마다 전 토큰 mean L1:
  p_full   = |p_j − LN(h_full)_j|          h_full = target_encoder(32 프레임 전부)     ← 표준 채점 (슬롯 평균 = 표준 surprise)
  copyz_full = |LN(z)_T − LN(h_full)_j|      (문맥 마지막 튜블릿 복사)
  p_causal / copyz_causal                   h_causal_j = LN(target_encoder(프레임 [0, 2(8+j+1))))[튜블릿 8+j]
표본: block 을 --every 개마다 하나 (기본 4 → 약 10,752 clip), matched pair 를 온전히 남긴다.
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

S_TOK, D = 256, 1280
OUT_DEFAULT = "/data2/local_datasets/world/world_analysis/cache/auto_research/v11_h6f"


def _worker(rank, world, a, sel):
    dev = torch.device(f"cuda:{rank}"); torch.cuda.set_device(dev)
    cfg = cfg_v11(); ds = arlib.WMADataset(cfg); bundle = arlib.build_bundle(cfg, dev, window=32)
    mine = [sel[i] for i in range(rank, len(sel), world)]; pos_of = {g: k for k, g in enumerate(sel)}
    loader = torch.utils.data.DataLoader(ClipDS(ds, mine), batch_size=8, num_workers=a.workers, collate_fn=collate)
    mm = np.lib.format.open_memmap(Path(a.out) / "l1.npy", mode="r+")        # (n, 8, 4)
    ac = arlib.autocast_ctx(cfg); t0 = time.time(); done = 0
    for ids, clips in loader:
        B = len(ids); x = clips.to(dev, bundle.dtype)
        with torch.inference_mode(), ac():
            ci, ti = arlib.ci_ti(B, 16, 16, dev)
            z = bundle.context_encoder(x, masks=[ci]); p = bundle.predictor(z, ci, ti, mask_index=0).float().reshape(B, 8, S_TOK, D)
            hf = F.layer_norm(bundle.target_encoder(x), (D,)).float().reshape(B, 16, S_TOK, D)[:, 8:]
            zT = F.layer_norm(z.float(), (D,)).reshape(B, 8, S_TOK, D)[:, 7:8]
            hc = torch.stack([F.layer_norm(bundle.target_encoder(x[:, :, :2 * (8 + j + 1)]), (D,)).float()
                              .reshape(B, 8 + j + 1, S_TOK, D)[:, 8 + j] for j in range(8)], 1)
            r = torch.stack([(p - hf).abs().mean((-1, -2)), (zT - hf).abs().mean((-1, -2)),
                             (p - hc).abs().mean((-1, -2)), (zT - hc).abs().mean((-1, -2))], -1)
        mm[np.array([pos_of[i] for i in ids])] = r.cpu().numpy()
        done += B
        if rank == 0 and (done // B) % 50 == 0:
            el = time.time() - t0
            print(f"[rank0] {done}/{len(mine)}  {el/60:.1f} 분  남은 {el/done*(len(mine)-done)/60:.1f} 분", flush=True)
    mm.flush(); print(f"[rank{rank}] 끝", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DEFAULT); ap.add_argument("--gpus", type=int, default=torch.cuda.device_count())
    ap.add_argument("--workers", type=int, default=6); ap.add_argument("--every", type=int, default=4)
    a = ap.parse_args(); Path(a.out).mkdir(parents=True, exist_ok=True)
    ds = arlib.WMADataset(cfg_v11())
    sel = [i for i, r in enumerate(ds.records) if int(r.raw["block_id"]) % a.every == 0]
    m = np.lib.format.open_memmap(Path(a.out) / "l1.npy", mode="w+", dtype=np.float32, shape=(len(sel), 8, 4)); m[:] = np.nan; del m
    keys = ("video_id", "block_id", "pair_id", "plausible", "occ_timing", "motion", "violation_type", "variant", "sym_k", "condition")
    meta = {k: [ds.records[i].raw[k] for i in sel] for k in keys}
    meta.update(n=len(sel), cols=["p_full", "copyz_full", "p_causal", "copyz_causal"])
    json.dump(meta, open(Path(a.out) / "meta.json", "w"))
    mp.spawn(_worker, args=(a.gpus, a, sel), nprocs=a.gpus, join=True); print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
