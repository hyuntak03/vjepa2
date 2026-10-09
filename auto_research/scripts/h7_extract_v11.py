#!/usr/bin/env python3
"""H7 추출 — IntPhysGen v11_full: predictor 층별로 미래 토큰의 정체성 (모양·색) 은 어디서 "보이던 형식" 을 잃나.

알려진 것 (CLAUDE.md §5-2): late 가림에서 최종 p 의 정체성 self probe 98~99 %, 그런데 비가림→가림 이식은 17 %.
→ 정보는 있는데 형식 (자리·좌표) 이 다르다. **predictor 의 몇 번째 층에서 그 형식이 갈라지나** 를 잰다.
표준 프로토콜 그대로 (stride 3, 문맥 16 / 미래 16 프레임, fp32 + autocast fp16, mask_index 0).
대상: v11_full **전부** 43,008 clip (probing 은 probe_type == obj 18,816 만 쓴다; 채점은 전부).
clip 마다 저장 (전 토큰 평균 풀링):
  lens  (n, 12, 8, 1280) fp16  predictor 렌즈 p^(l) 의 미래 튜블릿 8 개
  enc   (n, 8, 8, 1280)  fp16  문맥 encoder 층 {3,7,...,31} 의 문맥 튜블릿 8 개 (LN)
  h     (n, 16, 1280)    fp16  LN(target_encoder) 32 프레임 (천장)
  l1    (n, 12, 8)       fp32  층별 렌즈 채점: mean|p^(l)_t − LN(h)_t| (슬롯 t 전 토큰) — L11 의 슬롯 평균 = 표준 surprise (검증)
"""
from __future__ import annotations
import argparse, csv, json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arlib  # noqa: E402
from h23_extract_v3 import ClipDS, collate  # noqa: E402

S_TOK, D = 256, 1280
ENC_L = (3, 7, 11, 15, 19, 23, 27, 31)
OUT_DEFAULT = "/data2/local_datasets/world/world_analysis/cache/auto_research/v11_h7"


def cfg_v11():
    cfg = arlib.load_cfg("surprise_c16t32__v11_vith")
    cfg["data"] = dict(cfg["data"], root=str(arlib.ROOT / "data_csv/intphysgen_v11_full"), index_csv="index_probe.csv")
    cfg["features"] = {"cache_dir": "/tmp"}
    return cfg


def _worker(rank, world, a, sel):
    dev = torch.device(f"cuda:{rank}"); torch.cuda.set_device(dev)
    cfg = cfg_v11(); ds = arlib.WMADataset(cfg)
    bundle = arlib.build_bundle(cfg, dev, window=32)
    mine = list(range(rank, len(sel), world))
    loader = torch.utils.data.DataLoader(ClipDS(ds, [sel[i] for i in mine]), batch_size=8, num_workers=a.workers, collate_fn=collate)
    pos_of = {g: k for k, g in enumerate(sel)}
    mm = {k: np.lib.format.open_memmap(Path(a.out) / f"{k}.npy", mode="r+") for k in ("lens", "enc", "h", "l1")}
    t0 = time.time(); done = 0
    for ids, clips in loader:
        B = len(ids)
        o = arlib.forward(bundle, clips, cfg, ctx_frames=16, enc_layers=ENC_L, pred_layers=range(12), want_h=True, want_lens=True)
        rows = np.array([pos_of[i] for i in ids])
        mm["lens"][rows] = torch.stack([o["lens"][l].float().reshape(B, 8, S_TOK, D).mean(2) for l in range(12)], 1).half().cpu().numpy()
        mm["enc"][rows] = torch.stack([F.layer_norm(o["z_layers"][l].float(), (D,)).reshape(B, 8, S_TOK, D).mean(2) for l in ENC_L], 1).half().cpu().numpy()
        hh = o["h"].float().reshape(B, 16, S_TOK, D)
        mm["h"][rows] = hh.mean(2).half().cpu().numpy()
        mm["l1"][rows] = torch.stack([(o["lens"][l].float().reshape(B, 8, S_TOK, D) - hh[:, 8:]).abs().mean((-1, -2)) for l in range(12)], 1).cpu().numpy()
        del o
        done += B
        if rank == 0 and (done // B) % 25 == 0:
            el = time.time() - t0
            print(f"[rank0] {done}/{len(mine)}  {el/60:.1f} 분  남은 {el/done*(len(mine)-done)/60:.1f} 분", flush=True)
    for v in mm.values():
        v.flush()
    print(f"[rank{rank}] 끝", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--gpus", type=int, default=torch.cuda.device_count())
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    ds = arlib.WMADataset(cfg_v11())
    sel = list(range(len(ds.records)))
    if a.limit:
        sel = sel[:: max(1, len(sel) // a.limit)][: a.limit]
    n = len(sel)
    for k, sh, dt in (("lens", (n, 12, 8, D), np.float16), ("enc", (n, len(ENC_L), 8, D), np.float16), ("h", (n, 16, D), np.float16),
                      ("l1", (n, 12, 8), np.float32)):
        m = np.lib.format.open_memmap(Path(a.out) / f"{k}.npy", mode="w+", dtype=dt, shape=sh); del m
    keys = ("video_id", "block_id", "condition", "motion", "occ_timing", "sym_k", "shape_pre", "color_pre", "shape_post", "color_post",
            "env", "violation_type", "variant", "plausible", "pair_id", "probe_type", "direction")
    meta = {k: [ds.records[i].raw[k] for i in sel] for k in keys}
    meta.update(n=n, enc_layers=ENC_L, numerics="fp32 + autocast fp16, C16/P16 stride 3, mask_index 0; 전 토큰 평균 풀링")
    json.dump(meta, open(Path(a.out) / "meta.json", "w"))
    print(f"n={n}", flush=True)
    mp.spawn(_worker, args=(a.gpus, a, sel), nprocs=a.gpus, join=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
