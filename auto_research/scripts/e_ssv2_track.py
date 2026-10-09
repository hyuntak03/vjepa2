#!/usr/bin/env python3
"""SC2 — 라벨 없는 실영상 지평 곡선 (SSv2 좌/우, 2026-09-25 oral 방향 sanity check).

합성 (v3) 에서는 외형 템플릿을 **진짜 미래 target h** 에 걸면 진실 물체 칸을 0.75–0.91 찾는다 (P2 apph, encoder 천장).
그러니 실영상에서도 "target encoder 가 진짜 미래 프레임에서 그 토큰을 따라간 궤적" 을 **유사 정답** 으로 쓸 수 있다 (합성에서 보정된 판독).
질문: predictor 가 같은 토큰을 미래 슬롯에서 **어디에** 두나 — encoder 궤적을 몇 슬롯까지 따라가나.

clip: e_ssv2_extract.py 와 같은 전처리 (균등 32 장, 256², 문맥 16 / 미래 16 = 8 튜블릿), 정방향만 (역재생은 --rev).
질의 토큰 q: 인과 표적 hc (target encoder 를 문맥 16 장에만) 의 마지막 두 튜블릿 변화량 |hc_7 − hc_6| 이 가장 큰 칸 (움직이는 것).
템플릿: T = h_7[q] (표준 양방향), Tc = hc_7[q] (인과).
저장 (clip, 슬롯 i = 0..7):
  enc  argmax_s cos(h_{8+i}[s], T)   — 유사 정답 궤적 (진짜 미래를 본 target encoder)
  encc argmax_s cos(h_{8+i}[s], Tc)
  p    argmax_s cos(p_i[s], T),  pc  argmax_s cos(p_i[s], Tc)
  각각 (y, x, maxcos, median).  q 칸, 변화량 최대값.
분석 (e_ssv2_track_analyze.py): encoder 궤적이 문맥 끝에서 ≥ 2 칸 움직인 clip 에서
  hit_i = ‖p_i − enc_i‖∞ ≤ 1 − 귀무 (다른 clip 의 enc_i), 복사 (q 칸 그대로) 기준선, 진행률 = (p − q)·(enc − q)/|enc − q|².

  ARLIB_PRED_CKPT=<ckpt> python e_ssv2_track.py --gpus 8 --out <dir>      (predictor 교체; kind 는 체크포인트가 정함)
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
from h4_single_query_v3 import idx_range, locate  # noqa: E402
from e_ssv2_extract import load_clip, SSV2  # noqa: E402

S_TOK, D, NF, TC, G = 256, 1280, 32, 8, 16
OUT_DEFAULT = "/data2/local_datasets/world/world_analysis/cache/auto_research/ssv2_track"


def _worker(rank, world, a, files):
    dev = torch.device(f"cuda:{rank}"); torch.cuda.set_device(dev)
    cfg = arlib.v3_cfg(16, 16); bundle = arlib.build_bundle(cfg, dev, window=NF); pred = bundle.predictor; ac = arlib.autocast_ctx(cfg)
    M = np.lib.format.open_memmap(Path(a.out) / "track.npy", mode="r+")        # (n, dirs, 4 [enc, encc, p, pc], 8, 4)
    Q = np.lib.format.open_memmap(Path(a.out) / "query.npy", mode="r+")        # (n, dirs, 3) qy, qx, energy
    ci = idx_range(0, TC, 1, dev); ti = idx_range(TC, TC, 1, dev)
    mine = list(range(rank, len(files), world)); t0 = time.time()
    for n_done, row in enumerate(mine):
        x0 = load_clip(files[row])
        for di, x in enumerate((x0, torch.flip(x0, dims=[1])) if a.rev else (x0,)):
            x = x[None].to(dev, bundle.dtype)
            with torch.inference_mode(), ac():
                h = F.layer_norm(bundle.target_encoder(x), (D,)).float().reshape(16, S_TOK, D)
                hc = F.layer_norm(bundle.target_encoder(x[:, :, :16]), (D,)).float().reshape(8, S_TOK, D)
                z = bundle.context_encoder(x, masks=[ci])
                p = pred(z, ci, ti, mask_index=0).float().reshape(8, S_TOK, D)
            e = (hc[7] - hc[6]).abs().mean(-1); q = int(e.argmax())
            T, Tc = h[7, q][None], hc[7, q][None]
            Q[row, di] = [q // G, q % G, float(e[q])]
            for k, (src, tpl) in enumerate(((h[8:], T), (h[8:], Tc), (p, T), (p, Tc))):
                M[row, di, k] = locate(src, tpl.expand(8, -1)).cpu().numpy()
        if rank == 0 and n_done % 20 == 0:
            el = time.time() - t0; print(f"[rank0] {n_done+1}/{len(mine)} {el/60:.1f} 분 남은 {el/(n_done+1)*(len(mine)-n_done-1)/60:.1f} 분", flush=True)
    M.flush(); Q.flush(); print(f"[rank{rank}] 끝", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DEFAULT); ap.add_argument("--gpus", type=int, default=torch.cuda.device_count())
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--root", default=str(SSV2)); ap.add_argument("--rev", action="store_true")
    a = ap.parse_args(); Path(a.out).mkdir(parents=True, exist_ok=True)
    import decord
    files, labels = [], []
    for lab, c in enumerate(("left", "right")):
        for f in sorted((Path(a.root) / c).glob("*.mp4")):
            if len(decord.VideoReader(str(f), num_threads=1)) >= NF: files.append(f); labels.append(lab)
    if a.limit:
        k = np.linspace(0, len(files) - 1, a.limit).round().astype(int); files = [files[i] for i in k]; labels = [labels[i] for i in k]
    n = len(files); nd = 2 if a.rev else 1
    m = np.lib.format.open_memmap(Path(a.out) / "track.npy", mode="w+", dtype=np.float32, shape=(n, nd, 4, 8, 4)); m[:] = np.nan; del m
    m = np.lib.format.open_memmap(Path(a.out) / "query.npy", mode="w+", dtype=np.float32, shape=(n, nd, 3)); m[:] = np.nan; del m
    import os
    json.dump(dict(n=n, files=[str(f) for f in files], labels=labels, dirs=["forward", "reversed"][:nd], rows=["enc", "encc", "p", "pc"],
                   predictor=os.environ.get("ARLIB_PRED_CKPT", "release")), open(Path(a.out) / "meta.json", "w"))
    print("clips", n, flush=True)
    mp.spawn(_worker, args=(a.gpus, a, files), nprocs=a.gpus, join=True); print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
