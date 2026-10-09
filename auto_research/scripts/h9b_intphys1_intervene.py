#!/usr/bin/env python3
"""H9b — IntPhys 1 (Garrido 격자) 에 H9 개입: 표준 · 튜블릿별 단일 질의 · 미래 튜블릿 사이 차단 (mmx_all).
창마다 미래 튜블릿 j 의 mean|p_j − LN(h_full)_j| 를 저장 → h9b_analyze 가 Garrido 규칙 (시작점마다 C 최솟값 → 평균 → matched pair → macro) 로 채점.
검증: clean 의 창 평균 = per_window.json surprise.
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arlib  # noqa: E402
from evals.world_model_analysis.eval import _intphys1_windows  # noqa: E402
from h4_single_query_v3 import idx_range  # noqa: E402
from analysis.attention.knockout.runner import clean_forward, resume  # noqa: E402

S_TOK, D = 256, 1280
OUT = arlib.ROOT / "auto_research/exp_results/h9"


def mmx_mask(Tc, U, dev):
    N = U * S_TOK; ii = torch.arange(N, device=dev); t = ii // S_TOK; fut = t >= Tc
    cut = fut[:, None] & fut[None, :] & (t[:, None] != t[None, :]); k = ~cut; k.fill_diagonal_(True); return k


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--rank", type=int, default=0); ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()
    dev = torch.device("cuda:0")
    base = arlib.load_cfg("intphys1_sliding__intphys1_dev_vith"); cfg = dict(base, model=dict(base["model"], window_size=64))
    ds = arlib.WMADataset(cfg); bundle = arlib.build_bundle(cfg, dev, window=64); pred = bundle.predictor
    W = base["surprise"]["intphys1"]; combos = []
    for sk, wz in ((2, 32), (2, 16)):
        cl = [m * wz // 16 for m in W["context_mult"]]
        combos.append((f"skip{sk}_w{wz}", _intphys1_windows(ds.n_frames, sk, wz, cl, int(W["stride"]), 2, str(W["frame_budget"]))))
    idx = list(range(a.rank, len(ds), a.world))[: a.limit or None]
    ac = arlib.autocast_ctx(cfg); rows = []; t0 = time.time(); masks = {}
    for nd, i in enumerate(idx):
        clip = ds.clip(i)
        for ci_, (name, wins) in enumerate(combos):
            by_start = {}
            for C, cf, tf in wins:
                by_start.setdefault(cf[0], []).append((C, cf, tf))
            for st, lst in sorted(by_start.items()):
                x = clip[:, lst[0][1] + lst[0][2]][None].to(dev, bundle.dtype); wz = x.shape[2]; U = wz // 2
                with torch.inference_mode(), ac():
                    hf = F.layer_norm(bundle.target_encoder(x), (D,)).float().reshape(U, S_TOK, D)
                    for C, cf, tf in lst:
                        Tc = C // 2; ci = idx_range(0, Tc, 1, dev); ti = idx_range(Tc, U - Tc, 1, dev)
                        z = bundle.context_encoder(x, masks=[ci])
                        p0, prefix, rm, n_ctx = clean_forward(pred, z, ci, ti, 0, keep_prefix=True)
                        key = (Tc, U)
                        if key not in masks: masks[key] = mmx_mask(Tc, U, dev)
                        pm = resume(pred, prefix[0], 0, rm, n_ctx, attn_masks=[masks[key]] * 12)
                        ps = torch.cat([pred(z, ci, idx_range(Tc + j, 1, 1, dev), mask_index=0) for j in range(U - Tc)], 1)
                        for v, P in enumerate((p0, pm, ps)):
                            l1 = (P.float().reshape(U - Tc, S_TOK, D) - hf[Tc:]).abs().mean((-1, -2)).cpu().numpy()
                            for j, val in enumerate(l1):
                                rows.append((i, ci_, st, C, j, v, float(val)))
        if nd % 5 == 0:
            el = time.time() - t0; print(f"[{a.rank}] {nd+1}/{len(idx)} {el/60:.1f} 분 남은 {el/(nd+1)*(len(idx)-nd-1)/60:.1f} 분", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"ip1_intervene_r{a.rank}.npz", rows=np.array(rows), combos=np.array([c for c, _ in combos]),
             variants=np.array(["clean", "mmx_all", "single"]), video_ids=np.array([r.video_id for r in ds.records]))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
