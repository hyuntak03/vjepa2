"""블록 9 를 만드는 query 가 문맥 블록마다 물체에 얼마나 attend 하나 — V-JEPA 2 · V-JEPA 2 인과 문맥 (R1) · 블록 AR (2026-11-06).

query = 블록 9 시점의 실제 물체 위치 (가려져 있어도) 둘레 3×3 칸.
  V-JEPA 2 (R1 포함): 블록 9 자리의 mask token. key = 문맥 블록 1–8 + 미래 mask 9–16.
  AR               : 블록 8 자리 입력 토큰 (출력 = 블록 9 예측). key = 블록 1–8.
key 물체 자리 = 블록 t 시점의 실제 물체 위치 둘레 3×3 (가려진 블록이면 가림막 뒤 자리).
클립마다 (층, key 블록) 별로 [물체 자리 질량, 블록 전체 질량] 을 헤드 · query 평균으로 남긴다.
attention weight 는 SDPA 를 predictor 호출 동안만 감싸 q·k 로 다시 계산한다 (출력은 원래 SDPA 그대로).
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from attn_release_vs_ar import (C, L, NB, ROOT, S, load_cfg, obj_track, region)  # noqa: E402

from analysis.intphys2.model import build_from_config  # noqa: E402
from analysis.intphys2.surprise import _context_target_indices  # noqa: E402
from evals.world_model_analysis.data import WMADataset  # noqa: E402

F9 = 8              # 블록 9 (0 부터 센 인덱스)
_ORIG = F.scaled_dot_product_attention


class St:
    on = False
    calls = 0
    mode = ""
    oq = None       # 클립별 query 칸 (블록 안 인덱스)
    ok = None       # 클립별 (NB, 9) key 물체 칸 텐서
    out = None      # 클립별 (L, NB, 2) 누적
    verify = {}


def _rec(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kw):
    out = _ORIG(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kw)
    if not St.on:
        return out
    li, step = St.calls % L, St.calls // L
    St.calls += 1
    B, Hh, Nq, D = q.shape
    Nk = k.size(2)
    sc = scale if scale is not None else D ** -0.5
    if St.mode == "ar":
        if step != F9 - 1:                      # 걸음 7 = 블록 8 입력 → 블록 9 예측
            return out
        assert Nq == S and Nk == F9 * S, (Nq, Nk)
        base = 0
    else:
        assert Nq == Nk == NB * S, (Nq, Nk)
        base = F9 * S
    nkb = Nk // S
    for bi in range(B):
        qi = base + St.oq[bi]
        p = (torch.matmul(q[bi, :, qi].float(), k[bi].float().transpose(-1, -2)) * sc).softmax(-1)   # (H, nq, Nk)
        if St.mode not in St.verify and li == 0:
            o = torch.matmul(p, v[bi].float())
            St.verify[St.mode] = float((o - out[bi, :, qi].float()).abs().max())
        pm = p.mean((0, 1)).view(nkb, S)                                                          # (nkb, S)
        ok, w = St.ok[bi][0][:nkb], St.ok[bi][1][:nkb]                                            # (nkb, 9)
        St.out[bi][li, :nkb, 0] += (torch.gather(pm, 1, ok) * w).sum(1).double().cpu().numpy()
        St.out[bi][li, :nkb, 1] += pm.sum(1).double().cpu().numpy()
    return out


@contextlib.contextmanager
def rec(mode, oq, ok, n):
    St.on, St.calls, St.mode, St.oq, St.ok = True, 0, mode, oq, ok
    St.out = [np.zeros((L, NB, 2)) for _ in range(n)]
    F.scaled_dot_product_attention = _rec
    try:
        yield
    finally:
        F.scaled_dot_product_attention = _ORIG
        St.on = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "z_analysis/results/release_ladder_20261106/attn_block9_object.json"))
    ap.add_argument("--models", default="release,release_causal,ar")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    dev = torch.device("cuda:0")

    rows = {r["video_id"]: r for r in csv.DictReader(open(os.path.join(ROOT, "data_csv/vobench_v1/index.csv")))}
    ds = WMADataset(load_cfg("release"))
    clips = [i for i, x in enumerate(ds.records)
             if rows[x.video_id]["violation_type"] == "permanence" and rows[x.video_id]["variant"] == "pos_a"]
    if a.limit:
        clips = clips[:a.limit]
    meta = []
    for i in clips:
        r = rows[ds.records[i].video_id]
        vis, rc = obj_track(r["file_name"])
        meta.append(dict(video_id=r["video_id"], occlusion=r["occlusion"], k=r["k"], motion=r["motion"],
                         vis=vis.tolist(), rc=rc))
    print(f"클립 {len(clips)}", flush=True)
    ci, ti = _context_target_indices(ctx_frames=16, tgt_frames=16, tubelet_size=2, spatial_tokens=S,
                                     batch_size=a.bs, device=dev)
    res = {"meta": meta, "models": {}}
    for name in a.models.split(","):
        cfg = load_cfg(name)
        t0 = time.time()
        bundle = build_from_config(cfg["model"], dev)
        mode = "ar" if name.startswith("ar") else "release"
        per = []
        for s in range(0, len(clips), a.bs):
            idx = list(range(s, min(s + a.bs, len(clips))))
            n = len(idx)
            x = torch.stack([ds.clip(clips[j]) for j in idx]).to(dev, bundle.dtype)
            oq = [torch.tensor(region(meta[j]["rc"][F9]), device=dev) for j in idx]
            regs = [[region(meta[j]["rc"][t]) for t in range(NB)] for j in idx]          # 가장자리는 9 칸보다 적다 → 0 으로 채우고 가중 0
            ok = [(torch.tensor([r + [0] * (9 - len(r)) for r in rr], device=dev),
                   torch.tensor([[1.0] * len(r) + [0.0] * (9 - len(r)) for r in rr], device=dev)) for rr in regs]
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                z = bundle.context_encoder(x, masks=[ci[:n]])
                with rec(mode, oq, ok, n):
                    bundle.predictor(z, ci[:n], ti[:n], mask_index=0)
            per += [o.tolist() for o in St.out]
        res["models"][name] = per
        print(f"[{name}] {time.time()-t0:.0f}s verify max|Δ| {St.verify.get(mode)}", flush=True)
        St.verify.pop(mode, None)
        del bundle
        torch.cuda.empty_cache()
    json.dump(res, open(a.out, "w"))
    print("저장", a.out)


if __name__ == "__main__":
    main()
