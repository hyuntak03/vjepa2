#!/usr/bin/env python3
"""V-JEPA 2 표적 토큰은 국소적인가 — 물체 하나가 바뀌면 토큰 지도의 어디가 바뀌나 (2026-10-13).

design_headroom 에서: matched 쌍 (같은 문맥, 미래만 다름) 의 점수 차가 물체 3×3 안에서 4 %, 11×11 (면적 41 %) 안에서도 10 % 만 나온다 —
토큰당 차가 화면 전체에 고르다. 그렇다면 '토큰을 옮겨 미래를 만든다 (warp / move-content)' 는 이 공간에서 성립하지 않는다.
여기서는 렌더 없이 표적 자체로 잰다: d = |LN h_pos − LN h_imp| (블록별 LN(target_encoder), Ariel AR 공간, 미래 8 블록).
  raw      : 그대로
  demean   : 블록마다 토큰 평균 (전역 성분) 을 뺀 뒤 (h − mean_tokens h)
  global   : 토큰 평균끼리의 차 (전역 성분만) — 토큰마다 같은 값
그리고 물체 자리 (두 미래의 물체 합집합, 반경 r) 안 몫을 면적과 비교한다. 국소적이면 몫 ≫ 면적.
같이: 전역 성분의 크기 ‖mean‖² / mean‖h‖² (토큰 분산 중 전역 몫).

  python token_locality.py --gpus 4 [--space block|window]   (block = Ariel 공간, window = 32 장 창 단위 = release 표준 표적)
산출: z_research/VOBench_v1/headroom/token_locality.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "z_research/scripts/analysis"))
import design_headroom as H  # noqa: E402

S, NB, C, D, G = 256, 16, 8, 1280, 16
OUT = H.OUT
RADII = (0, 1, 2, 3)


def masks(lab, arms, r):
    yy, xx = np.mgrid[0:G, 0:G]; M = np.zeros((8, G, G), bool)
    for arm in arms:
        L = lab.get(arm)
        if not L or L[0] is None: continue
        P = np.array([[L[3 * s]["u"], L[3 * s]["v"]] for s in range(32)])[16:].reshape(8, 2, 2).mean(1)
        for k, p in enumerate(P):
            if not (0 <= p[0] < 512 and 0 <= p[1] < 512): continue
            y, x = H.tok(p); M[k] |= np.maximum(abs(yy - y), abs(xx - x)) <= r
    return M


def _worker(rank, world, bs, space):
    import torch
    import yaml
    from importlib.util import module_from_spec, spec_from_file_location
    from analysis.intphys2.model import build_from_config
    from app.vjepa_frozen.ar import encode_blocks
    from evals.world_model_analysis.data import WMADataset
    torch.cuda.set_device(rank); torch.set_num_threads(4); dev = torch.device("cuda", rank)
    spec = spec_from_file_location("ir", REPO / "z_research/scripts/analysis/rollout2_identity_readout.py")
    ir = module_from_spec(spec); spec.loader.exec_module(ir)
    bundle = build_from_config(dict(ir.rp.MODEL), dev)
    cfg = yaml.safe_load(open(H.SETS["vobench"]["cfg"]))
    ds = WMADataset(dict(data=dict(cfg["data"], index_csv="index.csv"), model=dict(ir.rp.MODEL), features={"cache_dir": "/tmp"}, surprise={}))
    pos_of = {rec.video_id: i for i, rec in enumerate(ds.records)}
    rows = {r["video_id"]: r for r in csv.DictReader(open(H.SETS["vobench"]["index"]))}
    blk = defaultdict(dict)
    for v, r in rows.items(): blk[r["source_block"]][r["variant"]] = v
    blocks = sorted(b for b, a in blk.items() if len(a) == 4 and rows[a["pos_a"]]["occlusion"] in ("visible", "early", "late"))
    mine = blocks[rank::world]
    rec = defaultdict(list); t0 = time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for bi, sb in enumerate(mine):
            a = blk[sb]; order = ("pos_a", "imp_ab", "pos_b", "imp_ba")
            x = torch.stack([ds.clip(pos_of[a[v]]) for v in order]).to(dev, dtype=bundle.dtype)
            if space == "block":
                h = encode_blocks(bundle.target_encoder, x, 2).float().view(4, NB, G, G, D)[:, C:]     # (4, 8, G, G, D) LN 됨
            else:                                                                                   # 창 단위: 32 장 한 번 (release 표준 표적)
                hw = bundle.target_encoder(x); hw = hw[-1] if isinstance(hw, list) else hw
                h = torch.nn.functional.layer_norm(hw.float(), (D,)).view(4, NB, G, G, D)[:, C:]
            mu = h.mean((2, 3), keepdim=True)
            r = rows[a["pos_a"]]
            lab = json.load(open(H.FR / Path(r["file_name"]).parent / "label.json"))["labels"]
            for (p, q, fa, fb) in ((0, 1, "A", "B"), (2, 3, "B", "A")):
                draw = (h[p] - h[q]).abs().mean(-1).cpu().numpy()
                ddem = ((h[p] - mu[p]) - (h[q] - mu[q])).abs().mean(-1).cpu().numpy()
                dglb = (mu[p] - mu[q]).abs().mean(-1).squeeze((-1, -2)).cpu().numpy()               # (8,)
                share = {}
                for rr in RADII:
                    M = masks(lab, (fa, fb), rr)
                    if not M.any(): continue
                    share[rr] = (M.mean(), draw[M].sum() / draw.sum(), ddem[M].sum() / ddem.sum(),
                                 draw[M].mean(), draw[~M].mean() if (~M).any() else np.nan, ddem[M].mean(), ddem[~M].mean() if (~M).any() else np.nan)
                mot = "static" if r["motion"] == "flat_static" else "moving"
                rec["key"].append(f"{r['objset']}/{r['violation']}/{r['occlusion']}/{mot}")
                rec["share"].append(np.array([share.get(rr, (np.nan,) * 7) for rr in RADII], np.float32))
                rec["glb"].append(dglb.mean()); rec["raw_mean"].append(draw.mean()); rec["dem_mean"].append(ddem.mean())
            gfrac = (mu.squeeze((2, 3)).pow(2).sum(-1) / h.pow(2).sum(-1).mean((2, 3))).mean().item()
            rec["gfrac"].append(gfrac)
            if rank == 0 and bi % 20 == 0: print(f"[r0] {bi}/{len(mine)} ({time.time() - t0:.0f}s)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"locality{SUF[space]}_r{rank}.npz", key=np.array(rec["key"]), share=np.stack(rec["share"]), glb=np.array(rec["glb"]),
             raw_mean=np.array(rec["raw_mean"]), dem_mean=np.array(rec["dem_mean"]), gfrac=np.array(rec["gfrac"]))


SUF = {"block": "", "window": "_window"}


def summarize(space="block"):
    Z = defaultdict(list)
    for f in sorted(OUT.glob(f"locality{SUF[space]}_r*.npz")):
        z = np.load(f)
        for k in z.files: Z[k].append(z[k])
    Z = {k: np.concatenate(v) for k, v in Z.items()}
    res = {"gfrac_mean": float(Z["gfrac"].mean())}
    print(f"전역 성분 몫 ‖mean_tokens h‖² / mean‖h‖² = {res['gfrac_mean']:.3f}  (블록 {len(Z['gfrac'])})")
    print("물체 자리 (반경 r) 안 몫 — 면적 | raw 몫 | demean 몫 | 토큰당 raw 안/밖 | 토큰당 demean 안/밖 ;  전역 차 · raw 평균 · demean 평균")
    for key in sorted(set(Z["key"])):
        m = Z["key"] == key; sh = np.nanmean(Z["share"][m], 0)
        res[key] = {"n": int(m.sum()), "radius": {str(rr): sh[i].tolist() for i, rr in enumerate(RADII)},
                    "glb": float(Z["glb"][m].mean()), "raw_mean": float(Z["raw_mean"][m].mean()), "dem_mean": float(Z["dem_mean"][m].mean())}
        print(f"  {key:36s} n{m.sum():3d} " + " ".join(f"r{rr}[{sh[i,0]:.2f}|{sh[i,1]:.2f}|{sh[i,2]:.2f}|{sh[i,3]:.2f}/{sh[i,4]:.2f}|{sh[i,5]:.2f}/{sh[i,6]:.2f}]" for i, rr in enumerate(RADII[:3]))
              + f" ; glb {Z['glb'][m].mean():.3f} raw {Z['raw_mean'][m].mean():.3f} dem {Z['dem_mean'][m].mean():.3f}")
    json.dump(res, open(OUT / f"token_locality{SUF[space]}.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--gpus", type=int, default=4); ap.add_argument("--summarize", action="store_true"); ap.add_argument("--space", default="block", choices=["block", "window"])
    a = ap.parse_args()
    if not a.summarize:
        import torch
        torch.multiprocessing.spawn(_worker, args=(a.gpus, 1, a.space), nprocs=a.gpus, join=True)
    summarize(a.space)
