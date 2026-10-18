#!/usr/bin/env python3
"""문맥 끝에 가려졌다 다시 나타나는 물체 (VOBench late) — 정체가 predictor 안에 남는가, 남는다면 제자리에 놓이는가 (2026-10-13).

대상: VOBench 가능 영상 (pos_a · pos_b) 중 물체가 있는 것, objset 둘 × 가림 (visible · early · mid · late) × 운동.
모델: release (문맥 16 장 → 8 블록 한 번) · AR ep40 · AR-state ep40 (블록별 LN(h) 문맥 8 블록 → rollout 8).
걸음 k 마다 저장: 은닉 st (predictor_norm, 384) 와 그린 출력 r (1280) 의 블록 평균 (g) · 참 물체 자리 3×3 평균 (l),
  참 h 블록의 같은 둘 (천장), 문맥 안 마지막 '보이는' 블록의 h (정체가 문맥에 있었다는 대조).
분석: objset · 가림 · (정지/이동) 칸 안에서 block 단위 5-fold ridge-one-hot → 모양 (6) · 색 정확도.
  '은닉 g 로는 읽히는데 l 로는 안 읽힌다' = 정체는 있으나 제자리에 안 놓임. 둘 다 안 읽히면 정체를 안 가져옴.

  python predictor_late_identity.py extract --name ar40 --ckpt z_training/ariel/ar_future_1e_5/latest.pt --gpus 4
  python predictor_late_identity.py analyze --names release ar40 ar_state40
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
FR = Path("/data2/local_datasets/world/Benchmark/VOBench_v1")
INDEX = REPO / "data_csv/vobench_v1/index.csv"
RES_CFG = REPO / "z_research/VOBench_v1/exp_results/surprise_c16t32__vobench_v1_vith/_resolved.yaml"
OUT = REPO / "z_research/VOBench_v1/bottleneck/late_identity"
S, NB, C, K, D, GRID, CELL = 256, 16, 8, 8, 1280, 16, 32.0


def obj_of(r):
    return r["obj_a"] if r["variant"] == "pos_a" else r["obj_b"]


def wanted():
    return {r["video_id"]: r for r in csv.DictReader(INDEX.open()) if r["variant"] in ("pos_a", "pos_b") and obj_of(r) != "empty"}


def track(r):
    """블록 중심 (16, 2) px@512 · 블록 가려짐 (16,) · 문맥 안 마지막 보이는 블록."""
    lab = json.load(open(FR / Path(r["file_name"]).parent / "label.json"))["labels"]
    L = lab["A" if r["variant"] == "pos_a" else "B"]
    P = np.array([[L[3 * s]["u"], L[3 * s]["v"]] for s in range(32)], np.float32).reshape(16, 2, 2).mean(1)
    hid = np.array([L[3 * s]["hidden"] for s in range(32)], bool).reshape(16, 2).any(1)
    vis_ctx = [b for b in range(C) if not hid[b]]
    return P, hid, (vis_ctx[-1] if vis_ctx else 0)


def loc9(xy):
    c = int(np.clip(xy[0] // CELL, 0, GRID - 1)); rr = int(np.clip(xy[1] // CELL, 0, GRID - 1))
    return [min(max(rr + a, 0), GRID - 1) * GRID + min(max(c + b, 0), GRID - 1) for a in (-1, 0, 1) for b in (-1, 0, 1)]


def _worker(rank, world, name, ckpt, bs):
    import torch
    import torch.nn.functional as F
    import yaml
    from importlib.util import module_from_spec, spec_from_file_location
    from analysis.intphys2.model import build_from_config
    from app.vjepa_frozen.ar import block_indices, encode_blocks
    from evals.world_model_analysis.data import WMADataset
    torch.cuda.set_device(rank); torch.set_num_threads(4); dev = torch.device("cuda", rank)
    os.environ.setdefault("PRESENCE_SET", "vob_readout_primitive")
    spec = spec_from_file_location("ir", REPO / "z_research/scripts/analysis/rollout2_identity_readout.py")
    ir = module_from_spec(spec); spec.loader.exec_module(ir)
    model = dict(ir.rp.MODEL) if name == "release" else dict(ir.rp.MODEL, predictor_checkpoint=str(REPO / ckpt))
    bundle = build_from_config(model, dev); pred = bundle.predictor
    is_ar = getattr(pred, "mask_mode", "release") in ("ar", "ar_state")
    cfg = yaml.safe_load(open(RES_CFG))
    ds = WMADataset(dict(data=dict(cfg["data"], index_csv="index.csv"), model=model, features={"cache_dir": "/tmp"}, surprise={}))
    W = wanted()
    ids = [i for i, r in enumerate(ds.records) if r.video_id in W]
    mine = ids[rank * len(ids) // world:(rank + 1) * len(ids) // world]
    LN = lambda x: F.layer_norm(x.float(), (x.size(-1),))
    rec = defaultdict(list); t0 = time.time()

    def pools(x, loc):
        g = x.float().mean(1); l = torch.gather(x.float(), 1, loc.unsqueeze(-1).expand(-1, -1, x.size(-1))).mean(1)
        return g.half().cpu().numpy(), l.half().cpu().numpy()

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for b0 in range(0, len(mine), bs):
            ch = mine[b0:b0 + bs]
            x = torch.stack([ds.clip(i) for i in ch]).to(dev, dtype=bundle.dtype); B = x.size(0)
            tr = [track(W[ds.records[i].video_id]) for i in ch]
            loc = torch.tensor(np.array([[loc9(P[t]) for t in range(NB)] for P, _, _ in tr]), device=dev)
            lastvis = torch.tensor([lv for _, _, lv in tr], device=dev)
            if is_ar:
                hb = encode_blocks(bundle.target_encoder, x, 2)
                cap = []
                hk = pred.predictor_norm.register_forward_hook(lambda m, i, o: cap.append(o))
                p = pred.rollout(hb[:, :C * S], block_indices(B, C, S, dev), K); hk.remove()
                if getattr(pred, "mask_mode", "") == "ar_state":
                    st = torch.cat([c_[:, -S:] for c_ in cap[-K:]], 1)       # 걸음마다 마지막 블록의 상태
                else:
                    st = torch.cat([c_[:, -S:] for c_ in cap[-K:]], 1)
                h_all = hb
            else:
                from analysis.intphys2.surprise import _context_target_indices
                ci, ti = _context_target_indices(ctx_frames=2 * C, tgt_frames=2 * K, tubelet_size=2, spatial_tokens=S, batch_size=B, device=dev)
                cap = []
                hk = pred.predictor_norm.register_forward_hook(lambda m, i, o: cap.append(o))
                z = bundle.context_encoder(x, masks=[ci]); z = z[-1] if isinstance(z, list) else z
                p = pred(z, ci, ti, mask_index=0); hk.remove()
                st = cap[-1][:, -K * S:]
                h = bundle.target_encoder(x); h_all = LN(h[-1] if isinstance(h, list) else h)
            for k in range(1, K + 1):
                t = C + k - 1; lc = loc[:, t]
                g, l = pools(st[:, (k - 1) * S:k * S], lc); rec[f"st_g{k}"].append(g); rec[f"st_l{k}"].append(l)
                g, l = pools(p[:, (k - 1) * S:k * S], lc); rec[f"r_g{k}"].append(g); rec[f"r_l{k}"].append(l)
                g, l = pools(h_all[:, t * S:(t + 1) * S], lc); rec[f"h_g{k}"].append(g); rec[f"h_l{k}"].append(l)
            hv = torch.stack([h_all[b, lastvis[b] * S:(lastvis[b] + 1) * S] for b in range(B)])
            g, l = pools(hv, torch.stack([loc[b, lastvis[b]] for b in range(B)])); rec["ctxvis_g"].append(g); rec["ctxvis_l"].append(l)
            rec["video_id"].append(np.array([ds.records[i].video_id for i in ch]))
            if rank == 0 and (b0 // bs) % 10 == 0:
                print(f"[r0] {name} {b0 + B}/{len(mine)} ({time.time() - t0:.0f}s)", flush=True)
    d = OUT / name; d.mkdir(parents=True, exist_ok=True)
    np.savez(d / f"shard{rank}.npz", **{k: np.concatenate(v) for k, v in rec.items()})
    print(f"[r{rank}] {name} done ({time.time() - t0:.0f}s)", flush=True)


def ridge_cv_acc(F, y, groups, lam=100.0, folds=5):
    """block 단위 fold · one-hot ridge → 정확도."""
    cls = np.unique(y); Y = (y[:, None] == cls[None]).astype(np.float64)
    ug = np.unique(groups); rng = np.random.default_rng(0); rng.shuffle(ug); parts = np.array_split(ug, folds)
    hit = np.zeros(len(y), bool)
    for pt in parts:
        te = np.isin(groups, pt); tr = ~te
        mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-6
        A = (F[tr] - mu) / sd; Bm = (F[te] - mu) / sd
        Wt = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ (Y[tr] - Y[tr].mean(0)))
        hit[te] = (Bm @ Wt).argmax(1) == Y[te].argmax(1)
    return float(hit.mean())


def analyze(names):
    W = wanted(); res = {}
    for name in names:
        Z = defaultdict(list)
        for f in sorted((OUT / name).glob("shard*.npz")):
            z = np.load(f, allow_pickle=True)
            for k in z.files: Z[k].append(z[k])
        if not Z: print("없음", name); continue
        Z = {k: np.concatenate(v) for k, v in Z.items()}
        rows = [W[str(v)] for v in Z["video_id"]]
        shape = np.array([obj_of(r).split("/")[0] for r in rows]); color = np.array([obj_of(r).split("/")[-1] for r in rows])
        objset = np.array([r["objset"] for r in rows]); occ = np.array([r["occlusion"] for r in rows])
        mov = np.array(["static" if r["motion"] == "flat_static" else "moving" for r in rows]); blk = np.array([r["block_id"] for r in rows])
        out = {}
        for o in ("primitive", "realistic"):
            for oc in ("visible", "early", "mid", "late"):
                m = (objset == o) & (occ == oc) & (mov == "moving")
                if m.sum() < 30: continue
                cell = {"n": int(m.sum()), "chance_shape": float(1 / len(np.unique(shape[m])))}
                for feat in ("st_g", "st_l", "r_g", "r_l", "h_g", "h_l"):
                    cell[feat] = [ridge_cv_acc(Z[f"{feat}{k}"][m].astype(np.float64), shape[m], blk[m]) for k in range(1, K + 1)]
                cell["ctxvis_g"] = ridge_cv_acc(Z["ctxvis_g"][m].astype(np.float64), shape[m], blk[m])
                if o == "primitive":
                    cell["color_st_g"] = [ridge_cv_acc(Z[f"st_g{k}"][m].astype(np.float64), color[m], blk[m]) for k in (1, 4, 8)]
                    cell["color_h_g"] = [ridge_cv_acc(Z[f"h_g{k}"][m].astype(np.float64), color[m], blk[m]) for k in (1, 4, 8)]
                out[f"{o}/{oc}"] = cell
        res[name] = out
        print(f"\n===== {name} — 이동 물체, 모양 정확도 (block 5-fold ridge), 걸음 k1..k8")
        for key, cell in out.items():
            print(f"-- {key} (n {cell['n']}, chance {cell['chance_shape']:.2f}, 문맥 마지막 보이는 h {cell['ctxvis_g']:.2f})")
            for feat in ("st_g", "st_l", "r_g", "r_l", "h_g", "h_l"):
                print(f"   {feat:5s} " + " ".join(f"{v:5.2f}" for v in cell[feat]))
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / "late_identity.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["extract", "analyze"])
    ap.add_argument("--name", default="ar40"); ap.add_argument("--ckpt", default="")
    ap.add_argument("--names", nargs="+", default=["release", "ar40", "ar_state40"])
    ap.add_argument("--gpus", type=int, default=4); ap.add_argument("--bs", type=int, default=8)
    a = ap.parse_args()
    if a.phase == "extract":
        import torch
        torch.multiprocessing.spawn(_worker, args=(a.gpus, a.name, a.ckpt, a.bs), nprocs=a.gpus, join=True)
    else:
        analyze(a.names)
