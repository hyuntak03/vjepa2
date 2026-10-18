#!/usr/bin/env python3
"""학습 목표는 역사를 쓰라고 요구하나 — 문맥을 마지막 1 · 2 블록으로 잘랐을 때 손실이 얼마나 오르나 (2026-10-13).

anchor 이식 (predictor_anchor_test) 에서: 미래 자리 · 속도는 마지막 문맥 블록의 내용을 따르고 앞 7 블록 (역사) 은 거의 안 쓰인다.
원인 후보 셋 — (A) encoder 에 역사 정보가 없다 (기각: AR 입력 블록 0–6 은 조건 사이 비트 동일, z 자가 late 미래를 5–10 px 로 읽음),
(B) 구조가 역사를 못 쓴다, (C) 목표 · 데이터가 역사를 요구하지 않는다. 이 스크립트는 (C) 를 잰다:
같은 모델에 문맥 8 블록 vs 마지막 2 블록 vs 마지막 1 블록 (절대 시간 인덱스 유지) 을 주고 미래 8 블록 L1 손실을 비교한다.
  역사 가치 = (L_ctx2 − L_ctx8) / (L_copy − L_ctx8)   — 복사 대비 이득 중 역사 (블록 0–5) 가 몫으로 내는 것.
판: SSv2 val (AR 둘의 학습 분포 안) 과 VOB_readout (등속, 전부 보임). 토큰 전체와 '변한 토큰' (복사 오차 상위 10 %) 따로.
release 는 문맥 encoder 에 마지막 블록 토큰만 넣는다 (masks 로 앞 블록을 뺀다).

  python predictor_history_value.py --name ar40 --ckpt z_training/ariel/ar_future_1e_5/latest.pt --set ssv2 --gpus 4 --n 800
산출: z_research/VOBench_v1/bottleneck/history/<set>_<name>.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "z_research/scripts/analysis"))
import predictor_bottleneck as PB  # noqa: E402  (VOB_readout 환경)

S, NB, C, K, D, NF = 256, 16, 8, 8, 1280, 32
OUT = REPO / "z_research/VOBench_v1/bottleneck/history"
VAL = REPO / "data_csv/ssv2/val_min48.csv"
CTX = (8, 4, 2, 1)
TOPQ = 0.10


def _worker(rank, world, name, ckpt, setname, n, bs):
    import decord
    import torch
    import torch.nn.functional as F
    from importlib.util import module_from_spec, spec_from_file_location
    from analysis.intphys2.dataset import build_video_transform
    from analysis.intphys2.model import build_from_config
    from app.vjepa_frozen.ar import encode_blocks
    from evals.world_model_analysis.data import WMADataset
    torch.cuda.set_device(rank); torch.set_num_threads(4); dev = torch.device("cuda", rank)
    spec = spec_from_file_location("ir", REPO / "z_research/scripts/analysis/rollout2_identity_readout.py")
    ir = module_from_spec(spec); spec.loader.exec_module(ir); rp = ir.rp
    model = dict(rp.MODEL) if name == "release" else dict(rp.MODEL, predictor_checkpoint=str(REPO / ckpt))
    bundle = build_from_config(model, dev); pred = bundle.predictor
    is_ar = getattr(pred, "mask_mode", "release") in ("ar", "ar_state")
    LN = lambda x: F.layer_norm(x.float(), (x.size(-1),))
    if setname == "ssv2":
        rows = [r for r in csv.reader(open(VAL), delimiter=" ")]
        rng = np.random.default_rng(0); rng.shuffle(rows); files = [r[0] for r in rows[:n]]
        tf_ = build_video_transform(256, 1.0)

        def load(f):
            try:
                vr = decord.VideoReader(f, num_threads=1)
                return tf_(vr.get_batch(list(range(NF))).asnumpy()) if len(vr) >= NF else None
            except Exception:
                return None
    elif setname.startswith("vob_"):
        import yaml
        import design_headroom as H
        cfg = yaml.safe_load(open(H.SETS["vobench"]["cfg"]))
        ds = WMADataset(dict(data=dict(cfg["data"], index_csv="index.csv"), model=model, features={"cache_dir": "/tmp"}, surprise={}))
        want = setname.split("_")[1]
        ix = {r["video_id"]: r for r in csv.DictReader(open(H.SETS["vobench"]["index"]))}
        files = [i for i, rec in enumerate(ds.records) if (r := ix.get(rec.video_id)) and r["occlusion"] == want
                 and r["motion"] != "flat_static" and r["variant"] in ("pos_a", "pos_b") and r["obj_a" if r["variant"] == "pos_a" else "obj_b"] != "empty"][:n]
        load = ds.clip
    else:
        ds = WMADataset(dict(data=rp.DATA, model=model, features={"cache_dir": "/tmp"}, surprise={}))
        items, _ = PB.select(1600, 1200); files = [i for i, p in items if p == "test"][:n]
        load = ds.clip
    mine = files[rank * len(files) // world:(rank + 1) * len(files) // world]
    acc = {}

    def put(k, v): acc.setdefault(k, []).append(v)

    t0 = time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for b0 in range(0, len(mine), bs):
            xs = [v for v in (load(f) for f in mine[b0:b0 + bs]) if v is not None]
            if not xs: continue
            x = torch.stack(xs).to(dev, dtype=bundle.dtype); B = x.size(0)
            preds = {}
            if is_ar:
                hb = encode_blocks(bundle.target_encoder, x, 2)
                tgt = hb[:, C * S:].float(); copy = hb[:, (C - 1) * S:C * S].float()
                for c in CTX:
                    idx = torch.arange((C - c) * S, C * S, device=dev).unsqueeze(0).expand(B, -1).contiguous()
                    preds[c] = pred.rollout(hb[:, (C - c) * S:C * S], idx, K).float()
            else:
                from analysis.intphys2.surprise import _context_target_indices
                ci, ti = _context_target_indices(ctx_frames=2 * C, tgt_frames=2 * K, tubelet_size=2, spatial_tokens=S, batch_size=B, device=dev)
                h = bundle.target_encoder(x); h = LN(h[-1] if isinstance(h, list) else h)
                tgt = h[:, C * S:]
                for c in CTX:
                    cs = ci[:, (C - c) * S:]
                    z = bundle.context_encoder(x, masks=[cs]); z = z[-1] if isinstance(z, list) else z
                    preds[c] = pred(z, cs, ti, mask_index=0).float()
                    if c == 8: copy = LN(z[:, (C - 1) * S:C * S])
            if is_ar:                                                                       # 모델 없는 데이터 통계: 미래 토큰의 최근접 내용이 최근 2 블록에 있나, 그 앞에만 있나
                hn = F.normalize(hb.float(), dim=-1)
                rec_ = hn[:, (C - 2) * S:C * S]; his_ = hn[:, :(C - 2) * S]
                for k in range(K):
                    q = hn[:, (C + k) * S:(C + k + 1) * S]
                    mr = torch.bmm(q, rec_.transpose(1, 2)).amax(-1).float(); mh = torch.bmm(q, his_.transpose(1, 2)).amax(-1).float()
                    put(f"needhist{k}", (mh - mr > 0.05).float().mean(1).cpu().numpy()); put(f"gaphist{k}", (mh - mr).clamp(min=0).mean(1).cpu().numpy())
            for k in range(K):
                hk = tgt[:, k * S:(k + 1) * S]
                ce = (copy - hk).abs().mean(-1)
                chg = ce >= torch.quantile(ce, 1 - TOPQ, dim=1, keepdim=True)
                put(f"copy_all{k}", ce.mean(1).cpu().numpy()); put(f"copy_chg{k}", ((ce * chg).sum(1) / chg.sum(1)).cpu().numpy())
                for c, p in preds.items():
                    e = (p[:, k * S:(k + 1) * S] - hk).abs().mean(-1)
                    put(f"c{c}_all{k}", e.mean(1).cpu().numpy()); put(f"c{c}_chg{k}", ((e * chg).sum(1) / chg.sum(1)).cpu().numpy())
            if rank == 0 and (b0 // bs) % 10 == 0: print(f"[r0] {name} {setname} {b0 + B}/{len(mine)} ({time.time() - t0:.0f}s)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"_{setname}_{name}_r{rank}.npz", **{k: np.concatenate(v) for k, v in acc.items()})


def summarize(name, setname):
    Z = {}
    for f in sorted(OUT.glob(f"_{setname}_{name}_r*.npz")):
        z = np.load(f)
        for k in z.files: Z.setdefault(k, []).append(z[k])
    Z = {k: np.concatenate(v) for k, v in Z.items()}
    res = {"n": int(len(Z["copy_all0"]))}
    for part in ("all", "chg"):
        cp = np.array([Z[f"copy_{part}{k}"].mean() for k in range(K)])
        L = {c: np.array([Z[f"c{c}_{part}{k}"].mean() for k in range(K)]) for c in CTX}
        res[part] = {"copy": cp.tolist(), **{f"ctx{c}": L[c].tolist() for c in CTX},
                     **{f"history_value_ctx{c}": ((L[c] - L[8]) / (cp - L[8])).tolist() for c in CTX if c != 8}}
    if "needhist0" in Z:
        res["need_history_frac"] = [float(Z[f"needhist{k}"].mean()) for k in range(K)]
        res["history_cos_gap"] = [float(Z[f"gaphist{k}"].mean()) for k in range(K)]
    json.dump(res, open(OUT / f"{setname}_{name}.json", "w"), indent=1)
    if "need_history_frac" in res:
        print("  [데이터] 미래 토큰 중 최근접 내용이 블록 0–5 에만 (cos 차 > .05) 있는 몫: " + " ".join(f"{v:.3f}" for v in res["need_history_frac"]))
    print(f"\n== {name} · {setname} ({res['n']} clip) — 미래 블록 k1..k8 L1")
    for part in ("all", "chg"):
        print(f"  [{'전체 토큰' if part == 'all' else '변한 토큰 (복사 오차 상위 10 %)'}]")
        print("    copy      " + " ".join(f"{v:.3f}" for v in res[part]["copy"]))
        for c in CTX: print(f"    ctx{c:<2d}     " + " ".join(f"{v:.3f}" for v in res[part][f"ctx{c}"]))
        for c in CTX[1:]: print(f"    역사가치 ctx{c} " + " ".join(f"{v:+.2f}" for v in res[part][f"history_value_ctx{c}"]))
    for f in OUT.glob(f"_{setname}_{name}_r*.npz"): f.unlink()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--ckpt", default="")
    ap.add_argument("--set", default="ssv2", choices=["ssv2", "vobro", "vob_late", "vob_visible"]); ap.add_argument("--gpus", type=int, default=4)
    ap.add_argument("--n", type=int, default=800); ap.add_argument("--bs", type=int, default=8)
    a = ap.parse_args()
    import torch
    torch.multiprocessing.spawn(_worker, args=(a.gpus, a.name, a.ckpt, a.set, a.n, a.bs), nprocs=a.gpus, join=True)
    summarize(a.name, a.set)
