#!/usr/bin/env python3
"""물체 위치는 블록 하나의 (N, D) 토큰에 어떻게 들어 있나 (2026-10-13, 사용자 질문).

(1) 위치 디코더는 어디를 보나 — 각 predictor (release · Ariel AR · Ariel AR-state) 의 미래 출력 p 에 그 predictor 의 위치 + 정체 자
    (identity_vobprim_<name>/p, attentive pooling 쿼리 1 개) 를 걸고, attention 지도 (256) 에서
    · 최대 attention 토큰이 참 물체 칸에서 몇 칸 떨어져 있나 · 물체 3×3 에 실린 attention 질량 (균등이면 9/256 = 3.5 %) · 최대 attention 값.
(2) 위치는 어느 토큰에 쓰여 있나 — target encoder 토큰 (블록별 = Ariel 공간 / 32 장 창 단위 = release 표준 표적) 에서
    물체로부터 체비셰프 거리 d 인 토큰 **하나** (무작위) 만 가지고 물체 위치 (x, y) 를 ridge 로 회귀 (clip 단위 train / test).
    d 가 커져도 위치가 읽히면 위치가 화면 전체 토큰에 퍼져 (broadcast) 있다는 뜻. 대조: 토큰 평균 (전역) · 학습 평균 (기준선).
판: VOB_readout_primitive test (등속 직선, 가림 없음), 이동 clip, bottleneck split.

  python position_representation.py --gpus 4 --n 600
산출: z_research/VOBench_v1/bottleneck/position_rep/{attn_<name>.npz, tokens_<space>.npz, result.json}
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "z_research/scripts/analysis"))
import predictor_bottleneck as PB  # noqa: E402

S, NB, C, K, D, G, CELL = 256, 16, 8, 8, 1280, 16, 18.0
DIST = (0, 1, 2, 3, 5, 7)
OUT = REPO / "z_research/VOBench_v1/bottleneck/position_rep"
MODELS = {"release": "", "ar40": "z_training/ariel/ar_future_1e_5/latest.pt", "ar_state40": "z_training/ariel/ar_state_future_1e_5/latest.pt"}


def items(n):
    its, rows = PB.select(1600, 1200)
    tr = [i for i, p in its if p == "train"][:n]; te = [i for i, p in its if p == "test"][:n]
    return tr, te, rows


def _worker(rank, world, n, bs):
    import torch
    from importlib.util import module_from_spec, spec_from_file_location
    from analysis.intphys2.model import build_from_config
    from app.vjepa_frozen.ar import block_indices, encode_blocks
    from evals.world_model_analysis.data import WMADataset
    torch.cuda.set_device(rank); torch.set_num_threads(4); dev = torch.device("cuda", rank)
    spec = spec_from_file_location("ir", REPO / "z_research/scripts/analysis/rollout2_identity_readout.py")
    ir = module_from_spec(spec); spec.loader.exec_module(ir); rp = ir.rp
    tr, te, rows = items(n); allc = [(i, "train") for i in tr] + [(i, "test") for i in te]
    mine = allc[rank::world]
    rng = np.random.default_rng(rank)
    rec = {}

    def put(k, v): rec.setdefault(k, []).append(v)

    for mi, (name, ck) in enumerate(MODELS.items()):
        model = dict(rp.MODEL) if not ck else dict(rp.MODEL, predictor_checkpoint=str(REPO / ck))
        bundle = build_from_config(model, dev); pred = bundle.predictor
        R = ir.IdReadout().to(dev).eval()
        R.load_state_dict(torch.load(REPO / f"z_research/VOBench_v1/readout/identity_vobprim_{name}/p/readout.pt", map_location=dev))
        ds = WMADataset(dict(data=rp.DATA, model=model, features={"cache_dir": "/tmp"}, surprise={}))
        t0 = time.time()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for b0 in range(0, len(mine), bs):
                ch = mine[b0:b0 + bs]
                x = torch.stack([ds.clip(i) for i, _ in ch]).to(dev, dtype=bundle.dtype); B = x.size(0)
                if name == "release":
                    from analysis.intphys2.surprise import _context_target_indices
                    ci, ti = _context_target_indices(ctx_frames=2 * C, tgt_frames=2 * K, tubelet_size=2, spatial_tokens=S, batch_size=B, device=dev)
                    z = bundle.context_encoder(x, masks=[ci]); z = z[-1] if isinstance(z, list) else z
                    p = pred(z, ci, ti, mask_index=0)
                else:
                    hb = encode_blocks(bundle.target_encoder, x, 2)
                    p = pred.rollout(hb[:, :C * S], block_indices(B, C, S, dev), K)
                maps = []
                for k in range(K):
                    _, _, a = R(p[:, k * S:(k + 1) * S].half()); maps.append(a.float().cpu().numpy())
                put(f"attn_{name}", np.stack(maps, 1).astype(np.float16))                 # (B, K, 256)
                if mi == 0:                                                                  # 토큰 표본 (모델 무관): target encoder 두 공간
                    put("row", np.array([i for i, _ in ch])); put("part", np.array([pp for _, pp in ch]))
                    hb = encode_blocks(bundle.target_encoder, x, 2).float()
                    hw = bundle.target_encoder(x); hw = hw[-1] if isinstance(hw, list) else hw
                    hw = torch.nn.functional.layer_norm(hw.float(), (D,))
                    T = np.stack([PB.truth(rows, i)[0] for i, _ in ch])                     # (B, 16, 2) 288 px
                    for space, H in (("block", hb), ("window", hw)):
                        Hg = H.view(B, NB, G, G, D).half().cpu().numpy()
                        sel = np.zeros((B, NB, len(DIST) + 1, D), np.float16)
                        for b in range(B):
                            for t in range(NB):
                                cy = int(np.clip(T[b, t, 1] // CELL, 0, G - 1)); cx = int(np.clip(T[b, t, 0] // CELL, 0, G - 1))
                                yy, xx = np.mgrid[0:G, 0:G]; dd = np.maximum(abs(yy - cy), abs(xx - cx))
                                for di, d in enumerate(DIST):
                                    cand = np.argwhere(dd == d)
                                    if len(cand) == 0: continue
                                    y, x_ = cand[rng.integers(len(cand))]
                                    sel[b, t, di] = Hg[b, t, y, x_]
                                sel[b, t, -1] = Hg[b, t].astype(np.float32).mean((0, 1))
                        put(f"tok_{space}", sel)
                if rank == 0 and (b0 // bs) % 10 == 0: print(f"[r0] {name} {b0 + B}/{len(mine)} ({time.time() - t0:.0f}s)", flush=True)
        del bundle, pred; torch.cuda.empty_cache()
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"_shard_r{rank}.npz", **{k: np.concatenate(v) for k, v in rec.items()})


def ridge(Xtr, Ytr, Xte, lam):
    mu = Xtr.mean(0); Xc = Xtr - mu; ym = Ytr.mean(0)
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    W = Vt.T @ np.diag(s / (s ** 2 + lam)) @ U.T @ (Ytr - ym)
    return (Xte - mu) @ W + ym


def analyze():
    Z = {}
    for f in sorted(OUT.glob("_shard_r*.npz")):
        z = np.load(f)
        for k in z.files: Z.setdefault(k, []).append(z[k])
    Z = {k: np.concatenate(v) for k, v in Z.items()}
    rows = PB.select(1600, 1200)[1]
    T = np.stack([PB.truth(rows, int(i))[0] for i in Z["row"]])                           # (n, 16, 2)
    tr = Z["part"] == "train"; te = ~tr
    res = {"n_train": int(tr.sum()), "n_test": int(te.sum())}
    # (1) 디코더 attention (미래 블록, test)
    cy = np.clip(T[:, C:, 1] // CELL, 0, G - 1).astype(int); cx = np.clip(T[:, C:, 0] // CELL, 0, G - 1).astype(int)
    yy, xx = np.mgrid[0:G, 0:G]
    print("(1) 위치 디코더 attention — predictor 미래 출력 (test, k1–k8 평균; [k1 / k8])")
    for name in MODELS:
        A = Z[f"attn_{name}"].astype(np.float32)[te]                                         # (n, K, 256)
        am = A.argmax(-1); ay, ax = am // G, am % G
        dist = np.maximum(abs(ay - cy[te]), abs(ax - cx[te]))
        dmap = np.maximum(abs(yy[None, None] - cy[te][..., None, None]), abs(xx[None, None] - cx[te][..., None, None])).reshape(*A.shape)
        mass = (A * (dmap <= 1)).sum(-1); peak = A.max(-1)
        res[f"attn_{name}"] = {"argmax_dist_med": np.median(dist, 0).tolist(), "argmax_on_obj3x3": (dist <= 1).mean(0).tolist(),
                               "mass_obj3x3": mass.mean(0).tolist(), "peak": peak.mean(0).tolist()}
        r = res[f"attn_{name}"]
        print(f"  {name:10s} 최대 attention 토큰이 물체 3×3 안: {np.mean(r['argmax_on_obj3x3']):.2f} [{r['argmax_on_obj3x3'][0]:.2f} / {r['argmax_on_obj3x3'][-1]:.2f}]"
              f" · 물체 3×3 질량 {np.mean(r['mass_obj3x3']):.2f} [{r['mass_obj3x3'][0]:.2f} / {r['mass_obj3x3'][-1]:.2f}] (균등 0.035) · 최대값 {np.mean(r['peak']):.2f}")
    # (2) 토큰 하나로 위치 회귀 (target encoder, 모든 16 블록)
    print("\n(2) 물체로부터 거리 d 인 토큰 하나 → 물체 위치 ridge (test 오차, 칸; 기준선 = 학습 평균)")
    Y = T.reshape(-1, 2) / CELL
    for space in ("block", "window"):
        X = Z[f"tok_{space}"].astype(np.float32)                                              # (n, 16, nd+1, D)
        n = len(X); trb = np.repeat(tr, NB); teb = np.repeat(te, NB)
        out = {}
        base = float(np.median(np.linalg.norm(Y[teb] - Y[trb].mean(0), axis=1)))
        for di, d in enumerate(list(DIST) + ["mean"]):
            Xd = X[:, :, di].reshape(n * NB, D)
            ok = np.abs(Xd).sum(1) > 0
            best = None
            for lam in (1e1, 1e2, 1e3, 1e4):
                # 검증: train 안에서 clip 절반
                trc = np.nonzero(tr)[0]; half = np.zeros(n, bool); half[trc[: len(trc) // 2]] = True
                a_, b_ = np.repeat(half, NB) & ok, np.repeat(tr & ~half, NB) & ok
                e = np.median(np.linalg.norm(ridge(Xd[a_], Y[a_], Xd[b_], lam) - Y[b_], axis=1))
                if best is None or e < best[0]: best = (e, lam)
            m_tr, m_te = trb & ok, teb & ok
            pr = ridge(Xd[m_tr], Y[m_tr], Xd[m_te], best[1])
            out[str(d)] = float(np.median(np.linalg.norm(pr - Y[m_te], axis=1)))
        res[f"tokpos_{space}"] = {"err_cells": out, "baseline": base}
        print(f"  {space:6s} " + "  ".join(f"d{d}: {v:.2f}" for d, v in out.items()) + f"   (학습 평균 기준선 {base:.2f})")
    json.dump(res, open(OUT / "result.json", "w"), indent=1)
    print("saved", OUT / "result.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--gpus", type=int, default=4); ap.add_argument("--n", type=int, default=600); ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    if not a.analyze:
        import torch
        torch.multiprocessing.spawn(_worker, args=(a.gpus, a.n, a.bs), nprocs=a.gpus, join=True)
    analyze()
