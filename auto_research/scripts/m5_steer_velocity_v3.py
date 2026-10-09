#!/usr/bin/env python3
"""M5 — 개입: predictor 는 문맥 encoder 토큰의 **속도 부호** 를 쓰는가 (v3, C16/P16). "encoder is enough" 의 3 단 (읽힌다 → 충분하다 → 쓰인다).

M3: 속도는 z_last 에서 R² 0.985 로 읽히고 p 토큰에도 남는다. 그런데 p 는 물체를 멀리 옮기지 않는다.
질문: z 의 속도 방향을 인위로 바꾸면 p 가 둔 물체 자리가 따라 움직이는가 — 움직인다면 몇 걸음까지.

적합 (적합 궤적, 평가와 궤적이 겹치지 않음): raw z (predictor 입력, LN 전) 의 문맥 마지막 튜블릿 물체 3×3 평균 → 문맥 끝 속도 (vx, vy px/프레임) 능형.
  raw 공간 방향 u_x = w_x / ‖w_x‖²  (특징 표준화를 되돌린 가중치) → z_obj 에 s·u_x 를 더하면 능형 읽기가 정확히 s px/프레임 바뀐다.
개입 (평가 clip): 문맥 마지막 **2 튜블릿** 의 물체 3×3 토큰 (18 개) 에 같은 Δ 를 더한다.
  팔: base, vx ± {1.5, 3} px/프레임, vy ± 3, 무작위 방향 (‖Δ‖ = vx+3 과 같음, 2 개).
읽기: p 슬롯마다 외형 템플릿 (문맥 마지막 LN(h) 물체 토큰) argmax (y, x) · maxcos · median.
  이득 gain_t = (Δx_t 칸) / (s × 경과 프레임 / 18);  경과 프레임 = (32 + 2t + 0.5) − 30.5.  gain 1 = 속도 바뀐 만큼 정확히 옮김, 0 = 안 씀.

  python m5_steer_velocity_v3.py --gpus 8         /  --gpus 1 --limit 4 --fit-clips 24 --out /tmp/m5_smoke
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
from h4_single_query_v3 import idx_range, locate, cellxy  # noqa: E402
from h23_extract_v3 import arr, win_idx  # noqa: E402
from p3_closure_v3 import select  # noqa: E402

G, S_TOK, D, SPLIT, C, K = 16, 256, 1280, 32, 16, 8
TC = C // 2
ARMS = [("base", None, 0.0), ("vx+1.5", "x", 1.5), ("vx-1.5", "x", -1.5), ("vx+3", "x", 3.0), ("vx-3", "x", -3.0),
        ("vy+3", "y", 3.0), ("vy-3", "y", -3.0), ("rand1", "r1", 3.0), ("rand2", "r2", 3.0)]
OUT_DEFAULT = "/data2/local_datasets/world/world_analysis/cache/auto_research/v3_m5"


def vel(raw):
    x, y = arr(raw["px_x_by_sample"]), arr(raw["px_y_by_sample"])
    return np.array([(x[31] - x[27]) / 4, (y[31] - y[27]) / 4])


def zobj(bundle, x, raw, dev):
    ci = idx_range(0, TC, 1, dev)
    z = bundle.context_encoder(x, masks=[ci]).float().reshape(TC, S_TOK, D)
    lc = cellxy(raw, SPLIT - 2)
    return z, lc, z[TC - 1][win_idx(*lc)].mean(0)


def fit_dirs(a, fit_idx, dev):
    ds64 = arlib.WMADataset(dict(arlib.v3_cfg(0, 64, split=0))); R = ds64.records
    bundle = arlib.build_bundle(arlib.v3_cfg(4, 60), dev, window=64); ac = arlib.autocast_ctx(arlib.v3_cfg(16, 16))
    rng = np.random.default_rng(0); pick = rng.choice(fit_idx, min(a.fit_clips, len(fit_idx)), replace=False)
    Xf, Yf = [], []
    for i in pick:
        x = ds64.clip(int(i))[None, :, SPLIT - C:SPLIT + 2 * K].to(dev, bundle.dtype)
        with torch.inference_mode(), ac():
            _, _, zo = zobj(bundle, x, R[int(i)].raw, dev)
        Xf.append(zo.cpu().numpy().astype(np.float64)); Yf.append(vel(R[int(i)].raw))
    X, Y = np.stack(Xf), np.stack(Yf); mu, sd = X.mean(0), X.std(0) + 1e-6; Xs = (X - mu) / sd
    lam = 10.0 * len(X)
    W = np.linalg.solve(Xs.T @ Xs + lam * np.eye(D), Xs.T @ (Y - Y.mean(0)))           # (D, 2) 표준화 공간
    Wr = W / sd[:, None]                                                                  # raw 공간 가중치
    pred = Xs @ W + Y.mean(0); r2 = 1 - ((pred - Y) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)
    u = {k: Wr[:, j] / (Wr[:, j] ** 2).sum() for k, j in (("x", 0), ("y", 1))}          # s·u → 읽기가 s 만큼 변함
    g = np.random.default_rng(1)
    for k in ("r1", "r2"):
        r = g.standard_normal(D); r = r / np.linalg.norm(r) * np.linalg.norm(u["x"]); u[k] = r
    np.savez(Path(a.out) / "dirs.npz", **{k: v.astype(np.float32) for k, v in u.items()}, mu=mu, sd=sd, W=W, ymean=Y.mean(0))
    json.dump(dict(fit_clips=len(pick), train_r2=[float(v) for v in r2], norm_ux=float(np.linalg.norm(u["x"])), norm_uy=float(np.linalg.norm(u["y"])),
                   mean_token_norm=float(np.linalg.norm(X, axis=1).mean())), open(Path(a.out) / "dirs.json", "w"))
    print("방향 적합: clip", len(pick), "train R²", np.round(r2, 3), "‖u_x‖", round(float(np.linalg.norm(u['x'])), 3), "토큰 평균 ‖z‖", round(float(np.linalg.norm(X, axis=1).mean()), 2), flush=True)
    del bundle; torch.cuda.empty_cache()


def _worker(rank, world, a, ev):
    dev = torch.device(f"cuda:{rank}"); torch.cuda.set_device(dev)
    ds64 = arlib.WMADataset(dict(arlib.v3_cfg(0, 64, split=0))); R = ds64.records
    bundle = arlib.build_bundle(arlib.v3_cfg(4, 60), dev, window=64); pred = bundle.predictor; ac = arlib.autocast_ctx(arlib.v3_cfg(16, 16))
    dz = np.load(Path(a.out) / "dirs.npz"); U = {k: torch.tensor(dz[k], device=dev) for k in ("x", "y", "r1", "r2")}
    M = np.lib.format.open_memmap(Path(a.out) / "steer.npy", mode="r+")                 # (n, arms, K, 4)
    mine = list(range(rank, len(ev), world)); t0 = time.time()
    ci = idx_range(0, TC, 1, dev); ti = idx_range(TC, K, 1, dev)
    for n_done, row in enumerate(mine):
        i = ev[row]; raw = R[i].raw
        x = ds64.clip(i)[None, :, SPLIT - C:SPLIT + 2 * K].to(dev, bundle.dtype)
        with torch.inference_mode(), ac():
            h = F.layer_norm(bundle.target_encoder(x), (D,)).float().reshape(TC + K, S_TOK, D)
            z, lc, _ = zobj(bundle, x, raw, dev)
            lc2 = cellxy(raw, SPLIT - 4)
            idx = [(TC - 1) * S_TOK + j for j in win_idx(*lc)] + [(TC - 2) * S_TOK + j for j in win_idx(*lc2)]
            app = h[TC - 1, lc[1] * G + lc[0]][None]
            zf = z.reshape(1, TC * S_TOK, D)
            for ai, (nm, key, s) in enumerate(ARMS):
                zz = zf.clone()
                if key is not None: zz[0, idx] += s * U[key]
                p = pred(zz.to(bundle.dtype), ci, ti, mask_index=0).float().reshape(K, S_TOK, D)
                M[row, ai] = locate(p, app.expand(K, -1)).cpu().numpy()
        if rank == 0 and n_done % 25 == 0:
            el = time.time() - t0; print(f"[rank0] {n_done+1}/{len(mine)} {el/60:.1f} 분 남은 {el/(n_done+1)*(len(mine)-n_done-1)/60:.1f} 분", flush=True)
    M.flush(); print(f"[rank{rank}] 끝", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DEFAULT); ap.add_argument("--gpus", type=int, default=torch.cuda.device_count())
    ap.add_argument("--every", type=int, default=2); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--fit-clips", type=int, default=240)
    a = ap.parse_args(); Path(a.out).mkdir(parents=True, exist_ok=True)
    ds = arlib.WMADataset(dict(arlib.v3_cfg(0, 64, split=0))); R = ds.records
    fit, ev = select(R, a.every)
    if a.limit: ev = ev[:: max(1, len(ev) // a.limit)][: a.limit]
    n = len(ev)
    m = np.lib.format.open_memmap(Path(a.out) / "steer.npy", mode="w+", dtype=np.float32, shape=(n, len(ARMS), K, 4)); m[:] = np.nan; del m
    json.dump(dict(n=n, arms=[x[0] for x in ARMS], steps=[x[2] for x in ARMS], keys=[x[1] for x in ARMS], C=C, K=K,
                   video_ids=[R[i].video_id for i in ev], scenario=[R[i].raw["scenario"] for i in ev],
                   traj=[[R[i].raw["scenario"], R[i].raw["primary"], R[i].raw["secondary"]] for i in ev]), open(Path(a.out) / "meta.json", "w"))
    print("eval clips", n, "fit pool", len(fit), flush=True)
    fit_dirs(a, fit, torch.device("cuda:0"))
    mp.spawn(_worker, args=(a.gpus, a, ev), nprocs=a.gpus, join=True); print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
