#!/usr/bin/env python3
"""P3 — 닫힘 검정 (RollOut_v3, C16): 자기 예측을 되먹이면 멈추는 것 (H5 ar_p) 은 **p 에 상태가 없어서** 인가, **인터페이스** 때문인가.

H5 는 채널별 어댑터 A (LN(z)→z, clip 자기 문맥에서 적합) 로 p 를 문맥에 되먹였고 두 걸음 뒤 멈췄다.
반론 (리뷰 P3): p 는 표적 공간 LN(h) 에 있고 predictor 입력은 문맥 encoder 공간 z 다 — 사상이 나빠서 멈췄을 수 있다.
그리고 H5 ar_h 는 양방향 표적 (뒤 프레임이 번진다) 을 되먹였다.

사상 W: LN(h) → z, 전 채널 선형 (1280×1281, 능형 회귀). **적합용 궤적** (법칙마다 궤적 셋 중 하나; 평가 clip 과 궤적이 겹치지 않는다)
        의 문맥 창에서 같은 프레임의 (인과 표적 LN(h), 문맥 encoder z) 토큰 쌍으로 적합한다.
팔 (한 튜블릿씩 되먹임, j = 0..7):
  ar_z     참 관측 z*_j (문맥 encoder 를 j 까지 늘려 새 튜블릿)             — 상한 (H5 와 같음)
  ar_pA    A(p_j)   (H5 ar_p 재현)
  ar_pW    W(p_j)   더 좋은 사상으로 자기 예측
  ar_hcW   W(hc_j)  인과 표적: target encoder 를 [문맥 시작, 튜블릿 j 끝) 에만 돌린 마지막 튜블릿 (미래가 안 번진다)
  ar_hcA   A(hc_j)
판정: ar_hcW 가 새 칸으로 나아가고 ar_pW 가 멈추면 → p 의 내용 (상태 없음) 이 원인, 인터페이스가 아니다.
      ar_hcW 도 멈추면 → 표적 공간 입력 자체를 predictor 가 못 쓴다 (인터페이스 교란) → H5 ar_p 멈춤은 "상태 없음" 증거가 못 된다.
사상 품질: 평가 clip 에서 rel L1 = mean|W(hc_j) − z*_j| / mean|z*_j| 과 A 의 같은 값 (저장).
지표: loc_tru (템플릿 = 표준 h_j 진실 칸), loc_app (문맥 마지막 외형), l1 = mean|pred_j − h_j| (표준 h).

  python p3_closure_v3.py --gpus 8           (적합은 rank 0 이 먼저, 그다음 8 rank 평가)
  python p3_closure_v3.py --gpus 1 --limit 4 --fit-clips 8 --out /tmp/p3_smoke
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
from h4_single_query_v3 import cellxy, idx_range, locate  # noqa: E402
from h5_autoregress_v3 import fit_adapter  # noqa: E402

G, S_TOK, D, SPLIT, J, C = 16, 256, 1280, 32, 8, 16
TC = C // 2
ARMS = ("ar_z", "ar_pA", "ar_pW", "ar_hcW", "ar_hcA")
OUT_DEFAULT = "/data2/local_datasets/world/world_analysis/cache/auto_research/v3_p3"
FREE = ("flat_v", "flat_a", "flat_d", "ramp_a", "ramp_d", "arc")


def select(R, every):
    traj = lambda r: (r.raw["scenario"], r.raw["primary"], r.raw["secondary"])
    free = [i for i, r in enumerate(R) if r.plausible == "1" and r.raw["scenario"] in FREE]
    trajs = sorted({traj(R[i]) for i in free})
    fit_t = {t for k, t in enumerate(trajs) if k % 3 == 0}
    fit = [i for i in free if traj(R[i]) in fit_t]; ev = [i for i in free if traj(R[i]) not in fit_t][::every]
    return fit, ev


def fit_W(a, fit_idx, dev):
    ds64 = arlib.WMADataset(dict(arlib.v3_cfg(0, 64, split=0)))
    bundle = arlib.build_bundle(arlib.v3_cfg(4, 60), dev, window=64); ac = arlib.autocast_ctx(arlib.v3_cfg(16, 16))
    rng = np.random.default_rng(0); pick = rng.choice(fit_idx, min(a.fit_clips, len(fit_idx)), replace=False)
    XtX = torch.zeros(D + 1, D + 1, dtype=torch.float64, device=dev); XtY = torch.zeros(D + 1, D, dtype=torch.float64, device=dev)
    val = []
    for k, i in enumerate(pick):
        x = ds64.clip(int(i))[None, :, SPLIT - C:SPLIT + 2 * J].to(dev, bundle.dtype)
        with torch.inference_mode(), ac():
            ci = idx_range(0, TC + J, 1, dev)
            z = bundle.context_encoder(x, masks=[ci]).float()[0]                       # (N, D) 문맥+미래 전 구간 z (인과 아님; 사상 적합용)
            hc = F.layer_norm(bundle.target_encoder(x), (D,)).float()[0]
        X = torch.cat([hc, torch.ones(len(hc), 1, device=dev)], 1).double(); Y = z.double()
        if k < len(pick) - 8:
            XtX += X.T @ X; XtY += X.T @ Y
        else:
            val.append((X, Y))
    lam = 1e-3 * XtX.diagonal().mean()
    W = torch.linalg.solve(XtX + lam * torch.eye(D + 1, device=dev, dtype=torch.float64), XtY)
    rel = [float(((X @ W - Y).abs().mean() / Y.abs().mean()).item()) for X, Y in val]
    torch.save(W.float().cpu(), Path(a.out) / "W.pt")
    json.dump(dict(fit_clips=int(len(pick) - 8), val_clips=len(val), val_rel_l1=float(np.mean(rel)), lam=float(lam)), open(Path(a.out) / "W.json", "w"))
    print(f"W 적합: clip {len(pick)-8}, 검증 rel L1 {np.mean(rel):.4f}", flush=True)
    del bundle; torch.cuda.empty_cache()


def _worker(rank, world, a, ev):
    dev = torch.device(f"cuda:{rank}"); torch.cuda.set_device(dev)
    ds64 = arlib.WMADataset(dict(arlib.v3_cfg(0, 64, split=0))); recs = ds64.records
    bundle = arlib.build_bundle(arlib.v3_cfg(4, 60), dev, window=64); pred, enc, tgt = bundle.predictor, bundle.context_encoder, bundle.target_encoder
    ac = arlib.autocast_ctx(arlib.v3_cfg(16, 16))
    W = torch.load(Path(a.out) / "W.pt").to(dev)
    mapW = lambda y: torch.cat([y.float(), torch.ones(*y.shape[:-1], 1, device=dev)], -1) @ W
    mm = np.lib.format.open_memmap(Path(a.out) / "p3.npy", mode="r+")          # (n, J, arms, 9)
    q = np.lib.format.open_memmap(Path(a.out) / "mapq.npy", mode="r+")         # (n, J, 2) rel L1 of W(hc), A(hc) vs z*
    mine = list(range(rank, len(ev), world)); t0 = time.time()
    for n_done, row in enumerate(mine):
        i = ev[row]
        x = ds64.clip(i)[None, :, SPLIT - C:SPLIT + 2 * J].to(dev, bundle.dtype)
        with torch.inference_mode(), ac():
            h = F.layer_norm(tgt(x), (D,)).float().reshape(TC + J, S_TOK, D)
            ci = idx_range(0, TC, 1, dev); z = enc(x, masks=[ci])
            Aa, Ab = fit_adapter(z.float()); mapA = lambda y: Aa * F.layer_norm(y.float(), (D,)) + Ab
            zstar, hc = [], []
            for j in range(J):
                zstar.append(enc(x, masks=[idx_range(0, TC + j + 1, 1, dev)])[:, (TC + j) * S_TOK:])
                hc.append(F.layer_norm(tgt(x[:, :, :C + 2 * (j + 1)]), (D,))[:, (TC + j) * S_TOK:])
            for j in range(J):
                zs = zstar[j].float()
                q[row, j] = [((mapW(hc[j]) - zs).abs().mean() / zs.abs().mean()).item(), ((mapA(hc[j]) - zs).abs().mean() / zs.abs().mean()).item()]
            preds = {}
            for arm in ARMS:
                Z = z; out = []
                for j in range(J):
                    p = pred(Z, idx_range(0, TC + j, 1, dev), idx_range(TC + j, 1, 1, dev), mask_index=0)
                    out.append(p.float())
                    new = {"ar_z": lambda: zstar[j], "ar_pA": lambda: mapA(p), "ar_pW": lambda: mapW(p),
                           "ar_hcW": lambda: mapW(hc[j]), "ar_hcA": lambda: mapA(hc[j])}[arm]()
                    Z = torch.cat([Z, new.to(Z.dtype)], 1)
                preds[arm] = out
        lc = cellxy(recs[i].raw, SPLIT - 2); app = h[TC - 1, lc[1] * G + lc[0]][None]
        for j in range(J):
            tc = cellxy(recs[i].raw, SPLIT + 2 * j); tt = h[TC + j, tc[1] * G + tc[0]][None]
            for m, arm in enumerate(ARMS):
                P = preds[arm][j]
                mm[row, j, m, 0:4] = locate(P, tt).cpu().numpy()[0]
                mm[row, j, m, 4:8] = locate(P, app).cpu().numpy()[0]
                mm[row, j, m, 8] = (P[0] - h[TC + j]).abs().mean().item()
        if rank == 0 and n_done % 25 == 0:
            el = time.time() - t0; print(f"[rank0] {n_done+1}/{len(mine)} {el/60:.1f} 분 남은 {el/(n_done+1)*(len(mine)-n_done-1)/60:.1f} 분", flush=True)
    mm.flush(); q.flush(); print(f"[rank{rank}] 끝", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DEFAULT); ap.add_argument("--gpus", type=int, default=torch.cuda.device_count())
    ap.add_argument("--every", type=int, default=2); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--fit-clips", type=int, default=160)
    a = ap.parse_args(); Path(a.out).mkdir(parents=True, exist_ok=True)
    ds = arlib.WMADataset(dict(arlib.v3_cfg(0, 64, split=0))); R = ds.records
    fit, ev = select(R, a.every)
    if a.limit: ev = ev[:: max(1, len(ev) // a.limit)][: a.limit]
    n = len(ev)
    m = np.lib.format.open_memmap(Path(a.out) / "p3.npy", mode="w+", dtype=np.float32, shape=(n, J, len(ARMS), 9)); m[:] = np.nan; del m
    m = np.lib.format.open_memmap(Path(a.out) / "mapq.npy", mode="w+", dtype=np.float32, shape=(n, J, 2)); m[:] = np.nan; del m
    json.dump(dict(n=n, arms=ARMS, J=J, C=C, video_ids=[R[i].video_id for i in ev], scenario=[R[i].raw["scenario"] for i in ev],
                   traj=[[R[i].raw["scenario"], R[i].raw["primary"], R[i].raw["secondary"]] for i in ev],
                   cols=["tru_y", "tru_x", "tru_max", "tru_med", "app_y", "app_x", "app_max", "app_med", "l1"]),
              open(Path(a.out) / "meta.json", "w"))
    print("eval clips", n, "fit pool", len(fit), flush=True)
    fit_W(a, fit, torch.device("cuda:0"))
    mp.spawn(_worker, args=(a.gpus, a, ev), nprocs=a.gpus, join=True); print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
