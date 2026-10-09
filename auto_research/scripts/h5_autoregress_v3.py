#!/usr/bin/env python3
"""H5 — frozen predictor 를 자기회귀로 되먹이면 상태가 진화하나. RollOut_v3.

기제 R: 미래 토큰끼리 상태를 안 넘긴다 (mask→mask 차단 −1.4). 추론 때 되먹임을 강제해 본다.
문맥 C ∈ {8, 16} (분할 32), 미래 튜블릿 j = 0..7 을 한 튜블릿씩 질의한다. 팔:
  os      한 번에 질의 (P = 16, 표준)
  single  원래 문맥으로 튜블릿 j 하나만 질의 (H4 와 같음)
  ar_p    Z_{j+1} = [Z_j ; A(p_j)]                      자기 예측을 문맥으로 되먹임
  ar_z    Z_{j+1} = [Z_j ; z*_j]                        참 관측 (문맥 encoder 를 j 까지 늘린 문맥에 돌려 새 튜블릿 토큰) — 상한
  ar_zA   Z_{j+1} = [Z_j ; A(LN(z*_j))]                 참 관측을 어댑터에 통과 — ar_z 와의 차이 = 어댑터 손실
  ar_h    Z_{j+1} = [Z_j ; A(h_j)]                      target 쪽 참 미래 (LN(target_encoder)) 를 어댑터로
어댑터 A: clip 자신의 문맥 토큰에서 채널별 1 차 사상 LN(z)_c → z_c 를 최소제곱으로 (미래 정보 안 씀, 2×1280 스칼라).
지표 (j 마다, 표적 h = LN(target_encoder([32−C, 48)))): loc_tru / loc_app (H4 와 같음), l1 = mean|p_j − h_j|.
저장 (n, 2 C, 8 j, 6 arm, 9).

  sbatch 로, 또는 python ... --gpus 1 --limit 8
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
from h23_extract_v3 import ClipDS, collate  # noqa: E402
from h4_single_query_v3 import cellxy, idx_range, locate  # noqa: E402

G, S_TOK, D, SPLIT, J = 16, 256, 1280, 32, 8
CS = (8, 16)
ARMS = ("os", "single", "ar_p", "ar_z", "ar_zA", "ar_h")
OUT_DEFAULT = "/data2/local_datasets/world/world_analysis/cache/auto_research/v3_h5"


def fit_adapter(z):
    """z (B,N,D) raw → 채널별 (a, b): z_c ≈ a_c·LN(z)_c + b_c."""
    x = F.layer_norm(z, (D,))
    xm, zm = x.mean(1, keepdim=True), z.mean(1, keepdim=True)
    a = ((x - xm) * (z - zm)).sum(1, keepdim=True) / ((x - xm) ** 2).sum(1, keepdim=True).clamp_min(1e-6)
    return a, zm - a * xm


def _worker(rank, world, a, n):
    dev = torch.device(f"cuda:{rank}"); torch.cuda.set_device(dev)
    ds64 = arlib.WMADataset(dict(arlib.v3_cfg(0, 64, split=0)))
    bundle = arlib.build_bundle(arlib.v3_cfg(4, 60), dev, window=64)
    pred, enc = bundle.predictor, bundle.context_encoder
    recs = ds64.records
    idx = list(range(rank, n, world))
    loader = torch.utils.data.DataLoader(ClipDS(ds64, idx), batch_size=4, num_workers=a.workers, collate_fn=collate)
    mm = np.lib.format.open_memmap(Path(a.out) / "h5.npy", mode="r+")
    ac = arlib.autocast_ctx(arlib.v3_cfg(16, 16))
    t0 = time.time(); done = 0; adapter_err = []
    for ids, clips in loader:
        B = len(ids)
        for ci_, C in enumerate(CS):
            Tc = C // 2
            x = clips[:, :, SPLIT - C:SPLIT + 2 * J].to(dev, bundle.dtype)
            out = np.full((B, J, len(ARMS), 9), np.nan, np.float32)
            with torch.inference_mode(), ac():
                h = F.layer_norm(bundle.target_encoder(x), (D,)).float().reshape(B, Tc + J, S_TOK, D)
                cidx = idx_range(0, Tc, B, dev)
                z = enc(x, masks=[cidx])
                A_a, A_b = fit_adapter(z.float())
                adapt = lambda y: (A_a * F.layer_norm(y.float(), (D,)) + A_b).to(z.dtype)
                preds = {arm: [] for arm in ARMS}
                preds["os"] = list(pred(z, cidx, idx_range(Tc, J, B, dev), mask_index=0).float().reshape(B, J, S_TOK, D).unbind(1))
                preds["single"] = [pred(z, cidx, idx_range(Tc + j, 1, B, dev), mask_index=0).float() for j in range(J)]
                zstar = []                                              # 참 관측: 문맥을 j 까지 늘려 새 튜블릿 토큰
                for j in range(J):
                    zz = enc(x, masks=[idx_range(0, Tc + j + 1, B, dev)])
                    zstar.append(zz[:, (Tc + j) * S_TOK:])
                adapter_err.append(float(((adapt(zstar[0]).float() - zstar[0].float()).abs().mean() / zstar[0].float().abs().mean()).item()))
                for arm in ("ar_p", "ar_z", "ar_zA", "ar_h"):
                    Z = z
                    for j in range(J):
                        cj = idx_range(0, Tc + j, B, dev)
                        p = pred(Z, cj, idx_range(Tc + j, 1, B, dev), mask_index=0)
                        preds[arm].append(p.float())
                        new = {"ar_p": lambda: adapt(p), "ar_z": lambda: zstar[j], "ar_zA": lambda: adapt(zstar[j]),
                               "ar_h": lambda: adapt(h[:, Tc + j])}[arm]()
                        Z = torch.cat([Z, new.to(Z.dtype)], 1)
            last = [cellxy(recs[i].raw, SPLIT - 2) for i in ids]
            app = torch.stack([h[b, Tc - 1, last[b][1] * G + last[b][0]] for b in range(B)])
            for j in range(J):
                tru = [cellxy(recs[i].raw, SPLIT + 2 * j) for i in ids]
                tt = torch.stack([h[b, Tc + j, tru[b][1] * G + tru[b][0]] for b in range(B)])
                for m, arm in enumerate(ARMS):
                    P = preds[arm][j]
                    out[:, j, m, 0:4] = locate(P, tt).cpu().numpy()
                    out[:, j, m, 4:8] = locate(P, app).cpu().numpy()
                    out[:, j, m, 8] = (P - h[:, Tc + j]).abs().mean((-1, -2)).cpu().numpy()
            mm[np.asarray(ids), ci_] = out
        done += B
        if rank == 0 and (done // B) % 10 == 0:
            el = time.time() - t0
            print(f"[rank0] {done}/{len(idx)}  {el/60:.1f} 분  남은 {el/done*(len(idx)-done)/60:.1f} 분  어댑터 상대오차 {np.mean(adapter_err):.3f}", flush=True)
    mm.flush()
    json.dump({"adapter_rel_l1_mean": float(np.mean(adapter_err))}, open(Path(a.out) / f"adapter_r{rank}.json", "w"))
    print(f"[rank{rank}] 끝  어댑터 상대오차 {np.mean(adapter_err):.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--gpus", type=int, default=torch.cuda.device_count())
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    ds = arlib.WMADataset(dict(arlib.v3_cfg(0, 64, split=0)))
    n = a.limit or len(ds)
    m = np.lib.format.open_memmap(Path(a.out) / "h5.npy", mode="w+", dtype=np.float32, shape=(n, len(CS), J, len(ARMS), 9)); m[:] = np.nan; del m
    json.dump(dict(n=n, Cs=CS, J=J, split=SPLIT, arms=ARMS, video_ids=[r.video_id for r in ds.records[:n]],
                   cols=["tru_y", "tru_x", "tru_max", "tru_med", "app_y", "app_x", "app_max", "app_med", "l1"]),
              open(Path(a.out) / "meta.json", "w"))
    mp.spawn(_worker, args=(a.gpus, a, n), nprocs=a.gpus, join=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
