#!/usr/bin/env python3
"""P3b — 닫힘 검정 2 판 (적대 검증 2026-09-25 권고): 자기 예측 되먹임의 멈춤을 **형식 · 누적 · 토큰 내용** 으로 가른다.

P3 1 판: 같은 채널 어댑터 A 로 인과 표적 hc 를 넣으면 ar_z 처럼 전진, 자기 예측 p 는 j3 부터 정체.
검증: 양방향 표적 **참** 토큰 h_j (p 가 회귀하는 형식) 도 같은 A 로 j≥4 에 정체 → "관측 대 예측" 이 "인과 창 끝 토큰 대 양방향 표적형 토큰" 과 교락.
팔 (C16, 8 걸음, 모두 같은 784 clip):
  ar_z      참 관측 (문맥 encoder 를 j 까지 늘림; 인과, 문맥 공간)                    — 상한
  ar_zfull  문맥 encoder 를 **창 전체 (32 장)** 에 돌린 튜블릿 j 토큰 (비인과, 문맥 공간, 사상 없음)
  ar_hcA    A(인과 표적 hc_j)                                                        — P3 재현
  ar_hA     A(양방향 표적 참 h_j) (창 전체 target encoder)                           — H5 ar_h 재현
  ar_pA     A(자기 예측 p_j)                                                          — H5 ar_p 재현
  tf_last   문맥 = [z ; z*_0 … z*_{j−2} ; A(p^z_{j−1})]  — 참 역사 + **마지막 하나만** 자기 예측 (p^z = ar_z 팔의 예측)
  tf_first  문맥 = [z ; A(p_0) … A(p_{j−2}) ; z*_{j−1}]   — 자기 예측 역사 (ar_pA 사슬) + **마지막만** 참 관측
판정: tf_last 가 정체 → 토큰 하나의 내용이 원인. tf_last 는 전진 · ar_pA 만 정체 → 누적이 원인.
      tf_first 가 전진 → 마지막 토큰이 다음 걸음을 정한다 (역사 무관). ar_zfull 이 정체 → 비인과 (창 중간) 형식이 원인.
읽기: 표준 h_j 진실 칸 템플릿 argmax + 문맥 마지막 외형 템플릿 argmax (둘 다 저장), L1.

  python p3b_closure_v2.py --gpus 8    /   --gpus 1 --limit 4 --out /tmp/p3b_smoke
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
from p3_closure_v3 import select  # noqa: E402

G, S_TOK, D, SPLIT, J, C = 16, 256, 1280, 32, 8, 16
TC = C // 2
ARMS = ("ar_z", "ar_zfull", "ar_hcA", "ar_hA", "ar_pA", "tf_last", "tf_first")
OUT_DEFAULT = "/data2/local_datasets/world/world_analysis/cache/auto_research/v3_p3b"


def _worker(rank, world, a, ev):
    dev = torch.device(f"cuda:{rank}"); torch.cuda.set_device(dev)
    ds64 = arlib.WMADataset(dict(arlib.v3_cfg(0, 64, split=0))); recs = ds64.records
    bundle = arlib.build_bundle(arlib.v3_cfg(4, 60), dev, window=64); pred, enc, tgt = bundle.predictor, bundle.context_encoder, bundle.target_encoder
    ac = arlib.autocast_ctx(arlib.v3_cfg(16, 16))
    mm = np.lib.format.open_memmap(Path(a.out) / "p3b.npy", mode="r+")        # (n, J, arms, 9)
    mine = list(range(rank, len(ev), world)); t0 = time.time()
    q = lambda Z, j: pred(Z, idx_range(0, TC + j, 1, dev), idx_range(TC + j, 1, 1, dev), mask_index=0)
    for n_done, row in enumerate(mine):
        i = ev[row]
        x = ds64.clip(i)[None, :, SPLIT - C:SPLIT + 2 * J].to(dev, bundle.dtype)
        with torch.inference_mode(), ac():
            h = F.layer_norm(tgt(x), (D,)).float().reshape(TC + J, S_TOK, D)
            ci = idx_range(0, TC, 1, dev); z = enc(x, masks=[ci])
            Aa, Ab = fit_adapter(z.float()); mapA = lambda y: (Aa * F.layer_norm(y.float(), (D,)) + Ab)
            zfull = enc(x)                                                           # 창 전체, 마스크 없음
            zstar, hc = [], []
            for j in range(J):
                zstar.append(enc(x, masks=[idx_range(0, TC + j + 1, 1, dev)])[:, (TC + j) * S_TOK:])
                hc.append(F.layer_norm(tgt(x[:, :, :C + 2 * (j + 1)]), (D,))[:, (TC + j) * S_TOK:])
            feed = {"ar_z": lambda j, p: zstar[j], "ar_zfull": lambda j, p: zfull[:, (TC + j) * S_TOK:(TC + j + 1) * S_TOK],
                    "ar_hcA": lambda j, p: mapA(hc[j]), "ar_hA": lambda j, p: mapA(h[TC + j][None]), "ar_pA": lambda j, p: mapA(p)}
            preds = {}
            for arm in ("ar_z", "ar_zfull", "ar_hcA", "ar_hA", "ar_pA"):
                Z = z; out = []
                for j in range(J):
                    p = q(Z, j); out.append(p.float())
                    Z = torch.cat([Z, feed[arm](j, p).to(Z.dtype)], 1)
                preds[arm] = out
            # 섞은 역사 팔
            tl, tf = [preds["ar_z"][0]], [preds["ar_z"][0]]
            for j in range(1, J):
                hist_true = [zstar[k] for k in range(j - 1)]
                Zl = torch.cat([z] + hist_true + [mapA(preds["ar_z"][j - 1]).to(z.dtype)], 1); tl.append(q(Zl, j).float())
                hist_self = [mapA(preds["ar_pA"][k]).to(z.dtype) for k in range(j - 1)]
                Zf = torch.cat([z] + hist_self + [zstar[j - 1]], 1); tf.append(q(Zf, j).float())
            preds["tf_last"], preds["tf_first"] = tl, tf
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
    mm.flush(); print(f"[rank{rank}] 끝", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DEFAULT); ap.add_argument("--gpus", type=int, default=torch.cuda.device_count())
    ap.add_argument("--every", type=int, default=2); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(); Path(a.out).mkdir(parents=True, exist_ok=True)
    ds = arlib.WMADataset(dict(arlib.v3_cfg(0, 64, split=0))); R = ds.records
    _, ev = select(R, a.every)
    if a.limit: ev = ev[:: max(1, len(ev) // a.limit)][: a.limit]
    n = len(ev)
    m = np.lib.format.open_memmap(Path(a.out) / "p3b.npy", mode="w+", dtype=np.float32, shape=(n, J, len(ARMS), 9)); m[:] = np.nan; del m
    json.dump(dict(n=n, arms=ARMS, J=J, C=C, video_ids=[R[i].video_id for i in ev], scenario=[R[i].raw["scenario"] for i in ev],
                   traj=[[R[i].raw["scenario"], R[i].raw["primary"], R[i].raw["secondary"]] for i in ev],
                   cols=["tru_y", "tru_x", "tru_max", "tru_med", "app_y", "app_x", "app_max", "app_med", "l1"]),
              open(Path(a.out) / "meta.json", "w"))
    print("eval clips", n, flush=True)
    mp.spawn(_worker, args=(a.gpus, a, ev), nprocs=a.gpus, join=True); print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
