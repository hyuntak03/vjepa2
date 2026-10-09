#!/usr/bin/env python3
"""M6 — 역사 시간 이동 (v3, 가림 없음): predictor 는 역사 토큰 속 **과거 물체 자리** 에서 물체를 가져와 뒤처지는가.

동기 (M2 적대 검증 2026-09-25): v11 쌍둥이에서 같은 경계에 역사만 바꾸면 위치 적중이 −0.18, 역사를 지우면 +0.09 — 역사 토큰이 자리를 뒤로 끄는 것으로 보였다.
그러나 v11 early 의 역사에는 이른 가림이 섞여 교란이 크다. 가림 없는 v3 에서 **같은 clip 의 역사를 시간만 옮겨** 직접 잰다.

팔 (C16 = 문맥 raw [16,32), 8 튜블릿; 예측 raw [32,64), 16 튜블릿; 경계 = 튜블릿 6–7 은 원래 인코딩 그대로, 위치 인덱스 원래 자리):
  base      원래 문맥
  hist-8    튜블릿 0–5 를 raw [8,20) (8 프레임 과거) 를 인코딩한 토큰으로   (창 [8,24) 인코딩의 앞 6 튜블릿)
  hist-4    raw [12,24) 로                                                (창 [12,28))
  hist+4    raw [20,32) 로 (경계 쪽으로 4 프레임, 미래 누수 없음)           (창 [20,36) 은 미래를 보므로 대신 창 [16,32) 를 4 프레임 민 [20,32)+패딩? → 아래 주)
  hist_none 역사 삭제 (경계 2 튜블릿만; M1 pred_last2 와 같음)
  주: hist+4 는 창 [20,36) 을 인코딩하면 raw 32–35 (미래) 를 보게 된다. 그래서 창 [20,32) (12 프레임, 6 튜블릿) 만 인코딩해 그대로 쓴다.
      hist-8 / hist-4 도 같은 방식 (12 프레임 창만 인코딩) 으로 맞춘다 — 세 팔 모두 "역사 12 프레임만 본 encoder 토큰".
      base_iso = 원래 역사 raw [16,28) 도 12 프레임 창만 인코딩한 토큰으로 바꾼 팔 (인코딩 방식 대조).
2×2 (교차 비평 C12 권고): base (enc 역사 ○, pred 역사 ○) · hist_none (○, ×) · histZ+bndM (×, ○) · bndM_only (×, ×) — bndM = 경계 튜블릿만 마스크로 인코딩 (RoPE 위치 보존).
예측: 역사에서 과거 자리를 가져와 뒤처진다면 진행 (진실 방향 투영) 이 hist-8 < hist-4 < base_iso < hist+4 순, 거리에 비례.
읽기: 슬롯마다 템플릿 두 개 (표준 h 진실 칸 / 문맥 마지막 외형) argmax, 진행 = (a − L)·u (칸).

  python m6_history_shift_v3.py --gpus 8    /  --gpus 1 --limit 4 --out /tmp/m6_smoke
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
from p3_closure_v3 import select  # noqa: E402

G, S_TOK, D, SPLIT, C, K = 16, 256, 1280, 32, 16, 16
TC = C // 2
ARMS = ["base", "base_iso", "hist-8", "hist-4", "hist+4", "hist_none", "bndM_only", "histZ+bndM"]
OUT_DEFAULT = "/data2/local_datasets/world/world_analysis/cache/auto_research/v3_m6"


def _worker(rank, world, a, ev):
    dev = torch.device(f"cuda:{rank}"); torch.cuda.set_device(dev)
    ds64 = arlib.WMADataset(dict(arlib.v3_cfg(0, 64, split=0))); recs = ds64.records
    bundle = arlib.build_bundle(arlib.v3_cfg(4, 60), dev, window=64); pred, enc, tgt = bundle.predictor, bundle.context_encoder, bundle.target_encoder
    ac = arlib.autocast_ctx(arlib.v3_cfg(16, 16))
    mm = np.lib.format.open_memmap(Path(a.out) / "m6.npy", mode="r+")          # (n, arms, K, 8)
    mine = list(range(rank, len(ev), world)); t0 = time.time()
    ci = idx_range(0, TC, 1, dev); ti = idx_range(TC, K, 1, dev)
    for n_done, row in enumerate(mine):
        i = ev[row]
        full = ds64.clip(i)[None].to(dev, bundle.dtype)                          # (1,3,64,H,W)
        x = full[:, :, SPLIT - C:SPLIT + 2 * K]                                   # raw [16, 64)
        with torch.inference_mode(), ac():
            h = F.layer_norm(tgt(x), (D,)).float().reshape(TC + K, S_TOK, D)
            z = enc(x, masks=[ci])                                                # (1, 8*256, D)
            bnd = z[:, 6 * S_TOK:]
            def hist12(r0):                                                       # raw [r0, r0+12) 만 인코딩 → 6 튜블릿
                return enc(full[:, :, r0:r0 + 12])
            Z = {"base": z, "base_iso": torch.cat([hist12(16), bnd], 1), "hist-8": torch.cat([hist12(8), bnd], 1),
                 "hist-4": torch.cat([hist12(12), bnd], 1), "hist+4": torch.cat([hist12(20), bnd], 1)}
            P = {k: pred(v, ci, ti, mask_index=0).float().reshape(K, S_TOK, D) for k, v in Z.items()}
            P["hist_none"] = pred(bnd, idx_range(6, 2, 1, dev), ti, mask_index=0).float().reshape(K, S_TOK, D)
            # 2×2 (교차 비평 권고): encoder 수준 역사 (경계 토큰이 역사 프레임을 봤나) × predictor 수준 역사 (역사 토큰이 있나)
            bndM = enc(x, masks=[idx_range(6, 2, 1, dev)])                             # 경계만 보고 인코딩, 위치 보존
            P["bndM_only"] = pred(bndM, idx_range(6, 2, 1, dev), ti, mask_index=0).float().reshape(K, S_TOK, D)
            P["histZ+bndM"] = pred(torch.cat([z[:, :6 * S_TOK], bndM], 1), ci, ti, mask_index=0).float().reshape(K, S_TOK, D)
        lc = cellxy(recs[i].raw, SPLIT - 2); app = h[TC - 1, lc[1] * G + lc[0]][None]
        for t in range(K):
            tc = cellxy(recs[i].raw, SPLIT + 2 * t); tt = h[TC + t, tc[1] * G + tc[0]][None]
            for ai, arm in enumerate(ARMS):
                mm[row, ai, t, 0:4] = locate(P[arm][t][None], tt).cpu().numpy()[0]
                mm[row, ai, t, 4:8] = locate(P[arm][t][None], app).cpu().numpy()[0]
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
    m = np.lib.format.open_memmap(Path(a.out) / "m6.npy", mode="w+", dtype=np.float32, shape=(n, len(ARMS), K, 8)); m[:] = np.nan; del m
    json.dump(dict(n=n, arms=ARMS, K=K, C=C, video_ids=[R[i].video_id for i in ev], scenario=[R[i].raw["scenario"] for i in ev],
                   traj=[[R[i].raw["scenario"], R[i].raw["primary"], R[i].raw["secondary"]] for i in ev],
                   cols=["tru_y", "tru_x", "tru_max", "tru_med", "app_y", "app_x", "app_max", "app_med"]), open(Path(a.out) / "meta.json", "w"))
    print("eval clips", n, flush=True)
    mp.spawn(_worker, args=(a.gpus, a, ev), nprocs=a.gpus, join=True); print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
