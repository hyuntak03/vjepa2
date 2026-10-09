#!/usr/bin/env python3
"""H4 — 미래 튜블릿 하나만 물으면 (단일 질의) predictor 는 시간 k 뒤의 물체를 어디에 두나. RollOut_v3.

predictor 와 encoder 는 위치를 RoPE 로만 받는다 → 질의가 "얼마나 먼 미래인가" 는 문맥 끝과의 **상대 시간 k** 로만 전달된다.
그래서 k 를 바꿔 가며 같은 문맥에 한 튜블릿 (256 mask 토큰) 만 물으면 predictor 의 "시간 조건 예측" 곡선이 나온다.
다른 미래 토큰이 없으므로 미래↔미래 상호작용도 빠진다 (joint 질의와 비교).

창: 문맥 [32−C, 32),  C ∈ {4, 8, 16, 32};  질의 k = 0..15 (미래 프레임 32+2k, 32+2k+1).
각 (clip, C, k) 저장 — 표적 h = LN(target_encoder([32−C, 64))):
  single  : 단일 질의 p^(k)
  joint   : 16 튜블릿 한 번 질의 (P = 32) 의 k 번째
  둘 다에 대해  loc_tru (y, x, maxcos, median)  템플릿 = h 의 진실 칸 토큰 (슬롯 k)          ← 진실 물체가 어디에 있나
               loc_app (y, x, maxcos, median)  템플릿 = h 의 문맥 마지막 튜블릿 물체 토큰      ← "마지막으로 본 그 물체" 가 어디에 있나
               l1      mean|p − h_k| (전 토큰)
⚠️ 정정 (2026-09-24): RoPE 가 순수 상대적이라 절대 위치 효과가 0 일 거라 예상했으나 **아니다** —
   `rotate_queries_or_keys` 의 주파수 복제 버그 (코드 주석 "subtle bug", 릴리즈 모델이 그대로 학습됨) 로 회전쌍 두 성분이
   다른 주파수를 받아 q·k 가 절대 위치에 의존한다 (smoke: 4 튜블릿 이동에 max|Δp| 1.34, |p| 평균 0.38).
   그래서 **위치 이동 팔** 을 둔다: C16 에서 문맥·질의 위치 인덱스를 통째로 s ∈ {4, 8} 튜블릿 민다 (내용은 같다).
   모드: single, joint, single_s4, single_s8 (앞의 둘은 모든 C, 뒤의 둘은 C16 만).

  sbatch 로 (8 GPU) 또는 python ... --gpus 1 --limit 8 --check
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
from h23_extract_v3 import ClipDS, collate, arr  # noqa: E402

G, S_TOK, D, CELL, SPLIT, K = 16, 256, 1280, 18.0, 32, 16
CS = (4, 8, 16, 32)
OUT_DEFAULT = "/data2/local_datasets/world/world_analysis/cache/auto_research/v3_h4"


def cellxy(raw, f0):
    x, y = arr(raw["px_x_by_sample"]), arr(raw["px_y_by_sample"])
    return np.clip(np.floor(np.array([(x[f0] + x[f0 + 1]) / 2, (y[f0] + y[f0 + 1]) / 2]) / CELL).astype(int), 0, G - 1)


def idx_range(t0, n, B, dev, shift=0):
    base = torch.arange((t0 + shift) * S_TOK, (t0 + shift + n) * S_TOK, device=dev)
    return base.unsqueeze(0).repeat(B, 1)


def locate(P, tpl):
    """P (B,256,D), tpl (B,D) → (B,4) = y, x, maxcos, median."""
    cs = F.normalize(P, dim=-1) @ F.normalize(tpl, dim=-1).unsqueeze(-1)
    cs = cs[..., 0]
    mx, am = cs.max(-1)
    return torch.stack([(am // G).float(), (am % G).float(), mx, cs.median(-1).values], -1)


def _worker(rank, world, a, n):
    dev = torch.device(f"cuda:{rank}"); torch.cuda.set_device(dev)
    ds64 = arlib.WMADataset(dict(arlib.v3_cfg(0, 64, split=0)))
    bundle = arlib.build_bundle(arlib.v3_cfg(4, 60), dev, window=64)
    pred = bundle.predictor
    recs = ds64.records
    idx = list(range(rank, n, world))
    loader = torch.utils.data.DataLoader(ClipDS(ds64, idx), batch_size=4, num_workers=a.workers, collate_fn=collate)
    mm = np.lib.format.open_memmap(Path(a.out) / "h4.npy", mode="r+")        # (n, 4C, K, 4 mode, 9)
    cfg = arlib.v3_cfg(16, 16)
    ac = arlib.autocast_ctx(cfg)
    t0 = time.time(); done = 0
    for ids, clips in loader:
        B = len(ids)
        for ci_, C in enumerate(CS):
            Tc = C // 2
            x = clips[:, :, SPLIT - C:].to(dev, bundle.dtype)                      # 문맥 + 미래 32 프레임
            with torch.inference_mode(), ac():
                h = bundle.target_encoder(x); h = F.layer_norm(h, (D,)).float().reshape(B, Tc + K, S_TOK, D)
                cidx = idx_range(0, Tc, B, dev)
                z = bundle.context_encoder(x, masks=[cidx])
                pj = pred(z, cidx, idx_range(Tc, K, B, dev), mask_index=0).float().reshape(B, K, S_TOK, D)
                ps = torch.stack([pred(z, cidx, idx_range(Tc + k, 1, B, dev), mask_index=0).float() for k in range(K)], 1)
                shifted = {}
                if C == 16:
                    for sft in (4, 8):
                        shifted[sft] = torch.stack([pred(z, idx_range(0, Tc, B, dev, sft), idx_range(Tc + k, 1, B, dev, sft), mask_index=0).float()
                                                    for k in range(K)], 1)
                    if done == 0:
                        for sft, v in shifted.items():
                            print(f"[shift] s={sft}: 상대 L1 |p(s)−p(0)|/|p(0)| = {((v - ps).abs().mean() / ps.abs().mean()).item():.3f}", flush=True)
            last = [cellxy(recs[i].raw, SPLIT - 2) for i in ids]
            app = torch.stack([h[b, Tc - 1, last[b][1] * G + last[b][0]] for b in range(B)])
            out = np.full((B, K, 4, 9), np.nan, np.float32)
            for k in range(K):
                tru = [cellxy(recs[i].raw, SPLIT + 2 * k) for i in ids]
                tt = torch.stack([h[b, Tc + k, tru[b][1] * G + tru[b][0]] for b in range(B)])
                modes = [ps[:, k], pj[:, k]] + ([shifted[4][:, k], shifted[8][:, k]] if shifted else [])
                for m, P in enumerate(modes):
                    out[:, k, m, 0:4] = locate(P, tt).cpu().numpy()
                    out[:, k, m, 4:8] = locate(P, app).cpu().numpy()
                    out[:, k, m, 8] = (P - h[:, Tc + k]).abs().mean((-1, -2)).cpu().numpy()
            mm[np.asarray(ids), ci_] = out
            del h, z, pj, ps, shifted
        done += B
        if rank == 0 and (done // B) % 10 == 0:
            el = time.time() - t0
            print(f"[rank0] {done}/{len(idx)}  {el/60:.1f} 분  남은 {el/done*(len(idx)-done)/60:.1f} 분", flush=True)
    mm.flush()
    print(f"[rank{rank}] 끝", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--gpus", type=int, default=torch.cuda.device_count())
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    ds = arlib.WMADataset(dict(arlib.v3_cfg(0, 64, split=0)))
    n = a.limit or len(ds)
    m = np.lib.format.open_memmap(Path(a.out) / "h4.npy", mode="w+", dtype=np.float32, shape=(n, len(CS), K, 4, 9)); m[:] = np.nan; del m
    json.dump(dict(n=n, Cs=CS, K=K, split=SPLIT, video_ids=[r.video_id for r in ds.records[:n]],
                   modes=["single", "joint", "single_s4", "single_s8"], cols=["tru_y", "tru_x", "tru_max", "tru_med", "app_y", "app_x", "app_max", "app_med", "l1"]),
              open(Path(a.out) / "meta.json", "w"))
    mp.spawn(_worker, args=(a.gpus, a, n), nprocs=a.gpus, join=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
