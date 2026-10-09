#!/usr/bin/env python3
"""H2 + H3 추출 — RollOut_v3 16 창 × predictor 12 층 렌즈 (frozen 릴리즈 ViT-H).

창 = 문맥 [split−C, split) + 예측 [split, split+P), C·P ∈ {4,8,16,32} (RollOutV3 와 같은 16 창).
bundle 은 window 64 로 **한 번만** 짓는다 — 창 총합대로 지은 것과 z·p·h 비트 동일 (check_window_build.py).
clip 마다 64 프레임을 한 번 읽고 창마다 잘라 쓴다. 수치 경로는 표준 채점 (fp32 + autocast fp16, mask_index 0).

렌즈  p^(l) = predictor_proj(predictor_norm(q^(l)[미래]))   l = 0..11   (L11 = p, 비트 동일 — validate_arlib.py)

clip × 창 × 미래 슬롯 t 마다 저장 (분석은 h23_analyze_v3.py 가 CPU 로):
  vec   (n, S, 29, 1280) fp16 — 3×3 창 평균 벡터
          0..11  p^(l)_t @ W_t          W_t = 슬롯 t 의 진실 물체 칸 3×3
          12..23 p^(l)_t @ W_T          W_T = 문맥 마지막 튜블릿 T 의 물체 칸 3×3 ("마지막으로 본 자리")
          24 h_t @ W_t   25 h_t @ W_T   26 h_T @ W_t   27 h_T @ W_T   28 LN(z)_T @ W_T
        (h = LN(target_encoder(창 전체)),  z = 문맥 encoder 최종 출력)
  l1    (n, S, 14) fp32 — 슬롯 t 전 토큰 mean|x − h_t|:  0..11 렌즈 p^(l),  12 복사 h_T (마지막 문맥 튜블릿 그대로),  13 복사 LN(z)_T
  loc   (n, S, 12, 4) fp32 — 파라미터 0 위치 읽기: 템플릿 = h_t 의 진실 칸 토큰, p^(l)_t 256 토큰과 cos →
          (argmax y, argmax x, max cos, median cos)          (RollOutV2 §4B 의 자 없는 검사와 같은 식)
  step  (n, S, 13) fp32 — mean|x_{t+1} − x_t| (전 토큰): 0..11 렌즈, 12 h.  마지막 슬롯은 NaN
  conv  (n, S, 12) fp32 — 렌즈 수렴: 토큰 평균 cos(p^(l)_t, p_t)
가능 clip 만 합 (중심화용): sum_p[l,t,pos,:], sumsq_p[l,t,pos], sum_h[s,pos,:], sumsq_h[s,pos] (s = 창 전체 튜블릿), n.

  sbatch 로 (auto_research/scripts/h23_extract_v3.sbatch) — 8 GPU, vll6
  python auto_research/scripts/h23_extract_v3.py --limit 8 --windows 16x16 --gpus 1     # 배관 점검
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arlib  # noqa: E402

G, S_TOK, D, CELL = 16, 256, 1280, 18.0
NL = 12
OUT_DEFAULT = "/data2/local_datasets/world/world_analysis/cache/auto_research/v3_h23"


def arr(s):
    return np.array([float(v) for v in s.split()], dtype=np.float32)


def cells_of(raw, split, C, P):
    """튜블릿 중심 (프레임 2 장 평균, 288 px 좌표) → 칸 (x, y).  반환: last (2,), fut (S, 2)."""
    x, y = arr(raw["px_x_by_sample"]), arr(raw["px_y_by_sample"])
    def c(f0):
        cx = (x[f0] + x[f0 + 1]) / 2; cy = (y[f0] + y[f0 + 1]) / 2
        return np.clip(np.floor(np.array([cx, cy]) / CELL).astype(int), 0, G - 1)
    last = c(split - 2)
    fut = np.stack([c(split + 2 * t) for t in range(P // 2)])
    return last, fut


def win_idx(cx, cy):
    ys = range(max(0, cy - 1), min(G, cy + 2)); xs = range(max(0, cx - 1), min(G, cx + 2))
    return [yy * G + xx for yy in ys for xx in xs]


class ClipDS(torch.utils.data.Dataset):
    def __init__(self, ds, idx):
        self.ds, self.idx = ds, idx
    def __len__(self):
        return len(self.idx)
    def __getitem__(self, k):
        i = self.idx[k]
        return i, self.ds.clip(i)


def collate(b):
    return [x[0] for x in b], torch.stack([x[1] for x in b])


def mm_path(out, name, C, P):
    return Path(out) / f"{name}_C{C}_P{P}.npy"


def _worker(rank, world, a, n, windows):
    torch.set_num_threads(4)
    dev = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(dev)
    ds64 = arlib.WMADataset(dict(arlib.v3_cfg(0, 64, split=0)))           # 프레임 0..63 전부
    cfg64 = arlib.v3_cfg(4, 60)
    bundle = arlib.build_bundle(cfg64, dev, window=64)
    pred = bundle.predictor
    idx = list(range(rank, n, world))
    loader = torch.utils.data.DataLoader(ClipDS(ds64, idx), batch_size=a.load_bs, num_workers=a.workers,
                                         collate_fn=collate,
                                         prefetch_factor=2, persistent_workers=False)
    recs = ds64.records
    plaus = np.array([r.plausible == "1" for r in recs])
    # 창별 memmap / 합
    mm, sums = {}, {}
    for C, P in windows:
        S = P // 2; Stot = (C + P) // 2
        mm[(C, P)] = {k: np.lib.format.open_memmap(mm_path(a.out, k, C, P), mode="r+")
                      for k in ("vec", "l1", "loc", "step", "conv")}
        sums[(C, P)] = dict(sum_p=torch.zeros(NL, S, S_TOK, D, device=dev, dtype=torch.float32),
                            sumsq_p=torch.zeros(NL, S, S_TOK, device=dev, dtype=torch.float32),
                            sum_h=torch.zeros(Stot, S_TOK, D, device=dev, dtype=torch.float32),
                            sumsq_h=torch.zeros(Stot, S_TOK, device=dev, dtype=torch.float32),
                            n=0)   # fp32 합: rank 당 가능 clip ~350 개라 충분
    cfgs = {(C, P): arlib.v3_cfg(C, P, a.split) for C, P in windows}
    t0 = time.time(); done = 0
    for ids, clips in loader:
        B0 = len(ids)
        for C, P in windows:
            S = P // 2; Tc = C // 2
            cfg = cfgs[(C, P)]
            ntok = (C + P) // 2 * S_TOK
            sb = max(1, min(B0, a.token_budget // ntok))
            for s0 in range(0, B0, sb):
                bid = ids[s0:s0 + sb]; B = len(bid)
                x = clips[s0:s0 + sb, :, a.split - C:a.split + P]
                o = arlib.forward(bundle, x, cfg, ctx_frames=C, pred_layers=range(NL), want_h=True, want_lens=True)
                h = o["h"].float().reshape(B, (C + P) // 2, S_TOK, D)
                hf = h[:, Tc:]                                          # (B, S, 256, D) 미래
                hT = h[:, Tc - 1]                                       # (B, 256, D) 문맥 마지막 튜블릿
                zT = F.layer_norm(o["z"].float(), (D,)).reshape(B, Tc, S_TOK, D)[:, Tc - 1]
                L = torch.stack([o["lens"][l].float().reshape(B, S, S_TOK, D) for l in range(NL)], 1)  # (B,12,S,256,D)
                assert torch.equal(o["lens"][NL - 1], o["p"]), "렌즈 L11 != p"
                del o
                # --- 칸
                cells = [cells_of(recs[i].raw, a.split, C, P) for i in bid]
                vec = torch.zeros(B, S, 29, D, device=dev)
                for b, (last, fut) in enumerate(cells):
                    wT = win_idx(*last)
                    for t in range(S):
                        wt = win_idx(*fut[t])
                        vec[b, t, 0:12] = L[b, :, t][:, wt].mean(1)
                        vec[b, t, 12:24] = L[b, :, t][:, wT].mean(1)
                        vec[b, t, 24] = hf[b, t, wt].mean(0); vec[b, t, 25] = hf[b, t, wT].mean(0)
                        vec[b, t, 26] = hT[b, wt].mean(0); vec[b, t, 27] = hT[b, wT].mean(0)
                        vec[b, t, 28] = zT[b, wT].mean(0)
                # --- L1 (전 토큰)
                l1 = torch.empty(B, S, 14, device=dev)
                l1[:, :, :12] = (L - hf[:, None]).abs().mean((-1, -2)).permute(0, 2, 1)
                l1[:, :, 12] = (hT[:, None] - hf).abs().mean((-1, -2))
                l1[:, :, 13] = (zT[:, None] - hf).abs().mean((-1, -2))
                # --- 자 없는 위치 읽기
                loc = torch.empty(B, S, NL, 4, device=dev)
                Ln = F.normalize(L, dim=-1)
                for b, (last, fut) in enumerate(cells):
                    for t in range(S):
                        tpl = F.normalize(hf[b, t, fut[t][1] * G + fut[t][0]], dim=-1)
                        cs = Ln[b, :, t] @ tpl                          # (12, 256)
                        mx, am = cs.max(-1)
                        loc[b, t, :, 0] = (am // G).float(); loc[b, t, :, 1] = (am % G).float()
                        loc[b, t, :, 2] = mx; loc[b, t, :, 3] = cs.median(-1).values
                # --- 걸음 · 수렴
                step = torch.full((B, S, 13), float("nan"), device=dev)
                if S > 1:
                    step[:, :-1, :12] = (L[:, :, 1:] - L[:, :, :-1]).abs().mean((-1, -2)).permute(0, 2, 1)
                    step[:, :-1, 12] = (hf[:, 1:] - hf[:, :-1]).abs().mean((-1, -2))
                conv = (Ln * Ln[:, NL - 1:NL]).sum(-1).mean(-1).permute(0, 2, 1)   # (B, S, 12)
                # --- 쓰기
                ib = np.asarray(bid)
                M = mm[(C, P)]
                M["vec"][ib] = vec.half().cpu().numpy(); M["l1"][ib] = l1.cpu().numpy()
                M["loc"][ib] = loc.cpu().numpy(); M["step"][ib] = step.cpu().numpy(); M["conv"][ib] = conv.cpu().numpy()
                # --- 합 (가능 clip 만)
                pm = torch.tensor(plaus[ib], device=dev)
                if pm.any():
                    Lp, hp = L[pm], h[pm]
                    Sm = sums[(C, P)]
                    Sm["sum_p"] += Lp.sum(0); Sm["sumsq_p"] += (Lp ** 2).sum((0, -1))
                    Sm["sum_h"] += hp.sum(0); Sm["sumsq_h"] += (hp ** 2).sum((0, -1))
                    Sm["n"] += int(pm.sum())
                del L, Ln, h, hf, vec
        done += B0
        if rank == 0 and (done // B0) % 5 == 0:
            el = time.time() - t0
            print(f"[rank0] {done}/{len(idx)} clip  {el/60:.1f} 분  남은 {el/done*(len(idx)-done)/60:.1f} 분", flush=True)
    for (C, P), Sm in sums.items():
        np.savez(Path(a.out) / f"sums_C{C}_P{P}_r{rank}.npz",
                 sum_p=Sm["sum_p"].float().cpu().numpy(), sumsq_p=Sm["sumsq_p"].cpu().numpy(),
                 sum_h=Sm["sum_h"].float().cpu().numpy(), sumsq_h=Sm["sumsq_h"].cpu().numpy(), n=Sm["n"])
        for v in mm[(C, P)].values():
            v.flush()
    print(f"[rank{rank}] 끝 {len(idx)} clip {(time.time()-t0)/60:.1f} 분", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--windows", nargs="*", default=None, help="예: 16x16 4x32 (없으면 16 창 전부)")
    ap.add_argument("--split", type=int, default=32)
    ap.add_argument("--gpus", type=int, default=torch.cuda.device_count())
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--load-bs", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--token-budget", type=int, default=16384)
    a = ap.parse_args()
    windows = ([tuple(int(v) for v in w.split("x")) for w in a.windows] if a.windows else arlib.V3_WINDOWS)
    for C, P in windows:
        assert a.split - C >= 0 and a.split + P <= 64, (C, P, a.split)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    ds = arlib.WMADataset(dict(arlib.v3_cfg(0, 64, split=0)))
    n = len(ds) if not a.limit else min(a.limit, len(ds))
    for C, P in windows:
        S = P // 2
        shapes = dict(vec=((n, S, 29, D), np.float16), l1=((n, S, 14), np.float32), loc=((n, S, NL, 4), np.float32),
                      step=((n, S, 13), np.float32), conv=((n, S, NL), np.float32))
        for k, (sh, dt) in shapes.items():
            m = np.lib.format.open_memmap(mm_path(a.out, k, C, P), mode="w+", dtype=dt, shape=sh)
            if dt == np.float32:
                m[:] = np.nan
            del m
    meta = dict(n=n, windows=windows, split=a.split, video_ids=[r.video_id for r in ds.records[:n]],
                scenario=[r.raw["scenario"] for r in ds.records[:n]], plausible=[r.plausible for r in ds.records[:n]],
                vec_slots=["p_L%d@Wt" % l for l in range(NL)] + ["p_L%d@WT" % l for l in range(NL)]
                + ["h_t@Wt", "h_t@WT", "h_T@Wt", "h_T@WT", "lnz_T@WT"],
                l1_cols=["p_L%d" % l for l in range(NL)] + ["copy_h_T", "copy_lnz_T"],
                loc_cols=["argmax_y", "argmax_x", "max_cos", "median_cos"],
                numerics="fp32 weights + autocast fp16, mask_index 0, bundle window 64 (bit-identical to per-window build)")
    json.dump(meta, open(Path(a.out) / "meta.json", "w"))
    print(f"n={n} windows={len(windows)} gpus={a.gpus} out={a.out}", flush=True)
    mp.spawn(_worker, args=(a.gpus, a, n, windows), nprocs=a.gpus, join=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
