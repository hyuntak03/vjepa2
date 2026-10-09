#!/usr/bin/env python3
"""H6e — IntPhys 1 을 **인과 표적** 으로 채점하면 점수가 얼마나 남나.

H6c/H6d 에서 확인한 것: 표준 채점의 표적 h = LN(target_encoder(창 전체)) 는 창 안의 **미래 프레임까지 양방향으로 본다** →
보이는 분기가 일어나기 전 튜블릿에서도 pos/imp 의 h 가 달라 판별이 된다 (이동+보임 pre 89.5 %).
그 몫을 빼려면 표적을 인과적으로 만든다:
  h^c_u = LN(target_encoder(창 프레임 [0, 2(u+1))))[튜블릿 u]      (튜블릿 u 까지만 본다; 시작점 s 와 u 에만 의존 → C 끼리 재사용)
p 와 copyz 는 표준과 같다. 같은 실행에서 표준 표적 h (창 전체) 도 같이 남긴다.
행 (video, combo, start, C, j) → p_causal, copyz_causal, p_full, copyz_full  (전 토큰 mean L1).

  sbatch 로 (GPU 8, rank 8) — python ... --rank r --world 8
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arlib  # noqa: E402
from evals.world_model_analysis.eval import _intphys1_windows  # noqa: E402

S_TOK, D = 256, 1280
OUT = arlib.ROOT / "auto_research/exp_results/h6"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    dev = torch.device("cuda:0")
    base = arlib.load_cfg("intphys1_sliding__intphys1_dev_vith")
    cfg = dict(base, model=dict(base["model"], window_size=64))
    ds = arlib.WMADataset(cfg)
    bundle = arlib.build_bundle(cfg, dev, window=64)
    W = base["surprise"]["intphys1"]
    combos = []
    for sk in (2, 5):
        for wz in (16, 32):
            cl = [m * wz // 16 for m in W["context_mult"]]
            wins = _intphys1_windows(ds.n_frames, sk, wz, cl, int(W["stride"]), 2, str(W["frame_budget"]))
            if wins:
                combos.append((f"skip{sk}_w{wz}", wins))
    idx = list(range(a.rank, len(ds), a.world))
    if a.limit:
        idx = idx[: a.limit]
    ac = arlib.autocast_ctx(cfg)
    rows = []; t0 = time.time()
    for n_done, i in enumerate(idx):
        clip = ds.clip(i)
        for ci_, (name, wins) in enumerate(combos):
            by_start = {}
            for C, cf, tf in wins:
                by_start.setdefault(cf[0], []).append((C, cf, tf))
            starts = sorted(by_start)
            X = torch.stack([clip[:, by_start[st][0][1] + by_start[st][0][2]] for st in starts]).to(dev, bundle.dtype)
            nS, _, wz = X.shape[:3]
            U = wz // 2
            with torch.inference_mode(), ac():
                hfull = F.layer_norm(bundle.target_encoder(X), (D,)).float().reshape(nS, U, S_TOK, D)
                hc = torch.empty(nS, U, S_TOK, D, device=dev)
                for u in range(U):
                    o = bundle.target_encoder(X[:, :, :2 * (u + 1)])
                    hc[:, u] = F.layer_norm(o, (D,)).float().reshape(nS, u + 1, S_TOK, D)[:, u]
                for si, st in enumerate(starts):
                    for C, cf, tf in by_start[st]:
                        Tc = C // 2
                        x = X[si:si + 1]
                        ci, ti = arlib.ci_ti(1, C, wz - C, dev)
                        z = bundle.context_encoder(x, masks=[ci])
                        p = bundle.predictor(z, ci, ti, mask_index=0).float().reshape(U - Tc, S_TOK, D)
                        zT = F.layer_norm(z.float(), (D,)).reshape(Tc, S_TOK, D)[Tc - 1]
                        for j in range(U - Tc):
                            u = Tc + j
                            rows.append((i, ci_, st, C, j,
                                         (p[j] - hc[si, u]).abs().mean().item(), (zT - hc[si, u]).abs().mean().item(),
                                         (p[j] - hfull[si, u]).abs().mean().item(), (zT - hfull[si, u]).abs().mean().item()))
        if n_done % 5 == 0:
            el = time.time() - t0
            print(f"[{a.rank}] {n_done+1}/{len(idx)}  {el/60:.1f} 분  남은 {el/(n_done+1)*(len(idx)-n_done-1)/60:.1f} 분", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"causal_l1_vith_r{a.rank}.npz", rows=np.array(rows, np.float64),
             cols=np.array(["video_i", "combo_i", "start", "C", "j", "p_causal", "copyz_causal", "p_full", "copyz_full"]),
             combos=np.array([c for c, _ in combos]), video_ids=np.array([r.video_id for r in ds.records]))
    print("DONE", len(rows), flush=True)


if __name__ == "__main__":
    main()
