#!/usr/bin/env python3
"""M1 + M3 — RollOut_v3 (C16/P32): predictor 의 유효 기억 (M1, 개입) 과 미래 토큰의 속도 상태 (M3, 표현).

M1 유효 기억 — predictor 입력의 앞쪽 문맥을 지우거나 바꿔도 미래가 같은가 (같으면 역사를 통합하지 않는다):
  full        문맥 8 튜블릿 (encoder 는 16 프레임 전부)
  pred_last1/2/4   predictor 입력을 마지막 m 튜블릿으로 (encoder 는 전체를 봤다 → 역사는 경계 토큰 안에만 남는다; 위치 인덱스는 원래 자리)
  enc_last2/4      encoder 도 마지막 2m 프레임만 본다 (역사가 시스템 어디에도 없다)
  hist_swap   앞쪽 6 튜블릿을 **같은 법칙 · 다른 궤적** clip 의 것으로 교체 (토큰 수 불변 — 삭제의 분포 밖 효과를 뺀 인과 대조)
  측정 (미래 16 슬롯): Δp = mean|p_arm − p_full| / mean|p_full|;  위치 읽기 (템플릿 = h 진실 칸 토큰) argmax · contrast.
M3 속도 상태 — full 팔에서, p 가 스스로 둔 물체 자리 (템플릿 = 문맥 마지막 튜블릿의 LN(h) 물체 토큰 = 마지막으로 본 외형; 미래 진실 안 씀) 의
  3×3 창 평균 토큰 (슬롯마다) 과 h 의 진실 칸 3×3 (천장), z 의 문맥 마지막 물체 창 (출발점) 을 저장 → CPU 에서 궤적 단위 ridge 로 문맥 끝 속도·방향 readout.
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
from h23_extract_v3 import ClipDS, collate, arr, win_idx  # noqa: E402
from h4_single_query_v3 import idx_range, locate, cellxy  # noqa: E402

G, S_TOK, D, SPLIT, C, K = 16, 256, 1280, 32, 16, 16
TC = C // 2
ARMS = ["full", "pred_last1", "pred_last2", "pred_last4", "enc_last2", "enc_last4", "hist_swap"]
OUT_DEFAULT = "/data2/local_datasets/world/world_analysis/cache/auto_research/v3_m13"
FREE = ("flat_v", "flat_a", "flat_d", "ramp_a", "ramp_d", "arc")


def _worker(rank, world, a, sel, donor):
    dev = torch.device(f"cuda:{rank}"); torch.cuda.set_device(dev)
    ds64 = arlib.WMADataset(dict(arlib.v3_cfg(0, 64, split=0)))
    bundle = arlib.build_bundle(arlib.v3_cfg(4, 60), dev, window=64); pred = bundle.predictor; recs = ds64.records
    mine = [sel[i] for i in range(rank, len(sel), world)]; pos_of = {g: k for k, g in enumerate(sel)}
    ac = arlib.autocast_ctx(arlib.v3_cfg(16, 16))
    Mloc = np.lib.format.open_memmap(Path(a.out) / "loc.npy", mode="r+")       # (n, arms, K, 4)
    Mdp = np.lib.format.open_memmap(Path(a.out) / "dp.npy", mode="r+")         # (n, arms, K)
    Mp = np.lib.format.open_memmap(Path(a.out) / "p_obj.npy", mode="r+")       # (n, K, D) p @ 자기 봉우리 (full)
    Mpk = np.lib.format.open_memmap(Path(a.out) / "p_peak.npy", mode="r+")     # (n, K, 4) 봉우리 y, x, max, med (마지막 관측 템플릿)
    Mh = np.lib.format.open_memmap(Path(a.out) / "h_obj.npy", mode="r+")       # (n, K, D) h @ 진실 칸
    Mz = np.lib.format.open_memmap(Path(a.out) / "z_last.npy", mode="r+")      # (n, D) LN(z) @ 문맥 마지막 물체 칸
    Mhl = np.lib.format.open_memmap(Path(a.out) / "h_last.npy", mode="r+")     # (n, D) LN(h) @ 문맥 마지막 물체 칸 (외형 등변성 검사)
    t0 = time.time()
    for n_done, i in enumerate(mine):
        x = ds64.clip(i)[None, :, SPLIT - C:SPLIT + 2 * K].to(dev, bundle.dtype)
        xd = ds64.clip(donor[i])[None, :, SPLIT - C:SPLIT].to(dev, bundle.dtype)
        with torch.inference_mode(), ac():
            h = F.layer_norm(bundle.target_encoder(x), (D,)).float().reshape(TC + K, S_TOK, D)
            ci = idx_range(0, TC, 1, dev); ti = idx_range(TC, K, 1, dev)
            z = bundle.context_encoder(x, masks=[ci])
            zd = bundle.context_encoder(torch.cat([xd, x[:, :, C:]], 2), masks=[ci])        # donor 문맥
            P = {"full": pred(z, ci, ti, mask_index=0)}
            for m in (1, 2, 4):
                P[f"pred_last{m}"] = pred(z[:, -m * S_TOK:], idx_range(TC - m, m, 1, dev), ti, mask_index=0)
            for m in (2, 4):
                ze = bundle.context_encoder(x[:, :, C - 2 * m:C])                              # 마지막 2m 프레임만 (위치 0 부터)
                P[f"enc_last{m}"] = pred(ze, idx_range(TC - m, m, 1, dev), ti, mask_index=0)
            zs = torch.cat([zd[:, :(TC - 2) * S_TOK], z[:, (TC - 2) * S_TOK:]], 1)
            P["hist_swap"] = pred(zs, ci, ti, mask_index=0)
            P = {k_: v.float().reshape(K, S_TOK, D) for k_, v in P.items()}
            lc = cellxy(recs[i].raw, SPLIT - 2)
            app = h[TC - 1, lc[1] * G + lc[0]]
            zl = F.layer_norm(z.float(), (D,)).reshape(TC, S_TOK, D)[TC - 1][win_idx(*lc)].mean(0)
            row = pos_of[i]
            for t in range(K):
                tc = cellxy(recs[i].raw, SPLIT + 2 * t)
                tpl = h[TC + t, tc[1] * G + tc[0]][None]
                for ai, arm in enumerate(ARMS):
                    Mloc[row, ai, t] = locate(P[arm][t][None], tpl).cpu().numpy()[0]
                    Mdp[row, ai, t] = ((P[arm][t] - P["full"][t]).abs().mean() / P["full"][t].abs().mean()).item()
                pk = locate(P["full"][t][None], app[None]).cpu().numpy()[0]; Mpk[row, t] = pk
                Mp[row, t] = P["full"][t][win_idx(int(pk[1]), int(pk[0]))].mean(0).half().cpu().numpy()
                Mh[row, t] = h[TC + t][win_idx(*tc)].mean(0).half().cpu().numpy()
            Mz[row] = zl.half().cpu().numpy()
            Mhl[row] = h[TC - 1][win_idx(*lc)].mean(0).half().cpu().numpy()
        if rank == 0 and n_done % 50 == 0:
            el = time.time() - t0; print(f"[rank0] {n_done+1}/{len(mine)} {el/60:.1f} 분 남은 {el/(n_done+1)*(len(mine)-n_done-1)/60:.1f} 분", flush=True)
    for v in (Mloc, Mdp, Mp, Mpk, Mh, Mz, Mhl): v.flush()
    print(f"[rank{rank}] 끝", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DEFAULT); ap.add_argument("--gpus", type=int, default=torch.cuda.device_count())
    ap.add_argument("--every", type=int, default=2, help="외형 복사 중 몇 개마다 하나 (궤적은 전부 유지)"); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(); Path(a.out).mkdir(parents=True, exist_ok=True)
    ds = arlib.WMADataset(dict(arlib.v3_cfg(0, 64, split=0)))
    R = ds.records
    traj = lambda r: (r.raw["scenario"], r.raw["primary"], r.raw["secondary"])
    pool = [i for i, r in enumerate(R) if r.plausible == "1" and r.raw["scenario"] in FREE]
    sel = pool[:: a.every]
    if a.limit: sel = sel[:: max(1, len(sel) // a.limit)][: a.limit]
    rng = np.random.default_rng(0); donor = {}
    by_law = {}
    for i in pool: by_law.setdefault(R[i].raw["scenario"], []).append(i)       # donor 는 전체 풀에서 (같은 법칙 · 다른 궤적)
    for i in sel:
        cand = [j for j in by_law[R[i].raw["scenario"]] if traj(R[j]) != traj(R[i])]
        donor[i] = int(rng.choice(cand))
    n = len(sel)
    for nm, sh, dt in (("loc", (n, len(ARMS), K, 4), np.float32), ("dp", (n, len(ARMS), K), np.float32), ("p_obj", (n, K, D), np.float16),
                       ("p_peak", (n, K, 4), np.float32), ("h_obj", (n, K, D), np.float16), ("z_last", (n, D), np.float16), ("h_last", (n, D), np.float16)):
        x = np.lib.format.open_memmap(Path(a.out) / f"{nm}.npy", mode="w+", dtype=dt, shape=sh); del x
    json.dump(dict(n=n, arms=ARMS, video_ids=[R[i].video_id for i in sel], donor=[R[donor[i]].video_id for i in sel],
                   traj=[list(traj(R[i])) for i in sel]), open(Path(a.out) / "meta.json", "w"))
    mp.spawn(_worker, args=(a.gpus, a, sel, donor), nprocs=a.gpus, join=True); print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
