#!/usr/bin/env python3
"""M4 — 조회 반경 기전 (v3, stride 1 · 2 · 4): P2 의 "절벽 거리 ≈ 2.6 칸 + 1.9 걸음" 이 attention 의 공간 도달 범위에서 오나.

측정 설계는 RollOutV2 locality 분석 (`z_research/scripts/analysis/rollout2_predictor_locality.py`) 을 따랐지만, 그 수치는 근거로 쓰지 않는다
(사용자 지시 "RollOutV2가 아니라 V3로"). hook 은 2026-09-25 에 `predictor_hooks.py` 로 옮겼다 (동작 동일, v3 파이프라인이 V2 스크립트에 기대지 않게).
여기서는 같은 측정을 stride 별로 한다. 사전 예측: 반경이 칸 단위로 고정이면 (stride 무관), 세 stride 의 곡선이 거리 (칸) 축에서 겹친다.
⚠️ 적대 검증 (VERIFY_0925, M4 문서 정정판) 에서 미충족. 알려진 한계 (재추출 전까지 그대로 둔다, 동작 변경 없음):
  - obj_traj 에 화면 밖 문맥 튜블릿 (cell_at 이 가장자리 칸으로 자름) 이 들어간다 → s2·s4 ×균등이 약 30–35 % 낮게 잡힌다 (s1 은 0 개).
  - disp 는 Euclid, knockout 반경 r 은 Chebyshev.  knockout 팔의 argmax 는 저장 안 함 (적중 boolean 만).
  - last_obj 창은 자기 기둥을 빼지 않는다.  attention 은 head 평균만 저장한다.
  - repro_check 는 clip 0 · s2 · kmask=None 만.  all-True kmask 검사와 ring 귀무는 m4_ring_null_cpu.py (CPU 부분집합).

팔: s1_C16 (raw [16,64) 문맥 8 / 미래 16), s2_C16 (raw [0,64) 8 / 8), s4_C8 (raw [0,64) 4 / 4).
exp1 (attention, head 평균, 층별): query = 슬롯 t 의 진실 물체 칸 미래 토큰. key 질량:
  obj_traj = 물체의 문맥 튜블릿별 3×3 칸 (query 자기 3×3 기둥과 겹치는 key 제외),  last_obj = 문맥 마지막 튜블릿 물체 3×3,
  self_col = query 칸 3×3 × 문맥 전 튜블릿,  ctx = 문맥 전체.  + 문맥 key 까지의 attention 가중 공간 거리 (칸).
exp2 (knockout): 미래 query → 문맥 key 를 Chebyshev 반경 r ∈ {1,2,3,5,8} 칸 밖이면 전 층에서 막는다 (미래↔미래 · 문맥↔문맥 은 연다).
  적중 = argmax cos(p_t, h_t 진실 칸 토큰) 이 진실 1 칸 안.  r = ∞ 는 원래.
재현 검사: hook 을 단 predictor (막음 없음) 의 p 와 원래 p 의 상대 L1 (첫 clip) 을 로그에 남긴다.

  python m4_locality_stride_v3.py --gpus 8        /   --gpus 1 --limit 4 --out /tmp/m4_smoke
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
from h4_single_query_v3 import idx_range  # noqa: E402
from p2_stride_sweep_v3 import arm_frames, cell_at, FREE  # noqa: E402
from h23_extract_v3 import arr  # noqa: E402
from predictor_hooks import Hook, patch_attention  # noqa: E402   (2026-09-25: rollout2_predictor_locality 에서 복사, 동작 동일)

G, S_TOK, D = 16, 256, 1280
ARMS = [("s1_C16", 1, 16), ("s2_C16", 2, 0), ("s4_C8", 4, 0)]
RADII = [1, 2, 3, 5, 8]
KMAX, NL = 16, 12
OUT_DEFAULT = "/data2/local_datasets/world/world_analysis/cache/auto_research/v3_m4"


def win(cx, cy):
    return [(yy, xx) for yy in range(max(0, cy - 1), min(G, cy + 2)) for xx in range(max(0, cx - 1), min(G, cx + 2))]


def knock_mask(nc, nf, r, dev):
    N = (nc + nf) * S_TOK; i = torch.arange(N, device=dev); hh, ww = (i % S_TOK) // G, i % G; fut = i >= nc * S_TOK
    cheb = torch.maximum((hh[:, None] - hh[None]).abs(), (ww[:, None] - ww[None]).abs())
    allow = torch.ones(N, N, dtype=torch.bool, device=dev); allow[fut[:, None] & (~fut)[None] & (cheb > r)] = False
    return allow


def _worker(rank, world, a, sel):
    dev = torch.device(f"cuda:{rank}"); torch.cuda.set_device(dev)
    ds64 = arlib.WMADataset(dict(arlib.v3_cfg(0, 64, split=0))); recs = ds64.records
    bundle = arlib.build_bundle(arlib.v3_cfg(4, 60), dev, window=64); pred = bundle.predictor
    ac = arlib.autocast_ctx(arlib.v3_cfg(16, 16))
    hook = Hook(); patch_attention(pred, hook)
    MA = np.lib.format.open_memmap(Path(a.out) / "attn.npy", mode="r+")    # (n, arm, KMAX, NL, 5) obj_traj, last_obj, self_col, ctx, 가중 거리
    MK = np.lib.format.open_memmap(Path(a.out) / "knock.npy", mode="r+")   # (n, arm, KMAX, len(RADII)+1) 적중
    MD = np.lib.format.open_memmap(Path(a.out) / "disp.npy", mode="r+")    # (n, arm, KMAX) 변위 거리 (칸, 문맥 끝 → 진실)
    kms = {}
    mine = list(range(rank, len(sel), world)); t0 = time.time()
    for n_done, row in enumerate(mine):
        i = sel[row]; raw = recs[i].raw; x, y = arr(raw["px_x_by_sample"]), arr(raw["px_y_by_sample"])
        full = ds64.clip(i)[None].to(dev, bundle.dtype)
        for ai, (nm, s, r0) in enumerate(ARMS):
            fr, nc, nf = arm_frames(s, r0); xc = full[:, :, fr]
            lc = cell_at(x, y, fr[2 * nc - 2], fr[2 * nc - 1])
            ctx_cells = [cell_at(x, y, fr[2 * u], fr[2 * u + 1]) for u in range(nc)]
            tcs = [cell_at(x, y, fr[2 * (nc + t)], fr[2 * (nc + t) + 1]) for t in range(nf)]
            if lc is None or any(c is None for c in tcs): continue
            ci = idx_range(0, nc, 1, dev); ti = idx_range(nc, nf, 1, dev)
            with torch.inference_mode(), ac():
                h = F.layer_norm(bundle.target_encoder(xc), (D,)).float().reshape(nc + nf, S_TOK, D)
                z = bundle.context_encoder(xc, masks=[ci])
                hook.kmask = None; hook.store = []
                hook.rec_rows = torch.tensor([[nc * S_TOK + t * S_TOK + tcs[t][1] * G + tcs[t][0] for t in range(nf)]], device=dev)
                p0 = pred(z, ci, ti, mask_index=0).float().reshape(nf, S_TOK, D)
                A = torch.stack(hook.store, 1)[0].float()                                  # (NL, nf, N)
                hook.rec_rows = None
                ps = {}
                for r in RADII:
                    key = (nc, nf, r)
                    if key not in kms: kms[key] = knock_mask(nc, nf, r, dev)
                    hook.kmask = kms[key]; ps[r] = pred(z, ci, ti, mask_index=0).float().reshape(nf, S_TOK, D)
                hook.kmask = None
            Nc = nc * S_TOK
            ctxpos = torch.arange(Nc, device=dev); cy_, cx_ = (ctxpos % S_TOK) // G, ctxpos % G
            for t in range(nf):
                tx, ty = tcs[t]
                selfm = torch.zeros(Nc, dtype=torch.bool, device=dev)
                for (yy, xx) in win(tx, ty): selfm |= (cy_ == yy) & (cx_ == xx)
                objm = torch.zeros(Nc, dtype=torch.bool, device=dev)
                for u, c in enumerate(ctx_cells):
                    if c is None: continue
                    for (yy, xx) in win(*c): objm[u * S_TOK + yy * G + xx] = True
                lastm = torch.zeros(Nc, dtype=torch.bool, device=dev)
                for (yy, xx) in win(*lc): lastm[(nc - 1) * S_TOK + yy * G + xx] = True
                At = A[:, t, :Nc]                                                          # (NL, Nc)
                dist = torch.maximum((cy_ - ty).abs(), (cx_ - tx).abs()).float()
                MA[row, ai, t] = torch.stack([At[:, objm & ~selfm].sum(-1), At[:, lastm].sum(-1), At[:, selfm].sum(-1), At.sum(-1),
                                              (At * dist).sum(-1) / At.sum(-1).clamp_min(1e-9)], -1).cpu().numpy()
                tpl = h[nc + t, ty * G + tx]
                hits = []
                for pp in [ps[r] for r in RADII] + [p0]:
                    am = int((F.normalize(pp[t], dim=-1) @ F.normalize(tpl, dim=-1)).argmax()); ay, ax = divmod(am, G)
                    hits.append(float(max(abs(ax - tx), abs(ay - ty)) <= 1))
                MK[row, ai, t] = hits
                MD[row, ai, t] = np.hypot(tx - lc[0], ty - lc[1])
        if rank == 0 and n_done % 20 == 0:
            el = time.time() - t0; print(f"[rank0] {n_done+1}/{len(mine)} {el/60:.1f} 분 남은 {el/(n_done+1)*(len(mine)-n_done-1)/60:.1f} 분", flush=True)
    for v in (MA, MK, MD): v.flush()
    print(f"[rank{rank}] 끝", flush=True)


def repro_check(a):
    """hook 을 단 predictor (막음 없음) 가 원래 predictor 와 같은 p 를 내는가."""
    dev = torch.device("cuda:0"); ds64 = arlib.WMADataset(dict(arlib.v3_cfg(0, 64, split=0)))
    bundle = arlib.build_bundle(arlib.v3_cfg(4, 60), dev, window=64); pred = bundle.predictor; ac = arlib.autocast_ctx(arlib.v3_cfg(16, 16))
    fr, nc, nf = arm_frames(2, 0); xc = ds64.clip(0)[None].to(dev, bundle.dtype)[:, :, fr]
    ci = idx_range(0, nc, 1, dev); ti = idx_range(nc, nf, 1, dev)
    with torch.inference_mode(), ac():
        z = bundle.context_encoder(xc, masks=[ci]); p_ref = pred(z, ci, ti, mask_index=0).float()
        hook = Hook(); patch_attention(pred, hook); p_hk = pred(z, ci, ti, mask_index=0).float()
    rel = float(((p_hk - p_ref).abs().mean() / p_ref.abs().mean()).item())
    json.dump(dict(hook_vs_original_rel_l1=rel), open(Path(a.out) / "repro.json", "w")); print("재현 검사 hook vs 원래 rel L1", rel, flush=True)
    del bundle; torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DEFAULT); ap.add_argument("--gpus", type=int, default=torch.cuda.device_count())
    ap.add_argument("--every", type=int, default=4); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(); Path(a.out).mkdir(parents=True, exist_ok=True)
    ds = arlib.WMADataset(dict(arlib.v3_cfg(0, 64, split=0))); R = ds.records
    sel = [i for i, r in enumerate(R) if r.plausible == "1" and r.raw["scenario"] in FREE][:: a.every]
    if a.limit: sel = sel[:: max(1, len(sel) // a.limit)][: a.limit]
    n = len(sel)
    for nm, sh in (("attn", (n, len(ARMS), KMAX, NL, 5)), ("knock", (n, len(ARMS), KMAX, len(RADII) + 1)), ("disp", (n, len(ARMS), KMAX))):
        v = np.lib.format.open_memmap(Path(a.out) / f"{nm}.npy", mode="w+", dtype=np.float32, shape=sh); v[:] = np.nan; del v
    json.dump(dict(n=n, arms=[dict(name=nm, stride=s, raw0=r0, n_ctx=arm_frames(s, r0)[1], n_fut=arm_frames(s, r0)[2]) for nm, s, r0 in ARMS],
                   radii=RADII + ["inf"], attn_cols=["obj_traj", "last_obj", "self_col", "ctx", "wdist"],
                   video_ids=[R[i].video_id for i in sel], scenario=[R[i].raw["scenario"] for i in sel],
                   traj=[[R[i].raw["scenario"], R[i].raw["primary"], R[i].raw["secondary"]] for i in sel]), open(Path(a.out) / "meta.json", "w"))
    repro_check(a)
    mp.spawn(_worker, args=(a.gpus, a, sel), nprocs=a.gpus, join=True); print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
