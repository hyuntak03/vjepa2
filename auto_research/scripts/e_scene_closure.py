#!/usr/bin/env python3
"""장면 단위 closure (실패 3: 자기 예측은 관측 대신 못 쓰는가) — 인공 팬 (K400 val) 위에서, 출처 판독 (2026-09-26).

P3 · P3b (v3 물체) 의 세 팔을 토큰 장 전체 · 실영상 프레임으로 옮긴다. 문맥 = 블록 0..7 (16 장), 미래 블록 8..15.
  os    : 한 번에 8 블록 (기준, e_scene_reach 의 base 와 같다)
  ar_z  : 걸음 j 마다 문맥 = **진짜** 프레임 0..16+2j 의 encoder z (블록 0..7+j) → 블록 8+j 예측 (탐침: 미래를 본다)
  ar_pA : 걸음 j 마다 문맥 = 진짜 블록 0..7 + **자기 예측** 블록 8..7+j (어댑터 A 로 z 공간에 놓음) → 블록 8+j 예측
  어댑터 A: 채널별 affine — p (LN(h) 공간) → z 공간: A(p) = p · std_c + mean_c, mean_c · std_c 는 그 clip 문맥 z 토큰의 채널 통계 (P3 의 A 와 같은 뜻).
판독: 블록 8+j 의 토큰 s → argmax_r cos(·, hc_L[r]) (hc_L = 문맥 16 장만 본 target 의 마지막 튜블릿) 가 출처 u_j(s) 1 칸 안.
저장: am (n, P, 3, 8, 256) int16 · ct (n, P, 3, 8, 256) fp16 cos(진짜 출처) · cc (…) cos(복사). P = release · Ariel · pv1 (+SCENE_EXTRA).
분석 (e_scene_closure_analyze.py): 걸음 j 별 · 변위 d 별 hit 를 세 팔 비교. ar_pA 가 os 보다 낫지 않고 ar_z 가 낫다 = 자기 예측은 관측 대신 못 쓴다.

  python e_scene_closure.py --gpus 8 --n 400 --out <dir>
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
from h4_single_query_v3 import idx_range  # noqa: E402
import e_scene_reach as ES  # noqa: E402

S_TOK, D, NF, TC, G = 256, 1280, 32, 8, 16
ARMS = ["os", "ar_z", "ar_pA"]
PREDS = ["R", "A", "P"] + [kv.split("=", 1)[0] for kv in os.environ.get("SCENE_EXTRA", "").split(",") if "=" in kv]


def _worker(rank, world, a, items):
    dev = torch.device(f"cuda:{rank}"); torch.cuda.set_device(dev)
    cfg = arlib.v3_cfg(16, 16); os.environ.pop("ARLIB_PRED_CKPT", None)
    bundle = arlib.build_bundle(cfg, dev, window=NF); ac = arlib.autocast_ctx(cfg)
    preds = {"R": bundle.predictor}
    preds["A"], _ = ES.load_predictor(cfg["model"], ES.ARIEL, dev, bundle.dtype, NF); preds["P"], _ = ES.load_predictor(cfg["model"], ES.PV1, dev, bundle.dtype, NF)
    for _k, _p in ES.EXTRA: preds[_k], _ = ES.load_predictor(cfg["model"], _p, dev, bundle.dtype, NF)
    O = {nm: np.lib.format.open_memmap(Path(a.out) / f"{nm}.npy", mode="r+") for nm in ("am", "ct", "cc")}
    mine = list(range(rank, len(items), world)); t0 = time.time()
    for n_done, row in enumerate(mine):
        it = items[row]; x = ES.load_pan(it)[None].to(dev, bundle.dtype); src, _ = ES.pan_src(it["v"], it["dir"])
        srcv = torch.as_tensor(src, device=dev)
        with torch.inference_mode(), ac():
            hc = F.layer_norm(bundle.target_encoder(x[:, :, :16]), (D,)).float().reshape(8, S_TOK, D); hcL = hc[7]
            z_all = [bundle.context_encoder(x[:, :, :16 + 2 * j], masks=[idx_range(0, TC + j, 1, dev)]) for j in range(0, 8)]   # z_all[j]: 블록 0..7+j
        z0 = z_all[0]; mu = z0.float().mean((0, 1)); sd = z0.float().std((0, 1)) + 1e-6
        Hn = F.normalize(hcL, dim=-1)

        def readout(p8, arm_i, pi):
            cs = F.normalize(p8.float(), dim=-1) @ Hn.T                       # (8, 256, 256)
            am = cs.argmax(-1); ct = torch.gather(cs, 2, srcv.clamp(min=0)[..., None])[..., 0].masked_fill(srcv < 0, float("nan"))
            O["am"][row, pi, arm_i] = am.short().cpu().numpy(); O["ct"][row, pi, arm_i] = ct.half().cpu().numpy(); O["cc"][row, pi, arm_i] = cs.diagonal(dim1=1, dim2=2).half().cpu().numpy()

        for pi, pk in enumerate(PREDS):
            pred = preds[pk]
            with torch.inference_mode(), ac():
                p_os = pred(z0, idx_range(0, TC, 1, dev), idx_range(TC, TC, 1, dev), mask_index=0).float().reshape(8, S_TOK, D)
                readout(p_os, 0, pi)
                # ar_z: 진짜 관측 되먹임
                outs = []
                for j in range(8):
                    pj = pred(z_all[j], idx_range(0, TC + j, 1, dev), idx_range(TC + j, 1, 1, dev), mask_index=0).float().reshape(1, S_TOK, D)
                    outs.append(pj[0])
                readout(torch.stack(outs), 1, pi)
                # ar_pA: 자기 예측 되먹임 (어댑터 A)
                ctx = z0; outs = []
                for j in range(8):
                    pj = pred(ctx, idx_range(0, TC + j, 1, dev), idx_range(TC + j, 1, 1, dev), mask_index=0).float().reshape(1, S_TOK, D)
                    outs.append(pj[0]); ctx = torch.cat([ctx, (pj * sd + mu).to(ctx.dtype)], 1)
                readout(torch.stack(outs), 2, pi)
        if rank == 0 and n_done % 10 == 0:
            el = time.time() - t0; print(f"[rank0] {n_done+1}/{len(mine)} {el/60:.1f} 분 남은 {el/(n_done+1)*(len(mine)-n_done-1)/60:.1f} 분", flush=True)
    for v_ in O.values(): v_.flush()
    print(f"[rank{rank}] 끝", flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--gpus", type=int, default=torch.cuda.device_count())
    ap.add_argument("--n", type=int, default=400); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(); Path(a.out).mkdir(parents=True, exist_ok=True)
    items = ES.pan_items(a.limit or a.n); n = len(items)
    for nm, dt in (("am", np.int16), ("ct", np.float16), ("cc", np.float16)):
        m = np.lib.format.open_memmap(Path(a.out) / f"{nm}.npy", mode="w+", dtype=dt, shape=(n, len(PREDS), len(ARMS), 8, S_TOK)); m[:] = (-1 if dt == np.int16 else np.nan); del m
    json.dump(dict(n=n, items=items, preds=PREDS, arms=ARMS, speeds=ES.SPEEDS), open(Path(a.out) / "meta.json", "w"))
    print("clips", n, "preds", PREDS, flush=True); torch.set_num_threads(4)
    mp.spawn(_worker, args=(a.gpus, a, items), nprocs=a.gpus, join=True); print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
