#!/usr/bin/env python3
"""P2 — stride sweep (RollOut_v3): predictor 가 물체를 놓는 한계는 **튜블릿 수** 인가, **물리 시간·거리** 인가.

RollOutV3 (attn 자) 는 "튜블릿 수가 정한다 (속도 1.6 배에도 절벽 t8 고정)", auto_research H23 (자 없는 읽기, null 보정) 은
"이동 거리가 정한다 (−1.43/칸, 슬롯 +0.07 n.s.)" 로 갈렸다. 속도 1.6 배는 좁다. 프레임 stride s 를 바꾸면 튜블릿 하나의
물리 시간·이동 거리가 s 배가 되고 튜블릿 수는 그대로다 → 둘을 크게 떼어 놓는다.

팔 (분할 = raw f32, 문맥은 분할 앞, 미래는 분할 뒤; 같은 raw 구간을 stride 만 바꿔 본다):
  s1_C32  raw [0,64) stride 1 — 문맥 16 튜블릿, 미래 16 튜블릿 (튜블릿 = 2 raw 프레임)
  s2_C16  raw [0,64) stride 2 — 문맥  8,       미래  8      (튜블릿 = 4 raw)
  s4_C8   raw [0,64) stride 4 — 문맥  4,       미래  4      (튜블릿 = 8 raw)
  s1_C16  raw [16,64) stride 1 — 문맥 8, 미래 16   (H23 창과 같음, 연속성 확인)
  s2_C8   raw [16,64) stride 2 — 문맥 4, 미래 8
예측: 튜블릿 수 한계면 s1 은 t≈8 (raw ≈48) 에서 무너지고 s2·s4 는 raw 64 까지 안 무너진다.
      물리 (시간·거리) 한계면 세 팔이 **같은 raw 프레임** 에서 무너진다 (s2 는 t≈4, s4 는 t≈2).
      둘 다 아니면 (예: 문맥 길이) 팔마다 문맥 튜블릿 수에 따라 달라진다 (s1_C16 대 s2_C8 로 따로 본다).
거리 대 시간은 팔 안의 clip 속도 차이 (같은 법칙 안에서 궤적마다 다름) 로 분리한다 — 분석에서 회귀.

측정 (미래 슬롯 t, 표준 표적 h = LN(target_encoder(창 전체))):
  loc_tru  p_t 에서 템플릿 = h_t 의 진실 칸 토큰 → argmax (y, x), maxcos, median          (null 은 분석에서 다른 궤적의 진실 칸)
  loc_app  p_t 에서 템플릿 = 문맥 마지막 튜블릿의 LN(h) 물체 토큰 (마지막으로 본 외형; 미래 진실 안 씀)
  loc_appc p_t 에서 **인과** 외형 템플릿 (target encoder 를 문맥 프레임에만 돌린 마지막 튜블릿 물체 토큰 — 양방향 번짐 없음, 리뷰 B5)
  loc_apph h_t 에서 같은 외형 템플릿 (encoder 천장: 진짜 미래 장면에서 외형 템플릿이 물체를 찾는가)
  l1p / l1c  전 토큰 mean|p_t − h_t| / mean|h_last − h_t| (복사 기준선, h_last = 문맥 마지막 튜블릿의 표준 h)
  l1po / l1co  같은 것을 진실 칸 3×3 창에서만
수치 경로는 표준 채점과 같다 (fp32 + autocast fp16, mask_index 0).

  sub.sh -J ar_p2 -o auto_research/logs/p2_%j.out --export=ALL,CMD="python auto_research/scripts/p2_stride_sweep_v3.py --gpus 8" run8.sbatch
  python auto_research/scripts/p2_stride_sweep_v3.py --gpus 1 --limit 4 --out /tmp/p2_smoke      # 배관 점검
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
from h23_extract_v3 import arr, win_idx  # noqa: E402
from h4_single_query_v3 import idx_range, locate  # noqa: E402

G, S_TOK, D, SPLIT, CELL = 16, 256, 1280, 32, 18.0
ARMS_ALL = {"s1_C32": (1, 0, 64), "s2_C16": (2, 0, 64), "s4_C8": (4, 0, 64), "s1_C16": (1, 16, 64), "s2_C8": (2, 16, 64),
            "s3_C10": (3, 2, 62),
            # SC1 (2026-09-25, oral 방향 sanity check): 분할점 raw 16 — 미래를 길게 (학습 지평 8 튜블릿을 넘어서까지) 물어 "멈춤이 학습 지평에서 오나" 를 본다
            "s2_S16": (2, 0, 64, 16), "s1_S16": (1, 8, 64, 16)}   # (stride, raw 시작, raw 끝) — s3: raw 2..59 → 문맥 5 / 미래 5 튜블릿 (pv1 · v11_postft 학습 stride, 리뷰 B7)
import os as _os
ARMS = [(k, *ARMS_ALL[k][:2]) for k in (_os.environ.get("P2_ARMS", "s1_C32,s2_C16,s4_C8,s1_C16,s2_C8").split(","))]   # (이름, stride, raw 시작)
KMAX = 24                                                                   # s1_S16 은 미래 24 튜블릿
OUT_DEFAULT = "/data2/local_datasets/world/world_analysis/cache/auto_research/v3_p2"
FREE = ("flat_v", "flat_a", "flat_d", "ramp_a", "ramp_d", "arc")


def arm_frames(s, r0, name=None):
    """창의 raw 프레임 번호, 문맥 튜블릿 수, 미래 튜블릿 수. name 이 있으면 ARMS_ALL[name] 규격 (끝 · 분할점) 을 쓴다."""
    spec = ARMS_ALL[name] if name in ARMS_ALL else next((v for v in ARMS_ALL.values() if v[0] == s and v[1] == r0), (s, r0, 64, SPLIT))
    end = spec[2]; split = spec[3] if len(spec) > 3 else SPLIT
    fr = list(range(r0, end, s)); nc = sum(f < split for f in fr)
    assert nc % 2 == 0 and len(fr) % 2 == 0
    return fr, nc // 2, (len(fr) - nc) // 2


def cell_at(x, y, fa, fb):
    cx, cy = (x[fa] + x[fb]) / 2, (y[fa] + y[fb]) / 2
    if not (np.isfinite(cx) and np.isfinite(cy)): return None
    return int(np.clip(cx // CELL, 0, G - 1)), int(np.clip(cy // CELL, 0, G - 1))


def _worker(rank, world, a, sel):
    dev = torch.device(f"cuda:{rank}"); torch.cuda.set_device(dev)
    ds64 = arlib.WMADataset(dict(arlib.v3_cfg(0, 64, split=0)))
    bundle = arlib.build_bundle(arlib.v3_cfg(4, 60), dev, window=64); pred = bundle.predictor; recs = ds64.records
    ac = arlib.autocast_ctx(arlib.v3_cfg(16, 16))
    mine = list(range(rank, len(sel), world))
    M = {nm: np.lib.format.open_memmap(Path(a.out) / f"{nm}.npy", mode="r+") for nm in ("loc_tru", "loc_app", "loc_apph", "l1", "loc_appc")}
    t0 = time.time()
    for n_done, row in enumerate(mine):
        i = sel[row]; raw = recs[i].raw
        x, y = arr(raw["px_x_by_sample"]), arr(raw["px_y_by_sample"])
        full = ds64.clip(i)[None].to(dev, bundle.dtype)                                   # (1, 3, 64, H, W)
        for ai, (nm, s, r0) in enumerate(ARMS):
            fr, nc, nf = arm_frames(s, r0, nm)
            xc = full[:, :, fr]
            with torch.inference_mode(), ac():
                h = F.layer_norm(bundle.target_encoder(xc), (D,)).float().reshape(nc + nf, S_TOK, D)
                ci = idx_range(0, nc, 1, dev); ti = idx_range(nc, nf, 1, dev)
                z = bundle.context_encoder(xc, masks=[ci])
                p = pred(z, ci, ti, mask_index=0).float().reshape(nf, S_TOK, D)
                hcx = F.layer_norm(bundle.target_encoder(xc[:, :, :2 * nc]), (D,)).float().reshape(nc, S_TOK, D)   # 인과 표적 (문맥 프레임만)
            lc = cell_at(x, y, fr[2 * nc - 2], fr[2 * nc - 1])
            if lc is None: continue
            hl = h[nc - 1]; app = hl[lc[1] * G + lc[0]]; appc = hcx[nc - 1, lc[1] * G + lc[0]]   # 인과 외형 템플릿 (미래 번짐 없음)
            for t in range(nf):
                tc = cell_at(x, y, fr[2 * (nc + t)], fr[2 * (nc + t) + 1])
                ht = h[nc + t]
                M["loc_app"][row, ai, t] = locate(p[t][None], app[None]).cpu().numpy()[0]
                M["loc_appc"][row, ai, t] = locate(p[t][None], appc[None]).cpu().numpy()[0]
                M["loc_apph"][row, ai, t] = locate(ht[None], app[None]).cpu().numpy()[0]
                w = win_idx(*tc) if tc is not None else None
                l1 = [(p[t] - ht).abs().mean().item(), (hl - ht).abs().mean().item(), np.nan, np.nan]
                if tc is not None:
                    M["loc_tru"][row, ai, t] = locate(p[t][None], ht[tc[1] * G + tc[0]][None]).cpu().numpy()[0]
                    l1[2] = (p[t][w] - ht[w]).abs().mean().item(); l1[3] = (hl[w] - ht[w]).abs().mean().item()
                M["l1"][row, ai, t] = l1
        if rank == 0 and n_done % 25 == 0:
            el = time.time() - t0; print(f"[rank0] {n_done+1}/{len(mine)} {el/60:.1f} 분 남은 {el/(n_done+1)*(len(mine)-n_done-1)/60:.1f} 분", flush=True)
    for v in M.values(): v.flush()
    print(f"[rank{rank}] 끝", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DEFAULT); ap.add_argument("--gpus", type=int, default=torch.cuda.device_count())
    ap.add_argument("--every", type=int, default=2, help="외형 복사 중 몇 개마다 하나 (궤적은 전부 유지)"); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(); Path(a.out).mkdir(parents=True, exist_ok=True)
    ds = arlib.WMADataset(dict(arlib.v3_cfg(0, 64, split=0))); R = ds.records
    sel = [i for i, r in enumerate(R) if r.plausible == "1" and r.raw["scenario"] in FREE][:: a.every]
    if a.limit: sel = sel[:: max(1, len(sel) // a.limit)][: a.limit]
    n = len(sel)
    for nm, sh in (("loc_tru", (n, len(ARMS), KMAX, 4)), ("loc_app", (n, len(ARMS), KMAX, 4)), ("loc_apph", (n, len(ARMS), KMAX, 4)), ("loc_appc", (n, len(ARMS), KMAX, 4)),
                   ("l1", (n, len(ARMS), KMAX, 4))):
        v = np.lib.format.open_memmap(Path(a.out) / f"{nm}.npy", mode="w+", dtype=np.float32, shape=sh); v[:] = np.nan; del v
    json.dump(dict(n=n, arms=[dict(name=nm, stride=s, raw0=r0, frames=arm_frames(s, r0, nm)[0], n_ctx=arm_frames(s, r0, nm)[1], n_fut=arm_frames(s, r0, nm)[2])
                              for nm, s, r0 in ARMS],
                   video_ids=[R[i].video_id for i in sel], scenario=[R[i].raw["scenario"] for i in sel],
                   traj=[[R[i].raw["scenario"], R[i].raw["primary"], R[i].raw["secondary"]] for i in sel]),
              open(Path(a.out) / "meta.json", "w"))
    print("clips", n, flush=True)
    mp.spawn(_worker, args=(a.gpus, a, sel), nprocs=a.gpus, join=True); print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
