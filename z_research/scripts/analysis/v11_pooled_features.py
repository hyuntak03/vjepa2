#!/usr/bin/env python3
"""IntPhysGen v11 — shape · color probing 용 **튜블릿별 평균 풀링 특징**을 온라인으로 뽑는다 (visible · 늦은 가림).

대상: v11 `index_probe.csv` 의 **가능 변이 · 물체 있는 clip** (`probe_type=obj`) — visible 3 조건 (k=0) + late 가림 3 조건 (k=1..4).
  CLAUDE.md §1-4: probing 은 가능 변이만. 라벨 = `shape_pre`, `color_pre` (가능 변이는 pre = post).
창: 표준 v11 — raw 100 프레임 stride 3 → 32 샘플, 문맥 16 샘플 (8 튜블릿), 미래 16 샘플 (8 튜블릿).

저장 (튜블릿마다 256 토큰을 평균, fp16):
  z  (n, 8, D)   context encoder, **문맥 16 샘플만** (`ctx_masked` = 미래 토큰을 transformer 전에 떨군 것 = predictor 의 입력), LN
  p  (n, 8, D)   predictor 출력, 미래 8 튜블릿 (LN(target) 공간 추정치 — 다시 LN 하지 않는다, forward.py `_ln` 설명)
  h  (n, 16, D)  target encoder, 창 전체 32 샘플, LN — **대조군** (p 가 맞추도록 학습된 공간)
  평균 풀링 (T·N, D) → (1, D) 은 튜블릿 평균들의 평균과 같다 (튜블릿마다 N=256 로 같다). 가려진 튜블릿만 등 부분 풀링은 이 값으로 한다.
  hidden (n, 32)  가려진 샘플 (metadata hidden_start..hidden_end). late k 는 샘플 16−k .. 16+k−1

  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P z_research/scripts/analysis/v11_pooled_features.py --limit 12 --gpus 1     # 배관 점검
  $P z_research/scripts/analysis/v11_pooled_features.py --gpus 8                # → cache/v11_pooled_vith/
"""
from __future__ import annotations
import argparse, csv, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "z_research/scripts/analysis"))
from rollout3_window_readout import MODEL, prefetch            # noqa: E402  (모델 정의를 복제하지 않는다)
import v11_readout_online as vro                                # noqa: E402  (clip 선택 · 가림 샘플 규약 공유)

OUT = Path(os.environ.get("V11P_OUT", "/local_datasets/world/world_analysis/cache/v11_pooled_vith"))
TMP = Path(os.environ.get("V11P_TMP", "/dev/shm/v11_pooled"))
NS, CS, S, D = 32, 16, 256, 1280
TC, TP = CS // 2, (NS - CS) // 2
KEEP = ["video_id", "block_id", "variant", "condition", "motion", "violation_type", "sym_k", "occ_timing",
        "shape_pre", "color_pre", "env"]


def _worker(rank, world, args, n):
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    from analysis.intphys2.model import build_from_config
    from evals.world_model_analysis.data import WMADataset
    from evals.analysis_vlm.occlusion_identity.forward import extract_batch
    import decord  # noqa: F401

    bundle = build_from_config({**MODEL, "window_size": NS}, dev)
    DATA = {**vro.DATA, "root": str(TMP)}
    ds = WMADataset(dict(data=DATA, model={**MODEL, "window_size": NS}, features={"cache_dir": "/tmp"}, surprise={}))
    bs = max(1, min(args.max_bs, args.token_budget // ((NS // 2) * S)))
    mm = {r: np.lib.format.open_memmap(TMP / f"{r}.npy", mode="r+") for r in ("z", "p", "h")}
    idx = list(range(rank, n, world)); pool = ThreadPoolExecutor(max_workers=args.workers)
    t0, done = time.time(), 0
    with torch.no_grad():
        for ids, clips in prefetch(ds, idx, bs, pool):
            B = clips.size(0); ids = np.asarray(ids)
            o = extract_batch(clips, bundle, [{"base": "ctx_masked"}, {"base": "predictor"}, {"base": "target"}],
                              context_length=CS, mask_index=0, out_dtype=torch.float32)
            for r, base, nt in (("z", "ctx_masked", TC), ("p", "predictor", TP), ("h", "target", NS // 2)):
                x = o[base].reshape(B, nt, S, D).mean(2)                          # 튜블릿별 평균 풀링
                mm[r][ids] = x.numpy().astype(np.float16)
            done += B
            if rank == 0 and done % (bs * 10) < bs:
                el = time.time() - t0; frac = done / len(idx)
                print(f"    bs={bs} {done*world:,}/{n:,} clip  {el:.0f}s  남은 {el*(1-frac)/max(frac,1e-9)/60:.1f}분  "
                      f"최대 VRAM {torch.cuda.max_memory_allocated(dev)/2**30:.1f} GiB", flush=True)
    for v in mm.values():
        v.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=int, default=torch.cuda.device_count() or 1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--token-budget", type=int, default=131072)
    ap.add_argument("--max-bs", type=int, default=32)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()

    rows, L, inf, hid, obj, tdir = vro.select("late")
    keep = np.where(obj)[0]                                                     # 물체 있는 clip 만
    if a.limit:
        keep = np.array(sorted({i for c in vro.CONDS for i in [j for j in keep if rows[j]["condition"] == c][:max(1, a.limit // 6)]}))
    rows = [rows[i] for i in keep]; hid = hid[keep]; L = L[keep]; tdir = tdir[keep]
    n = len(rows)
    print(f"[data] {n:,} clip (가능 · 물체 있음) · 조건 {sorted({r['condition'] for r in rows})}"
          + (f"  ⚠️ --limit {a.limit}" if a.limit else ""), flush=True)
    TMP.mkdir(parents=True, exist_ok=True)
    with (TMP / "index.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    for r, nt in (("z", TC), ("p", TP), ("h", NS // 2)):
        np.lib.format.open_memmap(TMP / f"{r}.npy", mode="w+", dtype=np.float16, shape=(n, nt, D))
    t0 = time.time()
    mp.spawn(_worker, args=(a.gpus, a, n), nprocs=a.gpus, join=True)
    print(f"[extract] 끝 {(time.time()-t0)/60:.1f}분", flush=True)

    out = OUT / ("_smoke" if a.limit else ""); out.mkdir(parents=True, exist_ok=True)
    for r in ("z", "p", "h"):
        np.save(out / f"{r}.npy", np.load(TMP / f"{r}.npy"))
    np.savez(out / "meta.npz", **{k: np.array([r[k] for r in rows]) for k in KEEP},
             hidden=hid, truth=L, travel_dir=tdir)
    (out / "README.json").write_text(json.dumps(dict(
        n_clip=n, z="context encoder on 16 context samples (ctx_masked), LN, per-tubelet mean (n,8,D)",
        p="predictor output for 8 future tubelets (LN target space), per-tubelet mean (n,8,D)",
        h="target encoder on all 32 samples, LN, per-tubelet mean (n,16,D)",
        hidden="(n,32) hidden samples from metadata", script="z_research/scripts/analysis/v11_pooled_features.py"), indent=1))
    print(f"→ {out}", flush=True)


if __name__ == "__main__":
    main()
