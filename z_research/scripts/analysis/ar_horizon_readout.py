"""AR rollout 의 운동 감쇠가 'rollout 몇 걸음째' 에 묶였나 '절대 시점' 에 묶였나 — 문맥 길이 C 를 바꿔 rollout (2026-10-12).

데이터: VOB_readout_primitive 의 test 절반 (자 학습 split 과 같은 block 분할) 중 물체가 움직이는 clip.
모델: AR (블록별 LN(target_encoder) 문맥 C 블록 → rollout K = 16 − C 블록). C ∈ {4, 6, 8, 10, 12}. 인코딩 한 번, rollout 다섯 번.
읽기: identity_vobprim_<name>/p/readout.pt (C = 8 rollout 으로 배운 p 자) 로 rollout 블록마다 위치 · 57 확률.
분석: 운동 방향 속도 = 이웃 rollout 블록의 읽은 위치 차 / 정답 위치 차. rollout 걸음 k 와 절대 블록 t 두 축으로.

  python z_research/scripts/analysis/ar_horizon_readout.py extract --gpus 4 [--name ar40 --ckpt z_training/ariel/ar_future_1e_5/latest.pt]
  python z_research/scripts/analysis/ar_horizon_readout.py analyze [--name ar40]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
os.environ.setdefault("PRESENCE_SET", "vob_readout_primitive")
os.environ.setdefault("PRESENCE_STRIDE", "1")
os.environ.setdefault("PRESENCE_FRAMES", "/data2/local_datasets/world/world_analysis/VOB_readout_primitive")
CS = (4, 6, 8, 10, 12)
S, NB, D, NONE = 256, 16, 1280, 56
META = Path("/data2/local_datasets/world/world_analysis/VOB_readout_primitive/metadata.csv")


def out_dir(name):
    return REPO / f"z_research/VOBench_v1/readout/horizon_{name}"


def ir_module():
    from importlib.util import module_from_spec, spec_from_file_location
    spec = spec_from_file_location("ir", REPO / "z_research/scripts/analysis/rollout2_identity_readout.py")
    ir = module_from_spec(spec); spec.loader.exec_module(ir); return ir


def pick(name):
    """test 절반 · 움직이는 물체 clip 의 (index 행 번호, video_id)."""
    z = np.load(REPO / f"z_research/VOBench_v1/readout/identity_vobprim_{name}/p/preds.npz", allow_pickle=True)
    meta = {r["name"]: r for r in csv.DictReader(META.open())}
    out = []
    for i, v in enumerate(z["video_id"]):
        m = meta[str(v)]
        if z["test"][i] and m["has_object"] in ("1", "True", "true") and np.hypot(float(m["vx_cm_s"] or 0), float(m["vz_cm_s"] or 0)) > 1:
            out.append((i, str(v)))
    return out


def _worker(rank, world, name, ckpt):
    import torch
    from analysis.intphys2.model import build_from_config
    from app.vjepa_frozen.ar import block_indices, encode_blocks
    from evals.world_model_analysis.data import WMADataset
    torch.cuda.set_device(rank); torch.set_num_threads(4); dev = torch.device("cuda", rank)
    ir = ir_module(); rp = ir.rp
    model = dict(rp.MODEL, predictor_checkpoint=str(REPO / ckpt))
    bundle = build_from_config(model, dev)
    R = ir.IdReadout().to(dev).eval()
    R.load_state_dict(torch.load(REPO / f"z_research/VOBench_v1/readout/identity_vobprim_{name}/p/readout.pt", map_location=dev))
    ds = WMADataset(dict(data=rp.DATA, model=model, features={"cache_dir": "/tmp"}, surprise={}))
    items = pick(name); mine = items[rank * len(items) // world:(rank + 1) * len(items) // world]
    assert all(ds.records[i].video_id == v for i, v in mine[:5])
    out = defaultdict(list); t0 = time.time()
    with torch.no_grad():
        for k in range(0, len(mine), 16):
            ch = mine[k:k + 16]
            x = torch.stack([ds.clip(i) for i, _ in ch]).to(dev, dtype=bundle.dtype); B = x.size(0)
            hb = encode_blocks(bundle.target_encoder, x, 2); idx = block_indices(B, NB, S, dev)
            for C in CS:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    p = bundle.predictor.rollout(hb[:, :C * S], idx[:, :C * S], NB - C)
                K = NB - C
                xy, lg, a = R(p.float().reshape(B * K, S, D).half())
                xyf = np.full((B, NB, 2), np.nan, np.float32); pnf = np.full((B, NB), np.nan, np.float32)
                xyf[:, C:] = xy.float().view(B, K, 2).cpu().numpy()
                pnf[:, C:] = torch.softmax(lg.float(), -1)[:, NONE].view(B, K).cpu().numpy()
                out[f"xy_C{C}"].append(xyf); out[f"pnone_C{C}"].append(pnf)
            out["video_id"] += [v for _, v in ch]
            if rank == 0 and k % 160 == 0:
                print(f"[r0] {k + B}/{len(mine)} ({time.time() - t0:.0f}s)", flush=True)
    d = out_dir(name); d.mkdir(parents=True, exist_ok=True)
    np.savez(d / f"shard{rank}.npz", video_id=np.array(out.pop("video_id")), **{k: np.concatenate(v) for k, v in out.items()})
    print(f"[r{rank}] done {len(mine)} ({time.time() - t0:.0f}s)", flush=True)


def analyze(name):
    Z = defaultdict(list)
    for f in sorted(out_dir(name).glob("shard*.npz")):
        z = np.load(f)
        for k in z.files: Z[k].append(z[k])
    Z = {k: np.concatenate(v) for k, v in Z.items()}
    rows = {r["video_id"]: r for r in csv.DictReader(open(REPO / "data_csv/rollout_v2_vob_readout_primitive/index.csv"))}
    meta = {r["name"]: r for r in csv.DictReader(META.open())}
    thr = json.load(open(REPO / f"z_research/VOBench_v1/readout/identity_vobprim_{name}/summary.json"))["reps"]["p"]["thr_val_fpr5"]
    CELL = 18.0
    res = {}
    for C in CS:
        by_k = defaultdict(list); by_t = defaultdict(list)
        for i, v in enumerate(Z["video_id"]):
            r = rows[str(v)]; sp = int(round(np.hypot(float(meta[str(v)]["vx_cm_s"] or 0), float(meta[str(v)]["vz_cm_s"] or 0))))
            px = np.array([float(x) for x in r["px_x_by_sample"].split()]).reshape(16, 2).mean(1)
            py = np.array([float(x) for x in r["px_y_by_sample"].split()]).reshape(16, 2).mean(1)
            vis = np.array([int(x) for x in r["visible_by_sample"].split()]).reshape(16, 2).all(1)
            T = np.stack([px, py], -1); X = (Z[f"xy_C{C}"][i] + 1) * 144
            d = T[15] - T[0]; n = np.linalg.norm(d)
            if n < 1e-3: continue
            d /= n
            ok = vis & (1 - Z[f"pnone_C{C}"][i] > thr)
            for t in range(C + 1, NB):
                if not (ok[t] and ok[t - 1]): continue
                tv = (T[t] - T[t - 1]) @ d / CELL; av = (X[t] - X[t - 1]) @ d / CELL
                by_k[(sp, t - C)].append((tv, av)); by_t[(sp, t)].append((tv, av))
        ratio = lambda L: float(np.mean([a for _, a in L]) / np.mean([t for t, _ in L])) if L else float("nan")
        res[C] = {"by_k": {f"{sp}/{k}": [ratio(L), len(L)] for (sp, k), L in by_k.items()},
                  "by_t": {f"{sp}/{t}": [ratio(L), len(L)] for (sp, t), L in by_t.items()}}
    json.dump(res, open(out_dir(name) / "horizon.json", "w"), indent=1)
    speeds = sorted({int(k.split("/")[0]) for k in res[CS[0]]["by_k"]})
    for sp in speeds:
        print(f"\n속도 {sp} cm/s — AR 속도 / 정답 속도")
        print("  rollout 걸음 k        " + " ".join(f"{'k' + str(k):>6s}" for k in range(1, 12)))
        for C in CS:
            print(f"  C={C:2d} (K={NB - C:2d})         " + " ".join(f"{res[C]['by_k'].get(f'{sp}/{k}', [np.nan])[0]:6.2f}" for k in range(1, 12)))
        print("  절대 블록 t           " + " ".join(f"{'t' + str(t):>6s}" for t in range(5, 16)))
        for C in CS:
            print(f"  C={C:2d}                 " + " ".join(f"{res[C]['by_t'].get(f'{sp}/{t}', [np.nan])[0]:6.2f}" for t in range(5, 16)))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["extract", "analyze"])
    ap.add_argument("--gpus", type=int, default=4)
    ap.add_argument("--name", default="ar40")
    ap.add_argument("--ckpt", default="z_training/ariel/ar_future_1e_5/latest.pt")
    a = ap.parse_args()
    if a.phase == "extract":
        import torch
        torch.multiprocessing.spawn(_worker, args=(a.gpus, a.name, a.ckpt), nprocs=a.gpus, join=True)
    else:
        analyze(a.name)
