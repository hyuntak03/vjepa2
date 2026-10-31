"""VOBench_v1 late — GEOPHYS 각도 일관성 (angle consistency) 을 V-JEPA 2 target encoder 로 (2026-10-29).

  GEOPHYS (Internò et al. 2026, papers/GEOPHYS-...pdf §3.1) 의 단일 신호를 predictor 없이 encoder 궤적에 건다.
    1. 32 샘플 (raw 0,3,…,93) 을 한 번에 target encoder (EMA, 릴리즈 ViT-H, 마지막 층) 에 넣는다 → (16 튜블릿, 256, 1280)
    2. 공간 평균 → 궤적 Γ (16, 1280).  변위 v_t = Γ[t+1] − Γ[t] (15)  →  꺾임각 θ_t = arccos cos(v_t, v_{t+1}) (14)
    3. 영상 점수 u = std(θ).  쌍 (matched, 문맥 일치) 에서 u(imp) > u(pos) 이면 정답.  학습 파라미터 0, threshold 0.
  특징 두 판: raw (encoder 출력 그대로 — 논문 규칙, 정규화 없음) · ln (토큰마다 affine-free LN 뒤 평균 — 우리 채점 공간).
  보조로 mean(θ) (φ_curv) · std(‖v‖) (φ_speed) 도 같이 낸다 (같은 궤적, 추가 비용 0).
  인코딩 두 방식 (마지막 인자): window = 32 장 한 번에 (기본) · block = 튜블릿 (2 장) 마다 따로 (app/vjepa_frozen/ar.py encode_blocks 와 같은 입력, 2026-10-29).

  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P z_research/scripts/analysis/vobench_geophys_angle.py extract <n_gpu> [window|block]   # 궤적 캐시 (late 864 clip)
  $P z_research/scripts/analysis/vobench_geophys_angle.py score [window|block]            # → z_research/VOBench_v1/vjepa2/geophys/
"""
from __future__ import annotations
import json, os, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
CFG = REPO / "z_research/VOBench_v1/exp_results/surprise_c16t32__vobench_v1_vith/_resolved.yaml"
FRAMES_ROOT = "/local_datasets/world/testbed/VOBench_v1"          # vll7 사본 (vll5 는 /data2/…/Benchmark/VOBench_v1)
OUT = REPO / "z_research/VOBench_v1/vjepa2/geophys"
MODE = sys.argv[-1] if sys.argv[-1] in ("window", "block") else "window"
ALL = os.environ.get("GEOPHYS_OCC", "late") == "all"          # GEOPHYS_OCC=all: 가림 시점 네 칸 전부 (2,880 clip), 별도 캐시
SUF = ("" if MODE == "window" else "_block") + ("_all" if ALL else "")
CACHE = OUT / f"_traj{SUF}"
RESULTS = OUT / f"results{SUF}.json"
NT, S = 16, 256


def cfg_load():
    cfg = yaml.safe_load(open(CFG))
    cfg["data"]["frames_root"] = FRAMES_ROOT
    return cfg


def late_index(cfg):
    from evals.world_model_analysis.data import WMADataset
    ds = WMADataset(cfg)
    return ds, [i for i, r in enumerate(ds.records) if ALL or r.raw["occlusion"] == "late"]


def _worker(rank, world, cfg, idx):
    from analysis.intphys2.model import build_from_config
    from evals.world_model_analysis.attn_probe import _ClipDS, _ac_dtype, _autocast
    torch.cuda.set_device(rank); dev = torch.device(f"cuda:{rank}"); torch.set_num_threads(4)
    lo, hi = len(idx) * rank // world, len(idx) * (rank + 1) // world
    model = build_from_config(cfg["model"], dev)
    ac = _ac_dtype(cfg["model"].get("autocast", "none"))
    loader = torch.utils.data.DataLoader(_ClipDS(cfg, idx[lo:hi]), batch_size=4, shuffle=False, num_workers=6, pin_memory=True)
    raw, ln, t0 = [], [], time.time()
    with torch.inference_mode():
        for x in loader:
            b = x.size(0)
            x = x.to(dev, model.dtype, non_blocking=True)
            if MODE == "block":                                                      # 튜블릿마다 따로: (b·16, 3, 2, H, W)
                C_, T_ = x.shape[1], x.shape[2]
                x = x.reshape(b, C_, NT, T_ // NT, *x.shape[3:]).permute(0, 2, 1, 3, 4, 5).reshape(b * NT, C_, T_ // NT, *x.shape[3:])
            with _autocast(ac):
                h = model.target_encoder(x)
            h = (h[-1] if isinstance(h, list) else h).float().reshape(b, NT, S, -1)      # (b, 16, 256, 1280)
            raw.append(h.mean(2).cpu().numpy())
            ln.append(nn.functional.layer_norm(h, (h.size(-1),)).mean(2).cpu().numpy())
    np.save(CACHE / f"raw_s{rank}.npy", np.concatenate(raw)); np.save(CACHE / f"ln_s{rank}.npy", np.concatenate(ln))
    print(f"[r{rank}] {hi - lo} clip {time.time() - t0:.0f}s", flush=True)


def extract(gpus):
    cfg = cfg_load()
    ds, idx = late_index(cfg)
    CACHE.mkdir(parents=True, exist_ok=True)
    meta = [dict(video_id=ds.records[i].video_id, **{k: ds.records[i].raw[k] for k in
             ("block_id", "variant", "plausible", "pair_id", "objset", "violation", "occlusion", "k", "motion")}) for i in idx]
    json.dump(dict(frames_root=FRAMES_ROOT, checkpoint=cfg["model"]["checkpoint"], n=len(idx), world=gpus, meta=meta),
              open(CACHE / "meta.json", "w"))
    print(f"extract {len(idx)} late clips on {gpus} GPU", flush=True)
    torch.multiprocessing.spawn(_worker, args=(gpus, cfg, idx), nprocs=gpus, join=True)
    for f in ("raw", "ln"):
        np.save(CACHE / f"{f}.npy", np.concatenate([np.load(CACHE / f"{f}_s{r}.npy") for r in range(gpus)]))
        for r in range(gpus):
            (CACHE / f"{f}_s{r}.npy").unlink()


# ------------------------------------------------------------------ score
def signals(G):
    """G (n, T, D) → dict of (n,) — 값이 클수록 불규칙 (= 위반 쪽)."""
    v = np.diff(G, axis=1)                                             # (n, T-1, D)
    s = np.linalg.norm(v, axis=-1)
    c = (v[:, :-1] * v[:, 1:]).sum(-1) / (s[:, :-1] * s[:, 1:])
    th = np.arccos(np.clip(c, -1, 1))                                  # (n, T-2)
    return dict(angle_std=th.std(1), angle_mean=th.mean(1), speed_std=s.std(1), _theta=th)


def score():
    M = json.load(open(CACHE / "meta.json"))["meta"]
    vid = {m["video_id"]: j for j, m in enumerate(M)}
    pairs = defaultdict(dict)                                          # (block, pair A|B) → {pos, imp}
    for j, m in enumerate(M):
        pairs[(m["block_id"], m["pair_id"])]["pos" if m["plausible"] == "1" else "imp"] = j
    pairs = {k: v for k, v in pairs.items() if len(v) == 2}
    res, lines = {}, []
    for feat in ("raw", "ln"):
        sig = signals(np.load(CACHE / f"{feat}.npy").astype(np.float64))
        th = sig.pop("_theta")
        res[feat] = {"theta_mean_rad": float(th.mean())}
        for name, u in sig.items():
            cells = defaultdict(list)
            for (blk, pid), d in pairs.items():
                m = M[d["pos"]]
                ok = float(u[d["imp"]] > u[d["pos"]]) + 0.5 * float(u[d["imp"]] == u[d["pos"]])
                for key in ("all", f"{m['objset']}", f"{m['objset']}/{m.get('occlusion', 'late')}/{m['violation']}", f"{m['objset']}/{m['violation']}",
                            f"{m['objset']}/{m['violation']}/{pid}", f"{m['objset']}/{m['violation']}/{m['motion']}/{pid}"):
                    cells[key].append(ok)
            # 블록 부트스트랩 CI (objset 전체)
            rng = np.random.default_rng(0)
            r = {}
            for k, v in sorted(cells.items()):
                a = np.array(v)
                bs = [rng.choice(a, len(a)).mean() for _ in range(1000)] if k.count("/") <= 1 else None
                r[k] = dict(n=len(a), acc=100 * a.mean(),
                            ci=[100 * float(np.percentile(bs, 2.5)), 100 * float(np.percentile(bs, 97.5))] if bs else None)
            res[feat][name] = r
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(RESULTS, "w"), indent=1)
    print(f"[{MODE}]")
    for feat in ("raw", "ln"):
        print(f"\n== {feat}  (mean θ {res[feat]['theta_mean_rad']:.3f} rad)")
        for name in ("angle_std", "angle_mean", "speed_std"):
            r = res[feat][name]
            print(f"  {name:11s} all {r['all']['acc']:5.1f}  " + "  ".join(
                f"{o} {r[o]['acc']:5.1f}[{r[o]['ci'][0]:.0f},{r[o]['ci'][1]:.0f}]" for o in ("primitive", "realistic")))
            for o in ("primitive", "realistic"):
                print("     " + o[:4] + " " + "  ".join(
                    f"{v[:4]} {r[f'{o}/{v}']['acc']:5.1f} ({r[f'{o}/{v}/A']['acc']:.0f}/{r[f'{o}/{v}/B']['acc']:.0f})"
                    for v in ("permanence", "shape", "color")))


if __name__ == "__main__":
    if sys.argv[1] == "extract":
        extract(int(sys.argv[2]))
    else:
        score()
