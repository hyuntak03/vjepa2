#!/usr/bin/env python3
"""상태 안의 속도는 위치를 움직이는 레버인가 — 조향 실험 (2026-10-13). AR ep40 · AR-state ep40.

predictor_bottleneck 에서: 은닉 (predictor_norm 출력 = AR-state 의 되먹임 상태) 에 속도가 모든 걸음에서 선형으로 읽히는데 (R² ≈ 0.9),
그려진 물체의 진행은 4–5 걸음 뒤 멈춘다. 그렇다면 그 속도 코드를 바꾸면 물체 자리가 바뀌는가?

방향 u: train clip 의 걸음 1 은닉 블록 평균 (roll_st_g1) 에서 '오른쪽으로 가는 clip 평균 − 왼쪽으로 가는 clip 평균' (수평 운동만).
대조 u_rand: 같은 노름의 무작위 방향.
개입: rollout 걸음 2 의 출력 은닉 (모든 토큰) 에 α·u 를 더하고 (α ∈ {−2, −1, 0, +1, +2}), 그 은닉으로 그린 것을 되먹임 (AR) / 그 은닉을 되먹임 (AR-state).
측정: test 수평 운동 clip 에서 걸음 2..8 의 읽은 x (그 모델 p 자) − α = 0 의 x, 운동 방향 부호로 맞춘 칸 단위. 레버라면 α 에 비례해 앞으로 / 뒤로 움직인다.

  python predictor_velocity_steer.py --name ar_state40 --ckpt z_training/ariel/ar_state_future_1e_5/latest.pt --gpus 4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "z_research/scripts/analysis"))
import predictor_bottleneck as PB  # noqa: E402

S, NB, C, K, D = 256, 16, 8, 8, 1280
ALPHAS = (-2.0, -1.0, 0.0, 1.0, 2.0)
OUT = REPO / "z_research/VOBench_v1/bottleneck/steer"


def direction(name):
    Z = {}
    for f in sorted((PB.OUT / name).glob("shard*.npz")):
        z = np.load(f, allow_pickle=True)
        for k in ("roll_st_g1", "truth", "part", "row"): Z.setdefault(k, []).append(z[k])
    Z = {k: np.concatenate(v) for k, v in Z.items()}
    v = (Z["truth"][:, 15] - Z["truth"][:, 0]) / 15.0
    horiz = np.abs(v[:, 0]) > 3 * np.abs(v[:, 1])
    tr = (Z["part"] == "train") & horiz; te = (Z["part"] == "test") & horiz
    g = Z["roll_st_g1"].astype(np.float32)
    u = g[tr & (v[:, 0] > 0)].mean(0) - g[tr & (v[:, 0] < 0)].mean(0)
    rng = np.random.default_rng(0); ur = rng.standard_normal(u.shape).astype(np.float32); ur *= np.linalg.norm(u) / np.linalg.norm(ur)
    return u, ur, Z["row"][te], np.sign(v[te, 0])


def _worker(rank, world, name, ckpt, bs):
    import torch
    import torch.nn.functional as F
    from importlib.util import module_from_spec, spec_from_file_location
    from analysis.intphys2.model import build_from_config
    from app.vjepa_frozen.ar import encode_blocks
    from evals.world_model_analysis.data import WMADataset
    from src.models.utils.modules import PrefixSpec
    torch.cuda.set_device(rank); torch.set_num_threads(4); dev = torch.device("cuda", rank)
    spec = spec_from_file_location("ir", REPO / "z_research/scripts/analysis/rollout2_identity_readout.py")
    ir = module_from_spec(spec); spec.loader.exec_module(ir); rp = ir.rp
    model = dict(rp.MODEL, predictor_checkpoint=str(REPO / ckpt)); bundle = build_from_config(model, dev); pred = bundle.predictor
    is_state = pred.mask_mode == "ar_state"
    R = ir.IdReadout().to(dev).eval()
    R.load_state_dict(torch.load(REPO / f"z_research/VOBench_v1/readout/identity_vobprim_{name}/p/readout.pt", map_location=dev))
    ds = WMADataset(dict(data=rp.DATA, model=model, features={"cache_dir": "/tmp"}, surprise={}))
    u, ur, rows, sgn = direction(name)
    mine = list(range(rank * len(rows) // world, (rank + 1) * len(rows) // world))
    U = {"u": torch.tensor(u, device=dev), "rand": torch.tensor(ur, device=dev)}

    def run(tok):
        N = tok.size(1); nb = N // S
        idx = torch.arange(N, device=dev).unsqueeze(0).expand(tok.size(0), -1)
        h = pred._run_blocks(tok, idx, PrefixSpec(1, nb - 1, S, future_causal=True) if nb > 1 else None)
        st = pred.predictor_norm(h[:, -S:]); return st, F.layer_norm(pred.predictor_proj(st), (D,))

    def emb_own(st, rd):
        return st + pred.type_embed[1] if is_state else pred.predictor_embed(rd)

    out = {f"{d}_{a}": [] for d in U for a in ALPHAS}; t0 = time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for b0 in range(0, len(mine), bs):
            ii = mine[b0:b0 + bs]
            x = torch.stack([ds.clip(int(rows[i])) for i in ii]).to(dev, dtype=bundle.dtype)
            hb = encode_blocks(bundle.target_encoder, x, 2)
            obs = pred.predictor_embed(hb[:, :C * S]); obs = obs + pred.type_embed[0] if is_state else obs
            for dn, uu in U.items():
                for a in ALPHAS:
                    tok = obs; xs = []
                    for k in range(1, K + 1):
                        st, rd = run(tok)
                        if k == 2 and a != 0:
                            st = st + a * uu.to(st.dtype)
                            rd = F.layer_norm(pred.predictor_proj(st), (D,))
                        xy, _, _ = R(rd.half()); xs.append(((xy.float() + 1) * 144)[:, 0].cpu().numpy())
                        tok = torch.cat([tok, emb_own(st, rd)], 1)
                    out[f"{dn}_{a}"].append(np.stack(xs, 1))                          # (B, K) x px
            if rank == 0: print(f"[r0] {name} {b0 + len(ii)}/{len(mine)} ({time.time() - t0:.0f}s)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"{name}_r{rank}.npz", idx=np.array(mine), **{k: np.concatenate(v) for k, v in out.items()})


def summarize(name):
    u, ur, rows, sgn = direction(name)
    Z = {}
    for f in sorted(OUT.glob(f"{name}_r*.npz")):
        z = np.load(f)
        for k in z.files: Z.setdefault(k, []).append(z[k])
    Z = {k: np.concatenate(v) for k, v in Z.items()}
    s = sgn[Z["idx"]]
    res = {"n": int(len(Z["idx"])), "u_norm": float(np.linalg.norm(u))}
    for dn in ("u", "rand"):
        base = Z[f"{dn}_0.0"]
        # u = '오른쪽 운동' 코드 → +α 면 화면 x 가 + (오른쪽) 로 가야 한다 (clip 운동 방향과 무관하게 절대 x)
        res[dn] = {str(a): [float(np.median(Z[f"{dn}_{a}"][:, k] - base[:, k]) / 18.0) for k in range(K)] for a in ALPHAS}
        for part, m in (("right_movers", s > 0), ("left_movers", s < 0)):
            res[f"{dn}_{part}"] = {str(a): [float(np.median(Z[f"{dn}_{a}"][m, k] - base[m, k]) / 18.0) for k in range(K)] for a in (-2.0, 2.0)}
    json.dump(res, open(OUT / f"{name}.json", "w"), indent=1)
    print(f"\n== {name} 조향 (수평 운동 test {res['n']} clip) — 걸음 k1..k8 의 화면 x 변화 (칸, + = 오른쪽), α·u 를 걸음 2 은닉에 더함 (u = 오른쪽 − 왼쪽 운동 평균)")
    for dn in ("u", "rand"):
        for a in ALPHAS:
            print(f"  {dn:4s} α={a:+.0f}  " + " ".join(f"{v:6.2f}" for v in res[dn][str(a)]))
        for part in ("right_movers", "left_movers"):
            for a in (-2.0, 2.0):
                print(f"  {dn:4s} {part:12s} α={a:+.0f}  " + " ".join(f"{v:6.2f}" for v in res[f"{dn}_{part}"][str(a)]))
    for f in OUT.glob(f"{name}_r*.npz"): f.unlink()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--gpus", type=int, default=4); ap.add_argument("--bs", type=int, default=8)
    a = ap.parse_args()
    import torch
    torch.multiprocessing.spawn(_worker, args=(a.gpus, a.name, a.ckpt, a.bs), nprocs=a.gpus, join=True)
    summarize(a.name)
