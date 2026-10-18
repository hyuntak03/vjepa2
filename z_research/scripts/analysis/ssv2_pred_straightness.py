#!/usr/bin/env python3
"""predictor 가 만든 미래 튜블릿 궤적은 얼마나 곧은가 — vast AR (ar_state ep70) vs release, SSv2 200 영상 (2026-10-17, 사용자 질문).

영상 · 프레임은 ssv2_straightness.py 와 같다 (per_video.npz 의 200 경로, 연속 32 장 12 fps, 256²). 문맥 = 앞 16 장 (튜블릿 8), 미래 = 뒤 16 장 (튜블릿 8).
모델마다 자기 타깃 공간에서 잰다 (채점과 같은 공간):
  release : p = predictor(context_encoder(문맥), mask token) — 미래 8 튜블릿을 한 번에.  h = LN(target_encoder(32 장)).
  vast AR : 문맥 · 타깃 = 블록별 LN(target_encoder(튜블릿 하나씩)) (ARS.encode_blocks).  p = 8 걸음 rollout (ARS.rollout_from_blocks).
시점 표현 (튜블릿마다): mean = 패치 256 개 공간 평균 (1,280) · patch = 패치 이어 붙임 (327,680).
지표 (ssv2_straightness.py 와 같은 정의, 영상마다):
  θ = mean_t arccos(⟨v_t, v_{t+1}⟩/‖·‖‖·‖), v_t = x_{t+1} − x_t  (도; 무구조면 ~120°에 가깝다 — 고차원 무작위 걸음)
  R²_line = 직선 (a + bτ) 최소제곱 적합 R²
궤적 다섯:  fut_true (h 미래 8) · fut_pred (p 미래 8) · fut_shuffle (h 미래 8 순서 섞기, 대조) · seq_true (h 문맥 8 + h 미래 8 = 16) · seq_pred (h 문맥 8 + p 미래 8 = 16).
  추가: step 크기 ‖v_t‖ 평균 (pred / true) — 예측이 얼마나 덜 움직이는지 (멈추면 θ 가 커 보일 수 있다).

  CUDA_VISIBLE_DEVICES=<g> python z_research/scripts/analysis/ssv2_pred_straightness.py --cfg-release <yaml> --cfg-ar <yaml> [--n 200]
  → z_research/VOBench_v1/vjepa2/ssv2_straightness/pred_straightness.json (+ pred_straightness_traj.npz)
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "z_research/scripts/analysis"))
from ssv2_straightness import r2, turning  # noqa: E402

D = REPO / "z_research/VOBench_v1/vjepa2/ssv2_straightness"
NF, CTX = 32, 16


def run_model(cfg_path, paths, dev):
    import torch, yaml
    from decord import VideoReader, cpu
    from analysis.intphys2.model import build_from_config
    from analysis.intphys2.dataset import build_video_transform
    from analysis.intphys2.surprise import _context_target_indices
    from analysis import predictors as PV
    from analysis.predictors import ar_scoring as ARS
    cfg = yaml.safe_load(open(cfg_path)); B = build_from_config(cfg["model"], dev); tf = build_video_transform(256, 1.0)
    ts, sp = B.tubelet_size, B.num_spatial_tokens; ar = PV.is_ar(B.predictor)
    ac = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(str(cfg["model"].get("autocast", "none")).lower())
    ci, ti = _context_target_indices(ctx_frames=CTX, tgt_frames=NF - CTX, tubelet_size=ts, spatial_tokens=sp, batch_size=1, device=dev)
    H, P = [], []
    with torch.no_grad():
        for k, p in enumerate(paths):
            fr = VideoReader(p, ctx=cpu(0)).get_batch(list(range(NF))).asnumpy()
            x = tf(list(fr)).unsqueeze(0).to(dev, B.dtype)
            with (torch.autocast("cuda", dtype=ac) if ac else contextlib.nullcontext()):
                if ar:
                    hb = ARS.encode_blocks(B.target_encoder, x, ts)                       # (1, 16*S, D) 블록별 LN
                    pr, _, _ = ARS.rollout_from_blocks(B.predictor, hb, CTX // ts, sp)     # (1, 8*S, D)
                    h = hb
                else:
                    z = B.context_encoder(x, masks=[ci[:1]])
                    pr = B.predictor(z, ci[:1], ti[:1], mask_index=int(cfg["surprise"].get("mask_index", 0)))
                    h = torch.nn.functional.layer_norm(B.target_encoder(x).float(), (B.target_encoder(x).size(-1),))
            Dm = h.size(-1)
            H.append(h.float().view(NF // ts, sp, Dm).cpu().numpy().astype(np.float16)); P.append(pr.float().view((NF - CTX) // ts, sp, Dm).cpu().numpy().astype(np.float16))
            if k % 25 == 0: print(f"  {cfg_path.name if hasattr(cfg_path, 'name') else cfg_path} {k}/{len(paths)}", flush=True)
    del B; torch.cuda.empty_cache()
    return np.stack(H), np.stack(P)          # (n, 16, S, D), (n, 8, S, D)


def metrics(H, P):
    out = {}
    nc = CTX // 2
    for rep in ("mean", "patch"):
        f = (lambda a: a.astype(np.float32).mean(-2)) if rep == "mean" else (lambda a: a.astype(np.float32).reshape(*a.shape[:-2], -1))
        res = {k: {"theta": [], "r2_line": [], "step": []} for k in ("fut_true", "fut_pred", "fut_shuffle", "seq_true", "seq_pred")}
        for i in range(len(H)):
            h = f(H[i]); p = f(P[i]); perm = np.random.default_rng(1000 + i).permutation(8)
            T = {"fut_true": h[nc:], "fut_pred": p, "fut_shuffle": h[nc:][perm], "seq_true": h, "seq_pred": np.concatenate([h[:nc], p])}
            for k, Z in T.items():
                res[k]["theta"].append(turning(Z)); res[k]["r2_line"].append(r2(Z, 1)); res[k]["step"].append(float(np.linalg.norm(np.diff(Z, axis=0), axis=1).mean()))
        out[rep] = {k: {m: {"mean": float(np.mean(v)), "median": float(np.median(v))} for m, v in d.items()} for k, d in res.items()}
        out[rep]["_per_video"] = {k: {m: v for m, v in d.items()} for k, d in res.items()}
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cfg-release", required=True); ap.add_argument("--cfg-ar", required=True); ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    import torch
    paths = list(np.load(D / "per_video.npz", allow_pickle=True)["paths"])[: a.n]; dev = torch.device("cuda:0")
    summ = {"n_videos": len(paths), "frames": "연속 32 장 (12 fps), 문맥 16 · 미래 16", "models": {}}; traj = {}
    for name, cp in (("release", Path(a.cfg_release)), ("vast_ar", Path(a.cfg_ar))):
        H, P = run_model(cp, paths, dev)
        m = metrics(H, P); summ["models"][name] = {r: {k: v for k, v in m[r].items() if k != "_per_video"} for r in m}
        traj[f"{name}_h_mean"] = H.astype(np.float32).mean(-2); traj[f"{name}_p_mean"] = P.astype(np.float32).mean(-2)
        json.dump({name: {r: m[r]["_per_video"] for r in m}}, open(D / f"pred_straightness_{name}_per_video.json", "w"))
        for r in ("mean", "patch"):
            print(f"[{name} · {r}] " + " | ".join(f"{k} θ {m[r][k]['theta']['mean']:.1f}° R² {m[r][k]['r2_line']['mean']:.3f} step {m[r][k]['step']['mean']:.3g}"
                                               for k in ("fut_true", "fut_pred", "fut_shuffle", "seq_true", "seq_pred")), flush=True)
    json.dump(summ, open(D / "pred_straightness.json", "w"), indent=1, ensure_ascii=False)
    np.savez(D / "pred_straightness_traj.npz", paths=np.array(paths), **traj)
    print("saved", D / "pred_straightness.json")


if __name__ == "__main__":
    main()
