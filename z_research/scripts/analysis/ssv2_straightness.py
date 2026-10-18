#!/usr/bin/env python3
"""SSv2 영상의 시간 궤적이 표현 공간에서 직선인가 — Hénaff 꺾임 각 · LiFT 직선 적합 (2026-10-15, 사용자 지시).

데이터: SSv2 val (48 프레임 이상, data_csv/ssv2/val_min48.csv) 에서 seed 0 무작위 200 개. 연속 32 프레임 (12 fps ≈ 2.7 s), 256×256
  (analysis/intphys2 의 build_video_transform(256, 1.0): 짧은 변 256 → 가운데 256). 튜블릿 2 프레임 → 시점 T = 16.
z_t (공간 넷):
  pixel      : 튜블릿 두 장 평균, 64×64 회색 (4,096 차원)
  enc_mean   : context encoder (online) 마지막 층 토큰의 공간 평균 (1,280)
  tgt_mean   : target encoder (EMA) 공간 평균 (1,280)
  enc_patch  : context encoder 패치 전체 이어 붙임 (256 × 1,280 = 327,680)
지표 (영상마다, 평균 · 중앙값 · 95 % 부트스트랩):
  (a) 꺾임 각  v_t = z_{t+1} − z_t,  θ_t = arccos(⟨v_t, v_{t+1}⟩ / (‖v_t‖‖v_{t+1}‖)),  영상 값 = mean_t θ_t (도)
  (b) 직선 · 2 차 적합  τ = t / (T−1),  최소제곱 z_t ≈ a + bτ  /  a + bτ + cτ²,  R² = 1 − Σ‖z_t − ẑ_t‖² / Σ‖z_t − z̄‖²
대조군: 같은 영상의 시점 순서를 무작위로 섞은 궤적 (shuffle) — 곧음의 기준선.
⚠️ encoder 는 32 프레임을 한 번에 보는 시공간 attention 이라 시점 t 의 토큰에 다른 시점 정보가 섞인다 — 곧음의 일부는 구조 탓일 수 있다.

  CUDA_VISIBLE_DEVICES=<gpu> python z_research/scripts/analysis/ssv2_straightness.py [--n 200]
  → z_research/VOBench_v1/vjepa2/ssv2_straightness/results.json (+ per_video.npz)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
OUT = REPO / "z_research/VOBench_v1/vjepa2/ssv2_straightness"
CFG = REPO / "analysis/intphys2/configs/bench_intphys2_main_vith.yaml"
NF = 32


def turning(Z):
    v = np.diff(Z, axis=0); a, b = v[:-1], v[1:]
    c = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))).mean())


def r2(Z, deg):
    T = len(Z); tau = np.arange(T) / (T - 1); X = np.stack([tau ** k for k in range(deg + 1)], 1)
    B, *_ = np.linalg.lstsq(X, Z, rcond=None); res = ((Z - X @ B) ** 2).sum(); tot = ((Z - Z.mean(0)) ** 2).sum()
    return float(1 - res / tot)


def pca_proj(Z, k=3):
    """영상 하나의 T 개 점을 그 영상 안에서 PCA → 앞 k 개 축 좌표 (T, k) 와 분산 설명 비율 (k,)."""
    C = Z - Z.mean(0); U, S, _ = np.linalg.svd(C, full_matrices=False)
    ev = S ** 2 / (S ** 2).sum(); return (U[:, :k] * S[:k]).astype(np.float32), ev[:k].astype(np.float32)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=200); a = ap.parse_args()
    import torch, yaml
    from decord import VideoReader, cpu
    from analysis.intphys2.model import build_from_config
    from analysis.intphys2.dataset import build_video_transform
    paths = [l.split()[0] for l in open(REPO / "data_csv/ssv2/val_min48.csv") if l.strip()]
    rng = np.random.default_rng(0); paths = [paths[i] for i in sorted(rng.choice(len(paths), a.n, replace=False))]
    cfg = yaml.safe_load(open(CFG)); m = dict(cfg["model"]); m["window_size"] = NF; m["dtype"] = "bfloat16"
    dev = torch.device("cuda:0"); B = build_from_config(m, dev); tf = build_video_transform(256, 1.0)
    names = ("pixel", "enc_mean", "tgt_mean", "enc_patch"); res = {n: {"real": {"theta": [], "r2_line": [], "r2_quad": []},
                                                                    "shuffle": {"theta": [], "r2_line": [], "r2_quad": []}} for n in names}
    proj, evr = {}, {}                       # 시각화용: 영상마다 PCA 앞 3 축 좌표 (16, 3) 와 분산 설명 비율
    for k, p in enumerate(paths):
        vr = VideoReader(p, ctx=cpu(0)); fr = vr.get_batch(list(range(NF))).asnumpy()
        x = tf(list(fr)).unsqueeze(0).to(dev)                                            # (1, 3, 32, 256, 256)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            e = B.context_encoder(x)[0].float().view(NF // 2, 256, -1).cpu().numpy()      # (16, 256, 1280)
            t = B.target_encoder(x)[0].float().view(NF // 2, 256, -1).cpu().numpy()
        g = torch.nn.functional.interpolate(torch.from_numpy(fr).float().mean(-1)[:, None], size=(64, 64), mode="bilinear", align_corners=False)[:, 0].numpy()
        Zs = {"pixel": g.reshape(NF // 2, 2, -1).mean(1), "enc_mean": e.mean(1), "tgt_mean": t.mean(1), "enc_patch": e.reshape(NF // 2, -1)}
        perm = np.random.default_rng(1000 + k).permutation(NF // 2)
        for n, Z in Zs.items():
            pr, ev = pca_proj(Z); proj.setdefault(n, []).append(pr); evr.setdefault(n, []).append(ev)
            for kind, ZZ in (("real", Z), ("shuffle", Z[perm])):
                res[n][kind]["theta"].append(turning(ZZ)); res[n][kind]["r2_line"].append(r2(ZZ, 1)); res[n][kind]["r2_quad"].append(r2(ZZ, 2))
        if k % 25 == 0: print(f"{k}/{len(paths)}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True); summ = {"n_videos": len(paths), "T": NF // 2, "frames": "연속 32 장 (12 fps)", "spaces": {}}
    bs = np.random.default_rng(0)
    for n in names:
        summ["spaces"][n] = {}
        for kind in ("real", "shuffle"):
            d = {}
            for key, v in res[n][kind].items():
                v = np.array(v); boot = [v[bs.integers(0, len(v), len(v))].mean() for _ in range(1000)]
                d[key] = {"mean": float(v.mean()), "median": float(np.median(v)), "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}
            summ["spaces"][n][kind] = d
            print(f"{n:10s} {kind:7s} | θ {d['theta']['mean']:6.1f}° | R²_line {d['r2_line']['mean']:.3f} | R²_quad {d['r2_quad']['mean']:.3f}", flush=True)
    json.dump(summ, open(OUT / "results.json", "w"), indent=1, ensure_ascii=False)
    np.savez(OUT / "per_video.npz", paths=np.array(paths), **{f"{n}_{kind}_{key}": np.array(v) for n in names for kind in ("real", "shuffle") for key, v in res[n][kind].items()},
             **{f"{n}_pca": np.stack(proj[n]) for n in names}, **{f"{n}_evr": np.stack(evr[n]) for n in names})
    print("saved", OUT / "results.json")


if __name__ == "__main__":
    main()
