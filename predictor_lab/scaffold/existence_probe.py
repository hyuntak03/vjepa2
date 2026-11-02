#!/usr/bin/env python3
"""움직이는 물체를 "안 그리나" vs "다른 자리에 그리나" — 자리 허용 채점 (2026-11-02, 사용자: pixel decoder 로 그려 보니 안 나온다 · "해보세요").

정의 (파라미터 없음). 미래 칸마다 예측 p (256 토큰) 와 실제 h (256 토큰) 사이 토큰 L1 (차원 평균) 행렬 D[i, j] 를 만들고,
  허용 반경 s (체비쇼프, 토큰) 안에서 양방향 최솟값의 평균:
  d_s(p, h) = ½ · ( mean_i min_{‖i−j‖∞ ≤ s} D[i, j]  +  mean_j min_{‖i−j‖∞ ≤ s} D[i, j] ),  미래 8 칸 평균.
  s = 0 → 표준 채점과 같다 (D 의 대각 평균) — 점검.  s = 16 → 화면 어디든.
쌍 채점 = 표준과 같다: 같은 문맥의 예측 p 로 d_s(p, h_가능) < d_s(p, h_불가능) 이면 정답.
  물체를 지운 예측: s 를 키워도 가능 영상 (물체 있음) 의 물체 토큰이 예측 어디서도 짝을 못 찾는다 → 계속 오답.
  물체를 엉뚱한 자리 (s 안) 에 둔 예측: 짝을 찾아 가능 영상 쪽이 가까워진다 → 정답으로 바뀐다.
대상: VOBench late · VO_moving_occluder 의 영속 block (A = 사라짐 · B = 나타남, 움직임 / 멈춤). 모델 = 릴리즈 · VOBench PFT · v3 학습판들 + 복사 기준선.
채점 규약은 표준과 같다 (te 엔진 encode · predict, fp32 + autocast fp16, 목표 LN).

  sbatch 본체: predictor_lab/scaffold/existence_probe_job.sh  → predictor_lab/scaffold/results/existence_probe.json
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "z_research/scripts/analysis")); sys.path.insert(0, str(Path(__file__).resolve().parent))
from adapter_probe import PANELS  # noqa: E402  (eval_v3.sh 와 같은 데이터 · 창)

OUT = REPO / "predictor_lab/scaffold/results/existence_probe.json"
RUNS = REPO / "z_training/runs"
MODELS = [("release", None), ("vobftg", RUNS / "vobench_release_postft_g/latest.pt"), ("v3_mix1600_s0", RUNS / "v3_mix1600_s0/latest.pt"),
          ("v3_mix1600_t25_s0", RUNS / "v3_mix1600_t25_s0/latest.pt"), ("v3t_only_s0", RUNS / "v3t_only_s0/latest.pt"), ("v3t_voblike_s0", RUNS / "v3t_voblike_s0/latest.pt")]
RADII = [0, 1, 2, 3, 16]
G, C, S = 16, 8, 256


def cheb_masks(dev):
    yy, xx = torch.meshgrid(torch.arange(G), torch.arange(G), indexing="ij")
    y, x = yy.flatten(), xx.flatten()
    cd = torch.maximum((y[:, None] - y[None]).abs(), (x[:, None] - x[None]).abs()).to(dev)    # (256, 256)
    return {s: cd <= s for s in RADII}


def tol_dist(p, h, masks):
    """p, h (N, 8, 256, D) → {s: (N,)} 자리 허용 거리 (미래 8 칸 평균)."""
    N, K, _, D = p.shape
    Dm = torch.cdist(p.reshape(N * K, S, D).float(), h.reshape(N * K, S, D).float(), p=1) / D   # (N·K, 256, 256)
    out = {}
    for s, m in masks.items():
        X = Dm.masked_fill(~m[None], float("inf"))
        d = 0.5 * (X.min(2).values.mean(1) + X.min(1).values.mean(1))
        out[s] = d.view(N, K).mean(1)
    return out


def run():
    import te_v11_score as te
    import transfer_gap_te as tte
    dev = torch.device("cuda:0"); torch.cuda.set_device(dev)
    masks = cheb_masks(dev)
    recs = []
    for pn in ("vob_late", "vomo"):
        ds_name, sets = PANELS[pn]
        sets = [(kv.split("=", 1)[0] + "=" + str((REPO / kv.split("=", 1)[1]).resolve())) if kv.startswith("data.root=") and not kv.split("=", 1)[1].startswith("/") else kv
                for kv in sets]
        cfg, _ = tte.resolve(ds_name, "vith", sets, None, 0)
        te.N_SLOT = 8
        sc = te.Scorer(cfg, dev, [(t, str(p) if p else None) for t, p in MODELS], keep_own=True)
        ds = te.WMADataset(cfg)
        blocks = defaultdict(dict)
        for i, r in enumerate(ds.records):
            raw = r.raw
            if raw.get("violation_type") == "permanence":
                blocks[raw["block_id"]][raw["variant"]] = i
        blocks = {b: v for b, v in blocks.items() if all(k in v for k in ("pos_a", "pos_b", "imp_ab", "imp_ba"))}
        bl = sorted(blocks)
        t0 = time.time()
        for b0 in range(0, len(bl), 2):
            bb = bl[b0:b0 + 2]
            order = [(b, v) for b in bb for v in ("pos_a", "imp_ab", "pos_b", "imp_ba")]
            clips = torch.stack([ds.clip(blocks[b][v]) for b, v in order])
            z, h, ci, ti, _ = sc.encode(clips)
            B, D = z.shape[0], sc.D
            hf = h.float().reshape(B, 2 * C, S, D)[:, C:]
            with sc.autocast():
                zT = F.layer_norm(z.float(), (D,)).reshape(B, C, S, D)[:, C - 1:C]
            zT = zT.float().expand(-1, C, -1, -1)
            preds = {"copy": zT}
            for tag, mod in zip(sc.tags, sc.mods):
                preds[tag] = sc.predict(mod, z, ci, ti).float().reshape(B, C, S, D)
            for j, b in enumerate(bb):
                raw = ds.records[blocks[b]["pos_a"]].raw
                moving = "static" not in str(raw.get("motion", ""))
                for direction, ip, ii in (("A", 4 * j + 0, 4 * j + 1), ("B", 4 * j + 2, 4 * j + 3)):
                    for tag, P in preds.items():
                        pp = P[ip:ip + 1]                                                   # 같은 문맥 (가능 영상의 문맥 = 불가능 영상의 문맥)
                        dp = tol_dist(pp, hf[ip:ip + 1], masks); di = tol_dist(pp, hf[ii:ii + 1], masks)
                        recs.append({"panel": pn, "block": b, "dir": direction, "moving": moving, "model": tag,
                                     **{f"pos_s{s}": float(dp[s][0]) for s in RADII}, **{f"imp_s{s}": float(di[s][0]) for s in RADII}})
            if b0 % 20 == 0:
                print(f"[{pn}] {b0 + len(bb)}/{len(bl)} block {time.time() - t0:.0f}s", flush=True)
        del sc; torch.cuda.empty_cache()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"radii": RADII, "records": recs}, open(OUT, "w"))
    summary()


def summary():
    d = json.load(open(OUT)); recs = d["records"]
    groups = defaultdict(list)
    for r in recs:
        mv = "moving" if r["moving"] else "static"
        groups[(r["panel"], r["dir"], mv, r["model"])].append(r)
    res = {}
    models = ["copy"] + [t for t, _ in MODELS]
    for pn in ("vob_late", "vomo"):
        for direction, mv in (("A", "moving"), ("A", "static"), ("B", "moving"), ("B", "static")):
            print(f"== {pn} · {'사라짐' if direction == 'A' else '나타남'} · {'움직임' if mv == 'moving' else '멈춤'}")
            for m in models:
                g = groups.get((pn, direction, mv, m), [])
                if not g:
                    continue
                acc = {s: round(100 * float(np.mean([1.0 if r[f"imp_s{s}"] > r[f"pos_s{s}"] else 0.0 for r in g])), 1) for s in d["radii"]}
                res.setdefault(pn, {}).setdefault(f"{direction}_{mv}", {})[m] = {"n": len(g), **{f"s{s}": v for s, v in acc.items()}}
                print(f"  {m:20s} n {len(g):3d} | " + " · ".join(f"s{s} {acc[s]:5.1f}" for s in d["radii"]))
    json.dump({"radii": d["radii"], "accuracy": res}, open(OUT.with_name("existence_probe_summary.json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    (run if (len(sys.argv) > 1 and sys.argv[1] == "run") else summary)()
