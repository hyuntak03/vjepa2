#!/usr/bin/env python3
"""E1b (paper_lab 진행자 2, 2026-10-21) — 입력 (문맥 블록 토큰) × 정답 (미래 표적 토큰) 의 떨림을 따로 뺀 2 × 2.

왜 (R5 반박): 정답 쪽 떨림은 (문맥과 독립 · 평균 0 · 대칭이면) 최적 예측을 바꾸지 않는다. 문맥 평균을 보상하는 것은
입력 쪽 떨림이다 (정지 칸에서 여러 블록 평균 = 입력 잡음 제거). E1 의 잡음 평균 공간은 문맥과 정답을 같이 평활해서 둘을 못 가른다.
그래서 같은 영상에서 입력 ∈ {원래, 잡음 평균} × 정답 ∈ {원래, 잡음 평균} 네 칸의 보상을 잰다.

잡음 평균 = 가우시안 2/255 × 8 판, 블록별 layer norm 뒤 평균 (E1 · 다른 팀과 같은 정의). bf16.
학습 없는 예측 (문맥만으로): 복사 · 문맥 2 / 8 블록 평균 · 역사 없는 수축 (원점 쪽 0.7, 마지막 블록 3 × 3 이웃 쪽 0.7, 칸 평균 쪽 0.7) ·
VOBench 는 추가로 한 블록 옮겨 붙이기 · 운반 오라클 (정답 자리) · 물체 없는 블록 평균.
판: VOBench 가능 영상 · 움직이는 물체 (가림 시점 전부, 약 1,200) + SSv2 검증 평가 쪽 200 개 (E1 보정에 안 쓴 것).

  python paper_lab/scripts/tspace_2x2.py extract --set vobench
  python paper_lab/scripts/tspace_2x2.py extract --set ssv2 --n 200
  python paper_lab/scripts/tspace_2x2.py analyze
산출: $TSR_OUT/x2_*.npz · x2_result.json
"""
import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "z_research/scripts/analysis"))
_spec = spec_from_file_location("tsr", REPO / "paper_lab/scripts/target_space_reward.py")
T = module_from_spec(_spec); _spec.loader.exec_module(T)
OUT = T.OUT
C, K = T.C, T.K
COMBOS = (("raw", "raw"), ("raw", "smooth"), ("smooth", "raw"), ("smooth", "smooth"))   # (입력, 정답)


def _renders(torch, Xc, tr):
    R = {"copy": Xc[C - 1], "mean2": Xc[C - 2:C].mean(0), "mean8": Xc.mean(0)}
    R.update(T._histfree(torch, Xc[C - 1]))
    R = {k: v.expand(K, -1, -1, -1) for k, v in R.items()}
    if tr is not None:
        Q = T._ctx_renders(torch, Xc, tr)
        for m in ("bg", "paste_or", "mc_or"): R[m] = Q[m]
    return R


def extract(setname, n):
    import torch
    torch_, dev, bundles, model = T._setup(0)
    enc, enc32, gauss, pm1, smooth_from = T._encoders(torch, dev, bundles)
    gen = torch.Generator(device=dev); gen.manual_seed(4000)
    rec = defaultdict(list); t0 = time.time()
    with torch.no_grad():
        if setname == "ssv2":
            files, load = T._ssv2_loader(n, 2 * n)
            items = [(f, None, None) for f in files]
            if T.LIMIT: items = items[:T.LIMIT]
            getx = lambda it: load(it[0])
        else:
            import yaml
            import design_headroom as H
            from evals.world_model_analysis.data import WMADataset
            cfg = yaml.safe_load(open(H.SETS["vobench"]["cfg"]))
            ds = WMADataset(dict(data=dict(cfg["data"], index_csv="index.csv"), model=model, features={"cache_dir": "/tmp"}, surprise={}))
            rows = {r["video_id"]: r for r in csv.DictReader(open(H.SETS["vobench"]["index"]))}
            items = []
            for i, rec_ in enumerate(ds.records):
                r = rows.get(rec_.video_id)
                if r is None or r["variant"] not in ("pos_a", "pos_b") or r["motion"] == "flat_static": continue
                t = H.track_vobench(r)
                if t is None: continue
                items.append((i, rec_.video_id, (t, *H.plan_positions(*t))))
            if T.LIMIT: items = items[:T.LIMIT]
            getx = lambda it: ds.clip(it[0])
        for j, it in enumerate(items):
            x = getx(it)
            if x is None: continue
            x = x.unsqueeze(0).to(dev).float()
            V = {"raw": enc(x)[0], "smooth": smooth_from([enc(gauss(x, gen)) for _ in range(T.KN)])[0]}
            for inp, tgt in COMBOS:
                R = _renders(torch, V[inp][:C], it[2])
                Fu = V[tgt][C:]
                Wt = (Fu - V[inp][:C].mean(0)).abs().mean(-1)                       # (K, G, G) 변화 가중: 정답이 문맥 평균과 다른 칸
                Wt = Wt / Wt.mean((-1, -2), keepdim=True).clamp(min=1e-6)
                for m, Rm in R.items():
                    E = (Rm - Fu).abs().mean(-1)                                        # (K, G, G)
                    rec[f"{inp}>{tgt}/{m}"].append(E.mean((-1, -2)).cpu().numpy())        # (K,)
                    rec[f"{inp}>{tgt}/w/{m}"].append((Wt * E).mean((-1, -2)).cpu().numpy())   # 변화 가중 손실 (운반 가중 손실 후보의 보상)
            rec["id"].append(str(it[1] if it[1] is not None else it[0]))
            if j % 50 == 0: print(f"[x2 {setname}] {j + 1}/{len(items)} ({time.time() - t0:.0f}s)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"x2_{setname}.npz", **{k: np.array(v) for k, v in rec.items()})
    print(f"[x2 {setname}] done ({time.time() - t0:.0f}s)", flush=True)


def analyze():
    res = {}
    for setname in ("vobench", "ssv2"):
        f = OUT / f"x2_{setname}.npz"
        if not f.exists(): continue
        Z = np.load(f, allow_pickle=True); d = {}
        print(f"\n[{setname}] 입력 > 정답: 평균 보상 (1 − L(8 블록 평균)/L(복사)) · 2 블록 · 역사 없는 수축 (z70 · nb70 · sp70) · 운반 보상 (1 − L(오라클)/L(8 블록 평균)) · 한 블록 옮겨 붙이기  [n]")
        for inp, tgt in COMBOS:
            key = f"{inp}>{tgt}"
            L = {m.split("/")[1]: float(Z[m].mean()) for m in Z.files if m.startswith(key + "/") and "/w/" not in m}
            Lw = {m.split("/")[2]: float(Z[m].mean()) for m in Z.files if m.startswith(key + "/w/")}
            e = dict(L=L, adv8=1 - L["mean8"] / L["copy"], adv2=1 - L["mean2"] / L["copy"],
                     shrink={m: 1 - L[m] / L["copy"] for m in ("z70", "nb70", "sp70")}, n=int(len(Z["id"])))
            if Lw:
                e["w_adv8"] = 1 - Lw["mean8"] / Lw["copy"]
                if "mc_or" in Lw: e["w_carry"] = 1 - Lw["mc_or"] / Lw["mean8"]
            if "mc_or" in L:
                e["carry"] = 1 - L["mc_or"] / L["mean8"]; e["carry_paste"] = 1 - L["paste_or"] / L["mean8"]
                e["carry_by_k"] = (1 - Z[f"{key}/mc_or"].mean(0) / Z[f"{key}/mean8"].mean(0)).tolist()
            d[key] = e
            print(f"  {key:15s} 평균 {100 * e['adv8']:5.1f} % · 2 블록 {100 * e['adv2']:5.1f} % · 수축 " + " ".join(f"{m}:{100 * v:5.1f}" for m, v in e["shrink"].items())
                  + (f" · 운반 {100 * e['carry']:5.2f} % (한 블록 {100 * e['carry_paste']:5.2f}) · 걸음별 " + " ".join(f"{100 * x:4.1f}" for x in e["carry_by_k"]) if "carry" in e else "")
                  + (f" · 변화 가중: 평균 {100 * e['w_adv8']:5.1f} % 운반 {100 * e.get('w_carry', float('nan')):5.2f} %" if "w_adv8" in e else "")
                  + f"  [{e['n']}]")
        res[setname] = d
    json.dump(res, open(OUT / "x2_result.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("phase", choices=["extract", "analyze"])
    ap.add_argument("--set", default="vobench", choices=["vobench", "ssv2"]); ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    if a.phase == "extract": extract(a.set, a.n)
    analyze()
