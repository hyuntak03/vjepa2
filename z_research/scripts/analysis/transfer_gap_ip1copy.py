#!/usr/bin/env python3
"""transfer_gap Q3 · E36 보조 — IntPhys 1 Garrido 격자의 복사 기준선을 잡음 판 (확률적 반올림 seed) 에서. 2026-10-30, [X2] R3.

왜: E36 은 잡음 판에서 Garrido 규칙을 seed 마다 다시 적용해 (PFT − 릴리즈) 를 본다. 같은 판에서 복사 기준선 (예측 없음) 도 내야
    "릴리즈 88.89 vs 복사 85.0" (leakage) 비교가 잡음에서도 서는지 알 수 있다. te_intphys1_score.py 는 인과 표적까지 계산해 seed 당
    GPU 4 장 · 약 40 분이라, 여기서는 **복사 열만** (인과 표적 · predictor 없음) 같은 창 · 같은 배치 · 같은 수치 경로로 뽑는다.
정의 (te_intphys1_score.py 머리말과 같다)
  창 격자 = Layout (skip2_w16 · skip2_w32 · skip5_w16, C ∈ {2..10}×W/16, stride 2), 영상 하나의 시작점 전부를 한 배치로.
  h = LN(target_encoder(창 W 장)) · z = context_encoder(창, mask = 앞 C 장) · copy = LN(z)[문맥 마지막 튜블릿]
  copy_std[j] = mean |copy − h[문맥 뒤 j]| (토큰 256 · 채널 1280), 창 surprise = mean_j → Filtered (min over C) → AvgSurprise → 쌍 (strict <)
  칸 점수 = 성질 평균, 규칙 최고 = 칸 최대 (garrido_rescore.select).
관문: --seed 0 (원본 프레임) 앞 N 영상의 copy_std 가 TrainingEffects ip1score_final 의 copy_std 와 같은가 (max 상대 차 보고).
입력  프레임 /data2/local_datasets/world/world_analysis/cache/transfer_gap/frames_src{s}_ip1 (seed 0 = 원본 IntPhys1_dev_frame_png)
출력  /data2/local_datasets/world/world_analysis/cache/transfer_gap/E36copy/src{s}.npz (copy_std (V, 925), video_ids) ·
      결과 z_analysis/transfer_gap/results/q3/E36_copy.json

  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  CUDA_VISIBLE_DEVICES=0 $P z_research/scripts/analysis/transfer_gap_ip1copy.py extract --seed 0 --limit 8      # 관문 (GPU 1, 몇 분)
  CUDA_VISIBLE_DEVICES=0 $P z_research/scripts/analysis/transfer_gap_ip1copy.py extract --seed 1               # seed 하나 (GPU 1)
  $P z_research/scripts/analysis/transfer_gap_ip1copy.py score
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "z_research/scripts/analysis"))
import numpy as np  # noqa: E402

TG = Path("/data2/local_datasets/world/world_analysis/cache/transfer_gap")
ARR = TG / "E36copy"
OUT = REPO / "z_analysis/transfer_gap/results/q3/E36_copy.json"
REF = Path("/data2/local_datasets/world/world_analysis/cache/training_effects/ip1score_final/arrays_v360.npz")
COMBOS = ("skip2_w16", "skip2_w32", "skip5_w16")


def frames_root(seed):
    return "/local_datasets/world/world_analysis/IntPhys1_dev_frame_png" if seed == 0 else str(TG / f"frames_src{seed}_ip1")


def extract(seed, limit, shard, nshard):
    import torch
    import torch.nn.functional as F
    import te_intphys1_score as T
    from analysis.intphys2.model import build_from_config
    from analysis.intphys2.surprise import _context_target_indices
    from evals.world_model_analysis.data import WMADataset
    cfg = T.resolve_cfg()
    cfg["data"]["frames_root"] = frames_root(seed)
    lay = T.Layout(cfg)
    ds = WMADataset(cfg)
    dev = torch.device("cuda", 0); torch.cuda.set_device(dev)
    torch.set_num_threads(6)
    bundle = build_from_config(cfg["model"], dev); bundle.predictor = None; torch.cuda.empty_cache()
    ac = T._autocast(cfg)
    S, D = T.S_TOK, T.D
    ln = lambda t: F.layer_norm(t.float(), (D,))  # noqa: E731
    idx = list(range(len(ds.records)))[: limit or None][shard::nshard]
    out = np.full((len(idx), lay.R), np.nan, np.float32)
    t0 = time.time()
    for n, vi in enumerate(idx):
        clip = ds.clip(vi)
        with torch.inference_mode():
            for cb in lay.combos:
                W, U = cb["W"], cb["W"] // 2
                si = np.arange(len(cb["starts"]))
                X = torch.stack([clip[:, cb["frames"][i]] for i in si]).to(dev, bundle.dtype)
                m = len(si)
                with ac():
                    hfull = ln(bundle.target_encoder(X)).view(m, U, S, D)
                for C in cb["Cs"]:
                    Tc = C // 2
                    ci, ti = _context_target_indices(ctx_frames=C, tgt_frames=W - C, tubelet_size=2, spatial_tokens=S, batch_size=m, device=dev)
                    with ac():
                        z = bundle.context_encoder(X, masks=[ci])
                    zT = ln(z).view(m, Tc, S, D)[:, Tc - 1][:, None]
                    rid = lay.idx(cb, si, C).reshape(-1)
                    out[n, rid] = (zT - hfull[:, Tc:]).abs().mean((2, 3)).reshape(-1).cpu().numpy()
                del X, hfull
        if n % 20 == 0:
            el = time.time() - t0
            print(f"[seed {seed} shard {shard}] {n + 1}/{len(idx)} {el / 60:.1f}분 남은 {el / (n + 1) * (len(idx) - n - 1) / 60:.1f}분", flush=True)
    assert np.isfinite(out).all()
    ARR.mkdir(parents=True, exist_ok=True)
    sfx = f"_lim{limit}" if limit else ""
    np.savez(ARR / f"src{seed}{sfx}_sh{shard}of{nshard}.npz", copy_std=out, video_ids=np.array([ds.records[i].video_id for i in idx]),
             rows=lay.rows, combos=np.array(lay.names))
    print(f"[seed {seed} shard {shard}] done {len(idx)} {time.time() - t0:.0f}s", flush=True)
    if seed == 0:                      # 관문
        z = np.load(REF)
        ref = {str(v): i for i, v in enumerate(z["video_ids"])}
        assert (z["rows"] == lay.rows).all()
        vids = [ds.records[i].video_id for i in idx]
        R = np.stack([z["copy_std"][ref[v]] for v in vids])
        rel = np.abs(out - R) / np.abs(R)
        print(f"GATE vs ip1score_final copy_std: max rel {rel.max():.2e} median rel {np.median(rel):.2e} (n video {len(vids)})", flush=True)


def load_seed(seed):
    fs = sorted(ARR.glob(f"src{seed}_sh*of*.npz"))
    if not fs:
        return None
    V, C = [], []
    for f in fs:
        z = np.load(f); V += [str(v) for v in z["video_ids"]]; C.append(z["copy_std"])
    return V, np.concatenate(C)


def score():
    import te_intphys1_score as T
    import importlib.util
    spec = importlib.util.spec_from_file_location("gr", REPO / "z_research/scripts/analysis/garrido_rescore.py")
    GR = importlib.util.module_from_spec(spec); spec.loader.exec_module(GR)
    cfg = T.resolve_cfg()
    lay = T.Layout(cfg)
    meta = T.index_meta(cfg["data"])
    res = {"definition": __doc__.split("정의 (te_intphys1_score.py 머리말과 같다)")[1].split("관문:")[0].strip(), "versions": {}}
    vers = {"orig_te(ip1score_final)": None}
    z = np.load(REF)
    vers["orig_te(ip1score_final)"] = ([str(v) for v in z["video_ids"]], z["copy_std"])
    for s in range(0, 5):
        x = load_seed(s)
        if x is not None and len(x[0]) == 360:
            vers[f"src{s}" if s else "orig_lean"] = x
    hits = {}
    for name, (V, Cst) in vers.items():
        wd = T.window_dict({"video_ids": np.array(V)}, lay, Cst)
        combos = {}
        per = {}
        for combo in COMBOS:
            vids, starts, Cs, arr = GR.score_table(wd, combo)
            sc = {v: float(np.nanmean(np.nanmin(arr[i], 1))) for i, v in enumerate(vids)}
            g, bad = GR.pairwise(sc, meta)
            combos[combo] = {"reductions": {"filtered": {"overall": 100 * sum(x[1] for x in g.values()) / sum(x[0] for x in g.values()),
                                                         "per_group": {k: {"n": v[0], "acc": v[1] / v[0] * 100} for k, v in g.items()}}}}
            pr = defaultdict(dict)
            for v, r in meta.items():
                if v in sc:
                    pr[GR.pair_key(r)]["pos" if r["plausible"] == "1" else "imp"] = v
            per[combo] = {k: sc[x["pos"]] < sc[x["imp"]] for k, x in pr.items() if "pos" in x and "imp" in x}
        best = GR.select(combos, "intphys")
        res["versions"][name] = {"best_combo": best["combo"], "best_acc": round(best["accuracy"], 3),
                                 "per_combo": {c: round(e["selected"]["accuracy"], 3) for c, e in combos.items()}}
        hits[name] = per
    seeds = [k for k in res["versions"] if k.startswith("src")]
    if seeds:
        b = [res["versions"][k]["best_acc"] for k in seeds]
        res["best_over_seeds"] = {"values": b, "mean": round(float(np.mean(b)), 3), "min": min(b), "max": max(b)}
        base = "orig_lean" if "orig_lean" in hits else "orig_te(ip1score_final)"
        res["flip_vs"] = base
        for c in COMBOS:
            h0 = hits[base][c]
            ks = sorted(h0)
            corr = np.array([h0[k] for k in ks])
            f = np.array([sum(hits[s][c][k] != h0[k] for s in seeds) for k in ks])
            res.setdefault("flips", {})[c] = {"orig_correct": int(corr.sum()), "lost": int(f[corr].sum()), "gained": int(f[~corr].sum()),
                                              "flip_rate_correct": round(float(f[corr].sum() / (len(seeds) * corr.sum())), 3),
                                              "flip_rate_wrong": round(float(f[~corr].sum() / (len(seeds) * (~corr).sum())), 3) if (~corr).any() else None}
    # 같은 판의 릴리즈 · PFT (하네스, E36) 와 규칙 최고끼리 비교 — 경로가 다르다 (복사 = encoder 한 번 통과, 비트 관문 통과)
    g = REPO / "z_analysis/transfer_gap/results/q3/E36_garrido_noise.json"
    if g.exists():
        pv = json.load(open(g))["per_version"]
        cmp_ = {}
        for v, key in [("orig", "orig_te(ip1score_final)")] + [(k, k) for k in seeds]:
            if v in pv and key in res["versions"]:
                c = res["versions"][key]["best_acc"]
                cmp_[v] = {"copy": c, "release_minus_copy": round(pv[v]["release"]["acc"] - c, 3), "pft_minus_copy": round(pv[v]["vobpft"]["acc"] - c, 3)}
        res["vs_models_garrido_best"] = cmp_
        rm = [cmp_[k]["release_minus_copy"] for k in seeds if k in cmp_]
        if rm:
            mm = float(np.mean(rm))
            res["verdict_release_minus_copy"] = {"values": rm, "mean": round(mm, 3), "orig": cmp_.get("orig", {}).get("release_minus_copy"),
                                                 "verdict": ("잡음 판에서도 릴리즈가 복사를 넘는다" if all(x > 0 for x in rm) and mm >= 3 else
                                                             "잡음 판에서는 복사와 구분 안 됨" if mm <= 1 else "보류")}
        res["copy_cell_drop_mean"] = {c: round(float(np.mean([res["versions"][k]["per_combo"][c] for k in seeds])) - res["versions"]["orig_te(ip1score_final)"]["per_combo"][c], 3) for c in COMBOS}
    if "orig_lean" in hits:
        res["orig_lean_vs_te_pair_agree"] = {c: round(float(np.mean([hits["orig_lean"][c][k] == hits["orig_te(ip1score_final)"][c][k] for k in hits["orig_lean"][c]])), 4) for c in COMBOS}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(res, ensure_ascii=False, indent=0)[:3000]); print("→", OUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["extract", "score"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    a = ap.parse_args()
    extract(a.seed, a.limit, a.shard, a.nshard) if a.part == "extract" else score()


if __name__ == "__main__":
    main()
