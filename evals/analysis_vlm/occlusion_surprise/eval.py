"""Predictor-surprise scoring on the Blender occlusion set — entry point.

    python -m evals.analysis_vlm.occlusion_surprise.eval --config <cfg.yaml> [--device cuda:0]

Zero-shot: nothing is trained. We reuse V-JEPA 2's own training loss as a "surprise"
score, exactly as `app/vjepa/train.py` computes it:

    z_ctx  = context_encoder(clip, masks=[ctx_idx])       # online encoder, MASKED context
    z_pred = predictor(z_ctx, ctx_idx, tgt_idx)           # imagined future, frames 33..40
    h      = LayerNorm(target_encoder(clip))[tgt_idx]     # EMA teacher, what was rendered
    surprise = mean(|z_pred - h|)                         # element-wise L1

The context (frames 1..32) is shared by all four variants of a scene, so `z_ctx`/`z_pred`
are computed ONCE per scene and only the target encode changes. Every surprise difference
is therefore attributable to the future frames alone.

Scoring (both are config-driven):
  pairwise : surprise(impossible) > surprise(possible)?   chance 0.5
  argmin   : is `possible` the least surprising of the candidate set?   chance 1/|candidates|

The forward pass and the two scorers are reused verbatim from the parabolic eval, which
implements the same shared-context / multi-target pattern.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import yaml

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from analysis.intphys2.model import build_from_config                        # noqa: E402
from evals.analysis_vlm.parabolic import scoring                             # noqa: E402
from evals.analysis_vlm.parabolic.forward import scene_surprises             # noqa: E402
from evals.analysis_vlm.occlusion_surprise.dataset import OcclusionScenes    # noqa: E402


def _load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run(cfg: dict, device: torch.device, limit=None, smoke: bool = False) -> dict:
    dcfg, ecfg = cfg["data"], cfg.get("eval", {})
    variants = tuple(dcfg.get("variants", ["possible", "imp_shape", "imp_color", "imp_vanish"]))
    ctx_variant = dcfg.get("context_variant", "possible")

    ds = OcclusionScenes(
        dcfg["root"], dcfg.get("index_csv", "index.csv"),
        variants=variants, context_variant=ctx_variant,
        n_frames=int(dcfg.get("n_frames", 40)),
        splice_context=bool(dcfg.get("splice_context", True)),
        context_length=int(ecfg.get("context_length", 32)),
        split=dcfg.get("split"), limit_scenes=limit,
    )
    print(f"[data] {ds.summary()}")

    print(f"[model] building on {device} ...")
    bundle = build_from_config(cfg["model"], device)
    ctx_len = int(ecfg.get("context_length", 32))
    tgt_len = ds.n_frames - ctx_len
    print(f"[model] window={bundle.num_frames}f tubelet={bundle.tubelet_size} "
          f"spatial={bundle.num_spatial_tokens} embed={bundle.embed_dim}")
    print(f"[model] context={ctx_len}f -> {(ctx_len//bundle.tubelet_size)*bundle.num_spatial_tokens} tokens | "
          f"target={tgt_len}f -> {(tgt_len//bundle.tubelet_size)*bundle.num_spatial_tokens} tokens")

    fkw = dict(context_length=ctx_len, distance=ecfg.get("distance", "l1"),
               loss_exp=float(ecfg.get("loss_exp", 1.0)),
               target_layer_norm=bool(ecfg.get("target_layer_norm", True)),
               mask_index=int(ecfg.get("mask_index", 0)), context_variant=ctx_variant)

    records, t0 = [], time.time()
    for i in range(len(ds)):
        sid, clips, meta = ds[i]
        s = scene_surprises(clips, bundle, debug=(smoke and i == 0), **fkw)
        records.append({"scene_id": sid, "surprise": s, "meta": meta})
        if smoke:
            order = sorted(s, key=s.get)
            print(f"  scene {sid:4d}  " + "  ".join(f"{v}={s[v]:.5f}" for v in variants)
                  + f"   argmin={order[0]} {'OK' if order[0]=='possible' else 'x'}")
        elif (i + 1) % 25 == 0 or i + 1 == len(ds):
            el = time.time() - t0
            print(f"  [{i+1}/{len(ds)}] {el/(i+1):.2f}s/scene  eta={(len(ds)-i-1)*el/(i+1)/60:.1f}min")

    # ---- scoring -------------------------------------------------------------
    scfg = cfg.get("scoring", {})
    results = {"argmin": {}, "pairwise": {}}
    for name, cands in (scfg.get("argmin_sets") or {}).items():
        results["argmin"][name] = scoring.score_argmin(
            records, cands, correct=scfg.get("argmin_correct", "possible"))
    pairs = [tuple(p) for p in scfg.get("pairwise_pairs", [])]
    if pairs:
        results["pairwise"] = scoring.score_pairwise(records, pairs)

    # ---- divergence-frame breakdown -----------------------------------------
    # 변이는 divergence_frame(33~36)부터 나타나므로, 그 값이 클수록 target window(33~40)
    # 안에서 possible 과 impossible 이 겹치는 프레임이 많아져 신호가 희석된다.
    by_div = defaultdict(lambda: defaultdict(list))
    for r in records:
        d = r["meta"].get("divergence_frame", "")
        for key, v in results["pairwise"].items():
            hit = next((p["hit"] for p in v["per_scene"] if p["scene_id"] == r["scene_id"]), None)
            if hit is not None:
                by_div[key][d].append(hit)
    results["by_divergence_frame"] = {
        k: {d: {"n": len(h), "accuracy": float(np.mean(h))} for d, h in sorted(v.items())}
        for k, v in by_div.items()
    }
    return {"records": records, "results": results, "variants": list(variants),
            "n_scenes": len(records), "context_length": ctx_len}


def write_outputs(out: dict, cfg: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    variants, res = out["variants"], out["results"]

    with open(os.path.join(out_dir, "per_scene.csv"), "w", newline="") as f:
        cols = (["scene_id", "split", "divergence_frame", "reappear_frame", "n_hidden_frames",
                 "shape", "color"] + [f"surprise_{v}" for v in variants] + ["argmin"]
                + [f"delta_{v}" for v in variants if v != "possible"])
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in out["records"]:
            s, m = r["surprise"], r["meta"]
            row = {"scene_id": r["scene_id"], "argmin": min(s, key=s.get)}
            row.update({k: m.get(k, "") for k in
                        ("split", "divergence_frame", "reappear_frame", "n_hidden_frames",
                         "shape", "color")})
            row.update({f"surprise_{v}": round(s[v], 6) for v in variants})
            # delta = surprise(impossible) - surprise(possible); >0 이면 정답 방향
            row.update({f"delta_{v}": round(s[v] - s["possible"], 6)
                        for v in variants if v != "possible"})
            w.writerow(row)

    slim = {"n_scenes": out["n_scenes"], "context_length": out["context_length"],
            "variants": variants, "config": cfg, "results": {"argmin": {}, "pairwise": {},
            "by_divergence_frame": res.get("by_divergence_frame", {})}}
    for k, v in res["argmin"].items():
        slim["results"]["argmin"][k] = {kk: vv for kk, vv in v.items() if kk != "per_scene"}
    for k, v in res["pairwise"].items():
        slim["results"]["pairwise"][k] = {kk: vv for kk, vv in v.items() if kk != "per_scene"}
    json.dump(slim, open(os.path.join(out_dir, "summary.json"), "w"), indent=2)

    L = ["=" * 92,
         "PREDICTOR SURPRISE — Blender occlusion (zero-shot, 학습 없음)",
         "=" * 92,
         f"scenes={out['n_scenes']}  variants={variants}  context={out['context_length']}f",
         "",
         "surprise = mean(|predictor(context) - LayerNorm(target_encoder(clip))[33:40]|)",
         "         = V-JEPA 2 학습 손실 그 자체. 값이 클수록 '예상 밖'.",
         "context(1~32)는 variant 간 동일하므로 z_pred 는 scene 당 하나뿐이고,",
         "차이는 전부 target encode 에서만 나온다.",
         ""]

    if res["pairwise"]:
        L += ["-" * 92,
              "[pairwise]  surprise(impossible) > surprise(possible) 이면 정답. chance 0.500",
              "-" * 92,
              f"  {'pair':28s} {'n':>5s} {'accuracy':>9s} {'tie':>5s}"]
        for k, v in res["pairwise"].items():
            L.append(f"  {k:28s} {v['n']:5d} {v['accuracy']:9.4f} {v['n_tie']:5d}")
        L.append("")

    if res["argmin"]:
        L += ["-" * 92,
              "[argmin]  surprise 가 가장 작은 것이 possible 이면 정답",
              "-" * 92,
              f"  {'candidate set':28s} {'n':>5s} {'accuracy':>9s} {'chance':>7s}"]
        for k, v in res["argmin"].items():
            L.append(f"  {k:28s} {v['n']:5d} {v['accuracy']:9.4f} {v['chance']:7.3f}"
                     f"   {v['candidates']}")
        L.append("")

    if res.get("by_divergence_frame"):
        L += ["-" * 92,
              "[divergence_frame 별]  변이가 늦게 시작될수록 target window(33~40) 안에서",
              "possible 과 겹치는 프레임이 많아져 신호가 희석된다.",
              "-" * 92]
        for k, v in res["by_divergence_frame"].items():
            cells = "  ".join(f"f{d}: {x['accuracy']:.3f}(n={x['n']})" for d, x in v.items())
            L.append(f"  {k:28s} {cells}")
        L.append("")

    txt = "\n".join(L)
    open(os.path.join(out_dir, "summary.txt"), "w").write(txt + "\n")
    print("\n" + txt + f"\n-> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--limit", type=int, default=None, help="첫 N scene 만")
    ap.add_argument("--smoke", action="store_true", help="텐서 shape 디버그 + scene 별 출력")
    a = ap.parse_args()

    cfg = _load_cfg(a.config)
    out = run(cfg, torch.device(a.device), limit=a.limit, smoke=a.smoke)
    write_outputs(out, cfg, a.out_dir or cfg.get("output_dir", "./occlusion_surprise_out"))


if __name__ == "__main__":
    main()
