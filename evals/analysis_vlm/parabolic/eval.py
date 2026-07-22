"""Parabolic predictor-surprise eval — entry point.

    python -m evals.analysis_vlm.parabolic.eval --config <cfg.yaml> [--device cuda:0]

Uses the pretrained V-JEPA2 encoder+predictor (no training). For every scene it
computes a per-variant surprise = L1( predictor(context) , target_encoder(variant)[future] )
then applies the configured scoring mode(s). Everything (model, split of context/target,
distance, scoring) is driven by the YAML config.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import torch
import yaml

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from analysis.intphys2.model import build_from_config          # noqa: E402  (reused, unmodified)
from evals.analysis_vlm.parabolic.dataset import ParabolicScenes  # noqa: E402
from evals.analysis_vlm.parabolic.forward import scene_surprises   # noqa: E402
from evals.analysis_vlm.parabolic import scoring                   # noqa: E402


def _load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run(cfg: dict, device: torch.device, limit: int = None, smoke: bool = False) -> dict:
    ecfg = cfg.get("eval", {})
    dcfg = cfg["data"]
    variants = tuple(dcfg.get("variants", ["possible", "higher", "frozen"]))
    ctx_variant = dcfg.get("context_variant", "possible")

    print(f"[parabolic] building model on {device} ...")
    bundle = build_from_config(cfg["model"], device)
    ds = ParabolicScenes(dcfg["root"], variants=variants, context_variant=ctx_variant)
    context_length = int(ecfg.get("context_length", ds.ctx_frames))
    print(f"[parabolic] {len(ds)} scenes | frames={ds.n_frames} context={ds.ctx_frames} "
          f"| window={bundle.num_frames} tubelet={bundle.tubelet_size} "
          f"embed_dim={bundle.embed_dim} | variants={variants}")
    print(f"[parabolic] token grid: T'={bundle.num_temporal_tokens} spatial={bundle.num_spatial_tokens} "
          f"N={bundle.num_tokens} | context_length={context_length}f -> "
          f"ctx_tokens={(context_length//bundle.tubelet_size)*bundle.num_spatial_tokens} "
          f"tgt_tokens={((ds.n_frames-context_length)//bundle.tubelet_size)*bundle.num_spatial_tokens}")

    fkw = dict(context_length=context_length, distance=ecfg.get("distance", "l1"),
               loss_exp=float(ecfg.get("loss_exp", 1.0)),
               target_layer_norm=bool(ecfg.get("target_layer_norm", True)),
               mask_index=int(ecfg.get("mask_index", 0)), context_variant=ctx_variant)

    n_run = min(limit, len(ds)) if limit else len(ds)
    if smoke:
        print(f"[SMOKE] running {n_run} scenes with shape debug on the first")
    records, t0 = [], time.time()
    for i in range(n_run):
        sid, clips, meta = ds[i]
        if smoke and i == 0:
            print(f"  scene {sid}: clip shapes " + ", ".join(f"{v}={tuple(c.shape)}" for v, c in clips.items()))
        s = scene_surprises(clips, bundle, debug=(smoke and i == 0), **fkw)
        records.append({"scene_id": sid, "surprise": s, "meta": meta})
        if smoke:
            order = sorted(s, key=s.get)
            print(f"  scene {sid:4d}  " + "  ".join(f"{v}={s[v]:.4f}" for v in variants)
                  + f"   -> argmin={order[0]}  ({'OK' if order[0]=='possible' else 'x'})")
        elif (i + 1) % 25 == 0 or i + 1 == n_run:
            print(f"  [{i+1}/{n_run}] {(time.time()-t0)/(i+1):.2f}s/scene")

    # ---- scoring (config-driven) --------------------------------------------
    scfg = cfg.get("scoring", {})
    modes = scfg.get("modes", ["argmin", "pairwise"])
    results = {}
    if "argmin" in modes:
        results["argmin"] = scoring.score_argmin(
            records, scfg.get("argmin_candidates", list(variants)),
            correct=scfg.get("argmin_correct", "possible"))
    if "pairwise" in modes:
        pairs = [tuple(p) for p in scfg.get("pairwise_pairs", [["possible", "frozen"], ["possible", "higher"]])]
        results["pairwise"] = scoring.score_pairwise(records, pairs)
    return {"records": records, "results": results, "context_length": context_length,
            "variants": list(variants), "n_scenes": len(records)}


def write_outputs(out: dict, cfg: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    variants = out["variants"]
    # per-scene surprises + argmin
    with open(os.path.join(out_dir, "per_scene.csv"), "w", newline="") as f:
        cols = ["scene_id"] + [f"surprise_{v}" for v in variants] + ["argmin", "apex_m", "vx_mps", "g_mps2"]
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in out["records"]:
            s = r["surprise"]; m = r["meta"]
            row = {"scene_id": r["scene_id"], "argmin": min(s, key=s.get)}
            row.update({f"surprise_{v}": round(s.get(v, float('nan')), 6) for v in variants})
            row.update({k: m.get(k, "") for k in ("apex_m", "vx_mps", "g_mps2")})
            w.writerow(row)
    # summary (drop bulky per_scene lists)
    summary = {"n_scenes": out["n_scenes"], "context_length": out["context_length"],
               "variants": variants, "results": {}}
    for mode, res in out["results"].items():
        if mode == "argmin":
            summary["results"]["argmin"] = {k: v for k, v in res.items() if k != "per_scene"}
        else:
            summary["results"]["pairwise"] = {k: {kk: vv for kk, vv in v.items() if kk != "per_scene"}
                                              for k, v in res.items()}
    summary["config"] = cfg
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=2)

    # human-readable
    lines = ["=" * 60, "PARABOLIC PREDICTOR-SURPRISE EVAL", "=" * 60,
             f"scenes={out['n_scenes']}  context_length={out['context_length']}  variants={variants}", ""]
    if "argmin" in out["results"]:
        a = out["results"]["argmin"]
        lines.append(f"[argmin]  acc={a['accuracy']:.4f}  (chance {a['chance']:.3f}, n={a['n']})  "
                     f"candidates={a['candidates']}")
    if "pairwise" in out["results"]:
        for k, v in out["results"]["pairwise"].items():
            lines.append(f"[pairwise {k}]  acc={v['accuracy']:.4f}  (chance 0.5, n={v['n']})")
    txt = "\n".join(lines)
    open(os.path.join(out_dir, "summary.txt"), "w").write(txt + "\n")
    print("\n" + txt + f"\n\n-> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out_dir", default=None, help="override cfg.output_dir")
    ap.add_argument("--limit", type=int, default=None, help="only run the first N scenes")
    ap.add_argument("--smoke", action="store_true", help="verbose shape debug + per-scene surprises")
    a = ap.parse_args()
    cfg = _load_cfg(a.config)
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    out = run(cfg, device, limit=a.limit, smoke=a.smoke)
    default_out = cfg.get("output_dir", os.path.join(os.path.dirname(a.config), "results"))
    out_dir = a.out_dir or (default_out + "/smoke" if a.smoke else default_out)
    write_outputs(out, cfg, out_dir)


if __name__ == "__main__":
    main()
