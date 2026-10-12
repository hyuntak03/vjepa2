"""encoder 표현에 위반 (possible vs impossible) 이 선형으로 남아 있나 — VOBench · IntPhys 1 · IntPhys 2 (2026-10-12).

특징: 영상 32 장 전체 → context encoder (online `encoder`) · target encoder (EMA) 토큰 (T×N, 1280) → T×N 평균 (1, 1280).
  VOBench · IntPhys 1: 원본 PNG stride 3 (raw 0–93), surprise_c16t32 와 같은 로더 · bf16. 튜블릿별 공간 평균 (16, 1280) 을 저장한다.
  IntPhys 2: auto_research/scripts/intphys2_linear_probe.py 의 저장본 재사용 (60 fps stride 18 × 32 장, 위반 포함, 같은 두 encoder · bf16).
probe: 표준화 + L2 로지스틱 (intphys2_linear_probe.logreg_cv) · group (block / 장면) 단위 5-fold (위반 유형 층화) · 내부 4-fold λ.
  지표: 영상 정확도 · 쌍 정확도 (같은 쌍에서 P(imp) 가 impossible 쪽이 큰가) · 라벨 섞기 (쌍 단위 뒤집기) 기준선.

  python z_research/scripts/analysis/encoder_voe_probe.py extract vobench_v1 --gpus 4
  python z_research/scripts/analysis/encoder_voe_probe.py extract intphys1_dev --gpus 4
  python z_research/scripts/analysis/encoder_voe_probe.py probe          # → z_research/VOBench_v1/vjepa2/encoder_voe_probe/
  python z_research/scripts/analysis/encoder_voe_probe.py plot
"""
from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "auto_research/scripts"))
CACHE = Path("/local_datasets/world/world_analysis/cache/encoder_voe_probe")
IP2_FEATS = REPO / "auto_research/exp_results/intphys2_probe"
OUT = REPO / "z_research/VOBench_v1/vjepa2/encoder_voe_probe"
PY = "/data/hyuntak/anaconda3/envs/vjepa2/bin/python"
BENCHES = ("vobench_v1", "intphys1_dev", "intphys2_main")
NT, D = 16, 1280


def cfg_for(bench: str) -> dict:
    import yaml
    out = CACHE / bench / "_resolved.yaml"; out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([PY, str(REPO / "z_research/scripts/harness/resolve.py"), "surprise_c16t32", bench, "vith", "-o", str(out), "--quiet"],
                   check=True, cwd=REPO)
    cfg = yaml.safe_load(open(out))
    cfg["model"]["dtype"], cfg["model"]["autocast"] = "bfloat16", "none"     # probing 관례 (CLAUDE.md §1-7)
    return cfg


def _worker(rank, world, cfg, bench):
    import torch
    from analysis.intphys2.model import build_from_config
    from evals.world_model_analysis.attn_probe import _ClipDS
    from evals.world_model_analysis.data import WMADataset
    torch.cuda.set_device(rank); torch.set_num_threads(4); dev = torch.device(f"cuda:{rank}")
    n = len(WMADataset(cfg)); lo, hi = n * rank // world, n * (rank + 1) // world
    model = build_from_config(cfg["model"], dev)
    loader = torch.utils.data.DataLoader(_ClipDS(cfg, range(lo, hi)), batch_size=4, num_workers=6, pin_memory=True)
    out = {"enc": [], "tgt": []}; t0 = time.time()
    with torch.inference_mode():
        for x in loader:
            x = x.to(dev, model.dtype, non_blocking=True)
            for k, e in (("enc", model.context_encoder), ("tgt", model.target_encoder)):
                tok = e(x); tok = tok[-1] if isinstance(tok, list) else tok                 # (b, 16·256, D)
                out[k].append(tok.float().view(x.size(0), NT, -1, D).mean(2).cpu().numpy().astype(np.float16))
            if rank == 0 and len(out["enc"]) % 20 == 0:
                print(f"[r0] {len(out['enc']) * 4}/{hi - lo} ({time.time() - t0:.0f}s)", flush=True)
    np.savez(CACHE / bench / f"shard{rank}.npz", lo=lo, hi=hi, enc=np.concatenate(out["enc"]), tgt=np.concatenate(out["tgt"]))
    print(f"[r{rank}] done {hi - lo} ({time.time() - t0:.0f}s)", flush=True)


def extract(bench: str, gpus: int):
    import torch
    from evals.world_model_analysis.data import WMADataset
    cfg = cfg_for(bench)
    ds = WMADataset(cfg)
    meta = [{**{k: v for k, v in r.raw.items() if k not in ("samples", "file")}, "video_id": r.video_id} for r in ds.records]
    json.dump(meta, open(CACHE / bench / "meta.json", "w"))
    print(f"extract {bench}: {len(ds)} clips × 2 encoders on {gpus} GPU → {CACHE / bench}", flush=True)
    torch.multiprocessing.spawn(_worker, args=(gpus, cfg, bench), nprocs=gpus, join=True)


def load(bench: str):
    """-> enc (n,16,D), tgt (n,16,D), y (1 = impossible), group, strata, pair, attrs (dict of arrays)."""
    if bench == "intphys2_main":
        enc, tgt, meta = [], [], []
        for f in sorted(glob.glob(str(IP2_FEATS / "shard*.npz"))):
            z = np.load(f, allow_pickle=True); enc.append(z["enc"]); tgt.append(z["tgt"]); meta += json.loads(str(z["meta"]))
        y = np.array([m["label"] for m in meta])
        attrs = {"condition": np.array([m["condition"] for m in meta]), "camera": np.array([m["camera"] for m in meta])}
        return (np.concatenate(enc).astype(np.float32), np.concatenate(tgt).astype(np.float32), y,
                np.array([m["scene"] for m in meta]), attrs["condition"], np.array([f"{m['scene']}_{m['pair']}" for m in meta]), attrs)
    meta = json.load(open(CACHE / bench / "meta.json"))
    sh = sorted((np.load(f) for f in glob.glob(str(CACHE / bench / "shard*.npz"))), key=lambda z: int(z["lo"]))
    enc = np.concatenate([z["enc"] for z in sh]).astype(np.float32); tgt = np.concatenate([z["tgt"] for z in sh]).astype(np.float32)
    assert len(enc) == len(meta), (len(enc), len(meta))
    y = np.array([1 - int(m["plausible"]) for m in meta])
    g = np.array([str(m["block_id"]) for m in meta])
    pair = np.array([f"{m['block_id']}_{m['pair_id']}" for m in meta])
    if bench == "vobench_v1":
        attrs = {k: np.array([m[k] for m in meta]) for k in ("objset", "violation", "occlusion")}
        strata = np.char.add(np.char.add(attrs["objset"], "/"), attrs["violation"])
    else:
        attrs = {"block_type": np.array([m["block_type"] for m in meta])}
        strata = attrs["block_type"]
    return enc, tgt, y, g, strata, pair, attrs


def pair_hits(p, y, pair):
    out = {}
    for pk in np.unique(pair):
        m = np.where(pair == pk)[0]
        if len(m) == 2 and y[m].sum() == 1:
            out[pk] = float(p[m][y[m] == 1][0] > p[m][y[m] == 0][0])
    return out


def boot(vals_by_group: dict, rng, n=2000):
    keys = list(vals_by_group); arr = [np.array(vals_by_group[k]) for k in keys]
    b = [np.mean(np.concatenate([arr[i] for i in rng.integers(0, len(arr), len(arr))])) * 100 for _ in range(n)]
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def probe(n_perm: int):
    from intphys2_linear_probe import logreg_cv
    rng = np.random.default_rng(0); R = {}
    for bench in BENCHES:
        enc, tgt, y, g, strata, pair, attrs = load(bench)
        print(f"{bench}: n {len(y)} imp {y.sum()} groups {len(set(g))}", flush=True)
        R[bench] = {"n": int(len(y)), "n_groups": int(len(set(g)))}
        for name, X in (("context_encoder", enc.mean(1)), ("target_encoder", tgt.mean(1))):
            p, lams = logreg_cv(X, y, g, strata)
            hits = pair_hits(p, y, pair)
            grp_of_pair = {pk: g[pair == pk][0] for pk in hits}
            by_g = defaultdict(list)
            for pk, h in hits.items(): by_g[grp_of_pair[pk]].append(h)
            r = {"acc": float(((p > 0.5) == y).mean() * 100), "pair_acc": float(np.mean(list(hits.values())) * 100),
                 "pair_ci": boot(by_g, rng), "n_pairs": len(hits), "lam": [float(l) for l in lams], "breakdown": {}}
            for ax, vals in attrs.items():
                for v in np.unique(vals):
                    hv = [h for pk, h in hits.items() if vals[np.where(pair == pk)[0][0]] == v]
                    r["breakdown"].setdefault(ax, {})[str(v)] = {"pair_acc": float(np.mean(hv) * 100), "n": len(hv)}
            if bench == "vobench_v1":
                for o in np.unique(attrs["objset"]):
                    for vi in np.unique(attrs["violation"]):
                        for oc in np.unique(attrs["occlusion"]):
                            hv = [h for pk, h in hits.items() if (lambda i: attrs["objset"][i] == o and attrs["violation"][i] == vi and attrs["occlusion"][i] == oc)(np.where(pair == pk)[0][0])]
                            r["breakdown"].setdefault("objset/violation/occlusion", {})[f"{o}/{vi}/{oc}"] = {"pair_acc": float(np.mean(hv) * 100), "n": len(hv)}
            perm = []
            for s in range(n_perm):
                yp = y.copy()
                for pk in np.unique(pair):
                    if rng.random() < 0.5: yp[pair == pk] = 1 - yp[pair == pk]
                pp, _ = logreg_cv(X, yp, g, strata, lam_grid=(1e-1, 1.0), k_inner=2, seed=s + 1)
                perm.append(float(np.mean(list(pair_hits(pp, yp, pair).values())) * 100))
            r["perm_pair_acc"] = perm
            R[bench][name] = r
            print(f"  {name}: acc {r['acc']:.1f} pair {r['pair_acc']:.1f} {r['pair_ci']} perm {np.mean(perm):.1f}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "results.json", "w"), indent=1)
    print("saved", OUT / "results.json")


def plot():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    sys.path.insert(0, str(REPO / "z_research/scripts/figures"))
    from plot_v11_surprise import BLUE, ORANGE, MUTED, frame
    R = json.load(open(OUT / "results.json")); (OUT / "figures").mkdir(parents=True, exist_ok=True)
    names = {"vobench_v1": "VOBench v1", "intphys1_dev": "IntPhys 1 dev", "intphys2_main": "IntPhys 2 Main"}
    fig, ax = plt.subplots(figsize=(5.2, 3.0)); frame(ax, (30, 112), (40, 50, 60, 70, 80, 90, 100))
    x = np.arange(len(BENCHES)); w = 0.34
    for j, (enc, col, lab) in enumerate((("context_encoder", BLUE, "context encoder"), ("target_encoder", ORANGE, "target encoder"))):
        v = [R[b][enc]["pair_acc"] for b in BENCHES]; ci = np.array([R[b][enc]["pair_ci"] for b in BENCHES])
        xs = x + (j - 0.5) * w
        ax.bar(xs, v, w * 0.92, color=col, label=lab)
        ax.errorbar(xs, v, yerr=[np.array(v) - ci[:, 0], ci[:, 1] - np.array(v)], fmt="none", ecolor="k", lw=0.8, capsize=2)
        for xi, hi, vi in zip(xs, ci[:, 1], v): ax.text(xi, hi + 1.0, f"{vi:.0f}", ha="center", va="bottom", fontsize=7)
        pm = [np.mean(R[b][enc]["perm_pair_acc"]) for b in BENCHES]
        ax.scatter(xs, pm, marker="_", s=120, color=MUTED, zorder=3, label="shuffled labels" if j == 0 else None)
    ax.axhline(50, color=MUTED, lw=0.8, ls=":")
    ax.set_xticks(x); ax.set_xticklabels([f"{names[b]}\n(n={R[b]['context_encoder']['n_pairs']} pairs)" for b in BENCHES], fontsize=7.5)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    ax.set_ylabel("pair accuracy (%)")
    ax.legend(handles=[Patch(color=BLUE, label="context encoder"), Patch(color=ORANGE, label="target encoder"),
                       Line2D([], [], color=MUTED, lw=2, label="shuffled labels")], fontsize=7, frameon=False, loc="upper right")
    fig.tight_layout(); fig.savefig(OUT / "figures/fig_encoder_probe_bench.png", dpi=200); fig.savefig(OUT / "figures/fig_encoder_probe_bench.pdf"); plt.close(fig)

    B = R["vobench_v1"]; occ = ["visible", "early", "mid", "late"]; viol = ["permanence", "shape", "color"]
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.2), sharey=True)
    for ri, o in enumerate(("primitive", "realistic")):
        for ci_, vi in enumerate(viol):
            ax = axes[ri, ci_]; frame(ax, (30, 104), (50, 75, 100))
            for enc, col, lab in (("context_encoder", BLUE, "context encoder"), ("target_encoder", ORANGE, "target encoder")):
                bd = B[enc]["breakdown"]["objset/violation/occlusion"]
                ax.plot(range(4), [bd[f"{o}/{vi}/{oc}"]["pair_acc"] for oc in occ], "-o", color=col, ms=3.5, lw=1.3, label=lab)
            ax.axhline(50, color=MUTED, lw=0.8, ls=":")
            ax.set_xticks(range(4)); ax.set_xticklabels(occ, fontsize=7); ax.set_title(f"{o} · {vi}", fontsize=8)
            if ci_ == 0: ax.set_ylabel("pair accuracy (%)", fontsize=8)
    axes[0, 0].legend(fontsize=6.5, frameon=False, loc="lower left")
    fig.tight_layout(); fig.savefig(OUT / "figures/fig_encoder_probe_vobench.png", dpi=200); fig.savefig(OUT / "figures/fig_encoder_probe_vobench.pdf"); plt.close(fig)
    print("saved figures →", OUT / "figures")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["extract", "probe", "plot"])
    ap.add_argument("bench", nargs="?")
    ap.add_argument("--gpus", type=int, default=4)
    ap.add_argument("--n-perm", type=int, default=5)
    a = ap.parse_args()
    if a.phase == "extract":
        extract(a.bench, a.gpus)
    elif a.phase == "probe":
        probe(a.n_perm)
    else:
        plot()
