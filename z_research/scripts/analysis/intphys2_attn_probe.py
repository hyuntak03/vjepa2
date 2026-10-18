#!/usr/bin/env python3
"""IntPhys 2 Main — encoder 토큰 위 attentive probe: possible vs impossible (2026-10-15, 사용자 지시).

linear probe (auto_research/scripts/intphys2_linear_probe.py, 32 장 · T×N 평균) 는 작은 · 짧은 위반을 평균에 묻는다 →
토큰을 평균하지 않고 attentive probe (V-JEPA 2 표준 AttentiveClassifier: query 1 · cross-attention 블록 1 개 (MLP 포함) · head 16) 로 읽는다.

프레임: 영상 전체에서 **48 장 균등** (np.linspace(0, T−1, 48).round(), 원본 60 fps) — 위반 시점 추정에 기대지 않는다.
전처리: analysis/intphys2 채점과 같음 (build_video_transform(256, crop_margin_ratio=1.0), ImageNet 정규화), bf16.
encoder: ViT-H context encoder (online `encoder`) · target encoder (EMA) — 토큰 (24 튜블릿 × 256, 1280) 을 fp16 로 영상마다 저장.

split (5-fold, 위반 종류 층화) — 사용자 (10-15): 샘플 단위 · 쌍 단위 둘. 장면 단위는 참고로 같이.
  video : 샘플 (영상) 단위 — 영상마다 무작위. 한 쌍의 가능은 train, 불가능은 test 에 갈 수 있다 (가장 느슨)
  pair  : 쌍 (SceneIndex, k) 단위 — 한 쌍을 통째로. 같은 장면의 다른 쌍은 반대쪽에 있을 수 있다
  scene : 장면 (SceneIndex) 단위 — 처음 보는 장면 (가장 엄격, 참고)
지표: 영상 정확도 · AUC (주) · 쌍 정확도 (같은 쌍에서 불가능 쪽 P(imp) 가 큰가, surprise 와 같은 꼴, 보조) · 학습 정확도 (과적합) ·
  라벨 섞기 (쌍마다 50 % 로 두 영상 라벨을 서로 뒤집음) 기준선.
probe 학습: AdamW lr 3e-4 · wd 0.05 · 20 epoch · batch 16 · 상수 lr (내부 탐색 없이 고정 — 결과에 같이 적는다).

  python z_research/scripts/analysis/intphys2_attn_probe.py extract --gpus 4          # 특징 (vll5, 약 5 분)
  python z_research/scripts/analysis/intphys2_attn_probe.py probe --gpus 4            # probe (fold 를 GPU 에 나눠)
  python z_research/scripts/analysis/intphys2_attn_probe.py sweep --gpus 4            # context encoder · 장면 · 쌍 단위 · head 4 개 (lr {1e-4,3e-4} × wd {0.01,0.1}) · 40 epoch → results_sweep.json
  → z_research/VOBench_v1/vjepa2/encoder_voe_probe/attn48/results.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
SRC = Path("/data/dataset/world/IntPhys2/Main")
CACHE = Path("/data2/local_datasets/world/world_analysis/cache/intphys2_attn48")
OUT = REPO / "z_research/VOBench_v1/vjepa2/encoder_voe_probe/attn48"
CFG = REPO / "analysis/intphys2/configs/bench_intphys2_main_vith.yaml"
NF = 48
PY = "/data/hyuntak/anaconda3/envs/vjepa2/bin/python"


def items():
    import csv
    rows = list(csv.DictReader(open(SRC / "metadata.csv")))
    return [dict(i=i, video=r["name"], path=str(SRC / r["file_name"]), scene=r["SceneIndex"], pair=int(r["type"][0]),
                 label=int("Impossible" in r["type"]), condition=r["condition"], camera=r["Camera"]) for i, r in enumerate(rows)]


def extract_shard(shard, nshards):
    import torch, yaml
    from decord import VideoReader, cpu
    from analysis.intphys2.model import build_from_config
    from analysis.intphys2.dataset import build_video_transform
    its = items()[shard::nshards]
    cfg = yaml.safe_load(open(CFG)); m = dict(cfg["model"]); m["window_size"] = NF; m["dtype"] = "bfloat16"
    dev = torch.device("cuda:0"); B = build_from_config(m, dev); tf = build_video_transform(256, 1.0)
    for e in ("enc", "tgt"): (CACHE / e).mkdir(parents=True, exist_ok=True)
    for j, it in enumerate(its):
        if all((CACHE / e / f"{it['i']:04d}.npy").exists() for e in ("enc", "tgt")):
            continue
        vr = VideoReader(it["path"], num_threads=2, ctx=cpu(0)); T = len(vr)
        idx = np.linspace(0, T - 1, NF).round().astype(int).tolist()
        x = tf(vr.get_batch(idx).asnumpy()).unsqueeze(0).to(dev)                     # (1, 3, 48, 256, 256)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for e, enc in (("enc", B.context_encoder), ("tgt", B.target_encoder)):
                np.save(CACHE / e / f"{it['i']:04d}.npy", enc(x)[0].float().cpu().numpy().astype(np.float16))   # (6144, 1280)
        if j % 25 == 0: print(f"[shard {shard}] {j}/{len(its)}", flush=True)
    print(f"[shard {shard}] done", flush=True)


def folds(groups, strata, k=5, seed=0):
    rng = np.random.default_rng(seed); f = {}
    for st in sorted(set(strata)):
        ug = sorted({g for g, s in zip(groups, strata) if s == st}); rng.shuffle(ug)
        for n, g in enumerate(ug): f[g] = n % k
    return np.array([f[g] for g in groups])


FEAT = {}


def load_feats(enc, dev):
    """encoder 하나의 특징 전부 (1012 × 6144 × 1280 fp16 ≈ 16 GB) 를 GPU 에 한 번만 올린다 — 매 batch 디스크 (HDD) 에서 읽으면 job 당 수백 GB 읽기."""
    import torch
    if enc not in FEAT:
        n = len(items()); T = torch.empty((n, NF // 2 * 256, 1280), dtype=torch.float16, device=dev)
        for i in range(n): T[i] = torch.from_numpy(np.load(CACHE / enc / f"{i:04d}.npy")).to(dev)
        FEAT[enc] = T
    return FEAT[enc]


def run_fold(args):
    """한 (encoder, split, fold, shuffle) 학습 → test 영상의 P(imp) 와 학습 정확도."""
    enc, split, fold, shuffle, gpu = args
    import torch
    from src.models.attentive_pooler import AttentiveClassifier
    torch.manual_seed(fold); dev = torch.device(f"cuda:{gpu}"); torch.cuda.set_device(dev)
    its = items(); y = np.array([it["label"] for it in its])
    if shuffle:
        rng = np.random.default_rng(100 + fold)
        for key in {(it["scene"], it["pair"]) for it in its}:
            if rng.random() < 0.5:
                m = [n for n, it in enumerate(its) if (it["scene"], it["pair"]) == key]; y[m] = 1 - y[m]
    g = [it["scene"] if split == "scene" else (f"{it['scene']}_{it['pair']}" if split == "pair" else str(it["i"])) for it in its]
    fo = folds(g, [it["condition"] for it in its])
    tr, te = np.where(fo != fold)[0], np.where(fo == fold)[0]
    F = load_feats(enc, dev)
    X = lambda ids: F[torch.as_tensor(np.asarray(ids), device=dev)].float()
    model = AttentiveClassifier(embed_dim=1280, num_heads=16, depth=1, num_classes=2).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    yt = torch.from_numpy(y).to(dev)
    for ep in range(20):
        model.train(); perm = np.random.default_rng(ep + 10 * fold).permutation(tr)
        for s in range(0, len(perm), 16):
            b = perm[s:s + 16]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = torch.nn.functional.cross_entropy(model(X(b)), yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval(); P = {}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for ids, name in ((te, "te"), (tr, "tr")):
            p = []
            for s in range(0, len(ids), 32):
                p.append(torch.softmax(model(X(ids[s:s + 32])).float(), -1)[:, 1].cpu().numpy())
            P[name] = np.concatenate(p)
    tr_acc = float(((P["tr"] > 0.5) == y[tr]).mean())
    return dict(enc=enc, split=split, fold=fold, shuffle=shuffle, te=te.tolist(), p=P["te"].tolist(), y=y[te].tolist(), train_acc=tr_acc)


def metrics(its, idx, p, y):
    p, y = np.array(p), np.array(y); r = {"video_acc": float(((p > 0.5) == y).mean())}
    pos, neg = p[y == 1], p[y == 0]; r["auc"] = float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())
    by = {}
    for i, pi, yi in zip(idx, p, y): by.setdefault((its[i]["scene"], its[i]["pair"]), {})[int(yi)] = (pi, its[i]["condition"])
    pa = [(v[1][0] > v[0][0], v[1][1]) for v in by.values() if 0 in v and 1 in v]
    r["pair_acc"] = float(np.mean([a for a, _ in pa])); r["n_pairs"] = len(pa)
    r["pair_acc_per_condition"] = {c: float(np.mean([a for a, cc in pa if cc == c])) for c in sorted({c for _, c in pa})}
    r["video_acc_per_condition"] = {c: float(np.mean([((pi > 0.5) == yi) for i, pi, yi in zip(idx, p, y) if its[i]["condition"] == c]))
                                    for c in sorted({its[i]["condition"] for i in idx})}
    return r


def run_lane(jobs):
    out = []
    for j in jobs:
        out.append(run_fold(j)); print(f"[lane gpu{j[-1]}] {j[:4]} 끝", flush=True)
    return out


def probe(gpus):
    import torch.multiprocessing as mp
    jobs = [(e, s, f, sh) for e in ("enc", "tgt") for s in ("video", "pair", "scene") for sh in (False, True) for f in range(5)]
    # GPU 마다 encoder 하나 (앞 절반 GPU = enc, 뒤 절반 = tgt) — 특징 16 GB 를 GPU 에 한 번 올리고 그 encoder 의 job 만 돈다
    half = max(1, gpus // 2); lanes = {g: [] for g in range(gpus)}
    for n, j in enumerate([j for j in jobs if j[0] == "enc"]): lanes[n % half].append(j + (n % half,))
    for n, j in enumerate([j for j in jobs if j[0] == "tgt"]): g = half + n % (gpus - half) if gpus > 1 else 0; lanes[g].append(j + (g,))
    ctx = mp.get_context("spawn")
    with ctx.Pool(gpus) as pool:
        res = sum(pool.map(run_lane, list(lanes.values()), chunksize=1), [])
    its = items(); out = {"n_videos": len(its), "frames": f"{NF} uniform", "probe": "AttentiveClassifier depth 1 · heads 16 · AdamW 3e-4 · wd 0.05 · 20 ep · bs 16", "runs": {}}
    for e in ("enc", "tgt"):
        for s in ("video", "pair", "scene"):
            for sh in (False, True):
                R = [r for r in res if r["enc"] == e and r["split"] == s and r["shuffle"] == sh]
                idx = sum((r["te"] for r in R), []); p = sum((r["p"] for r in R), []); y = sum((r["y"] for r in R), [])
                m = metrics(its, idx, p, y); m["train_acc_mean"] = float(np.mean([r["train_acc"] for r in R]))
                out["runs"][f"{e}/{s}/{'shuffle' if sh else 'real'}"] = m
                print(f"{e} {s:5s} {'shuffle' if sh else 'real   '} | video {100*m['video_acc']:.1f} · AUC {m['auc']:.3f} · pair {100*m['pair_acc']:.1f} | train {100*m['train_acc_mean']:.1f}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True); json.dump(out, open(OUT / "results.json", "w"), indent=1)
    print("saved", OUT / "results.json")


HEADS = [(1e-4, 0.01), (1e-4, 0.1), (3e-4, 0.01), (3e-4, 0.1)]     # sweep (2026-10-15 사용자: 40 epoch · head 4 개)
EVAL_EP = (10, 20, 30, 40)


def run_fold_sweep(args):
    """head 4 개 (lr × wd) 를 같은 batch 로 동시에 40 epoch — epoch 10 · 20 · 30 · 40 의 test P(imp) 와 학습 정확도."""
    enc, split, fold, shuffle, gpu = args
    import torch
    from src.models.attentive_pooler import AttentiveClassifier
    torch.manual_seed(fold); dev = torch.device(f"cuda:{gpu}"); torch.cuda.set_device(dev)
    its = items(); y = np.array([it["label"] for it in its])
    if shuffle:
        rng = np.random.default_rng(100 + fold)
        for key in {(it["scene"], it["pair"]) for it in its}:
            if rng.random() < 0.5:
                m = [n for n, it in enumerate(its) if (it["scene"], it["pair"]) == key]; y[m] = 1 - y[m]
    g = [it["scene"] if split == "scene" else (f"{it['scene']}_{it['pair']}" if split == "pair" else str(it["i"])) for it in its]
    fo = folds(g, [it["condition"] for it in its]); tr, te = np.where(fo != fold)[0], np.where(fo == fold)[0]
    F = load_feats(enc, dev); X = lambda ids: F[torch.as_tensor(np.asarray(ids), device=dev)].float()
    models = [AttentiveClassifier(embed_dim=1280, num_heads=16, depth=1, num_classes=2).to(dev) for _ in HEADS]
    opts = [torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd) for m, (lr, wd) in zip(models, HEADS)]
    yt = torch.from_numpy(y).to(dev); out = {"te": te.tolist(), "y": y[te].tolist(), "p": {}, "train_acc": {}}

    def predict(m, ids):
        m.eval(); p = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for s in range(0, len(ids), 32): p.append(torch.softmax(m(X(ids[s:s + 32])).float(), -1)[:, 1].cpu().numpy())
        return np.concatenate(p)
    for ep in range(1, 41):
        for m in models: m.train()
        perm = np.random.default_rng(ep + 100 * fold).permutation(tr)
        for s in range(0, len(perm), 16):
            b = perm[s:s + 16]; xb = X(b)
            for m, o in zip(models, opts):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = torch.nn.functional.cross_entropy(m(xb), yt[b])
                o.zero_grad(); loss.backward(); o.step()
        if ep in EVAL_EP:
            for h, m in enumerate(models):
                out["p"][f"h{h}_e{ep}"] = predict(m, te).tolist()
                out["train_acc"][f"h{h}_e{ep}"] = float(((predict(m, tr) > 0.5) == y[tr]).mean())
    return dict(enc=enc, split=split, fold=fold, shuffle=shuffle, **out)


def run_lane_sweep(jobs):
    out = []
    for j in jobs:
        out.append(run_fold_sweep(j)); print(f"[lane gpu{j[-1]}] {j[:4]} 끝", flush=True)
    return out


def sweep(gpus):
    import torch.multiprocessing as mp
    jobs = [("enc", s, f, sh) for s in ("scene", "pair") for sh in (False, True) for f in range(5)]
    lanes = {g: [] for g in range(gpus)}
    for n, j in enumerate(jobs): lanes[n % gpus].append(j + (n % gpus,))
    with mp.get_context("spawn").Pool(gpus) as pool:
        res = sum(pool.map(run_lane_sweep, list(lanes.values()), chunksize=1), [])
    its = items(); out = {"heads": [f"lr {lr:g} · wd {wd:g}" for lr, wd in HEADS], "epochs": list(EVAL_EP), "runs": {}}
    for s in ("scene", "pair"):
        for sh in (False, True):
            R = [r for r in res if r["split"] == s and r["shuffle"] == sh]; idx = sum((r["te"] for r in R), []); y = sum((r["y"] for r in R), [])
            for h in range(len(HEADS)):
                for ep in EVAL_EP:
                    key = f"h{h}_e{ep}"; p = sum((r["p"][key] for r in R), [])
                    m = metrics(its, idx, p, y); m["train_acc_mean"] = float(np.mean([r["train_acc"][key] for r in R]))
                    out["runs"][f"{s}/{'shuffle' if sh else 'real'}/{key}"] = m
                    if ep == 40 or not sh:
                        print(f"{s:5s} {'shuffle' if sh else 'real   '} {out['heads'][h]:18s} e{ep:2d} | video {100*m['video_acc']:.1f} · AUC {m['auc']:.3f} · pair {100*m['pair_acc']:.1f} | train {100*m['train_acc_mean']:.1f}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True); json.dump(out, open(OUT / "results_sweep.json", "w"), indent=1); print("saved", OUT / "results_sweep.json")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["extract", "probe", "sweep", "_shard"]); ap.add_argument("--gpus", type=int, default=4)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1); a = ap.parse_args()
    if a.cmd == "_shard":
        return extract_shard(a.shard, a.nshards)
    if a.cmd == "extract":
        ps = [subprocess.Popen([PY, __file__, "_shard", "--shard", str(i), "--nshards", str(a.gpus)], env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(i))) for i in range(a.gpus)]
        sys.exit(max(p.wait() for p in ps))
    sweep(a.gpus) if a.cmd == "sweep" else probe(a.gpus)


if __name__ == "__main__":
    main()
