#!/usr/bin/env python3
"""IntPhys 2 (Main) — 문맥 encoder (ViT-H `encoder`) 의 T×N 평균 풀링 위에 linear probe: possible vs impossible (2026-09-30, 사용자 지시).

프레임: 위반 시점 (`intphys2_onset_main_v2.json`, 국소 셀 통계) 을 포함하도록 stride 18 × 32 장 (span 559 프레임 = 9.3 s @ 60 fps).
  start = 0; 위반 뒤 표본이 2 장 미만이면 (onset > start + 29·18) start 를 뒤로 민다 (같은 쌍은 같은 start). 시점 없는 쌍은 start 0.
전처리: analysis/intphys2 와 동일 (Resize 256, CenterCrop 256 = no-op, ImageNet 정규화). bf16 autocast.
저장: 영상마다 encoder · target_encoder 각각 **튜블릿별 공간 평균 (16, 1280)** fp16 — T×N 평균은 여기서 다시 평균하면 된다 (다른 풀링도 재추출 없이).

  extract: python intphys2_linear_probe.py extract --shard i --nshards 8 --out-dir <dir>     (GPU 1 장 / 프로세스)
  probe:   python intphys2_linear_probe.py probe --feat-dir <dir> --out <json>                  (CPU)
    장면 (SceneIndex) 단위 5-fold (condition 층화) · 내부 4-fold 로 L2 λ 선택 · 표준화 · 로지스틱 회귀 (LBFGS).
    지표: 정확도 (chance 50) · condition 별 · 쌍 정확도 (같은 k 쌍에서 P(imp) 가 impossible 쪽이 큰가) · 라벨 섞기 기준선.
"""
import argparse, json, os, sys, glob
import numpy as np, pandas as pd

ROOT = "/data/hyuntak/project/2026/2027_cvpr/vjepa2"; sys.path.insert(0, ROOT)
IP2 = "/data2/local_datasets/world/IntPhys2/Main"
ONSET = f"{ROOT}/auto_research/exp_results/verify/intphys2_onset_main_v2.json"
CFG = f"{ROOT}/analysis/intphys2/configs/bench_intphys2_main_vith.yaml"
N_FRAMES, STRIDE = 32, 18


def plan():
    df = pd.read_csv(f"{IP2}/metadata.csv"); J = json.load(open(ONSET))
    onset = {(str(r["scene"]), r["pair"]): r.get("onset_loc_frame", -1) for r in J["rows"] if "error" not in r}
    nfr = {(str(r["scene"]), r["pair"]): r["n_frames"] for r in J["rows"] if "error" not in r}
    items = []
    for _, r in df.iterrows():
        t = str(r["type"]); k = int(t[0]); sc = str(r["SceneIndex"])
        # ⚠️ metadata 의 SceneIndex 에 "154_0" 같은 문자열이 있고, onset 스크립트는 int(sc) 로 저장했다 (파이썬은 밑줄을 허용해 1540 이 된다). 같은 변환으로 맞춘다.
        try: sk = str(int(sc))
        except ValueError: sk = sc
        o = onset.get((sk, k), -1); T = nfr.get((sk, k), 636)
        span = (N_FRAMES - 1) * STRIDE + 1; start = 0
        if o >= 0 and o > start + (N_FRAMES - 3) * STRIDE: start = max(0, min(T - span, o + 2 * STRIDE - (N_FRAMES - 1) * STRIDE))
        fn = str(r["file_name"]); fn = fn if fn.endswith(".mp4") else fn + ".mp4"; path = f"{IP2}/{fn}" if fn.startswith("Videos/") else f"{IP2}/Videos/{fn}"
        items.append(dict(video=str(r["name"]), path=path, scene=sc, pair=k, label=int("Impossible" in t), condition=str(r["condition"]), difficulty=str(r.get("Difficulty", "")),
                          camera=str(r.get("Camera", "")), onset_frame=int(o), start=int(start), n_frames=int(T)))
    return items


def extract(a):
    import torch, yaml
    from decord import VideoReader, cpu
    from analysis.intphys2.model import build_from_config
    from analysis.intphys2.dataset import build_video_transform
    items = plan()[a.shard :: a.nshards]; os.makedirs(a.out_dir, exist_ok=True)
    cfg = yaml.safe_load(open(CFG)); m = dict(cfg["model"]); m["window_size"] = N_FRAMES; m["dtype"] = "bfloat16"
    dev = torch.device("cuda:0"); B = build_from_config(m, dev); tf = build_video_transform(256, 1.0)
    encs = {"enc": B.context_encoder, "tgt": B.target_encoder}
    out = {k: [] for k in encs}; meta = []
    for i, it in enumerate(items):
        vr = VideoReader(it["path"], num_threads=2, ctx=cpu(0)); T = len(vr)
        idx = [min(it["start"] + j * STRIDE, T - 1) for j in range(N_FRAMES)]
        x = tf(vr.get_batch(idx).asnumpy()).unsqueeze(0).to(dev)                     # (1, 3, 32, 256, 256)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for k, e in encs.items():
                tok = e(x)                                                              # (1, 16·256, 1280)
                tub = tok.float().view(1, N_FRAMES // 2, -1, tok.shape[-1]).mean(2)[0]  # (16, 1280) 공간 평균
                out[k].append(tub.cpu().numpy().astype(np.float16))
        meta.append({**it, "idx": idx})
        if i % 20 == 0: print(f"[shard {a.shard}] {i}/{len(items)}", flush=True)
    np.savez(f"{a.out_dir}/shard{a.shard}.npz", enc=np.stack(out["enc"]), tgt=np.stack(out["tgt"]), meta=json.dumps(meta))
    print(f"[shard {a.shard}] done {len(items)}", flush=True)


def logreg_cv(X, y, groups, strata, lam_grid=(1e-3, 1e-2, 1e-1, 1.0, 10.0), k_outer=5, k_inner=4, seed=0):
    import torch
    rng = np.random.default_rng(seed)
    def folds(g, s, k):
        ug = np.unique(g); gs = {u: s[g == u][0] for u in ug}; f = {}
        for st in set(gs.values()):
            us = [u for u in ug if gs[u] == st]; rng.shuffle(us)
            for j, u in enumerate(us): f[u] = j % k
        return np.array([f[u] for u in g])
    def fit(Xtr, ytr, lam):
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; Xt = torch.tensor((Xtr - mu) / sd, dtype=torch.float64); yt = torch.tensor(ytr, dtype=torch.float64)
        w = torch.zeros(Xt.shape[1], dtype=torch.float64, requires_grad=True); b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([w, b], lr=1, max_iter=200, line_search_fn="strong_wolfe")
        def closure():
            opt.zero_grad(); z = Xt @ w + b; loss = torch.nn.functional.binary_cross_entropy_with_logits(z, yt) + lam * (w * w).sum() / len(yt); loss.backward(); return loss
        opt.step(closure); return (w.detach().numpy(), b.item(), mu, sd)
    def predict(model, X):
        w, b, mu, sd = model; return 1 / (1 + np.exp(-(((X - mu) / sd) @ w + b)))
    fo = folds(groups, strata, k_outer); p = np.zeros(len(y)); lam_pick = []
    for f in range(k_outer):
        tr = fo != f; Xtr, ytr, gtr, strr = X[tr], y[tr], groups[tr], strata[tr]; fi = folds(gtr, strr, k_inner)
        best = None
        for lam in lam_grid:
            acc = []
            for g in range(k_inner):
                m = fit(Xtr[fi != g], ytr[fi != g], lam); acc.append(((predict(m, Xtr[fi == g]) > 0.5) == ytr[fi == g]).mean())
            if best is None or np.mean(acc) > best[0]: best = (np.mean(acc), lam)
        lam_pick.append(best[1]); m = fit(Xtr, ytr, best[1]); p[~tr] = predict(m, X[~tr])
    return p, lam_pick


def probe(a):
    F = sorted(glob.glob(f"{a.feat_dir}/shard*.npz")); enc, tgt, meta = [], [], []
    for f in F:
        z = np.load(f, allow_pickle=True); enc.append(z["enc"].astype(np.float32)); tgt.append(z["tgt"].astype(np.float32)); meta += json.loads(str(z["meta"]))
    enc, tgt = np.concatenate(enc), np.concatenate(tgt); y = np.array([m["label"] for m in meta]); g = np.array([m["scene"] for m in meta]); cond = np.array([m["condition"] for m in meta])
    pair = np.array([f"{m['scene']}_{m['pair']}" for m in meta]); print("n", len(y), "pos/imp", (y == 0).sum(), (y == 1).sum(), "scenes", len(set(g)))
    res = {"n": int(len(y)), "n_scenes": int(len(set(g))), "frames": dict(n=N_FRAMES, stride=STRIDE), "variants": {}}
    variants = {"enc_meanTN": enc.mean(1), "tgt_meanTN": tgt.mean(1), "enc_maxT_meanN": enc.max(1), "enc_lastT_meanN": enc[:, -1]}
    rng = np.random.default_rng(0)
    for name, X in variants.items():
        p, lams = logreg_cv(X, y, g, cond); pred = (p > 0.5).astype(int); r = {"acc": float((pred == y).mean()), "lam": [float(l) for l in lams]}
        r["per_condition"] = {c: float((pred[cond == c] == y[cond == c]).mean()) for c in sorted(set(cond))}
        pa = []
        for pk in sorted(set(pair)):
            m = pair == pk
            if m.sum() == 2 and y[m].sum() == 1: pa.append(float(p[m][y[m] == 1][0] > p[m][y[m] == 0][0]))
        r["pair_acc"] = float(np.mean(pa)); r["n_pairs"] = len(pa)
        r["pair_acc_per_condition"] = {}
        for c in sorted(set(cond)):
            v = []
            for pk in sorted(set(pair[cond == c])):
                m = pair == pk
                if m.sum() == 2 and y[m].sum() == 1: v.append(float(p[m][y[m] == 1][0] > p[m][y[m] == 0][0]))
            r["pair_acc_per_condition"][c] = float(np.mean(v)) if v else None
        # AUC
        pos, neg = p[y == 1], p[y == 0]; r["auc"] = float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())
        if name == "enc_meanTN":
            perm = []
            for s in range(a.n_perm):
                yp = y.copy()
                for pk in set(pair):
                    m = np.where(pair == pk)[0]
                    if rng.random() < 0.5: yp[m] = 1 - yp[m]
                pp, _ = logreg_cv(X, yp, g, cond, lam_grid=(1e-1, 1.0), k_inner=2, seed=s + 1); perm.append(float(((pp > 0.5) == yp).mean()))
            r["perm_acc"] = perm
        res["variants"][name] = r; print(name, json.dumps(r), flush=True)
    json.dump(dict(res, meta=meta), open(a.out, "w"), indent=1); print("saved", a.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract"); e.add_argument("--shard", type=int, default=0); e.add_argument("--nshards", type=int, default=1); e.add_argument("--out-dir", required=True)
    p = sub.add_parser("probe"); p.add_argument("--feat-dir", required=True); p.add_argument("--out", required=True); p.add_argument("--n-perm", type=int, default=10)
    pl = sub.add_parser("plan")
    a = ap.parse_args()
    if a.cmd == "extract": extract(a)
    elif a.cmd == "probe": probe(a)
    else:
        it = plan(); st = np.array([i["start"] for i in it]); o = np.array([i["onset_frame"] for i in it])
        print("videos", len(it), "start>0", (st > 0).sum(), "onset none", (o < 0).sum()); print(it[0])
        ok = o >= 0; last = st + (N_FRAMES - 1) * STRIDE; print("onset 포함 (o in [start, last-18])", ((o[ok] >= st[ok]) & (o[ok] <= last[ok] - STRIDE)).mean())
        import collections; c = collections.Counter(str(int(i["scene"])) if i["scene"].replace("_", "").isdigit() else i["scene"] for i in it); print("scene 키 충돌", [k for k, v in c.items() if v != 4])
