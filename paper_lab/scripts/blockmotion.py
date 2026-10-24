#!/usr/bin/env python3
"""G23 — 블록 토큰 하나 (연속 2 장) 에 한 장 인코딩에는 없는 순간 움직임 정보가 있나 (진행자 1, 2026-10-22, 학습 없음).
정의 · 판정 규칙 = paper_lab/EVIDENCE.md §3-2a 의 G23 (결과 전에 적음, 해시 사본 paper_lab/_scratch/prereg_G23.{txt,sha256}).

판: K400 공식 val, 하네스 natural 과 같은 순서 (rng 0) 의 앞 n 영상, 4 fps 가운데 32 장, 블록 7 = 프레임 14 · 15.
정답: 칸별 (16 × 16 화소) Farneback 흐름 14 → 15 평균 (px, x y). 일관 칸 = 앞-뒤 흐름 오차 칸 평균 ≤ 1 px.
특징 (칸 토큰 하나):
  a  V-JEPA 2 표적 encoder, 블록 (14, 15)            1280
  b  V-JEPA 2, 한 장 복제 (15, 15)  (V-JEPA 2-AC 꼴)   1280
  c  [ (14, 14) ; (15, 15) ]                             2560
  d  DINOv2 ViT-B/14-reg 최종 층, 프레임 15              768
  e  [ DINOv2(14) ; DINOv2(15) ]                         1536
  pos 칸 좌표 2 차 다항식 (판 사용 조건)                 6
모든 토큰은 토큰별 LN (affine 없음). 판독 = ridge (특징 표준화, 세기는 학습 반 안의 반 나눔으로 고름), 보류 = 영상 단위 반 (rng 0).

  CUDA_VISIBLE_DEVICES=<g> python paper_lab/scripts/blockmotion.py extract --n 400      # GPU 10–20 분 [추정]
  python paper_lab/scripts/blockmotion.py analyze                                       # CPU
산출: 캐시 /local_datasets/world/world_analysis/cache/paper_lab_blockmotion/ (npy) · 결과 paper_lab/_scratch/blockmotion/result.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import carry_coef as CC   # noqa: E402  (sys.path: repo · paper_lab/eval · z_research/scripts/analysis)
import carry_ext as CE    # noqa: E402

G, CELL = 16, 16
CACHE = Path(os.environ.get("BM_CACHE", "/local_datasets/world/world_analysis/cache/paper_lab_blockmotion"))
OUT = Path(os.environ.get("BM_OUT", str(CC.REPO / "paper_lab/_scratch/blockmotion")))
FEATS = ("a", "b", "c", "d", "e")


def cell_flow(x):
    """x (3, T, 256, 256) 정규화 → 칸 흐름 14 → 15 (G·G, 2) px · 일관 칸 (G·G,) · 칸 평균 흐름 크기."""
    import cv2
    g = CC._denorm_grey(x)
    fw = cv2.calcOpticalFlowFarneback(g[14], g[15], None, 0.5, 3, 15, 3, 5, 1.2, 0)   # 14(p) ≈ 15(p + fw(p))
    bw = cv2.calcOpticalFlowFarneback(g[15], g[14], None, 0.5, 3, 15, 3, 5, 1.2, 0)
    yy, xx = np.mgrid[0:256, 0:256].astype(np.float32)
    bws = np.stack([cv2.remap(bw[..., i], xx + fw[..., 0], yy + fw[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                    for i in range(2)], -1)
    err = np.linalg.norm(fw + bws, axis=-1)
    flow = fw.reshape(G, CELL, G, CELL, 2).mean((1, 3)).reshape(-1, 2)
    cons = err.reshape(G, CELL, G, CELL).mean((1, 3)).reshape(-1) <= 1.0
    return flow.astype(np.float32), cons


def extract(n, bs):
    import torch
    import torch.nn.functional as F
    import passline as PL
    dev = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    model, bundle = PL._load_model("z_training/vast/ar_state/latest.pt", dev)       # 표적 encoder 만 쓴다
    CACHE.mkdir(parents=True, exist_ok=True)
    acc = {k: [] for k in ("a", "d14", "d15", "g14", "g15", "flow", "cons", "video")}
    buf, t0 = [], time.time()

    def flush():
        if not buf:
            return
        x = torch.stack([b[1] for b in buf]).to(dev)                                  # (B, 3, 32, 256, 256)
        f14, f15 = x[:, :, 14:15], x[:, :, 15:16]
        clips = torch.cat([torch.cat([f14, f15], 2), torch.cat([f14, f14], 2), torch.cat([f15, f15], 2)], 0)
        with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            h = CE.enc_vjepa2(bundle, clips.to(bundle.dtype)).float()[:, 0]           # (3B, G, G, 1280) LN
            gd = CE._dino_frames(x, [14, 15]).float()                                  # (B, 2, G, G, 768)
        gd = F.layer_norm(gd, (gd.shape[-1],))
        B = x.shape[0]
        for k, t in (("a", h[:B]), ("d14", h[B:2 * B]), ("d15", h[2 * B:]), ("g14", gd[:, 0]), ("g15", gd[:, 1])):
            acc[k].append(t.reshape(B, G * G, -1).half().cpu().numpy())
        for b in buf:
            acc["flow"].append(b[2]["flow"][None]); acc["cons"].append(b[2]["cons"][None]); acc["video"].append(b[0])
        buf.clear()
        print(f"[blockmotion extract] {len(acc['video'])} videos ({time.time() - t0:.0f}s, {dev})", flush=True)

    for vid, x, _ in CC.natural_items(n, 4):                                           # 하네스 natural 과 같은 순서 · 같은 32 장
        flow, cons = cell_flow(x)
        buf.append((vid, x, {"flow": flow, "cons": cons}))
        if len(buf) == bs:
            flush()
    flush()
    for k, v in acc.items():
        arr = np.array(v) if k == "video" else np.concatenate(v)
        np.save(CACHE / f"{k}.npy", arr)
    json.dump({"n": len(acc["video"]), "device": str(dev), "fps": 4, "block": [14, 15]}, open(CACHE / "meta.json", "w"))
    print(f"[blockmotion extract] done {len(acc['video'])} -> {CACHE}", flush=True)


# ---------------------------------------------------------------- analyze (CPU)
def _ridge_fit(X, Y, lam):
    A = X.T @ X
    A[np.diag_indices_from(A)] += lam * len(X)
    return np.linalg.solve(A, X.T @ Y)


def _r2(Y, P):
    return 1.0 - ((P - Y) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum()


def fit_predict(Xtr, Ytr, Xte, lams=(1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0), vtr=None, seed=1):
    """표준화 + ridge. 세기는 학습 반 안을 영상 단위로 다시 반 나눠 고른다. 반환 (예측, 고른 세기).
    Gram 행렬을 한 번만 만들고 세기마다 풀기만 한다 (같은 결과, 공용 노드에서 빠르게)."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr_ = np.concatenate([(Xtr - mu) / sd, np.ones((len(Xtr), 1))], 1)
    Xte_ = np.concatenate([(Xte - mu) / sd, np.ones((len(Xte), 1))], 1)
    uv = np.unique(vtr); rng = np.random.default_rng(seed); inner = set(int(v) for v in rng.choice(uv, len(uv) // 2, replace=False))
    m = np.isin(vtr, list(inner))
    Xa, Ya, Xb, Yb = Xtr_[~m], Ytr[~m], Xtr_[m], Ytr[m]
    dual = Xtr_.shape[1] > Xtr_.shape[0]                    # 특징 수 > 표본 수 (DINOv2 16 장 이어 붙임 등): 같은 ridge 를 쌍대형으로
    if dual:
        Ka, Kba = Xa @ Xa.T, Xb @ Xa.T
    else:
        Ga, ba = Xa.T @ Xa, Xa.T @ Ya
    best = None
    for lam in lams:
        if dual:
            A = Ka.copy(); A[np.diag_indices_from(A)] += lam * len(Xa)
            r = _r2(Yb, Kba @ np.linalg.solve(A, Ya))
        else:
            A = Ga.copy(); A[np.diag_indices_from(A)] += lam * len(Xa)
            r = _r2(Yb, Xb @ np.linalg.solve(A, ba))
        if best is None or r > best[0]:
            best = (r, lam)
    if dual:
        A = Xtr_ @ Xtr_.T; A[np.diag_indices_from(A)] += best[1] * len(Xtr_)
        return (Xte_ @ Xtr_.T) @ np.linalg.solve(A, Ytr), best[1]
    A = Xtr_.T @ Xtr_; A[np.diag_indices_from(A)] += best[1] * len(Xtr_)
    W = np.linalg.solve(A, Xtr_.T @ Ytr)
    return Xte_ @ W, best[1]


def analyze(B=1000):
    z = {k: np.load(CACHE / f"{k}.npy", mmap_mode="r") for k in ("a", "d14", "d15", "g14", "g15", "flow", "cons")}
    vids = np.load(CACHE / "video.npy")
    nv = len(vids)
    vid_of_cell = np.repeat(np.arange(nv), G * G)
    rng = np.random.default_rng(0)
    test_v = set(rng.choice(nv, nv // 2, replace=False))
    te = np.array([v in test_v for v in vid_of_cell])
    Y = np.asarray(z["flow"], np.float64).reshape(-1, 2)
    cons = np.asarray(z["cons"]).reshape(-1)
    yy, xx = np.divmod(np.tile(np.arange(G * G), nv), G)
    u, v = (xx + 0.5) / G, (yy + 0.5) / G
    feats = {"pos": np.stack([u, v, u * u, v * v, u * v, np.ones_like(u)], 1)}
    get = lambda k: np.asarray(z[k], np.float32).reshape(nv * G * G, -1)
    feats["a"] = get("a"); feats["b"] = get("d15"); feats["c"] = np.concatenate([get("d14"), get("d15")], 1)
    feats["d"] = get("g15"); feats["e"] = np.concatenate([get("g14"), get("g15")], 1)
    out = {"n_videos": int(nv), "n_cells": int(nv * G * G), "n_test_cells": int(te.sum()),
           "consistent_frac": float(cons.mean()), "flow_px_abs_mean": float(np.abs(Y).mean()),
           "flow_px_std": [float(s) for s in Y.std(0)], "arms": {}}
    preds = {}
    for name, X in feats.items():
        X = X.astype(np.float64)
        P, lam = fit_predict(X[~te], Y[~te], X[te], vtr=vid_of_cell[~te])
        Ysh = Y[~te][np.random.default_rng(2).permutation((~te).sum())]
        Psh, _ = fit_predict(X[~te], Ysh, X[te], vtr=vid_of_cell[~te])
        preds[name] = P
        ct = cons[te]
        out["arms"][name] = {"dim": int(X.shape[1]), "lambda": lam, "R2_test": float(_r2(Y[te], P)),
                             "R2_test_consistent": float(_r2(Y[te][ct], P[ct])), "R2_shuffled_label": float(_r2(Y[te], Psh))}
        print(f"[blockmotion] {name:4s} dim {X.shape[1]:5d} lam {lam:g}  R2 {out['arms'][name]['R2_test']:+.3f}"
              f"  (consistent {out['arms'][name]['R2_test_consistent']:+.3f}, shuffled {out['arms'][name]['R2_shuffled_label']:+.3f})", flush=True)
    # paired differences, bootstrap over held-out videos
    tv = np.array(sorted(test_v)); vte = vid_of_cell[te]
    idx_of = {v_: np.where(vte == v_)[0] for v_ in tv}
    Yt = Y[te]
    def r2_on(P, idx):
        return _r2(Yt[idx], P[idx])
    pairs = (("a", "b"), ("a", "c"), ("a", "e"), ("c", "b"), ("e", "d"), ("a", "d"))
    boot = {p: [] for p in pairs}
    rr = np.random.default_rng(3)
    for _ in range(B):
        pick = rr.choice(tv, len(tv), replace=True)
        idx = np.concatenate([idx_of[v_] for v_ in pick])
        for p in pairs:
            boot[p].append(r2_on(preds[p[0]], idx) - r2_on(preds[p[1]], idx))
    out["paired"] = {f"{p[0]}-{p[1]}": {"diff": float(out["arms"][p[0]]["R2_test"] - out["arms"][p[1]]["R2_test"]),
                                         "ci95": [float(np.percentile(boot[p], 2.5)), float(np.percentile(boot[p], 97.5))]} for p in pairs}
    for k, d in out["paired"].items():
        print(f"[blockmotion] {k}: {d['diff']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT / "result.json", "w"), indent=1)


# ---------------------------------------------------------------- G23b: 문맥 통째 인코딩 (f_t · f_o) + 합성 가림 (정답 2)
def _cell_flow_pair(x, s, t):
    import cv2
    g = CC._denorm_grey(x)
    fw = cv2.calcOpticalFlowFarneback(g[s], g[t], None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return fw.reshape(G, CELL, G, CELL, 2).mean((1, 3)).astype(np.float32)          # (G, G, 2)


def _box(flow_gg, k=4):
    mag = np.linalg.norm(flow_gg, axis=-1)
    best, rc = -1, (0, 0)
    for r in range(G - k + 1):
        for c in range(G - k + 1):
            v = mag[r:r + k, c:c + k].sum()
            if v > best:
                best, rc = v, (r, c)
    return rc


def extract2(n, bs, k=4):
    import torch
    import torch.nn.functional as F
    import passline as PL
    dev = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    model, bundle = PL._load_model("z_training/vast/ar_state/latest.pt", dev)       # dual_encoder: context = online 'encoder', target = EMA
    assert bundle.context_encoder is not bundle.target_encoder, "online / target encoder must be different modules"
    CACHE.mkdir(parents=True, exist_ok=True)
    acc = {key: [] for key in ("f_t", "f_o", "o_a", "o_b", "o_ft", "o_fo", "o_d", "o_e16", "o_flow", "o_box", "video")}
    buf, t0 = [], time.time()
    ln = lambda t: F.layer_norm(t.float(), (t.shape[-1],))

    def last_tubelet(enc, clip):                                                       # clip (B, 3, 16, H, W) → (B, G·G, D)
        h = enc(clip)
        return ln(h[:, -G * G:])

    def flush():
        if not buf:
            return
        x = torch.stack([b[1] for b in buf]).to(dev)
        B = x.shape[0]
        xo = x.clone()
        for j, b in enumerate(buf):
            r, c = b[2]["box"]
            xo[j, :, 14:16, r * CELL:(r + k) * CELL, c * CELL:(c + k) * CELL] = 0.0      # 마지막 블록만 회색 (정규화 0)
        with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            ctx, ctxo = x[:, :, :16].to(bundle.dtype), xo[:, :, :16].to(bundle.dtype)
            f_t, f_o = last_tubelet(bundle.target_encoder, ctx), last_tubelet(bundle.context_encoder, ctx)
            o_ft, o_fo = last_tubelet(bundle.target_encoder, ctxo), last_tubelet(bundle.context_encoder, ctxo)
            f14, f15 = xo[:, :, 14:15], xo[:, :, 15:16]
            hb = CE.enc_vjepa2(bundle, torch.cat([torch.cat([f14, f15], 2), torch.cat([f15, f15], 2)], 0).to(bundle.dtype)).float()[:, 0]
            o_a, o_b = hb[:B].reshape(B, G * G, -1), hb[B:].reshape(B, G * G, -1)
            gd = ln(CE._dino_frames(xo, list(range(16))))                              # (B, 16, G, G, 768)
        for j, b in enumerate(buf):
            r, c = b[2]["box"]
            cells = np.array([(rr * G + cc) for rr in range(r, r + k) for cc in range(c, c + k)])
            acc["f_t"].append(f_t[j].half().cpu().numpy()[None]); acc["f_o"].append(f_o[j].half().cpu().numpy()[None])
            for key, t in (("o_a", o_a), ("o_b", o_b), ("o_ft", o_ft), ("o_fo", o_fo)):
                acc[key].append(t[j, cells].half().cpu().numpy()[None])
            acc["o_d"].append(gd[j, 15].reshape(G * G, -1)[cells].half().cpu().numpy()[None])
            acc["o_e16"].append(gd[j].reshape(16, G * G, -1)[:, cells].permute(1, 0, 2).reshape(len(cells), -1).half().cpu().numpy()[None])
            acc["o_flow"].append(b[2]["flow1314"].reshape(G * G, 2)[cells][None]); acc["o_box"].append(np.array([[r, c]])); acc["video"].append(b[0])
        buf.clear()
        print(f"[blockmotion extract2] {len(acc['video'])} videos ({time.time() - t0:.0f}s, {dev})", flush=True)

    for vid, x, _ in CC.natural_items(n, 4):
        f1314 = _cell_flow_pair(x, 13, 14)
        buf.append((vid, x, {"flow1314": f1314, "box": _box(f1314, k)}))
        if len(buf) == bs:
            flush()
    flush()
    vids0 = list(np.load(CACHE / "video.npy")) if (CACHE / "video.npy").exists() else None
    assert vids0 is None or vids0 == acc["video"], "extract2 must use the same videos as extract"
    for key, v in acc.items():
        np.save(CACHE / f"{key}.npy", np.array(v) if key == "video" else np.concatenate(v))
    print(f"[blockmotion extract2] done {len(acc['video'])}", flush=True)


def analyze2(B=1000):
    """G23b: 정답 1 에 (f_t · f_o) 를 더하고, 정답 2 (합성 가림, 가려지는 순간 움직임) 의 팔 · 기준선 · 짝지은 차."""
    nv = len(np.load(CACHE / "video.npy"))
    rng = np.random.default_rng(0)
    test_v = set(rng.choice(nv, nv // 2, replace=False))                               # G23 과 같은 보류 분할
    out = {"target1": {}, "target2": {}}
    # ---- target 1 (보이는 내용, 흐름 14 → 15), 같은 칸 · 같은 분할
    vid_of = np.repeat(np.arange(nv), G * G)
    te = np.array([v in test_v for v in vid_of])
    Y = np.asarray(np.load(CACHE / "flow.npy"), np.float64).reshape(-1, 2)
    getf = lambda key: np.asarray(np.load(CACHE / f"{key}.npy", mmap_mode="r"), np.float32).reshape(nv * G * G, -1).astype(np.float64)
    preds1 = {}
    for name in ("a", "f_t", "f_o"):
        X = getf("a") if name == "a" else getf(name)
        P, lam = fit_predict(X[~te], Y[~te], X[te], vtr=vid_of[~te])
        preds1[name] = P
        out["target1"][name] = {"R2_test": float(_r2(Y[te], P)), "lambda": lam}
        print(f"[G23b t1] {name:4s} R2 {out['target1'][name]['R2_test']:+.3f}", flush=True)
    # ---- target 2 (합성 가림 16 칸)
    k2 = np.load(CACHE / "o_flow.npy").shape[1]
    vid2 = np.repeat(np.arange(nv), k2)
    te2 = np.array([v in test_v for v in vid2])
    Y2 = np.asarray(np.load(CACHE / "o_flow.npy"), np.float64).reshape(-1, 2)
    box = np.load(CACHE / "o_box.npy")                                                  # (nv, 2) r, c
    kk = int(round(np.sqrt(k2)))
    rr, cc = np.divmod(np.tile(np.arange(k2), nv), kk)
    br, bc = np.repeat(box[:, 0], k2), np.repeat(box[:, 1], k2)
    cy, cx = (br + rr + 0.5) / G, (bc + cc + 0.5) / G
    poly = lambda u, v: np.stack([u, v, u * u, v * v, u * v, np.ones_like(u)], 1)
    feats = {"pos_cell": poly(cx, cy),
             "occluder_pos": np.concatenate([poly((bc + kk / 2) / G, (br + kk / 2) / G), np.eye(k2)[np.tile(np.arange(k2), nv)]], 1)}
    for name, key in (("a", "o_a"), ("b", "o_b"), ("f_t", "o_ft"), ("f_o", "o_fo"), ("d", "o_d"), ("e16", "o_e16")):
        feats[name] = np.asarray(np.load(CACHE / f"{key}.npy"), np.float32).reshape(nv * k2, -1).astype(np.float64)
    preds2 = {}
    for name, X in feats.items():
        P, lam = fit_predict(X[~te2], Y2[~te2], X[te2], vtr=vid2[~te2])
        Ysh = Y2[~te2][np.random.default_rng(2).permutation((~te2).sum())]
        Psh, _ = fit_predict(X[~te2], Ysh, X[te2], vtr=vid2[~te2])
        preds2[name] = P
        out["target2"][name] = {"dim": int(X.shape[1]), "lambda": lam, "R2_test": float(_r2(Y2[te2], P)), "R2_shuffled": float(_r2(Y2[te2], Psh))}
        print(f"[G23b t2] {name:12s} dim {X.shape[1]:6d} R2 {out['target2'][name]['R2_test']:+.3f} (shuffled {out['target2'][name]['R2_shuffled']:+.3f})", flush=True)
    # ---- paired bootstrap over held-out videos
    def boot_pairs(preds, Yt, vte, pairs):
        tv = np.array(sorted(set(vte.tolist()))); idx_of = {v_: np.where(vte == v_)[0] for v_ in tv}
        bs = {p: [] for p in pairs}; rr_ = np.random.default_rng(3)
        for _ in range(B):
            idx = np.concatenate([idx_of[v_] for v_ in rr_.choice(tv, len(tv), replace=True)])
            for p in pairs:
                bs[p].append(_r2(Yt[idx], preds[p[0]][idx]) - _r2(Yt[idx], preds[p[1]][idx]))
        return {f"{p[0]}-{p[1]}": {"diff": float(_r2(Yt, preds[p[0]]) - _r2(Yt, preds[p[1]])),
                                   "ci95": [float(np.percentile(bs[p], 2.5)), float(np.percentile(bs[p], 97.5))]} for p in pairs}
    out["target1_paired"] = boot_pairs(preds1, Y[te], vid_of[te], (("f_t", "a"), ("f_o", "a")))
    out["target2_paired"] = boot_pairs(preds2, Y2[te2], vid2[te2], (("f_t", "a"), ("f_o", "a"), ("f_t", "d"), ("f_o", "d"), ("f_t", "e16"), ("f_o", "e16"), ("e16", "a"), ("a", "b")))
    for tgt in ("target1_paired", "target2_paired"):
        for kk_, d in out[tgt].items():
            print(f"[G23b {tgt}] {kk_}: {d['diff']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT / "result_g23b.json", "w"), indent=1)


# ---------------------------------------------------------------- G23c: 양성 대조 (원 화소 MLP) + 같은 MLP 를 토큰에
def frames(n):
    """같은 400 영상의 프레임 14 · 15 회색조 (256 px) 를 캐시한다 (CPU, 디코드만 다시)."""
    vids0 = list(np.load(CACHE / "video.npy"))
    out, vids = [], []
    for vid, x, _ in CC.natural_items(n, 4):
        out.append(CC._denorm_grey(x)[14:16][None]); vids.append(vid)
        if len(vids) % 50 == 0:
            print(f"[blockmotion frames] {len(vids)}", flush=True)
    assert vids == vids0, "frames must use the same videos as extract"
    np.save(CACHE / "grey1415.npy", np.concatenate(out))
    print(f"[blockmotion frames] done {len(vids)}", flush=True)


def _pix_feats(grey):
    """grey (nv, 2, 256, 256) uint8 → 칸마다 이웃 3 × 3 칸 (48 × 48 px) 두 장 이어 붙임 (nv·256, 4608) float32."""
    nv = grey.shape[0]
    pad = np.pad(grey, ((0, 0), (0, 0), (CELL, CELL), (CELL, CELL)), mode="edge").astype(np.float32) / 255.0
    X = np.empty((nv * G * G, 2 * 48 * 48), np.float32)
    k = 0
    for v in range(nv):
        for r in range(G):
            for c in range(G):
                X[k] = pad[v, :, r * CELL:r * CELL + 48, c * CELL:c * CELL + 48].reshape(-1); k += 1
    return X


def _mlp_fit_predict(Xtr, Ytr, Xte, vtr, seed=0, hidden=256, epochs=40, patience=5, bs=512):
    import torch
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    ym, ys = Ytr.mean(0), Ytr.std(0)
    uv = np.unique(vtr); rng = np.random.default_rng(seed + 1)
    val_v = set(int(v) for v in rng.choice(uv, max(1, len(uv) // 5), replace=False))
    mv = np.isin(vtr, list(val_v))
    T = lambda a: torch.as_tensor(a, dtype=torch.float32)
    Xa, Ya = T((Xtr[~mv] - mu) / sd), T((Ytr[~mv] - ym) / ys)
    Xv, Yv = T((Xtr[mv] - mu) / sd), T((Ytr[mv] - ym) / ys)
    net = torch.nn.Sequential(torch.nn.Linear(Xtr.shape[1], hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 2))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    best, best_state, bad, ep_used = 1e9, None, 0, 0
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(Xa))
        for i in range(0, len(Xa), bs):
            idx = perm[i:i + bs]
            loss = ((net(Xa[idx]) - Ya[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl = ((net(Xv) - Yv) ** 2).mean().item()
        if vl < best - 1e-5:
            best, best_state, bad, ep_used = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0, ep + 1
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state); net.eval()
    with torch.no_grad():
        P = net(T((Xte - mu) / sd)).numpy() * ys + ym
    return P, ep_used


def analyze3(B=1000):
    nv = len(np.load(CACHE / "video.npy"))
    vid_of = np.repeat(np.arange(nv), G * G)
    rng = np.random.default_rng(0)
    test_v = set(rng.choice(nv, nv // 2, replace=False))                               # G23 과 같은 보류 분할
    te = np.array([v in test_v for v in vid_of])
    Y = np.asarray(np.load(CACHE / "flow.npy"), np.float64).reshape(-1, 2)
    get = lambda k: np.asarray(np.load(CACHE / f"{k}.npy", mmap_mode="r"), np.float32).reshape(nv * G * G, -1)
    feats = {"pixel": _pix_feats(np.load(CACHE / "grey1415.npy")), "a": get("a"), "b": get("d15"),
             "e": np.concatenate([get("g14"), get("g15")], 1)}
    out = {"arms": {}}; preds = {}
    for name, X in feats.items():
        t0 = time.time()
        P, ep = _mlp_fit_predict(X[~te], Y[~te], X[te], vid_of[~te])
        preds[name] = P
        out["arms"][name] = {"dim": int(X.shape[1]), "epochs": ep, "R2_test": float(_r2(Y[te], P))}
        print(f"[G23c] {name:6s} dim {X.shape[1]:5d} epochs {ep:2d}  MLP R2 {out['arms'][name]['R2_test']:+.3f}  ({time.time() - t0:.0f}s)", flush=True)
    tv = np.array(sorted(test_v)); vte = vid_of[te]; idx_of = {v_: np.where(vte == v_)[0] for v_ in tv}; Yt = Y[te]
    pairs = (("a", "b"), ("pixel", "a"), ("a", "e"))
    bs = {p: [] for p in pairs}; rr = np.random.default_rng(3)
    for _ in range(B):
        idx = np.concatenate([idx_of[v_] for v_ in rr.choice(tv, len(tv), replace=True)])
        for p in pairs:
            bs[p].append(_r2(Yt[idx], preds[p[0]][idx]) - _r2(Yt[idx], preds[p[1]][idx]))
    out["paired"] = {f"{p[0]}-{p[1]}": {"diff": float(out["arms"][p[0]]["R2_test"] - out["arms"][p[1]]["R2_test"]),
                                         "ci95": [float(np.percentile(bs[p], 2.5)), float(np.percentile(bs[p], 97.5))]} for p in pairs}
    for k_, d in out["paired"].items():
        print(f"[G23c] {k_}: {d['diff']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT / "result_g23c.json", "w"), indent=1)


def main():
    import torch
    torch.set_num_threads(int(os.environ.get("CARRY_THREADS", 4)))
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("extract"); p.add_argument("--n", type=int, default=400); p.add_argument("--bs", type=int, default=8)
    sub.add_parser("analyze")
    p = sub.add_parser("extract2"); p.add_argument("--n", type=int, default=400); p.add_argument("--bs", type=int, default=4)
    sub.add_parser("analyze2")
    p = sub.add_parser("frames"); p.add_argument("--n", type=int, default=400)
    sub.add_parser("analyze3")
    a = ap.parse_args()
    {"extract": lambda: extract(a.n, a.bs), "analyze": analyze, "extract2": lambda: extract2(a.n, a.bs), "analyze2": analyze2,
     "frames": lambda: frames(a.n), "analyze3": analyze3}[a.cmd]()


if __name__ == "__main__":
    main()
