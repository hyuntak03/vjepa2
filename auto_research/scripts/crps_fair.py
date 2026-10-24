#!/usr/bin/env python3
"""[T] 분포 예측 (CRPS) 팔의 공정 비교 (학습 없음) — 사전 등록 진행판 [T] KST 16:46 (auto_research, 2026-10-22).

  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=7 /venv/vjepa/bin/python -W ignore auto_research/scripts/crps_fair.py [--limit N] [--K 8]

모델: 기준 블록 AR (점 예측) · 분포 예측 팔 (/workspace/auto_research_cache/ar_ft/ft_crpsxi.pt).
표본 k = g(문맥, z_k) + s ⊙ ξ_k, z_k ~ N(0, I_32) (창마다 하나, 블록 공유), ξ_k = Laplace(0,1)/√2 좌표마다 (학습과 같음), s = 저장된 척도.
세트: K710 보류 (150 창) · SSv2 수동 · 행위자 (각 300 창), TF 한 걸음 (블록 0..6 → 1..7).
(1) L1: 기준 · 표본 하나 · g 하나 · 표본 평균 · g 평균.
(2) 좌표별 공정 CRPS (표본 K) · 블록 단위 에너지 점수 (칸 × 차원 L2 ÷ √(256·1280), 표본 K). 기준은 한 점 분포 (CRPS = L1, ES = L2/√N).
    보충 (스모크 뒤 추가, 등록 판정 밖): 기준 점 예측 + 같은 ξ 잡음 (표본 K) — 한 점 분포는 너무 약한 기준이라 "잡음 모형만으로 얻는 몫" 을 뺀다.
(3) 위약 뺀 운반 (transport_placebo.py 와 같은 회귀: 움직이는 칸, t = 4 · 5 · 6, y ≈ a · 같은 칸 5 블록 평균 + b · 옮긴 h_t; 옮김 = 정답 흐름 · 90° 돌린 흐름):
    데이터 · 복사 정답 · 기준 · 표본 하나 · g 하나 · 표본 평균 · g 평균. CI = 창 bootstrap, 기준 대비 차이는 짝 bootstrap.
출력: results/crps_fair.json."""
import argparse, json, os, sys, time
import numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--nb", type=int, default=300)
ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "../results/crps_fair.json"))
a = ap.parse_args()
sys.path.insert(0, os.path.dirname(__file__))
_argv = sys.argv; sys.argv = [sys.argv[0], "--loss", "l1", "--name", "x"]
import ar_crps_ft as FT                                                     # noqa: E402
sys.argv = _argv
torch.set_num_threads(8)
dev = torch.device("cuda")
G, S, D = 16, 256, 1280
cy, cx = np.divmod(np.arange(S), G)
P0 = np.stack([cx + 0.5, cy + 0.5], 1)
R90 = np.array([[0.0, -1.0], [1.0, 0.0]], np.float32)
SETS = {"k710_holdout": ("/workspace/auto_research_cache/natural_holdout/k710_320_holdout_h.npy", "/workspace/auto_research_cache/natural_holdout/k710_320_holdout_U.npy"),
        "ssv2_passive": ("/workspace/auto_research_cache/ssv2_classes/passive_h.npy", "/workspace/auto_research_cache/ssv2_classes/passive_U.npy"),
        "ssv2_actor": ("/workspace/auto_research_cache/ssv2_classes/actor_h.npy", "/workspace/auto_research_cache/ssv2_classes/actor_U.npy")}

base = FT.NoiseWrap(FT.load_pred("natural_ar_vw_state", "latest.pt"), 32, 0.0).to(dev).eval()
ck = torch.load("/workspace/auto_research_cache/ar_ft/ft_crpsxi.pt", map_location="cpu", weights_only=False)
cm = FT.NoiseWrap(FT.load_pred("natural_ar_vw_state", "latest.pt"), int(ck["args"].get("nz", 32)), 0.0)
cm.pred.load_state_dict(ck["predictor"]); cm.proj.load_state_dict(ck["noise_proj"])
cm = cm.to(dev).eval()
scale = ck["log_s"].exp().to(dev)                                           # (1280,)
NZ = int(ck["args"].get("nz", 32))
lap = torch.distributions.Laplace(0.0, 1.0)
VARS = ["base", "X1", "g1", "Xmean", "gmean"]


def warp(ht, u):
    s = np.floor(P0 + u).astype(int)
    ok = (s >= 0).all(1) & (s < G).all(1)
    j = np.clip(s[:, 1], 0, G - 1) * G + np.clip(s[:, 0], 0, G - 1)
    out = ht[j].copy(); out[~ok] = ht[~ok]
    return out


def fair_pair_mean(Xs, fn):                                                # Σ_{k≠l} fn(X_k − X_l) / (K(K−1))
    K = len(Xs); tot = 0.0
    for k in range(K):
        for l in range(k + 1, K):
            tot += 2 * fn(Xs[k] - Xs[l])
    return tot / (K * (K - 1))


RES = {}
for sn, (hp, up) in SETS.items():
    H = np.load(hp, mmap_mode="r"); U = np.load(up).astype(np.float32).reshape(-1, 8, S, 2)
    n = H.shape[0] if not a.limit else min(a.limit, H.shape[0])
    sc = {k: [0.0, 0] for k in ["l1_" + v for v in VARS] + ["crps_X", "crps_g", "crps_baseXi", "es_X", "es_g", "es_base", "es_baseXi"]}
    recs = []
    t0 = time.time()
    gen = torch.Generator(device=dev)
    for w in range(n):
        h = torch.from_numpy(np.asarray(H[w], np.float32)).to(dev)
        x = h.reshape(1, -1, D); y = h[1:]
        gen.manual_seed(1000 * w + 7)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            pb = base.tf(x).float()[0].view(8, S, D)[:7]
            gs = [cm.tf(x, torch.randn(1, NZ, device=dev, generator=gen)).float()[0].view(8, S, D)[:7] for _ in range(a.K)]
        Xs = [g + scale * (lap.sample(g.shape).to(dev) / np.sqrt(2)) for g in gs]
        P = {"base": pb, "X1": Xs[0], "g1": gs[0], "Xmean": torch.stack(Xs).mean(0), "gmean": torch.stack(gs).mean(0)}
        for v in VARS:
            sc["l1_" + v][0] += float((P[v] - y).abs().mean()); sc["l1_" + v][1] += 1
        Bs = [pb + scale * (lap.sample(pb.shape).to(dev) / np.sqrt(2)) for _ in range(a.K)]     # 보충 (스모크 뒤 추가): 기준 점 예측 + 같은 ξ 잡음
        for nm, smp in (("X", Xs), ("g", gs), ("baseXi", Bs)):
            fit = float(torch.stack([(s_ - y).abs().mean() for s_ in smp]).mean())
            sc["crps_" + nm][0] += fit - 0.5 * fair_pair_mean(smp, lambda d: float(d.abs().mean())); sc["crps_" + nm][1] += 1
            nrm = lambda d: float(d.pow(2).sum((1, 2)).sqrt().mean() / np.sqrt(S * D))          # 블록마다 L2 → 7 블록 평균
            sc["es_" + nm][0] += float(np.mean([nrm(s_ - y) for s_ in smp])) - 0.5 * fair_pair_mean(smp, nrm); sc["es_" + nm][1] += 1
        sc["es_base"][0] += float((pb - y).pow(2).sum((1, 2)).sqrt().mean() / np.sqrt(S * D)); sc["es_base"][1] += 1
        hn = h.cpu().numpy(); Pn = {v: P[v][4:7].cpu().numpy() for v in VARS}
        for t in (4, 5, 6):
            sel = np.linalg.norm(U[w, t + 1], axis=1) >= 0.5
            if sel.sum() < 2:
                continue
            m_ = hn[t - 4:t + 1].mean(0)
            r = dict(win=w)
            ys = {"data": hn[t + 1], "copy": hn[t], **{v: Pn[v][t - 4] for v in VARS}}
            for kind in ("true", "rot"):
                Wk = warp(hn[t], U[w, t + 1] if kind == "true" else U[w, t + 1] @ R90.T)
                X = np.stack([m_[sel].ravel(), Wk[sel].ravel()], 1).astype(np.float64)
                r[f"XtX_{kind}"] = X.T @ X
                for nm, yy in ys.items():
                    r[f"Xty_{kind}_{nm}"] = X.T @ yy[sel].ravel().astype(np.float64)
            recs.append(r)
        if (w + 1) % 50 == 0:
            print(f"  {sn} {w + 1}/{n} ({time.time() - t0:.0f}s)", flush=True)

    def bfit(rr, kind, nm):
        A = sum(r[f"XtX_{kind}"] for r in rr); bv = sum(r[f"Xty_{kind}_{nm}"] for r in rr)
        return np.linalg.solve(A, bv)[1]

    rng = np.random.RandomState(0)
    wins = np.array([r["win"] for r in recs]); uw = np.unique(wins); ix = {u: np.where(wins == u)[0] for u in uw}
    boots = [np.concatenate([ix[u] for u in rng.choice(uw, len(uw))]) for _ in range(a.nb)]
    E = {"n_windows": int(n), "n_reg_windows": int(len(uw)), "scores": {k: v[0] / max(v[1], 1) for k, v in sc.items()}, "transport": {}}
    bb = {}
    for nm in ["data", "copy"] + VARS:
        bt, br = bfit(recs, "true", nm), bfit(recs, "rot", nm)
        bs = []
        for ii in boots:
            rr = [recs[i] for i in ii]; bs.append(bfit(rr, "true", nm) - bfit(rr, "rot", nm))
        bb[nm] = np.array(bs)
        e = dict(b_true=float(bt), b_rot=float(br), net=float(bt - br), ci=np.percentile(bs, [2.5, 97.5]).tolist())
        if nm in VARS and nm != "base":
            d = bb[nm] - bb["base"]
            e["diff_vs_base"] = float((bt - br) - E["transport"]["base"]["net"]); e["diff_ci"] = np.percentile(d, [2.5, 97.5]).tolist()
        E["transport"][nm] = e
    RES[sn] = E
    s_ = E["scores"]
    print(f"== {sn} (창 {n}): L1 기준 {s_['l1_base']:.4f} · 표본 하나 {s_['l1_X1']:.4f} · g 하나 {s_['l1_g1']:.4f} · 표본 평균 {s_['l1_Xmean']:.4f} · g 평균 {s_['l1_gmean']:.4f}", flush=True)
    print(f"   CRPS: 기준 (= L1) {s_['l1_base']:.4f} · 기준 + 같은 ξ {s_['crps_baseXi']:.4f} · 분포 (g + ξ) {s_['crps_X']:.4f} · g 만 {s_['crps_g']:.4f} | 에너지 점수: 기준 {s_['es_base']:.4f} · 기준 + ξ {s_['es_baseXi']:.4f} · 분포 {s_['es_X']:.4f} · g 만 {s_['es_g']:.4f}", flush=True)
    for nm, e in E["transport"].items():
        print(f"   운반 {nm:6s} b_true {e['b_true']:.4f} · b_rot {e['b_rot']:.4f} · 위약 뺀 {e['net']:+.4f} {np.round(e['ci'], 4).tolist()}"
              + (f" · 기준 대비 {e['diff_vs_base']:+.4f} {np.round(e['diff_ci'], 4).tolist()}" if "diff_vs_base" in e else ""), flush=True)
json.dump(RES, open(a.out if not a.limit else a.out.replace(".json", "_smoke.json"), "w"), indent=1)
print("저장", flush=True)
