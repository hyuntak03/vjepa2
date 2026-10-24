#!/usr/bin/env python3
"""[T] 방법 개입 판정: 3D 합성 val 에서 기준 블록 AR · L1 이어 학습 · 공정 CRPS 이어 학습 (표본별) 의 새 자리 존재감 · 옛 자리 잔상 (auto_research, 2026-10-21).

  CUDA_VISIBLE_DEVICES=1 /venv/vjepa/bin/python -W ignore auto_research/scripts/ar_crps_eval.py --ft ft_l1 ft_crpsxi --K 8

정의는 [P] 의 고리 3 · 4 측정과 같다: TF 한 걸음 (문맥 = 정답 블록 0..t, t = 7..14 → 블록 t+1), 움직인 쌍 (변위 > 8 px, 두 블록 모두 보임).
물체 칸 obj_b = 블록 b 두 장 중 bbox 덮임 ≥ 0.25. new = obj_{t+1} − obj_t, old = obj_t − obj_{t+1}, far = 창 내내 물체가 안 닿은 칸 (2 칸 팽창의 여집합).
물체 점수 s(x) = x · T̂ (T̂ = 블록 t 물체 칸 평균 토큰, 길이 1), 대비 = 무리 칸 평균 − far 칸 평균.
존재감 (new) = [대비(x) − 대비(copy)] / [대비(정답) − 대비(copy)] · 잔상 (old) = [대비(x) − 대비(정답)] / [대비(copy) − 대비(정답)] (쌍 전체 평균의 비).
분포 예측은 g(H, z) 표본 K 개 (ξ 없이). 표본별: 쌍마다 정규화한 존재감 · 잔상의 분포, 쌍 안 표본 사이 퍼짐, 쌍 안에서 존재감 ↔ 잔상 상관 (갈래면 음수)."""
import argparse, glob, json, os, sys
import numpy as np, torch, torch.nn.functional as F

sys.argv_backup = list(sys.argv)
ap = argparse.ArgumentParser()
ap.add_argument("--ft", nargs="*", default=[])
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--ftdir", default="/workspace/auto_research_cache/ar_ft")
ap.add_argument("--cache", default="/workspace/auto_research_cache/phys3d")
ap.add_argument("--limit", type=int, default=0)
args = ap.parse_args()
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.dirname(__file__))
sys.argv = [sys.argv[0], "--loss", "l1", "--name", "x"]                      # ar_crps_ft 의 인자 해석을 통과시키기 위함
import ar_crps_ft as FT                                                      # noqa: E402
dev = torch.device("cuda")
G, S, NB = 16, 256, 16


def cover(bbox):
    x0, y0, x1, y1 = [bbox[..., i, None] for i in range(4)]
    c = np.arange(G) * 16.0
    ox = np.clip(np.minimum(c + 16, x1) - np.maximum(c, x0), 0, 16) / 16
    oy = np.clip(np.minimum(c + 16, y1) - np.maximum(c, y0), 0, 16) / 16
    return oy[..., :, None] * ox[..., None, :]


def dil(m, k=1):
    for _ in range(k):
        o = m.copy()
        o[1:] |= m[:-1]; o[:-1] |= m[1:]; o[:, 1:] |= m[:, :-1]; o[:, :-1] |= m[:, 1:]
        o[1:, 1:] |= m[:-1, :-1]; o[:-1, :-1] |= m[1:, 1:]; o[1:, :-1] |= m[:-1, 1:]; o[:-1, 1:] |= m[1:, :-1]
        m = o
    return m


models = {"base": (FT.NoiseWrap(FT.load_pred("natural_ar_vw_state", "latest.pt"), 32, 0.0).to(dev).eval(), "l1")}
import disp_ft as DF                                                         # noqa: E402
for name in args.ft:
    ck = torch.load(os.path.join(args.ftdir, f"{name}.pt"), map_location="cpu", weights_only=False)
    if ck.get("kind") == "disp":
        Wd = DF.DispWrap(FT.load_pred("natural_ar_vw_state", "latest.pt"))
        Wd.pred.load_state_dict(ck["predictor"]); Wd.u_head.load_state_dict(ck["u_head"]); Wd.m_head.load_state_dict(ck["m_head"])
        models[name] = (Wd.to(dev).eval(), "l1")
        continue
    W = FT.NoiseWrap(FT.load_pred("natural_ar_vw_state", "latest.pt"), int(ck["args"].get("nz", 32)), 0.0)
    W.pred.load_state_dict(ck["predictor"])
    if "noise_proj" in ck:
        W.proj.load_state_dict(ck["noise_proj"])
    if ck.get("codes") is not None:
        W.codes = torch.nn.Parameter(ck["codes"])
    models[name] = (W.to(dev).eval(), ck["args"].get("loss", "l1"))
REC = []
REG = {}                                                                     # (name, k) → [XtX (9,9), Xty (9)] — 새 칸에서 y ≈ Σ_lag w h_{t−lag} + β · 옮긴 h_t
def reg_add(key, Xn, yn):
    A = REG.setdefault(key, [np.zeros((9, 9)), np.zeros(9)])
    A[0] += Xn.T @ Xn; A[1] += Xn.T @ yn
for f in sorted(glob.glob(os.path.join(args.cache, "h_val_target_s*.npy"))):
    sh = f.split("_s")[-1].split(".")[0]
    lab = np.load(os.path.join(args.cache, f"lab_val_s{sh}.npz"))
    X = np.load(f, mmap_mode="r")
    for w in range(X.shape[0] if not args.limit else min(args.limit, X.shape[0])):
        h = torch.from_numpy(X[w].astype(np.float32)).to(dev)                      # (16,256,D)
        cv = cover(np.nan_to_num(lab["bbox"][w], nan=-100.0)); cvb = np.maximum(cv[0::2], cv[1::2])
        objb = cvb >= 0.25
        ok = lab["inframe_frac"][w] > 0.9
        vis_b = (lab["vis_frac"][w][0::2] > 0.7) & (lab["vis_frac"][w][1::2] > 0.7) & ok[0::2] & ok[1::2]
        pos = lab["pos"][w]
        far = torch.from_numpy(~dil(objb.any(0), 2).reshape(-1)).to(dev)
        pairs = []
        for t in range(7, NB - 1):
            d = pos[2 * t + 3] - pos[2 * t + 1]; dn = np.linalg.norm(d)
            if not (np.isfinite(dn) and dn > 8 and vis_b[t] and vis_b[t + 1]):
                continue
            new = (objb[t + 1] & ~objb[t]).reshape(-1); old = (objb[t] & ~objb[t + 1]).reshape(-1)
            if new.sum() == 0 or old.sum() == 0 or far.sum() == 0:
                continue
            pairs.append((t, torch.from_numpy(new).to(dev), torch.from_numpy(old).to(dev), torch.from_numpy(objb[t].reshape(-1)).to(dev), d))
        if not pairs:
            continue
        x = h[:NB - 1].reshape(1, -1, h.size(-1))
        outs = {}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for name, (W, kind) in models.items():
                if kind == "crps":
                    outs[name] = torch.stack([W.tf(x, torch.randn(1, W.nz, device=dev)).float()[0].view(NB - 1, S, -1) for _ in range(args.K)])
                elif kind == "wta":
                    outs[name] = torch.stack([W.tf(x, code=k).float()[0].view(NB - 1, S, -1) for k in range(W.codes.shape[0])])
                else:
                    outs[name] = W.tf(x).float()[0].view(NB - 1, S, -1)[None]
        for t, new, old, objt, d in pairs:
            dx, dy = int(round(d[0] / 16)), int(round(d[1] / 16))
            mc = torch.roll(torch.roll(h[t].view(G, G, -1), dy, 0), dx, 1).reshape(S, -1)
            Xn = torch.stack([h[t - k][new] for k in range(8)] + [mc[new]], -1).reshape(-1, 9).double().cpu().numpy()
            reg_add(("true", 0), Xn, h[t + 1][new].reshape(-1).double().cpu().numpy())
            for name, o in outs.items():
                for k in range(o.shape[0]):
                    reg_add((name, k), Xn, o[k, t][new].reshape(-1).double().cpu().numpy())
            T_ = F.normalize(h[t][objt].mean(0), dim=-1)
            con = lambda blk, m: float((blk[m] @ T_).mean() - (blk[far] @ T_).mean())
            r = {"t": t, "new": {"copy": con(h[t], new), "true": con(h[t + 1], new)}, "old": {"copy": con(h[t], old), "true": con(h[t + 1], old)}}
            for name, o in outs.items():
                r["new"][name] = [con(o[k, t], new) for k in range(o.shape[0])]
                r["old"][name] = [con(o[k, t], old) for k in range(o.shape[0])]
            REC.append(r)
    print(f"{f}: 쌍 {len(REC)}", flush=True)

res = {"n_pairs": len(REC)}
print(f"\n쌍 {len(REC)} (TF 한 걸음, 움직인 쌍)")
cn = lambda g, k: np.array([r[g][k] for r in REC])
for name in models:
    Pn, Po = np.array([r["new"][name] for r in REC]), np.array([r["old"][name] for r in REC])      # (n, K)
    dn_ = cn("new", "true") - cn("new", "copy"); do_ = cn("old", "copy") - cn("old", "true")
    pres_mean = float((Pn.mean(1).mean() - cn("new", "copy").mean()) / dn_.mean())
    resid_mean = float((Po.mean(1).mean() - cn("old", "true").mean()) / do_.mean())
    ent = {"presence_new_samplemean": pres_mean, "residual_old_samplemean": resid_mean, "K": int(Pn.shape[1])}
    if Pn.shape[1] > 1:
        pres_k = float((Pn.mean() - cn("new", "copy").mean()) / dn_.mean())
        # 쌍마다 정규화 (분모가 큰 절반만)
        big = (np.abs(dn_) > np.median(np.abs(dn_))) & (np.abs(do_) > np.median(np.abs(do_)))
        pn = (Pn[big] - cn("new", "copy")[big, None]) / dn_[big, None]
        po = (Po[big] - cn("old", "true")[big, None]) / do_[big, None]
        within = float(pn.std(1).mean()); between = float(pn.mean(1).std())
        cc = [np.corrcoef(pn[i], po[i])[0, 1] for i in range(pn.shape[0]) if pn[i].std() > 1e-6 and po[i].std() > 1e-6]
        ent.update(presence_new_per_sample_quantiles=np.percentile(pn, [5, 25, 50, 75, 95]).tolist(),
                   residual_old_per_sample_quantiles=np.percentile(po, [5, 25, 50, 75, 95]).tolist(),
                   within_pair_sample_std=within, between_pair_std=between, corr_new_old_within_pair_median=float(np.median(cc)) if cc else None,
                   hist_presence=np.histogram(pn, bins=12, range=(-0.5, 1.5))[0].tolist())
    bt = np.linalg.solve(REG[("true", 0)][0], REG[("true", 0)][1])[8]
    betas = [float(np.linalg.solve(REG[(name, k)][0], REG[(name, k)][1])[8] / bt) for k in range(Pn.shape[1])]
    pk = [float((Pn[:, k].mean() - cn("new", "copy").mean()) / dn_.mean()) for k in range(Pn.shape[1])]
    rk = [float((Po[:, k].mean() - cn("old", "true").mean()) / do_.mean()) for k in range(Pn.shape[1])]
    ent.update(beta_true=float(bt), transport_ratio_per_output=betas, presence_per_output=pk, residual_per_output=rk,
               corr_presence_residual_across_outputs=float(np.corrcoef(pk, rk)[0, 1]) if len(pk) > 2 else None)
    res[name] = ent
    print(f"  {name:12s} 운반 비 (출력별) {np.round(betas, 2).tolist()} · 존재감 (출력별) {np.round(pk, 2).tolist()} · 잔상 (출력별) {np.round(rk, 2).tolist()}", flush=True)
    print(f"  {name:12s} 존재감 (표본 평균) {pres_mean:.2f} · 잔상 {resid_mean:.2f}"
          + (f" | 표본별 존재감 분위 5/25/50/75/95 {np.round(ent['presence_new_per_sample_quantiles'], 2).tolist()} · 쌍 안 퍼짐 {ent['within_pair_sample_std']:.3f}"
             f" (쌍 사이 {ent['between_pair_std']:.3f}) · 쌍 안 존재감↔잔상 상관 중앙 {ent['corr_new_old_within_pair_median']}" if Pn.shape[1] > 1 else ""), flush=True)
# ---- 자연 영상 ([E] 짧은 창, 흐름 기반): 출력마다 b = (같은 칸 5 블록 평균, 실제 흐름으로 옮긴 h_t) 회귀의 옮긴 쪽 계수 vs 데이터 b
EN = "/workspace/auto_research_cache/enc_noise"; RD = os.path.join(os.path.dirname(__file__), "../results/natural_transport_ctx5")
cy, cx = np.divmod(np.arange(S), G); P0 = np.stack([cx + 0.5, cy + 0.5], 1)
res["natural"] = {}
for src in ("k710", "ssv2", "ht100m"):
    U = np.load(os.path.join(RD, f"rows_{src}.npz"))["Uwin"].astype(np.float32)
    NR = {}
    w0 = 0
    for sh in (0, 1):
        T_ = np.load(os.path.join(EN, f"{src}_target_encoder_s{sh}_tok.npy"), mmap_mode="r")
        nw = T_.shape[0] if not args.limit else min(args.limit, T_.shape[0])
        for w in range(nw):
            h = torch.from_numpy(np.asarray(T_[w, 0]).astype(np.float32)).to(dev)     # (8,256,D)
            x = h[:7].reshape(1, -1, h.size(-1))
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                outs = {}
                for name, (W, kind) in models.items():
                    if kind == "crps":
                        outs[name] = torch.stack([W.tf(x, torch.randn(1, W.nz, device=dev)).float()[0].view(7, S, -1) for _ in range(args.K)])
                    elif kind == "wta":
                        outs[name] = torch.stack([W.tf(x, code=k).float()[0].view(7, S, -1) for k in range(W.codes.shape[0])])
                    else:
                        outs[name] = W.tf(x).float()[0].view(7, S, -1)[None]
            for t in (4, 5, 6):
                u1 = U[w0 + w, t + 1].reshape(S, 2)
                mov = torch.from_numpy(np.linalg.norm(u1, axis=1) >= 0.5).to(dev)
                if mov.sum() < 4:
                    continue
                sp = np.floor(P0 + u1).astype(int); okp = (sp >= 0).all(1) & (sp < G).all(1)
                j = np.where(okp, np.clip(sp[:, 1], 0, G - 1) * G + np.clip(sp[:, 0], 0, G - 1), np.arange(S))
                wc = h[t][torch.from_numpy(j).to(dev)]
                m5 = h[t - 4:t + 1].mean(0)
                Xn = torch.stack([m5[mov], wc[mov]], -1).reshape(-1, 2).double().cpu().numpy()
                def acc(key, yv):
                    A = NR.setdefault(key, [np.zeros((2, 2)), np.zeros(2)]); A[0] += Xn.T @ Xn; A[1] += Xn.T @ yv
                acc(("true", 0), h[t + 1][mov].reshape(-1).double().cpu().numpy())
                for name, o in outs.items():
                    for k in range(o.shape[0]):
                        acc((name, k), o[k, t][mov].reshape(-1).double().cpu().numpy())
        w0 += T_.shape[0]
    bd = float(np.linalg.solve(*NR[("true", 0)])[1])
    rr = {"data_b": bd}
    for name, (W, kind) in models.items():
        ks = sorted(k for (n_, k) in NR if n_ == name)
        rr[name] = [float(np.linalg.solve(*NR[(name, k)])[1]) for k in ks]
    res["natural"][src] = rr
    print(f"  자연 {src}: 데이터 b {bd:.3f} | " + " · ".join(f"{n_} {np.round(v, 3).tolist()}" for n_, v in rr.items() if n_ != "data_b"), flush=True)
json.dump(res, open(os.path.join(os.path.dirname(__file__), "../results/ar_crps_eval.json"), "w"), indent=1)
