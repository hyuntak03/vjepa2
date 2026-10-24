#!/usr/bin/env python3
"""[T] 문맥 가중 개입 판정: 모델마다 (1) 안정도 구간별 모델 b vs 데이터 b (기울기 · 최상위 구간 비), (2) 자연 보류 세트 TF 한 걸음 L1 (전체 · 정지 칸),
    그리고 데이터에서 "조건화의 손실 보상" 상한 (auto_research, 2026-10-21).

  CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 /venv/vjepa/bin/python -W ignore auto_research/scripts/ft_eval_reliability.py --ft ft_l1 ft_l1_lr1e4 ft_wl1 ft_wl1_lr1e4

데이터: K710 보류 (val_min224, 학습 표와 겹침 0; /workspace/auto_research_cache/natural_holdout, 칸 흐름 포함) · HT100M · 3D 합성 ([E] 캐시 + natural_transport_ctx5 흐름).
안정도 s · 회귀 · 구간 경계는 context_reliability.py 와 같다 (자연 5 분위 경계 0.35 · 0.449 · 0.557 · 0.724 고정).
조건화의 손실 보상 (데이터, 제곱 오차 근사): 움직이는 칸에서 전역 회귀 (a, b) 대신 구간별 (a_k, b_k) 를 쓸 때 줄어드는 제곱 오차 Σ_k (β_g − β_k)ᵀ A_k (β_g − β_k),
  운반 자체의 보상 (b = 0 → 전역 b) 과 화면 전체 제곱 오차에 대한 몫으로 함께 낸다."""
import argparse, json, os, sys
import numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument("--ft", nargs="*", default=[])
ap.add_argument("--ftdir", default="/workspace/auto_research_cache/ar_ft")
ap.add_argument("--runs", nargs="*", default=[], help="학습 run 저장본 run:ckpt (예: natural_ar_vw_state_phys3d_single:e10.pt)")
ap.add_argument("--out", default="")
args = ap.parse_args()
sys.path.insert(0, os.path.dirname(__file__))
sys.argv = [sys.argv[0], "--loss", "l1", "--name", "x"]
import ar_crps_ft as FT                                                     # noqa: E402
torch.set_num_threads(8)
dev = torch.device("cuda")
G, S = 16, 256
EDGES = np.array([0.35, 0.449, 0.557, 0.724])
cy, cx = np.divmod(np.arange(S), G); P0 = np.stack([cx + 0.5, cy + 0.5], 1)
models = {"base": FT.NoiseWrap(FT.load_pred("natural_ar_vw_state", "latest.pt"), 32, 0.0).to(dev).eval()}
import disp_ft as DF                                                         # noqa: E402
for n in args.ft:
    ck = torch.load(os.path.join(args.ftdir, f"{n}.pt"), map_location="cpu", weights_only=False)
    if ck.get("kind") == "disp":
        Wn = DF.DispWrap(FT.load_pred("natural_ar_vw_state", "latest.pt"))
        Wn.pred.load_state_dict(ck["predictor"]); Wn.u_head.load_state_dict(ck["u_head"]); Wn.m_head.load_state_dict(ck["m_head"])
    else:
        Wn = FT.NoiseWrap(FT.load_pred("natural_ar_vw_state", "latest.pt"), 32, 0.0); Wn.pred.load_state_dict(ck["predictor"])
    models[n] = Wn.to(dev).eval()


for rc in args.runs:
    r_, c_ = rc.split(":")
    models[f"{r_.replace('natural_ar_vw_state', 'AR')}:{c_.replace('.pt', '')}"] = FT.NoiseWrap(FT.load_pred(r_, c_), 32, 0.0).to(dev).eval()


def cell_of(p):
    s = np.floor(p).astype(int); ok = (s >= 0).all(1) & (s < G).all(1)
    return np.where(ok, np.clip(s[:, 1], 0, G - 1) * G + np.clip(s[:, 0], 0, G - 1), np.arange(S)), ok


def windows(src):
    if src == "k710_holdout":
        H = np.load("/workspace/auto_research_cache/natural_holdout/k710_320_holdout_h.npy", mmap_mode="r")
        U = np.load("/workspace/auto_research_cache/natural_holdout/k710_320_holdout_U.npy", mmap_mode="r")
        for w in range(H.shape[0]):
            yield np.asarray(H[w], np.float32), np.asarray(U[w], np.float32).reshape(8, S, 2)
    else:
        EN = "/workspace/auto_research_cache/enc_noise"
        U = np.load(os.path.join(os.path.dirname(__file__), f"../results/natural_transport_ctx5/rows_{src}.npz"))["Uwin"].astype(np.float32)
        w0 = 0
        for sh in (0, 1):
            T_ = np.load(os.path.join(EN, f"{src}_target_encoder_s{sh}_tok.npy"), mmap_mode="r")
            for w in range(T_.shape[0]):
                yield np.asarray(T_[w, 0], np.float32), U[w0 + w].reshape(8, S, 2)
            w0 += T_.shape[0]


res = {"models": list(models)}
for src in ("k710_holdout", "ht100m", "phys3d"):
    ST = {n: [] for n in models}; L1 = {n: [0.0, 0.0, 0, 0] for n in models}; DSTAT = {}
    for hn, U in windows(src):
        h = torch.from_numpy(hn).to(dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            P = {n: W.tf(h[:7].reshape(1, -1, h.size(-1))).float()[0].view(7, S, -1) for n, W in models.items()}
        # TF 한 걸음 L1 (위치 1..6 → 블록 2..7), 전체 · 정지 칸 (블록 b+1 과 b 의 흐름 모두 < 0.25 칸)
        for n, W in models.items():
            if hasattr(W, "u_head"):
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    _, uu, mm_ = W.heads(h[:7].reshape(1, -1, h.size(-1)))
                for b in range(1, 7):
                    stc = torch.from_numpy(np.linalg.norm(U[b + 1], axis=1) < 0.25).to(dev); mvc = torch.from_numpy(np.linalg.norm(U[b + 1], axis=1) >= 0.5).to(dev)
                    DSTAT.setdefault(n, []).append([float(uu[0, b][stc].norm(dim=-1).mean()) if stc.any() else np.nan, float(mm_[0, b, :, 0][stc].mean()) if stc.any() else np.nan,
                                                     float(mm_[0, b, :, 0][mvc].mean()) if mvc.any() else np.nan,
                                                     float((uu[0, b][mvc] - torch.from_numpy(U[b + 1]).to(dev)[mvc]).norm(dim=-1).mean()) if mvc.any() else np.nan])
        for b in range(1, 7):
            st_m = torch.from_numpy((np.linalg.norm(U[b + 1], axis=1) < 0.25) & (np.linalg.norm(U[b], axis=1) < 0.25)).to(dev)
            for n in models:
                e = (P[n][b] - h[b + 1]).abs().mean(-1)
                L1[n][0] += float(e.mean()); L1[n][2] += 1
                if st_m.any():
                    L1[n][1] += float(e[st_m].mean()); L1[n][3] += 1
        for t in (4, 5, 6):
            u1, u0 = U[t + 1], U[t]
            mov = np.linalg.norm(u1, axis=1) >= 0.5
            if mov.sum() == 0:
                continue
            js, ok1 = cell_of(P0 + u1); jp, ok0 = cell_of(P0[js] + u0[js])
            sel = np.where(mov & ok1 & ok0)[0]
            if sel.size == 0:
                continue
            c = torch.from_numpy(sel).to(dev)
            x1 = h[t - 4:t + 1].mean(0)[c]; x2 = h[t][torch.from_numpy(js[sel]).to(dev)]; y = h[t + 1][c]
            stab = torch.nn.functional.cosine_similarity(x2, h[t - 1][torch.from_numpy(jp[sel]).to(dev)], dim=-1)
            base_cols = [(x1 * x1).sum(-1), (x1 * x2).sum(-1), (x2 * x2).sum(-1), (x1 * y).sum(-1), (x2 * y).sum(-1), (y * y).sum(-1), stab]
            for n in models:
                ym = P[n][t][c]
                ST[n].append(torch.stack(base_cols + [(x1 * ym).sum(-1), (x2 * ym).sum(-1)], -1).double().cpu().numpy())
    r = {}
    for n in models:
        st = np.concatenate(ST[n]); bins = np.digitize(st[:, 6], EDGES)
        def fit(m, yi):
            A = np.array([[st[m, 0].sum(), st[m, 1].sum()], [st[m, 1].sum(), st[m, 2].sum()]])
            return np.linalg.solve(A, [st[m, yi[0]].sum(), st[m, yi[1]].sum()]), A
        allm = np.ones(len(st), bool)
        bg, Ag = fit(allm, (3, 4)); mg, _ = fit(allm, (7, 8))
        rows = []
        gain_cond = 0.0
        for k in range(5):
            m = bins == k
            bd, Ak = fit(m, (3, 4)); bm, _ = fit(m, (7, 8))
            d = bg - bd; gain_cond += float(d @ Ak @ d)
            rows.append({"n": int(m.sum()), "b_data": float(bd[1]), "b_model": float(bm[1])})
        # 운반 자체의 보상: b = 0 (a 만 다시 맞춤) → 전역 (a, b)
        a0 = st[:, 3].sum() / st[:, 0].sum()
        d0 = bg - np.array([a0, 0.0]); gain_tr = float(d0 @ Ag @ d0)
        sse_g = float(st[:, 5].sum() - 2 * (bg[0] * st[:, 3].sum() + bg[1] * st[:, 4].sum()) + bg @ Ag @ bg)
        r[n] = {"bins": rows, "slope_data": rows[4]["b_data"] - rows[0]["b_data"], "slope_model": rows[4]["b_model"] - rows[0]["b_model"],
                "top_ratio": rows[4]["b_model"] / rows[4]["b_data"], "b_model_all": float(mg[1]), "b_data_all": float(bg[1]),
                "l1_all": L1[n][0] / max(L1[n][2], 1), "l1_static": L1[n][1] / max(L1[n][3], 1),
                "gain_conditioning_rel_sse_moving": gain_cond / sse_g, "gain_transport_rel_sse_moving": gain_tr / sse_g}
    for n, v in DSTAT.items():
        d = np.nanmean(np.array(v), 0)
        r[n].update(u_static=float(d[0]), m_static=float(d[1]), m_moving=float(d[2]), u_err_moving=float(d[3]))
    res[src] = r
    print(f"\n== {src}", flush=True)
    for n in DSTAT:
        print(f"  {n}: 정지 칸 |u| {r[n]['u_static']:.3f} 칸 · m 정지 / 움직임 {r[n]['m_static']:.2f} / {r[n]['m_moving']:.2f} · 움직이는 칸 u 오차 {r[n]['u_err_moving']:.3f} 칸", flush=True)
    for n, v in r.items():
        print(f"  {n:24s} 모델 b 구간 0→4 {[round(x['b_model'], 3) for x in v['bins']]} (기울기 {v['slope_model']:+.3f}, 데이터 {v['slope_data']:+.3f}) · 최상위 비 {v['top_ratio']:.2f} · "
              f"TF L1 전체 {v['l1_all']:.4f} · 정지 칸 {v['l1_static']:.4f}", flush=True)
    v = r["base"]
    print(f"  데이터: 구간별 b {[round(x['b_data'], 3) for x in v['bins']]} · 조건화 보상 (움직이는 칸 제곱 오차 대비) {100 * v['gain_conditioning_rel_sse_moving']:.2f} % · 운반 자체 보상 {100 * v['gain_transport_rel_sse_moving']:.2f} %", flush=True)
json.dump(res, open(args.out or os.path.join(os.path.dirname(__file__), "../results/ft_eval_reliability.json"), "w"), indent=1)
