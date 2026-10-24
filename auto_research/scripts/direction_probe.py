#!/usr/bin/env python3
"""[P] 블록 AR 은 움직임 방향을 어디서 잃나 — 층별 선형 판독 (ridge, 학습 없음) (auto_research, 2026-10-22 KST).

  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=5 /venv/vjepa/bin/python -W ignore auto_research/scripts/direction_probe.py [--limit N]

사전 등록: 진행판 [P] 08:01 UTC (KST 17:01).
TF 한 걸음 (forward_seq) 에서 위치 t 의 토큰 = 관측 블록 t → 층을 거쳐 블록 t+1 예측. 두 자리를 칸으로 나눈다:
  (i) 문맥 쪽 = 위치 t 의 출처 칸 j = c + U[t+1](c) · (ii) 예측 쪽 = 위치 t 의 도착 칸 c.
표본: |U[t+1](c)| ≥ 0.5, |U[t](j)| ≥ 0.5, j 화면 안, t = 4 · 5 · 6. 세트마다 최대 MAXN 칸 (무작위, seed 0).
라벨: 문맥 방향 dctx = U[t](j)/|·|, 다음 움직임 dnext = U[t+1](c)/|·| (2 차원 단위 벡터).
판독 지점: 층 0 (입력 임베딩) · 블록 1..12 출력 · out (투영 + LN 출력 토큰).
모델: base = natural_ar_vw_state latest · hyp1 = ar_ft/ft_wta3.pt 의 predictor + 부호 codes[1] (입력 임베딩에 더함).
encoder 상한: h_t(j) · [h_{t−1}(j), h_t(j)] · h_t(c).
ridge: 창 단위 반 / 반 보류 (seed 0), λ 는 맞춤 쪽 창 단위 5 겹 교차. 섞은 라벨 대조 (맞춤 쪽 라벨 순열).
출력: auto_research/results/direction_probe.json."""
import argparse, json, os, sys, time
import numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--maxn", type=int, default=40000)
ap.add_argument("--out", default="/workspace/world_model/vjepa2/auto_research/results/direction_probe.json")
ap.add_argument("--models", nargs="+", default=["base", "hyp1"], help="base · hyp1 · syn (합성 전용 AR) · vob (VOBench 섞은 AR)")
ap.add_argument("--sets", nargs="+", default=["ssv2_passive", "ssv2_actor", "k710_holdout"], help="+ phys3d · vobench (lad_synth 캐시)")
a = ap.parse_args()
REPO = "/workspace/world_model/vjepa2"
sys.path.insert(0, os.path.join(REPO, "auto_research/scripts")); sys.path.insert(0, REPO)
_argv = list(sys.argv); sys.argv = [sys.argv[0], "--loss", "l1", "--name", "x"]
import ar_crps_ft as FT  # noqa: E402
sys.argv = _argv
from analysis.intphys2.model import _clean_backbone_key  # noqa: E402
torch.set_num_threads(8)
dev = torch.device("cuda")
S, G, D = 256, 16, 1280
cy, cx = np.divmod(np.arange(S), G); P0 = np.stack([cx + 0.5, cy + 0.5], 1)
LAMS = (1.0, 10.0, 100.0, 1e3, 1e4, 1e5)


def cell_of(p):
    s = np.floor(p).astype(int); ok = (s >= 0).all(1) & (s < G).all(1)
    return np.clip(s[:, 1], 0, G - 1) * G + np.clip(s[:, 0], 0, G - 1), ok


RUNS = {"base": "natural_ar_vw_state", "syn": "natural_ar_vw_state_phys3d_only", "vob": "natural_ar_vw_state_vob"}
MODELS = {}
for nm in a.models:
    if nm in RUNS:
        MODELS[nm] = FT.load_pred(RUNS[nm], "latest.pt").to(dev).eval()
    elif nm == "hyp1":
        hyp = FT.load_pred("natural_ar_vw_state", "latest.pt")
        ck = torch.load("/workspace/auto_research_cache/ar_ft/ft_wta3.pt", map_location="cpu", weights_only=False)
        hyp.load_state_dict({k: v for k, v in _clean_backbone_key(ck["predictor"]).items() if not k.startswith(("hint_attn.", "dyn_head.", "recon_proj."))}, strict=True)
        hyp = hyp.to(dev).eval()
        CODE1 = ck["codes"][1].to(dev)
        _orig = hyp._ar_embed_obs

        def _emb_code(x, _o=_orig, _c=CODE1):
            e = _o(x)
            return e + _c[None, None, :].to(e.dtype)
        hyp._ar_embed_obs = _emb_code
        MODELS["hyp1"] = hyp
CAP = {}


def hook(i):
    def f(mod, inp, out):
        CAP[i] = (out[0] if isinstance(out, tuple) else out).detach()
    return f


for nm, m in MODELS.items():
    for i, blk in enumerate(m.predictor_blocks):
        blk.register_forward_hook(hook((nm, i + 1)))


def windows():
    for name in a.sets:
        if name.startswith("ssv2_"):
            sc = "/workspace/auto_research_cache/ssv2_classes"; grp = name[5:]
            Hs = np.load(f"{sc}/{grp}_h.npy", mmap_mode="r"); Us = np.load(f"{sc}/{grp}_U.npy", mmap_mode="r")
        elif name in ("phys3d", "vobench"):
            sc = "/workspace/auto_research_cache/lad_synth"
            Hs = np.load(f"{sc}/{name}_h.npy", mmap_mode="r"); Us = np.load(f"{sc}/{name}_U.npy", mmap_mode="r")
        else:
            nh = "/workspace/auto_research_cache/natural_holdout"
            Hs = np.load(f"{nh}/k710_320_holdout_h.npy", mmap_mode="r"); Us = np.load(f"{nh}/k710_320_holdout_U.npy", mmap_mode="r")
        for w in range(min(Hs.shape[0], a.limit or 10 ** 9)):
            yield name, w, np.asarray(Hs[w], np.float32), np.asarray(Us[w], np.float32).reshape(8, S, 2)


FEATS = {}                                                                      # src → dict(name → list of arrays)
t0 = time.time()
rng0 = np.random.default_rng(0)
for n_done, (src, w, h, U) in enumerate(windows()):
    hx = torch.from_numpy(h).to(dev)
    idx = torch.arange(7 * S, device=dev).unsqueeze(0)
    layers = {}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for nm, m in MODELS.items():
            emb = m._ar_embed_obs(hx[:7].reshape(1, -1, D))
            out = m.forward_seq(hx[:7].reshape(1, -1, D), idx)
            layers[(nm, 0)] = emb.float()[0].view(7, S, -1)
            for li in range(1, 13):
                layers[(nm, li)] = CAP[(nm, li)].float()[0].view(7, S, -1)
            layers[(nm, "out")] = out.float()[0].view(7, S, -1)
    F_ = FEATS.setdefault(src, {})
    for t in (4, 5, 6):
        u1, u0 = U[t + 1], U[t]
        j, ok = cell_of(P0 + u1)
        c = np.where((np.linalg.norm(u1, axis=1) >= 0.5) & ok & (np.linalg.norm(u0[j], axis=1) >= 0.5))[0]
        if c.size == 0:
            continue
        jj = j[c]
        dctx = u0[jj] / np.linalg.norm(u0[jj], axis=1, keepdims=True); dnext = u1[c] / np.linalg.norm(u1[c], axis=1, keepdims=True)
        F_.setdefault("win", []).append(np.full(c.size, w)); F_.setdefault("dctx", []).append(dctx); F_.setdefault("dnext", []).append(dnext)
        F_.setdefault("enc_hj", []).append(h[t][jj].astype(np.float16)); F_.setdefault("enc_hc", []).append(h[t][c].astype(np.float16))
        F_.setdefault("enc_h2j", []).append(np.concatenate([h[t - 1][jj], h[t][jj]], 1).astype(np.float16))
        cj = torch.from_numpy(jj).to(dev); cc = torch.from_numpy(c).to(dev)
        for (nm, li), L in layers.items():
            F_.setdefault(f"{nm}|{li}|ctx", []).append(L[t][cj].cpu().numpy().astype(np.float16))
            F_.setdefault(f"{nm}|{li}|pred", []).append(L[t][cc].cpu().numpy().astype(np.float16))
    if n_done % 50 == 0:
        print(f"{src} {w} · {n_done} 창 · {time.time() - t0:.0f}s", flush=True)


def r2(Y, P):
    return float(1 - ((Y - P) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())


def probe(X, Y, win, train_w, shuffle=False, rng=None):
    """ridge: 맞춤 쪽 통계로 표준화, λ 는 맞춤 쪽 창 단위 5 겹 (Gram 을 한 번 만들고 겹마다 뺀다)."""
    tr = np.isin(win, train_w); te = ~tr
    X = X.astype(np.float32); Y = Y.astype(np.float32)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xs = (X - mu) / sd
    Ytr = Y[tr].copy()
    if shuffle:
        Ytr = Ytr[rng.permutation(len(Ytr))]
    ym = Ytr.mean(0); Xtr = Xs[tr]; Yc = Ytr - ym
    wtr = win[tr]; uw = np.unique(wtr); perm = rng.permutation(uw); fold = {u: k % 5 for k, u in enumerate(perm)}; fo = np.array([fold[u] for u in wtr])
    Gall = (Xtr.T @ Xtr).astype(np.float64); Ball = (Xtr.T @ Yc).astype(np.float64)
    score = {lam: [] for lam in LAMS}
    for k in range(5):
        m = fo == k
        Gk = Gall - (Xtr[m].T @ Xtr[m]).astype(np.float64); Bk = Ball - (Xtr[m].T @ Yc[m]).astype(np.float64)
        ev, V = np.linalg.eigh(Gk); VB = V.T @ Bk                                 # 한 번 분해하고 λ 마다 대각만 바꾼다
        for lam in LAMS:
            Wt = V @ (VB / (ev + lam)[:, None])
            score[lam].append(r2(Ytr[m], Xtr[m] @ Wt.astype(np.float32) + ym))
    bl = max(LAMS, key=lambda l: np.mean(score[l]))
    Wt = np.linalg.solve(Gall + bl * np.eye(X.shape[1]), Ball).astype(np.float32)
    return dict(r2=round(r2(Y[te], Xs[te] @ Wt + ym), 4), lam=bl, cv=round(float(np.mean(score[bl])), 4), n_tr=int(tr.sum()), n_te=int(te.sum()))


res = {"definition": "사전 등록 [P] 08:01 UTC (KST 17:01)"}
for src, F_ in FEATS.items():
    cat = {k: np.concatenate(v) for k, v in F_.items()}
    n = len(cat["win"]); sel = np.sort(rng0.choice(n, min(n, a.maxn), replace=False))
    cat = {k: v[sel] for k, v in cat.items()}
    win = cat["win"]; uw = np.unique(win); rng = np.random.default_rng(1)
    train_w = rng.permutation(uw)[: len(uw) // 2]
    r = {"n": int(len(win)), "n_windows": int(len(uw))}
    for lab in ("dctx", "dnext"):
        Y = cat[lab].astype(np.float64)
        rr = {}
        for k in [k for k in cat if k.startswith(("enc_",) + tuple(f"{nm}|" for nm in MODELS))]:
            X = cat[k].astype(np.float64)
            rr[k] = probe(X, Y, win, train_w, rng=np.random.default_rng(2))
        m0 = list(MODELS)[0]
        for k in ("enc_h2j", f"{m0}|12|pred", f"{m0}|1|pred"):
            rr[k + "|shuffled"] = probe(cat[k].astype(np.float64), Y, win, train_w, shuffle=True, rng=np.random.default_rng(3))
        # 데이터 상한: 정답 문맥 방향 → 다음 움직임 (같은 반 / 반)
        if lab == "dnext":
            tr = np.isin(win, train_w); A_ = np.c_[cat["dctx"], np.ones(len(win))]
            coef, *_ = np.linalg.lstsq(A_[tr], Y[tr], rcond=None)
            rr["label_ceiling"] = dict(r2=round(r2(Y[~tr], A_[~tr] @ coef), 4))
        r[lab] = rr
        print(f"\n== {src} · {lab} (n {r['n']}, 창 {r['n_windows']})", flush=True)
        print("   encoder: " + " · ".join(f"{k} {rr[k]['r2']:.3f}" for k in ("enc_hj", "enc_h2j", "enc_hc")) + " | 섞은 라벨 " + " · ".join(f"{rr[k]['r2']:.3f}" for k in rr if k.endswith("|shuffled"))
              + (f" | 데이터 상한 {rr['label_ceiling']['r2']:.3f}" if "label_ceiling" in rr else ""), flush=True)
        for nm in MODELS:
            for pos in ("ctx", "pred"):
                print(f"   {nm:4s} {pos:4s} " + " ".join(f"{rr[f'{nm}|{li}|{pos}']['r2']:.2f}" for li in list(range(13)) + ["out"]), flush=True)
    res[src] = r
json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
print("저장", a.out, f"{time.time() - t0:.0f}s")
