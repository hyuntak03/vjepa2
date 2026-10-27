#!/usr/bin/env python3
"""motion 을 어디에 걸까 — encoder 출력 → predictor 층 1..12 → 출력 (TF) 에서 움직임 방향이 얼마나 남나 (2026-10-24, 사용자 "인코더부터 하나씩 다 진단").
auto_research/scripts/direction_probe.py (2026-10-22 [P]) 를 그대로 쓰고 두 가지를 더했다:
  · 모델: 기준 AR · motion (MotionJEPA 이식, 학습 중) · 정답 토큰 판독기 감독 · 3D 합성 전용
  · 이식 판독: **정답 토큰으로 맞춘 자를 예측 출력에 그대로 씌움** — 예측이 움직임을 정답 토큰과 같은 방식으로 담나 (채점에 들어가는 형태)
      enc_hn = h_{t+1}(c) (정답 다음 블록 · 도착 칸) → 예측 out_t(c) 에 씌움 / enc_pn = [h_t(c), h_{t+1}(c)] → [h_t(c), out_t(c)] 에 씌움
광류는 판독용 정답으로만 쓴다 (학습 신호 아님).

  CUDA_VISIBLE_DEVICES=7 /venv/vjepa/bin/python -W ignore z_research/scripts/analysis/motion_where_probe.py [--limit N]

[원 문서 — direction_probe.py]

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
ap.add_argument("--maxn", type=int, default=20000)
ap.add_argument("--out", default="/workspace/world_model/vjepa2/z_training/runs/natural_ar_vw_motion/motion_where_probe.json")
ap.add_argument("--mem_gb", type=float, default=4.5)
ap.add_argument("--device", default="cuda")
ap.add_argument("--models", nargs="+", default=["base", "motion", "gthead", "syn"], help="base · motion · gthead · syn · vob · hyp1")
ap.add_argument("--sets", nargs="+", default=["k710_holdout", "ssv2_actor", "ssv2_passive", "phys3d"], help="+ vobench (lad_synth 캐시)")
a = ap.parse_args()
REPO = "/workspace/world_model/vjepa2"
sys.path.insert(0, os.path.join(REPO, "auto_research/scripts")); sys.path.insert(0, REPO)
_argv = list(sys.argv); sys.argv = [sys.argv[0], "--loss", "l1", "--name", "x"]
import ar_crps_ft as FT  # noqa: E402
sys.argv = _argv
from analysis.intphys2.model import _clean_backbone_key  # noqa: E402
torch.set_num_threads(int(os.environ.get("PROBE_THREADS", 8)))
dev = torch.device(a.device)        # 2026-10-24: cpu 도 (학습이 GPU 를 꽉 잡을 때). cpu 면 fp32 · autocast 없음
if dev.type == "cuda":
    torch.cuda.set_per_process_memory_fraction(a.mem_gb * 1024**3 / torch.cuda.get_device_properties(0).total_memory, 0)
S, G, D = 256, 16, 1280
cy, cx = np.divmod(np.arange(S), G); P0 = np.stack([cx + 0.5, cy + 0.5], 1)
LAMS = (1.0, 10.0, 100.0, 1e3, 1e4, 1e5)


def cell_of(p):
    s = np.floor(p).astype(int); ok = (s >= 0).all(1) & (s < G).all(1)
    return np.clip(s[:, 1], 0, G - 1) * G + np.clip(s[:, 0], 0, G - 1), ok


LAMW = {}                                                                         # 잠재 행동 감싸개 (persist 규칙일 때만)
RUNS = {"base": "natural_ar_vw_state", "syn": "natural_ar_vw_state_phys3d_only", "vob": "natural_ar_vw_state_vob",
        "motion": "natural_ar_vw_motion", "gthead": "natural_ar_vw_state_phys3d_event_gthead", "motion2": "natural_ar_vw_motion_v2", "motion3": "natural_ar_vw_motion_v3", "lam": "natural_ar_vw_lam"}
MODELS = {}
for nm in a.models:
    if "@" in nm:                                    # 2026-10-24: run@ckpt (예: base@e4) — 같은 epoch 끼리 비교
        _k, _c = nm.split("@")
        _c, _pol = (_c.split("+") + [""])[:2]        # 2026-10-24: lam@e8+persist — 잠재 행동 "마지막 z 유지" 추론
        MODELS[nm] = FT.load_pred(RUNS[_k], _c + ".pt").to(dev).eval()
        if _pol == "persist":
            from app.vjepa_frozen import latent_action as LAM_
            _ck = torch.load(f"{REPO}/z_training/runs/{RUNS[_k]}/{_c}.pt", map_location="cpu", weights_only=False)
            _w = LAM_.LatentActionAR(MODELS[nm], _ck["latent_action_cfg"], D).to(dev).eval()
            _w.idm.load_state_dict(_ck["latent_action"]["idm"]); _w.zmod.load_state_dict(_ck["latent_action"]["zmod"])
            LAMW[nm] = _w; del _ck
    elif nm in RUNS:
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
    with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
        for nm, m in MODELS.items():
            emb = m._ar_embed_obs(hx[:7].reshape(1, -1, D))
            if nm in LAMW:      # 위치 t (블록 t 입력 → t+1 예측) 에 관측 전이 t−1→t 의 z (μ). 위치 0 은 z = 0. 미래 블록은 안 쓴다
                _w = LAMW[nm]
                _, _mu, _ = _w.infer(hx[:7].reshape(1, -1, D), sample=False)            # z_0..z_5 (전이 0→1 .. 5→6, 모두 입력 블록)
                _zs = torch.cat([torch.zeros_like(_mu[:, :1]), _mu], 1)                 # 위치 0..6
                _w._m = _w.mods(_zs)
                try:
                    out = m.forward_seq(hx[:7].reshape(1, -1, D), idx)
                finally:
                    _w._m = None
            else:
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
        F_.setdefault("enc_hn", []).append(h[t + 1][c].astype(np.float16))                                   # 정답 다음 블록 (도착 칸)
        F_.setdefault("enc_pn", []).append(np.concatenate([h[t][c], h[t + 1][c]], 1).astype(np.float16))     # 정답 쌍 (도착 칸)
        cj = torch.from_numpy(jj).to(dev); cc = torch.from_numpy(c).to(dev)
        for (nm, li), L in layers.items():
            F_.setdefault(f"{nm}|{li}|ctx", []).append(L[t][cj].cpu().numpy().astype(np.float16))
            F_.setdefault(f"{nm}|{li}|pred", []).append(L[t][cc].cpu().numpy().astype(np.float16))
        for nm in MODELS:                                                                                   # 이식용: 예측 출력 · [정답 앞 블록, 예측 출력]
            o = layers[(nm, "out")][t][cc].cpu().numpy()
            F_.setdefault(f"T|{nm}|out", []).append(o.astype(np.float16))
            F_.setdefault(f"T|{nm}|pair", []).append(np.concatenate([h[t][c], o], 1).astype(np.float16))
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


def probe_transfer(Xs_, Xd_, Y, win, train_w, rng):
    """정답 특징 Xs_ 로 맞춘 ridge (λ 는 같은 5 겹) 를 예측 특징 Xd_ 에 그대로 씌운다 (표준화도 정답 쪽 통계). 보류 창에서 R²."""
    tr = np.isin(win, train_w); te = ~tr
    Xs_ = Xs_.astype(np.float32); Xd_ = Xd_.astype(np.float32); Y = Y.astype(np.float32)
    mu, sd = Xs_[tr].mean(0), Xs_[tr].std(0) + 1e-6
    Xtr = (Xs_[tr] - mu) / sd; ym = Y[tr].mean(0); Yc = Y[tr] - ym
    wtr = win[tr]; uw = np.unique(wtr); perm = rng.permutation(uw); fold = {u: k % 5 for k, u in enumerate(perm)}; fo = np.array([fold[u] for u in wtr])
    Gall = (Xtr.T @ Xtr).astype(np.float64); Ball = (Xtr.T @ Yc).astype(np.float64)
    score = {lam: [] for lam in LAMS}
    for k in range(5):
        m = fo == k
        Gk = Gall - (Xtr[m].T @ Xtr[m]).astype(np.float64); Bk = Ball - (Xtr[m].T @ Yc[m]).astype(np.float64)
        ev, V = np.linalg.eigh(Gk); VB = V.T @ Bk
        for lam in LAMS:
            score[lam].append(r2(Y[tr][m], Xtr[m] @ (V @ (VB / (ev + lam)[:, None])).astype(np.float32) + ym))
    bl = max(LAMS, key=lambda l: np.mean(score[l]))
    Wt = np.linalg.solve(Gall + bl * np.eye(Xtr.shape[1]), Ball).astype(np.float32)
    return dict(gt=round(r2(Y[te], (Xs_[te] - mu) / sd @ Wt + ym), 4), transfer=round(r2(Y[te], (Xd_[te] - mu) / sd @ Wt + ym), 4), lam=bl)


res = {"definition": "motion_where_probe 2026-10-24 (direction_probe [P] 정의 + 이식 판독)", "models": {k: RUNS.get(k, k) for k in MODELS}}
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
        for k in [k for k in cat if k.startswith(("enc_",) + tuple(f"{nm}|" for nm in MODELS)) and k not in ("enc_hn", "enc_pn")]:
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
        for nm in MODELS:
            rr[f"transfer|{nm}|single"] = probe_transfer(cat["enc_hn"], cat[f"T|{nm}|out"], Y, win, train_w, np.random.default_rng(4))
            rr[f"transfer|{nm}|pair"] = probe_transfer(cat["enc_pn"], cat[f"T|{nm}|pair"], Y, win, train_w, np.random.default_rng(4))
        r[lab] = rr
        print(f"\n== {src} · {lab} (n {r['n']}, 창 {r['n_windows']})", flush=True)
        print("   encoder: " + " · ".join(f"{k} {rr[k]['r2']:.3f}" for k in ("enc_hj", "enc_h2j", "enc_hc")) + " | 섞은 라벨 " + " · ".join(f"{rr[k]['r2']:.3f}" for k in rr if k.endswith("|shuffled"))
              + (f" | 데이터 상한 {rr['label_ceiling']['r2']:.3f}" if "label_ceiling" in rr else ""), flush=True)
        for nm in MODELS:
            for pos in ("ctx", "pred"):
                print(f"   {nm:6s} {pos:4s} " + " ".join(f"{rr[f'{nm}|{li}|{pos}']['r2']:.2f}" for li in list(range(13)) + ["out"]), flush=True)
        print("   이식 (정답 자 → 예측 출력) 단일/쌍: " + " · ".join(f"{nm} {rr[f'transfer|{nm}|single']['transfer']:.3f}/{rr[f'transfer|{nm}|pair']['transfer']:.3f}" for nm in MODELS)
              + f" | 정답 자 자기 {rr[f'transfer|{list(MODELS)[0]}|single']['gt']:.3f}/{rr[f'transfer|{list(MODELS)[0]}|pair']['gt']:.3f}", flush=True)
    res[src] = r
json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
print("저장", a.out, f"{time.time() - t0:.0f}s")
