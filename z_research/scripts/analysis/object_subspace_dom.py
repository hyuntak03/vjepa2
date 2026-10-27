"""물체 벡터의 정체 · 움직임 부분공간을 평균 차로 구하고, 지웠을 때 정말 망가지는지로 검증한다 (2026-10-25, 학습 없음).
사용자: "(shape x color) identity 평균 벡터 − 전체 평균으로 identity 의 방향을 구하던가" · "difference in means 로 확실하게 구해봐" ·
"이걸 뺐을 때 decoder 가 렌더 못하면 맞는거고, 학습된 head 성능이 떨어지면 맞는거고".
참고: Joseph et al. 2026 "Interpreting Physics in Video World Models" — 방향은 수십 차원 (직교 probe 연쇄로 14–136), 조작에는 수십 방향이 함께 필요.

표본 = 물체 칸 토큰 하나하나 (정답, LN(target_encoder)). 블록 AR 공간 (블록 2 장) · 릴리즈 공간 (32 장 창) 따로.
  train = pred_objvec train 캐시 400 편 (물체 칸 토큰), val = val 캐시 미래 블록 (350 편).
  시나리오 평균을 먼저 뺀다 (장면 배치 · 배경이 시나리오마다 달라 움직임 방향 · 위치와 얽히는 것을 막는다; 정체는 시나리오와 독립으로 뽑혔다).
부분공간 (평균 차):
  정체   48 정체 (모양 6 × 색 8) 의 평균 − 전체 평균 → SVD. 차원 = split-half 로 재현되는 성분까지 (반 A · 반 B 평균 차의 같은 성분 cos > 0.5)
  방향   블록 안 변위 방향 8 구간 (속도 ≥ 1 px/장 인 칸만) 평균 − 전체 평균 → 최대 7 차원
  속력   속력 5 분위 평균 − 전체 평균 → 최대 4 차원
  비교   직교 probe 연쇄 (논문 C.11) 로 정체 · 방향 차원 수도 잰다 (val 성능이 우연 근처가 될 때까지)
검증:
  V1 val 판독: 부분공간 좌표만으로 최근접 평균 분류 (정체 48 · 모양 · 색 · 방향 8). 우연 = 1/48 · 1/6 · 1/8 · 1/8
  V2 head: 부분공간을 만들 때 안 쓴 train 반으로 학습한 ridge 판독기 (전체 1280 차원) 의 val 정확도 — 원본 · 부분공간 제거 · 같은 차원 무작위 (공분산 따름) 제거
  V3 decoder: val 정답 블록의 물체 칸에서 부분공간 성분을 전체 평균 값으로 바꿔 decoder 로 그린다 → 물체 상자 안 (a) 원본 그림과의 차이, (b) 물체 지운 그림 대비 물체 세기, (c) 물체 평균 RGB 가 원래 색에서 얼마나 멀어졌나. 같은 차원 무작위 제거와 비교

  CUDA_VISIBLE_DEVICES=6 /venv/vjepa/bin/python -W ignore z_research/scripts/analysis/object_subspace_dom.py
출력: auto_research/results/object_subspace_dom.json
"""
import json, os, sys
import numpy as np, torch, torch.nn.functional as F

REPO = "/workspace/world_model/vjepa2"; sys.path.insert(0, os.path.join(REPO, "z_pixel_decoder"))
from model import PixelDecoder  # noqa: E402
CACHE = "/workspace/auto_research_cache/pred_objvec"; OUT = os.path.join(REPO, "auto_research/results/object_subspace_dom.json")
DECODER = os.path.join(REPO, "z_pixel_decoder/runs/vob_grav_i1_i2_vitb_both/latest.pt")
IDX = [3 * k for k in range(32)]; NBC = 8; D = 1280
dev = torch.device("cuda")
rng = np.random.default_rng(0); torch.manual_seed(0)
tmeta = json.load(open(f"{CACHE}/train_meta.json")); vmeta = json.load(open(f"{CACHE}/val_meta.json"))
STATES = {}


def disp(clip, b):
    if clip not in STATES:
        STATES[clip] = np.load(clip[:-4] + "_states.npz")["pos"][:, 0]
    p = STATES[clip]; return p[IDX[2 * b + 1]] - p[IDX[2 * b]]


def scen(clip):
    return clip.split("/")[-2]


BIG = "/workspace/auto_research_cache/objtok_train"   # 2026-10-25 사용자 "학습 데이터로 해봐 그건 샘플 많잖아": extract_objtok.py 4,000 편


def collect_train(sp):
    if os.path.isdir(BIG):
        import glob as _g
        X, lab = [], []
        for f in sorted(_g.glob(f"{BIG}/shard*.npz")):
            z = np.load(f, allow_pickle=True); sh = int(os.path.basename(f)[5:-4]); scn = z["scen"]
            L = z[f"lab_{sp}"]
            if len(L) == 0:
                continue
            X.append(z[f"tok_{sp}"].astype(np.float32))
            lab += [(int(r[0]), int(r[1]), int(r[2]), str(scn[int(r[0]) - sh * 100000]), float(r[5]), float(r[6])) for r in L]
        return np.concatenate(X), lab
    X, lab = [], []
    for i, m in enumerate(tmeta):
        tok = np.load(f"{CACHE}/train/{i:05d}.npz")[f"objtok_{sp}"]; r = 0
        for b in range(16):
            L = m["labels"][b]
            if L["ok"] and L["cells"]:
                n = len(L["cells"]); d = disp(m["clip"], b)
                X.append(tok[r: r + n]); lab += [(i, m["shape"], m["color"], scen(m["clip"]), *d)] * n; r += n
    return np.concatenate(X).astype(np.float32), lab


def collect_val(sp):
    X, lab, blocks = [], [], []
    for i, m in enumerate(vmeta):
        z = np.load(f"{CACHE}/val/{i:05d}.npz"); Y = z[f"Y_{sp}"]
        for j in range(8):
            b = NBC + j; L = m["labels"][b]
            if L["ok"] and L["cells"]:
                d = disp(m["clip"], b)
                X.append(Y[j, L["cells"]].astype(np.float32)); lab += [(i, m["shape"], m["color"], scen(m["clip"]), *d)] * len(L["cells"])
                blocks.append((i, j))
    return np.concatenate(X), lab, blocks


def dirbin(dx, dy, thr=2.0):          # 블록 안 변위 (3 장 간격 = 1 장 4 fps 의 절반) — px 단위, 1 px/장 ≈ 3 px
    s = np.hypot(dx, dy)
    return np.where(s >= thr, ((np.arctan2(dy, dx) + np.pi) / (2 * np.pi / 8)).astype(int) % 8, -1), s


def class_means(X, y, classes):
    return np.stack([X[y == c].mean(0) for c in classes])


def dom_subspace(X, y, groups, classes, rmax):
    """평균 차 부분공간 + split-half 재현 성분 수."""
    classes = [c for c in classes if (y == c).sum() > 0]
    mu = X.mean(0)
    M = class_means(X, y, classes) - mu
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    ug = np.unique(groups); hA = np.isin(groups, rng.permutation(ug)[: len(ug) // 2])
    ok = [c for c in classes if (y[hA] == c).sum() > 5 and (y[~hA] == c).sum() > 5]
    MA = class_means(X[hA], y[hA], ok) - X[hA].mean(0); MB = class_means(X[~hA], y[~hA], ok) - X[~hA].mean(0)
    VA = np.linalg.svd(MA, full_matrices=False)[2]; VB = np.linalg.svd(MB, full_matrices=False)[2]
    k = 0
    for c in range(min(rmax, len(ok) - 1)):
        if abs(float(VA[c] @ VB[c])) > 0.5:
            k = c + 1
        else:
            break
    # 부분공간 수준 재현도 (성분 순서가 바뀌어도 되게): 상위 r 성분 공간 겹침 vs 같은 r 무작위 (등방) 기댓값 r/1280
    sub = {r: round(float(np.linalg.norm(VA[:r] @ VB[:r].T) ** 2 / r), 3) for r in (1, 3, 5, 10, 20, 40) if r <= min(rmax, len(ok) - 1)}
    dom_subspace.last = {"subspace_overlap_halves": sub}
    return Vt[: max(k, 1)].T, Vt[:rmax].T, k, (s ** 2 / (s ** 2).sum())[: rmax].round(3).tolist()


def nearest_mean_acc(Xtr, ytr, Xte, yte, Q, classes):
    classes = [c for c in classes if (ytr == c).sum() > 0]
    mu = Xtr.mean(0); C = class_means(Xtr, ytr, classes) - mu
    a, b = (Xte - mu) @ Q, C @ Q
    pred = np.array(classes)[np.argmin(((a[:, None] - b[None]) ** 2).sum(-1), 1)]
    return round(100 * float((pred == yte).mean()), 1)


def ridge(X, Y, lam=1e2):
    u, s, vt = np.linalg.svd(X, full_matrices=False)
    return vt.T @ ((s / (s ** 2 + lam))[:, None] * (u.T @ Y))


def inlp_dims(Xtr, ytr, Xte, yte, K, chance, maxit=60):
    """직교 probe 연쇄 (논문 C.11): 정확도가 우연 + 5 %p 아래로 떨어질 때까지 몇 번 (× (K−1) 차원)."""
    Xa, Xb = Xtr.copy(), Xte.copy(); accs = []
    for it in range(maxit):
        Y = np.eye(K)[ytr]; W = ridge(Xa, Y - Y.mean(0))
        acc = float(((Xb @ W).argmax(1) == yte).mean() * 100); accs.append(round(acc, 1))
        if acc < chance + 5:
            break
        Q, _ = np.linalg.qr(W); Xa = Xa - (Xa @ Q) @ Q.T; Xb = Xb - (Xb @ Q) @ Q.T
    return {"n_probes": len(accs) - 1, "dims": (len(accs) - 1) * (K - 1), "acc_seq": accs[:12] + (["…"] if len(accs) > 12 else []) + accs[-2:]}


st_ = torch.load(DECODER, map_location="cpu", weights_only=False); mc = st_["cfg"]["model"]
dec = PixelDecoder(in_dim=D, dim=mc["dim"], depth=mc["depth"], heads=mc["heads"]).to(dev).eval(); dec.load_state_dict(st_["decoder"]); del st_
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1, 1)

out = {}
for sp in ("ar", "rel"):
    Xt, Lt = collect_train(sp); Xv, Lv, vblocks = collect_val(sp)
    g_t = np.array([l[0] for l in Lt]); sh_t = np.array([l[1] for l in Lt]); co_t = np.array([l[2] for l in Lt]); sc_t = np.array([l[3] for l in Lt]); d_t = np.array([l[4:] for l in Lt])
    sh_v = np.array([l[1] for l in Lv]); co_v = np.array([l[2] for l in Lv]); sc_v = np.array([l[3] for l in Lv]); d_v = np.array([l[4:] for l in Lv])
    # 시나리오 평균 빼기 (train 평균으로 train · val 모두)
    smu = {s: Xt[sc_t == s].mean(0) for s in np.unique(sc_t)}
    Xtc = Xt - np.stack([smu[s] for s in sc_t]); Xvc = Xv - np.stack([smu.get(s, Xt.mean(0)) for s in sc_v])
    id_t, id_v = sh_t * 8 + co_t, sh_v * 8 + co_v
    db_t, sp_t = dirbin(d_t[:, 0], d_t[:, 1]); db_v, sp_v = dirbin(d_v[:, 0], d_v[:, 1])
    res = {"n_train_tokens": len(Xt), "n_val_tokens": len(Xv)}
    subs = {}
    Q, Qfull, k, ev = dom_subspace(Xtc, id_t, g_t, list(range(48)), 47); subs["identity"] = Q
    res["identity"] = {"dims_reproducible": k, "top_var_share": ev[:10], **dom_subspace.last}
    Qid_by_r = {r: Qfull[:, :r] for r in (5, 10, 20)}
    mv = db_t >= 0
    Qd, _, kd, evd = dom_subspace(Xtc[mv], db_t[mv], g_t[mv], list(range(8)), 7); subs["direction"] = Qd
    res["direction"] = {"dims_reproducible": kd, "top_var_share": evd, "n_moving_tokens": int(mv.sum()), **dom_subspace.last}
    qb = np.quantile(sp_t, [0.2, 0.4, 0.6, 0.8]); sb_t = np.digitize(sp_t, qb); sb_v = np.digitize(sp_v, qb)
    Qs, _, ks, evs = dom_subspace(Xtc, sb_t, g_t, list(range(5)), 4); subs["speed"] = Qs
    res["speed"] = {"dims_reproducible": ks, "top_var_share": evs, **dom_subspace.last}
    # V1 부분공간 좌표만으로 최근접 평균 판독 (val)
    mvv = db_v >= 0
    res["V1_nearest_mean"] = {
        "identity48": nearest_mean_acc(Xtc, id_t, Xvc, id_v, subs["identity"], list(range(48))), "identity48_chance": round(100 / 48, 1),
        "shape_via_identity": nearest_mean_acc(Xtc, sh_t, Xvc, sh_v, subs["identity"], list(range(6))),
        "color_via_identity": nearest_mean_acc(Xtc, co_t, Xvc, co_v, subs["identity"], list(range(8))),
        "direction8": nearest_mean_acc(Xtc[mv], db_t[mv], Xvc[mvv], db_v[mvv], subs["direction"], list(range(8))), "direction_chance": 12.5,
        "direction8_via_identity": nearest_mean_acc(Xtc[mv], db_t[mv], Xvc[mvv], db_v[mvv], subs["identity"], list(range(8))),
        "identity48_via_direction": nearest_mean_acc(Xtc, id_t, Xvc, id_v, subs["direction"], list(range(48)))}
    res["V1_identity_top_r"] = {r_: {"identity48": nearest_mean_acc(Xtc, id_t, Xvc, id_v, Qid_by_r[r_], list(range(48))),
                                     "shape": nearest_mean_acc(Xtc, sh_t, Xvc, sh_v, Qid_by_r[r_], list(range(6))),
                                     "color": nearest_mean_acc(Xtc, co_t, Xvc, co_v, Qid_by_r[r_], list(range(8)))} for r_ in (5, 10, 20)}
    # 부분공간끼리 겹침
    ov = lambda A, B: round(float(np.linalg.norm(A.T @ B) ** 2 / min(A.shape[1], B.shape[1])), 3)
    res["overlap"] = {"identity-direction": ov(subs["identity"], subs["direction"]), "identity-speed": ov(subs["identity"], subs["speed"]), "direction-speed": ov(subs["direction"], subs["speed"])}
    # 직교 probe 연쇄 차원 (논문 C.11)
    sel = rng.choice(len(Xtc), min(len(Xtc), 40000), replace=False)
    mu_, sd_ = Xtc.mean(0), Xtc.std(0) + 1e-6
    res["inlp"] = {"shape": inlp_dims((Xtc[sel] - mu_) / sd_, sh_t[sel], (Xvc - mu_) / sd_, sh_v, 6, 100 / 6),
                   "color": inlp_dims((Xtc[sel] - mu_) / sd_, co_t[sel], (Xvc - mu_) / sd_, co_v, 8, 12.5),
                   "direction": inlp_dims((Xtc[mv][: 40000] - mu_) / sd_, db_t[mv][: 40000], (Xvc[mvv] - mu_) / sd_, db_v[mvv], 8, 12.5)}
    # V2 head: 부분공간 만들 때와 같은 train 이지만 다른 판독기 (전체 1280 차원 ridge) — 원본 · 제거 · 무작위 제거
    Sig = np.cov(Xtc[sel], rowvar=False); ev_, evec_ = np.linalg.eigh(Sig); sq = evec_ * np.sqrt(np.clip(ev_, 0, None))
    def heads_after(Q):
        P = lambda X: X - ((X - mu_) @ Q) @ Q.T if Q is not None else X
        r = {}
        for nm, ytr, yte, K, m_tr, m_te in (("shape", sh_t, sh_v, 6, slice(None), slice(None)), ("color", co_t, co_v, 8, slice(None), slice(None)),
                                            ("direction", db_t, db_v, 8, mv, mvv)):
            Xa = (P(Xtc[m_tr])[: 40000] - mu_) / sd_; Xb = (P(Xvc[m_te]) - mu_) / sd_; ya = ytr[m_tr][: 40000]
            W = ridge(Xa, np.eye(K)[ya] - np.eye(K)[ya].mean(0)); r[nm] = round(100 * float(((Xb @ W).argmax(1) == yte[m_te]).mean()), 1)
        return r
    def rand_like(k_):
        Qr, _ = np.linalg.qr(sq @ rng.standard_normal((D, k_))); return Qr
    res["V2_head"] = {"original": heads_after(None)}
    for nm in ("identity", "direction", "speed"):
        res["V2_head"][f"remove_{nm}({subs[nm].shape[1]})"] = heads_after(subs[nm])
        res["V2_head"][f"remove_random({subs[nm].shape[1]})"] = heads_after(rand_like(subs[nm].shape[1]))
    for r_ in (10, 20):
        res["V2_head"][f"remove_identity_top{r_}"] = heads_after(Qid_by_r[r_])
        res["V2_head"][f"remove_random({r_})"] = heads_after(rand_like(r_))
    # V3 decoder: val 블록 물체 칸에서 부분공간 성분을 평균으로 (원 좌표: 시나리오 평균을 뺀 공간에서 정의 → 같은 평행이동으로 적용)
    dres = {k_: [] for k_ in ("identity", "identity_top10", "rand_10", "direction", "speed", "rand_identity", "rand_direction")}
    Qt = {"identity": subs["identity"], "identity_top10": Qid_by_r[10], "rand_10": rand_like(10), "direction": subs["direction"], "speed": subs["speed"],
          "rand_identity": rand_like(subs["identity"].shape[1]), "rand_direction": rand_like(subs["direction"].shape[1])}
    Qt = {k_: torch.from_numpy(v).float().to(dev) for k_, v in Qt.items()}
    mu_t = torch.from_numpy(mu_).float().to(dev)
    for (i, j) in vblocks[:: max(1, len(vblocks) // 400)]:
        z = np.load(f"{CACHE}/val/{i:05d}.npz"); L = vmeta[i]["labels"][NBC + j]; cells = L["cells"]
        Y = torch.from_numpy(z[f"Y_{sp}"][j]).float().to(dev); bg = torch.from_numpy(z[f"bg_{sp}"]).float().to(dev)
        smu_c = torch.from_numpy(smu.get(scen(vmeta[i]["clip"]), Xt.mean(0))).float().to(dev)
        Xs = [Y]
        B = Y.clone(); hasbg = not torch.isnan(bg[cells]).any()
        if hasbg:
            B[cells] = bg[cells]
        Xs.append(B)
        for k_, Q in Qt.items():
            X = Y.clone(); xc = X[cells] - smu_c
            X[cells] = xc - ((xc - mu_t) @ Q) @ Q.T + smu_c; Xs.append(X)
        with torch.no_grad():
            y_ = (dec(torch.stack(Xs)) * STD + MEAN).clamp(0, 1)
        x0, y0, x1, y1 = [int(np.clip(v, 0, 255)) for v in L["box"]]
        if x1 <= x0 or y1 <= y0:
            continue
        R = lambda t: t[..., y0:y1 + 1, x0:x1 + 1]
        base = R(y_[0]); rgb0 = base.mean((-1, -2, -3))
        dObj = float((base - R(y_[1])).abs().mean()) if hasbg else float("nan")
        for n_, k_ in enumerate(Qt):
            yk = R(y_[2 + n_])
            dres[k_].append({"diff_vs_orig": float((yk - base).abs().mean()), "rel_to_object": float((yk - base).abs().mean()) / dObj if hasbg and dObj > 1e-4 else float("nan"),
                             "rgb_shift": float((yk.mean((-1, -2, -3)) - rgb0).norm())})
    res["V3_decoder"] = {k_: {m_: round(float(np.nanmedian([r[m_] for r in v])), 4) for m_ in ("diff_vs_orig", "rel_to_object", "rgb_shift")} | {"n": len(v)} for k_, v in dres.items()}
    # V4 (2026-10-25): 예측이 그린 물체 벡터 vs encoder 물체 벡터 — 부분공간 안에서. val 미래 블록 · 참 자리 칸 · bg 있는 블록.
    #   e_Y = Y − bg, e_P = (P − μ_tok) − β (bg − μ_tok) (β = 블록마다 예측을 bg 에 회귀). 부분공간 Q 에 투영한 좌표끼리 cos 와 크기 비.
    #   귀무 = 같은 차원 무작위 부분공간 (공분산 따름) 의 cos.
    Q4 = {"identity_top10": Qid_by_r[10], "identity_top20": Qid_by_r[20], "direction": subs["direction"], "rand10": rand_like(10), "rand3": rand_like(3)}
    Q4 = {k_: torch.from_numpy(v).float().to(dev) for k_, v in Q4.items()}
    rows4 = []
    for i, m in enumerate(vmeta):
        z = np.load(f"{CACHE}/val/{i:05d}.npz")
        Yf = torch.from_numpy(z[f"Y_{sp}"]).float().to(dev); Pf = torch.from_numpy(z[f"P_{sp}"]).float().to(dev); bgm = torch.from_numpy(z[f"bg_{sp}"]).float().to(dev)
        mtok = Yf.mean((0, 1))
        for j in range(8):
            L = m["labels"][NBC + j]; cells = L["cells"]
            if not L["ok"] or not cells or torch.isnan(bgm[cells]).any():
                continue
            Y, P, B = Yf[j, cells], Pf[j, cells], bgm[cells]
            pb, bb = (P - mtok).reshape(-1), (B - mtok).reshape(-1); beta = (pb @ bb) / (bb @ bb)
            eP, eY = (P - mtok) - beta * (B - mtok), Y - B
            r = {"step": j + 1, "whole_cos": float(F.cosine_similarity(eP, eY, dim=-1).mean()), "whole_norm": float((eP.norm(dim=1) / eY.norm(dim=1)).mean())}
            for k_, Q in Q4.items():
                a_, b_ = eP @ Q, eY @ Q
                r[f"{k_}_cos"] = float(F.cosine_similarity(a_, b_, dim=-1).mean()); r[f"{k_}_norm"] = float((a_.norm(dim=1) / b_.norm(dim=1).clamp_min(1e-6)).mean())
            rows4.append(r)
    keys4 = [k_ for k_ in rows4[0] if k_ != "step"]
    res["V4_pred_vs_enc"] = {k: {kk: round(float(np.median([r[kk] for r in rows4 if r["step"] == k])), 3) for kk in keys4} | {"n": sum(r["step"] == k for r in rows4)} for k in range(1, 9)}
    out[sp] = res
    print(sp, json.dumps(res, ensure_ascii=False), flush=True)
json.dump(out, open(OUT, "w"), indent=1, ensure_ascii=False)
print("saved", OUT)
