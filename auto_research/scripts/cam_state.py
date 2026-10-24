#!/usr/bin/env python3
"""[물리] 카메라 상태 조건화의 정당성 (학습 없음, CPU) (auto_research, 2026-10-22 KST).

  OMP_NUM_THREADS=8 /venv/vjepa/bin/python auto_research/scripts/cam_state.py [--parts 1234]

사전 등록: 진행판 [물리] 06:30 UTC (KST 15:30) + 리더 수정 06:32 UTC (KST 15:32, [비판] KST 15:30 다섯 지적 채택).
흐름 U[b] (칸 단위 RAFT 칸 평균, 블록 b → b−1, U[0] = 0) 는 기존 캐시. 카메라 c_b = U[b] 에 강건하게 맞춘 호모그래피의 네 모서리 변위 (8 수, 칸 단위).
카메라 움직임 창 = 문맥 카메라 (블록 t 의 사상) 의 칸 변위 중앙값 ≥ 0.3 칸. 설명 몫 = 블록 t+1 사상이 인라이어 배경 흐름 에너지 중 설명한 몫.
 (1) c_(t+1) ≈ A c_t + 상수 R² (짝수 창 맞춤 · 홀수 창 평가) · 지속 판 R²
 (2) 카메라 뺀 물체 흐름 지속 k · R² (r = U − F_c, 물체 칸 |r_(t+1)| ≥ 0.5)
 (3) 데이터 운반 신뢰 b (class_transport 회귀) — 문맥 외삽만 · 카메라 앎 (물체 몫 외삽 + 실제 미래 카메라) · 카메라만 · 실제 흐름.
     칸 = 물체 칸 (블록 t+1 사상 잔차 ≥ 0.5 칸) / 배경 칸 (잔차 < 0.5 이고 |U[t+1]| ≥ 0.5). 간극 닫힘 G = (b_카메라앎 − b_문맥) / (b_참 − b_문맥).
 (4) 흡수 검사 — IntPhys 2 Main: 짝 c 차 사건 전 (바닥) vs 사건 뒤 ([E] ip2_decomp 사건 블록), 카메라 상태만의 surprise 표준 보고.
     VOBench 끝 가림: 카메라 상태만의 surprise 표준 보고 (사건 전 = 같은 문맥이라 바닥 없음).
토큰은 옮기지 않는다 — 옮긴 토큰은 데이터 최적 b 를 재는 회귀 도구로만 쓴다."""
import argparse, collections, json, os, random, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--parts", default="1234")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--nb", type=int, default=1000)
a = ap.parse_args()
REPO = "/workspace/world_model/vjepa2"
RD = os.path.join(REPO, "auto_research/results")
EN = "/workspace/auto_research_cache/enc_noise"
sys.path.insert(0, os.path.join(REPO, "auto_research/scripts"))
G, S = 16, 256
cy, cx = np.divmod(np.arange(S), G)
P0 = np.stack([cx + 0.5, cy + 0.5], 1).astype(np.float64)
CN = np.array([[0, 0], [16, 0], [0, 16], [16, 16]], np.float64)
rng0 = np.random.RandomState(0)


# ------------------------------------------------------------------ 호모그래피
def _n(p):
    return (p - 8.0) / 8.0


def h_apply(H, p):
    q = np.c_[_n(p), np.ones(len(p))] @ H.T
    return q[:, :2] / q[:, 2:3] * 8.0 + 8.0


def h_dlt(p, q):
    x, y = _n(p).T; u, v = _n(q).T; o, z = np.ones_like(x), np.zeros_like(x)
    A = np.empty((2 * len(x), 9))
    A[0::2] = np.stack([-x, -y, -o, z, z, z, u * x, u * y, u], 1)
    A[1::2] = np.stack([z, z, z, -x, -y, -o, v * x, v * y, v], 1)
    H = np.linalg.svd(A, full_matrices=False)[2][-1].reshape(3, 3)
    return H / H[2, 2] if abs(H[2, 2]) > 1e-9 else np.eye(3)


def fit_cam(U):
    """U (256, 2) 블록 b → b−1 칸 변위. return H (b 의 칸 → b−1 의 칸), 인라이어, 설명 몫."""
    U = np.asarray(U, np.float64)
    q = P0 + U
    inl = np.ones(S, bool)
    H = np.eye(3)
    for _ in range(4):
        H = h_dlt(P0[inl], q[inl])
        r = np.linalg.norm(h_apply(H, P0) - q, axis=1)
        new = r <= max(0.25, 2.5 * np.median(r))
        if new.sum() < 32:
            new = r <= np.percentile(r, 50)
        if (new == inl).all():
            break
        inl = new
    F = h_apply(H, P0) - P0
    e = (U[inl] ** 2).sum()
    share = float(1 - ((U[inl] - F[inl]) ** 2).sum() / e) if e > 1e-6 else float("nan")
    return H, inl, share


def cvec(H):
    return (h_apply(H, CN) - CN).ravel()


def hinv(H):
    Hi = np.linalg.inv(H)
    return Hi / Hi[2, 2]


def cam_flow(H):
    return h_apply(H, P0) - P0


def cam_fwd(H):                                           # 블록 b−1 의 칸 s 가 블록 b 에서 갈 자리 − s
    return h_apply(hinv(H), P0) - P0


def cam_mag(H):
    return float(np.median(np.linalg.norm(cam_flow(H), axis=1)))


# ------------------------------------------------------------------ 데이터
class EncTok:
    def __init__(s, src):
        s.T = [np.load(os.path.join(EN, f"{src}_target_encoder_s{sh}_tok.npy"), mmap_mode="r") for sh in (0, 1)]
        s.n0 = s.T[0].shape[0]

    def __getitem__(s, w):
        return s.T[0][w, 0] if w < s.n0 else s.T[1][w - s.n0, 0]


def sources():
    nh = "/workspace/auto_research_cache/natural_holdout"
    out = [("k710_holdout", lambda: np.load(f"{nh}/k710_320_holdout_h.npy", mmap_mode="r"), np.load(f"{nh}/k710_320_holdout_U.npy").astype(np.float32).reshape(-1, 8, S, 2))]
    out.append(("ht100m", lambda: EncTok("ht100m"), np.load(os.path.join(RD, "natural_transport_ctx5/rows_ht100m.npz"))["Uwin"].astype(np.float32).reshape(-1, 8, S, 2)))
    sc = "/workspace/auto_research_cache/ssv2_classes"
    for grp in ("passive", "actor"):
        out.append((f"ssv2_{grp}", (lambda g=grp: np.load(f"{sc}/{g}_h.npy", mmap_mode="r")), np.load(f"{sc}/{grp}_U.npy").astype(np.float32).reshape(-1, 8, S, 2)))
    out.append(("phys3d", lambda: EncTok("phys3d"), np.load(os.path.join(RD, "natural_transport_ctx5/rows_phys3d.npz"))["Uwin"].astype(np.float32).reshape(-1, 8, S, 2)))
    return out


NAT = ("k710_holdout", "ht100m", "ssv2_passive", "ssv2_actor")
CAMS = {}


def cams_for(src, U):
    if src not in CAMS:
        W, NBk = U.shape[:2]
        Hs = np.tile(np.eye(3), (W, NBk, 1, 1)); C = np.zeros((W, NBk, 8)); SH = np.full((W, NBk), np.nan); INL = np.ones((W, NBk))
        for w in range(W):
            for b in range(1, NBk):
                Hs[w, b], inl, SH[w, b] = fit_cam(U[w, b]); C[w, b] = cvec(Hs[w, b]); INL[w, b] = inl.mean()
        CAMS[src] = (Hs, C, SH, INL)
    return CAMS[src]


def cell_of(p):
    s = np.floor(p).astype(int); ok = (s >= 0).all(1) & (s < G).all(1)
    return np.clip(s[:, 1], 0, G - 1) * G + np.clip(s[:, 0], 0, G - 1), ok


def warp(ht, u):
    j, ok = cell_of(P0 + u)
    out = ht[j].copy(); out[~ok] = ht[~ok]
    return out


def splat(ht, u0):
    d = np.floor(P0 - u0).astype(int)
    ok = (d >= 0).all(1) & (d < G).all(1)
    j = d[:, 1] * G + d[:, 0]
    acc = np.zeros_like(ht); n = np.zeros(S)
    np.add.at(acc, j[ok], ht[ok]); np.add.at(n, j[ok], 1)
    out = ht.copy(); m = n > 0
    out[m] = acc[m] / n[m][:, None]
    return out


def r2(Y, P, mu):
    return float(1 - ((Y - P) ** 2).sum() / max(((Y - mu) ** 2).sum(), 1e-12))


def grp_of(Hs, SH, w, t):
    cam = "moving" if cam_mag(Hs[w, t]) >= 0.3 else "still"                  # 문맥 카메라로 고른다
    sh = "hi" if (SH[w, t + 1] >= 0.9) else "lo"
    return cam, sh


srcs = sources()
if a.limit:
    srcs = [(n, f, U[: a.limit]) for n, f, U in srcs]
SUMM = {}

# ------------------------------------------------------------------ (1)
if "1" in a.parts:
    out = {}
    for src, _, U in srcs:
        Hs, C, SH, INL = cams_for(src, U)
        X, Y, wid, mov = [], [], [], []
        for w in range(U.shape[0]):
            for t in range(1, 7):
                X.append(C[w, t]); Y.append(C[w, t + 1]); wid.append(w); mov.append(cam_mag(Hs[w, t]) >= 0.3)
        X, Y, wid, mov = map(np.array, (X, Y, wid, mov))
        e = {"n_pairs": int(len(X)), "frac_cam_moving_ctx": float(mov.mean()), "inlier_frac_median": float(np.median(INL[:, 1:])),
             "share_median_moving": float(np.nanmedian(SH[:, 2:][np.array([[cam_mag(Hs[w, b]) >= 0.3 for b in range(2, 8)] for w in range(U.shape[0])])])) if mov.any() else None}
        for sub, msk in (("all", np.ones(len(X), bool)), ("cam_moving", mov)):
            tr = msk & (wid % 2 == 0); te = msk & (wid % 2 == 1)
            if te.sum() < 10 or tr.sum() < 10:
                continue
            Xa = np.c_[X, np.ones(len(X))]
            Wl = np.linalg.lstsq(Xa[tr], Y[tr], rcond=None)[0]
            e[sub] = dict(n_test=int(te.sum()), R2_linear=r2(Y[te], Xa[te] @ Wl, Y[te].mean(0)), R2_persist=r2(Y[te], X[te], Y[te].mean(0)), c_rms=float(np.sqrt((Y[te] ** 2).mean())))
        out[src] = e
        print(f"(1) {src:13s} 카메라 움직임 (문맥) {e['frac_cam_moving_ctx']:.2f} · 인라이어 {e['inlier_frac_median']:.2f} · 설명 몫 (움직임) {e['share_median_moving']} | "
              + " | ".join(f"{sub}: 선형 R² {e[sub]['R2_linear']:.3f} · 지속 R² {e[sub]['R2_persist']:.3f} (n {e[sub]['n_test']}, c rms {e[sub]['c_rms']:.2f})" for sub in ("all", "cam_moving") if sub in e), flush=True)
    json.dump(out, open(os.path.join(RD, "cam_state_pred.json"), "w"), indent=1)

# ------------------------------------------------------------------ (2)
if "2" in a.parts:
    out = {}
    for src, _, U in srcs:
        Hs, C, SH, _ = cams_for(src, U)
        acc = collections.defaultdict(lambda: {"Y": [], "Xf": [], "Xs": []})
        for w in range(U.shape[0]):
            for t in (4, 5, 6):
                cam, sh = grp_of(Hs, SH, w, t)
                s_, ok = cell_of(P0 + U[w, t + 1])
                for kind, (u1, u0) in (("raw", (U[w, t + 1], U[w, t])), ("cam_removed", (U[w, t + 1] - cam_flow(Hs[w, t + 1]), U[w, t] - cam_flow(Hs[w, t])))):
                    sel = (np.linalg.norm(u1, axis=1) >= 0.5) & ok
                    for g in ([f"{cam}", f"{cam}_{sh}"] if cam == "moving" else [cam]):
                        d = acc[(kind, g)]
                        d["Y"].append(u1[sel]); d["Xf"].append(u0[s_[sel]]); d["Xs"].append(u0[sel])
        e = {}
        for (kind, g), d in acc.items():
            Y = np.concatenate(d["Y"]); row = {"n_cells": int(len(Y))}
            if len(Y) < 20:
                continue
            for xn, nm in (("Xf", "follow"), ("Xs", "same_cell")):
                X = np.concatenate(d[xn]); k = float((Y * X).sum() / max((X * X).sum(), 1e-9))
                row[nm] = dict(k=round(k, 3), R2=round(1 - float(((Y - k * X) ** 2).sum() / max((Y ** 2).sum(), 1e-9)), 3))
            e[f"{kind}|{g}"] = row
        out[src] = e
        print(f"(2) {src:13s} " + " · ".join(f"{g} {kind}: k {e[f'{kind}|{g}']['follow']['k']:.2f} R² {e[f'{kind}|{g}']['follow']['R2']:.2f} (같은 칸 {e[f'{kind}|{g}']['same_cell']['k']:.2f}) n {e[f'{kind}|{g}']['n_cells']}"
                                      for g in ("moving", "moving_hi", "moving_lo", "still") for kind in ("raw", "cam_removed") if f"{kind}|{g}" in e), flush=True)
    json.dump(out, open(os.path.join(RD, "cam_state_persist.json"), "w"), indent=1)

# ------------------------------------------------------------------ (3)
KINDS = ("lag", "camlag", "camonly", "true")


def bstats(XtX, Xty, nb):
    n = XtX.shape[0]

    def fit(cnt):
        A = np.einsum("w,wkij->kij", cnt, XtX); v = np.einsum("w,wki->ki", cnt, Xty)
        return np.array([np.linalg.solve(A[k] + 1e-9 * np.eye(2), v[k])[1] for k in range(len(KINDS))])
    b0 = fit(np.ones(n))
    B = np.array([fit(np.bincount(rng0.randint(0, n, n), minlength=n).astype(float)) for _ in range(nb)])
    ki = {k: i for i, k in enumerate(KINDS)}
    r = {"n_windows": int(n)}
    for k in KINDS:
        r[f"b_{k}"] = float(b0[ki[k]]); r[f"ci_{k}"] = np.percentile(B[:, ki[k]], [2.5, 97.5]).tolist()
    for nm, k2 in (("camlag", "camlag"), ("camonly", "camonly")):
        g = B[:, ki[k2]] - B[:, ki["lag"]]
        r[f"gain_{nm}"] = float(b0[ki[k2]] - b0[ki["lag"]]); r[f"ci_gain_{nm}"] = np.percentile(g, [2.5, 97.5]).tolist()
        den = B[:, ki["true"]] - B[:, ki["lag"]]
        Gs = g / np.where(np.abs(den) > 1e-6, den, np.nan)
        r[f"G_{nm}"] = float((b0[ki[k2]] - b0[ki["lag"]]) / (b0[ki["true"]] - b0[ki["lag"]])); r[f"ci_G_{nm}"] = np.nanpercentile(Gs, [2.5, 97.5]).tolist()
    return r


if "3" in a.parts:
    out = {}
    POOL = collections.defaultdict(lambda: [[], []])
    CELLS = ("obj", "bg", "allmov")
    for src, hget, U in srcs:
        Hs, C, SH, _ = cams_for(src, U)
        Hh = hget()
        W = U.shape[0]
        acc = collections.defaultdict(lambda: [np.zeros((W, len(KINDS), 2, 2)), np.zeros((W, len(KINDS), 2)), np.zeros(W)])
        for w in range(W):
            h = np.asarray(Hh[w]).astype(np.float32)
            for t in (4, 5, 6):
                cam, sh = grp_of(Hs, SH, w, t)
                res = np.linalg.norm(U[w, t + 1] - cam_flow(Hs[w, t + 1]), axis=1); mag = np.linalg.norm(U[w, t + 1], axis=1)
                cellsets = {"obj": res >= 0.5, "bg": (res < 0.5) & (mag >= 0.5), "allmov": mag >= 0.5}
                m = h[t - 4:t + 1].mean(0)
                obj = U[w, t] - cam_flow(Hs[w, t]); Df = cam_fwd(Hs[w, t + 1])
                Ws = {"lag": splat(h[t], U[w, t]), "camlag": splat(h[t], obj - Df), "camonly": splat(h[t], -Df), "true": warp(h[t], U[w, t + 1])}
                groups = [cam] + ([f"{cam}_{sh}"] if cam == "moving" else []) + ["all"]
                for cs, sel in cellsets.items():
                    if sel.sum() < 2:
                        continue
                    y = h[t + 1][sel].ravel().astype(np.float64)
                    for ki, kind in enumerate(KINDS):
                        X = np.stack([m[sel].ravel(), Ws[kind][sel].ravel()], 1).astype(np.float64)
                        xtx, xty = X.T @ X, X.T @ y
                        for g in groups:
                            A_ = acc[(g, cs)]; A_[0][w, ki] += xtx; A_[1][w, ki] += xty
                            if ki == 0:
                                A_[2][w] += sel.sum()
            if w % 100 == 0:
                print(f"(3) {src} {w}/{W}", flush=True)
        e = {}
        for (g, cs), (XtX, Xty, n) in acc.items():
            okw = n > 0
            if okw.sum() < 10:
                continue
            r = bstats(XtX[okw], Xty[okw], a.nb); r["n_cells"] = int(n.sum())
            e[f"{g}|{cs}"] = r
            if src in NAT:
                POOL[(g, cs)][0].append(XtX[okw]); POOL[(g, cs)][1].append(Xty[okw])
        out[src] = e
        for key in ("moving|obj", "moving_hi|obj", "moving_lo|obj", "still|obj", "moving|bg", "moving|allmov"):
            if key in e:
                r = e[key]
                print(f"(3) {src:13s} {key:14s} 창 {r['n_windows']:3d} 칸 {r['n_cells']:6d} | b 문맥 {r['b_lag']:.3f} · 카메라앎 {r['b_camlag']:.3f} · 카메라만 {r['b_camonly']:.3f} · 참 {r['b_true']:.3f} [{r['ci_true'][0]:.3f}, {r['ci_true'][1]:.3f}]"
                      f" | 이득 {r['gain_camlag']:+.3f} [{r['ci_gain_camlag'][0]:+.3f}, {r['ci_gain_camlag'][1]:+.3f}] · G {r['G_camlag']:.2f} [{r['ci_G_camlag'][0]:.2f}, {r['ci_G_camlag'][1]:.2f}]", flush=True)
    pooled = {}
    for (g, cs), (xs, ys) in POOL.items():
        r = bstats(np.concatenate(xs), np.concatenate(ys), a.nb)
        pooled[f"{g}|{cs}"] = r
    out["natural_pooled"] = pooled
    for key in ("moving|obj", "moving_hi|obj", "moving_lo|obj", "still|obj", "moving|bg", "moving|allmov"):
        if key in pooled:
            r = pooled[key]
            print(f"(3) 자연 합침      {key:14s} 창 {r['n_windows']:3d} | b 문맥 {r['b_lag']:.3f} · 카메라앎 {r['b_camlag']:.3f} · 카메라만 {r['b_camonly']:.3f} · 참 {r['b_true']:.3f}"
                  f" | 이득 {r['gain_camlag']:+.3f} [{r['ci_gain_camlag'][0]:+.3f}, {r['ci_gain_camlag'][1]:+.3f}] · G {r['G_camlag']:.2f} [{r['ci_G_camlag'][0]:.2f}, {r['ci_G_camlag'][1]:.2f}]", flush=True)
    json.dump(out, open(os.path.join(RD, "cam_state_b.json"), "w"), indent=1)

# ------------------------------------------------------------------ (4)
if "4" in a.parts:
    import crit_ip2_pmi as PMI
    out = {}
    D = "/workspace/auto_research_cache/intphys2/Main"
    meta = json.load(open(os.path.join(D, "meta.json")))
    Ui = np.load(os.path.join(D, "U.npy")).astype(np.float32).reshape(len(meta), 32, S, 2)
    Vi = np.load(os.path.join(D, "V.npy")).astype(np.float32).reshape(len(meta), 32, S, 2)
    CU = np.zeros((len(meta), 32, 8)); CV = np.zeros((len(meta), 32, 8)); SHi = np.full((len(meta), 32), np.nan); MAGi = np.zeros((len(meta), 32))
    for i in range(len(meta)):
        for b in range(1, 32):
            H, _, SHi[i, b] = fit_cam(Ui[i, b]); CU[i, b] = cvec(H); MAGi[i, b] = cam_mag(H)
            Hv, _, _ = fit_cam(Vi[i, b - 1]); CV[i, b] = cvec(hinv(Hv))                    # V[b−1] = b−1 → b, 역사상 = b → b−1
        if i % 200 == 0:
            print(f"(4) IntPhys 2 사상 {i}/{len(meta)}", flush=True)
    rows_by = {(m["scene_index"], m["type"]): m["row"] for m in meta}
    recs = json.load(open(os.path.join(RD, "ip2_decomp.json")))["records"]
    ev = {(r["scene"], r["pair"]): r.get("b0") for r in recs}
    nz = np.linalg.norm(CU - CV, axis=-1)                                                      # 반복 (U 대 V⁻¹) 차 — 참고
    pairs = []
    for (scn, pid), b0 in ev.items():
        p, q = rows_by[(scn, f"{pid}_Possible")], rows_by[(scn, f"{pid}_Impossible")]
        d = np.linalg.norm(CU[p] - CU[q], axis=-1)
        r = dict(scene=scn, pair=pid, cam=meta[p]["camera"], cond=meta[p]["condition"], b0=b0, noise_UV=float(nz[[p, q], 1:].mean()))
        if b0 is not None:
            K = min(8, 32 - b0)
            r.update(D_pre=float(d[1:b0].mean()), D_post=float(d[b0:b0 + K].mean()), D_pre_max=float(d[1:b0].max()), D_post_max=float(d[b0:b0 + K].max()))
        pairs.append(r)
    ab = {}
    for cam in ("all", "Fixed", "Moving"):
        xs = [r for r in pairs if r.get("D_pre") is not None and (cam == "all" or r["cam"] == cam)]
        if not xs:
            continue
        pre = np.array([r["D_pre"] for r in xs]); post = np.array([r["D_post"] for r in xs])
        ratio = post / np.maximum(pre, 1e-6)
        ab[cam] = dict(n=len(xs), D_pre_median=float(np.median(pre)), D_post_median=float(np.median(post)), ratio_median=float(np.median(ratio)),
                       frac_ratio_gt3=float((ratio > 3).mean()), frac_post_gt_pre=float((post > pre).mean()),
                       noise_UV_median=float(np.median([r["noise_UV"] for r in xs])))
        print(f"(4) IntPhys 2 {cam:6s} 짝 {len(xs)}: 짝 c 차 사건 전 {ab[cam]['D_pre_median']:.3f} · 사건 뒤 {ab[cam]['D_post_median']:.3f} 칸 · 비 중앙 {ab[cam]['ratio_median']:.2f} · 비 > 3 {ab[cam]['frac_ratio_gt3']:.2f} · (U 대 V⁻¹ {ab[cam]['noise_UV_median']:.3f})", flush=True)
    out["intphys2_pair_c"] = dict(summary=ab, pairs=pairs)
    sh_mov = SHi[[m["row"] for m in meta if m["camera"] == "Moving"], 1:]
    mg_mov = MAGi[[m["row"] for m in meta if m["camera"] == "Moving"], 1:]
    out["intphys2_share_moving"] = dict(median=float(np.nanmedian(sh_mov[mg_mov >= 0.3])), frac_ge_0p9=float(np.nanmean(sh_mov[mg_mov >= 0.3] >= 0.9)), n=int((mg_mov >= 0.3).sum()))
    print(f"(4) IntPhys 2 이동 카메라 블록 (카메라 ≥ 0.3 칸) 설명 몫 중앙 {out['intphys2_share_moving']['median']:.3f} · ≥ 0.9 비율 {out['intphys2_share_moving']['frac_ge_0p9']:.2f} (n {out['intphys2_share_moving']['n']})", flush=True)
    # 카메라 상태만의 surprise (미래 c − 문맥 외삽 c, 외삽 = 직전 블록 c 그대로) → 표준 보고
    rng = random.Random(0)
    rows_all, rows_post = [], []
    for m in meta:
        i = m["row"]; e = np.linalg.norm(CU[i, 5:32] - CU[i, 4:31], axis=-1)                   # 블록 b+1 = 5..31 (crit_ip2_pmi B0 = 4 와 같은 구간)
        base = dict(scene=m["scene_index"], pair=str(m["pair_id"]), imp=bool(m["is_impossible"]), cond=m["condition"], cam=m["camera"])
        rows_all.append(dict(base, s=float(e.mean())))
        b0 = ev.get((m["scene_index"], str(m["pair_id"])))
        if b0 is not None:
            K = min(8, 32 - b0)
            rows_post.append(dict(base, s=float(np.linalg.norm(CU[i, b0:b0 + K] - CU[i, b0 - 1:b0 + K - 1], axis=-1).mean())))
    std = {}
    for nm, rows in (("all_blocks", rows_all), ("post_event", rows_post)):
        for cam in ("all", "Fixed", "Moving"):
            xs = [r for r in rows if cam == "all" or r["cam"] == cam]
            st = PMI.decomp(xs, "s", False, rng)
            std[f"{nm}|{cam}"] = st
            print(f"(4) IntPhys 2 카메라만 surprise [{nm} · {cam}] n {st['n']} 정확도 {st['acc']:.1f} [{st['acc_ci'][0]:.1f}, {st['acc_ci'][1]:.1f}] 짝 1/2 {st['pair1']:.1f}/{st['pair2']:.1f} "
                  f"결과 항 몫 {st['outcome_share']:.2f} 문맥 항 {st['mean_i']:+.4f} [{st['mean_i_ci'][0]:+.4f}, {st['mean_i_ci'][1]:+.4f}]", flush=True)
    out["intphys2_cam_surprise"] = std
    # VOBench
    vp = "/workspace/auto_research_cache/cam_state/vob_UV.npz"
    if os.path.exists(vp):
        Z = np.load(vp); Uv = Z["U"].astype(np.float32).reshape(-1, 4, 8, S, 2); Vv = Z["V"].astype(np.float32).reshape(-1, 4, 8, S, 2)
        vm = json.load(open("/workspace/auto_research_cache/vobench_late/meta.json"))
        Cv = np.zeros(Uv.shape[:3] + (8,)); Cw = np.zeros_like(Cv); MGv = np.zeros(Uv.shape[:3])
        for i in range(Uv.shape[0]):
            for k in range(4):
                for j in range(8):
                    H, _, _ = fit_cam(Uv[i, k, j]); Cv[i, k, j] = cvec(H); MGv[i, k, j] = cam_mag(H)
                    Hv, _, _ = fit_cam(Vv[i, k, j]); Cw[i, k, j] = cvec(hinv(Hv))
        rows = []
        for i, m in enumerate(vm):
            for k, (arm, pid, imp) in enumerate((("pos_a", "1", False), ("pos_b", "2", False), ("imp_ab", "1", True), ("imp_ba", "2", True))):
                rows.append(dict(scene=m["block_id"], pair=pid, imp=imp, s=float(np.linalg.norm(Cv[i, k, 1:] - Cv[i, k, :-1], axis=-1).mean()), motion=m["motion"], viol=m["violation"]))
        st = PMI.decomp(rows, "s", False, rng)
        dpair = np.concatenate([np.linalg.norm(Cv[:, 0] - Cv[:, 2], axis=-1).mean(1), np.linalg.norm(Cv[:, 1] - Cv[:, 3], axis=-1).mean(1)])
        out["vobench"] = dict(cam_surprise=st, cam_mag_median=float(np.median(MGv)), pair_c_diff_median=float(np.median(dpair)),
                              noise_UV_median=float(np.median(np.linalg.norm(Cv - Cw, axis=-1))))
        print(f"(4) VOBench 카메라 크기 중앙 {out['vobench']['cam_mag_median']:.3f} 칸 · 짝 c 차 중앙 {out['vobench']['pair_c_diff_median']:.3f} · (U 대 V⁻¹ {out['vobench']['noise_UV_median']:.3f}) | "
              f"카메라만 surprise n {st['n']} 정확도 {st['acc']:.1f} [{st['acc_ci'][0]:.1f}, {st['acc_ci'][1]:.1f}] 짝 1/2 {st['pair1']:.1f}/{st['pair2']:.1f} 결과 항 몫 {st['outcome_share']:.2f} 문맥 항 {st['mean_i']:+.4f} [{st['mean_i_ci'][0]:+.4f}, {st['mean_i_ci'][1]:+.4f}]", flush=True)
    json.dump(out, open(os.path.join(RD, "cam_state_absorb.json"), "w"), indent=1, default=float)
print("끝")
