"""개념 벡터 빼기 · 조작 (steering) — 고정 probe 로 검증, encoder · 블록 AR 층별 (2026-10-26, 학습 없음, CPU).
사용자: "그냥 원본 (1, 1280) − concept vector (1, 1280) 선형 결합으로 하면 되는 거 아니야?" → "빼기를 해야지. 그래야 steering 되는지도 봐야 됨".

입력·표현·묶음·표준화·probe 는 probe_concept_erasure.py 와 같다 (블록마다 칸 256 개 평균 (1, D), 물체 보이는 블록, ridge probe 고정).
개념 벡터 (train, 표현마다 그 공간에서):
  shape · color : c_k = m_k − (클래스 평균들의 평균)                                  (K, D)
  direction (cos θ, sin θ) · position (x, y) · speed : B = Cov(z, y) Cov(y)⁻¹ (라벨 한 단위당 특징 평균 이동, (D, q))
빼기 (라벨 사용):  shape · color z − α c_y  ·  연속값 z − α B (y − ȳ)
조작 (라벨 사용):  shape · color z − α c_y + α c_t (t = y 가 아닌 모든 클래스)  ·  direction 목표 = θ + 90°  ·  position 목표 = (x + 64, y)  ·  speed 목표 = s + 4 px
  → 연속값 조작은 z + α B (t − y)
지표 (val): 빼기 = 원래 라벨 기준 정확도 / 각 오차 / R² · 조작 = 목표로 읽힌 비율 (shape · color) · 목표와의 각 오차 중앙값 (direction) ·
  읽힌 이동 / 의도한 이동 (position x 이득, y 는 0 이어야) · speed 이득.
대조: 조작 벡터와 같은 크기의 무작위 벡터 (train 공분산 방향) 를 더했을 때 같은 지표 (3 회 평균).

  /venv/vjepa/bin/python z_research/scripts/analysis/concept_steering.py
출력: z_analysis/results/concept_steering.json
"""
import json, os
import numpy as np

P = "/workspace/auto_research_cache/ar_layers_pooled"
OUTJ = "/workspace/world_model/vjepa2/z_analysis/results/concept_steering.json"
TR, VA = dict(np.load(f"{P}/train.npz")), dict(np.load(f"{P}/val.npz"))
rng = np.random.default_rng(0)
LAMS = (1e0, 1e1, 1e2, 1e3, 1e4)
K = {"shape": 6, "color": 8}
ALPHAS = (1.0, 2.0)


def views():
    V = {}
    b16 = np.arange(16)
    for g, sl in (("ctx", range(0, 8)), ("fut", range(8, 16)), ("s12", range(8, 10)), ("s35", range(10, 13)), ("s68", range(13, 16))):
        V[("enc", g)] = (lambda Z: Z["enc"], b16, list(sl))
    for l in range(14):
        for g, sl in (("ctx", range(0, 8)), ("fut", range(8, 16))):
            V[(f"tf_L{l}", g)] = (lambda Z, l=l: Z["tfh"][:, l], b16, list(sl))
        for g, sl in (("s12", range(0, 2)), ("s35", range(2, 5)), ("s67", range(5, 7))):
            V[(f"ro_L{l}", g)] = (lambda Z, l=l: Z["roh"][:, l], np.arange(8) + 8, list(sl))
    for g, sl in (("ctx", range(1, 8)), ("fut", range(8, 16))):
        V[("tf_out", g)] = (lambda Z: Z["tfout"], b16, list(sl))
    for g, sl in (("s12", range(0, 2)), ("s35", range(2, 5)), ("s68", range(5, 8))):
        V[("ro_out", g)] = (lambda Z: Z["roout"], np.arange(8) + 8, list(sl))
    return V


def gather(Z, getter, lb, sl):
    X = getter(Z)[:, sl].astype(np.float32); L = Z["lab"][:, lb[sl]]
    clip = np.repeat(np.arange(X.shape[0])[:, None], len(sl), 1)
    X, L, clip = X.reshape(-1, X.shape[-1]), L.reshape(-1, 8), clip.reshape(-1)
    ok = (L[:, 0] > 0) & ~np.isnan(X).any(1)
    return X[ok], L[ok], clip[ok]


def targets(L):
    dx, dy = L[:, 4], L[:, 5]; s = np.hypot(dx, dy); th = np.arctan2(dy, dx)
    return {"shape": (L[:, 1].astype(int), np.ones(len(L), bool)), "color": (L[:, 2].astype(int), np.ones(len(L), bool)),
            "direction": (np.stack([np.cos(th), np.sin(th)], 1), s >= 2.0), "speed": (s[:, None], np.ones(len(L), bool)),
            "position": (L[:, 6:8].astype(np.float64), np.ones(len(L), bool))}


def fit_ridge(Z, Y, tr_mask):
    ym = Y[tr_mask].mean(0); ev, V = np.linalg.eigh(Z[tr_mask].T @ Z[tr_mask]); ZtY = Z[tr_mask].T @ (Y[tr_mask] - ym)
    best = None
    for lam in LAMS:
        W = V @ ((V.T @ ZtY) / (ev + lam)[:, None]); e = (((Z[~tr_mask] @ W + ym) - Y[~tr_mask]) ** 2).sum()
        if best is None or e < best[0]:
            best = (e, lam)
    ym = Y.mean(0); ev, V = np.linalg.eigh(Z.T @ Z); W = V @ ((V.T @ (Z.T @ (Y - ym))) / (ev + best[1])[:, None])
    return W, ym


ang = lambda P_: np.arctan2(P_[:, 1], P_[:, 0])
aerr = lambda a, b: np.degrees(np.abs(np.angle(np.exp(1j * (a - b)))))
r2 = lambda P_, y: float(1 - ((P_ - y) ** 2).sum() / ((y - y.mean(0)) ** 2).sum())

out = {}
for key, (getter, lb, sl) in views().items():
    Xt, Lt, ct = gather(TR, getter, lb, sl); Xv, Lv, _ = gather(VA, getter, lb, sl)
    if len(Xt) < 500 or len(Xv) < 50:
        continue
    mu, sd = Xt.mean(0), Xt.std(0) + 1e-6; Zt, Zv = (Xt - mu) / sd, (Xv - mu) / sd
    tr_mask = (ct % 5) != 0; Tt, Tv = targets(Lt), targets(Lv)
    Sig = np.cov(Zt[rng.choice(len(Zt), min(len(Zt), 30000), replace=False)], rowvar=False)
    evs, Ves = np.linalg.eigh(Sig); sq = Ves * np.sqrt(np.clip(evs, 0, None))
    def randlike(V_):                                                        # V_ (n, D) 와 행마다 같은 크기의 무작위 (공분산 방향)
        g = rng.standard_normal(V_.shape) @ sq.T
        return g / np.linalg.norm(g, axis=1, keepdims=True) * np.linalg.norm(V_, axis=1, keepdims=True)
    res = {"n_train": int(len(Zt)), "n_val": int(len(Zv))}
    for fam in ("shape", "color"):
        yt, _ = Tt[fam]; yv, _ = Tv[fam]
        W, b = fit_ridge(Zt, np.eye(K[fam])[yt], tr_mask)
        M = np.stack([Zt[yt == k].mean(0) for k in range(K[fam])]); C = M - M.mean(0)
        pred = lambda Zx: (Zx @ W + b).argmax(1)
        r = {"original": round(float((pred(Zv) == yv).mean() * 100), 1), "chance": round(100 / K[fam], 1), "concept_norm_over_sample_norm": round(float(np.linalg.norm(C, axis=1).mean() / np.linalg.norm(Zv, axis=1).mean()), 3)}
        for al in ALPHAS:
            r[f"subtract_a{al:g}"] = round(float((pred(Zv - al * C[yv]) == yv).mean() * 100), 1)
            succ, keep, nsucc = [], [], []
            for t in range(K[fam]):
                m = yv != t
                Vst = al * (C[t] - C[yv[m]]); Zs = Zv[m] + Vst; p = pred(Zs)
                succ.append((p == t).mean()); keep.append((p == yv[m]).mean())
                ns = []
                for _ in range(3):
                    ns.append((pred(Zv[m] + randlike(Vst)) == t).mean())
                nsucc.append(np.mean(ns))
            r[f"steer_a{al:g}_to_target"] = round(float(np.mean(succ) * 100), 1); r[f"steer_a{al:g}_kept_original"] = round(float(np.mean(keep) * 100), 1)
            r[f"steer_a{al:g}_random_to_target"] = round(float(np.mean(nsucc) * 100), 1)
        res[fam] = r
    for fam in ("direction", "position", "speed"):
        yt, mt = Tt[fam]; yv, mv = Tv[fam]
        W, b = fit_ridge(Zt[mt], yt[mt], tr_mask[mt])
        yc = yt[mt] - yt[mt].mean(0); B = ((yc.T @ Zt[mt]) / len(yc)).T @ np.linalg.pinv(np.atleast_2d(np.cov(yc, rowvar=False)))   # (D, q)
        Zm, ym_ = Zv[mv], yv[mv]; Pm = Zm @ W + b
        if fam == "direction":
            th = ang(ym_); tgt = np.stack([np.cos(th + np.pi / 2), np.sin(th + np.pi / 2)], 1)
            r = {"original_err_deg": round(float(np.median(aerr(ang(Pm), th))), 1)}
        elif fam == "position":
            tgt = ym_ + np.array([64.0, 0.0]); r = {"original_r2": round(r2(Pm, ym_), 3)}
        else:
            tgt = ym_ + 4.0; r = {"original_r2": round(r2(Pm, ym_), 3)}
        for al in ALPHAS:
            Zsub = Zm - al * (ym_ - yt[mt].mean(0)) @ B.T
            Psub = Zsub @ W + b
            Vst = al * (tgt - ym_) @ B.T; Pst = (Zm + Vst) @ W + b
            Pnull = np.mean([(Zm + randlike(Vst)) @ W + b for _ in range(3)], 0)
            if fam == "direction":
                r[f"subtract_a{al:g}_err_deg"] = round(float(np.median(aerr(ang(Psub), th))), 1)
                r[f"steer_a{al:g}_err_to_target_deg"] = round(float(np.median(aerr(ang(Pst), ang(tgt)))), 1)
                r[f"steer_a{al:g}_random_err_to_target_deg"] = round(float(np.median(aerr(ang(Pnull), ang(tgt)))), 1)
            elif fam == "position":
                r[f"subtract_a{al:g}_r2"] = round(r2(Psub, ym_), 3)
                d = Pst - Pm; dn = Pnull - Pm
                r[f"steer_a{al:g}_x_gain"] = round(float(d[:, 0].mean() / 64.0), 3); r[f"steer_a{al:g}_y_shift_px"] = round(float(d[:, 1].mean()), 1)
                r[f"steer_a{al:g}_random_x_gain"] = round(float(dn[:, 0].mean() / 64.0), 3)
            else:
                r[f"subtract_a{al:g}_r2"] = round(r2(Psub, ym_), 3)
                r[f"steer_a{al:g}_gain"] = round(float((Pst - Pm).mean() / 4.0), 3); r[f"steer_a{al:g}_random_gain"] = round(float((Pnull - Pm).mean() / 4.0), 3)
        res[fam] = r
    out["|".join(key)] = res
    s, c, d, p = res["shape"], res["color"], res["direction"], res["position"]
    print(f"{'|'.join(key):14s} shape {s['original']:5.1f} sub {s['subtract_a1']:5.1f} steer→t {s['steer_a1_to_target']:5.1f}/{s['steer_a2_to_target']:5.1f} (rand {s['steer_a1_random_to_target']:4.1f}) | "
          f"color {c['original']:5.1f} sub {c['subtract_a1']:5.1f} steer {c['steer_a1_to_target']:5.1f}/{c['steer_a2_to_target']:5.1f} (rand {c['steer_a1_random_to_target']:4.1f}) | "
          f"dir {d['original_err_deg']:5.1f}° sub {d['subtract_a1_err_deg']:5.1f}° steer→+90 {d['steer_a1_err_to_target_deg']:5.1f}° (rand {d['steer_a1_random_err_to_target_deg']:5.1f}) | "
          f"pos x-gain {p['steer_a1_x_gain']:.2f} (rand {p['steer_a1_random_x_gain']:.2f})", flush=True)
json.dump(out, open(OUTJ, "w"), indent=1, ensure_ascii=False)
print("saved", OUTJ, len(out))
