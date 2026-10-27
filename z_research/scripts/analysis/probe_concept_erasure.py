"""블록별 전체 평균 (1, D) 선형 probe + 개념 벡터 지우기 (probe 고정) — encoder · 블록 AR 층별 (2026-10-26, 학습 없음, CPU).
사용자: "(N, D) → mean pool (1, D) 해서 각 block 별로 object identity (shape, color), object motion direction 이렇게 linear probe 학습해봐.
그리고 (1, D) 에서 concept vector 빼고 linear probe (frozen) forward 시켰을 때 acc drop" · "probe 는 왜 구역을 또 나눠" → 연속값은 구간 없이 회귀.

입력 pool_objcells.py 출력 (블록마다 칸 256 개 평균). 표현 × 묶음:
  enc (1280): ctx 블록 0–7 · fut 8–15 · s12 / s35 / s68 (블록 8–9 / 10–12 / 13–15)
  tf_L{0..13} (384) · tf_out (1280): ctx · fut          ro_L{0..13} (384): s12 · s35 · s67 (걸음 1–2 / 3–5 / 6–7) · ro_out (1280): s12 · s35 · s68
표본 = 물체가 보이는 블록. 특징은 train 평균 · 표준편차로 표준화 (z).
probe (ridge, λ 는 train 클립 80/20 로 고르고 train 전체로 다시 맞춤):
  shape 6 · color 8 분류 (정확도) · direction = 블록 안 변위의 (cos θ, sin θ) 회귀 (변위 ≥ 2 px 블록만, 각 오차 중앙값 · 45° 안 비율) ·
  speed = |변위| 회귀 (R²) · position = 물체 중심 (x, y) 회귀 (R²)
개념 벡터 (train, z 공간): shape · color = 클래스 평균 − 클래스 평균들의 평균 (K, D) · 연속값 = 중심화한 라벨로 가중한 특징 평균 (공분산 방향):
  direction (2, D) · speed (1, D) · position (2, D). 지우기 = 그 벡터들이 펼치는 부분공간을 투영으로 제거 z' = z − (z Q) Qᵀ.
val 에서 원본 · 개념 제거 · 같은 차원 무작위 제거 (train 공분산을 따르는 방향, 5 회 평균) 에 고정 probe 를 그대로 forward.

  /venv/vjepa/bin/python z_research/scripts/analysis/probe_concept_erasure.py
출력: z_analysis/results/probe_concept_erasure.json
"""
import json, os
import numpy as np

P = "/workspace/auto_research_cache/ar_layers_pooled"
OUTJ = "/workspace/world_model/vjepa2/z_analysis/results/probe_concept_erasure.json"
TR, VA = dict(np.load(f"{P}/train.npz")), dict(np.load(f"{P}/val.npz"))
rng = np.random.default_rng(0)
LAMS = (1e0, 1e1, 1e2, 1e3, 1e4)
FAMS = ("shape", "color", "direction", "speed", "position")


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
            "position": (L[:, 6:8], np.ones(len(L), bool))}


K = {"shape": 6, "color": 8}


def Ymat(fam, y):
    return np.eye(K[fam])[y] if fam in K else y


def fit_ridge(Z, Y, tr_mask):
    """λ 를 tr_mask (80 %) → 나머지 20 % 로 고르고 전체로 다시. 고유분해 재사용."""
    def solver(Zs):
        ev, V = np.linalg.eigh(Zs.T @ Zs); return ev, V
    ym = Y[tr_mask].mean(0)
    ev, V = solver(Z[tr_mask]); ZtY = Z[tr_mask].T @ (Y[tr_mask] - ym)
    best = None
    for lam in LAMS:
        W = V @ ((V.T @ ZtY) / (ev + lam)[:, None]); e = (((Z[~tr_mask] @ W + ym) - Y[~tr_mask]) ** 2).sum()
        if best is None or e < best[0]:
            best = (e, lam)
    ym = Y.mean(0); ev, V = solver(Z); W = V @ ((V.T @ (Z.T @ (Y - ym))) / (ev + best[1])[:, None])
    return W, ym, best[1]


def metric(fam, P_, y):
    if fam in K:
        return round(float((P_.argmax(1) == y).mean() * 100), 1)
    if fam == "direction":
        d = np.degrees(np.abs(np.angle(np.exp(1j * (np.arctan2(P_[:, 1], P_[:, 0]) - np.arctan2(y[:, 1], y[:, 0]))))))
        return round(float(np.median(d)), 1)                                     # 각 오차 중앙값 (°), 우연 ≈ 90
    return round(float(1 - ((P_ - y) ** 2).sum() / ((y - y.mean(0)) ** 2).sum()), 3)  # R²


def concept_basis(fam, Z, y):
    if fam in K:
        M = np.stack([Z[y == k].mean(0) for k in range(K[fam]) if (y == k).sum() >= 5]); C = M - M.mean(0)
    else:
        yc = y - y.mean(0); C = (yc.T @ Z) / len(Z)                              # (dim, D) 공분산 방향
    U, s, Vt = np.linalg.svd(C, full_matrices=False)
    r = (K[fam] - 1) if fam in K else C.shape[0]
    return Vt[:r].T                                                              # (D, r)


out = {}
for key, (getter, lb, sl) in views().items():
    Xt, Lt, ct = gather(TR, getter, lb, sl); Xv, Lv, _ = gather(VA, getter, lb, sl)
    if len(Xt) < 500 or len(Xv) < 50:
        continue
    mu, sd = Xt.mean(0), Xt.std(0) + 1e-6
    Zt, Zv = (Xt - mu) / sd, (Xv - mu) / sd
    tr_mask = (ct % 5) != 0                                                       # 클립 단위 80 / 20
    Tt, Tv = targets(Lt), targets(Lv)
    probes, bases = {}, {}
    for fam in FAMS:
        yt, mt = Tt[fam]
        W, b, lam = fit_ridge(Zt[mt], Ymat(fam, yt[mt]), tr_mask[mt]); probes[fam] = (W, b, lam)
        bases[fam] = concept_basis(fam, Zt[mt], yt[mt])
    Sig = np.cov(Zt[rng.choice(len(Zt), min(len(Zt), 30000), replace=False)], rowvar=False)
    ev, Ve = np.linalg.eigh(Sig); sq = Ve * np.sqrt(np.clip(ev, 0, None))
    def evaluate(Zx):
        r = {}
        for fam in FAMS:
            yv, mv = Tv[fam]; W, b, _ = probes[fam]
            r[fam] = metric(fam, Zx[mv] @ W + b, yv[mv])
        return r
    res = {"n_train": int(len(Zt)), "n_val": int(len(Zv)), "lambda": {f: probes[f][2] for f in FAMS}, "original": evaluate(Zv)}
    for fam in FAMS:
        Q = bases[fam]; res[f"remove_{fam}({Q.shape[1]})"] = evaluate(Zv - (Zv @ Q) @ Q.T)
    for r_ in sorted({bases[f].shape[1] for f in FAMS}):
        rs = []
        for _ in range(5):
            Qr, _ = np.linalg.qr(sq @ rng.standard_normal((Zt.shape[1], r_))); rs.append(evaluate(Zv - (Zv @ Qr) @ Qr.T))
        res[f"remove_random({r_})"] = {f: round(float(np.mean([x[f] for x in rs])), 3) for f in FAMS}
    out["|".join(key)] = res
    o = res["original"]
    print(f"{'|'.join(key):14s} shape {o['shape']:5.1f}→{res['remove_shape(5)']['shape']:5.1f} color {o['color']:5.1f}→{res['remove_color(7)']['color']:5.1f} "
          f"dir {o['direction']:5.1f}°→{res['remove_direction(2)']['direction']:5.1f}° speed {o['speed']:.2f}→{res['remove_speed(1)']['speed']:.2f} "
          f"pos {o['position']:.2f}→{res['remove_position(2)']['position']:.2f}", flush=True)
os.makedirs(os.path.dirname(OUTJ), exist_ok=True)
json.dump(out, open(OUTJ, "w"), indent=1, ensure_ascii=False)
print("saved", OUTJ, len(out))
