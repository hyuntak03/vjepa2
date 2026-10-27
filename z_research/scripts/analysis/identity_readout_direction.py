"""정체 (모양 · 색) 를 읽는 방향 찾기 — 라벨 있는 다른 정보를 먼저 빼고 개념 방향을 다시 구한다 (2026-10-26, 학습 없음, CPU).
사용자: "정체를 읽는 방향을 찾으려면 어떻게 해야 되는 거지".
배경: 블록 전체 평균 (1, 1280) 에서 개념 벡터 (클래스 평균 − 클래스 평균들의 평균) 방향 위 분산 중 정체 몫이 모양 16 % · 색 4 % 뿐이라
  개념 방향만으로는 encoder 에서도 모양 40 % · 색 20 % (identity_probe_transfer.py). 그 방향 위 흔들림이 라벨 있는 다른 정보
  (장면 종류 · 바닥 · 벽 색 · 위치 · 움직임 · 물리 상수) 라면, 그것을 선형으로 빼고 남은 것 (잔차) 에서 개념 방향이 곧 읽는 방향이 된다.
다른 정보 설계 (블록마다): 장면 13 종 원핫 · 바닥 RGB · 벽 RGB (1 · 2 차) · 위치 (x, y) 와 4×4 칸 원핫 · 블록 변위 (dx, dy, 크기) · g · 반발 · 마찰.
  공간 (encoder / AR 굴린 예측 / 릴리즈 정답 / 릴리즈 예측) 마다 자기 train 에서 x ≈ [다른 정보, 1]·A 를 최소제곱으로 맞추고 잔차 = x − 맞춘 값 (val 은 val 의 라벨로).
판독기 (블록 b = 7+k 마다, train 8,000 편 중 물체 보이는 표본 → val 350 편 중 물체 보이는 표본):
  중심 = (z − 자기 train 평균) · 단위 개념 벡터 최대 / ridge = identity_probe_transfer.py 와 같음 (one-hot 릿지, λ train 클립 80/20).
  원본과 잔차 각각에서 encoder → encoder · encoder → 예측 · 예측 → 예측. 개념 부분공간 위 분산 중 정체 몫 (η²) 도.
  /venv/vjepa/bin/python z_research/scripts/analysis/identity_readout_direction.py   → z_analysis/results/identity_readout_direction.json
"""
import json
import numpy as np
P = "/workspace/auto_research_cache"
LAMS = (1e-1, 1e0, 1e1, 1e2, 1e3, 1e4)


def fit(Z, Y, clip):
    tr = (clip % 5) != 0; ev, V = np.linalg.eigh(Z[tr].T @ Z[tr]); ym = Y[tr].mean(0); ZtY = Z[tr].T @ (Y[tr] - ym)
    errs = [(((Z[~tr] @ (V @ ((V.T @ ZtY) / (ev + l)[:, None])) + ym) - Y[~tr]) ** 2).sum() for l in LAMS]
    l = LAMS[int(np.argmin(errs))]; ev, V = np.linalg.eigh(Z.T @ Z); ym = Y.mean(0)
    return V @ ((V.T @ (Z.T @ (Y - ym))) / (ev + l)[:, None]), ym


def concepts(X, y, K):
    C = np.stack([X[y == c].mean(0) for c in range(K)]); C -= C.mean(0)
    return C / np.linalg.norm(C, axis=1, keepdims=True)


def eta2(X, y, K):                                   # 개념 부분공간 (K−1 차원) 위 분산 중 클래스 몫
    Xc = X - X.mean(0); M = np.stack([Xc[y == c].mean(0) for c in range(K)]); U = np.linalg.svd(M - M.mean(0), full_matrices=False)[2][:K - 1]
    Z = Xc @ U.T; Zm = np.stack([Z[y == c].mean(0) for c in range(K)])[y]
    return float((Zm ** 2).sum() / (Z ** 2).sum())


def design(lab, env, b):
    n = len(lab); s = np.eye(13)[lab[:, b, 3].astype(int)]
    rgb = env[:, :6]; phys = env[:, 6:]
    cx, cy = lab[:, b, 6] / 256, lab[:, b, 7] / 256; dx, dy = lab[:, b, 4] / 64, lab[:, b, 5] / 64
    grid = np.eye(16)[np.clip((cx * 4).astype(int), 0, 3) * 4 + np.clip((cy * 4).astype(int), 0, 3)]
    return np.concatenate([s, rgb, rgb ** 2, np.stack([cx, cy, dx, dy, np.hypot(dx, dy)], 1), grid, phys, np.ones((n, 1))], 1)


def resid(Xtr, Ntr, Xva, Nva):
    A = np.linalg.solve(Ntr.T @ Ntr + 1e-3 * np.eye(Ntr.shape[1]), Ntr.T @ Xtr)
    r2 = 1 - ((Xva - Nva @ A) ** 2).sum() / ((Xva - Xtr.mean(0)) ** 2).sum()
    return Xtr - Ntr @ A, Xva - Nva @ A, float(r2)


def load(split):
    L = dict(np.load(f"{P}/ar_layers_pooled/{split}.npz"))
    env = []
    for c in L["clip"]:
        e = json.load(open(str(c)[:-4] + ".json"))["env"]
        env.append(list(np.array(e["floor_rgb"]) / 255) + list(np.array(e["wall_rgb"]) / 255) + [e["g"] / 9.8, e["bounce"], e["friction"]])
    R = dict(np.load(f"{P}/release_pooled/{split}.npz")); o = {int(v): n for n, v in enumerate(L["i"])}; sel = [o[int(v)] for v in R["i"]]
    return {"ar": (L["enc"], L["roout"], L["lab"], np.array(env)), "release": (R["gt"], R["pred"], L["lab"][sel], np.array(env)[sel])}


TRD, VAD = load("train"), load("val")
out = {}
for model in ("ar", "release"):
    (GT, PR, LT, ET), (GV, PV, LV, EV) = TRD[model], VAD[model]; out[model] = {}
    for fam, col, K in (("shape", 1, 6), ("color", 2, 8)):
        out[model][fam] = {}
        for k in range(1, 9):
            b = 7 + k; mt = LT[:, b, 0] > 0; mv = LV[:, b, 0] > 0
            yt, yv, clip = LT[mt, b, col].astype(int), LV[mv, b, col].astype(int), np.nonzero(mt)[0]
            Ntr, Nva = design(LT[mt], ET[mt], b), design(LV[mv], EV[mv], b)
            sp = {"enc": (GT[mt, b].astype(np.float64), GV[mv, b].astype(np.float64)), "pred": (PR[mt, k - 1].astype(np.float64), PV[mv, k - 1].astype(np.float64))}
            row = {"n_val": int(mv.sum())}
            for ver in ("raw", "resid"):
                S = {}
                for nm, (Xt, Xv) in sp.items():
                    if ver == "resid":
                        Xt, Xv, row[f"nuisance_r2_{nm}"] = resid(Xt, Ntr, Xv, Nva)
                    S[nm] = (Xt, Xv, Xt.mean(0), Xt.std(0) + 1e-6)
                acc = lambda Sc: round(float((Sc.argmax(1) == yv).mean() * 100), 1)
                (Et, Ev, me, se), (Pt, Pv, mp, sp_) = S["enc"], S["pred"]
                Ce, Cp = concepts(Et, yt, K), concepts(Pt, yt, K)
                We, be = fit((Et - me) / se, np.eye(K)[yt], clip); Wp, bp = fit((Pt - mp) / sp_, np.eye(K)[yt], clip)
                row[ver] = {"eta2_enc": round(eta2(Et, yt, K), 3), "eta2_pred": round(eta2(Pt, yt, K), 3),
                            "centroid_enc_on_enc": acc((Ev - me) @ Ce.T), "centroid_enc_on_pred": acc((Pv - mp) @ Ce.T), "centroid_pred_on_pred": acc((Pv - mp) @ Cp.T),
                            "ridge_enc_on_enc": acc((Ev - me) / se @ We + be), "ridge_enc_on_pred": acc((Pv - me) / se @ We + be),
                            "ridge_enc_on_pred_restd": acc((Pv - mp) / sp_ @ We + be), "ridge_pred_on_pred": acc((Pv - mp) / sp_ @ Wp + bp)}
            out[model][fam][k] = row
        print(model, fam, flush=True)
        for key in ("nuisance_r2_enc", "nuisance_r2_pred"):
            print(f"   {key:26s}", [round(out[model][fam][k][key], 3) for k in range(1, 9)])
        for ver in ("raw", "resid"):
            for key in out[model][fam][1][ver]:
                print(f"   {ver:5s} {key:26s}", [out[model][fam][k][ver][key] for k in range(1, 9)])
json.dump(out, open("/workspace/world_model/vjepa2/z_analysis/results/identity_readout_direction.json", "w"), indent=1)
