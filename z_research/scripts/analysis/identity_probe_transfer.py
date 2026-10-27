"""encoder 임베딩으로 학습한 모양 · 색 판독기를 predictor 출력에 적용 — 블록 전체 평균 (1, 1280) (2026-10-26, 학습 없음, CPU).
사용자: "shape, color 읽는 head 를 encoder embedding 으로 학습하고 predictor embedding 에 test 하면 성능 안 나와?"
블록 b (8–15, 예측 걸음 k = b−7) 마다 train 정답 (1, 1280) 으로 판독기를 맞추고 같은 블록의 val 에 적용한다.
  ridge 판독기 (one-hot, λ 는 train 클립 80/20): encoder → encoder (상한) · encoder → 예측 · encoder → 예측 재표준화
    (예측 특징을 train 예측의 평균 · 표준편차로 표준화한 뒤 encoder 표준화 좌표로 넣음 — 평균 이동 · 크기 차이 제거) · 예측 → 예측
  중심 판독기 (클래스 평균 − 클래스 평균들의 평균 = 개념 벡터, (z − 자기 평균) 과 단위 개념 벡터의 내적 최대): encoder 개념 → 예측 · 예측 개념 → 예측
모델: AR = 굴린 예측 출력 vs 블록 단위 LN(target_encoder) (ar_layers_pooled) / 릴리즈 = 한 번에 8 블록 예측 vs 32 장 LN(target_encoder) (release_pooled).
  /venv/vjepa/bin/python z_research/scripts/analysis/identity_probe_transfer.py   → z_analysis/results/identity_probe_transfer.json
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
def load(model, split):
    L = dict(np.load(f"{P}/ar_layers_pooled/{split}.npz"))
    if model == "ar":
        return L["enc"], L["roout"], L["lab"]
    R = dict(np.load(f"{P}/release_pooled/{split}.npz")); o = {int(v): n for n, v in enumerate(L["i"])}
    return R["gt"], R["pred"], L["lab"][[o[int(v)] for v in R["i"]]]
out = {}
for model in ("ar", "release"):
    (GT, PR, LT), (GV, PV, LV) = load(model, "train"), load(model, "val"); out[model] = {}
    for fam, col, K in (("shape", 1, 6), ("color", 2, 8)):
        out[model][fam] = {}
        for k in range(1, 9):
            b = 7 + k; mt = LT[:, b, 0] > 0; mv = LV[:, b, 0] > 0
            Xe, Xp = GT[mt, b].astype(np.float64), PR[mt, k - 1].astype(np.float64)
            yt, yv, clip = LT[mt, b, col].astype(int), LV[mv, b, col].astype(int), np.nonzero(mt)[0]
            me, se = Xe.mean(0), Xe.std(0) + 1e-6; mp, sp = Xp.mean(0), Xp.std(0) + 1e-6
            We, be = fit((Xe - me) / se, np.eye(K)[yt], clip); Wp, bp = fit((Xp - mp) / sp, np.eye(K)[yt], clip)
            Ce, Cp = concepts(Xe, yt, K), concepts(Xp, yt, K)
            Ve, Vp = GV[mv, b].astype(np.float64), PV[mv, k - 1].astype(np.float64)
            acc = lambda S: round(float((S.argmax(1) == yv).mean() * 100), 1)
            out[model][fam][k] = {"n_val": int(mv.sum()),
                "ridge_enc_on_enc": acc((Ve - me) / se @ We + be), "ridge_enc_on_pred": acc((Vp - me) / se @ We + be),
                "ridge_enc_on_pred_restd": acc((Vp - mp) / sp @ We + be), "ridge_pred_on_pred": acc((Vp - mp) / sp @ Wp + bp),
                "centroid_enc_on_enc": acc((Ve - me) @ Ce.T), "centroid_enc_on_pred": acc((Vp - mp) @ Ce.T), "centroid_pred_on_pred": acc((Vp - mp) @ Cp.T)}
json.dump(out, open("/workspace/world_model/vjepa2/z_analysis/results/identity_probe_transfer.json", "w"), indent=1)
for model in out:
    for fam in out[model]:
        print(model, fam)
        for key in out[model][fam][1]:
            if key != "n_val":
                print(f"   {key:24s}", [out[model][fam][k][key] for k in range(1, 9)])
