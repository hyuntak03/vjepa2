"""encoder 와 예측 양쪽에서 함께 읽히는 정체 방향이 있나 — 두 공간을 합쳐 판독기 하나를 맞춘다 (2026-10-26, 학습 없음, CPU).
사용자: "정체를 읽는 방향을 찾으려면 어떻게 해야 되는 거지". identity_probe_transfer.py · identity_readout_direction.py 의 후속.
블록 b = 7+k 마다 train 의 encoder 표본과 예측 표본 (같은 클립 · 같은 라벨) 을 이어 붙여 릿지 판독기 하나 (λ 는 클립 80/20) 를 맞추고
val 의 encoder · 예측에 각각 적용. 표준화: 공통 (합친 표본의 평균 · 표준편차) / 공간별 (각 공간이 자기 평균 · 표준편차로).
비교 기준: 각 공간 전용 판독기 (identity_probe_transfer.json 의 encoder → encoder, 예측 → 예측).
같이 본다: 모양 × 장면 종류 분할표 (모양이 장면에 묶여 있으면 판독기가 장면을 읽을 수 있다).
  /venv/vjepa/bin/python z_research/scripts/analysis/identity_joint_readout.py   → z_analysis/results/identity_joint_readout.json
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


def load(split):
    L = dict(np.load(f"{P}/ar_layers_pooled/{split}.npz"))
    R = dict(np.load(f"{P}/release_pooled/{split}.npz")); o = {int(v): n for n, v in enumerate(L["i"])}; sel = [o[int(v)] for v in R["i"]]
    return {"ar": (L["enc"], L["roout"], L["lab"]), "release": (R["gt"], R["pred"], L["lab"][sel])}


TRD, VAD = load("train"), load("val")
lab = TRD["ar"][2]
ct = np.zeros((13, 6), int)
for b in range(8, 16):
    m = lab[:, b, 0] > 0; np.add.at(ct, (lab[m, b, 3].astype(int), lab[m, b, 1].astype(int)), 1)
names = list(np.load(f"{P}/ar_layers_pooled/train.npz")["scen_names"])
out = {"shape_by_scenario": {n: ct[i].tolist() for i, n in enumerate(names)}}
print("모양 (sphere cube cylinder cone pyramid torus) × 장면, 블록 8–15 물체 보이는 표본 합")
for i, n in enumerate(names):
    print(f"   {n:16s}", ct[i].tolist())
for model in ("ar", "release"):
    (GT, PR, LT), (GV, PV, LV) = TRD[model], VAD[model]; out[model] = {}
    for fam, col, K in (("shape", 1, 6), ("color", 2, 8)):
        out[model][fam] = {}
        for k in range(1, 9):
            b = 7 + k; mt = LT[:, b, 0] > 0; mv = LV[:, b, 0] > 0
            yt, yv, clip = LT[mt, b, col].astype(int), LV[mv, b, col].astype(int), np.nonzero(mt)[0]
            Et, Pt = GT[mt, b].astype(np.float64), PR[mt, k - 1].astype(np.float64)
            Ev, Pv = GV[mv, b].astype(np.float64), PV[mv, k - 1].astype(np.float64)
            acc = lambda S: round(float((S.argmax(1) == yv).mean() * 100), 1)
            Y2, c2 = np.eye(K)[np.concatenate([yt, yt])], np.concatenate([clip, clip])
            X = np.concatenate([Et, Pt]); m, s = X.mean(0), X.std(0) + 1e-6
            W, b0 = fit((X - m) / s, Y2, c2)
            me, se, mp, sp = Et.mean(0), Et.std(0) + 1e-6, Pt.mean(0), Pt.std(0) + 1e-6
            W2, b2 = fit(np.concatenate([(Et - me) / se, (Pt - mp) / sp]), Y2, c2)
            out[model][fam][k] = {"joint_common_on_enc": acc((Ev - m) / s @ W + b0), "joint_common_on_pred": acc((Pv - m) / s @ W + b0),
                                  "joint_perspace_on_enc": acc((Ev - me) / se @ W2 + b2), "joint_perspace_on_pred": acc((Pv - mp) / sp @ W2 + b2)}
        print(model, fam, flush=True)
        for key in out[model][fam][1]:
            print(f"   {key:24s}", [out[model][fam][k][key] for k in range(1, 9)])
json.dump(out, open("/workspace/world_model/vjepa2/z_analysis/results/identity_joint_readout.json", "w"), indent=1)
