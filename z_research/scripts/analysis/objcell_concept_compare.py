"""물체 칸 기준 정체 (모양 · 색) 개념 벡터 · 판독기 — encoder vs 예측 (AR 굴린 출력 · 릴리즈), 예측 걸음별 (2026-10-26, CPU).
사용자: "물체 칸 concept vector 로 지금 결과들 다 바꿔줘. predictor 는 그게 안 될 것 같은데 일단 해봐".
계기: 화면 평균 개념 벡터는 물체가 아니라 화면 전체에 퍼진 정체 신호였다 (concept_vector_regions.json: 물체 칸 개념과 cos 0.31 / 0.47).
입력: objcell_concept_cache.py (train 8,000 · val 350 편, 블록 8–15). 라벨 = ar_layers meta (모양 · 색 · 블록별 보임).
정답 (encoder) 개념 = 블록 b 의 진짜 물체 칸 평균 (AR: 블록 단위 encoder / 릴리즈: 32 장 encoder). 예측 개념 두 자리:
  진짜 자리 = 블록 b 의 진짜 물체 칸에서 예측 토큰 평균 · 마지막에 본 자리 = 블록 7 의 물체 칸에서 예측 토큰 평균 (블록 7 · b 모두 물체 보일 때)
(1) 개념 벡터 (클래스 평균 − 전체 평균, train 전부): 정답 vs 예측 직접 cos · 크기 비 (클래스마다 → 평균 ± 표준편차) · 반쪽 재현도 (클립 짝 / 홀) · 라벨 섞은 영 기준
(2) 판독기 (선형층, 모양 6 · 색 8 따로, 입력 표준화, AdamW lr 1e-3 · wd 1e-2 · 묶음 512 · 최대 60 epoch, train 클립 20 % 로 epoch 선택, 블록 8–15 합쳐 학습):
    정답으로 학습 → 정답 val (상한) · 예측 진짜 자리 val · 예측 마지막 자리 val / 예측 진짜 자리로 학습 → 같은 것 val (정보가 있나), 걸음별
  /venv/vjepa/bin/python z_research/scripts/analysis/objcell_concept_compare.py [ar,release]   → z_analysis/results/objcell_concept_compare.json (모델별로 덧붙인다)
"""
import copy, json
import numpy as np, torch, torch.nn.functional as F
torch.set_num_threads(48)
D = "/workspace/auto_research_cache/objcell_concept"; M = "/workspace/auto_research_cache/ar_layers"
rng = np.random.default_rng(0)
cv = lambda X, y, K: np.stack([X[y == c].mean(0) for c in range(K)]) - X.mean(0)
pcos = lambda A, B: (A * B).sum(1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1))


def labels(split, I):
    meta = {m["i"]: m for m in json.load(open(f"{M}/{split}_meta.json"))}
    shape = np.array([meta[int(i)]["shape"] for i in I]); color = np.array([meta[int(i)]["color"] for i in I])
    ok = np.array([[meta[int(i)]["blocks"][b]["ok"] for b in range(16)] for i in I])
    return shape, color, ok


def fit(X, y, clip, K):
    torch.manual_seed(0); X, y = torch.from_numpy(X), torch.from_numpy(y); h = torch.from_numpy(clip % 5 == 0)
    head = torch.nn.Linear(X.shape[1], K); opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-2); best = (-1, None)
    Xa, ya, Xh, yh = X[~h], y[~h], X[h], y[h]
    for ep in range(60):
        perm = torch.randperm(len(Xa))
        for i in range(0, len(Xa), 512):
            idx = perm[i:i + 512]; loss = F.cross_entropy(head(Xa[idx]), ya[idx]); opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            acc = (head(Xh).argmax(1) == yh).float().mean().item()
        if acc > best[0]:
            best = (acc, copy.deepcopy(head.state_dict()))
    head.load_state_dict(best[1]); return head


import sys
MODELS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["ar", "release"]   # 릴리즈 캐시가 늦게 끝나 AR 먼저 돌릴 수 있게
OUTJ = "/workspace/world_model/vjepa2/z_analysis/results/objcell_concept_compare.json"
try:
    res = json.load(open(OUTJ))
except FileNotFoundError:
    res = {}
for model in MODELS:
    Z, V = np.load(f"{D}/{model}_train.npz"), np.load(f"{D}/{model}_val.npz")
    sh, co, ok = labels("train", Z["i"]); shv, cov, okv = labels("val", V["i"]); res[model] = {}
    T, PT, PL = Z["tgt_true"].astype(np.float32), Z["pred_true"].astype(np.float32), Z["pred_last"].astype(np.float32)
    Tv, PTv, PLv = V["tgt_true"].astype(np.float32), V["pred_true"].astype(np.float32), V["pred_last"].astype(np.float32)
    for fam, y_all, yv_all, K in (("shape", sh, shv, 6), ("color", co, cov, 8)):
        r = {"concept": {}, "readout": {}}
        # (1) 개념 벡터, 걸음별
        for k in range(1, 9):
            j = k - 1; b = 7 + k; mb = ok[:, b]; ml = mb & ok[:, 7]; clip = np.arange(len(y_all)); half = clip % 2 == 0
            row = {"n_true": int(mb.sum()), "n_last": int(ml.sum())}
            E = cv(T[mb, j], y_all[mb], K)
            for nm, X, m in (("pred_true", PT, mb), ("pred_last", PL, ml)):
                P = cv(X[m, j], y_all[m], K); c = pcos(E if nm == "pred_true" else cv(T[m, j], y_all[m], K), P)
                Eref = E if nm == "pred_true" else cv(T[m, j], y_all[m], K)
                nr = np.linalg.norm(P, axis=1) / np.linalg.norm(Eref, axis=1); yp = rng.permutation(y_all[m])
                row[nm] = {"cos": round(float(c.mean()), 3), "cos_sd": round(float(c.std()), 3), "norm_ratio": round(float(nr.mean()), 3), "norm_ratio_sd": round(float(nr.std()), 3),
                           "cos_null": round(float(pcos(cv(T[m, j], yp, K), cv(X[m, j], yp, K)).mean()), 3),
                           "reliability_pred": round(float(pcos(cv(X[m & half, j], y_all[m & half], K), cv(X[m & ~half, j], y_all[m & ~half], K)).mean()), 3)}
            row["reliability_target"] = round(float(pcos(cv(T[mb & half, j], y_all[mb & half], K), cv(T[mb & ~half, j], y_all[mb & ~half], K)).mean()), 3)
            r["concept"][k] = row
        # (2) 판독기: 블록 8–15 합쳐 학습, 걸음별 시험
        def stack(X, okm, y, last=False):
            xs, ys, cs, ks = [], [], [], []
            for k in range(1, 9):
                m = okm[:, 7 + k] & (okm[:, 7] if last else True)
                xs.append(X[m, k - 1]); ys.append(y[m]); cs.append(np.nonzero(m)[0]); ks.append(np.full(m.sum(), k))
            return np.concatenate(xs), np.concatenate(ys).astype(np.int64), np.concatenate(cs), np.concatenate(ks)
        Xt, yt, ct, _ = stack(T, ok, y_all); mu, sd = Xt.mean(0), Xt.std(0) + 1e-6; nz = lambda X: ((X - mu) / sd).astype(np.float32)
        h_enc = fit(nz(Xt), yt, ct, K)
        Xp, yp_, cp, _ = stack(PT, ok, y_all); mu2, sd2 = Xp.mean(0), Xp.std(0) + 1e-6; nz2 = lambda X: ((X - mu2) / sd2).astype(np.float32)
        h_pred = fit(nz2(Xp), yp_, cp, K)
        tests = {"enc_head_on_target": (h_enc, nz, Tv, False), "enc_head_on_pred_true": (h_enc, nz, PTv, False), "enc_head_on_pred_last": (h_enc, nz, PLv, True),
                 "pred_head_on_pred_true": (h_pred, nz2, PTv, False)}
        for nm, (h, f, X, last) in tests.items():
            Xs, ys, _, ks = stack(X, okv, yv_all, last)
            with torch.no_grad():
                pr = h(torch.from_numpy(f(Xs))).argmax(1).numpy()
            r["readout"][nm] = [round(float((pr[ks == k] == ys[ks == k]).mean() * 100), 1) for k in range(1, 9)]
        r["readout"]["val_majority"] = round(float(np.bincount(yv_all, minlength=K).max() / len(yv_all) * 100), 1)
        res[model][fam] = r
        print(model, fam, "cos true", [r["concept"][k]["pred_true"]["cos"] for k in range(1, 9)], "cos last", [r["concept"][k]["pred_last"]["cos"] for k in range(1, 9)], flush=True)
        print("   norm true", [r["concept"][k]["pred_true"]["norm_ratio"] for k in range(1, 9)], "norm last", [r["concept"][k]["pred_last"]["norm_ratio"] for k in range(1, 9)], flush=True)
        print("   rel tgt", [r["concept"][k]["reliability_target"] for k in range(1, 9)], "rel pred_true", [r["concept"][k]["pred_true"]["reliability_pred"] for k in range(1, 9)], flush=True)
        for nm, v in r["readout"].items():
            print(f"   {nm:24s} {v}", flush=True)
json.dump(res, open(OUTJ, "w"), indent=1)
