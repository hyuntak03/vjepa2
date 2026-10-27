"""개념 벡터 재검토 — 정체 (모양 · 색) 신호가 물체 칸에 있나, 화면 전체로 퍼져 있나, 화면 평균 개념 벡터는 어느 구역의 것인가 (2026-10-26, CPU).
사용자: "concept vector 이거 의미 있는 건지 다시 검토해 줘". 계기: 배경만 평균해도 모양 73 % · 색 78 % 로 읽힌다 (identity_readout_object_cells.json).
입력: attnprobe_extract.py 의 encoder 구역 평균 (표본마다 (5, 1280): 물체 칸 · 1 칸 옆 · 2 칸 옆 · 3 칸 이상 · 화면 전체; 칸 거리 = 물체 칸까지 체스판 거리), train 4,000 편 · val 350 편, 블록 0–15 중 물체 보이는 블록.
보는 것:
  (a) 구역별 선형 판독 (모양 6 · 색 8 따로, 선형층, 입력 표준화, AdamW lr 1e-3 · wd 1e-2 · 묶음 512 · 최대 60 epoch, train 클립 20 % 로 epoch 선택) → val 정확도
  (b) 구역별 개념 벡터 (클래스 평균 − 전체 평균, train 전부): 반쪽 재현도 (클립 짝 / 홀) · 장면 일반성 (장면 묶음 A / B) · 라벨 섞은 영 기준
  (c) 화면 전체 개념 벡터와 각 구역 개념 벡터의 cos (클래스 평균) — 화면 평균 개념이 어디서 오나
  (d) 구역별 개념 크기: 개념 부분공간 위 분산 중 클래스 몫 (η²)
  /venv/vjepa/bin/python z_research/scripts/analysis/concept_vector_regions.py   → z_analysis/results/concept_vector_regions.json
"""
import copy, json
import numpy as np, torch, torch.nn.functional as F
torch.set_num_threads(32)
D = "/workspace/auto_research_cache/attnprobe"; M = "/workspace/auto_research_cache/ar_layers"
REG = ["object", "ring1", "ring2", "far3plus", "whole"]
SA = {"container", "drop_bounce", "moving_screen", "occ_pass", "projectile", "shelf_land", "wall_bounce"}
Rt, Lt = np.load(f"{D}/train_enc_regions.npy").astype(np.float32), np.load(f"{D}/train_enc_lab.npy")
Rv, Lv = np.load(f"{D}/val_enc_regions.npy").astype(np.float32), np.load(f"{D}/val_enc_lab.npy")
scen = {m["i"]: m["scen"] for m in json.load(open(f"{M}/train_meta.json"))}
sceneA = np.array([scen[int(i)] in SA for i in Lt[:, 6]])
rng = np.random.default_rng(0)


def fit_eval(Xt, yt, clip, Xv, yv, K):
    mu, sd = Xt.mean(0), Xt.std(0) + 1e-6; Xt, Xv = torch.from_numpy((Xt - mu) / sd), torch.from_numpy((Xv - mu) / sd)
    torch.manual_seed(0); y = torch.from_numpy(yt); h = torch.from_numpy(clip % 5 == 0)
    head = torch.nn.Linear(1280, K); opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-2); best = (-1, None)
    for ep in range(60):
        perm = torch.nonzero(~h)[:, 0][torch.randperm(int((~h).sum()))]
        for i in range(0, len(perm), 512):
            idx = perm[i:i + 512]; loss = F.cross_entropy(head(Xt[idx]), y[idx]); opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            acc = (head(Xt[h]).argmax(1) == y[h]).float().mean().item()
        if acc > best[0]:
            best = (acc, copy.deepcopy(head.state_dict()))
    head.load_state_dict(best[1])
    with torch.no_grad():
        return round(float((head(Xv).argmax(1).numpy() == yv).mean() * 100), 1)


cv = lambda X, y, K: np.stack([X[y == c].mean(0) for c in range(K)]) - X.mean(0)
cosm = lambda A, B: float(np.mean((A * B).sum(1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1))))


def eta2(X, y, K):
    Xc = X - X.mean(0); C = np.stack([Xc[y == c].mean(0) for c in range(K)]); U = np.linalg.svd(C - C.mean(0), full_matrices=False)[2][:K - 1]
    Z = Xc @ U.T; Zm = np.stack([Z[y == c].mean(0) for c in range(K)])[y]
    return float((Zm ** 2).sum() / (Z ** 2).sum())


res = {"n_train": int(len(Lt)), "n_val": int(len(Lv)), "regions": REG}
for fam, col, K in (("shape", 0, 6), ("color", 1, 8)):
    yt, yv = Lt[:, col].astype(np.int64), Lv[:, col].astype(np.int64); clip = Lt[:, 6].astype(int); r = {}
    Cw = cv(Rt[:, 4], yt, K)
    for j, nm in enumerate(REG):
        okt, okv = ~np.isnan(Rt[:, j]).any(1), ~np.isnan(Rv[:, j]).any(1)
        X, y, cl = Rt[okt, j], yt[okt], clip[okt]; half = cl % 2 == 0; sA = sceneA[okt]; yp = rng.permutation(y)
        C = cv(X, y, K)
        r[nm] = {"n_train": int(okt.sum()), "probe_val_acc": fit_eval(X, y, cl, Rv[okv, j], yv[okv], K),
                 "reliability": round(cosm(cv(X[half], y[half], K), cv(X[~half], y[~half], K)), 3),
                 "reliability_null": round(cosm(cv(X[half], yp[half], K), cv(X[~half], yp[~half], K)), 3),
                 "scene_general": round(cosm(cv(X[sA], y[sA], K), cv(X[~sA], y[~sA], K)), 3),
                 "scene_general_null": round(cosm(cv(X[sA], yp[sA], K), cv(X[~sA], yp[~sA], K)), 3),
                 "cos_with_whole_frame_concept": round(cosm(C, Cw[:len(C)]) if okt.all() else cosm(C, cv(Rt[okt, 4], y, K)), 3),
                 "cos_with_object_concept": round(cosm(C, cv(Rt[okt, 0], y, K)), 3), "eta2": round(eta2(X, y, K), 3)}
        print(fam, nm, r[nm], flush=True)
    res[fam] = r
json.dump(res, open("/workspace/world_model/vjepa2/z_analysis/results/concept_vector_regions.json", "w"), indent=1)
