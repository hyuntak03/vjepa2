"""고정 선형 판독기 정확도 — 원본 vs 자기 정답 개념 벡터 빼기, 물체 칸 기준 (2026-10-26 저녁, CPU). probe_subtract_bars 그림의 물체 칸 판.
사용자: "물체 칸 concept vector 로 지금 결과들 다 바꿔줘" · "figure 전부 갱신".
⚠️ 같은 공간 · 같은 데이터에서 구한 개념 벡터를 빼면 선형 판독기는 어느 방향을 읽든 우연 쪽으로 간다 (수학적으로 정해진 결과, 계산 확인용).
표현 (objcell_concept_cache.py, train 8,000 · val 350 편): encoder 블록 8–15 의 물체 칸 평균 · AR 굴린 예측 (진짜 물체 칸) 걸음 1–2 · 걸음 6–8.
판독기: 선형층 (모양 6 · 색 8 따로), 입력 표준화, AdamW lr 1e-3 · wd 1e-2 · 묶음 512 · 최대 60 epoch, train 클립 20 % 로 epoch 선택. 개념 벡터 = 같은 표현 train 의 클래스 평균 − 전체 평균 (표준화 좌표).
  /venv/vjepa/bin/python z_research/scripts/analysis/probe_subtract_objcell.py   → z_analysis/results/probe_subtract_objcell.json
"""
import copy, json
import numpy as np, torch, torch.nn.functional as F
torch.set_num_threads(48)
D = "/workspace/auto_research_cache/objcell_concept"; M = "/workspace/auto_research_cache/ar_layers"
Z, V = np.load(f"{D}/ar_train.npz"), np.load(f"{D}/ar_val.npz")
def lab(split, I):
    meta = {m["i"]: m for m in json.load(open(f"{M}/{split}_meta.json"))}
    return np.array([meta[int(i)]["shape"] for i in I]), np.array([meta[int(i)]["color"] for i in I])
sh, co = lab("train", Z["i"]); shv, cov = lab("val", V["i"])
def gather(A, steps, y):
    X = A[:, [k - 1 for k in steps]].astype(np.float32); n, s_ = X.shape[:2]; X = X.reshape(n * s_, -1); yy = np.repeat(y, s_); cl = np.repeat(np.arange(n), s_)
    m = ~np.isnan(X).any(1); return X[m], yy[m].astype(np.int64), cl[m]
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
PANELS = {"encoder": ("tgt_true", list(range(1, 9))), "ar_rollout_s12": ("pred_true", [1, 2]), "ar_rollout_s68": ("pred_true", [6, 7, 8])}
res = {}
for pn, (key, steps) in PANELS.items():
    res[pn] = {}
    for fam, y, yv, K in (("shape", sh, shv, 6), ("color", co, cov, 8)):
        Xt, yt, ct = gather(Z[key], steps, y); Xv, yv_, _ = gather(V[key], steps, yv)
        mu, sd = Xt.mean(0), Xt.std(0) + 1e-6; Xt, Xv = ((Xt - mu) / sd).astype(np.float32), ((Xv - mu) / sd).astype(np.float32)
        C = np.stack([Xt[yt == c].mean(0) for c in range(K)]) - Xt.mean(0)
        h = fit(Xt, yt, ct, K)
        with torch.no_grad():
            o = float((h(torch.from_numpy(Xv)).argmax(1).numpy() == yv_).mean() * 100); s_ = float((h(torch.from_numpy((Xv - C[yv_]).astype(np.float32))).argmax(1).numpy() == yv_).mean() * 100)
        res[pn][fam] = {"original": round(o, 1), "subtract_a1": round(s_, 1), "n_val": int(len(yv_)), "val_majority": round(float(np.bincount(yv_, minlength=K).max() / len(yv_) * 100), 1)}
        print(pn, fam, res[pn][fam], flush=True)
json.dump(res, open("/workspace/world_model/vjepa2/z_analysis/results/probe_subtract_objcell.json", "w"), indent=1)
