"""encoder 1280 화면 전체 평균의 모양 · 색 선형 판독 — 학습 표본 수를 늘리면 오르나 (학습 곡선, 2026-10-26, CPU).
사용자: "물체가 너무 작아서 그래? 전체 평균해서 shape, color 따로 읽으면 높게 나오나? 샘플 개수 더 늘려 봐".
표본: ar_layers_pooled train 의 앞 n 편 (n = 500 · 1,000 · 2,000 · 4,000 · 8,000) × 블록 0–15 중 물체 보이는 블록 (보이는 비율 ≥ 0.5 · 화면 안 ≥ 0.9). 시험 = val 350 편 같은 기준 (2,667 표본, 고정).
판독기: 선형층 하나 (모양 6 · 색 8 따로), 입력 표준화 (학습 표본 통계), AdamW lr 1e-3 · wd 1e-2 · 묶음 512, 최대 epoch = min(300, 60 × max(1, 4000 / n)), train 클립 20 % 로 epoch 선택.
  /venv/vjepa/bin/python z_research/scripts/analysis/identity_readout_learning_curve.py   → z_analysis/results/identity_readout_learning_curve.json
"""
import copy, json
import numpy as np, torch, torch.nn.functional as F
torch.set_num_threads(48)
P = "/workspace/auto_research_cache/ar_layers_pooled"
TR, VA = np.load(f"{P}/train.npz"), np.load(f"{P}/val.npz")


def flat(Z, n=None):
    E, L = Z["enc"][:n], Z["lab"][:n]; m = L[..., 0] > 0
    clip = np.repeat(np.arange(len(E)), 16).reshape(len(E), 16)
    return E[m].astype(np.float32), L[m][:, 1].astype(np.int64), L[m][:, 2].astype(np.int64), clip[m]


def fit(X, y, clip, K, epochs):
    torch.manual_seed(0); X, y = torch.from_numpy(X), torch.from_numpy(y); h = torch.from_numpy(clip % 5 == 0)
    head = torch.nn.Linear(X.shape[1], K); opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-2); best = (-1, None, -1)
    Xa, ya, Xh, yh = X[~h], y[~h], X[h], y[h]
    for ep in range(epochs):
        perm = torch.randperm(len(Xa))
        for i in range(0, len(Xa), 512):
            idx = perm[i:i + 512]; loss = F.cross_entropy(head(Xa[idx]), ya[idx]); opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            acc = (head(Xh).argmax(1) == yh).float().mean().item()
        if acc > best[0]:
            best = (acc, copy.deepcopy(head.state_dict()), ep)
    head.load_state_dict(best[1]); return head, best[2]


Xv, sv, cv, _ = flat(VA)
res = {"n_val": int(len(sv)), "curve": []}
for n in (500, 1000, 2000, 4000, 8000):
    Xt, st, ct, clt = flat(TR, n); mu, sd = Xt.mean(0), Xt.std(0) + 1e-6; Xtn, Xvn = (Xt - mu) / sd, (Xv - mu) / sd
    ep_max = int(min(300, 60 * max(1, 4000 / n))); row = {"n_clips": n, "n_train": int(len(st))}
    for task, yt, yv, K in (("shape", st, sv, 6), ("color", ct, cv, 8)):
        h, ep = fit(Xtn, yt, clt, K, ep_max)
        with torch.no_grad():
            row[task] = round(float((h(torch.from_numpy(Xvn)).argmax(1).numpy() == yv).mean() * 100), 1)
        row[f"epoch_{task}"] = f"{ep}/{ep_max}"
    res["curve"].append(row); print(row, flush=True)
json.dump(res, open("/workspace/world_model/vjepa2/z_analysis/results/identity_readout_learning_curve.json", "w"), indent=1)
