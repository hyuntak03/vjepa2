"""encoder 1280 화면 전체 평균의 모양 · 색 선형 판독 — 물체가 확실히 보이는 블록만, 표본 늘려서 (2026-10-26, CPU).
사용자: "화면 전체 평균은 문제가 안 될 것 같은데. 물체가 확실히 보이는 block 만 확인해서, 샘플도 늘리고".
표본: ar_layers_pooled train 8,000 편 · val 350 편의 블록 0–15 전부 (앞판은 블록 8–14 만). 표현 = 블록 단위 encoder 출력의 256 칸 평균 (1, 1280).
보임 기준 (블록의 두 장 모두, 클립 _states.npz 에서):
  ok     = 보이는 비율 ≥ 0.5 · 화면 안 비율 ≥ 0.9 (지금까지 쓴 기준)
  clear  = 보이는 비율 ≥ 0.95 · 화면 안 비율 ≥ 0.99
  clear_large = clear 이고 물체 상자 넓이 ≥ 중앙값 (clear 표본 기준)
판독기: 선형층 하나 (모양 6 · 색 8 따로), 입력 표준화, AdamW lr 1e-3 · wd 1e-2 · 묶음 512 · 최대 60 epoch, train 클립 20 % 로 epoch 선택.
  (1) ok 로 학습 → val 의 ok · clear · clear_large 에서 정확도  (2) clear 로 학습 → val clear 에서 정확도
  /venv/vjepa/bin/python z_research/scripts/analysis/identity_readout_strict_visible.py   → z_analysis/results/identity_readout_strict_visible.json
"""
import copy, json
import numpy as np, torch, torch.nn.functional as F
torch.set_num_threads(32)
P = "/workspace/auto_research_cache/ar_layers_pooled"
IDX = [3 * k for k in range(32)]


def vis_table(clips):
    V = np.zeros((len(clips), 16, 3), np.float32)                      # 보이는 비율 최소 · 화면 안 비율 최소 · 상자 넓이 (px²)
    for n, c in enumerate(clips):
        st = np.load(str(c)[:-4] + "_states.npz"); vis, inf, bb = st["vis_frac"][:, 0], st["inframe_frac"][:, 0], st["bbox"][:, 0]
        for b in range(16):
            f0, f1 = IDX[2 * b], IDX[2 * b + 1]
            area = np.mean([max(0.0, bb[f, 2] - bb[f, 0]) * max(0.0, bb[f, 3] - bb[f, 1]) for f in (f0, f1)])
            V[n, b] = [min(vis[f0], vis[f1]), min(inf[f0], inf[f1]), area if np.isfinite(area) else 0.0]
    return V


def flat(Z, V):
    X = Z["enc"].reshape(-1, 1280); lab = Z["lab"].reshape(-1, 8); v = V.reshape(-1, 3); clip = np.repeat(np.arange(len(V)), 16)
    return X, lab[:, 1].astype(np.int64), lab[:, 2].astype(np.int64), v, clip


def fit(X, y, clip, K):
    torch.manual_seed(0); X, y = torch.from_numpy(X), torch.from_numpy(y); h = torch.from_numpy(clip % 5 == 0)
    head = torch.nn.Linear(X.shape[1], K); opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-2); best = (-1, None, -1)
    Xa, ya, Xh, yh = X[~h], y[~h], X[h], y[h]
    for ep in range(60):
        perm = torch.randperm(len(Xa))
        for i in range(0, len(Xa), 512):
            idx = perm[i:i + 512]; loss = F.cross_entropy(head(Xa[idx]), ya[idx]); opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            acc = (head(Xh).argmax(1) == yh).float().mean().item()
        if acc > best[0]:
            best = (acc, copy.deepcopy(head.state_dict()), ep)
    head.load_state_dict(best[1]); return head, best[2]


TR, VA = np.load(f"{P}/train.npz"), np.load(f"{P}/val.npz")
VT, VV = vis_table(TR["clip"]), vis_table(VA["clip"])
Xt, st_, ct_, vt, clt = flat(TR, VT); Xv, sv, cv, vv, _ = flat(VA, VV)
def tiers(v, area_thr):
    ok = (v[:, 0] >= 0.5) & (v[:, 1] >= 0.9); clear = (v[:, 0] >= 0.95) & (v[:, 1] >= 0.99)
    return {"ok": ok, "clear": clear, "clear_large": clear & (v[:, 2] >= area_thr)}
thr = float(np.median(vt[(vt[:, 0] >= 0.95) & (vt[:, 1] >= 0.99), 2]))
Tt, Tv = tiers(vt, thr), tiers(vv, thr)
res = {"area_threshold_px2": round(thr, 1), "n_train": {k: int(m.sum()) for k, m in Tt.items()}, "n_val": {k: int(m.sum()) for k, m in Tv.items()}}
print("표본 수 train", res["n_train"], "val", res["n_val"], "상자 넓이 기준", thr, flush=True)
mu, sd = Xt[Tt["ok"]].astype(np.float32).mean(0), Xt[Tt["ok"]].astype(np.float32).std(0) + 1e-6
norm = lambda X: ((X.astype(np.float32) - mu) / sd).astype(np.float32)
for task, yt, yv, K in (("shape", st_, sv, 6), ("color", ct_, cv, 8)):
    r = {}
    for train_tier in ("ok", "clear"):
        m = Tt[train_tier]; h, ep = fit(norm(Xt[m]), yt[m], clt[m], K)
        with torch.no_grad():
            for test_tier in (("ok", "clear", "clear_large") if train_tier == "ok" else ("clear", "clear_large")):
                mv = Tv[test_tier]; r[f"train_{train_tier}__test_{test_tier}"] = round(float((h(torch.from_numpy(norm(Xv[mv]))).argmax(1).numpy() == yv[mv]).mean() * 100), 1)
        r[f"epoch_train_{train_tier}"] = int(ep)
    r["val_majority_clear"] = round(float(np.bincount(yv[Tv["clear"]], minlength=K).max() / Tv["clear"].sum() * 100), 1)
    res[task] = r; print(task, r, flush=True)
json.dump(res, open("/workspace/world_model/vjepa2/z_analysis/results/identity_readout_strict_visible.json", "w"), indent=1)
