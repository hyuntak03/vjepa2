"""물체 벡터 부분공간 (정체 · 속도 · 위치) 이 유의미한가 — 검증 (2026-10-25, 학습 없음).
사용자: "이 속도 위치 물체 벡터는 어떻게 뽑은거야? 이게 정말 유의미한 벡터 맞아?"
pred_object_direction.py 의 부분공간은 시험 성능 · 재현도 · 귀무 없이 썼다 → 여기서 잰다.

특징 = 블록의 물체 칸 정답 토큰 평균 (1280). train = pred_objvec train 캐시 (400 편), 시험 = val 캐시 미래 블록 (350 편, 블록 8..15).
라벨: 모양 6 · 색 8 · 위치 (블록 중심 px) · 속도_사이 (블록 중심 − 앞 블록 중심) · 속도_안 (블록 두 장 사이 변위, states.npz 에서).
  1 시험 성능: ridge (λ 는 train 5 겹 CV) → val 정확도 / R². 귀무 = train 라벨을 편 단위로 섞어 같은 절차.
  2 재현도: train 편을 반으로 나눠 두 probe 의 활성 패턴 부분공간 겹침 (‖U₁ᵀU₂‖²_F / r).
  3 에너지 몫: val 참 자리 칸의 encoder 물체 벡터 e_Y = Y − bg 의 에너지 중 각 부분공간 몫 vs 같은 차원 무작위 부분공간 (e_Y 공분산을 따름) 의 95 % 분위.
"""
import json, os
import numpy as np, torch

CACHE = "/workspace/auto_research_cache/pred_objvec"; OUT = "/workspace/world_model/vjepa2/auto_research/results/validate_object_subspaces.json"
IDX = [3 * k for k in range(32)]; NBC = 8
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tmeta = json.load(open(f"{CACHE}/train_meta.json")); vmeta = json.load(open(f"{CACHE}/val_meta.json"))
rng = np.random.default_rng(0)


def within(clip, b):
    st = np.load(clip[:-4] + "_states.npz"); p = st["pos"][:, 0]
    return p[IDX[2 * b + 1]] - p[IDX[2 * b]]


def ridge_cv(X, Y, lams=(1e0, 1e1, 1e2, 1e3, 1e4), groups=None):
    fold = (groups % 5) if groups is not None else np.arange(len(X)) % 5; best = None
    for lam in lams:
        e = 0.0
        for f in range(5):
            tr = fold != f; u, s, vt = np.linalg.svd(X[tr], full_matrices=False)
            W = vt.T @ ((s / (s ** 2 + lam))[:, None] * (u.T @ Y[tr])); e += ((X[~tr] @ W - Y[~tr]) ** 2).sum()
        if best is None or e < best[0]:
            best = (e, lam)
    u, s, vt = np.linalg.svd(X, full_matrices=False)
    return vt.T @ ((s / (s ** 2 + best[1]))[:, None] * (u.T @ Y)), best[1]


def collect_train(sp):
    X, L = [], []
    for i, m in enumerate(tmeta):
        pooled = np.load(f"{CACHE}/train/{i:05d}.npz")[f"objpool_{sp}"]; lab = m["labels"]; r = 0
        for b in range(16):
            if lab[b]["ok"] and lab[b]["cells"]:
                if b >= 1 and lab[b - 1]["ok"] and np.isfinite(lab[b - 1]["ctr"]).all():
                    X.append(pooled[r]); L.append((i, m["shape"], m["color"], *lab[b]["ctr"], *(np.array(lab[b]["ctr"]) - np.array(lab[b - 1]["ctr"])), *within(m["clip"], b)))
                r += 1
    return np.array(X, np.float32), np.array(L, np.float64)


def collect_val(sp):
    X, L, EY = [], [], []
    for i, m in enumerate(vmeta):
        z = np.load(f"{CACHE}/val/{i:05d}.npz"); Y = z[f"Y_{sp}"].astype(np.float32); bg = z[f"bg_{sp}"].astype(np.float32); lab = m["labels"]
        for j in range(8):
            b = NBC + j; Lb = lab[b]
            if not (Lb["ok"] and Lb["cells"] and lab[b - 1]["ok"] and np.isfinite(lab[b - 1]["ctr"]).all()):
                continue
            X.append(Y[j, Lb["cells"]].mean(0))
            L.append((i, m["shape"], m["color"], *Lb["ctr"], *(np.array(Lb["ctr"]) - np.array(lab[b - 1]["ctr"])), *within(m["clip"], b)))
            if not np.isnan(bg[Lb["cells"]]).any():
                EY.append(Y[j, Lb["cells"]] - bg[Lb["cells"]])
    return np.array(X, np.float32), np.array(L, np.float64), np.concatenate(EY)


TARGETS = {"shape": ("cls", 1, 6), "color": ("cls", 2, 8), "position": ("reg", (3, 5), 2), "vel_between": ("reg", (5, 7), 2), "vel_within": ("reg", (7, 9), 2)}
out = {}
for sp in ("ar", "rel"):
    Xt, Lt = collect_train(sp); Xv, Lv, EY = collect_val(sp)
    mu, sd = Xt.mean(0), Xt.std(0) + 1e-6; Xts, Xvs = (Xt - mu) / sd, (Xv - mu) / sd; Sig = np.cov(Xts, rowvar=False)
    groups = Lt[:, 0].astype(int)
    res = {"n_train": len(Xt), "n_val": len(Xv)}; pats = {}
    for t, (kind, col, k) in TARGETS.items():
        Y = np.eye(k)[Lt[:, col].astype(int)] if kind == "cls" else Lt[:, col[0]:col[1]]
        Yv = np.eye(k)[Lv[:, col].astype(int)] if kind == "cls" else Lv[:, col[0]:col[1]]
        def score(Ytr):
            ym = Ytr.mean(0); W, lam = ridge_cv(Xts, Ytr - ym, groups=groups); yh = Xvs @ W + ym
            s = float((yh.argmax(1) == Yv.argmax(1)).mean() * 100) if kind == "cls" else float(1 - ((yh - Yv) ** 2).sum() / ((Yv - Yv.mean(0)) ** 2).sum())
            return s, W, lam
        s, W, lam = score(Y)
        # 귀무: 편 단위로 라벨을 섞는다 (같은 편의 블록은 같은 라벨 묶음을 유지)
        u = np.unique(groups); perm = dict(zip(u, rng.permutation(u)))
        first = {g: np.nonzero(groups == g)[0] for g in u}
        Ysh = np.concatenate([Y[first[perm[g]]][: len(first[g])] if len(first[perm[g]]) >= len(first[g]) else np.resize(Y[first[perm[g]]], (len(first[g]), Y.shape[1])) for g in u])
        order = np.concatenate([first[g] for g in u]); Ynull = np.empty_like(Y); Ynull[order] = Ysh
        s0, _, _ = score(Ynull)
        r = k - 1 if kind == "cls" else 2
        # 재현도
        half = np.isin(groups, u[: len(u) // 2]); subs = []
        for hm in (half, ~half):
            ym = Y[hm].mean(0); Wh, _ = ridge_cv(Xts[hm], Y[hm] - ym, groups=groups[hm])
            U_, _, _ = np.linalg.svd((np.cov(Xts[hm], rowvar=False) @ Wh) / sd[:, None], full_matrices=False); subs.append(U_[:, :r])
        rel = float(np.linalg.norm(subs[0].T @ subs[1]) ** 2 / r)
        U_, _, _ = np.linalg.svd((Sig @ W) / sd[:, None], full_matrices=False); pats[t] = U_[:, :r]
        res[t] = {"val": round(s, 3), "null": round(s0, 3), "chance": round(100 / k, 1) if kind == "cls" else 0.0, "lambda": lam, "split_half": round(rel, 3), "dims": r}
    # 에너지 몫 vs 무작위
    E = torch.from_numpy(EY).to(dev); tot = (E ** 2).sum(1)
    ev, evec = torch.linalg.eigh(torch.cov(E.T)); sq = evec * ev.clamp_min(0).sqrt()
    for t, U_ in pats.items():
        U = torch.from_numpy(U_).float().to(dev); share = float(((E @ U) ** 2).sum(1).div(tot).mean())
        nulls = []
        for _ in range(100):
            Q, _ = torch.linalg.qr(sq @ torch.randn(1280, U.shape[1], device=dev)); nulls.append(float(((E @ Q) ** 2).sum(1).div(tot).mean()))
        res[t]["energy_share"] = round(share, 4); res[t]["energy_null95"] = round(float(np.percentile(nulls, 95)), 4); res[t]["energy_iso"] = round(U.shape[1] / 1280, 4)
    out[sp] = res
    print(sp, json.dumps(res, ensure_ascii=False), flush=True)
json.dump(out, open(OUT, "w"), indent=1, ensure_ascii=False)
print("saved", OUT)
