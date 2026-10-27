"""예측이 그린 물체 벡터의 방향은 encoder 의 물체 벡터와 같은가 (2026-10-25, 학습 없음, pred_object_vector.py 캐시).
사용자: "그 물체를 그리는 어떤 벡터의 방향은 encoder 와 동일하다?"

참 자리 칸 c (물체가 문맥에서 없던 새 자리, bg 있는 칸) 에서:
  encoder 물체 벡터  e_Y = Y_c − bg_c                       (그 칸에 이 물체가 들어와 바뀐 만큼, 정답)
  예측 물체 벡터    e_P = (P_c − μ) − β (bg_c − μ)          (예측에서 배경 성분을 뺀 나머지; β = 블록마다 P 를 bg 에 회귀한 계수)
  지표: cos(e_P, e_Y) · e_P 에너지 중 e_Y 방향 몫 (= cos²) · ‖e_P‖/‖e_Y‖
  기준: encoder 끼리 — 마지막 문맥 블록의 물체 (Ō − bg_c) 와 e_Y 의 cos (같은 물체가 다른 블록 · 다른 칸에서 얼마나 같은 방향인가)
        예측 e_P 와 (Ō − bg_c) 의 cos (예측이 "마지막에 본 물체" 방향을 쓰나)
  무작위 기준: 1280 차원 두 무작위 방향의 |cos| ≈ 0.02
"""
import json
import numpy as np, torch
CACHE = "/workspace/auto_research_cache/pred_objvec"; S, NBC, NBF = 256, 8, 8
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
meta = json.load(open(f"{CACHE}/val_meta.json"))
mu = {}
for sp in ("ar", "rel"):
    acc = torch.zeros(1280, device=dev, dtype=torch.float64); n = 0
    for i in range(len(meta)):
        Y = torch.from_numpy(np.load(f"{CACHE}/val/{i:05d}.npz")[f"Y_{sp}"]).to(dev).double().reshape(-1, 1280); acc += Y.sum(0); n += len(Y)
    mu[sp] = (acc / n).float()
cos = lambda a, b: torch.nn.functional.cosine_similarity(a, b, dim=-1)
rows = []
for i, m in enumerate(meta):
    z = np.load(f"{CACHE}/val/{i:05d}.npz"); lab = m["labels"]
    for sp in ("ar", "rel"):
        Yf, Pf = torch.from_numpy(z[f"Y_{sp}"]).to(dev).float(), torch.from_numpy(z[f"P_{sp}"]).to(dev).float()
        bg, ob = torch.from_numpy(z[f"bg_{sp}"]).to(dev).float(), torch.from_numpy(z[f"obar_{sp}"]).to(dev).float()
        if torch.isnan(ob).any():
            continue
        for j in range(NBF):
            L = lab[NBC + j]; cells = L["cells"]
            if not L["ok"] or not cells or torch.isnan(bg[cells]).any():
                continue
            Y, P, B = Yf[j, cells], Pf[j, cells], bg[cells]
            pb, bb = (P - mu[sp]).reshape(-1), (B - mu[sp]).reshape(-1)
            beta = (pb @ bb) / (bb @ bb)
            eP = (P - mu[sp]) - beta * (B - mu[sp]); eY = Y - B; eO = ob[None] - B
            c = cos(eP, eY)
            rows.append({"sp": sp, "step": j + 1, "cos_pred_enc": float(c.mean()), "share": float((c ** 2).mean()),
                         "norm_ratio": float((eP.norm(dim=1) / eY.norm(dim=1)).mean()), "cos_enc_last_vs_now": float(cos(eO, eY).mean()),
                         "cos_pred_last": float(cos(eP, eO).mean()), "beta": float(beta)})
out = {}
for sp in ("ar", "rel"):
    out[sp] = {}
    for k in range(1, NBF + 1):
        R = [r for r in rows if r["sp"] == sp and r["step"] == k]
        out[sp][k] = {"n": len(R), **{key: round(float(np.median([r[key] for r in R])), 3) for key in ("cos_pred_enc", "share", "norm_ratio", "cos_enc_last_vs_now", "cos_pred_last", "beta")}}
json.dump(out, open("/workspace/world_model/vjepa2/auto_research/results/pred_object_direction.json", "w"), indent=1)
for sp in ("ar", "rel"):
    for k in (1, 2, 4, 6, 8):
        print(sp, k, out[sp][k])


# ------------------------------------------------------------------ 2026-10-25 추가 (사용자: "배경 성분을 빼도 속도나 다양한 게 남잖아")
# 물체 벡터를 부분공간별로 나눠 비교: 정체 (모양 · 색) · 속도 · 위치 · 나머지.
# 부분공간 = train 정답 물체 벡터 (물체 칸 평균) 에 맞춘 선형 probe 의 활성 패턴 (Σ W) 열 공간. 정체 12 (5 + 7) · 속도 2 · 위치 2, 차례로 직교화.
# 칸 단위 벡터에 칸 평균으로 맞춘 부분공간을 쓰는 근사다.
tmeta = json.load(open(f"{CACHE}/train_meta.json"))
def ridge(X, Y, lam=1e2):
    u, s, vt = np.linalg.svd(X, full_matrices=False)
    return vt.T @ ((s / (s ** 2 + lam))[:, None] * (u.T @ Y))
def pattern(Xs, Sig, ps, Y):
    W = ridge(Xs, Y - Y.mean(0)); return (Sig @ W) / ps[:, None]
SUB = {}
for sp in ("ar", "rel"):
    X, sh, co, vel, pos = [], [], [], [], []
    for i, m in enumerate(tmeta):
        pooled = np.load(f"{CACHE}/train/{i:05d}.npz")[f"objpool_{sp}"]; lab = m["labels"]; r = 0
        for b in range(16):
            if lab[b]["ok"] and lab[b]["cells"]:
                if b >= 1 and lab[b - 1]["ok"] and np.isfinite(lab[b - 1]["ctr"]).all():
                    X.append(pooled[r]); sh.append(m["shape"]); co.append(m["color"])
                    vel.append(np.array(lab[b]["ctr"]) - np.array(lab[b - 1]["ctr"])); pos.append(lab[b]["ctr"])
                r += 1
    X = np.array(X, np.float32); pm, ps = X.mean(0), X.std(0) + 1e-6; Xs = (X - pm) / ps; Sig = np.cov(Xs, rowvar=False)
    A_id = np.concatenate([pattern(Xs, Sig, ps, np.eye(6, dtype=np.float32)[sh]), pattern(Xs, Sig, ps, np.eye(8, dtype=np.float32)[co])], 1)
    A_vel = pattern(Xs, Sig, ps, np.array(vel, np.float32)); A_pos = pattern(Xs, Sig, ps, np.array(pos, np.float32))
    def basis(A, r, prev):
        A = torch.from_numpy(A).float().to(dev)
        if prev is not None:
            A = A - prev @ (prev.T @ A)
        U, s_, _ = torch.linalg.svd(A, full_matrices=False); return U[:, :r]
    Qi = basis(A_id, 12, None); Qv = basis(A_vel, 2, Qi); Qp = basis(A_pos, 2, torch.cat([Qi, Qv], 1))
    SUB[sp] = {"identity": Qi, "velocity": Qv, "position": Qp}
    print(sp, "부분공간 학습 표본", len(X), flush=True)
rows2 = []
for i, m in enumerate(meta):
    z = np.load(f"{CACHE}/val/{i:05d}.npz"); lab = m["labels"]
    for sp in ("ar", "rel"):
        Yf, Pf = torch.from_numpy(z[f"Y_{sp}"]).to(dev).float(), torch.from_numpy(z[f"P_{sp}"]).to(dev).float()
        bg = torch.from_numpy(z[f"bg_{sp}"]).to(dev).float()
        for j in range(NBF):
            L = lab[NBC + j]; cells = L["cells"]
            if not L["ok"] or not cells or torch.isnan(bg[cells]).any():
                continue
            Y, P, B = Yf[j, cells], Pf[j, cells], bg[cells]
            pb, bb = (P - mu[sp]).reshape(-1), (B - mu[sp]).reshape(-1); beta = (pb @ bb) / (bb @ bb)
            eP = (P - mu[sp]) - beta * (B - mu[sp]); eY = Y - B
            Q_all = torch.cat(list(SUB[sp].values()), 1)
            r = {"sp": sp, "step": j + 1}
            for name, Q in list(SUB[sp].items()) + [("rest", None)]:
                if Q is None:
                    pP, pY = eP - (eP @ Q_all) @ Q_all.T, eY - (eY @ Q_all) @ Q_all.T
                else:
                    pP, pY = (eP @ Q) @ Q.T, (eY @ Q) @ Q.T
                r[f"{name}_cos"] = float(cos(pP, pY).mean())
                r[f"{name}_norm_ratio"] = float((pP.norm(dim=1) / pY.norm(dim=1).clamp_min(1e-6)).mean())
                r[f"{name}_share_enc"] = float(((pY ** 2).sum(1) / (eY ** 2).sum(1)).mean())
                r[f"{name}_share_pred"] = float(((pP ** 2).sum(1) / (eP ** 2).sum(1)).mean())
            rows2.append(r)
out2 = {}
keys = [k for k in rows2[0] if k not in ("sp", "step")]
for sp in ("ar", "rel"):
    out2[sp] = {}
    for k in range(1, NBF + 1):
        R = [r for r in rows2 if r["sp"] == sp and r["step"] == k]
        out2[sp][k] = {"n": len(R), **{key: round(float(np.median([r[key] for r in R])), 3) for key in keys}}
json.dump({"whole": out, "by_subspace": out2}, open("/workspace/world_model/vjepa2/auto_research/results/pred_object_direction.json", "w"), indent=1)
for sp in ("ar", "rel"):
    for k in (1, 4, 8):
        d = out2[sp][k]
        print(sp, "step", k, " | ".join(f"{n}: cos {d[n + '_cos']:.2f} 크기비 {d[n + '_norm_ratio']:.2f} 몫(enc/pred) {d[n + '_share_enc']:.3f}/{d[n + '_share_pred']:.3f}" for n in ("identity", "velocity", "position", "rest")))
