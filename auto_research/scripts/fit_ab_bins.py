#!/usr/bin/env python3
"""[물리] 자연 영상 (K710 + HT100M) 데이터 최적 (a, b) 를 안정도 5 구간별로 (학습 없음, 모델 없음). context_reliability.py 와 같은 칸 · 같은 회귀 ·
같은 구간 경계 (자연 영상 5 분위) 인데, b 만이 아니라 a (같은 칸 문맥 평균 계수) 도 저장한다 — 구조 상한 오라클 (VOBench) 의 κ(s) 예측에 쓴다.
  y ≈ a·(같은 칸 5 블록 평균) + b·(실제 흐름으로 옮긴 마지막 블록 토큰), 안정도 s = cos(옮긴 토큰, 그 내용의 한 블록 앞 토큰)."""
import json, os, numpy as np, torch
EN = "/workspace/auto_research_cache/enc_noise"; RD = os.path.join(os.path.dirname(__file__), "../results/natural_transport_ctx5")
G, S = 16, 256
cy, cx = np.divmod(np.arange(S), G); P0 = np.stack([cx + 0.5, cy + 0.5], 1)
torch.set_num_threads(8)
def cell_of(p):
    s = np.floor(p).astype(int); ok = (s >= 0).all(1) & (s < G).all(1)
    return np.where(ok, np.clip(s[:, 1], 0, G - 1) * G + np.clip(s[:, 0], 0, G - 1), np.arange(S)), ok
R = []
for src in ("k710", "ht100m"):
    U = np.load(os.path.join(RD, f"rows_{src}.npz"))["Uwin"].astype(np.float32); w0 = 0
    for sh in (0, 1):
        T_ = np.load(os.path.join(EN, f"{src}_target_encoder_s{sh}_tok.npy"), mmap_mode="r")
        for w in range(T_.shape[0]):
            h = torch.from_numpy(np.asarray(T_[w, 0]).astype(np.float32))
            for t in (4, 5, 6):
                u1 = U[w0 + w, t + 1].reshape(S, 2); u0 = U[w0 + w, t].reshape(S, 2)
                mov = np.linalg.norm(u1, axis=1) >= 0.5
                js, ok1 = cell_of(P0 + u1); jp, ok0 = cell_of(P0[js] + u0[js])
                sel = np.where(mov & ok1 & ok0)[0]
                if sel.size == 0:
                    continue
                c = torch.from_numpy(sel); sj = torch.from_numpy(js[sel]); pj = torch.from_numpy(jp[sel])
                x1 = h[t - 4:t + 1].mean(0)[c]; x2 = h[t][sj]; y = h[t + 1][c]; xp = h[t - 1][pj]
                stab = torch.nn.functional.cosine_similarity(x2, xp, dim=-1)
                R.append(torch.stack([(x1 * x1).sum(-1), (x1 * x2).sum(-1), (x2 * x2).sum(-1), (x1 * y).sum(-1), (x2 * y).sum(-1), stab], -1).double().numpy())
        w0 += T_.shape[0]
st = np.concatenate(R)
edges = np.quantile(st[:, 5], [0.2, 0.4, 0.6, 0.8])
def solve(z):
    A = np.array([[z[:, 0].sum(), z[:, 1].sum()], [z[:, 1].sum(), z[:, 2].sum()]])
    return np.linalg.solve(A, [z[:, 3].sum(), z[:, 4].sum()]).tolist()
bins = np.digitize(st[:, 5], edges)
out = dict(edges=edges.tolist(), all=solve(st), bins=[dict(a=solve(st[bins == k])[0], b=solve(st[bins == k])[1], n=int((bins == k).sum()),
                                                         s_med=float(np.median(st[bins == k, 5]))) for k in range(5)])
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(os.path.dirname(__file__), "../results/fit_ab_bins.json"), "w"), indent=1)
