#!/usr/bin/env python3
"""[T] 문맥 신뢰도 추론 검사: 데이터의 운반 신뢰 b 가 "문맥에서 보이는 내용 안정도" s 에 따라 달라지나, 그리고 기준 블록 AR 의 b 도 따라가나 (auto_research, 2026-10-21).

  CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 /venv/vjepa/bin/python -W ignore auto_research/scripts/context_reliability.py

데이터: [E] 수집 캐시 (블록 8 개 창, 학습 정답 16 비트 토큰) — K710 · HT100M (자연), SSv2 (⚠️ 기준 AR 학습 영상), 3D 합성 — 와 칸 흐름 (natural_transport_ctx5).
움직이는 칸 c (실제 흐름 |U[t+1](c)| ≥ 0.5 칸, t = 4, 5, 6), 그 내용의 블록 t 자리 s = c + U[t+1](c).
문맥 안정도 s_stab = cos( h_t(s), h_{t−1}(s + U[t](s)) ) — 블록 t−1 → t 사이 흐름을 따라 옮긴 같은 내용의 토큰 상관. 문맥 블록 두 개와 문맥 흐름만 쓴다
  (어느 내용인지 고르는 데에는 실제 흐름 U[t+1] 을 쓴다 — 운반 신뢰 b 의 회귀자와 같은 오라클).
회귀 (칸마다 차원 합 충분통계를 모아 안정도 구간별로 풂): y ≈ a·(같은 칸 5 블록 평균) + b·h_t(s). y = 정답 h_{t+1}(c) → 데이터 b, y = 기준 AR 의 TF 한 걸음 예측 → 모델 b.
사전 등록 판정: 데이터 b 가 s 하위 → 상위 구간에서 ≥ 0.1 오르는데 모델 b 가 평평하면 → "모델이 문맥으로 신뢰도를 추론하지 않는다". 둘 다 오르면 → 조건화는 하고 있음 (3D 합성의 s 분포를 본다)."""
import json, os, sys
import numpy as np, torch

sys.path.insert(0, os.path.dirname(__file__))
sys.argv = [sys.argv[0], "--loss", "l1", "--name", "x"]
import ar_crps_ft as FT                                                     # noqa: E402
torch.set_num_threads(8)
dev = torch.device("cuda")
EN = "/workspace/auto_research_cache/enc_noise"
RD = os.path.join(os.path.dirname(__file__), "../results/natural_transport_ctx5")
G, S = 16, 256
cy, cx = np.divmod(np.arange(S), G); P0 = np.stack([cx + 0.5, cy + 0.5], 1)
W = FT.NoiseWrap(FT.load_pred("natural_ar_vw_state", "latest.pt"), 32, 0.0).to(dev).eval()


def cell_of(p):
    s = np.floor(p).astype(int); ok = (s >= 0).all(1) & (s < G).all(1)
    return np.where(ok, np.clip(s[:, 1], 0, G - 1) * G + np.clip(s[:, 0], 0, G - 1), np.arange(S)), ok


ROWS = {}
for src in ("k710", "ht100m", "ssv2", "phys3d"):
    U = np.load(os.path.join(RD, f"rows_{src}.npz"))["Uwin"].astype(np.float32)
    R = []
    w0 = 0
    for sh in (0, 1):
        T_ = np.load(os.path.join(EN, f"{src}_target_encoder_s{sh}_tok.npy"), mmap_mode="r")
        for w in range(T_.shape[0]):
            h = torch.from_numpy(np.asarray(T_[w, 0]).astype(np.float32)).to(dev)            # (8,256,D)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                pr = W.tf(h[:7].reshape(1, -1, h.size(-1))).float()[0].view(7, S, -1)       # 위치 t → 블록 t+1
            for t in (4, 5, 6):
                u1 = U[w0 + w, t + 1].reshape(S, 2); u0 = U[w0 + w, t].reshape(S, 2)
                mov = np.linalg.norm(u1, axis=1) >= 0.5
                if mov.sum() == 0:
                    continue
                js, ok1 = cell_of(P0 + u1)
                jp, ok0 = cell_of(P0[js] + u0[js])
                sel = np.where(mov & ok1 & ok0)[0]
                if sel.size == 0:
                    continue
                c = torch.from_numpy(sel).to(dev); sj = torch.from_numpy(js[sel]).to(dev); pj = torch.from_numpy(jp[sel]).to(dev)
                x1 = h[t - 4:t + 1].mean(0)[c]; x2 = h[t][sj]; y = h[t + 1][c]; ym = pr[t][c]; xp = h[t - 1][pj]
                stab = torch.nn.functional.cosine_similarity(x2, xp, dim=-1)
                st = torch.stack([(x1 * x1).sum(-1), (x1 * x2).sum(-1), (x2 * x2).sum(-1), (x1 * y).sum(-1), (x2 * y).sum(-1),
                                  (x1 * ym).sum(-1), (x2 * ym).sum(-1), stab], -1).double().cpu().numpy()
                R.append(st)
        w0 += T_.shape[0]
    ROWS[src] = np.concatenate(R)
    print(f"{src}: 움직이는 칸 {len(ROWS[src])}", flush=True)


def solve(st):
    A = np.array([[st[:, 0].sum(), st[:, 1].sum()], [st[:, 1].sum(), st[:, 2].sum()]])
    bd = np.linalg.solve(A, [st[:, 3].sum(), st[:, 4].sum()])[1]
    bm = np.linalg.solve(A, [st[:, 5].sum(), st[:, 6].sum()])[1]
    return float(bd), float(bm)


res = {}
groups = {"natural (K710+HT100M)": np.concatenate([ROWS["k710"], ROWS["ht100m"]]), "ssv2 (학습 영상)": ROWS["ssv2"], "phys3d": ROWS["phys3d"]}
edges = np.quantile(groups["natural (K710+HT100M)"][:, 7], [0.2, 0.4, 0.6, 0.8])               # 자연 영상 5 분위 경계를 모든 그룹에 공통으로
print(f"\n안정도 구간 경계 (자연 영상 5 분위): {np.round(edges, 3).tolist()}")
for g, st in groups.items():
    bins = np.digitize(st[:, 7], edges)
    rows = []
    for k in range(5):
        m = bins == k
        if m.sum() < 50:
            rows.append(None); continue
        bd, bm = solve(st[m]); rows.append({"n": int(m.sum()), "frac": float(m.mean()), "b_data": bd, "b_model": bm, "s_med": float(np.median(st[m, 7]))})
    bd_all, bm_all = solve(st)
    res[g] = {"bins": rows, "b_data_all": bd_all, "b_model_all": bm_all, "s_quantiles": np.quantile(st[:, 7], [0.1, 0.25, 0.5, 0.75, 0.9]).tolist()}
    print(f"\n== {g}: 칸 {len(st)} · 전체 b 데이터 {bd_all:.3f} · 모델 {bm_all:.3f} · 안정도 분위 10/25/50/75/90 {np.round(res[g]['s_quantiles'], 3).tolist()}")
    for k, r in enumerate(rows):
        if r:
            print(f"   구간 {k} (s 중앙 {r['s_med']:.3f}, 몫 {r['frac']:.2f}, n {r['n']}): b 데이터 {r['b_data']:.3f} · 모델 {r['b_model']:.3f} · 모델/데이터 {r['b_model'] / r['b_data']:.2f}")
json.dump(res, open(os.path.join(os.path.dirname(__file__), "../results/context_reliability.json"), "w"), indent=1)
