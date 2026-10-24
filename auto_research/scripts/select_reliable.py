#!/usr/bin/env python3
"""[P] 1 순위 마지막 한 수 — 실제 영상 "고르기" 로 조건화 보상을 장난감 문턱까지 키울 수 있나 (학습 없음, CPU) (auto_research, 2026-10-21).

  OMP_NUM_THREADS=8 /venv/vjepa/bin/python auto_research/scripts/select_reliable.py

데이터 (블록 8 개 창, 블록별 LN(target_encoder) bf16, 칸 흐름 RAFT): 학습 캐시 natural_train (K710 · SSv2 · HT100M 300 창씩, Uflow) ·
  [E] 캐시 (K710 · SSv2 · HT100M 160 창씩, natural_transport_ctx5 흐름) · K710 보류 (150 창). 흐름은 측정 (칸 고르기 · s) 에만 쓴다.
칸 (t = 4 · 5 · 6): 움직이는 칸 c (|U[t+1](c)| ≥ 0.5 칸), 내용 자리 j = c + U[t+1](c), 블록 t−1 자리 p = j + U[t](j), 안정도 s = cos(h_t(j), h_{t−1}(p)) ([T] 정의).
회귀 ([T] ft_eval_reliability 와 같다): y = h_{t+1}(c) ≈ a·(같은 칸 5 블록 평균) + b·h_t(j).
  조건화 보상 = Σ_k (β_g − β_k)ᵀ A_k (β_g − β_k) (안정도 5 구간, 경계 0.35 · 0.449 · 0.557 · 0.724 고정) — 전역 (a, b) 대신 구간별을 쓸 때 줄어드는 제곱 오차.
  운반 보상 = b = 0 (a 만) → 전역 (a, b). 둘 다 움직이는 칸의 전역 회귀 잔차 제곱합 대비 %.
창의 "신뢰할 만한 움직임 몫" = 움직이는 칸 중 s ≥ 0.724 (자연 5 분위 최상위 경계) 인 칸의 비율.
고르기: 몫 상위 x % 창 (x = 5 · 10 · 20 · 100) 을 모았을 때 (소스별 · 전체) 그 부분집합의 몫 · 조건화 보상 · 운반 보상 · 비.
장난감 환산 ([T] rs_r10 · r25 · r50: 외형 계수 단단함 1 · 무름 0.3, 외형 분산 1, 정답 잡음 0.6² — 같은 식을 해석적으로):
  조건화 보상 / (전역 잔차 + 잡음) = f(1−f)(1−0.3)² / (f(1−b_g)² + (1−f)(1 − 0.6 b_g + b_g²) + 0.36), b_g = f + 0.3(1−f)."""
import json, os
import numpy as np

G, S = 16, 256
EDGES = np.array([0.35, 0.449, 0.557, 0.724])
cy, cx = np.divmod(np.arange(S), G); P0 = np.stack([cx + 0.5, cy + 0.5], 1)
EN = "/workspace/auto_research_cache/enc_noise"
RD = os.path.join(os.path.dirname(__file__), "../results/natural_transport_ctx5")


def cell_of(p):
    s = np.floor(p).astype(int); ok = (s >= 0).all(1) & (s < G).all(1)
    return np.where(ok, np.clip(s[:, 1], 0, G - 1) * G + np.clip(s[:, 0], 0, G - 1), np.arange(S)), ok


def windows():
    for src, nm in (("k710", "k710_320"), ("ssv2", "ssv2"), ("ht100m", "ht100m_1pct")):
        H = np.load(f"/workspace/auto_research_cache/natural_train/{nm}_h.npy", mmap_mode="r"); U = np.load(f"/workspace/auto_research_cache/natural_train/{nm}_Uflow.npy", mmap_mode="r")
        for w in range(H.shape[0]):
            yield src, "train", np.asarray(H[w], np.float32), np.asarray(U[w], np.float32).reshape(8, S, 2)
        Uw = np.load(os.path.join(RD, f"rows_{src}.npz"))["Uwin"].astype(np.float32); w0 = 0
        for sh in (0, 1):
            T_ = np.load(os.path.join(EN, f"{src}_target_encoder_s{sh}_tok.npy"), mmap_mode="r")
            for w in range(T_.shape[0]):
                yield src, "enc", np.asarray(T_[w, 0], np.float32), Uw[w0 + w].reshape(8, S, 2)
            w0 += T_.shape[0]
    H = np.load("/workspace/auto_research_cache/natural_holdout/k710_320_holdout_h.npy", mmap_mode="r"); U = np.load("/workspace/auto_research_cache/natural_holdout/k710_320_holdout_U.npy", mmap_mode="r")
    for w in range(H.shape[0]):
        yield "k710", "holdout", np.asarray(H[w], np.float32), np.asarray(U[w], np.float32).reshape(8, S, 2)


WIN = []                                                                        # 창마다 (소스, 세트, 칸 통계 (n, 7))
for src, part, h, U in windows():
    rows = []
    for t in (4, 5, 6):
        u1, u0 = U[t + 1], U[t]
        mov = np.linalg.norm(u1, axis=1) >= 0.5
        js, ok1 = cell_of(P0 + u1); jp, ok0 = cell_of(P0[js] + u0[js])
        sel = np.where(mov & ok1 & ok0)[0]
        if sel.size == 0:
            continue
        x1 = h[t - 4:t + 1].mean(0)[sel]; x2 = h[t][js[sel]]; y = h[t + 1][sel]; xp = h[t - 1][jp[sel]]
        s = (x2 * xp).sum(-1) / (np.linalg.norm(x2, axis=-1) * np.linalg.norm(xp, axis=-1))
        rows.append(np.stack([(x1 * x1).sum(-1), (x1 * x2).sum(-1), (x2 * x2).sum(-1), (x1 * y).sum(-1), (x2 * y).sum(-1), (y * y).sum(-1), s], -1))
    WIN.append((src, part, np.concatenate(rows) if rows else np.zeros((0, 7))))
print(f"창 {len(WIN)} 개 (움직이는 칸 있는 창 {sum(len(r) > 0 for _, _, r in WIN)})", flush=True)


def rewards(st):
    def fit(m):
        A = np.array([[st[m, 0].sum(), st[m, 1].sum()], [st[m, 1].sum(), st[m, 2].sum()]])
        return np.linalg.solve(A, [st[m, 3].sum(), st[m, 4].sum()]), A
    allm = np.ones(len(st), bool); bg, Ag = fit(allm)
    bins = np.digitize(st[:, 6], EDGES); gc = 0.0; bk = []
    for k in range(5):
        m = bins == k
        if m.sum() < 30:
            bk.append(np.nan); continue
        bd, Ak = fit(m); d = bg - bd; gc += float(d @ Ak @ d); bk.append(float(bd[1]))
    a0 = st[:, 3].sum() / st[:, 0].sum(); d0 = bg - np.array([a0, 0.0]); gt = float(d0 @ Ag @ d0)
    sse = float(st[:, 5].sum() - 2 * (bg[0] * st[:, 3].sum() + bg[1] * st[:, 4].sum()) + bg @ Ag @ bg)
    return dict(n_cells=int(len(st)), share_reliable=float((st[:, 6] >= EDGES[-1]).mean()), cond_pct=100 * gc / sse, transport_pct=100 * gt / sse,
                ratio=gc / gt, b_bins=bk, b_global=float(bg[1]))


res = {"windows": {}, "select": {}}
share = np.array([(r[:, 6] >= EDGES[-1]).mean() if len(r) else np.nan for _, _, r in WIN]); nmov = np.array([len(r) for _, _, r in WIN])
srcs = np.array([s for s, _, _ in WIN])
print("\n== 창별 '신뢰할 만한 움직임 몫' (움직이는 칸 중 s ≥ 0.724) 분포")
for s_ in ("k710", "ssv2", "ht100m"):
    v = share[(srcs == s_) & np.isfinite(share)]
    q = np.percentile(v, [10, 25, 50, 75, 90, 95])
    res["windows"][s_] = dict(n=int(len(v)), pct=q.tolist(), mean=float(v.mean()), frac_zero=float((v == 0).mean()))
    print(f"  {s_:7s} 창 {len(v)}: 평균 {v.mean():.3f} · 10/25/50/75/90/95 분위 {np.round(q, 3).tolist()} · 몫 0 인 창 {100 * (v == 0).mean():.0f} %")
print("\n== 고르기: 몫 상위 x % 창만 모았을 때 (전체 = 소스 합침, 창마다 순위)")
for scope in ("all", "k710", "ssv2", "ht100m"):
    idx = np.where(np.isfinite(share) & ((srcs == scope) if scope != "all" else True))[0]
    order = idx[np.argsort(-share[idx])]
    for x in (100, 20, 10, 5):
        k = max(1, int(round(len(order) * x / 100)))
        st = np.concatenate([WIN[i][2] for i in order[:k]])
        r = rewards(st); r["n_windows"] = int(k); r["min_window_share"] = float(share[order[k - 1]])
        res["select"][f"{scope}/top{x}"] = r
        print(f"  {scope:6s} 상위 {x:3d} % (창 {k:4d}, 칸 {r['n_cells']:6d}): 몫 {r['share_reliable']:.3f} · 조건화 보상 {r['cond_pct']:.2f} % · 운반 보상 {r['transport_pct']:.2f} % · 비 {r['ratio']:.3f} · "
              f"구간 b {np.round(r['b_bins'], 3).tolist()}")
print("\n== 장난감 환산 (해석적, [T] rs_r* 설정: 외형 계수 1 / 0.3, 외형 분산 1, 정답 잡음 0.6²) — 같은 단위 (움직이는 칸 잔차 대비 %) 와 비")
toy = {}
for f, split in ((0.10, 0.00), (0.25, 0.03), (0.50, 0.37)):
    bg = f + 0.3 * (1 - f)
    gc = f * (1 - f) * 0.7 ** 2; gt = bg ** 2
    sse = f * (1 - bg) ** 2 + (1 - f) * (1 - 0.6 * bg + bg ** 2) + 0.36
    toy[f] = dict(cond_pct=100 * gc / sse, transport_pct=100 * gt / sse, ratio=gc / gt, split_measured=split)
    print(f"  단단함 {int(f * 100):2d} %: 조건화 보상 {100 * gc / sse:.1f} % · 운반 보상 {100 * gt / sse:.1f} % · 비 {gc / gt:.2f} · (측정된 갈림 {split:.2f})")
res["toy"] = toy
json.dump(res, open(os.path.join(os.path.dirname(__file__), "../results/select_reliable.json"), "w"), indent=1)
