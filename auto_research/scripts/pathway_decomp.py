#!/usr/bin/env python3
"""[T] 고리 2 일반화: 손실을 줄이는 두 길 — (i) 같은 칸 문맥 블록 시간 평균, (ii) 마지막 블록 하나를 사전 · 이웃 칸 쪽으로 수축 —
   이 각각 복사 대비 손실을 얼마나 줄이나. 선형 예측기 (능선 최소제곱) 로 잰다, 학습된 predictor 없이 (auto_research, 2026-10-21).

  /venv/vjepa/bin/python auto_research/scripts/pathway_decomp.py

데이터: [E] 의 수집 캐시 (블록 8 개 창, 학습 정답과 같은 16 비트 블록 토큰) — K710 · SSv2 · HowTo100M · 3D 합성, 소스마다 반 (s0) 으로 맞추고 반 (s1) 으로 평가.
정답 = 블록 t+1 (t+1 = 5, 6, 7), 문맥 = 블록 t−4..t. 칸 평균 μ(c) = 맞춤 반의 칸별 평균 토큰 (사전).
예측기 (모두 μ 로 중심화한 특징의 선형 결합 + μ):
  copy        h_t
  mean5       같은 칸 5 블록 평균                                  (시간 평균, 맞춤 없음)
  shrink_k    μ + k (h_t − μ), 스칼라 k 하나                          (순수 사전 수축)
  lin_t       μ + W (h_t − μ), W 1280×1280                            (블록 하나, 차원 사이 수축)
  lin_ts      [h_t, 3×3 이웃 평균, 화면 평균] (블록 하나 + 공간)
  lin_tm      [h_t, 앞 4 블록 같은 칸 평균]                           (블록 하나 + 시간)
  lin_all     [h_t, 앞 4 블록 평균, 3×3 이웃, 화면 평균]
보고: 소스별 L1 · L2 와 복사 대비 감소율. 맞춤은 L2 (최소제곱) — L1 최적과 다르지만 길의 크기를 재기엔 충분하다."""
import glob, json, os
import numpy as np, torch

C = "/workspace/auto_research_cache/enc_noise"
SRC = ["k710", "ssv2", "ht100m", "phys3d"]
G, D = 16, 1280
TGT = [5, 6, 7]
torch.set_num_threads(64)


def load(src, sh):
    f = os.path.join(C, f"{src}_target_encoder_s{sh}_tok.npy")
    return np.load(f, mmap_mode="r")                                   # (N,5,8,256,D), 0 = 학습 정답과 같은 bf16


def feats(H, t, mu):
    """H (N,8,256,D) float32 torch, t = 마지막 문맥 블록 → dict of (N·256, D) 중심화 특징, y."""
    ht = H[:, t] - mu
    prev = H[:, t - 4:t].mean(1) - mu
    g = ht.view(-1, G, G, D).permute(0, 3, 1, 2)
    nb = torch.nn.functional.avg_pool2d(torch.nn.functional.pad(g, (1, 1, 1, 1), mode="replicate"), 3, 1).permute(0, 2, 3, 1).reshape(ht.shape)
    nb = (nb * 9 - ht) / 8                                              # 자기 칸 뺀 이웃 8 칸 평균
    glob_ = ht.mean(1, keepdim=True).expand_as(ht)
    y = H[:, t + 1] - mu
    r = lambda x: x.reshape(-1, D)
    return {"t": r(ht), "m": r(prev), "nb": r(nb), "g": r(glob_)}, r(y), r(H[:, t - 4:t + 1].mean(1) - mu)


SETS = {"lin_t": ["t"], "lin_ts": ["t", "nb", "g"], "lin_tm": ["t", "m"], "lin_all": ["t", "m", "nb", "g"]}
res = {}
for src in SRC:
    tr = torch.from_numpy(np.ascontiguousarray(load(src, 0)[:, 0])).float()      # (N,8,256,D)
    te = torch.from_numpy(np.ascontiguousarray(load(src, 1)[:, 0])).float()
    mu = tr.mean((0, 1))                                                         # (256,D) 칸별 사전 평균
    Gm = {k: None for k in SETS}; Bv = {k: None for k in SETS}
    kn, kd = 0.0, 0.0
    for t in [x - 1 for x in TGT]:
        F_, y, _ = feats(tr, t, mu)
        kn += float((F_["t"] * y).sum()); kd += float((F_["t"] ** 2).sum())
        for name, parts in SETS.items():
            X = torch.cat([F_[p] for p in parts], 1).double()
            g_, b_ = X.T @ X, X.T @ y.double()
            Gm[name] = g_ if Gm[name] is None else Gm[name] + g_
            Bv[name] = b_ if Bv[name] is None else Bv[name] + b_
    k_shr = kn / kd
    W = {}
    for name in SETS:
        A = Gm[name]; lam = 1e-3 * torch.trace(A) / A.shape[0]
        W[name] = torch.linalg.solve(A + lam * torch.eye(A.shape[0], dtype=A.dtype), Bv[name]).float()
    out = {k: [0.0, 0.0] for k in ["copy", "mean5", "shrink_k", *SETS]}
    n = 0
    for t in [x - 1 for x in TGT]:
        F_, y, m5 = feats(te, t, mu)
        P = {"copy": F_["t"], "mean5": m5, "shrink_k": k_shr * F_["t"]}
        for name, parts in SETS.items():
            P[name] = torch.cat([F_[p] for p in parts], 1) @ W[name]
        for k, p in P.items():
            e = p - y
            out[k][0] += float(e.abs().mean()); out[k][1] += float((e ** 2).mean())
        n += 1
    r = {k: {"l1": v[0] / n, "l2": v[1] / n} for k, v in out.items()}
    c1, c2 = r["copy"]["l1"], r["copy"]["l2"]
    for k in r:
        r[k]["l1_vs_copy_pct"] = 100 * (r[k]["l1"] - c1) / c1
        r[k]["rms_vs_copy_pct"] = 100 * (np.sqrt(r[k]["l2"] / c2) - 1)
    r["shrink_k_value"] = k_shr
    # 잡음 몫: 같은 블록 ±1/255 두 번 (변형 2, 3) 사이 차이 = 픽셀 미세 잡음 몫, 다른 창 같은 칸 거리
    te_all = load(src, 1)
    hn1 = torch.from_numpy(np.ascontiguousarray(te_all[:, 2, 7])).float(); hn2 = torch.from_numpy(np.ascontiguousarray(te_all[:, 3, 7])).float()
    r["pix_noise_l1"] = float((hn1 - hn2).abs().mean())
    r["other_window_l1"] = float((te[1:, 7] - te[:-1, 7]).abs().mean())
    r["block_to_block_l1"] = float((te[:, 7] - te[:, 6]).abs().mean())
    res[src] = r
    print(f"\n== {src} (맞춤 창 {tr.shape[0]}, 평가 창 {te.shape[0]}) — 스칼라 수축 k {k_shr:.3f}", flush=True)
    for k in ["copy", "mean5", "shrink_k", *SETS]:
        print(f"  {k:9s} L1 {r[k]['l1']:.4f} ({r[k]['l1_vs_copy_pct']:+.1f} %)   rms ({r[k]['rms_vs_copy_pct']:+.1f} %)")
    print(f"  척도: 다른 창 같은 칸 {r['other_window_l1']:.3f} · 이웃 블록 {r['block_to_block_l1']:.3f} · 같은 블록 ±1/255 두 번 {r['pix_noise_l1']:.3f}")
json.dump(res, open(os.path.join(os.path.dirname(__file__), "../results/pathway_decomp.json"), "w"), indent=1)
