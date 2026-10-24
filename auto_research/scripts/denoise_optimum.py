#!/usr/bin/env python3
"""[T] 고리 1 → 2 인과 (학습 없이, 최적해 수준): 블록 토큰의 nuisance 를 줄이면 데이터 최적 예측의 "평균 · 수축" 이 줄어드나 (auto_research, 2026-10-21).

  /venv/vjepa/bin/python auto_research/scripts/denoise_optimum.py

[E] 수집 캐시 (블록 8 개 창): 같은 프레임을 다섯 가지로 인코딩한 토큰 — 학습 정답과 같은 16 비트 · 32 비트 · ±1/255 잡음 두 번 · 재압축.
  원본   = 16 비트 하나 (학습 정답 그대로)
  평활 4 = 16 비트 · ±1/255 두 번 · 재압축 네 개의 평균 (무작위 평활 꼴, 입력 미세 변화에 대한 기대값 근사)
같은 창 · 같은 정답 블록 (t+1 = 5, 6, 7) 에서 두 토큰 공간 각각:
  (a) 합 1 최소제곱 시간 가중 (같은 칸, 블록 t−4..t)  → 마지막 블록 몫
  (b) 칸 평균 쪽 스칼라 수축 계수 k (블록 하나)        → k → 1 이면 수축이 필요 없다
  (c) 복사 · 5 블록 평균 · 수축 의 L1 (복사 대비)
가설 (고리 1 → 2): nuisance 가 평균의 원인이면 평활 토큰에서 마지막 블록 몫과 k 가 커지고 평균 보상이 줄어야 한다."""
import json, os
import numpy as np, torch

C = "/workspace/auto_research_cache/enc_noise"
G, D = 16, 1280
torch.set_num_threads(64)
res = {}
for src in ("k710", "ssv2", "ht100m", "phys3d"):
    r = {}
    for name, idx in (("orig", [0]), ("smooth4", [0, 2, 3, 4])):
        Ts = []
        for sh in (0, 1):
            T_ = np.load(os.path.join(C, f"{src}_target_encoder_s{sh}_tok.npy"), mmap_mode="r")
            Ts.append(torch.from_numpy(np.ascontiguousarray(T_[:, idx])).float().mean(1))     # (N,8,256,D)
        H = torch.cat(Ts)
        mu = H.mean((0, 1))
        M = torch.zeros(5, 5, dtype=torch.float64); v = torch.zeros(5, dtype=torch.float64)
        kn = kd = 0.0
        L = {"copy": 0.0, "mean5": 0.0, "shrink": 0.0, "lsw": 0.0}
        n = 0
        for t in (4, 5, 6):
            X = H[:, t - 4:t + 1].permute(1, 0, 2, 3).reshape(5, -1).double()
            y = H[:, t + 1].reshape(-1).double()
            M += X @ X.T; v += X @ y
            a_, b_ = (H[:, t] - mu).reshape(-1).double(), (H[:, t + 1] - mu).reshape(-1).double()
            kn += float(a_ @ b_); kd += float(a_ @ a_)
        Ai = torch.linalg.inv(M); one = torch.ones(5, dtype=torch.float64)
        wu = Ai @ v; w = wu + (1 - one @ wu) / (one @ Ai @ one) * (Ai @ one)
        k = kn / kd
        for t in (4, 5, 6):
            y = H[:, t + 1]
            L["copy"] += float((H[:, t] - y).abs().mean()); L["mean5"] += float((H[:, t - 4:t + 1].mean(1) - y).abs().mean())
            L["shrink"] += float((mu + k * (H[:, t] - mu) - y).abs().mean())
            L["lsw"] += float((torch.einsum("k,nkcd->ncd", w.float(), H[:, t - 4:t + 1]) - y).abs().mean())
            n += 1
        L = {kk: vv / n for kk, vv in L.items()}
        noise = float((H[:, 1:] - H[:, :-1]).abs().mean()); scale = float((H[1:] - H[:-1]).abs().mean())
        r[name] = dict(w=w.tolist(), k=k, L=L, avg_reward_pct=100 * (L["copy"] - L["mean5"]) / L["copy"], shrink_reward_pct=100 * (L["copy"] - L["shrink"]) / L["copy"],
                       lsw_reward_pct=100 * (L["copy"] - L["lsw"]) / L["copy"], block_change=noise, other_window=scale)
        print(f"{src:7s} {name:8s}: 시간 가중 (t−4..t) {np.round(w.numpy(), 3).tolist()} · 수축 k {k:.3f} · 보상 (복사 대비) 평균 {r[name]['avg_reward_pct']:+.1f} % · "
              f"수축 {r[name]['shrink_reward_pct']:+.1f} % · 최적 가중 {r[name]['lsw_reward_pct']:+.1f} % · 이웃 블록 변화 / 다른 창 {noise:.3f} / {scale:.3f}", flush=True)
    res[src] = r
json.dump(res, open(os.path.join(os.path.dirname(__file__), "../results/denoise_optimum.json"), "w"), indent=1)
