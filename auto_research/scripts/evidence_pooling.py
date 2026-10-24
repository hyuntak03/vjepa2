#!/usr/bin/env python3
"""[S] 후보 F 측정 (등록 KST 05:58, 리더 승인 KST 06:0x, 학습 없음, GPU 없음): 증거 잡음 감쇠 — 옛 자리 증거를 모으면 실제 영상 L1 이 보상하는 운반이 커지나,
그리고 기준 블록 AR 은 그 증거를 데이터만큼 쓰나.

  OMP_NUM_THREADS=8 /venv/vjepa/bin/python auto_research/scripts/evidence_pooling.py

⚠️ 증거로 흐름이 고른 옛 자리 토큰을 쓰는 것은 데이터 통계 진단이다 (GUARDRAILS 2-1, 정리 3′ 의 κ 와 같은 자). 예측기가 아니다.
데이터 = [P] 공유 forward 창 (/workspace/auto_research_cache/ctxt_compare): K710 보류 150 (흐름 natural_holdout U) · HT100M 160 (흐름 natural_transport_ctx5 Uwin).
  블록 토큰 h = ctx (블록 0..3) + gt (블록 4..7). 모델 = 기준 블록 AR 의 TF 한 걸음 예측 (pred_base, 블록 4..7).
칸 · 회귀 = fit_ab_bins 와 같다: t ∈ {4,5,6}, 움직이는 칸 c (|U[t+1](c)| ≥ 0.5 칸), 옛 자리 s = c + U[t+1](c) (블록 t), 그 앞 p = s + U[t](s) (블록 t−1).
  y ≈ a·x1 + b·E,  x1 = 같은 칸 5 블록 평균 (블록 t−4..t), 차원 공유 스칼라 계수.
증거 E 셋: E1 = h_t(s) (칸 하나, 지금 κ) · E3 = 블록 t 의 s 둘레 3×3 평균 · E3h = ½ (E3 + 블록 t−1 의 p 둘레 3×3 평균).
운반 보상 = (SSE(x1 만) − SSE(x1, E)) / SSE(x1 만), 데이터 y = 정답 h_{t+1}(c). 모델 계수 = 같은 회귀를 y = 기준 AR 예측으로.
판정 (등록): 운반 보상(E3) ≥ 1.5 × 운반 보상(E1) 이고 모델 b(E3) < 0.7 × 데이터 b(E3) → "실제 영상 L1 이 더 보상하는 운반을 지금 해가 놓친다". 아니면 닫는다."""
import json, os
import numpy as np, torch, torch.nn.functional as F

torch.set_num_threads(8)
CC = "/workspace/auto_research_cache/ctxt_compare"; REPO = "/workspace/world_model/vjepa2"
G, S = 16, 256
cy, cx = np.divmod(np.arange(S), G); P0 = np.stack([cx + 0.5, cy + 0.5], 1)
EV = ("E1", "E3", "E3h", "Ec", "E3x")   # Ec · E3x = 사후 진단 (판정 밖): 같은 칸 마지막 블록 (복사) · c 를 뺀 3×3
# 창별 충분통계: 증거마다 [x1x1, x1E, EE, x1y, Ey, yy, x1m, Em, mm] (m = 모델 예측), + n
NST = 9


def cell_of(p):
    s = np.floor(p).astype(int); ok = (s >= 0).all(1) & (s < G).all(1)
    return np.where(ok, np.clip(s[:, 1], 0, G - 1) * G + np.clip(s[:, 0], 0, G - 1), np.arange(S)), ok


def pool3(hm):                                                                       # (S, D) → (S, D) 3×3 평균 (화면 안 칸만)
    z = hm.T.reshape(1, -1, G, G)
    return F.avg_pool2d(z, 3, 1, 1, count_include_pad=False).reshape(-1, S).T


FLOW = {"k710_holdout": np.load("/workspace/auto_research_cache/natural_holdout/k710_320_holdout_U.npy").astype(np.float32),
        "ht100m": np.load(f"{REPO}/auto_research/results/natural_transport_ctx5/rows_ht100m.npz")["Uwin"].astype(np.float32)}
per = {}
for src, U in FLOW.items():
    meta = json.load(open(f"{CC}/{src}_meta.json"))
    CX, GT, PB = (np.load(f"{CC}/{src}_{k}.npy", mmap_mode="r") for k in ("ctx", "gt", "pred_base"))
    rows = []
    for i, w in enumerate(meta["windows"]):
        h = torch.from_numpy(np.concatenate([np.asarray(CX[i]), np.asarray(GT[i])]).astype(np.float32))     # (8, S, D)
        pm = torch.from_numpy(np.asarray(PB[i]).astype(np.float32))                                        # 블록 4..7
        st = np.zeros((len(EV), NST + 1))
        for t in (4, 5, 6):
            u1 = U[w, t + 1].reshape(S, 2); u0 = U[w, t].reshape(S, 2)
            mov = np.linalg.norm(u1, axis=1) >= 0.5
            js, ok1 = cell_of(P0 + u1); jp, ok0 = cell_of(P0[js] + u0[js])
            sel = np.where(mov & ok1 & ok0)[0]
            if sel.size == 0:
                continue
            c = torch.from_numpy(sel); sj = torch.from_numpy(js[sel]); pj = torch.from_numpy(jp[sel])
            x1 = h[t - 4:t + 1].mean(0)[c]; y = h[t + 1][c]; m = pm[t + 1 - 4][c]
            p3t, p3p = pool3(h[t]), pool3(h[t - 1])
            Es = {"E1": h[t][sj], "E3": p3t[sj], "E3h": 0.5 * (p3t[sj] + p3p[pj]), "Ec": h[t][c]}
            # E3x: s 둘레 3×3 에서 c 자신을 뺀 평균 (c 가 이웃이면 빼고, 아니면 E3 그대로)
            ys, xs = sj // G, sj % G; yc, xc = c // G, c % G
            cnt = torch.zeros(sel.size); acc = torch.zeros(sel.size, h.shape[-1])
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy_, xx_ = ys + dy, xs + dx
                    ok = (yy_ >= 0) & (yy_ < G) & (xx_ >= 0) & (xx_ < G) & ~((yy_ == yc) & (xx_ == xc))
                    idx = (yy_.clamp(0, G - 1) * G + xx_.clamp(0, G - 1))
                    acc += h[t][idx] * ok[:, None].float(); cnt += ok.float()
            Es["E3x"] = acc / cnt.clamp_min(1)[:, None]
            for k, e in enumerate(EV):
                E = Es[e]
                st[k, :NST] += [float((a * b).sum()) for a, b in ((x1, x1), (x1, E), (E, E), (x1, y), (E, y), (y, y), (x1, m), (E, m), (m, m))]
                st[k, NST] += sel.size
        rows.append(st)
    per[src] = np.stack(rows)                                                        # (W, 3, 10)


def fit(s, ycol):                                                                    # s (10,) 합 → (a, b, sse, sse_x1only, yy)
    xx, xe, ee = s[0], s[1], s[2]
    xy, ey, yy = (s[3], s[4], s[5]) if ycol == "y" else (s[6], s[7], s[8])
    a, b = np.linalg.solve([[xx, xe], [xe, ee]], [xy, ey])
    sse = yy - 2 * a * xy - 2 * b * ey + a * a * xx + 2 * a * b * xe + b * b * ee
    a0 = xy / xx; sse0 = yy - 2 * a0 * xy + a0 * a0 * xx
    return a, b, sse, sse0, yy


def stats(P):                                                                        # P (W, 3, 10) → dict
    s = P.sum(0); o = {}
    for k, e in enumerate(EV):
        a, b, sse, sse0, yy = fit(s[k], "y"); am, bm, *_ = fit(s[k], "m")
        o[e] = dict(a=a, b=b, reward=(sse0 - sse) / sse0, r2=1 - sse / yy, a_model=am, b_model=bm, ratio=bm / b)
    o["reward_ratio_E3_E1"] = o["E3"]["reward"] / o["E1"]["reward"]
    o["reward_ratio_E3h_E1"] = o["E3h"]["reward"] / o["E1"]["reward"]
    return o


rng = np.random.default_rng(0); out = {}
for name, P in (("k710_holdout", per["k710_holdout"]), ("ht100m", per["ht100m"]), ("natural", np.concatenate([per["k710_holdout"], per["ht100m"]]))):
    v = stats(P); W = P.shape[0]
    bs = [stats(P[rng.integers(0, W, W)]) for _ in range(500)]
    res = {"n_cells": int(P[:, 0, NST].sum()), "n_windows": W}
    for e in EV:
        res[e] = {kk: [round(float(v[e][kk]), 4)] + [round(float(q), 4) for q in np.percentile([b_[e][kk] for b_ in bs], [2.5, 97.5])] for kk in v[e]}
    for kk in ("reward_ratio_E3_E1", "reward_ratio_E3h_E1"):
        res[kk] = [round(float(v[kk]), 3)] + [round(float(q), 3) for q in np.percentile([b_[kk] for b_ in bs], [2.5, 97.5])]
    out[name] = res
    print(f"== {name} (창 {W}, 칸·걸음 {res['n_cells']})")
    for e in EV:
        r = res[e]
        print(f"  {e:4s} 데이터 a {r['a'][0]:.3f} b {r['b'][0]:.3f} [{r['b'][1]:.3f},{r['b'][2]:.3f}] · 운반 보상 {100 * r['reward'][0]:.2f} % [{100 * r['reward'][1]:.2f},{100 * r['reward'][2]:.2f}]"
              f" · 모델 b {r['b_model'][0]:.3f} · 모델/데이터 {r['ratio'][0]:.3f} [{r['ratio'][1]:.3f},{r['ratio'][2]:.3f}]")
    print(f"  보상 비 E3/E1 {res['reward_ratio_E3_E1']} · E3h/E1 {res['reward_ratio_E3h_E1']}")
nat = out["natural"]
verdict = (nat["reward_ratio_E3_E1"][0] >= 1.5) and (nat["E3"]["ratio"][0] < 0.7)
out["verdict_registered"] = dict(reward_ratio_ge_1p5=nat["reward_ratio_E3_E1"][0] >= 1.5, model_ratio_lt_0p7=nat["E3"]["ratio"][0] < 0.7, pass_=bool(verdict))
print("판정 (등록, 자연 합침):", out["verdict_registered"])
json.dump(out, open(f"{REPO}/auto_research/results/evidence_pooling.json", "w"), indent=1)
