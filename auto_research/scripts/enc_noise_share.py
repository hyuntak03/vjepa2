#!/usr/bin/env python3
"""고리 1 [E] — 손실이 쫓는 대상 (블록 간 정답 변화) 중 nuisance 몫, 한 숫자로 (auto_research, 2026-10-21).

  /venv/vjepa/bin/python auto_research/scripts/enc_noise_share.py

가정 [이론]: h_b = s_b + n_b, n_b 는 블록마다 독립 · 평균 0 · 공분산 N (칸 · 소스마다 같은 분포). 그러면
  E‖h_b − h_{b−1}‖² = E‖s_b − s_{b−1}‖² + 2 tr N    (교차항 0)
  → nuisance 몫 (제곱 기준) = 2 tr N / E‖h_b − h_{b−1}‖²  — "복사 예측의 L2 손실 중 nuisance 가 만든 몫" (모든 칸 평균).
N 추정 두 가지: N_st (픽셀 정지 칸의 블록 간 차 /√2, 번짐 포함 = 데이터의 실제 nuisance 상한 쪽) · N_rep (±1/255 주입, 미세 픽셀 nuisance 만 = 하한 쪽).
L1 판: 칸 · 차원 평균 |Δh| 대비 nuisance 만의 |n_b − n_{b−1}| (N_rep 은 |h(x+n1) − h(x+n2)|, N_st 는 정지 칸 |Δh|) — 가우시안 근사 없이 그대로 비.
픽셀 공간 기준점: 같은 양을 칸 픽셀 벡터 (16×16×2×3, 0–255) 로 — ±1/255 반복 차 E|n1 − n2| = 8/9 단계 (균등 {−1,0,1}), 다른 창 같은 칸 거리, 정지 칸 블록 간 차."""
import glob, json, os
import numpy as np, torch

D_ = "/workspace/auto_research_cache/enc_noise"
RES = {}
for src in ("phys3d", "vobench", "k710", "ssv2", "ht100m", "ek100"):
    fs = sorted(glob.glob(f"{D_}/{src}_target_encoder_s*.npz"))
    mf = f"{D_}/_moments_{src}.pt"
    if not fs or not os.path.exists(mf):
        continue
    Z = [np.load(f) for f in fs]
    vg_sq1 = float(np.concatenate([z["vg0_sq"][:, 0] for z in Z]).mean())             # E (Δh)² per dim, lag 1, 모든 칸
    d0 = np.concatenate([z["d0"] for z in Z]); pixD = np.concatenate([z["pixD"] for z in Z]); rep = np.concatenate([z["rep"] for z in Z])
    M = torch.load(mf)
    D = M["N_rep"].shape[0]
    trN = {k: float(torch.trace(M[k])) / D for k in ("N_st", "N_rep", "N_bf")}
    r = {"E_dh2_per_dim": vg_sq1, "trN_per_dim": trN,
         "share_sq_N_st": 2 * trN["N_st"] / vg_sq1, "share_sq_N_rep": 2 * trN["N_rep"] / vg_sq1, "share_sq_N_bf": trN["N_bf"] * 2 / vg_sq1,
         "L1_copy": float(d0.mean()), "L1_static": float(d0[pixD < 1].mean()) if (pixD < 1).any() else None, "L1_rep": float(rep.mean())}
    r["share_L1_static"] = r["L1_static"] / r["L1_copy"] if r["L1_static"] else None
    r["share_L1_rep"] = r["L1_rep"] / r["L1_copy"]
    # 픽셀 공간 기준점
    P = [np.load(f[:-4] + "_pix.npy", mmap_mode="r") for f in fs]
    cross, stat = [], []
    rng = np.random.default_rng(0)
    pp = np.concatenate([np.asarray(p[:, :2], np.float32) for p in P])                  # 블록 0 (n,2,R,R,3)
    for _ in range(200):
        i, j = rng.choice(len(pp), 2, replace=False); cross.append(np.abs(pp[i] - pp[j]).mean())
    r["pix_cross"] = float(np.median(cross)); r["pix_rep"] = 8 / 9
    r["pix_static_blockchange"] = float(pixD[pixD < 1].mean()) if (pixD < 1).any() else None
    r["pix_copy"] = float(pixD.mean())
    r["pix_rep_over_cross"] = r["pix_rep"] / r["pix_cross"]
    r["pix_share_rep_L1"] = r["pix_rep"] / r["pix_copy"]
    RES[src] = r
    print(f"{src:8s} | 블록 간 변화 제곱 {vg_sq1:.3f}/차원 · nuisance 몫 (제곱): 정지 칸 추정 {r['share_sq_N_st']:.0%} · ±1/255 추정 {r['share_sq_N_rep']:.0%} · bf16 만 {r['share_sq_N_bf']:.0%} | "
          f"L1: 복사 {r['L1_copy']:.3f} · 정지 칸 {r['L1_static'] if r['L1_static'] else float('nan'):.3f} ({r['share_L1_static'] or float('nan'):.0%}) · ±1/255 반복 {r['L1_rep']:.3f} ({r['share_L1_rep']:.0%}) | "
          f"픽셀: ±1/255 / 다른 창 {r['pix_rep_over_cross']:.1%} · ±1/255 / 복사 {r['pix_share_rep_L1']:.1%}", flush=True)
json.dump(RES, open(os.path.join(os.path.dirname(__file__), "../results/enc_noise_share.json"), "w"), indent=1)
