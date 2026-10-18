#!/usr/bin/env python3
"""context encoder 출력에서 '지금 상태' 가 읽히는가 — v11 late 가림 (2026-10-13, 사용자: "미래 state 를 읽는 건 이상한 분석").

미래 위치 자 (v11_future_from_context.py) 는 미래로 넘어가는 규칙 (외삽) 을 자가 라벨에서 배울 수 있어
'encoder 가 미래를 안다' 와 '자가 외삽을 배웠다' 를 못 가른다. 여기서는 **미래 라벨 없이** 문맥 안의 상태만 읽는다.

표현: z = V-JEPA 2 ViT-H **context encoder** (online `encoder` 가중치) 가 문맥 16 샘플만 보고 낸 토큰 (LN, 8 튜블릿 × 256 × 1280).
      특징은 10-02 추출본 재사용 (`auto_research/_stage/v11_ffc_feats/z.npy`, release 체크포인트의 context encoder).
대상: IntPhysGen v11 vanish, 물체 있는 pos_a 1,344 clip (가림 없음 k0 672 · late 가림 k1–4 672). late = 문맥 마지막 k 샘플이 가림막 뒤.
표적 (문맥 안, 288 px 좌표; 슬롯 = 튜블릿 2 샘플 평균):
  last_pos  마지막으로 **보인** 문맥 슬롯의 위치
  last_vel  그 슬롯과 바로 앞 슬롯의 위치 차 (px / 슬롯)
  now_pos   문맥 마지막 슬롯 (7) 의 위치 — late 에서는 가림막 뒤라 **안 보이는** 지금 자리 (가려진 동안 움직인 만큼을 추론해야 함)
자: 표적마다 쿼리 1 개가 T×N 전체 토큰을 attend → Linear(D, 2) (3 × 1280 + 3 × 2,562 = 11,526 파라미터). L1, Adam 1e-3, 60 epoch, batch 64.
split: block 단위 절반 (condition × k 층화) 학습 → 나머지 절반 test. 기준선: condition 별 학습 평균 (영상별 상태를 안 읽는 값).

  python v11_context_state_probe.py --gpu 0
산출: auto_research/exp_results/future_from_context/v11/state_probe.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[3]
FEAT = ROOT / "auto_research/_stage/v11_ffc_feats"
OUT = ROOT / "auto_research/exp_results/future_from_context/v11"
T, S, D, RES = 8, 256, 1280, 144.0
TARGETS = ("last_pos", "last_vel", "now_pos")


class Head(nn.Module):
    def __init__(s, nq):
        super().__init__(); s.q = nn.Parameter(torch.randn(nq, D) / D ** 0.5); s.out = nn.ModuleList(nn.Linear(D, 2) for _ in range(nq))

    def forward(s, tok):                                                     # (B, T*S, D) → (B, nq, 2)
        a = torch.softmax(torch.einsum("bnd,qd->bqn", tok, s.q) / D ** 0.5, -1)
        v = torch.einsum("bqn,bnd->bqd", a, tok)
        return torch.stack([o(v[:, i]) for i, o in enumerate(s.out)], 1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--gpu", type=int, default=0); ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); dev = torch.device("cuda", a.gpu); torch.manual_seed(a.seed)
    L = np.load(FEAT / "labels.npz", allow_pickle=True)
    P, hid, cond, k, block = (L["P"].astype(np.float32) + 1) * RES, L["hid"], L["cond"].astype(str), L["k"].astype(int), L["block"].astype(str)
    n = len(P)
    # 표적 (문맥 슬롯 0–7)
    Y = np.zeros((n, 3, 2), np.float32); ok = np.ones(n, bool)
    for i in range(n):
        vis = [j for j in range(T) if hid[i, j] < 0.5]
        if len(vis) < 2: ok[i] = False; continue
        j = vis[-1]; jp = j - 1 if (j - 1) in vis else vis[-2]
        Y[i, 0] = P[i, j]; Y[i, 1] = (P[i, j] - P[i, jp]) / (j - jp); Y[i, 2] = P[i, T - 1]
    scale = np.array([RES, 10.0, RES], np.float32)[None, :, None]; off = np.array([1.0, 0.0, 1.0], np.float32)[None, :, None]
    Yn = Y / scale - off
    # split: block 단위 절반, condition × k 층화
    rng = np.random.default_rng(a.seed); tr = np.zeros(n, bool)
    for key in sorted(set(zip(cond, k))):
        blks = sorted(set(block[(cond == key[0]) & (k == key[1])])); rng.shuffle(blks)
        for b in blks[:len(blks) // 2]: tr |= block == b
    tr &= ok; te = ~tr & ok
    z = np.load(FEAT / "z.npy", mmap_mode="r")
    Z = torch.from_numpy(np.ascontiguousarray(z[:])).to(dev)                 # (n, 2048, 1280) fp16
    Yt = torch.from_numpy(Yn).to(dev)
    head = Head(3).to(dev); opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    tri = np.nonzero(tr)[0]
    for ep in range(a.epochs):
        perm = rng.permutation(tri); tot = 0.0
        for b0 in range(0, len(perm), 64):
            ii = torch.from_numpy(perm[b0:b0 + 64]).to(dev)
            pr = head(Z[ii].float()); loss = (pr - Yt[ii]).abs().mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(ii)
        if ep % 20 == 0 or ep == a.epochs - 1: print(f"ep {ep} train L1 {tot / len(tri):.4f}", flush=True)
    with torch.no_grad():
        pred = torch.cat([head(Z[torch.arange(i, min(i + 64, n), device=dev)].float()) for i in range(0, n, 64)]).cpu().numpy()
    pred = (pred + off) * scale
    # 기준선: condition 별 학습 평균 (k 무관)
    base = np.zeros_like(Y)
    for c in set(cond): base[cond == c] = Y[tr & (cond == c)].mean(0)
    err = lambda X: np.linalg.norm(X - Y, axis=-1)                          # (n, 3) px (속도는 px/슬롯)
    E, B = err(pred), err(base)
    res = {"n_train": int(tr.sum()), "n_test": int(te.sum()), "targets": TARGETS, "cells": {}}
    print(f"\ncontext encoder (release ckpt `encoder`) z → 문맥 안 상태, held-out {te.sum()} clip. 오차 = 유클리드 (위치 px · 속도 px/슬롯)  [condition 평균 기준선]")
    print(f"{'condition':24s} {'k':>3s} " + " ".join(f"{t:>20s}" for t in TARGETS) + "   |true vel|  n")
    for c in ("static_visible", "moving_visible_flat", "moving_visible", "static_occlusion", "moving_occlusion_flat", "moving_occlusion"):
        for kk in sorted(set(k[cond == c])):
            m = te & (cond == c) & (k == kk)
            if not m.any(): continue
            row = {t: [float(np.median(E[m, i])), float(np.median(B[m, i]))] for i, t in enumerate(TARGETS)}
            row["true_speed"] = float(np.median(np.linalg.norm(Y[m, 1], axis=-1))); row["n"] = int(m.sum())
            res["cells"][f"{c}/k{kk}"] = row
            print(f"{c:24s} {kk:3d} " + " ".join(f"{row[t][0]:8.1f} [{row[t][1]:7.1f}]  " for t in TARGETS) + f"  {row['true_speed']:6.1f}  {row['n']}")
    OUT.mkdir(parents=True, exist_ok=True); json.dump(res, open(OUT / "state_probe.json", "w"), indent=1); print("saved", OUT / "state_probe.json")


if __name__ == "__main__":
    main()
