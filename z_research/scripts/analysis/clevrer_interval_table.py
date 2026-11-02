#!/usr/bin/env python3
"""CLEVRER 간격 시험 표 (2026-11-01) — 같은 장면에서 샘플 간격 0.08 → 0.16 초로 바꿨을 때 PFT 이득이 남는가.

입력: z_analysis/release/08_rubric_transfer/te/clevrer_<창>/ (transfer_gap_te.py 산출물) · data_csv/clevrer_interval_<창>/windows.json
정의 (장면 = quad 하나, 모델 m, S = 미래 블록 평균 surprise):
  m_A = S(vanish) − S(present)   (> 0 이면 사라짐 정답) · m_B = S(appear) − S(absent)
  문맥 효과 c = m_A + m_B (문맥의 물체 유무가 예측을 물체 쪽으로 당기는 정도) · 편향 b = (m_B − m_A)/2 · 둘 다 S(present) 로 나눠 %
  주 지표: (1) c 비율 = mean c(PFT) / mean c(릴리즈)  (2) A 이득 = A 정답률(PFT) − A 정답률(릴리즈)
  비교 집합: 두 c8t16 창 모두에서 미래 안에 다시 보이는 장면 (R ≥ 0) — 같은 장면끼리 짝지은 bootstrap (장면 단위, 10,000 회)
  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python; $P z_research/scripts/analysis/clevrer_interval_table.py
"""
import collections
import json
import os

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
TE = f"{REPO}/z_analysis/release/08_rubric_transfer/te"
M = ["release", "vobftg", "v3_full_s0", "copy"]
NAME = {"release": "릴리즈", "vobftg": "VOBench PFT", "v3_full_s0": "V3 PFT", "copy": "복사"}


def load(arm):
    d = f"{TE}/clevrer_{arm}"
    meta = json.load(open(f"{d}/meta.json")); tags = meta["pred_tags"]
    assert np.load(f"{d}/done.npy").all(), f"{arm} 미완료"
    l1 = np.load(f"{d}/l1.npy").mean(2); cp = np.load(f"{d}/copy.npy").mean(1)
    S = {m: (cp if m == "copy" else l1[:, tags.index(m)]) for m in M}
    g = collections.defaultdict(dict)
    for i, (vid, v) in enumerate(zip(meta["video_id"], meta["variant"])):
        g[vid.split("_")[0]][v] = i
    out = {}
    for sid, q in g.items():
        out[sid] = {}
        for m in M:
            s = S[m]
            mA = s[q["imp_ab"]] - s[q["pos_a"]]; mB = s[q["imp_ba"]] - s[q["pos_b"]]
            out[sid][m] = (mA, mB, s[q["pos_a"]])
    wj = json.load(open(f"{REPO}/data_csv/clevrer_interval_{arm}/windows.json")); win = wj["scenes"]
    for v in win.values():   # R = 두 미래가 갈리는 (이어 붙인 = 다시 보이는) 첫 미래 샘플. 창 밖이면 −1
        v["R"] = v["splice_sample"] - wj["context"] if v["splice_sample"] < wj["n_frames"] else -1
    return out, win


def summ(Q, ids):
    r = {}
    for m in M:
        mA = np.array([Q[i][m][0] for i in ids]); mB = np.array([Q[i][m][1] for i in ids]); S0 = np.array([Q[i][m][2] for i in ids])
        r[m] = dict(A=100 * np.mean(mA > 0), B=100 * np.mean(mB > 0), c=100 * np.mean((mA + mB) / S0), b=100 * np.mean((mB - mA) / 2 / S0),
                    cA=(mA > 0).astype(float), cc=(mA + mB) / S0)
    return r


def main():
    arms = [a for a in ("s2_c16t32", "s2_c8t16", "s4_c8t16") if os.path.isdir(f"{TE}/clevrer_{a}")]
    D = {a: load(a) for a in arms}
    # 재현 확인: s2_c16t32 (PNG) vs 원 채점 (mp4, predictor_lab v3_ablation clevrer_final)
    if "s2_c16t32" in D:
        Q, _ = D["s2_c16t32"]; r = summ(Q, sorted(Q))
        ref = json.load(open(f"{REPO}/predictor_lab/results/v3_ablation/table.json"))["clevrer"]["all"]
        print("== 재현 확인 (원 창, PNG) — 전체 쌍 정답률 vs 원 채점 (mp4)")
        for m in M:
            allacc = (r[m]["A"] + r[m]["B"]) / 2
            print(f"  {NAME[m]:12s} {allacc:5.1f}  (원 {ref[m][0]:5.1f})   A {r[m]['A']:5.1f} B {r[m]['B']:5.1f}  c/S {r[m]['c']:+.2f} %  c 비율 {r[m]['c'] / r['release']['c']:.2f}×")
    if not {"s2_c8t16", "s4_c8t16"} <= set(D):
        return
    (Q2, W2), (Q4, W4) = D["s2_c8t16"], D["s4_c8t16"]
    ids = sorted(i for i in Q2 if W2[i]["R"] >= 0 and W4[i]["R"] >= 0)
    print(f"\n== 간격 비교 (문맥 8 + 미래 8, 두 창 모두 미래 안에 다시 보이는 장면 {len(ids)} / {len(Q2)})")
    print(f"  {'':12s} | {'0.08 초':^30s} | {'0.16 초':^30s}")
    print(f"  {'':12s} | {'A':>6s} {'B':>6s} {'c/S %':>7s} {'b/S %':>7s} | {'A':>6s} {'B':>6s} {'c/S %':>7s} {'b/S %':>7s}")
    r2, r4 = summ(Q2, ids), summ(Q4, ids)
    for m in M:
        x, y = r2[m], r4[m]
        print(f"  {NAME[m]:12s} | {x['A']:6.1f} {x['B']:6.1f} {x['c']:+7.2f} {x['b']:+7.2f} | {y['A']:6.1f} {y['B']:6.1f} {y['c']:+7.2f} {y['b']:+7.2f}")
    rng = np.random.default_rng(0); n = len(ids); B = rng.integers(0, n, (10000, n))
    print("\n== PFT 이득 — 0.08 초 vs 0.16 초 (같은 장면 짝, 95 % CI)")
    for m in ("vobftg", "v3_full_s0"):
        gA2 = r2[m]["cA"] - r2["release"]["cA"]; gA4 = r4[m]["cA"] - r4["release"]["cA"]
        d = gA4 - gA2; bs = d[B].mean(1)
        rat2 = r2[m]["cc"][B].mean(1) / r2["release"]["cc"][B].mean(1); rat4 = r4[m]["cc"][B].mean(1) / r4["release"]["cc"][B].mean(1)
        print(f"  {NAME[m]:12s} A 이득 {100 * gA2.mean():+5.1f} → {100 * gA4.mean():+5.1f} pt  (차 {100 * d.mean():+5.1f} [{100 * np.percentile(bs, 2.5):+5.1f}, {100 * np.percentile(bs, 97.5):+5.1f}])"
              f" | c 비율 {r2[m]['c'] / r2['release']['c']:.2f}× [{np.percentile(rat2, 2.5):.2f}, {np.percentile(rat2, 97.5):.2f}] → "
              f"{r4[m]['c'] / r4['release']['c']:.2f}× [{np.percentile(rat4, 2.5):.2f}, {np.percentile(rat4, 97.5):.2f}]")
    print("\n== 다시 보이는 미래 샘플 R 별 사라짐 A 정답률 (장면 수)  — 0.16 초는 같은 실제 시간이 R 절반")
    for arm, Q, W in (("0.08 초", Q2, W2), ("0.16 초", Q4, W4)):
        by = collections.defaultdict(list)
        for i in ids:
            by[W[i]["R"]].append(i)
        line = []
        for k in sorted(by):
            r = summ(Q, by[k])
            line.append(f"R{k}(n{len(by[k])}): " + "/".join(f"{r[m]['A']:.0f}" for m in M))
        print(f"  {arm}: " + "  ".join(line) + "   [릴리즈/VOBench/V3/복사]")


if __name__ == "__main__":
    main()
