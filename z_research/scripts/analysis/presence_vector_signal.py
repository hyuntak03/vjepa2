#!/usr/bin/env python3
"""예측 속 물체 신호의 분리도 · 방향 (2026-11-10, 진단 → 처방) — presence_probe_layers.py extract 가 만든 npz 를 읽는다. CPU 만.

물체 벡터  = 같은 장면의 (물체 장면 − 빈 장면) 차이 (블록마다, 평균 풀링 절반만)
실제 물체 방향 u = 실제 표현 물체 벡터의 장면 평균, 단위 길이
cos        = 예측 물체 벡터 장면 평균과 u 의 cos (방향 일치)
norm       = 예측 물체 벡터 평균 길이 ÷ 실제 물체 벡터 평균 길이 (보조, 주 지표 아님)
dp / dh    = u 위 투영값으로 잰 물체 장면 무리 · 빈 장면 무리의 분리도 d′ = (평균 차) ÷ √(두 무리 분산 평균), 예측 / 실제
eta_p / eta_h = 물체 유무가 설명하는 전체 분산 비율 η², 예측 / 실제
scene_cos  = 장면별 예측 물체 벡터와 실제 물체 벡터의 cos 중앙값
실제 영상 영속성 정답률 (real_world_permanence_uniform, 짝 A 사라짐 · 짝 B 나타남) 을 함께 적고 분리도 비 (dp/dh) 와 순위를 비교한다.

  /venv/vjepa/bin/python z_research/scripts/analysis/presence_vector_signal.py
"""
import json, os, sys
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, os.path.dirname(__file__))
O = f"{REPO}/z_analysis/results/presence_probe_20261110"
MODELS = ["vith", "vith_release_v4bc4", "vith_release_v4bc4_negimp_vanish_app02",
          "vast_ar_vw_state", "vast_ar_vw_state_v4bc4", "vast_ar_vw_state_v4bc4_negimp_vanish_app02"]
LATE = slice(2, 8)          # 미래 블록 3–8 (물체가 다시 보이는 구간)


def signal(m):
    d = np.load(f"{O}/{m}.npz")
    y, blk = d["label"], d["block"]
    D = d["out"].shape[-1] // 2                                   # 앞 절반 = 평균 풀링
    P = d["out"][:, :, :D].astype(np.float64)
    H = d["tgt"][:, :, :D].astype(np.float64)
    ia = {b: i for i, (b, l) in enumerate(zip(blk, y)) if l == 1}
    ib = {b: i for i, (b, l) in enumerate(zip(blk, y)) if l == 0}
    bs = [b for b in ia if b in ib]
    A = np.array([ia[b] for b in bs]); B = np.array([ib[b] for b in bs])
    r = {k: [] for k in ("cos", "norm", "dp", "dh", "eta_p", "eta_h", "scene_cos")}
    for k in range(P.shape[1]):
        dP = P[A, k] - P[B, k]; dH = H[A, k] - H[B, k]
        vp, vh = dP.mean(0), dH.mean(0)
        r["cos"].append(float(vp @ vh / np.linalg.norm(vp) / np.linalg.norm(vh)))
        r["norm"].append(float(np.linalg.norm(vp) / np.linalg.norm(vh)))
        u = vh / np.linalg.norm(vh)

        def dprime(X):
            a, b = X[A, k] @ u, X[B, k] @ u
            return float((a.mean() - b.mean()) / np.sqrt(0.5 * (a.var() + b.var())))

        def eta(X):
            Xa, Xb = X[A, k], X[B, k]
            allx = np.concatenate([Xa, Xb]); mu = allx.mean(0)
            between = len(A) * ((Xa.mean(0) - mu) ** 2).sum() + len(B) * ((Xb.mean(0) - mu) ** 2).sum()
            return float(between / ((allx - mu) ** 2).sum())

        r["dp"].append(dprime(P)); r["dh"].append(dprime(H))
        r["eta_p"].append(eta(P)); r["eta_h"].append(eta(H))
        cs = (dP * dH).sum(1) / np.linalg.norm(dP, axis=1) / np.linalg.norm(dH, axis=1)
        r["scene_cos"].append(float(np.median(cs)))
    r["n_scenes"] = len(bs)
    return r


def rank(x):
    o = np.argsort(np.argsort(x)); return o.astype(float)


def main():
    import ar_full_table as T
    out = {}
    print(f"{'모델':45s} 분리도 d′ 예측/실제 (비)   cos   η² 예측/실제   실제 영상 A / B (짝 수)")
    for m in MODELS:
        r = signal(m)
        L = lambda k: float(np.mean(r[k][LATE]))
        rw = T.rwp(m)
        pv = T.pv_of(f"{T.RES}/testbed/real_world_permanence_uniform_{m}/per_block.json")
        P = T.idx_pairs(f"{REPO}/data_csv/real_world_permanence_uniform/index.csv")
        n = {ab: sum(1 for p, i in P if p["pair_id"] == ab and p["video_id"] in pv and i["video_id"] in pv) for ab in ("A", "B")}
        r["late"] = {k: L(k) for k in ("cos", "norm", "dp", "dh", "eta_p", "eta_h", "scene_cos")}
        r["late"]["dp_ratio"] = r["late"]["dp"] / r["late"]["dh"]
        r["rwp"] = {**rw, "n_A": n["A"], "n_B": n["B"]}
        out[m] = r
        print(f"{m:45s} {L('dp'):.2f} / {L('dh'):.2f} ({r['late']['dp_ratio']:.2f})   {L('cos'):.2f}   {L('eta_p'):.3f} / {L('eta_h'):.3f}"
              f"   {rw['rwp_A']:.1f} / {rw['rwp_B']:.1f} ({n['A']}, {n['B']})")
    x = np.array([out[m]["late"]["dp_ratio"] for m in MODELS]); a = np.array([out[m]["rwp"]["rwp_A"] for m in MODELS])
    rho = float(np.corrcoef(rank(x), rank(a))[0, 1])
    out["_rank"] = {"spearman_dpratio_vs_rwpA": rho, "n_models": len(MODELS)}
    print(f"분리도 비 대 실제 영상 A 순위 상관 (Spearman, 모델 {len(MODELS)} 개) = {rho:.3f}")
    json.dump(out, open(f"{O}/vector_signal.json", "w"), indent=1)
    print("저장", f"{O}/vector_signal.json")


if __name__ == "__main__":
    main()
