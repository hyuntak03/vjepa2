"""예측은 물체를 어느 칸에 그리나 — 맞는 칸에 작게인가, 다른 칸에인가 (2026-10-25, 학습 없음, pred_object_vector.py 의 캐시를 읽는다).
사용자: "(N, D) shape 인데 다른 칸에 물체 그리는거야? 아니면 맞는 칸에 그리는데 물체 벡터 magnitude 가 작은거야".

칸 c 의 물체 점수 o_c: (X_c − μ) 를 [(bg_c − μ), (Ō − μ)] 에 최소제곱 → Ō 의 계수. X = 예측 P 또는 정답 Y.
  μ = 그 공간 정답 토큰 평균 (val 전체), bg_c = 문맥에서 물체가 그 칸을 안 덮은 블록 평균, Ō = 마지막 문맥 블록의 물체 칸 평균 (마지막에 본 물체).
  배경 성분을 같이 맞추므로 "평균 쪽으로 줄어듦" 은 물체 점수에 안 들어간다. 토큰 내용에는 절대 위치가 거의 없어 Ō 를 어느 칸에나 쓸 수 있다고 본다.
칸 무리: 참 자리 (블록 j 물체 칸) · 옛 자리 (블록 7 물체 칸 중 참 자리 아님) · 그 사이 (두 상자를 감싼 상자 안, 앞 둘 제외) · 나머지.
지표 (걸음별, 두 모델 · 정답):
  mass_*   양수 물체 점수의 합 중 각 무리의 몫
  o_true   참 자리 칸 물체 점수 평균 (정답이면 ≈ 물체 크기, 예측이면 얼마나 진하게)
  peak     물체 점수가 가장 큰 칸까지 거리: 참 중심 · 마지막 관측 중심 (px)
  com      양수 물체 점수의 무게 중심이 마지막 관측 → 참 위치 사이 어디인가 (0 = 마지막 관측 자리, 1 = 참 자리, 이동 2 칸 이상만)

  CUDA_VISIBLE_DEVICES=6 /venv/vjepa/bin/python -W ignore z_research/scripts/analysis/pred_object_where.py
출력: auto_research/results/pred_object_where.json
"""
import json, os
import numpy as np, torch

CACHE = "/workspace/auto_research_cache/pred_objvec"
OUT = "/workspace/world_model/vjepa2/auto_research/results/pred_object_where.json"
S, G, NBC, NBF = 256, 16, 8, 8
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
meta = json.load(open(f"{CACHE}/val_meta.json"))
cc = torch.tensor([[(c % G) * 16 + 8.0, (c // G) * 16 + 8.0] for c in range(S)], device=dev)    # 칸 중심 (x, y)

mu = {}
for sp in ("ar", "rel"):
    acc = torch.zeros(1280, device=dev, dtype=torch.float64); n = 0
    for i in range(len(meta)):
        Y = torch.from_numpy(np.load(f"{CACHE}/val/{i:05d}.npz")[f"Y_{sp}"]).to(dev).double().reshape(-1, 1280)
        acc += Y.sum(0); n += Y.shape[0]
    mu[sp] = (acc / n).float()


def scores(X, bg, ob, m):
    """X, bg (S, D); ob (D) → 칸별 Ō 계수 (S,), bg 가 없는 칸은 NaN."""
    x, b, o = X - m, bg - m, (ob - m).expand_as(X)
    bb, oo, bo = (b * b).sum(1), (o * o).sum(1), (b * o).sum(1)
    xb, xo = (x * b).sum(1), (x * o).sum(1)
    det = bb * oo - bo ** 2
    return (bb * xo - bo * xb) / det


rows = []
for i, m_ in enumerate(meta):
    z = np.load(f"{CACHE}/val/{i:05d}.npz"); lab = m_["labels"]; last = lab[NBC - 1]
    if not last["ok"] or not last["cells"]:
        continue
    lc = torch.tensor(last["ctr"], device=dev)
    for sp in ("ar", "rel"):
        Yf = torch.from_numpy(z[f"Y_{sp}"]).to(dev).float(); Pf = torch.from_numpy(z[f"P_{sp}"]).to(dev).float()
        bg = torch.from_numpy(z[f"bg_{sp}"]).to(dev).float(); ob = torch.from_numpy(z[f"obar_{sp}"]).to(dev).float()
        if torch.isnan(ob).any():
            continue
        okc = ~torch.isnan(bg).any(1)
        for j in range(NBF):
            L = lab[NBC + j]
            if not L["ok"] or not L["cells"]:
                continue
            true = set(L["cells"]); old = set(last["cells"]) - true
            bx = [min(L["box"][0], last["box"][0]), min(L["box"][1], last["box"][1]), max(L["box"][2], last["box"][2]), max(L["box"][3], last["box"][3])]
            between = {c for c in range(S) if bx[0] <= cc[c, 0] <= bx[2] and bx[1] <= cc[c, 1] <= bx[3]} - true - old
            tc = torch.tensor(L["ctr"], device=dev); move = float((tc - lc).norm())
            for who, X in (("gt", Yf[j]), (sp, Pf[j])):
                o = scores(X, torch.nan_to_num(bg), ob, mu[sp]); o[~okc] = float("nan")
                v = torch.nan_to_num(o, nan=0.0).clamp_min(0)
                tot = float(v.sum())
                if tot <= 0:
                    continue
                grp = lambda s_: float(v[list(s_)].sum() / tot) if s_ else 0.0
                tl = [c for c in true if okc[c]]
                r = {"clip": i, "space": sp, "who": who, "step": j + 1, "move_px": move,
                     "mass_true": grp(true), "mass_old": grp(old), "mass_between": grp(between),
                     "o_true": float(torch.nan_to_num(o[tl]).mean()) if tl else float("nan")}
                r["mass_rest"] = 1 - r["mass_true"] - r["mass_old"] - r["mass_between"]
                # 2026-10-25 추가: 물체 총량 — 자리와 무관하게 물체가 얼마나 남았나 (최대 칸 점수 · 참 ∪ 옛 ∪ 사이 칸의 양수 점수 합)
                r["o_peak"] = float(torch.nan_to_num(o, nan=-1e9).max())
                reg_ = [c for c in (true | old | between) if okc[c]]
                r["o_region_sum"] = float(v[reg_].sum()) if reg_ else float("nan")
                ol = [c for c in old if okc[c]]
                r["o_old"] = float(torch.nan_to_num(o[ol]).mean()) if ol else float("nan")
                k = int(torch.argmax(torch.nan_to_num(o, nan=-1e9)))
                r["peak_to_true_px"] = float((cc[k] - tc).norm()); r["peak_to_last_px"] = float((cc[k] - lc).norm())
                if move >= 32:
                    com = (v[:, None] * cc).sum(0) / v.sum()
                    d = tc - lc; r["com_frac"] = float(((com - lc) * d).sum() / (d * d).sum())
                rows.append(r)

out = {}
for sp in ("ar", "rel"):
    out[sp] = {}
    for who in ("gt", sp):
        R = [r for r in rows if r["space"] == sp and r["who"] == who]
        by = {}
        for k in range(1, NBF + 1):
            Rk = [r for r in R if r["step"] == k]
            f = lambda key: round(float(np.nanmedian([r.get(key, np.nan) for r in Rk])), 3)
            by[k] = {"n": len(Rk), **{key: f(key) for key in ("mass_true", "mass_old", "mass_between", "mass_rest", "o_true", "o_old", "o_peak", "o_region_sum", "peak_to_true_px", "peak_to_last_px", "com_frac")}}
        out[sp]["gt" if who == "gt" else "pred"] = by
json.dump(out, open(OUT, "w"), indent=1, ensure_ascii=False)
for sp in ("ar", "rel"):
    for who in ("gt", "pred"):
        print(sp, who)
        for k in (1, 2, 4, 6, 8):
            print("  step", k, out[sp][who][k])
print("saved", OUT)
