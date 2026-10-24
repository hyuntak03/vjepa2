#!/usr/bin/env python3
"""고리 1 [E] — nuisance 부분공간 vs 물체 · 장면 신호 부분공간 (auto_research, 2026-10-21, 리더 우선순위 1).

  CUDA_VISIBLE_DEVICES=3 /venv/vjepa/bin/python -W ignore auto_research/scripts/enc_noise_subspace.py

질문: 블록 토큰 h (블록별 LN(target_encoder), bf16 = 학습 정답) 의 nuisance (같은 물리 내용인데 바뀌는 성분) 가
      물체 · 장면 신호와 다른 방향에 있나 (→ nuisance 불변 상태 공간 / Mahalanobis 거리가 길이 된다), 같은 방향에 퍼져 있나 (→ 막힌다).
모든 양은 2 차 모멘트 행렬 (1280×1280) 로: E[x xᵀ]. 창을 반으로 나눠 앞 반 (train) 으로 metric 을 만들고 뒤 반 (test) 으로 잰다.
nuisance (블록 하나당 한 번 뽑힌 크기로 맞춤):
  N_rep    (h(x+n1) − h(x+n2)) / √2,  n = ±1/255 정수 잡음                             (주입한 미세 섭동)
  N_st     정지 칸 (블록 간 칸 평균 |Δ픽셀| < 1) 의 (h_b − h_{b−1}) / √2                 (데이터의 실제 블록 간 nuisance — 번짐 포함)
  N_bf     h_bf16 − h_fp32                                                             (수치)
신호:
  S_cross  다른 창 같은 칸 · 같은 블록 (h_i − h_j)/√2 − N_st    (장면 내용 신호, nuisance 몫을 빼서 보정)
  S_mov    픽셀이 크게 바뀐 칸 (≥ 16 단계) 의 h_b − h_{b−1} − 2 N_st   (움직임 · 변화 신호, 보정)
  (phys3d 캐시) S_obj  정지 물체 칸 (덮임 > 0.6, 블록 m 개 평균) − 같은 칸 배경 (물체 없던 블록 평균), 보정 = (1/m + 1/m_bg) N_st3
                S_tr   물체가 움직인 쌍의 물체 칸 h_{t+1} − h_t − 2 N_st3     (운반해야 할 변화)
                N_st3  물체가 한 번도 안 닿은 칸 (2 칸 팽창 밖) 의 (h_b − h_{b−1})/√2
metric M (train nuisance N 로 만듦): I (지금 L1 · L2 손실의 등방 거리) · Mahalanobis (N + εI)⁻¹ (ε = 평균 고유값 × 1e-3, 1e-2, 1e-1) ·
         상위 k nuisance 주성분 제거 (k = 8, 32, 128) · 일반화 고유 (S_cross 대 N) 상위 k 방향만.
비 = [tr(M S) ] / tr(M N_test)  (신호 에너지 / nuisance 에너지). 이득 = 비(M) / 비(I).
공간 공분산: N_rep 의 칸 사이 상관 E[δ_c·δ_c'] / E|δ|² 를 체비셰프 거리별 · 전역 성분 몫 |mean_c δ_c|² / mean_c |δ_c|²."""
import argparse, glob, json, os
import numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="/workspace/auto_research_cache/enc_noise")
ap.add_argument("--p3", default="/workspace/auto_research_cache/phys3d")
ap.add_argument("--sources", nargs="+", default=["phys3d", "vobench", "k710", "ssv2", "ht100m", "ek100"])
ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "../results/enc_noise_subspace.json"))
a = ap.parse_args()
dev = torch.device("cuda"); G, D = 16, 1280
torch.backends.cuda.matmul.allow_tf32 = False
yy, xx = np.meshgrid(np.arange(G), np.arange(G), indexing="ij")
CHEB = torch.from_numpy(np.maximum(np.abs(yy.reshape(-1, 1) - yy.reshape(1, -1)), np.abs(xx.reshape(-1, 1) - xx.reshape(1, -1)))).to(dev)


class Mom:
    def __init__(self):
        self.S = torch.zeros(D, D, dtype=torch.float64, device=dev); self.n = 0

    def add(self, X):
        if X.shape[0]:
            X = X.to(dev, torch.float64); self.S += X.T @ X; self.n += X.shape[0]

    def get(self):
        return (self.S / max(self.n, 1)).float()


def metrics(Ntr, Nte, sigs, Sc_tr):
    """sigs: {name: (S_raw, bias_coef)} — S 는 보정 전 2 차 모멘트, bias = coef × Nte (test nuisance) 를 뺀다."""
    lam, U = torch.linalg.eigh(Ntr.double()); lam, U = lam.flip(0).clamp_min(0), U.flip(1)
    out = {"nuis_PR": float(lam.sum() ** 2 / (lam ** 2).sum()),
           "nuis_cum_var": {str(k): float(lam[:k].sum() / lam.sum()) for k in (1, 8, 32, 128, 512)}}
    Ms = {"I": torch.eye(D, device=dev, dtype=torch.float64)}
    mbar = float(lam.mean())
    for e in (1e-3, 1e-2, 1e-1):
        Ms[f"mahal_eps{e:g}"] = U @ torch.diag(1.0 / (lam + e * mbar)) @ U.T
    for k in (8, 32, 128):
        P = torch.eye(D, device=dev, dtype=torch.float64) - U[:, :k] @ U[:, :k].T
        Ms[f"drop_top{k}"] = P
    # 일반화 고유: S_cross (보정) 대 Ntr — 신호/nuisance 가 큰 방향
    if Sc_tr is not None:
        Sc = (Sc_tr - Ntr).double()                                       # train 신호로 방향을 고른다 (test 누수 없음)
        W = U @ torch.diag(1.0 / torch.sqrt(lam + 1e-2 * mbar)) @ U.T
        ev, V = torch.linalg.eigh(W @ Sc @ W); V = W @ V.flip(1)
        for k in (16, 64, 256):
            Q, _ = torch.linalg.qr(V[:, :k]); Ms[f"gen_top{k}"] = Q @ Q.T
    Nte = Nte.double()
    res = {}
    for mn, M in Ms.items():
        nte = float(torch.trace(M @ Nte))
        r = {"nuis_energy_frac": nte / float(torch.trace(Nte)) if mn.startswith(("drop", "gen")) else None}
        for sn, (S, c) in sigs.items():
            S = S.double(); sc = float(torch.trace(M @ (S - c * Nte)))
            r[sn] = sc / nte
            if mn.startswith(("drop", "gen")):
                r[sn + "_energy_frac"] = sc / float(torch.trace(S - c * Nte))
        res[mn] = r
    for mn in res:
        for sn in sigs:
            res[mn][sn + "_gain"] = res[mn][sn] / res["I"][sn]
    out["metrics"] = res
    return out


rng = np.random.default_rng(0)
RES = {}
for src in a.sources:
    tf = sorted(glob.glob(os.path.join(a.dir, f"{src}_target_encoder_s*_tok.npy")))
    if not tf:
        continue
    Z = [np.load(f[:-8] + ".npz") for f in tf]
    T = [np.load(f, mmap_mode="r") for f in tf]
    items = [(k, i) for k in range(len(T)) for i in range(T[k].shape[0])]
    rng.shuffle(items)
    half = len(items) // 2
    M = {s: {k: Mom() for k in ("N_rep", "N_st", "N_bf", "S_cross", "S_mov")} for s in ("tr", "te")}
    spat = torch.zeros(256, 256, dtype=torch.float64, device=dev); glob_share = []
    spat_st = torch.zeros(256, 256, dtype=torch.float64, device=dev); cnt_st = torch.zeros(256, 256, dtype=torch.float64, device=dev)
    prevH = None
    for j, (k, i) in enumerate(items):
        part = "tr" if j < half else "te"
        t = torch.from_numpy(np.asarray(T[k][i, :4], np.float32)).to(dev)               # (4,nb,256,D): h0 hf n1 n2
        h0, hf, n1, n2 = t
        pixD = torch.from_numpy(Z[k]["pixD"][i]).to(dev)                                 # (nb−1,256)
        dn = (n1 - n2) / np.sqrt(2)
        M[part]["N_rep"].add(dn.reshape(-1, D)); M[part]["N_bf"].add((h0 - hf).reshape(-1, D))
        dh = h0[1:] - h0[:-1]
        M[part]["N_st"].add(dh[pixD < 1] / np.sqrt(2)); M[part]["S_mov"].add(dh[pixD >= 16])
        if prevH is not None:
            M[part]["S_cross"].add((h0[0] - prevH) / np.sqrt(2))
        prevH = h0[0]
        # 공간 상관 (±1/255 nuisance, 블록마다 칸 Gram)
        gm = torch.einsum("bcd,bed->ce", dn.double(), dn.double()); spat += gm
        glob_share.append(float((dn.mean(1).pow(2).sum(-1) / dn.pow(2).sum(-1).mean(1)).mean()))
        # 정지 칸 블록 간 nuisance 의 공간 상관 (같은 블록 쌍에서 둘 다 정지인 칸)
        st = (pixD < 1).double()
        dd = dh.double() * st[..., None]
        spat_st += torch.einsum("bcd,bed->ce", dd, dd); cnt_st += torch.einsum("bc,be->ce", st, st)
    Mtr = {k: v.get() for k, v in M["tr"].items()}; Mte = {k: v.get() for k, v in M["te"].items()}
    R = {"n_windows": len(items), "n_static_vec": M["tr"]["N_st"].n + M["te"]["N_st"].n, "n_mov_vec": M["tr"]["S_mov"].n + M["te"]["S_mov"].n}
    # 공간 상관 (거리별)
    dg = torch.diagonal(spat); C = spat / torch.sqrt(dg[:, None] * dg[None, :])
    R["spatial_rep"] = {str(r): float(C[CHEB == r].mean()) for r in (1, 2, 3, 4, 6, 8, 12, 15)}
    R["global_share_rep"] = float(np.mean(glob_share))
    ok = cnt_st > 20
    dgs = torch.diagonal(spat_st) / torch.diagonal(cnt_st).clamp_min(1)
    Cs = (spat_st / cnt_st.clamp_min(1)) / torch.sqrt(dgs[:, None] * dgs[None, :]).clamp_min(1e-12)
    R["spatial_static"] = {str(r): float(Cs[(CHEB == r) & ok].mean()) if ((CHEB == r) & ok).any() else None for r in (1, 2, 3, 4, 6, 8, 12, 15)}
    sigs = {"S_cross": (Mte["S_cross"], 1.0), "S_mov": (Mte["S_mov"], 2.0)}
    for nn in ("N_st", "N_rep"):
        R[f"by_{nn}"] = metrics(Mtr[nn], Mte[nn], sigs, Mtr["S_cross"])
    # nuisance 종류끼리 방향 겹침 (공분산 cos) · 상위 32 주성분 부분공간 겹침
    def cc(A, B):
        return float(torch.trace(A @ B) / (A.norm() * B.norm()))
    def sub(A, B, k=32):
        Ua = torch.linalg.eigh(A.double())[1][:, -k:]; Ub = torch.linalg.eigh(B.double())[1][:, -k:]
        return float((Ua.T @ Ub).pow(2).sum() / k)
    allN = {k: (Mtr[k] + Mte[k]) / 2 for k in Mtr}
    R["cov_cos"] = {f"{x}~{y}": cc(allN[x], allN[y]) for x, y in (("N_rep", "N_st"), ("N_rep", "N_bf"), ("N_st", "S_mov"), ("N_rep", "S_cross"), ("N_st", "S_cross"), ("S_mov", "S_cross"))}
    R["sub32"] = {f"{x}~{y}": sub(allN[x], allN[y]) for x, y in (("N_rep", "N_st"), ("N_st", "S_mov"), ("N_st", "S_cross"), ("N_rep", "N_bf"))}
    RES[src] = R
    torch.save({k: v.cpu() for k, v in allN.items()}, os.path.join(a.dir, f"_moments_{src}.pt"))
    b = R["by_N_st"]
    print(f"\n== {src} (창 {len(items)}, 정지 칸 벡터 {R['n_static_vec']}, 변화 칸 벡터 {R['n_mov_vec']})", flush=True)
    print(f"  nuisance (정지 칸 블록 간) 유효 차원 PR {b['nuis_PR']:.1f} · 상위 k 주성분 누적 분산 " + " ".join(f"{k}:{v:.2f}" for k, v in b["nuis_cum_var"].items()))
    print(f"  nuisance (±1/255)           PR {R['by_N_rep']['nuis_PR']:.1f} · " + " ".join(f"{k}:{v:.2f}" for k, v in R['by_N_rep']["nuis_cum_var"].items()))
    for nn in ("N_st", "N_rep"):
        print(f"  [nuisance = {nn}] 신호/nuisance 에너지비 (이득):")
        for mn, r in R[f"by_{nn}"]["metrics"].items():
            ex = f" (nuisance 남는 몫 {r['nuis_energy_frac']:.2f}, 장면 신호 남는 몫 {r.get('S_cross_energy_frac', float('nan')):.2f}, 변화 신호 {r.get('S_mov_energy_frac', float('nan')):.2f})" if r["nuis_energy_frac"] is not None else ""
            print(f"    {mn:16s} 장면 {r['S_cross']:7.3f} (×{r['S_cross_gain']:.2f}) · 변화 {r['S_mov']:7.3f} (×{r['S_mov_gain']:.2f}){ex}")
    print("  공간 상관 (±1/255) 거리 " + " ".join(f"{k}:{v:.3f}" for k, v in R["spatial_rep"].items()) + f" · 전역 성분 몫 {R['global_share_rep']:.3f} (독립이면 0.004)")
    print("  공간 상관 (정지 칸 블록 간) " + " ".join(f"{k}:{(v if v is not None else float('nan')):.3f}" for k, v in R["spatial_static"].items()))
    print("  방향 겹침 cov cos: " + " · ".join(f"{k} {v:.2f}" for k, v in R["cov_cos"].items()) + " | 상위 32 부분공간 겹침: " + " · ".join(f"{k} {v:.2f}" for k, v in R["sub32"].items()))

# ---- phys3d 캐시: 물체 신호 (라벨)
def cover(bbox):
    x0, y0, x1, y1 = [bbox[..., i, None] for i in range(4)]
    c = np.arange(G) * 16.0
    ox = np.clip(np.minimum(c + 16, x1) - np.maximum(c, x0), 0, 16) / 16
    oy = np.clip(np.minimum(c + 16, y1) - np.maximum(c, y0), 0, 16) / 16
    return oy[..., :, None] * ox[..., None, :]


def dil(m, k=1):
    for _ in range(k):
        o = m.copy()
        o[1:] |= m[:-1]; o[:-1] |= m[1:]; o[:, 1:] |= m[:, :-1]; o[:, :-1] |= m[:, 1:]
        o[1:, 1:] |= m[:-1, :-1]; o[:-1, :-1] |= m[1:, 1:]; o[1:, :-1] |= m[:-1, 1:]; o[:-1, 1:] |= m[1:, :-1]
        m = o
    return m


hfiles = sorted(glob.glob(os.path.join(a.p3, "h_val_target_s*.npy")))
if hfiles:
    MM = {s: {k: Mom() for k in ("N_st3", "S_obj", "S_tr", "S_cross3")} for s in ("tr", "te")}
    bias_obj = {"tr": [], "te": []}
    loc = {"obj_share": [], "obj_area": []}
    allw = [(f, w) for f in hfiles for w in range(np.load(f, mmap_mode="r").shape[0])]
    rng.shuffle(allw); half = len(allw) // 2
    prev = None
    for j, (f, w) in enumerate(allw):
        part = "tr" if j < half else "te"
        sh = f.split("_s")[-1].split(".")[0]
        lab = np.load(os.path.join(a.p3, f"lab_val_s{sh}.npz"))
        h = torch.from_numpy(np.asarray(np.load(f, mmap_mode="r")[w], np.float32)).to(dev)     # (16,256,D)
        bb = np.nan_to_num(lab["bbox"][w], nan=-100.0); cv = cover(bb)                        # (32,G,G)
        touch = (cv[0::2] > 0) | (cv[1::2] > 0)                                               # (16,G,G)
        never = ~dil(touch.any(0), 2).reshape(-1)
        if never.sum() > 5:
            nv = torch.from_numpy(never).to(dev)
            MM[part]["N_st3"].add(((h[1:] - h[:-1])[:, nv] / np.sqrt(2)).reshape(-1, D))
        if prev is not None:
            MM[part]["S_cross3"].add((h[0] - prev) / np.sqrt(2))
        prev = h[0]
        okf = lab["inframe_frac"][w] > 0.9
        vis = (lab["vis_frac"][w][0::2] > 0.7) & (lab["vis_frac"][w][1::2] > 0.7) & okf[0::2] & okf[1::2]
        inside = (np.minimum(cv[0::2], cv[1::2]) > 0.6).reshape(16, -1)                        # (16,256)
        free = ~dil(touch, 1).reshape(16, -1) if False else ~np.stack([dil(touch[b], 1) for b in range(16)]).reshape(16, -1)
        pos = lab["pos"][w]; vis = (lab["vis_frac"][w][0::2] > 0.7) & (lab["vis_frac"][w][1::2] > 0.7)
        # 정지 물체: 물체 칸이고 그 칸이 물체에 덮인 블록 m ≥ 3 개 평균
        insv = inside & vis[:, None]                                                      # 보임 조건 (2026-10-21 정정: 가려진 물체의 bbox 칸 제외)
        for c in np.where(insv.sum(0) >= 3)[0]:
            bo = np.where(insv[:, c])[0]; bg = np.where(free[:, c])[0]
            if len(bg) < 3:
                continue
            s = h[bo, c].mean(0) - h[bg, c].mean(0)
            MM[part]["S_obj"].add(s[None]); bias_obj[part].append(1 / len(bo) + 1 / len(bg))
        # 운반: 물체가 8 px 넘게 움직인 쌍의 물체 칸 (t 또는 t+1)
        for t_ in range(15):
            d = pos[2 * t_ + 3] - pos[2 * t_ + 1]
            if not (np.isfinite(d).all() and np.linalg.norm(d) > 8 and vis[t_] and vis[t_ + 1]):
                continue
            m = torch.from_numpy((touch[t_] | touch[t_ + 1]).reshape(-1)).to(dev)
            MM[part]["S_tr"].add(h[t_ + 1, m] - h[t_, m])
    Ntr, Nte = MM["tr"]["N_st3"].get(), MM["te"]["N_st3"].get()
    bo = float(np.mean(bias_obj["te"])) if bias_obj["te"] else 0.0
    sigs = {"S_obj": (MM["te"]["S_obj"].get(), bo), "S_tr": (MM["te"]["S_tr"].get(), 2.0), "S_cross": (MM["te"]["S_cross3"].get(), 1.0)}
    R = metrics(Ntr, Nte, sigs, MM["tr"]["S_cross3"].get())
    R["n"] = {k: MM["tr"][k].n + MM["te"][k].n for k in MM["tr"]}
    R["bias_obj_coef"] = bo
    RES["phys3d_labels"] = R
    print(f"\n== phys3d 캐시 (라벨) — nuisance = 물체가 안 닿은 칸의 블록 간 /√2 · n {R['n']}")
    print(f"  nuisance PR {R['nuis_PR']:.1f} · 상위 k 누적 " + " ".join(f"{k}:{v:.2f}" for k, v in R["nuis_cum_var"].items()))
    for mn, r in R["metrics"].items():
        ex = f" (nuisance 남는 몫 {r['nuis_energy_frac']:.2f}, 물체 {r.get('S_obj_energy_frac', float('nan')):.2f}, 운반 {r.get('S_tr_energy_frac', float('nan')):.2f})" if r["nuis_energy_frac"] is not None else ""
        print(f"    {mn:16s} 물체 대비 {r['S_obj']:7.3f} (×{r['S_obj_gain']:.2f}) · 운반 변화 {r['S_tr']:7.3f} (×{r['S_tr_gain']:.2f}) · 장면 {r['S_cross']:7.3f} (×{r['S_cross_gain']:.2f}){ex}")
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(RES, open(a.out, "w"), indent=1)
print("→", a.out)
