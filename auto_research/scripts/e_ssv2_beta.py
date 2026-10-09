#!/usr/bin/env python3
"""E-SSv2 2 판 분석 (적대 검증 2026-09-25 권고) — 토큰 수준 진전 지표 β 와 양성 대조.
입력 ssv2_e3/ (e_ssv2_extract.py 3 판 산출물: proj, D_p, D_pn, D_z, D_oa, D_oe, meta).

β (슬롯 간 변화의 추종): δ_i = p_{i+1} − p_i,  Δ_i = h_{9+i} − h_{8+i}  (토큰별, 무작위 투영 64 차원).
  방향 (정/역) · 슬롯 i · 토큰마다 clip 평균을 빼고 (clip 과 무관한 인덱스 위치 서명 제거)  β_i = Σ⟨δc, Δc⟩ / Σ‖Δc‖².
  1 = 진실의 슬롯 간 변화를 그대로 따름, 0 = clip 특이적 변화 없음. 귀무 = p 를 다른 clip 의 Δ 와 짝.
  양성 대조: 같은 β 를 오라클 o_i = (1−a)h_{8+i} + a·hc_7 (a = .7/.8/.9) 에 — 흐려도 진전하는 예측의 β 는 1−a.
시간 정렬 (보조): 귀무 오프셋 보정 — D[i,j] 에서 같은 (방향, i, j) 의 귀무 평균을 빼고 argmin → 대각 비율. 오라클 D_oa · D_oe 에도.
모든 지표를 방향별로 병기, CI = clip bootstrap (정/역을 같은 clip 으로 묶음).

2 차 적대 검증 (2026-09-25) 추가 → 새 키 `token2|forward`, `token2|reversed`, `token2|rho_reversed_minus_forward`, `align|diag_rel_margin_median_fullD`, `align|_note`.
  도착/이탈 분해, 진실 추세 오라클 (크기비 상한 없음), 수준 상관, 인덱스 구조, 공간 척도 사다리, L1 을 맞춘 복사 쪽 수축 오라클 (투영 공간, 자기 귀무).
  기존 키는 값이 그대로다 (따로 rng2). 기존 `p_slots4_7` 은 전이 3→4 를 포함한다 — 이름과 맞는 값은 token2 의 `token|beta_transitions_within_slots4_7`.
사용: python e_ssv2_beta.py <ssv2_e3 디렉토리>   (OUT_TAG=_x → ssv2_beta_x.json)
"""
import json, os, sys
from pathlib import Path
import numpy as np

I = Path(sys.argv[1] if len(sys.argv) > 1 else "/data2/local_datasets/world/world_analysis/cache/auto_research/ssv2_e3")
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "auto_research/exp_results/ssv2"; OUT.mkdir(parents=True, exist_ok=True)
meta = json.load(open(I / "meta.json")); n = meta["n"]; rng = np.random.default_rng(0)
PR = np.load(I / "proj", mmap_mode="r") if (I / "proj").exists() else np.load(I / "proj.npy", mmap_mode="r")    # (n, 2, 25, 256, 64)
ok = np.isfinite(np.asarray(PR[:, 0, 0, 0, 0], dtype=np.float32))


def beta_parts(P, H, idx_pair=None):
    """P: (n, 8, T, d) 예측, H: (n, 8, T, d) 진실 미래 → 슬롯 간 (7) 분자 · 분모 clip 별."""
    dP = np.diff(P, axis=1); dH = np.diff(H, axis=1)                           # (n, 7, T, d)
    dPc = dP - dP.mean(0, keepdims=True); dHc = dH - dH.mean(0, keepdims=True)
    if idx_pair is not None: dHc = dHc[idx_pair]
    num = (dPc * dHc).sum((-1, -2)); den = (dHc ** 2).sum((-1, -2))           # (n, 7)
    return num, den


def boot_ratio(num, den, B=500):
    m = num.sum(0) / den.sum(0); bs = []
    for _ in range(B):
        s = rng.integers(0, len(num), len(num)); bs.append(num[s].sum(0) / den[s].sum(0))
    bs = np.array(bs)
    return np.round(m, 3).tolist(), np.round(np.percentile(bs, 2.5, 0), 3).tolist(), np.round(np.percentile(bs, 97.5, 0), 3).tolist()


# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
# 2 차 적대 검증 (2026-09-25, VERIFY_0925_R2 · 검증자 SSV2T s1/s2/s3) 이 요구한 추가. **기존 키 값은 바꾸지 않고 새 키만 더한다.**
# 기존 rng (seed 0) 의 소비 순서를 건드리지 않도록 rng2 를 따로 쓴다. 새 키는 모두 투영 64 차원 공간 값이다.
#   · ρ / 크기비 / β 의 clip bootstrap CI (B = 1000, 정/역 같은 재표본 → 역 − 정 ρ 차 CI)
#   · 라벨을 맞춘 섞음 귀무 ρ
#   · ρ = 도착 + 이탈 분해 (도착 = ⟨δp_i, h̃_{9+i}⟩, 이탈 = −⟨δp_i, h̃_{8+i}⟩) + 진실 자신의 같은 분해 (기준: 도착 ≈ 이탈)
#   · 매끄러운 진실 추세 오라클의 β / ρ / 크기비 (토큰 Δh 가 슬롯 간 거의 백색이라 크기비에 상한이 없음을 보인다)
#   · 수준 상관 corr(p̃_i, h̃_j) 의 대각 · 행 argmax, p 의 인덱스 구조 (수준 상관 · 변화 자기상관), 중심화가 지우는 변화 에너지 몫
#   · 공간 풀링 척도 사다리 (블록 1/2/4/8/16)
#   · 투영 공간 시간 정렬 (근사 동적 토큰 64 개, 자기 귀무): p · 독립 오차 오라클 · 상수 a 수축 · **L1 을 p 와 맞춘 복사 쪽 수축** (×1.0 / ×1.1) · 진실 선형 추세
r4 = lambda x: np.round(np.asarray(x, dtype=float), 4).tolist()
rng2 = np.random.default_rng(20260925)
LAB = np.asarray(meta["labels"])[ok]
M_OK = int(ok.sum())
BI = [rng2.integers(0, M_OK, M_OK) for _ in range(1000)]
STORE = {}
DP_FULL = np.load(I / "D_p.npy")


def parts64(P, H, perm=None):
    """clip 별 (n, 7) float64: ⟨δp, Δh⟩, ‖δp‖², ‖Δh‖²  (beta_parts 와 같은 중심화: 방향 · 슬롯 · 토큰 · 차원마다 clip 평균을 뺀 슬롯 간 차)."""
    dP = np.diff(P, axis=1); dH = np.diff(H, axis=1)
    dP -= dP.mean(0, keepdims=True); dH -= dH.mean(0, keepdims=True)
    if perm is not None: dH = dH[perm]
    ax = tuple(range(2, dP.ndim))
    return (dP * dH).sum(ax, dtype=np.float64), (dP * dP).sum(ax, dtype=np.float64), (dH * dH).sum(ax, dtype=np.float64)


def summ(num, nP, nH):
    """전이별 합의 비: β = Σ⟨δp,Δh⟩/Σ‖Δh‖², ρ = Σ⟨δp,Δh⟩/√(Σ‖δp‖²Σ‖Δh‖²), 크기비 = √(Σ‖δp‖²/Σ‖Δh‖²).  β = ρ × 크기비 (항등)."""
    return dict(beta=num.sum(0) / nH.sum(0), rho=num.sum(0) / np.sqrt(nP.sum(0) * nH.sum(0)), gain=np.sqrt(nP.sum(0) / nH.sum(0)))


def summ_ci(num, nP, nH):
    bs = {k: [] for k in ("beta", "rho", "gain")}; d06 = []
    for b in BI:
        s = summ(num[b], nP[b], nH[b])
        for k in bs: bs[k].append(s[k])
        d06.append(s["rho"][0] - s["rho"][6])
    res = {f"{k}_ci": [r4(np.percentile(v, 2.5, 0)), r4(np.percentile(v, 97.5, 0))] for k, v in bs.items()}
    res["rho0_minus_rho6_ci"] = r4(np.percentile(d06, [2.5, 97.5]))
    return res


def arrive_depart(P, H, with_ci=True):
    """ρ_i = 도착_i + 이탈_i (항등).  h̃, p̃ = clip 평균을 뺀 수준 (슬롯 · 토큰 · 차원마다).  분모 = ‖δp_i‖·‖Δh_i‖ (clip 합).
    도착 = ⟨δp_i, h̃_{9+i}⟩  (다음 진실 슬롯의 고유 내용 쪽으로 가는 성분)
    이탈 = −⟨δp_i, h̃_{8+i}⟩ (지금 진실 슬롯의 고유 내용과의 일치를 떠나는 성분)
    진실을 그대로 따르는 예측 (p = h) 은 도착 ≈ 이탈 (기준 truth_self)."""
    Pc = P - P.mean(0, keepdims=True); Hc = H - H.mean(0, keepdims=True)
    dP = np.diff(Pc, axis=1); dH = np.diff(Hc, axis=1)
    na = np.stack([(dP[:, i] * Hc[:, i + 1]).sum((1, 2), dtype=np.float64) for i in range(7)], 1)
    nd = np.stack([-(dP[:, i] * Hc[:, i]).sum((1, 2), dtype=np.float64) for i in range(7)], 1)
    # 강건성: h̃ = clip 공통 (미래 8 슬롯 평균 h̄) + 슬롯 고유 편차.  공통 항 s = ⟨δp_i, h̄⟩ 은 도착에 +s, 이탈에 −s 로 들어가 합에서 지워진다.
    #   p 가 clip 내용을 슬롯 따라 잃으면 (감쇠) s < 0 이라 '이탈' 로 보인다 → 슬롯 고유 성분만의 도착/이탈도 싣는다.
    Hbar = Hc.mean(1)
    ns = np.stack([(dP[:, i] * Hbar).sum((1, 2), dtype=np.float64) for i in range(7)], 1)
    pp = (dP * dP).sum((2, 3), dtype=np.float64); hh = (dH * dH).sum((2, 3), dtype=np.float64)

    def f(s):
        dn_ = np.sqrt(pp[s].sum(0) * hh[s].sum(0)); return na[s].sum(0) / dn_, nd[s].sum(0) / dn_, ns[s].sum(0) / dn_
    a0, d0, s0 = f(slice(None))
    res = dict(arrive=r4(a0), depart=r4(d0), total=r4(a0 + d0), arrive_share=r4(a0 / (a0 + d0)),
               clip_common=r4(s0), arrive_slotdev=r4(a0 - s0), depart_slotdev=r4(d0 + s0))
    if with_ci:
        bA, bD, bS = zip(*[f(b) for b in BI]); bA, bD, bS = np.array(bA), np.array(bD), np.array(bS)
        ci_ = lambda X: [r4(np.percentile(X, 2.5, 0)), r4(np.percentile(X, 97.5, 0))]
        res.update(arrive_ci=ci_(bA), depart_ci=ci_(bD), clip_common_ci=ci_(bS), arrive_slotdev_ci=ci_(bA - bS), depart_slotdev_ci=ci_(bD + bS))
    return res


def slot_corr(X, within_clip=False):
    """슬롯 × 슬롯 상관 (clip 평균을 뺀 뒤; within_clip 이면 clip 안 슬롯 평균도 뺀다)."""
    Xc = X - X.mean(0, keepdims=True)
    if within_clip: Xc = Xc - Xc.mean(1, keepdims=True)
    k = X.shape[1]; nr = np.sqrt([(Xc[:, i] * Xc[:, i]).sum(dtype=np.float64) for i in range(k)])
    return np.array([[(Xc[:, i] * Xc[:, j]).sum(dtype=np.float64) / (nr[i] * nr[j]) for j in range(k)] for i in range(k)])


def linear_trend(H):
    """진실 미래 8 슬롯에 clip · 토큰 · 차원마다 직선을 맞춘 오라클 (진전은 완벽, 슬롯 고유 떨림은 없음)."""
    t = np.arange(H.shape[1], dtype=np.float32) - (H.shape[1] - 1) / 2
    b = np.einsum('i,nitd->ntd', t, H) / (t ** 2).sum()
    return H.mean(1, keepdims=True) + t[None, :, None, None] * b[:, None]


def token_extras(A, H, P, HC, di):
    rec = {}
    num, nP, nH = parts64(P, H); STORE[len(STORE)] = (num, nP, nH)
    s = summ(num, nP, nH)
    rec["token|rho_gain_beta"] = {k: r4(v) for k, v in s.items()}
    rec["token|ci_B1000"] = summ_ci(num, nP, nH)
    rec["token|identity_beta_minus_rho_x_gain_maxabs"] = float(np.abs(s["beta"] - s["rho"] * s["gain"]).max())
    perm = np.arange(len(A))
    for L in np.unique(LAB):
        ix = np.where(LAB == L)[0]; perm[ix] = rng2.permutation(ix)
    rec["token|rho_null_label_matched"] = r4(summ(*parts64(P, H, perm))["rho"])
    # 슬롯 범위 합 (기존 p_slots4_7 은 전이 3→4 를 포함한다 — num[:, 3:]; 이름과 맞는 것은 아래)
    rec["token|beta_transitions_within_slots4_7"] = round(float(num[:, 4:].sum() / nH[:, 4:].sum()), 4)
    rec["token|beta_transition_3to4"] = round(float(num[:, 3].sum() / nH[:, 3].sum()), 4)
    # 도착 / 이탈 분해 + 기준
    rec["arrive_depart|p"] = arrive_depart(P, H)
    rec["arrive_depart|truth_self"] = arrive_depart(H, H, with_ci=False)
    OL = linear_trend(H)                    # (추세 오라클의 도착/이탈 분해는 δ 가 슬롯마다 같아 기준으로 읽을 수 없어 싣지 않는다)
    rec["ceiling|oracle_linear_trend"] = {k: r4(v) for k, v in summ(*parts64(OL, H)).items()}
    del OL
    # 수준 상관 (인덱스 서명 제거 = clip 평균 빼기): p_i 대 h_j (j = 0..15), hc_7
    Pc = P - P.mean(0, keepdims=True); Hall = A[:, 0:16] - A[:, 0:16].mean(0, keepdims=True); HCc = HC[:, 0] - HC[:, 0].mean(0, keepdims=True)
    nPi = np.sqrt([(Pc[:, i] * Pc[:, i]).sum(dtype=np.float64) for i in range(8)]); nHj = np.sqrt([(Hall[:, j] * Hall[:, j]).sum(dtype=np.float64) for j in range(16)])
    C = np.array([[(Pc[:, i] * Hall[:, j]).sum(dtype=np.float64) / (nPi[i] * nHj[j]) for j in range(16)] for i in range(8)])
    nHC = np.sqrt((HCc * HCc).sum(dtype=np.float64))
    rec["level_corr|p_vs_h"] = r4(C)
    rec["level_corr|diag_p_i_h_8+i"] = r4([C[i, 8 + i] for i in range(8)])
    rec["level_corr|argmax_j"] = [int(C[i].argmax()) for i in range(8)]
    rec["level_corr|p_vs_hc7"] = r4([(Pc[:, i] * HCc).sum(dtype=np.float64) / (nPi[i] * nHC) for i in range(8)])
    del Pc, Hall, HCc
    # 인덱스 구조: p 와 진실의 수준 상관 (clip 안 슬롯 평균도 뺌), 변화 자기상관, 중심화가 지우는 변화 에너지 몫
    Sp, Sh = slot_corr(P, True), slot_corr(H, True)
    dPc, dHc = slot_corr(np.diff(P, axis=1)), slot_corr(np.diff(H, axis=1))
    rec["index_structure|p_level_corr_within_clip"] = r4(Sp); rec["index_structure|h_level_corr_within_clip"] = r4(Sh)
    rec["index_structure|p_change_autocorr"] = r4(dPc); rec["index_structure|h_change_autocorr"] = r4(dHc)
    rec["index_structure|h_change_lag1"] = r4([dHc[i, i + 1] for i in range(6)])
    rec["index_structure|p_change_lag3"] = r4([dPc[i, i + 3] for i in range(4)])
    dPr, dHr = np.diff(P, axis=1), np.diff(H, axis=1)
    rec["index_structure|frac_change_energy_removed_by_centering"] = dict(p=r4(1 - nP.sum(0) / (dPr * dPr).sum((0, 2, 3), dtype=np.float64)),
                                                                       h=r4(1 - nH.sum(0) / (dHr * dHr).sum((0, 2, 3), dtype=np.float64)))
    del dPr, dHr
    # 공간 풀링 척도 사다리 (토큰 순서 = 16×16 행 우선)
    lad = {}
    for k in (1, 2, 4, 8, 16):
        g = 16 // k
        Pk = P.reshape(len(P), 8, g, k, g, k, 64).mean((3, 5)).reshape(len(P), 8, g * g, 64)
        Hk = H.reshape(len(H), 8, g, k, g, k, 64).mean((3, 5)).reshape(len(H), 8, g * g, 64)
        nu, pP, pH = parts64(Pk, Hk); ss = summ(nu, pP, pH)
        lad[f"block{k}"] = dict({kk: r4(v) for kk, v in ss.items()}, beta_all_transitions=round(float(nu.sum() / pH.sum()), 4),
                                truth_change_lag1=r4([slot_corr(np.diff(Hk, axis=1))[i, i + 1] for i in range(6)]))
    rec["scale_ladder"] = lad
    rec["align_proj"] = align_proj(A, P, H, HC, di)
    return rec


def align_proj(A, P, H, HC, di):
    """투영 공간 시간 정렬. 근사 동적 토큰 = 투영 h 16 슬롯 시간 분산 상위 64 개 (전체 차원 D_p 의 동적 토큰과 다를 수 있다).
    D[i, j] = mean |x_i − h_j| (64 토큰 × 64 차원). 귀무 = **각 예측 자신의** x_A 대 다른 clip h_B (A 의 토큰 집합, 같은 방향).
    보정 argmin = argmin_j (D − E_clip[D_null]) → 대각 (j = 8+i) 비율."""
    m = len(A)
    var = A[:, 0:16].var(1).mean(-1); order = np.argsort(-var, 1)[:, :64]
    take = lambda X: np.take_along_axis(X, order[:, None, :, None], 2)
    Hall = take(A[:, 0:16]); Pd = take(P); Hd = Hall[:, 8:16]; HCd = take(HC)

    def derange():
        d = (np.arange(m) + 1 + rng2.integers(0, m - 1, m)) % m
        return np.where(d == np.arange(m), (d + 1) % m, d)
    der_null, der_err = derange(), derange()
    HB = np.take_along_axis(A[der_null, 0:16], order[:, None, :, None], 2)          # 다른 clip 의 h, A 의 토큰 자리

    def Dmat(X, Hh):
        o = np.empty((m, 8, 16), np.float32)
        for s0 in range(0, m, 40):
            o[s0:s0 + 40] = np.abs(X[s0:s0 + 40, :, None] - Hh[s0:s0 + 40, None]).mean((-1, -2))
        return o

    def run(X):
        D, Dn = Dmat(X, Hall), Dmat(X, HB)
        raw = D.argmin(-1); cor = (D - Dn.mean(0, keepdims=True)).argmin(-1)
        dg = np.stack([D[:, i, 8 + i] for i in range(8)], 1)
        off = np.stack([np.delete(D[:, i], 8 + i, axis=-1).min(-1) for i in range(8)], 1)
        return dict(diag_raw=r4([(raw[:, i] == 8 + i).mean() for i in range(8)]), diag_nullcorr_own=r4([(cor[:, i] == 8 + i).mean() for i in range(8)]),
                    mean_diag_L1=r4(dg.mean(0)), median_rel_margin=r4(np.median((off - dg) / dg, 0)))
    res = {"p": run(Pd)}
    EbA = np.take_along_axis(A[der_err, 16:24] - A[der_err, 8:16], order[:, None, :, None], 2)
    res["oracle_truth_plus_otherclip_error"] = run(Hd + EbA); del EbA
    for a_ in (0.7, 0.75, 0.8):
        res[f"oracle_copy_a{a_}"] = run((1 - a_) * Hd + a_ * HCd)
    # L1 을 p 와 clip · 슬롯마다 정확히 맞춘 복사 쪽 수축: o = (1−a)h_{8+i} + a·hc_7,  a = L1(p_i − h_{8+i}) / L1(hc_7 − h_{8+i})  → L1(o − h) = L1(p − h)
    am = np.abs(Pd - Hd).mean((-1, -2)) / np.abs(HCd - Hd).mean((-1, -2))
    res["matched_a_mean_by_slot"] = r4(am.mean(0)); res["matched_a_quartiles_slot0"] = r4(np.percentile(am[:, 0], [25, 50, 75]))
    for sc in (1.0, 1.1):
        a2 = np.minimum(am * sc, 1.0)[:, :, None, None]
        res[f"oracle_copy_matched_a_x{sc}"] = run((1 - a2) * Hd + a2 * HCd)
    res["oracle_linear_trend"] = run(linear_trend(Hd))
    # 대리 척도 점검: 투영 공간 argmin 이 전체 차원 D_p (참 동적 토큰) argmin 과 같은 비율
    res["p_argmin_agreement_with_fullD_Dp"] = r4((Dmat(Pd, Hall).argmin(-1) == DP_FULL[ok, di].argmin(-1)).mean(0))
    res["_n_tokens"] = 64
    return res


out = {"n_ok": int(ok.sum())}
for di, dn in enumerate(("forward", "reversed")):
    A = np.asarray(PR[ok, di], dtype=np.float32)                              # (m, 25, 256, 64)
    H, P, HC = A[:, 8:16], A[:, 16:24], A[:, 24:25]
    rec = {}
    num, den = beta_parts(P, H); rec["p"] = boot_ratio(num, den)
    perm = rng.permutation(len(A)); num0, den0 = beta_parts(P, H, perm); rec["null_shuffled"] = boot_ratio(num0, den0)
    for a_ in (0.7, 0.8, 0.9):
        O = (1 - a_) * H + a_ * HC; nu, de = beta_parts(O, H); rec[f"oracle_blur_a{a_}"] = boot_ratio(nu, de)
    # 슬롯 4–7 (전이 3..6) 합침
    rec["p_slots4_7"] = round(float(num[:, 3:].sum() / den[:, 3:].sum()), 3)
    rec["p_slots0_3"] = round(float(num[:, :3].sum() / den[:, :3].sum()), 3)
    # β = ρ × (|δp|/|Δh|) 로 분해: 방향 (상관 ρ) 과 크기 (이득)
    dP = np.diff(P, axis=1); dH = np.diff(H, axis=1); dPc = dP - dP.mean(0, keepdims=True); dHc = dH - dH.mean(0, keepdims=True)
    nP = (dPc ** 2).sum((-1, -2)); nH = (dHc ** 2).sum((-1, -2)); nm = (dPc * dHc).sum((-1, -2))
    rec["rho_direction"] = np.round(nm.sum(0) / np.sqrt(nP.sum(0) * nH.sum(0)), 3).tolist()
    rec["gain_magnitude"] = np.round(np.sqrt(nP.sum(0) / nH.sum(0)), 3).tolist()
    rec["rho_null"] = np.round((dPc * dHc[perm]).sum((-1, -2)).sum(0) / np.sqrt(nP.sum(0) * nH.sum(0)), 3).tolist()
    rec["rho_clip_median"] = np.round(np.median(nm / np.sqrt(nP * nH + 1e-12), 0), 3).tolist()
    out[f"beta|{dn}"] = rec
    del dP, dH, dPc, dHc, O
    out[f"token2|{dn}"] = token_extras(A, H, P, HC, di)                        # 2 차 적대 검증 추가 (새 키)
    del A, H, P, HC

# 역재생 − 정방향 ρ 차 (같은 clip 재표본)
(nf, pf, hf), (nr, pr_, hr) = STORE[0], STORE[1]
_d = np.array([summ(nr[b], pr_[b], hr[b])["rho"] - summ(nf[b], pf[b], hf[b])["rho"] for b in BI])
out["token2|rho_reversed_minus_forward"] = dict(est=r4(summ(nr, pr_, hr)["rho"] - summ(nf, pf, hf)["rho"]),
                                                ci=[r4(np.percentile(_d, 2.5, 0)), r4(np.percentile(_d, 97.5, 0))])


def aligned(Dm, Dn):
    """귀무 오프셋 보정 argmin → 대각 비율 (방향별)."""
    res = {}
    for di, dn in enumerate(("forward", "reversed")):
        X = Dm[:, di]; Nn = Dn[:, di]; okr = np.isfinite(X).all((1, 2)) & np.isfinite(Nn).all((1, 2))
        corr = X[okr] - np.nanmean(Nn[okr], 0, keepdims=True)
        js = corr.argmin(-1); raw = X[okr].argmin(-1)
        res[dn] = dict(diag_corrected=np.round([(js[:, i] == 8 + i).mean() for i in range(8)], 3).tolist(),
                       diag_raw=np.round([(raw[:, i] == 8 + i).mean() for i in range(8)], 3).tolist(),
                       mean_j_raw=np.round(raw.mean(0), 2).tolist())
    return res


Dp, Dn = np.load(I / "D_p.npy"), np.load(I / "D_pn.npy")
out["align|p"] = aligned(Dp, Dn); out["align|null"] = aligned(Dn, Dn)
if (I / "D_oa.npy").exists():
    Doa = np.load(I / "D_oa.npy")
    for k_, a_ in enumerate((0.7, 0.8, 0.9)): out[f"align|oracle_blur_a{a_}"] = aligned(Doa[:, :, k_], Dn)
if (I / "D_oe.npy").exists(): out["align|oracle_truth_plus_other_p_error"] = aligned(np.load(I / "D_oe.npy"), Dn)
if (I / "D_oe.npy").exists():   # 전체 차원 대각 여유 (min 대각 밖 − 대각) / 대각 의 clip 중앙값 — 독립 오차 오라클은 대각을 옮기지 못한다
    Doe_ = np.load(I / "D_oe.npy"); okm = np.isfinite(Doe_).all((1, 2, 3))

    def _margin(X):
        return {dn_: r4([float(np.median((np.delete(X[okm, di_, i], 8 + i, axis=-1).min(-1) - X[okm, di_, i, 8 + i]) / X[okm, di_, i, 8 + i]))
                         for i in range(8)]) for di_, dn_ in enumerate(("forward", "reversed"))}
    out["align|diag_rel_margin_median_fullD"] = dict(n=int(okm.sum()), p=_margin(Dp), oracle_truth_plus_other_p_error=_margin(Doe_))
out["align|_note"] = ("align|oracle_* 는 p 의 귀무 오프셋 (D_pn) 으로 보정했다 — 오라클 자신의 귀무가 맞다 (2 차 검증). 자기 귀무 판은 투영 공간 "
                      "token2|<dir>.align_proj. D_pn 은 역재생에도 직전 clip 의 정방향 h 를 쓴다 (B35). "
                      "D_oe 의 오차는 직전 clip 의 정방향 오차라 현재 clip 과 독립 → 정렬 1.00 은 구성상 자명 (diag_raw 도 1.00).")
json.dump(out, open(OUT / f"ssv2_beta{os.environ.get('OUT_TAG', '')}.json", "w"), indent=1, default=float)
print("유효 clip", out["n_ok"])
for dn in ("forward", "reversed"):
    r = out[f"beta|{dn}"]; print(f"\n== β ({dn}) 슬롯 간 0..6   [p 슬롯 0–3 {r['p_slots0_3']} · 4–7 {r['p_slots4_7']}]")
    for k in ("p", "null_shuffled", "oracle_blur_a0.7", "oracle_blur_a0.8", "oracle_blur_a0.9"):
        print(f"  {k:18s} " + " ".join(f"{v:+.3f}" for v in r[k][0]) + f"   CI[{min(r[k][1]):+.3f}, {max(r[k][2]):+.3f}]")
for dn in ("forward", "reversed"):
    r = out[f"beta|{dn}"]
    print(f"\n== β 분해 ({dn}): 방향 ρ {r['rho_direction']}  크기비 {r['gain_magnitude']}  귀무 ρ {r['rho_null']}  clip 별 ρ 중앙값 {r['rho_clip_median']}")
print("\n== 시간 정렬, 귀무 오프셋 보정 대각 비율 (정 / 역)")
for k in [k for k in out if k.startswith("align|") and isinstance(out[k], dict) and "diag_corrected" in out[k].get("forward", {})]:
    print(f"  {k:40s} 정 {out[k]['forward']['diag_corrected']}  역 {out[k]['reversed']['diag_corrected']}")
# 2 차 적대 검증 추가 키
for dn in ("forward", "reversed"):
    t = out[f"token2|{dn}"]; ad = t["arrive_depart|p"]
    print(f"\n== [token2] {dn}")
    print("  ρ / 크기비 / β      ", t["token|rho_gain_beta"])
    print("  ρ CI               ", t["token|ci_B1000"]["rho_ci"], " ρ0−ρ6 CI", t["token|ci_B1000"]["rho0_minus_rho6_ci"])
    print("  라벨 맞춘 귀무 ρ     ", t["token|rho_null_label_matched"])
    print("  도착 / 이탈 (p)     ", ad["arrive"], ad["depart"], " 도착 CI", ad["arrive_ci"])
    print("  도착 몫: p / 진실 자신", ad["arrive_share"], t["arrive_depart|truth_self"]["arrive_share"])
    print("  clip 공통 항 · 슬롯 고유 도착 / 이탈", ad["clip_common"], ad["arrive_slotdev"], ad["depart_slotdev"], " 슬롯 고유 도착 CI", ad["arrive_slotdev_ci"])
    print("  진실 추세 오라클     ", t["ceiling|oracle_linear_trend"])
    print("  수준 상관 대각 · argmax", t["level_corr|diag_p_i_h_8+i"], t["level_corr|argmax_j"])
    print("  진실 Δh lag-1 · p Δ lag-3", t["index_structure|h_change_lag1"], t["index_structure|p_change_lag3"])
    print("  척도 사다리 β (전이 합)", {k: v["beta_all_transitions"] for k, v in t["scale_ladder"].items()}, " ρ0→1", {k: v["rho"][0] for k, v in t["scale_ladder"].items()})
    for k, v in t["align_proj"].items():
        if isinstance(v, dict): print(f"  정렬(투영, 자기 귀무) {k:34s} {v['diag_nullcorr_own']}  raw {v['diag_raw']}")
    print("  matched a", t["align_proj"]["matched_a_mean_by_slot"], " 전체 차원 argmin 일치", t["align_proj"]["p_argmin_agreement_with_fullD_Dp"])
print("\n== [token2] 역 − 정 ρ", out["token2|rho_reversed_minus_forward"])
if "align|diag_rel_margin_median_fullD" in out: print("== 전체 차원 대각 여유 (중앙값)", out["align|diag_rel_margin_median_fullD"])
