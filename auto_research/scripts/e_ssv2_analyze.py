#!/usr/bin/env python3
"""E-SSv2 분석 (계획 PLAN_ENCODER_PREDICTOR_2026-09-25 §6; 적대 검증 VERIFY_0925 반영 2026-09-25). 입력 ssv2_e2/ (또는 ssv2_e/). CPU.

사용: python e_ssv2_analyze.py <입력 dir>   (환경변수 OUT_TAG → exp_results/ssv2/ssv2_e{OUT_TAG}.json;
      SSV2_PIXEL=1|0 → 픽셀 기준선 (원본 mp4 디코드 약 90 초; 기본 = OUT_TAG 가 비었을 때만))

표적: h = LN(target_encoder(32 장)) **표준 양방향 표적**, hc = LN(target_encoder(앞 16 장)) **인과 표적** (복사 기준선 hc_7).
행 = clip × 방향 (정방향 / 역재생). **CI 는 전부 clip 군집 bootstrap (정/역을 같은 clip 으로 묶음, B=1000)** — 2026-09-25 전에는
(clip×방향) 평탄화 행을 재표집했다 (B32). 점추정은 그대로다.

(1) 방향 probe — 로지스틱 (표준화 + L2, torch CPU). 학습 = train clip 의 정방향 + 역재생 (역재생은 라벨 뒤집음 →
    장면 단서로는 못 맞힌다). 시험 = test clip 두 방향. split = clip 단위 층화 7:3 (seed 0; 역재생은 원본과 같은 쪽).
    특징 = 토큰 평균 (feats) 또는 동적 토큰 평균 (featd), 표현 · 슬롯마다 self probe;  이식: h_{8+i} 로 배운 probe → p_i.
    + probe_seeds: split seed 0–4 (z s0/s7, hc s7, p s0/s7, h s15).  + pixel_baseline: 원본 프레임 16×16 RGB (한 시점 · 두 프레임 차 ·
    열 프로파일, 슬롯 0–15, seed 0–2) 와 문맥 16 장 이어 붙임.
(2) 시간 정렬 (토큰 수준, 동적 토큰) — j*(i) = argmin_j D_p[i, j].  ⚠️ 평균 j* 기울기는 요약으로 성립하지 않는다
    (위치 인덱스 별칭 j = 8+i−12 과 argmin 평균의 가운데 회귀). 추가: 슬롯 0–3 기울기 · 열 방향 (진실 슬롯 k 에 가장 가까운 p 슬롯)
    · j* 히스토그램 · 별칭 비율 · 방향별 · **귀무 오프셋 보정** (D_p − E_clip[D_pn], 우연 = 귀무를 같은 오프셋으로 보정).
    zf 대조는 인덱스 정렬 확인일 뿐 (같은 미래 프레임의 encoding) — 양성 대조가 아니다.
(3) **진전 지표 β (풀링, 중심화)** — 주 지표. 인덱스마다 전 clip·방향 평균을 빼고 (clip 과 무관한 위치 서명 제거),
    clip 안 슬롯 평균을 뺀 δ_i (p) · Δ_i (h_{8+i}) 로 β = Σ_i⟨δ_i, Δ_i⟩ / Σ_i‖Δ_i‖² (행마다, clip 평균).
    1 = 진실의 슬롯 간 변화를 그대로 따름, 0 = clip 특이적 변화 없음. 귀무 = 다른 clip 의 Δ. 교정: 복사 쪽으로 a 만큼 흐린
    진전 오라클 (1−a)h_{8+i} + a·hc_7 은 β = 1−a (구성상) 인데 argmin 기울기는 a=0.9 에서 무너진다 → 두 지표를 같이 싣는다.
    정지 내용 감쇠 오라클 · 문맥 span 투영 제거 뒤 β 로 "지평별 흐림" 교란을 본다.
(4) 복사 대비 — EI_i (인과 hc_7 / 새는 h_7), 흐림 기준선 · p 슬롯 평균, 기준선별 재현 비율, **위치 보정 복사** (풀링:
    hc_7 − μ(hc_7) + μ(h_{8+i}), μ = 전 clip·방향 인덱스 평균), p 슬롯 평균 대비 EI 의 오차 맞춤 오라클 (진실 + 다른 clip 의 p 오차).
(5) 변화 지도 일치 ρ_i 대 귀무 (다른 clip, 같은 방향) — ⚠️ 두 지도가 기준 h_7 을 공유해 부풀려진다 (미통제). 주장에 쓰지 않는다.
"""
import json, os, sys, time
from pathlib import Path
import numpy as np
import torch

I = Path(sys.argv[1] if len(sys.argv) > 1 else "/data2/local_datasets/world/world_analysis/cache/auto_research/ssv2_e")
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "auto_research/exp_results/ssv2"; OUT.mkdir(parents=True, exist_ok=True)
TAG = os.environ.get("OUT_TAG", "")
DO_PIX = os.environ.get("SSV2_PIXEL", "0" if TAG else "1") == "1"
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
meta = json.load(open(I / "meta.json")); y0 = np.array(meta["labels"]); n = len(y0)
rng = np.random.default_rng(0)
te = np.zeros(n, bool)
for lab in (0, 1):
    ix = rng.permutation(np.where(y0 == lab)[0]); te[ix[: int(round(0.3 * len(ix)))]] = True
SL = dict(z=(0, 8), h=(8, 24), hc=(24, 32), p=(32, 40), zf=(40, 56))
DIRS = ("forward", "reversed")
AR8 = np.arange(8)
B_BOOT = 1000
_BIDX = {}


def bidx(m):
    """길이 m 의 clip 재표집 인덱스 (B_BOOT, m) — 길이마다 고정 (seed 1), 모든 지표가 같은 표본을 공유."""
    if m not in _BIDX:
        _BIDX[m] = np.random.default_rng(1).integers(0, m, (B_BOOT, m))
    return _BIDX[m]


BIDX = bidx(n)


def _r(x, k=3):
    return np.round(np.asarray(x, float), k).tolist()


def boot(v):
    """clip 군집 bootstrap. v: (n, 2) (clip × 방향) 또는 (n,) → [평균, 2.5 %, 97.5 %]. NaN 행 (귀무 없는 clip) 은 뺀다."""
    v = np.asarray(v, float)
    m = float(np.nanmean(v))
    pc = np.nanmean(v, axis=1) if v.ndim == 2 else v                     # clip 마다 방향 평균
    pc = pc[np.isfinite(pc)]
    bs = pc[bidx(len(pc))].mean(1)
    return [round(m, 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]


def boot_fn(fn, m_idx=None):
    """통계량 fn(clip 인덱스 배열) 의 clip 군집 bootstrap → [점추정, 2.5 %, 97.5 %]."""
    base = np.arange(n) if m_idx is None else m_idx
    est = fn(base)
    bs = np.array([fn(base[b]) for b in bidx(len(base))])
    return [round(float(est), 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]


def by_dir(v):
    """v: (n, 2, …) → {forward: 평균, reversed: 평균} (슬롯 축이 있으면 슬롯별)."""
    v = np.asarray(v, float)
    return {d: _r(np.nanmean(v[:, k], axis=0)) for k, d in enumerate(DIRS)}


def fit_logreg(Xtr, ytr, lam=1e-2, epochs=300):
    Xt = torch.tensor(Xtr, dtype=torch.float32); yt = torch.tensor(ytr, dtype=torch.float32)
    mu, sd = Xt.mean(0), Xt.std(0) + 1e-5; Xt = (Xt - mu) / sd
    w = torch.zeros(Xt.shape[1], requires_grad=True); b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1, max_iter=epochs, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad(); loss = torch.nn.functional.binary_cross_entropy_with_logits(Xt @ w + b, yt) + lam * (w ** 2).sum(); loss.backward(); return loss
    opt.step(closure)
    return lambda X: ((((torch.tensor(X, dtype=torch.float32) - mu) / sd) @ w + b) > 0).numpy().astype(int)


YD = np.stack([y0, 1 - y0], 1)                                          # (n, 2) 방향 라벨 (역재생 뒤집힘)


def split(seed):
    r = np.random.default_rng(seed); t = np.zeros(n, bool)
    for lab in (0, 1):
        ix = r.permutation(np.where(y0 == lab)[0]); t[ix[: int(round(0.3 * len(ix)))]] = True
    return t


assert (split(0) == te).all()


def probe_xy(X, tmask):
    """X (n, 2, d) → [정 acc, 역 acc] (train = ~tmask 두 방향, test = tmask)."""
    g = fit_logreg(X[~tmask].reshape(-1, X.shape[-1]).astype(np.float32), YD[~tmask].reshape(-1))
    return [round(float((g(X[tmask][:, d].astype(np.float32)) == YD[tmask][:, d]).mean()), 3) for d in (0, 1)]


def probe_block(F):
    """F (n, 2, 56, D) → dict rep→slot→{self acc fwd/rev, h→p 이식}."""
    y = YD
    res = {}
    def data(k, mask):
        Xk = F[mask][:, :, k].reshape(-1, F.shape[-1]).astype(np.float32); yk = y[mask].reshape(-1); return Xk, yk
    heads = {}
    for rep, (a, b) in SL.items():
        res[rep] = []
        for s in range(b - a):
            k = a + s; Xtr, ytr = data(k, ~te); g = fit_logreg(Xtr, ytr); heads[(rep, s)] = g
            accs = [float((g(F[te][:, d, k].astype(np.float32)) == y[te][:, d]).mean()) for d in (0, 1)]
            res[rep].append(dict(slot=s, fwd=round(accs[0], 3), rev=round(accs[1], 3)))
    res["h_to_p_centered"] = []
    for i in range(8):                                               # 표현마다 train 평균을 빼고 (전역 치우침 제거) 이식
        kh, kp = SL["h"][0] + 8 + i, SL["p"][0] + i
        Fh = F[:, :, kh].astype(np.float32); Fp = F[:, :, kp].astype(np.float32)
        Fh = Fh - Fh[~te].reshape(-1, Fh.shape[-1]).mean(0); Fp = Fp - Fp[~te].reshape(-1, Fp.shape[-1]).mean(0)
        g = fit_logreg(Fh[~te].reshape(-1, Fh.shape[-1]), y[~te].reshape(-1))
        accs = [float((g(Fp[te][:, d]) == y[te][:, d]).mean()) for d in (0, 1)]
        res["h_to_p_centered"].append(dict(slot=i, fwd=round(accs[0], 3), rev=round(accs[1], 3)))
    res["h_to_p"] = []
    for i in range(8):
        g = heads[("h", 8 + i)]; k = SL["p"][0] + i
        accs = [float((g(F[te][:, d, k].astype(np.float32)) == y[te][:, d]).mean()) for d in (0, 1)]
        res["h_to_p"].append(dict(slot=i, fwd=round(accs[0], 3), rev=round(accs[1], 3)))
    return res


fr = np.array(meta.get("n_frames", [np.nan] * n), float)
out = {"n": n, "n_test": int(te.sum()),
       "meta_summary": dict(n_clips=n, n_left=int((y0 == 0).sum()), n_right=int((y0 == 1).sum()),
                            src_frames_min=float(np.nanmin(fr)), src_frames_median=float(np.nanmedian(fr)), src_frames_max=float(np.nanmax(fr)),
                            eff_fps_at_12fps=[round(32 * 12 / float(np.nanmax(fr)), 2), round(32 * 12 / float(np.nanmin(fr)), 2)],
                            tubelet_sec_at_12fps=[round(float(np.nanmin(fr)) / 16 / 12, 3), round(float(np.nanmedian(fr)) / 16 / 12, 3), round(float(np.nanmax(fr)) / 16 / 12, 3)],
                            bootstrap="clip 군집 (정/역 묶음), B=1000", targets="h = 표준 양방향 표적 (32 장), hc = 인과 표적 (앞 16 장)")}
t0 = time.time()
for nm in ("feats", "featd"):
    F = np.load(I / f"{nm}.npy")
    if np.isnan(F.astype(np.float32)).any(): print(nm, "nan 있음 — 추출 미완?"); continue
    out[f"probe_{nm}"] = probe_block(F)
    # split seed 0–4 재현 (대표 슬롯)
    ps = {}
    for rep, k in (("z_s0", 0), ("z_s7", 7), ("hc_s7", 31), ("p_s0", 32), ("p_s7", 39), ("h_s15", 23)):
        ps[rep] = [probe_xy(F[:, :, k].astype(np.float32), split(sd)) for sd in range(5)]
    out[f"probe_seeds_{nm}"] = ps
    del F
print(f"[probe {time.time()-t0:.0f}s]", flush=True)

# ---------------------------------------------------------------- (2) 시간 정렬 (토큰 수준, 동적 토큰, 표준 양방향 h)
Dp = np.load(I / "D_p.npy"); Dz = np.load(I / "D_z.npy"); Dn = np.load(I / "D_pn.npy"); Dpa = np.load(I / "D_pa.npy")
ta = {}
EXTRA = [(nm, np.load(I / f"{f}.npy")) for nm, f in (("oracle_blur", "D_o1"), ("oracle_half_mean", "D_o2")) if (I / f"{f}.npy").exists()]


def slope(y, sl=slice(0, 8)):
    return float(np.polyfit(AR8[sl], np.asarray(y)[sl], 1)[0])


def align_extra(D4):
    """D4 (n, 2, 8, 16) → 추가 필드 (clip 군집 CI, 방향별, 히스토그램, 별칭, 열 방향)."""
    okc = np.isfinite(D4).all((1, 2, 3)); cl = np.where(okc)[0]
    js = D4.argmin(-1) if okc.all() else np.where(okc[:, None, None], np.nan_to_num(D4, nan=np.inf).argmin(-1), -1)
    ic = np.nan_to_num(D4[..., 8:], nan=np.inf).argmin(2)             # (n, 2, 8) 진실 슬롯 k 마다 가장 가까운 p 슬롯
    curve = lambda ix: js[ix].reshape(-1, 8).mean(0)
    ccurve = lambda ix: ic[ix].reshape(-1, 8).mean(0)
    J = js[cl]
    r = dict(n_clips=int(okc.sum()),
             slope_ci=boot_fn(lambda ix: slope(curve(ix)), cl),
             slope_slots0_3=boot_fn(lambda ix: slope(curve(ix), slice(0, 4)), cl),
             frac_pm1=_r([(np.abs(J[:, :, i] - (8 + i)) <= 1).mean() for i in range(8)]),
             frac_context_early=_r([(J[:, :, i] <= 6).mean() for i in range(8)]),
             frac_alias_minus12_slots4_7=_r([(J[:, :, i] == 8 + i - 12).mean() for i in range(4, 8)]),
             hist=[_r(np.bincount(J[:, :, i].ravel(), minlength=16) / J[:, :, i].size, 3) for i in range(8)],
             column_mean_istar=_r(ic[cl].reshape(-1, 8).mean(0), 2),
             column_diag=_r([(ic[cl][:, :, k] == k).mean() for k in range(8)]),
             column_slope=boot_fn(lambda ix: slope(ccurve(ix)), cl),
             frac_diag_ci=[boot_fn(lambda ix, i=i: (js[ix][:, :, i] == 8 + i).mean(), cl) for i in range(8)])
    r["by_direction"] = {}
    for k, d in enumerate(DIRS):
        Jd = J[:, k]; Icd = ic[cl][:, k]
        r["by_direction"][d] = dict(mean_jstar=_r(Jd.mean(0), 2), frac_diag=_r([(Jd[:, i] == 8 + i).mean() for i in range(8)]),
                                    frac_boundary=_r([np.isin(Jd[:, i], (7, 8)).mean() for i in range(8)]),
                                    slope=round(slope(Jd.mean(0)), 3), slope_slots0_3=round(slope(Jd.mean(0), slice(0, 4)), 3),
                                    column_slope=round(slope(Icd.mean(0)), 3), column_diag=_r([(Icd[:, q] == q).mean() for q in range(8)]))
    return r


for nm, Dm in [("p_dyn", Dp), ("p_all", Dpa), ("zf_control", Dz), ("p_null", Dn)] + EXTRA:
    Dr = Dm.reshape(-1, 8, 16); Dr = Dr[np.isfinite(Dr).all((1, 2))]
    js = Dr.argmin(-1)                                               # (유효 행, 8)
    ta[nm] = dict(mean_jstar=[round(float(np.nanmean(js[:, i])), 2) for i in range(8)],
                  frac_diag=[round(float(np.nanmean(np.abs(js[:, i] - (8 + i)) <= 0)), 3) for i in range(8)],
                  frac_boundary=[round(float(np.nanmean(np.isin(js[:, i], (7, 8)))), 3) for i in range(8)],
                  slope=round(float(np.polyfit(np.arange(8), np.nanmean(js, 0), 1)[0]), 3))
    ta[nm].update(align_extra(Dm))
ta["zf_control"]["note"] = "인덱스 정렬 확인 (같은 미래 프레임의 encoding) — 양성 대조 아님"
zdiag = np.stack([Dz[:, :, i, 8 + i] for i in range(8)], -1); zoff = np.stack([np.delete(Dz[:, :, i], 8 + i, -1).min(-1) for i in range(8)], -1)
pdiag = np.stack([Dp[:, :, i, 8 + i] for i in range(8)], -1); poff = np.stack([np.delete(Dp[:, :, i], 8 + i, -1).min(-1) for i in range(8)], -1)
ta["diag_over_min_offdiag"] = dict(zf=_r((zdiag / zoff).mean((0, 1))), p=_r((pdiag / poff).mean((0, 1))))
# min 거리 대 귀무 (같은 clip 의 h 에 더 가까운가). 귀무 B = 같은 rank 가 바로 앞에 처리한 clip (정렬상 대부분 같은 라벨), 역재생 A 에는 B 의 정방향 h.
okn = np.isfinite(Dn).all((1, 2, 3))
mn_p = np.where(okn[:, None, None], Dp.min(-1), np.nan); mn_n = np.where(okn[:, None, None], Dn.min(-1), np.nan)
ta["min_D_vs_null"] = [boot(np.where(okn[:, None], (mn_p[:, :, i] < mn_n[:, :, i]).astype(float), np.nan)) for i in range(8)]
# 귀무 오프셋 보정 (D_p − 귀무 평균 [i, j], 정/역 합쳐 한 오프셋) → argmin. 우연 = 귀무를 같은 오프셋으로 보정.
off = Dn[okn].mean((0, 1))                                            # (8, 16) clip 과 무관한 인덱스 구조
tac = {"offset": "E_clip,dir[D_pn]  (정/역 합침, 귀무 clip 669)"}
for nm, Dm in (("p_dyn", Dp), ("p_null_chance", Dn)):
    Dc = Dm - off
    r = align_extra(Dc)
    okc = np.isfinite(Dc).all((1, 2, 3)); J = Dc[okc].argmin(-1)
    r.update(mean_jstar=_r(J.reshape(-1, 8).mean(0), 2), frac_diag=_r([(J[:, :, i] == 8 + i).mean() for i in range(8)]),
             slope=round(slope(J.reshape(-1, 8).mean(0)), 3))
    tac[nm] = r
ta["null_offset_corrected"] = tac
out["time_alignment"] = ta
print(f"[align {time.time()-t0:.0f}s]", flush=True)

# ---------------------------------------------------------------- (4) 복사 대비 (토큰 수준, 동적 토큰)
L1 = np.load(I / "l1.npy")                                              # (n, 2, 8, 6)
EIc = 1 - L1[..., 0] / L1[..., 2]
out["EI_causal_dyn"] = [boot(EIc[:, :, i]) for i in range(8)]
out["EI_leaky_dyn"] = [boot(1 - L1[:, :, i, 0] / L1[:, :, i, 1]) for i in range(8)]
out["EI_causal_all"] = [boot(1 - L1[:, :, i, 4] / L1[:, :, i, 5]) for i in range(8)]
out["p_beats_causal_copy_dyn"] = [boot((L1[:, :, i, 0] < L1[:, :, i, 2]).astype(float)) for i in range(8)]
out["EI_causal_dyn_slotmean"] = boot(EIc.mean(2))
bd = {"EI_causal_dyn": by_dir(EIc), "EI_leaky_dyn": by_dir(1 - L1[..., 0] / L1[..., 1])}
# 변화 지도 일치 (보조; 공유 기준 h_7 인공물 미통제)
CM = np.load(I / "cmap.npy").astype(np.float32)                        # (n, 2, 3, 8, 256)
prng = np.random.default_rng(2); order = prng.permutation(n); perm = np.empty(n, int); perm[order] = order[np.roll(np.arange(n), 1)]   # 고정점 없는 순열


def corr(a, b):
    a = a - a.mean(-1, keepdims=True); b = b - b.mean(-1, keepdims=True)
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-9)


out["rho_p_vs_true"] = [boot(corr(CM[:, :, 1, i], CM[:, :, 0, i])) for i in range(8)]
out["rho_null"] = [boot(corr(CM[:, :, 1, i], CM[perm][:, :, 0, i])) for i in range(8)]
out["rho_pc_vs_true"] = [boot(corr(CM[:, :, 2, i], CM[:, :, 0, i])) for i in range(8)]
if (I / "l1b.npy").exists():
    LB = np.load(I / "l1b.npy")                                         # (n, 2, 8, 8)
    share = {}
    for j, nm in enumerate(("ctxmean", "blur3x3", "mix", "pmean")):
        out[f"EI_vs_{nm}_dyn"] = [boot(1 - L1[:, :, i, 0] / LB[:, :, i, 2 * j]) for i in range(8)]
        out[f"p_beats_{nm}_dyn"] = [boot((L1[:, :, i, 0] < LB[:, :, i, 2 * j]).astype(float)) for i in range(8)]
        bd[f"EI_vs_{nm}_dyn"] = by_dir(1 - L1[..., 0] / LB[..., 2 * j])
        bd[f"p_beats_{nm}_dyn"] = by_dir((L1[..., 0] < LB[..., 2 * j]).astype(float))
        gb = 1 - LB[..., 2 * j] / L1[..., 2]                              # 기준선 b 의 인과 복사 대비 이득
        share[nm] = dict(gain_over_causal_copy=boot(gb.mean(2)), share_of_p_gain=round(float(gb.mean() / EIc.mean()), 3),
                         share_per_slot=_r(gb.mean((0, 1)) / EIc.mean((0, 1)), 2))
    out["baseline_share_of_EI_causal"] = share
    CB = np.load(I / "cmapb.npy").astype(np.float32)                    # (n, 2, 2, 256)
    out["rho_lastmotion_vs_true"] = [boot(corr(CB[:, :, 1], CM[:, :, 0, i])) for i in range(8)]
    out["rho_lastmotion_causal_vs_true"] = [boot(corr(CB[:, :, 0], CM[:, :, 0, i])) for i in range(8)]
out["by_direction"] = bd

# ---------------------------------------------------------------- (3) 진전 지표 β (풀링, 중심화) + 교정 · 위치 보정 복사
def pooled_D(X, Hh):
    """X (n, 2, k, d), Hh (n, 2, 16, d) → (n, 2, k, 16) 풀링 L1."""
    D_ = np.empty(X.shape[:3] + (Hh.shape[2],), np.float32)
    for i in range(X.shape[2]):
        D_[:, :, i] = np.abs(X[:, :, i, None] - Hh).mean(-1)
    return D_


def argmin_summary(D_):
    js = D_.argmin(-1); mj = js.reshape(-1, 8).mean(0)
    return dict(mean_jstar=_r(mj, 2), slope=round(slope(mj), 3), slope_slots0_3=round(slope(mj, slice(0, 4)), 3),
                frac_diag=_r([(js[:, :, i] == 8 + i).mean() for i in range(8)], 2),
                slope_by_direction={d: round(slope(js[:, k].mean(0)), 3) for k, d in enumerate(DIRS)})


def beta_rows(X, Y):
    """δ = X − 슬롯 평균, Δ = Y − 슬롯 평균 (행마다) → β (n, 2) = Σ⟨δ, Δ⟩ / Σ‖Δ‖²."""
    dX = X - X.mean(2, keepdims=True); dY = Y - Y.mean(2, keepdims=True)
    return (dX * dY).sum((2, 3)) / (dY * dY).sum((2, 3))


def beta_report(X, Y):
    r = {}
    for sn, sl in (("slots0_7", slice(0, 8)), ("slots0_3", slice(0, 4)), ("slots4_7", slice(4, 8))):
        b = beta_rows(X[:, :, sl], Y[:, :, sl])
        r[sn] = boot(b); r[f"{sn}_by_direction"] = {d: boot(b[:, k]) for k, d in enumerate(DIRS)}
    return r


def beta_transitions(X, Y):
    """연속 슬롯 차 δ_t = X_{t+1} − X_t 에 대한 β_t (합의 비, t = 0..6) — 슬롯별 진전 윤곽."""
    dX = np.diff(X, axis=2); dY = np.diff(Y, axis=2)
    num = (dX * dY).sum(-1); den = (dY * dY).sum(-1)                   # (n, 2, 7)
    est = num.sum((0, 1)) / den.sum((0, 1))
    pcn, pcd = num.sum(1), den.sum(1)
    bs = np.array([pcn[b].sum(0) / pcd[b].sum(0) for b in BIDX])
    return dict(est=_r(est), lo=_r(np.percentile(bs, 2.5, 0)), hi=_r(np.percentile(bs, 97.5, 0)))


def perp(X, basis):
    """clip·방향마다 basis (n, 2, k, D) 의 span 을 X (n, 2, m, D) 에서 제거."""
    Q = np.linalg.qr(np.swapaxes(basis, -1, -2))[0]                     # (n, 2, D, k)
    return X - np.einsum('abdk,abik->abid', Q, np.einsum('abid,abdk->abik', X, Q))


bp, cp, eo = {}, {}, {}
for fname in ("featd", "feats"):
    F = np.load(I / f"{fname}.npy").astype(np.float32)
    Z, H, HC, P, ZF = F[:, :, 0:8], F[:, :, 8:24], F[:, :, 24:32], F[:, :, 32:40], F[:, :, 40:56]
    del F
    cen = lambda X: X - X.mean((0, 1), keepdims=True)
    Pc, Hc_, HCc, ZFc = cen(P), cen(H), cen(HC), cen(ZF)
    HF, ZFF, C7, CMn = Hc_[:, :, 8:], ZFc[:, :, 8:], HCc[:, :, 7], HCc.mean(2)
    prev = np.roll(np.arange(n), 1)                                    # 직전 clip (정렬상 대부분 같은 라벨)
    r = {"p": beta_report(Pc, HF), "null_prev_clip": beta_report(Pc, HF[prev]), "null_random_clip": beta_report(Pc, HF[perm]),
         "zf": beta_report(ZFF, HF), "p_reversed_slot_order": beta_report(Pc, HF[:, :, ::-1])}
    r["transitions"] = dict(p=beta_transitions(Pc, HF), null_random_clip=beta_transitions(Pc, HF[perm]), zf=beta_transitions(ZFF, HF))
    g = [float((Pc[:, :, i] * C7).sum() / (C7 * C7).sum()) for i in range(8)]
    r["gamma_p_on_c7"] = _r(g)
    r["oracle_static_decay_gamma"] = boot(beta_rows(np.stack([g[i] * C7 for i in range(8)], 2), HF))
    r["oracle_static_decay_linear"] = boot(beta_rows(np.stack([(1 - (0.3 + 0.6 * i / 7)) * C7 for i in range(8)], 2), HF))
    proj = {}
    for bn, basis in (("c7", C7[:, :, None]), ("c7_ctxmean", np.stack([C7, CMn], 2)), ("all8_hc", HCc)):
        dpp, dhh, dzz = perp(Pc - Pc.mean(2, keepdims=True), basis), perp(HF - HF.mean(2, keepdims=True), basis), perp(ZFF - ZFF.mean(2, keepdims=True), basis)
        late_p, late_h = perp(Pc[:, :, 4:] - Pc[:, :, 4:].mean(2, keepdims=True), basis), perp(HF[:, :, 4:] - HF[:, :, 4:].mean(2, keepdims=True), basis)
        proj[bn] = dict(p=boot(beta_rows(dpp, dhh)), p_slots4_7=boot(beta_rows(late_p, late_h)), zf=boot(beta_rows(dzz, dhh)),
                        null_random_clip=boot(beta_rows(dpp, dhh[perm])))
    r["after_projecting_out_context"] = proj
    # 교정: argmin 기울기 (풀링, 원 공간) 대 β — 흐려도 진전하는 오라클
    HFr = H[:, :, 8:]; hbar = HFr.mean(2, keepdims=True)
    cal = {"p_raw": dict(argmin=argmin_summary(pooled_D(P, H)), beta=r["p"]["slots0_7"]),
           "p_centered_argmin": argmin_summary(pooled_D(Pc, Hc_)),
           "zf_raw": dict(argmin=argmin_summary(pooled_D(ZF[:, :, 8:], H)))}
    for a_ in (0.5, 0.7, 0.8, 0.9):
        O = (1 - a_) * HFr + a_ * HC[:, :, 7:8]
        cal[f"oracle_copy_blur_a{a_}"] = dict(argmin=argmin_summary(pooled_D(O, H)), beta=boot(beta_rows(cen(O), HF)))
    for a_ in (0.8, 0.9):
        O = (1 - a_) * HFr + a_ * hbar
        cal[f"oracle_futmean_blur_a{a_}"] = dict(argmin=argmin_summary(pooled_D(O, H)), beta=boot(beta_rows(cen(O), HF)))
    E = P - HFr; Oe = HFr + E[np.roll(np.arange(n), 3)]                 # 진실 + 다른 clip 의 실제 p 오차 (오차 크기를 맞춘 완벽 진전)
    cal["oracle_truth_plus_other_p_error"] = dict(argmin=argmin_summary(pooled_D(Oe, H)), beta=boot(beta_rows(cen(Oe), HF)))
    r["calibration"] = cal
    bp[fname] = r
    # p 슬롯 평균 대비 EI (풀링) — p 와 오차 맞춤 오라클
    ei_own = lambda X: 1 - np.abs(X - HFr).mean(-1) / np.abs(X.mean(2, keepdims=True) - HFr).mean(-1)
    eo[fname] = dict(p=_r(ei_own(P).mean((0, 1))), p_by_direction=by_dir(ei_own(P)),
                     oracle_truth_plus_other_p_error=_r(ei_own(Oe).mean((0, 1))), oracle_by_direction=by_dir(ei_own(Oe)))
    # 위치 보정 복사 (풀링): 인덱스 평균 차만큼 옮긴 복사
    muH, muHC = H.mean((0, 1)), HC.mean((0, 1))
    L = lambda X: np.abs(X - HFr).mean(-1)                               # (n, 2, 8)
    Lp = L(P); pcorr = HC[:, :, 7:8] - muHC[7] + muH[8:]; cmc = HC.mean(2, keepdims=True) - muHC.mean(0) + muH[8:]
    rows = {"p": Lp, "hc7_copy": L(HC[:, :, 7:8]), "hc7_poscorr": L(pcorr), "ctxmean": L(HC.mean(2, keepdims=True)),
            "ctxmean_poscorr": L(cmc), "h7_copy_leaky": L(H[:, :, 7:8]), "h7_poscorr_leaky": L(H[:, :, 7:8] - muH[7] + muH[8:])}
    cp[fname] = {"L1": {k: _r(v.mean((0, 1))) for k, v in rows.items()},
                 "EI_p_vs_hc7_raw": _r((1 - Lp / rows["hc7_copy"]).mean((0, 1))),
                 "EI_p_vs_hc7_poscorr": _r((1 - Lp / rows["hc7_poscorr"]).mean((0, 1))),
                 "EI_p_vs_hc7_poscorr_ci": [boot(1 - Lp[:, :, i] / rows["hc7_poscorr"][:, :, i]) for i in range(8)],
                 "win_p_vs_hc7_poscorr": _r((Lp < rows["hc7_poscorr"]).mean((0, 1))),
                 "win_p_vs_ctxmean_poscorr": _r((Lp < rows["ctxmean_poscorr"]).mean((0, 1))),
                 "EI_p_vs_hc7_poscorr_by_direction": by_dir(1 - Lp / rows["hc7_poscorr"]),
                 "note": "μ = 전 clip·방향 인덱스 평균 (자기 기여 1/1354). 풀링 수준 — 토큰 수준은 ssv2_e3 (후속)"}
    del Z, H, HC, P, ZF, Pc, Hc_, HCc, ZFc
    print(f"[beta {fname} {time.time()-t0:.0f}s]", flush=True)
out["beta_pooled"] = bp
out["copy_poscorr_pooled"] = cp
out["EI_vs_own_slotmean_pooled"] = eo

# ---------------------------------------------------------------- (1b) 픽셀 기준선 (원본 mp4, 16×16 RGB)
if DO_PIX:
    import decord
    import torch.nn.functional as Fnn
    PIX = np.zeros((n, 32, 3, 16, 16), np.float32); fps = np.zeros(n)
    for r_, f in enumerate(meta["files"]):
        vr = decord.VideoReader(f, num_threads=4); fps[r_] = vr.get_avg_fps()
        idx = np.linspace(0, len(vr) - 1, 32).round().astype(int)
        x = torch.from_numpy(vr.get_batch(list(idx)).asnumpy()).permute(0, 3, 1, 2).float() / 255.
        PIX[r_] = Fnn.adaptive_avg_pool2d(x, 16).numpy()
    PR = PIX[:, ::-1]                                                   # 역재생
    pix = {"fps": dict(min=round(float(fps.min()), 2), median=round(float(np.median(fps)), 2), max=round(float(fps.max()), 2)), "slots": {}}
    for s in range(16):
        fr_ = np.stack([PIX[:, 2 * s:2 * s + 2].mean(1).reshape(n, -1), PR[:, 2 * s:2 * s + 2].mean(1).reshape(n, -1)], 1)
        a2 = min(2 * s + 2, 31)
        df_ = np.stack([(PIX[:, a2] - PIX[:, 2 * s]).reshape(n, -1), (PR[:, a2] - PR[:, 2 * s]).reshape(n, -1)], 1)
        cl_ = np.stack([PIX[:, 2 * s:2 * s + 2].mean(1).mean(-2).reshape(n, -1), PR[:, 2 * s:2 * s + 2].mean(1).mean(-2).reshape(n, -1)], 1)
        pix["slots"][s] = {k: [probe_xy(X, split(sd)) for sd in range(3)] for k, X in (("frame", fr_), ("diff2", df_), ("column_profile", cl_))}
    Xc = np.stack([PIX[:, :16].reshape(n, -1), PR[:, :16].reshape(n, -1)], 1)
    pix["context16_stacked"] = [probe_xy(Xc, split(sd)) for sd in range(3)]
    rng_ = lambda k: [round(min(min(v) for s in pix["slots"].values() for v in s[k]), 3), round(max(max(v) for s in pix["slots"].values() for v in s[k]), 3)]
    pix["range"] = {k: rng_(k) for k in ("frame", "diff2", "column_profile")}
    pix["range"]["context16_stacked"] = [min(min(v) for v in pix["context16_stacked"]), max(max(v) for v in pix["context16_stacked"])]
    out["pixel_baseline"] = pix
    print(f"[pixel {time.time()-t0:.0f}s]", flush=True)

json.dump(out, open(OUT / f"ssv2_e{TAG}.json", "w"), indent=1, default=float)

# ---------------------------------------------------------------- 출력
ms = out["meta_summary"]; print(f"\nclip {ms['n_clips']} (left {ms['n_left']} / right {ms['n_right']}), 원본 {ms['src_frames_min']:.0f}–{ms['src_frames_max']:.0f} 프레임 (중앙 {ms['src_frames_median']:.0f})")
for nm in ("feats", "featd"):
    if f"probe_{nm}" not in out: continue
    pr = out[f"probe_{nm}"]; print(f"\n== 방향 probe ({nm}) — test 정확도 정/역 (split seed 0)")
    for rep in ("z", "hc", "h", "p", "zf", "h_to_p", "h_to_p_centered"):
        print(f"{rep:7s} " + " ".join(f"{r['fwd']:.2f}/{r['rev']:.2f}" for r in pr[rep]))
    print("  seeds 0–4:", {k: " ".join(f"{a:.2f}/{b:.2f}" for a, b in v) for k, v in out[f"probe_seeds_{nm}"].items()})
if "pixel_baseline" in out:
    print("\n== 픽셀 기준선 범위 (정/역 · 슬롯 0–15 · seed 0–2)", out["pixel_baseline"]["range"], "fps", out["pixel_baseline"]["fps"])
print("\n== 시간 정렬 (토큰 수준, 동적 토큰, 표준 양방향 h; 슬롯 i → j*; 진전이면 8+i)")
for nm in ("p_dyn", "p_all", "zf_control", "p_null") + tuple(k for k, _ in EXTRA):
    r = ta[nm]
    print(f"{nm:11s} j* {r['mean_jstar']}  대각 {r['frac_diag']}  경계(7,8) {r['frac_boundary']}  기울기 {r['slope']} CI {r['slope_ci']}  슬롯0–3 {r['slope_slots0_3']}  열 {r['column_slope']}")
    print(f"{'':11s} 방향별 기울기 " + " ".join(f"{d} {v['slope']:+.3f}" for d, v in r["by_direction"].items()) + f"   별칭 j=8+i−12 (슬롯4–7) {r['frac_alias_minus12_slots4_7']}")
print("min D < 귀무", [r[0] for r in ta["min_D_vs_null"]])
c = ta["null_offset_corrected"]
print("귀무 오프셋 보정 대각  p", c["p_dyn"]["frac_diag"], " 우연", c["p_null_chance"]["frac_diag"])
print("   방향별 p", {d: v["frac_diag"] for d, v in c["p_dyn"]["by_direction"].items()})
print("diag / 대각 밖 최소  zf", ta["diag_over_min_offdiag"]["zf"], " p", ta["diag_over_min_offdiag"]["p"])
print("\n== β (풀링, 중심화; 표적 = 표준 양방향 h_{8+i})")
for fname, r in bp.items():
    print(f"  [{fname}] p {r['p']['slots0_7']} 0–3 {r['p']['slots0_3']} 4–7 {r['p']['slots4_7']} | 방향별 {r['p']['slots0_7_by_direction']}")
    print(f"           귀무 직전 {r['null_prev_clip']['slots0_7']} 무작위 {r['null_random_clip']['slots0_7']} | zf {r['zf']['slots0_7']} | 역순 {r['p_reversed_slot_order']['slots0_7']}")
    print(f"           전이별 p {r['transitions']['p']['est']}  귀무 {r['transitions']['null_random_clip']['est']}")
    print(f"           정지 감쇠 오라클 γ {r['oracle_static_decay_gamma']} 선형 {r['oracle_static_decay_linear']}")
    print("           문맥 투영 제거 " + " | ".join(f"{k}: p {v['p'][0]} 4–7 {v['p_slots4_7'][0]} zf {v['zf'][0]} 귀무 {v['null_random_clip'][0]}" for k, v in r["after_projecting_out_context"].items()))
    for k, v in r["calibration"].items():
        am = v["argmin"] if "argmin" in v else v
        print(f"           교정 {k:32s} argmin 기울기 {am['slope']:+.3f} (0–3 {am['slope_slots0_3']:+.3f})" + (f"  β {v['beta']}" if "beta" in v else ""))
print("\n== 복사 대비 (토큰 수준, 동적 토큰)")
print("EI 인과 (동적)", [r[0] for r in out["EI_causal_dyn"]], "슬롯 평균", out["EI_causal_dyn_slotmean"]); print("EI 새는 (동적)", [r[0] for r in out["EI_leaky_dyn"]])
print("EI 인과 (전 토큰)", [r[0] for r in out["EI_causal_all"]]); print("p < 인과 복사 비율", [r[0] for r in out["p_beats_causal_copy_dyn"]])
for nm in ("ctxmean", "blur3x3", "mix", "pmean"):
    if f"EI_vs_{nm}_dyn" in out:
        print(f"EI vs {nm:8s}", [r[0] for r in out[f"EI_vs_{nm}_dyn"]], " p 가 이기는 비율", [r[0] for r in out[f"p_beats_{nm}_dyn"]],
              " 재현 비율", out["baseline_share_of_EI_causal"][nm]["share_of_p_gain"], " 방향별", out["by_direction"][f"EI_vs_{nm}_dyn"])
for fname, v in cp.items():
    print(f"  [{fname}] L1 " + " ".join(f"{k} {vv}" for k, vv in v["L1"].items()))
    print(f"           EI p vs 위치 보정 복사 {v['EI_p_vs_hc7_poscorr']} 승률 {v['win_p_vs_hc7_poscorr']} | p vs 위치 보정 문맥 평균 승률 {v['win_p_vs_ctxmean_poscorr']}")
for fname, v in eo.items():
    print(f"  [{fname}] 슬롯 평균 대비 EI (풀링) p {v['p']}  오차 맞춤 진전 오라클 {v['oracle_truth_plus_other_p_error']}")
print("\nρ p↔진실", [r[0] for r in out["rho_p_vs_true"]])
if "rho_lastmotion_vs_true" in out:
    print("ρ 마지막 운동지도 (h) ↔ 진실", [r[0] for r in out["rho_lastmotion_vs_true"]]); print("ρ 마지막 운동지도 (hc) ↔ 진실", [r[0] for r in out["rho_lastmotion_causal_vs_true"]]); print("ρ 귀무", [r[0] for r in out["rho_null"]]); print("ρ (hc 기준)", [r[0] for r in out["rho_pc_vs_true"]])
print(f"\n[끝 {time.time()-t0:.0f}s] → {OUT / f'ssv2_e{TAG}.json'}")
