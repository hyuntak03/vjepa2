#!/usr/bin/env python3
"""[P] 고리 3 · 4 — 학습된 블록 AR predictor 의 출력이 "같은 칸 문맥 블록 평균" 해인가, 그 해가 운반 0.27 을 재현하나 (auto_research, 2026-10-21).

  CUDA_VISIBLE_DEVICES=4 /venv/vjepa/bin/python -W ignore auto_research/scripts/pred_mean_solution.py \
      --run natural_ar_vw_state latest.pt [--attn 1] [--tag base]

데이터: 캐시 /workspace/auto_research_cache/phys3d/ (phys3d_single_v1 val 358 창, raw 0 부터 간격 3 으로 32 장 = 블록 16 개, 블록별 LN(target_encoder)).
모델 입력 = 캐시의 정답 블록 (fp16 → fp32, bf16 autocast — transport_gain.py 와 같은 호출).

A. TF 한 걸음 (위치 t = 7..14 → 블록 t+1, 문맥 = 정답 블록 0..t).  물체 칸 obj_b = 블록 b 두 장 중 bbox 덮임 ≥ 0.25 인 칸.
   쌍 종류: mov (t → t+1 물체 이동 > 8 px, 두 블록 모두 보임 vis > 0.7) / sta (< 1 px, 두 블록 모두 보임).
   칸 무리: new = obj_{t+1} − obj_t (물체가 올 곳) · old = obj_t − obj_{t+1} (떠난 곳) · ovl = 둘 다 · sobj (sta 의 물체 칸) ·
            far = 창 전체에서 물체가 한 번도 안 닿은 칸을 2 칸 팽창한 것의 여집합.
   회귀 (무리별로 표본 전체를 모아 최소제곱, 상수 없음): y ≈ Σ_k w_k h_{t−k} (k = 0..7, 같은 칸) [+ β · mc],
     mc = h_t 를 정답 변위 (정수 칸) 만큼 옮긴 것 (= 물체를 옳게 옮긴 복사). y = 모델 예측 p 와 정답 h_{t+1} (대조군) 둘 다.
     창마다 충분통계 (XᵀX, Xᵀy, yᵀy) 를 남겨 clip bootstrap 이 가능하게 한다.
   물체 점수 (선형): s(x) = x · T̂, T̂ = 블록 t 물체 칸 평균 토큰 (정규화). 무리 칸 평균 − far 칸 평균 = 대비.
     new 의 존재감 = [대비(x) − 대비(h_t)] / [대비(h_{t+1}) − 대비(h_t)]   (1 = 정답처럼 그림, 0 = 복사처럼 없음)
     old 의 잔상   = [대비(x) − 대비(h_{t+1})] / [대비(h_t) − 대비(h_{t+1})] (1 = 옛 자리에 그대로, 0 = 정답처럼 지움)
     선형이라 가중합 예측기의 값은 성분 값의 가중합으로 정확히 계산된다.
   칸별 L1 거리: |p − x| (x = copy · mean_2/4/8 · 정답 · 무리별 적합 가중합 (B 단계)).
B. 운반 이득 — transport_gain.py 와 같은 읽기 · 같은 정의 (템플릿 = 블록 7 물체 중심 3×3, cos argmax 주변 3×3 가중 중심,
   TF 기준 = 직전 정답 블록 위치, rollout 기준 = 문맥 끝 위치, |d| > 16 px · 보이는 블록만).
   예측기: 모델 TF / 모델 rollout / 정답 / copy / mean_n / LA_far (far 칸 적합 가중치를 모든 칸에, rollout 은 재귀) /
           LA_grp (무리별 M0 가중치, 무리는 정답으로 앎) / M1_grp (무리별 M0 + β·mc).
C. rollout 존재감 · 잔상 (블록 8..15): 대비 = 물체 칸 s 평균 − far 칸 s 평균 (T̂ = 블록 7 물체), 정답 대비로 나눈 비율.
   정지 (블록 7 대비 변위 < 4 px) vs 움직임 (> 16 px). 잔상 = 블록 7 물체 칸 중 그 블록 물체 칸 (1 칸 팽창) 밖.
D. (--attn 1) attention — TF 위치 t = 7 의 query 를 무리별로 (new · old · ovl · sobj · far), key 를 지연 k (블록 7−k) 별로:
   같은 칸 · 3×3 이웃 · 블록 전체 질량, new query 가 블록 7 물체 칸 (옛 자리) 에 준 질량 (헤드별)."""
import argparse, glob, json, os, sys, time
import numpy as np, torch, torch.nn.functional as F, yaml
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, REPO)
from analysis.intphys2.model import _clean_backbone_key                       # noqa: E402
from src.models.rollout_predictor import vit_prefix_predictor                 # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="/workspace/auto_research_cache/phys3d")
ap.add_argument("--run", nargs=2, required=True, metavar=("RUN", "CKPT"))
ap.add_argument("--tag", default=None)
ap.add_argument("--attn", type=int, default=0)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--out", default=os.path.join(REPO, "auto_research/results/pred_mean"))
a = ap.parse_args()
dev = torch.device("cuda")
S, G, C, K, NB, L = 256, 16, 8, 8, 16, 8          # 칸 · 격자 · 문맥 · 미래 · 블록 · 지연 수
tag = a.tag or f"{a.run[0]}__{a.run[1].replace('.pt', '')}"
os.makedirs(a.out, exist_ok=True)


def load_pred(run, ck):
    D = os.path.join(REPO, "z_training/runs", run)
    pc = yaml.safe_load(open(os.path.join(D, "config.yaml")))["model"]["predictor"]
    assert pc.get("kind") == "ar", f"{run}: kind=ar 만"
    kw = dict(img_size=256, patch_size=16, tubelet_size=2, num_frames=64, embed_dim=1280, predictor_embed_dim=384, depth=12, num_heads=12,
              num_mask_tokens=10, mask_mode="ar", ar_feedback=pc.get("ar_feedback", "output"))
    for k_ in ("gating_residual", "gating_anchor", "gating_out_ln"):
        if pc.get(k_) is not None:
            kw[k_] = pc[k_]
    m = vit_prefix_predictor(**kw)
    w = {k: v for k, v in _clean_backbone_key(torch.load(os.path.join(D, ck), map_location="cpu", weights_only=False)["predictor"]).items()
         if not k.startswith(("hint_attn.", "dyn_head.", "recon_proj."))}
    m.load_state_dict(w, strict=True)
    return m.to(dev).eval()


model = load_pred(*a.run)

# ------------------------------------------------------------------ 데이터
H_list, LAB, CLIP = [], {k: [] for k in ("pos", "bbox", "vis_frac", "inframe_frac")}, []
for f in sorted(glob.glob(os.path.join(a.cache, "h_val_target_s*.npy"))):
    sh = f.split("_s")[-1].split(".")[0]
    X = np.load(f, mmap_mode="r"); lab = np.load(os.path.join(a.cache, f"lab_val_s{sh}.npz"))
    meta = json.load(open(os.path.join(a.cache, f"meta_val_s{sh}.json")))
    for w in range(X.shape[0]):
        H_list.append((f, w))
        for k in LAB:
            LAB[k].append(lab[k][w])
        CLIP.append(meta[w]["path"])
LAB = {k: np.stack(v) for k, v in LAB.items()}
_, CLIP = np.unique(np.array(CLIP), return_inverse=True)
NW = len(H_list) if not a.limit else min(a.limit, len(H_list))
_mm = {}


def get_h(i):
    f, w = H_list[i]
    if f not in _mm:
        _mm[f] = np.load(f, mmap_mode="r")
    return torch.from_numpy(np.asarray(_mm[f][w], dtype=np.float32)).to(dev)      # (16, 256, D)


def cover(bbox):
    x0, y0, x1, y1 = [bbox[..., i, None] for i in range(4)]
    c = np.arange(G) * 16.0
    ox = np.clip(np.minimum(c + 16, x1) - np.maximum(c, x0), 0, 16) / 16
    oy = np.clip(np.minimum(c + 16, y1) - np.maximum(c, y0), 0, 16) / 16
    return oy[..., :, None] * ox[..., None, :]


def dil(m, k=1):
    m = m.reshape(G, G).copy()
    for _ in range(k):
        o = m.copy()
        o[1:] |= m[:-1]; o[:-1] |= m[1:]; o[:, 1:] |= m[:, :-1]; o[:, :-1] |= m[:, 1:]
        o[1:, 1:] |= m[:-1, :-1]; o[:-1, :-1] |= m[1:, 1:]; o[1:, :-1] |= m[:-1, 1:]; o[:-1, 1:] |= m[1:, :-1]
        m = o
    return m.reshape(-1)


def masks(i):
    lab = {k: v[i] for k, v in LAB.items()}
    cv = cover(np.nan_to_num(lab["bbox"], nan=-100.0))                        # (32, G, G)
    objb = (np.maximum(cv[0::2], cv[1::2]) >= 0.25).reshape(NB, S)             # (16, 256)
    touch = ((cv[0::2] > 0) | (cv[1::2] > 0)).reshape(NB, S)
    ok = lab["inframe_frac"] > 0.9
    visb = (lab["vis_frac"][0::2] > 0.7) & (lab["vis_frac"][1::2] > 0.7) & ok[0::2] & ok[1::2]
    far = ~dil(touch.any(0), 2)
    return lab, objb, visb, far


cells3 = lambda cx, cy: [y * G + x for y in range(max(0, cy - 1), min(G, cy + 2)) for x in range(max(0, cx - 1), min(G, cx + 2))]
cell_of = lambda p: (int(np.clip(p[0] // 16, 0, G - 1)), int(np.clip(p[1] // 16, 0, G - 1)))
yy, xx = np.divmod(np.arange(S), G)
cen = np.stack([xx * 16 + 8, yy * 16 + 8], 1).astype(np.float32)


def readout(blk, tmpl):
    cs = (F.normalize(blk.float(), dim=-1) @ tmpl).cpu().numpy()
    j = int(cs.argmax()); nb = cells3(j % G, j // G)
    w = np.clip(cs[nb], 0, None) + 1e-6
    return (cen[nb] * w[:, None]).sum(0) / w.sum()


def shift(hb, d):
    dx, dy = int(round(d[0] / 16)), int(round(d[1] / 16))
    return torch.roll(hb.view(G, G, -1), shifts=(dy, dx), dims=(0, 1)).reshape(S, -1), (dx, dy)


@torch.no_grad()
def model_out(h):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        p_tf = model.forward_seq(h[:NB - 1].reshape(1, -1, h.size(-1)), torch.arange((NB - 1) * S, device=dev).unsqueeze(0)).float()[0]
        p_ro = model.rollout_kv(h[:C].reshape(1, C * S, -1), torch.arange(C * S, device=dev).unsqueeze(0), K).float()[0]
    return p_tf.view(NB - 1, S, -1), p_ro.view(K, S, -1)


GROUPS_MOV, GROUPS_STA = ("new", "old", "ovl", "far"), ("sobj", "far")
NR = L + 1                                                                     # 지연 8 + mc

# ------------------------------------------------------------------ A 단계: 모델 출력 + 충분통계
t0 = time.time()
REC = []                    # TF 기록 (창, t, 종류, 무리, 통계)
ROREC = []                  # rollout 기록
for i in range(NW):
    lab, objb, visb, far = masks(i)
    pos = lab["pos"]
    h = get_h(i)
    p_tf, p_ro = model_out(h)
    for t in range(L - 1, NB - 1):
        d = pos[2 * t + 3] - pos[2 * t + 1]; dn = float(np.linalg.norm(d))
        if not np.isfinite(dn) or not (visb[t] and visb[t + 1]):
            continue
        kind = "mov" if dn > 8 else ("sta" if dn < 1 else None)
        if kind is None or objb[t].sum() == 0:
            continue
        mc, (dx, dy) = shift(h[t], d)
        T_hat = F.normalize(h[t][torch.from_numpy(objb[t]).to(dev)].mean(0), dim=-1)
        grp = {"far": far}
        if kind == "mov":
            grp.update(new=objb[t + 1] & ~objb[t], old=objb[t] & ~objb[t + 1], ovl=objb[t] & objb[t + 1])
        else:
            grp.update(sobj=objb[t] & objb[t + 1])
        lags = torch.stack([h[t - k] for k in range(L)] + [mc])                # (9, S, D)
        y = {"pred": p_tf[t], "true": h[t + 1]}
        m4 = {f"mean_{n}": h[t - n + 1:t + 1].mean(0) for n in (2, 4, 8)}
        # 물체 점수 (선형): 성분 9 + pred + true
        sc = torch.cat([lags, y["pred"][None], y["true"][None]]) @ T_hat       # (11, S)
        farm = torch.from_numpy(far).to(dev)
        sfar = sc[:, farm].mean(1) if far.any() else torch.full((11,), float("nan"), device=dev)
        for g, msk in grp.items():
            if msk.sum() == 0:
                continue
            mt = torch.from_numpy(msk).to(dev)
            X = lags[:, mt]                                                    # (9, n, D)
            XtX = torch.einsum("knd,lnd->kl", X, X)
            r = dict(win=i, clip=int(CLIP[i]), t=t, kind=kind, grp=g, n=int(msk.sum()), dn=dn, shift=(dx, dy),
                     XtX=XtX.cpu().numpy())
            for nm, yv in y.items():
                Y = yv[mt]
                r[f"Xty_{nm}"] = torch.einsum("knd,nd->k", X, Y).cpu().numpy()
                r[f"yty_{nm}"] = float((Y * Y).sum())
            r["xy_pt"] = float((y["pred"][mt] * y["true"][mt]).sum())          # pred · true (거리용)
            r["s"] = sc[:, mt].mean(1).cpu().numpy()                           # (11,) 무리 평균 점수
            r["sfar"] = sfar.cpu().numpy()
            P = y["pred"][mt]
            dist = {"copy": h[t][mt], **{k_: v_[mt] for k_, v_ in m4.items()}, "true": y["true"][mt]}
            r["l1"] = {k_: float((P - v_).abs().mean()) for k_, v_ in dist.items()}
            r["l1_true"] = {k_: float((y["true"][mt] - v_).abs().mean()) for k_, v_ in dist.items() if k_ != "true"}
            REC.append(r)
    # rollout 존재감 · 잔상
    if visb[C - 1] and objb[C - 1].any():
        T7 = F.normalize(h[C - 1][torch.from_numpy(objb[C - 1]).to(dev)].mean(0), dim=-1)
        farm = torch.from_numpy(far).to(dev)
        if far.any():
            for j in range(K):
                b = C + j
                if not visb[b] or not objb[b].any():
                    continue
                D = float(np.linalg.norm(pos[2 * b + 1] - pos[2 * C - 1]))
                if not np.isfinite(D):
                    continue
                cand = torch.stack([p_ro[j], h[b], h[C - 1], h[:C].mean(0)])    # pred_ro · true · copy · mean8
                sc = cand @ T7
                om = torch.from_numpy(objb[b]).to(dev)
                oldm = objb[C - 1] & ~dil(objb[b], 1)
                ROREC.append(dict(win=i, clip=int(CLIP[i]), j=j, D=D, s_obj=sc[:, om].mean(1).cpu().numpy(), s_far=sc[:, farm].mean(1).cpu().numpy(),
                                  s_old=sc[:, torch.from_numpy(oldm).to(dev)].mean(1).cpu().numpy() if oldm.any() else None,
                                  l1=[float((p_ro[j][om] - x_[om]).abs().mean()) for x_ in (h[b], h[C - 1], h[:C].mean(0))]))
    if i % 50 == 0:
        print(f"[A] {i}/{NW} 기록 {len(REC)} ({time.time() - t0:.0f}s)", flush=True)


# ------------------------------------------------------------------ 적합
def fit(recs, nm, cols):
    """recs 의 충분통계를 모아 cols (회귀자 번호) 로 최소제곱. return w, R² (비중심: 1 − SSE/Σy²)."""
    if not recs:
        return None, float("nan")
    XtX = sum(r["XtX"][np.ix_(cols, cols)] for r in recs).astype(np.float64)
    Xty = sum(r[f"Xty_{nm}"][cols] for r in recs).astype(np.float64)
    yty = sum(r[f"yty_{nm}"] for r in recs)
    lam = 1e-6 * np.trace(XtX) / len(cols)
    w = np.linalg.solve(XtX + lam * np.eye(len(cols)), Xty)
    sse = yty - 2 * w @ Xty + w @ XtX @ w
    return w, float(1 - sse / yty)


M0 = list(range(L)); M1 = list(range(NR)); MCC = [0, L]          # 같은 칸 8 지연 / + mc / copy + mc
rng = np.random.RandomState(0)


def boot(recs, fn, nb=300):
    cl = np.array([r["clip"] for r in recs]); uc = np.unique(cl); ix = {c: np.where(cl == c)[0] for c in uc}
    vals = []
    for _ in range(nb):
        sel = np.concatenate([ix[c] for c in rng.choice(uc, len(uc))])
        vals.append(fn([recs[k] for k in sel]))
    vals = np.array(vals, dtype=np.float64)
    return np.nanpercentile(vals, [2.5, 97.5], axis=0)


FIT = {}
summary = {"run": a.run, "n_windows": NW, "tf": {}, "ro": {}, "attn": None}
print(f"\n== {tag}: TF 한 걸음 — 같은 칸 가중합 적합 (무리별, 표본 전체)")
for kind, groups in (("mov", GROUPS_MOV), ("sta", GROUPS_STA)):
    for g in groups:
        R_ = [r for r in REC if r["kind"] == kind and r["grp"] == g]
        if not R_:
            continue
        ent = {"n_rec": len(R_), "n_cells": int(sum(r["n"] for r in R_))}
        for nm in ("pred", "true"):
            w0, r20 = fit(R_, nm, M0)
            w1, r21 = (fit(R_, nm, M1) if kind == "mov" else (None, float("nan")))
            wc, r2c = (fit(R_, nm, MCC) if kind == "mov" else (None, float("nan")))
            ent[nm] = dict(M0_w=w0.tolist(), M0_R2=r20, M0_sum=float(w0.sum()), M0_w0_share=float(w0[0] / w0.sum()),
                           M1_w=None if w1 is None else w1.tolist(), M1_R2=r21, CM_w=None if wc is None else wc.tolist(), CM_R2=r2c)
            FIT[(kind, g, nm, "M0")] = w0
            if w1 is not None:
                FIT[(kind, g, nm, "M1")] = w1
        # bootstrap: pred 의 w0 몫 · M1 β · R²
        ci = boot(R_, lambda rr: [fit(rr, "pred", M0)[0][0] / fit(rr, "pred", M0)[0].sum(), fit(rr, "pred", M0)[1]]
                  + ([fit(rr, "pred", M1)[0][L], fit(rr, "true", M1)[0][L]] if kind == "mov" else [np.nan, np.nan]))
        ent["ci_pred_w0share_R2_beta_pred_beta_true"] = ci.tolist()
        # 존재감 · 잔상 (선형 점수; 성분 순서: lag0..7, mc, pred, true)
        s = np.stack([r["s"] for r in R_]); sf = np.stack([r["sfar"] for r in R_]); con = np.nanmean(s - sf, 0)
        ent["contrast"] = {"lags": con[:L].tolist(), "mc": float(con[L]), "pred": float(con[L + 1]), "true": float(con[L + 2])}
        ent["l1_pred_to"] = {k_: float(np.mean([r["l1"][k_] for r in R_])) for k_ in R_[0]["l1"]}
        ent["l1_true_to"] = {k_: float(np.mean([r["l1_true"][k_] for r in R_])) for k_ in R_[0]["l1_true"]}
        summary["tf"][f"{kind}/{g}"] = ent
        P, T_ = ent["pred"], ent["true"]
        print(f"  {kind}/{g:4s} rec {len(R_):4d} 칸 {ent['n_cells']:5d} | pred M0 R² {P['M0_R2']:.3f} 합 {P['M0_sum']:.2f} w0 몫 {P['M0_w0_share']:.2f} "
              f"w {np.round(P['M0_w'], 2).tolist()} | true M0 R² {T_['M0_R2']:.3f} w0 몫 {T_['M0_w0_share']:.2f}"
              + (f" | M1 β pred {P['M1_w'][L]:.2f} (R² {P['M1_R2']:.3f}) true {T_['M1_w'][L]:.2f} (R² {T_['M1_R2']:.3f})" if kind == "mov" else ""))
        print(f"        CI [w0몫, R², β_pred, β_true] {np.round(ci, 3).tolist()} | 대비 pred {con[L + 1]:.2f} true {con[L + 2]:.2f} copy {con[0]:.2f} mc {con[L]:.2f}"
              f" | L1 pred→ {json.dumps({k_: round(v_, 3) for k_, v_ in ent['l1_pred_to'].items()}, ensure_ascii=False)}")


# 존재감 (new) · 잔상 (old) — 선형 점수 비율, 가중합 예측기 포함
def ratio_new(recs, coef):
    """coef (11,) 성분 가중치 → [대비(x) − 대비(copy)] / [대비(true) − 대비(copy)] (평균 대비의 비)."""
    s = np.stack([r["s"] - r["sfar"] for r in recs]); x = s @ coef
    return float((x.mean() - s[:, 0].mean()) / (s[:, L + 2].mean() - s[:, 0].mean()))


def ratio_old(recs, coef):
    s = np.stack([r["s"] - r["sfar"] for r in recs]); x = s @ coef
    return float((x.mean() - s[:, L + 2].mean()) / (s[:, 0].mean() - s[:, L + 2].mean()))


E = lambda k_: np.eye(L + 3)[k_]
def comb(w):
    c = np.zeros(L + 3); c[:len(w)] = w; return c


pres = {}
for g, fn in (("new", ratio_new), ("old", ratio_old)):
    R_ = [r for r in REC if r["kind"] == "mov" and r["grp"] == g]
    if not R_:
        continue
    cands = {"pred": E(L + 1), "copy": E(0), "mean_2": comb([.5] * 2), "mean_4": comb([.25] * 4), "mean_8": comb([.125] * 8),
             "LA_far(pred 적합)": comb(FIT[("mov", "far", "pred", "M0")]), f"LA_{g}(pred 적합)": comb(FIT[("mov", g, "pred", "M0")]),
             f"M1_{g}(pred 적합)": comb(FIT[("mov", g, "pred", "M1")]), "mc (정답 이동)": E(L)}
    pres[g] = {k_: fn(R_, c_) for k_, c_ in cands.items()}
    ci = boot(R_, lambda rr: [fn(rr, cands["pred"])])
    pres[g]["pred_ci"] = ci[:, 0].tolist()
summary["presence_tf"] = pres
print("\n== TF 존재감 (new: 1 = 정답처럼 물체를 그림, 0 = 복사) · 잔상 (old: 1 = 옛 자리에 그대로, 0 = 지움)")
for g, d_ in pres.items():
    print(f"  {g}: " + " · ".join(f"{k_} {v_:.2f}" if not isinstance(v_, list) else f"{k_} [{v_[0]:.2f}, {v_[1]:.2f}]" for k_, v_ in d_.items()))

# ------------------------------------------------------------------ B 단계: 운반 이득 (같은 읽기)
wfar = FIT[("mov", "far", "pred", "M0")]
WG = {g: FIT.get(("mov", g, "pred", "M0")) for g in ("new", "old", "ovl", "far")}
WG1 = {g: FIT.get(("mov", g, "pred", "M1")) for g in ("new", "old", "ovl", "far")}
WS = {g: FIT.get(("sta", g, "pred", "M0")) for g in ("sobj", "far")}
NAMES_TF = ["model", "gt", "copy", "mean_2", "mean_4", "mean_8", "LA_far", "LA_grp", "M1_grp"]
NAMES_RO = ["model", "gt", "copy", "mean_8", "LA_far_rec"]
TFP = {n: [] for n in NAMES_TF}; ROP = {n: [] for n in NAMES_RO}
REF = {"pc": [], "futp": [], "futv": [], "clip": []}
L1B = {n: {"new": [], "old": [], "far": []} for n in ("LA_grp", "LA_far", "M1_grp")}
t0 = time.time()
for i in range(NW):
    lab, objb, visb, far = masks(i)
    pos = lab["pos"]; fe = 2 * C - 1
    pos_c = pos[fe]
    if not (lab["vis_frac"][fe] > 0.3 and np.isfinite(pos_c).all()):
        continue
    h = get_h(i)
    p_tf, p_ro = model_out(h)
    tmpl = F.normalize(h[C - 1][cells3(*cell_of(pos_c))].mean(0), dim=-1)
    REF["pc"].append(pos_c); REF["clip"].append(int(CLIP[i]))
    REF["futp"].append(np.stack([pos[2 * (C + j) + 1] for j in range(K)]))
    REF["futv"].append(np.array([lab["vis_frac"][2 * (C + j) + 1] > 0.3 for j in range(K)]))
    wt = torch.tensor(wfar, device=dev, dtype=torch.float32)
    rows = {n: [] for n in NAMES_TF}
    for j in range(K):
        t = C - 1 + j
        ctx = torch.stack([h[t - k] for k in range(L)])                                # (8, S, D)
        la_far = torch.einsum("k,ksd->sd", wt, ctx)
        # 무리별 (정답 무리 사용)
        d = pos[2 * t + 3] - pos[2 * t + 1]
        if np.isfinite(d).all():
            mc, _ = shift(h[t], d)
            new = objb[t + 1] & ~objb[t]; old = objb[t] & ~objb[t + 1]; ovl = objb[t] & objb[t + 1]
        else:
            mc, new, old, ovl = h[t], np.zeros(S, bool), np.zeros(S, bool), np.zeros(S, bool)
        la_g = la_far.clone(); m1_g = la_far.clone()
        for g, msk in (("new", new), ("old", old), ("ovl", ovl)):
            if msk.any() and WG[g] is not None:
                mt = torch.from_numpy(msk).to(dev)
                la_g[mt] = torch.einsum("k,ksd->sd", torch.tensor(WG[g], device=dev, dtype=torch.float32), ctx[:, mt])
                w1 = torch.tensor(WG1[g], device=dev, dtype=torch.float32)
                m1_g[mt] = torch.einsum("k,ksd->sd", w1[:L], ctx[:, mt]) + w1[L] * mc[mt]
        oth = ~(new | old | ovl)
        if WG1["far"] is not None:                                                     # 나머지 칸의 M1 = far 적합
            mt = torch.from_numpy(oth).to(dev); w1 = torch.tensor(WG1["far"], device=dev, dtype=torch.float32)
            m1_g[mt] = torch.einsum("k,ksd->sd", w1[:L], ctx[:, mt]) + w1[L] * mc[mt]
        if visb[t] and visb[t + 1] and np.isfinite(d).all() and np.linalg.norm(d) > 8:
            for g, msk in (("new", new), ("old", old), ("far", far)):
                if msk.any():
                    mt = torch.from_numpy(msk).to(dev)
                    for nm_, v_ in (("LA_grp", la_g), ("LA_far", la_far), ("M1_grp", m1_g)):
                        L1B[nm_][g].append(float((p_tf[t][mt] - v_[mt]).abs().mean()))
        cand = {"model": p_tf[t], "gt": h[t + 1], "copy": h[t], "mean_2": ctx[:2].mean(0), "mean_4": ctx[:4].mean(0),
                "mean_8": ctx.mean(0), "LA_far": la_far, "LA_grp": la_g, "M1_grp": m1_g}
        for n in NAMES_TF:
            rows[n].append(readout(cand[n], tmpl))
    for n in NAMES_TF:
        TFP[n].append(np.stack(rows[n]))
    # rollout: 모델 · 정답 · copy · mean8 · 재귀 선형 AR (far 적합 가중치)
    seq = [h[b] for b in range(C)]
    la_ro = []
    for j in range(K):
        nxt = sum(float(wfar[k]) * seq[-1 - k] for k in range(L))
        la_ro.append(nxt); seq.append(nxt)
    cand_ro = {"model": list(p_ro), "gt": [h[C + j] for j in range(K)], "copy": [h[C - 1]] * K, "mean_8": [h[:C].mean(0)] * K, "LA_far_rec": la_ro}
    for n in NAMES_RO:
        ROP[n].append(np.stack([readout(cand_ro[n][j], tmpl) for j in range(K)]))
    if i % 50 == 0:
        print(f"[B] {i}/{NW} ({time.time() - t0:.0f}s)", flush=True)

pc, futp, futv, clip = (np.asarray(REF[k]) for k in ("pc", "futp", "futv", "clip"))
prevt = np.concatenate([pc[:, None], futp[:, :-1]], 1)


def gain(Q, base):
    dp, dtr = Q - base, futp - base
    g = (dp * dtr).sum(-1) / np.maximum((dtr ** 2).sum(-1), 1e-6)
    ok = (np.linalg.norm(dtr, axis=-1) > 16) & futv & np.isfinite(g) & np.isfinite(dtr).all(-1)
    return g, ok


def gboot(g, ok):
    if ok.sum() < 5:
        return [float("nan"), float("nan")]
    cl = np.broadcast_to(clip[:, None], ok.shape)[ok]; gv = g[ok]
    uc = np.unique(cl); ix = {c: np.where(cl == c)[0] for c in uc}
    bs = [np.median(gv[np.concatenate([ix[c] for c in rng.choice(uc, len(uc))])]) for _ in range(1000)]
    return np.percentile(bs, [2.5, 97.5]).tolist()


print(f"\n== 운반 이득 중앙값 (transport_gain.py 와 같은 읽기 · 정의), 창 {len(pc)} · clip {len(np.unique(clip))}")
summary["gain"] = {}
for mode, P_, base in (("tf", TFP, prevt), ("ro", ROP, pc[:, None])):
    for n, Q in P_.items():
        g, ok = gain(np.stack(Q), base)
        med = float(np.median(g[ok])) if ok.any() else float("nan"); ci = gboot(g, ok)
        per = [float(np.median(g[ok[:, j], j])) if ok[:, j].sum() > 5 else None for j in range(K)]
        summary["gain"][f"{mode}/{n}"] = dict(median=med, ci=ci, n=int(ok.sum()), per_step=per)
        print(f"  {mode} {n:10s} {med:6.3f} [{ci[0]:.2f}, {ci[1]:.2f}] n {ok.sum():4d} | 걸음별 " + " ".join("—" if v is None else f"{v:.2f}" for v in per))
summary["l1_pred_to_fitted"] = {nm_: {g: float(np.mean(v)) if v else None for g, v in d_.items()} for nm_, d_ in L1B.items()}
print("  L1 (모델 TF 예측 ↔ 적합 재구성), mov 무리별: " + json.dumps(summary["l1_pred_to_fitted"], ensure_ascii=False))

# ------------------------------------------------------------------ C: rollout 존재감 · 잔상 (정지 vs 움직임)
print("\n== rollout 존재감 (물체 칸 대비 / 정답 대비) · 잔상 (블록 7 물체 칸 중 지금 물체 밖, 대비 / 블록 7 대비)")
summary["ro_presence"] = {}
for cls, sel in (("static(<4px)", lambda r: r["D"] < 4), ("moving(>16px)", lambda r: r["D"] > 16), ("moving(>32px)", lambda r: r["D"] > 32)):
    out = []
    for j in range(K):
        R_ = [r for r in ROREC if r["j"] == j and sel(r)]
        if len(R_) < 5:
            out.append(None); continue
        c = np.stack([r["s_obj"] - r["s_far"] for r in R_]).mean(0)                    # pred · true · copy · mean8
        Ro = [r for r in R_ if r["s_old"] is not None]
        c7 = None
        if len(Ro) >= 5:
            co = np.stack([r["s_old"] - r["s_far"] for r in Ro]).mean(0)
            # 잔상 비율: 블록 7 자기 자리 대비 (copy 가 그 자리에서 내는 대비 = 블록 7 물체 대비) 로 나눔
            c7 = (co / co[2]).tolist()
        out.append(dict(n=len(R_), pres=(c / c[1]).tolist(), trail=c7, n_old=len(Ro),
                        l1=np.mean([r["l1"] for r in R_], 0).tolist()))
    summary["ro_presence"][cls] = out
    print(f"  {cls}")
    for j, o in enumerate(out):
        if o is None:
            continue
        tr = "—" if o["trail"] is None else " / ".join(f"{v:.2f}" for v in o["trail"])
        print(f"    걸음 {j + 1}: n {o['n']:3d} 존재감 pred {o['pres'][0]:.2f} · copy {o['pres'][2]:.2f} · mean8 {o['pres'][3]:.2f} | "
              f"잔상 (pred / true / copy / mean8) {tr} (n {o['n_old']}) | L1 물체 칸 pred↔(true, copy, mean8) {np.round(o['l1'], 3).tolist()}")

# ------------------------------------------------------------------ D: attention (t = 7)
if a.attn:
    import src.models.utils.modules as M
    REC_A = {"on": False, "w": []}
    _sdpa = F.scaled_dot_product_attention

    def sdpa(q, k, v, *args, **kw):
        if REC_A["on"]:
            w = torch.softmax((q.float() @ k.float().transpose(-1, -2)) / q.size(-1) ** 0.5, dim=-1)   # (B, h, nq, M)
            REC_A["w"].append(w[0])                                                                 # (h, 256, M)
        return _sdpa(q, k, v, *args, **kw)
    M.F.scaled_dot_product_attention = sdpa
    t = C - 1
    AT = {g: [] for g in ("mov/new", "mov/old", "mov/ovl", "mov/far", "sta/sobj", "sta/far")}
    AT_old_head = []
    for i in range(NW):
        lab, objb, visb, far = masks(i)
        pos = lab["pos"]
        d = pos[2 * t + 3] - pos[2 * t + 1]; dn = float(np.linalg.norm(d))
        if not np.isfinite(dn) or not (visb[t] and visb[t + 1]):
            continue
        kind = "mov" if dn > 8 else ("sta" if dn < 1 else None)
        if kind is None:
            continue
        h = get_h(i)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            caches = None
            for b in range(t + 1):
                if b == t:
                    REC_A.update(on=True, w=[])
                _, caches, _ = model._ar_step_kv(h[b:b + 1], torch.arange(b * S, (b + 1) * S, device=dev).unsqueeze(0), caches)
                REC_A["on"] = False
        Wl = torch.stack(REC_A["w"])                                    # (12, h, 256, (t+1)*256)
        nl, nh = Wl.shape[:2]
        Wm = Wl.mean(1).view(nl, S, t + 1, S)                           # (layer, q, blk, key cell)
        diag = torch.diagonal(Wm, dim1=1, dim2=3)                       # (layer, blk, q) 같은 칸
        nbm = torch.zeros(S, S, device=dev)
        for c_ in range(S):
            nbm[c_, cells3(c_ % G, c_ // G)] = 1
        nb3 = torch.einsum("lqbk,qk->lbq", Wm, nbm)                     # 3×3 이웃
        blk = Wm.sum(-1).transpose(1, 2)                                # (layer, blk, q)
        grp = {"far": far}
        if kind == "mov":
            grp.update(new=objb[t + 1] & ~objb[t], old=objb[t] & ~objb[t + 1], ovl=objb[t] & objb[t + 1])
        else:
            grp.update(sobj=objb[t] & objb[t + 1])
        for g, msk in grp.items():
            if not msk.any():
                continue
            mq = torch.from_numpy(msk).to(dev)
            # 지연 k = 0..t 순서 (블록 t−k)
            lag_ix = torch.arange(t, -1, -1, device=dev)
            rec = np.stack([diag[:, lag_ix][:, :, mq].mean(-1).cpu().numpy(),
                            nb3[:, lag_ix][:, :, mq].mean(-1).cpu().numpy(),
                            blk[:, lag_ix][:, :, mq].mean(-1).cpu().numpy()])   # (3, layer, lag)
            AT[f"{kind}/{g}"].append(rec)
            if kind == "mov" and g == "new":
                om = torch.from_numpy(objb[t]).to(dev)
                # 헤드별: new query 가 블록 t 의 물체 칸 (옛 자리) 에 준 질량
                ho = Wl.view(nl, nh, S, t + 1, S)[:, :, mq][:, :, :, t][:, :, :, om].sum(-1).mean(-1)   # (layer, head)
                AT_old_head.append(ho.cpu().numpy())
        if i % 50 == 0:
            print(f"[D] {i}/{NW}", flush=True)
    M.F.scaled_dot_product_attention = _sdpa
    summary["attn"] = {}
    print("\n== attention (TF 위치 t = 7, 헤드 평균): 지연 k = 0..7 (블록 7−k) 의 [같은 칸 | 3×3 이웃 | 블록 전체] 질량 — 층 1 · 3 · 6 · 12 (균등이면 같은 칸 1/2048)")
    for g, v in AT.items():
        if not v:
            continue
        m_ = np.mean(np.stack(v), 0)                                    # (3, layer, lag)
        summary["attn"][g] = dict(n=len(v), same=m_[0].tolist(), nb3=m_[1].tolist(), block=m_[2].tolist())
        print(f"  {g} (n {len(v)})")
        for li in (0, 2, 5, 11):
            print(f"    층 {li + 1:2d} 같은 칸 " + " ".join(f"{x:.3f}" for x in m_[0][li]) + " | 3×3 " + " ".join(f"{x:.3f}" for x in m_[1][li])
                  + " | 블록 " + " ".join(f"{x:.2f}" for x in m_[2][li]))
    if AT_old_head:
        oh = np.mean(np.stack(AT_old_head), 0)                          # (layer, head)
        summary["attn"]["new_to_oldpos_by_head"] = oh.tolist()
        print("  mov/new query → 블록 7 물체 칸 (옛 자리) 질량, 층별 헤드 최대 / 헤드 평균: "
              + " ".join(f"L{li + 1} {oh[li].max():.3f}/{oh[li].mean():.3f}" for li in range(oh.shape[0])))

json.dump(summary, open(os.path.join(a.out, f"{tag}.json"), "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
print(f"\n저장: {os.path.join(a.out, tag + '.json')}")
