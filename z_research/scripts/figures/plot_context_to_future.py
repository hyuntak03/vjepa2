#!/usr/bin/env python3
"""**문맥이 바뀌면 p 가 만드는 미래도 바뀌나** — 운동 · 외형 · 정체 (2026-09-26). 정체 자 (identity) 로 읽은 값만 쓴다. CPU 1–2 분.

세 질문 (v3 = RollOut_v3, 궤적 112 개 = 8 법칙 × 14, 궤적마다 외형만 다른 clip 14–28 개):
  1 운동   문맥에서 빨리 / 가속해서 / 어디서 가던 물체는 p 의 미래에서도 그만큼 다르게 가 있나   → fig_motion_*.png
  2 외형   궤적이 같고 색 · 모양 · 배경만 다른 물체를 p 가 다른 자리에 두나 (물리적으로는 같아야 한다) → fig_appearance_*.png
  3 정체   p 의 미래 물체가 문맥 물체와 같은 색 · 모양인가 (v3 튜블릿별 · v11 가림 조건별)          → fig_identity_*.png

기호 (창 C/P, 미래 튜블릿 t = 0 … P/2 − 1, τ = t + 1 = 문맥 끝에서 몇 튜블릿 뒤인가):
  x⁰        문맥 마지막 튜블릿 (샘플 30 · 31) 의 진실 위치 (px)
  v, a      문맥 샘플 (32 − C … 31) 의 진실 위치에 x(s) = x⁰ + v·(s − 30.5) + ½·a·(s − 30.5)² 를 맞춘 속도 · 가속 (튜블릿 단위로 환산).
            **문맥만** 본다 — p 가 볼 수 있던 것
  u         v 방향 단위 벡터 (문맥 끝 운동 방향)
  D         진실 변위  (x_t − x⁰)·u          D̂  p 변위 (x̂_t − x⁰)·u,  x̂ = p 자가 읽은 위치 − 치우침, **'있음' 칸만**
  궤적 평균  한 궤적의 외형 복사본들 평균 (궤적 = 분석 단위, 구간 = 궤적 bootstrap 1,000 회)

  운동   β_v  법칙 안에서 궤적 평균 D̂ 을 |v|·τ 에 회귀한 기울기 (법칙 평균을 뺀 뒤, 시작 위치 x⁰ 를 같이 넣음). 1 = 문맥 속도대로 · 0 = 속도와 무관
         β_x  같은 회귀의 시작 위치 계수 (px / 100 px)
         β_a  flat 세 법칙 (flat_a · flat_v · flat_d, 42 궤적, 같은 바닥) 에서 D̂ = c + β_v·|v|τ + β_a·½(a·u)τ² 의 가속 계수. 1 = 문맥 가속대로 · 0 = 등속으로만
         사건 법칙 (ledge 낙하 · wall 정지) 은 문맥에 없는 일이 미래에 생겨 β_v · β_x 에서 뺀다 (산점도에는 보인다)
  외형   σ_외형  궤적 안 D̂ 의 외형 간 표준편차 (궤적 평균)
         η²     궤적 평균을 뺀 D̂ 의 분산 중 모양 / 색 / 배경 평균이 설명하는 몫. 귀무 = 궤적 안에서 외형 라벨을 섞은 η² 의 95 분위
  정체   조합 맞음 = 57-way argmax 가 문맥 물체의 (모양, 색). v3 는 전체 clip 중 · v11 은 '있음' 칸 중. 우연 1.8 %

같은 자로 **실제 미래 프레임**을 읽은 선을 같이 그린다 — v3 는 z (창 전체를 보는 encoder), v11 은 h. p 가 도달해야 할 목표선이다.

  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  R3_DECODER=identity_r8 R3_OUT=identity_r8 $P z_research/scripts/figures/plot_context_to_future.py   → RollOutV3/figures/context_to_future/
"""
from __future__ import annotations
import csv, json, os, sys
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
sys.path.insert(0, str(ROOT / "z_research/scripts/analysis"))
from rollout3_paths import PRES, WIN, V11, V11_MID, ARCHIVE, check, is_identity, stamp   # noqa: E402

OUT = ARCHIVE / "context_to_future"
# 2026-09-28: 실제 미래 프레임 기준선 표현. 릴리즈 · full · prefix = z (창 전체 online encoder), ar = h (블록별 LN(target_encoder))
REF = os.environ.get("R3_REF", "z")
NO_V11 = os.environ.get("R3_NO_V11") == "1"
# 기준선 이름 (2026-09-28 사용자 지시: "real future frames" 대신 무엇으로 읽었는지를 쓴다).
#   z = context encoder 를 창 전체 (문맥 + 미래 프레임) 에 통과시켜 미래 칸을 같은 자로 읽은 값 · h = target encoder (ar 는 블록별)
REF_LABEL = {"z": "context encoder z", "h": "target encoder h"}.get(REF, REF)                 # v11 읽은 값 없이 v3 그림만
R, SPLIT, NB = 144.0, 32, 1000
CS, PS = (4, 8, 16, 32), (4, 8, 16, 32)
MAIN = (16, 32)
LAWS = ["arc", "flat_a", "flat_v", "flat_d", "ramp_a", "ramp_d", "ledge", "wall"]
EVENT = {"ledge", "wall"}
FLAT = ["flat_a", "flat_v", "flat_d"]
LCOL = dict(zip(LAWS, plt.get_cmap("tab10").colors))
COL = {"p": "#eb6834", "z": "#2a78d6", "h": "#1baf7a", "truth": "#222222"}
CCOL = {4: "#fdd0a2", 8: "#fd8d3c", 16: "#d94801", 32: "#7f2704"}          # 문맥 길이: 옅음 → 짙음 (한 색상)
rng = np.random.default_rng(0)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5})
DEC = f"decoder '{PRES.name}' (position + 57-way identity)"


# ------------------------------------------------------------------ 데이터

def load_v3():
    z = np.load(WIN / "readings.npz", allow_pickle=True); check(z, PRES)
    S = json.loads((PRES / "summary.json").read_text()); b = json.loads((PRES / "attn_bias_px.json").read_text())
    idx = {r["video_id"]: r for r in csv.DictReader((ROOT / "data_csv/rollout_v3/index.csv").open())}
    POS = z["role"] == "roll"
    L = z["truth"][POS] * R + R
    law = z["scenario"][POS]; vid = z["video_id"][POS]
    key = [f"{s}|" + ",".join(map(str, np.round(l, 0).astype(int).ravel())) for s, l in zip(law, L)]
    _, tid = np.unique(key, return_inverse=True)
    shape = np.array([idx[v]["shape_pre"] for v in vid]); color = np.array([idx[v]["color_pre"] for v in vid])
    env = np.array([idx[v]["env"] for v in vid])
    combo = np.array([S["shapes"].index(s) * 8 + S["colors"].index(c) for s, c in zip(shape, color)])
    return dict(z=z, POS=POS, L=L, law=law, tid=tid, nt=tid.max() + 1, shape=shape, color=color, env=env, combo=combo, S=S, b=b)


def kin(L, c):
    """문맥 샘플 32 − c … 31 에 2 차식 → (x⁰, v, a) 튜블릿 단위. x⁰ = 샘플 30 · 31 평균."""
    s = np.arange(SPLIT - c, SPLIT) - 30.5
    A = np.stack([np.ones_like(s), s, 0.5 * s ** 2], 1)                     # (c, 3)
    coef = np.linalg.lstsq(A, L[:, SPLIT - c:SPLIT].transpose(1, 0, 2).reshape(c, -1), rcond=None)[0]
    coef = coef.reshape(3, len(L), 2).transpose(1, 0, 2)                      # (n, 3, 2)
    x0 = L[:, 30:32].mean(1)
    return x0, coef[:, 1] * 2, coef[:, 2] * 4                                 # 샘플 → 튜블릿 (Δs = 2)


def window(V, c, p, rep):
    """한 창의 clip 단위 양: D (진실) · Dh (rep 읽은 값, '있음' 아니면 nan) · 운동 회귀 변수."""
    z, POS, L = V["z"], V["POS"], V["L"]
    tp = p // 2; tau = np.arange(1, tp + 1, dtype=float)
    gt = L[:, SPLIT:SPLIT + p].reshape(len(L), tp, 2, 2).mean(2)
    x0, v, a = kin(L, c)
    sp = np.linalg.norm(v, axis=-1); u = v / np.maximum(sp, 1e-6)[:, None]
    r = z[f"C{c}_P{p}_{rep}"][POS]
    say = r[..., 2] > 0
    xy = r[..., :2] * R + R - np.array(V["b"][rep.split("@")[-1]], np.float32)
    D = ((gt - x0[:, None]) * u[:, None]).sum(-1)
    Dh = np.where(say, ((xy - x0[:, None]) * u[:, None]).sum(-1), np.nan)
    ok = z[f"C{c}_P{p}_{rep}_prob"][POS].argmax(-1) == V["combo"][:, None]
    return dict(tp=tp, tau=tau, D=D, Dh=Dh, say=say, ok=ok, sp=sp, apar=(a * u).sum(-1), x0=x0[:, 0], pred=r, xy=xy, gt=gt)


def traj_mean(V, a):
    """clip (n, …) → 궤적 (nt, …) 평균 (nan 무시)."""
    fin = np.isfinite(a)
    sm = np.zeros((V["nt"],) + a.shape[1:]); ct = np.zeros_like(sm)
    np.add.at(sm, V["tid"], np.where(fin, a, 0.0)); np.add.at(ct, V["tid"], fin)
    with np.errstate(all="ignore"):
        return np.where(ct > 0, sm / ct, np.nan)


def tfirst(V, a):
    return a[np.unique(V["tid"], return_index=True)[1]]


# ------------------------------------------------------------------ 운동 회귀

def fit_speed(y, spt, x0, g):
    """법칙 안 회귀: y = c_법칙 + β_v · |v|τ + β_x · x⁰ → (β_v, β_x per 100 px). g = 법칙 번호 (사건 법칙은 −1 로 빠짐)."""
    m = np.isfinite(y) & (g >= 0)
    if m.sum() < 10:
        return np.nan, np.nan
    G = g[m]; n_ = np.bincount(G, minlength=len(LAWS)).astype(float)
    dm = lambda v: v - (np.bincount(G, v, minlength=len(LAWS)) / np.maximum(n_, 1))[G]
    X = np.stack([dm(spt[m]), dm(x0[m])], 1)
    beta = np.linalg.lstsq(X, dm(y[m]), rcond=None)[0]
    return beta[0], beta[1] * 100


def fit_accel(y, spt, acc, flat):
    """flat 세 법칙: y = c + β_v · |v|τ + β_a · ½ a τ² → β_a."""
    m = np.isfinite(y) & flat
    if m.sum() < 10:
        return np.nan
    X = np.stack([np.ones(m.sum()), spt[m], acc[m]], 1)
    return np.linalg.lstsq(X, y[m], rcond=None)[0][2]


def boot_ix(law, nb):
    """궤적 bootstrap 색인 (법칙 안에서 궤적을 재표집) — 한 창의 모든 t · 표현이 같은 재표집을 쓴다."""
    groups = [np.where(law == g)[0] for g in np.unique(law)]
    return [np.arange(len(law))] + [np.concatenate([gg[rng.integers(0, len(gg), len(gg))] for gg in groups]) for _ in range(nb)]


def ci3(vals):
    v = np.array(vals, float)
    lo, hi = np.nanpercentile(v[1:], [2.5, 97.5]) if np.isfinite(v[1:]).any() else (np.nan, np.nan)
    return v[0], lo, hi


def motion_coefs(V, c, p, reps=("p", REF)):
    """창 하나: t 마다 β_v · β_x · β_a (궤적 bootstrap) — 진실 · p · z."""
    law_t = tfirst(V, V["law"])
    out = {}
    base = window(V, c, p, "p")
    spt = tfirst(V, base["sp"])[:, None] * base["tau"][None]
    acc = 0.5 * tfirst(V, base["apar"])[:, None] * base["tau"][None] ** 2
    x0 = tfirst(V, base["x0"])
    ys = {"truth": tfirst(V, base["D"])}
    for rep in reps:
        ys[rep] = traj_mean(V, window(V, c, p, rep)["Dh"])
    g = np.array([LAWS.index(l) if l not in EVENT else -1 for l in law_t]); flat = np.isin(law_t, FLAT)
    IX = boot_ix(law_t, NB if (c, p) == MAIN else 300)
    for name, Y in ys.items():
        rows = []
        for t in range(base["tp"]):
            y = Y[:, t]
            sv = [fit_speed(y[ix], spt[ix, t], x0[ix], g[ix]) for ix in IX]
            sa = [fit_accel(y[ix], spt[ix, t], acc[ix, t], flat[ix]) for ix in IX]
            rows.append(dict(bv=ci3([v[0] for v in sv]), bx=ci3([v[1] for v in sv]), ba=ci3(sa)))
        out[name] = rows
    return out, dict(spt=spt, acc=acc, x0=x0, law=law_t, Y=ys, tp=base["tp"])


# ------------------------------------------------------------------ 그림 1 운동

def band(ax, xs, rows, key, color, label, ls="-", lw=1.8):
    v = np.array([r[key] for r in rows], float)
    ax.plot(xs, v[:, 0], color=color, ls=ls, lw=lw, marker="o" if ls == "-" else None, ms=3, label=label)
    if ls == "-":
        ax.fill_between(xs, v[:, 1], v[:, 2], color=color, alpha=0.18, lw=0)


def fig_motion_main(V, coefs, aux):
    c, p = MAIN; tp = aux["tp"]; tt = np.arange(tp)
    f = plt.figure(figsize=(15.5, 8.6)); gs = f.add_gridspec(2, 4, hspace=0.42, wspace=0.28)
    for k, t in enumerate([1, 4, 8, 15]):
        a = f.add_subplot(gs[0, k])
        D, Dh = aux["Y"]["truth"][:, t], aux["Y"]["p"][:, t]
        for lw in LAWS:
            m = aux["law"] == lw
            a.scatter(D[m], Dh[m], s=16, color=LCOL[lw], label=lw, edgecolor="black" if lw in EVENT else "none", lw=0.5,
                      marker="X" if lw in EVENT else "o")
        lim = [min(np.nanmin(D), np.nanmin(Dh), 0) - 5, max(np.nanmax(D), np.nanmax(Dh)) + 5]
        a.plot(lim, lim, color="#888888", lw=0.8, ls="--"); a.axhline(0, color="#bbbbbb", lw=0.8, ls=":")
        a.set_xlim(lim); a.set_ylim(lim); a.set_aspect("equal")
        a.set_xlabel("true displacement from last seen (px)"); a.set_ylabel("p's displacement (px)" if k == 0 else "")
        a.set_title(f"future tubelet t{t}   (one dot = one trajectory)", fontsize=9, fontweight="bold")
        a.text(0.03, 0.97, "dashed = follows context exactly\ndotted = copy (stays put)", transform=a.transAxes, fontsize=7, va="top", color="#555555")
        if k == 0:
            hs, ls_ = a.get_legend_handles_labels()
    f.legend(hs, ls_, loc="upper center", ncol=8, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.955), title="law (X = event law: ledge falls, wall stops)", title_fontsize=8)
    panels = [("bv", "(e) SPEED: β_v within law\n1 = as fast as the context · 0 = ignores speed", (-0.2, 1.4)),
              ("ba", "(f) ACCELERATION: β_a, flat_a / flat_v / flat_d\n1 = as the context · 0 = constant speed", (-1.0, 2.0)),
              ("bx", "(g) START POSITION: β_x\npx of displacement per 100 px of start x", None)]
    for k, (key, title, yl) in enumerate(panels):
        a = f.add_subplot(gs[1, k])
        band(a, tt, coefs["truth"], key, COL["truth"], "truth (what the future needs)", ls="--", lw=1.4)
        band(a, tt, coefs[REF], key, COL["z"], REF_LABEL)
        band(a, tt, coefs["p"], key, COL["p"], "p (predictor)")
        a.axhline(0, color="#bbbbbb", lw=0.8)
        if key != "bx":
            a.axhline(1, color="#bbbbbb", lw=0.8, ls=":")
        if yl:
            a.set_ylim(*yl)
        a.set_xlabel("future tubelet"); a.set_title(title, fontsize=8.8, fontweight="bold")
        a.spines[["top", "right"]].set_visible(False)
        if k == 0:
            a.legend(fontsize=7.5, frameon=False, loc="lower left")
        if key == "ba":
            a.text(0.02, 0.97, "early tubelets: the acceleration term is < 2 px,\nbelow readout noise — z cannot measure it either",
                   transform=a.transAxes, fontsize=7, va="top", color="#555555")
    a = f.add_subplot(gs[1, 3])
    for j, lw in enumerate(FLAT):
        m = aux["law"] == lw
        for name, ls in (("truth", "--"), ("p", "-"), (REF, ":")):
            y = aux["Y"][name][m] - aux["spt"][m]
            a.plot(tt, np.nanmean(y, 0), color=["#2a78d6", "#888888", "#eb6834"][j], ls=ls, lw=1.6,
                   label=f"{lw} {name}" if name != REF else None)
    a.axhline(0, color="#bbbbbb", lw=0.8)
    a.set_xlabel("future tubelet"); a.set_ylabel("displacement − constant-speed extrapolation (px)")
    a.set_title("(h) flat_a / flat_v / flat_d: displacement beyond\nconstant speed (dashed truth · solid p · dotted z)", fontsize=8.8, fontweight="bold")
    h_, l_ = a.get_legend_handles_labels()
    h_ += [plt.Line2D([], [], color="#888888", ls=":", lw=1.6, label=f"{REF_LABEL} (each law)")]
    a.legend(handles=h_, fontsize=6.8, frameon=False, ncol=2, loc="lower left"); a.spines[["top", "right"]].set_visible(False)
    f.suptitle(f"RollOut v3, window C{c}/P{p} — does p's future change with the context's motion?   "
               f"112 trajectories × 14–28 appearances · bands = 95% trajectory bootstrap · {DEC}", fontsize=10)
    f.subplots_adjust(left=0.05, right=0.985, top=0.86, bottom=0.07)
    f.savefig(OUT / f"detail_motion_C{c}_P{p}.png", dpi=150); plt.close(f)


def fig_motion_windows(W):
    f, ax = plt.subplots(3, 4, figsize=(15.5, 9.2), sharey="row")
    rows = [("bv", "β_v  speed", (-0.2, 1.4)), ("ba", "β_a  acceleration", (-1.5, 2.5)), ("bx", "β_x  start position (px / 100 px)", None)]
    for j, p in enumerate(PS):
        for i, (key, lab, yl) in enumerate(rows):
            a = ax[i, j]
            for c in CS:
                co, _ = W[(c, p)]; tt = np.arange(len(co["p"]))
                band(a, tt, co["p"], key, CCOL[c], f"p, context C{c}")
                v = np.array([r[key][0] for r in co[REF]]); a.plot(tt, v, color=CCOL[c], lw=0.9, ls=":")
            v = np.array([r[key][0] for r in W[(16, p)][0]["truth"]]); a.plot(np.arange(len(v)), v, color="black", lw=1.2, ls="--", label="truth (C16)")
            a.axhline(0, color="#bbbbbb", lw=0.7)
            if yl:
                a.set_ylim(*yl)
            if i == 0:
                a.set_title(f"prediction P = {p} frames", fontsize=9.5, fontweight="bold")
            if j == 0:
                a.set_ylabel(lab)
            if i == 2:
                a.set_xlabel("future tubelet")
            a.spines[["top", "right"]].set_visible(False)
    h_, l_ = ax[0, 3].get_legend_handles_labels()
    h_ += [plt.Line2D([], [], color="#888888", lw=0.9, ls=":", label=f"{REF_LABEL} (colour = C)")]
    ax[0, 3].legend(handles=h_, fontsize=7, frameon=False, loc="lower left")
    f.suptitle("RollOut v3, all 16 windows — speed / acceleration / start-position coefficients of p's future\n"
               f"solid + band = p (colour = context length C) · dotted = {REF_LABEL} · dashed = truth", fontsize=10)
    f.subplots_adjust(left=0.06, right=0.99, top=0.9, bottom=0.07, hspace=0.3, wspace=0.08)
    f.savefig(OUT / "detail_motion_windows.png", dpi=150); plt.close(f)


# ------------------------------------------------------------------ 그림 2 외형

def appearance(V, c, p, rep, nperm=200):
    w = window(V, c, p, rep); tp = w["tp"]
    res = {"sd": np.full(tp, np.nan), "eta": {k: np.full(tp, np.nan) for k in ("shape", "color", "env")},
           "null": {k: np.full(tp, np.nan) for k in ("shape", "color", "env")},
           "by_shape": {s: np.full(tp, np.nan) for s in np.unique(V["shape"])},
           "say_shape": {s: np.full(tp, np.nan) for s in np.unique(V["shape"])},
           "say_color": {s: np.full(tp, np.nan) for s in np.unique(V["color"])},
           "by_env": {s: np.full(tp, np.nan) for s in np.unique(V["env"])},
           "say_env": {s: np.full(tp, np.nan) for s in np.unique(V["env"])}}
    tid = V["tid"]
    tm_all = traj_mean(V, w["Dh"])
    for t in range(tp):
        y = w["Dh"][:, t]; m = np.isfinite(y)
        tm = tm_all[:, t]; r = y - tm[tid]
        nj = np.bincount(tid[m], minlength=V["nt"])
        var = np.bincount(tid[m], r[m] ** 2, minlength=V["nt"]) / np.maximum(nj, 1)
        res["sd"][t] = np.sqrt(var[nj >= 3]).mean() if (nj >= 3).any() else np.nan
        rr, tt_ = r[m], tid[m]; tot = (rr ** 2).sum()
        base_ord = np.argsort(tt_, kind="stable")
        for k in ("shape", "color", "env"):
            _, code = np.unique(V[k][m], return_inverse=True); K = code.max() + 1
            def eta(cd):
                n_ = np.bincount(cd, minlength=K); mu = np.bincount(cd, rr, minlength=K) / np.maximum(n_, 1)
                return (n_ * mu ** 2).sum() / max(tot, 1e-9)
            res["eta"][k][t] = eta(code)
            nul = []
            for _ in range(nperm):
                perm = np.lexsort((rng.random(len(tt_)), tt_))          # 궤적 안에서만 섞는다
                cd = np.empty_like(code); cd[base_ord] = code[perm]
                nul.append(eta(cd))
            res["null"][k][t] = np.percentile(nul, 95)
        for s in res["by_shape"]:
            mm = m & (V["shape"] == s); res["by_shape"][s][t] = r[mm].mean() if mm.any() else np.nan
            res["say_shape"][s][t] = w["say"][V["shape"] == s, t].mean()
        for s in res["say_color"]:
            res["say_color"][s][t] = w["say"][V["color"] == s, t].mean()
        for s in res["by_env"]:
            mm = m & (V["env"] == s); res["by_env"][s][t] = r[mm].mean() if mm.any() else np.nan
            res["say_env"][s][t] = w["say"][V["env"] == s, t].mean()
    return res


def fig_appearance_main(A):
    c, p = MAIN; P_, Z_ = A[(c, p, "p")], A[(c, p, REF)]; tt = np.arange(p // 2)
    f, ax = plt.subplots(2, 3, figsize=(15.0, 8.4)); ax = ax.ravel()
    cm = plt.get_cmap("tab10")
    a = ax[0]
    a.plot(tt, P_["sd"], color=COL["p"], marker="o", ms=3, label="p"); a.plot(tt, Z_["sd"], color=COL["z"], marker="o", ms=3, label=REF_LABEL)
    a.set_ylabel("SD across appearances (px)")
    a.set_title("(a) same trajectory, different looks:\nhow far apart does p put them?", fontsize=9, fontweight="bold")
    a.legend(fontsize=7.5, frameon=False)
    a = ax[1]
    for k, ls in (("env", "-"), ("shape", "--"), ("color", ":")):
        a.plot(tt, 100 * P_["eta"][k], color=COL["p"], ls=ls, lw=1.8, label=f"p · {'background' if k == 'env' else k}")
        a.plot(tt, 100 * Z_["eta"][k], color=COL["z"], ls=ls, lw=1.2, label=f"{REF} · {'background' if k == 'env' else k}")
        a.plot(tt, 100 * P_["null"][k], color="#aaaaaa", ls=ls, lw=0.8)
    a.set_ylabel("% of within-trajectory variance")
    a.set_title("(b) which look moves p's position?\n(grey = 95% null, labels shuffled within trajectory)", fontsize=9, fontweight="bold")
    a.legend(fontsize=7, frameon=False, ncol=2)
    for a, key, name in ((ax[2], "by_env", "BACKGROUND"), (ax[3], "by_shape", "SHAPE")):
        for i, (g_, v) in enumerate(P_[key].items()):
            a.plot(tt, v, color=cm(i), lw=1.6, label=g_)
        a.axhline(0, color="#bbbbbb", lw=0.8)
        a.set_ylabel("p's displacement − trajectory mean (px)")
        a.set_title(f"({'c' if key == 'by_env' else 'd'}) p's displacement by {name}\n(+ = further along the motion)", fontsize=9, fontweight="bold")
        a.legend(fontsize=7, frameon=False, ncol=2)
    for a, key, name in ((ax[4], "say_shape", "SHAPE"), (ax[5], "say_env", "BACKGROUND")):
        for i, (g_, v) in enumerate(P_[key].items()):
            a.plot(tt, 100 * v, color=cm(i), lw=1.5, label=g_)
        a.set_ylabel("p reads 'present' (%)")
        a.set_title(f"({'e' if key == 'say_shape' else 'f'}) object still 'present' in p's future, by {name}", fontsize=9, fontweight="bold")
        a.legend(fontsize=7, frameon=False, ncol=2, loc="lower left")
    for a in ax:
        a.set_xlabel("future tubelet"); a.spines[["top", "right"]].set_visible(False)
    f.suptitle(f"RollOut v3, window C{c}/P{p} — does the object's look (shape, colour, background) change p's future? Physically it should not.\n{DEC}", fontsize=10)
    f.subplots_adjust(left=0.06, right=0.99, top=0.88, bottom=0.07, wspace=0.28, hspace=0.42)
    f.savefig(OUT / f"detail_appearance_C{c}_P{p}.png", dpi=150); plt.close(f)


def fig_appearance_windows(A):
    f, ax = plt.subplots(1, 4, figsize=(15.5, 4.0), sharey=True)
    for j, p in enumerate(PS):
        a = ax[j]; tt = np.arange(p // 2)
        for c in CS:
            P_, Z_ = A[(c, p, "p")], A[(c, p, REF)]
            a.plot(tt, 100 * sum(P_["eta"].values()), color=CCOL[c], lw=1.8, marker="o", ms=3, label=f"p, C{c}")
            a.plot(tt, 100 * sum(Z_["eta"].values()), color=CCOL[c], lw=0.9, ls=":")
        a.set_title(f"P = {p} frames", fontsize=9.5, fontweight="bold"); a.set_xlabel("future tubelet")
        a.spines[["top", "right"]].set_visible(False)
    ax[0].set_ylabel("% of within-trajectory variance\nexplained by shape + colour + background")
    h_, l_ = ax[0].get_legend_handles_labels()
    h_ += [plt.Line2D([], [], color="#888888", lw=0.9, ls=":", label=f"{REF_LABEL} (colour = C)")]
    ax[0].legend(handles=h_, fontsize=7.5, frameon=False)
    f.suptitle("RollOut v3, all 16 windows — how much of p's position spread (within a trajectory) is appearance?   solid = p · dotted = z (real future)", fontsize=10)
    f.subplots_adjust(left=0.07, right=0.99, top=0.84, bottom=0.14, wspace=0.08)
    f.savefig(OUT / "detail_appearance_windows.png", dpi=150); plt.close(f)


# ------------------------------------------------------------------ 그림 3 정체

def id_v3(V, c, p, rep):
    z = V["z"]; P_ = z[f"C{c}_P{p}_{rep}_prob"][V["POS"]].astype(np.float32)
    none = V["S"]["none_index"]
    pm = P_[..., :none].reshape(*P_.shape[:-1], 7, 8)
    ps, pc = pm.sum(-1).argmax(-1), pm.sum(-2).argmax(-1)
    s_true, c_true = V["combo"] // 8, V["combo"] % 8
    say = P_.argmax(-1) != none
    ok = P_.argmax(-1) == V["combo"][:, None]
    sh = ps == s_true[:, None]; co = pc == c_true[:, None]
    return dict(combo=ok.mean(0), shape=sh.mean(0), color=co.mean(0),
                err=dict(none=(~say).mean(0), shape_only=(say & ~ok & ~sh & co).mean(0), color_only=(say & ~ok & sh & ~co).mean(0),
                         both=(say & ~ok & ~sh & ~co).mean(0)),
                by_law={lw: ok[V["law"] == lw].mean(0) for lw in LAWS})


def fig_identity_v3(V):
    c, p = MAIN; I = {rep: id_v3(V, c, p, rep) for rep in ("p", REF)}; tt = np.arange(p // 2)
    f, ax = plt.subplots(1, 4, figsize=(15.5, 4.3))
    a = ax[0]
    for k, ls in (("combo", "-"), ("shape", "--"), ("color", ":")):
        a.plot(tt, 100 * I["p"][k], color=COL["p"], ls=ls, lw=1.8, label=f"p · {k}")
        a.plot(tt, 100 * I[REF][k], color=COL["z"], ls=ls, lw=1.2, label=f"{REF} · {k}")
    a.axhline(100 / 56, color="#999999", lw=0.8, ls="-."); a.text(15, 100 / 56 + 1.5, "combo chance 1.8%", fontsize=7, color="#777777", ha="right")
    a.set_ylim(0, 102); a.set_ylabel("correct (%) — all clips")
    a.set_title(f"(a) is p's future object the same object?\n({REF_LABEL} = same readout on the real frames)", fontsize=8.8, fontweight="bold")
    a.legend(fontsize=6.8, frameon=False, ncol=2, loc="lower left")
    a = ax[1]; e = I["p"]["err"]; bottom = np.zeros(len(tt))
    for k, col_, lab in (("shape_only", "#eb6834", "wrong shape only"), ("color_only", "#2a78d6", "wrong colour only"),
                         ("both", "#7a4bb3", "both wrong"), ("none", "#bbbbbb", "says 'none'")):
        a.bar(tt, 100 * e[k], bottom=100 * bottom, color=col_, label=lab, width=0.8); bottom += e[k]
    a.set_ylabel("share of all clips (%)"); a.set_title("(b) when p's identity is wrong, what is wrong?", fontsize=8.8, fontweight="bold")
    a.legend(fontsize=7, frameon=False)
    a = ax[2]
    for lw in LAWS:
        a.plot(tt, 100 * I["p"]["by_law"][lw], color=LCOL[lw], lw=1.5, label=lw)
    a.set_ylim(0, 102); a.set_ylabel("p combo correct (%)"); a.set_title("(c) by law", fontsize=8.8, fontweight="bold")
    a.legend(fontsize=7, frameon=False, ncol=2, loc="lower left")
    a = ax[3]
    for pp in PS:
        for cc in CS:
            v = id_v3(V, cc, pp, "p")["combo"]
            a.plot(np.arange(len(v)), 100 * v, color=CCOL[cc], lw=1.2, alpha=0.9, label=f"C{cc}" if pp == 32 else None)
    zc = np.mean([id_v3(V, cc, 32, REF)["combo"] for cc in CS], 0)
    a.plot(np.arange(16), 100 * zc, color=COL["z"], lw=1.2, ls="--", label=f"{REF} (mean of C)")
    a.set_ylim(0, 102); a.set_ylabel("p combo correct (%)"); a.set_title("(d) all 16 windows (colour = context length C)", fontsize=8.8, fontweight="bold")
    a.legend(fontsize=7, frameon=False, loc="lower left")
    for a in ax:
        a.set_xlabel("future tubelet"); a.spines[["top", "right"]].set_visible(False)
    f.suptitle(f"RollOut v3 (no occlusion) — does p keep the object's identity (shape × colour, 56 combos) into the future?   {DEC}", fontsize=10)
    f.subplots_adjust(left=0.05, right=0.99, top=0.8, bottom=0.13, wspace=0.3)
    f.savefig(OUT / "detail_identity_v3.png", dpi=150); plt.close(f)
    return I


def fig_identity_v11():
    S = json.loads((PRES / "summary.json").read_text())
    Z = {"last": np.load(V11 / "readings.npz", allow_pickle=True), "mid": np.load(V11_MID / "readings.npz", allow_pickle=True)}
    check(Z["last"], PRES); check(Z["mid"], PRES)
    f, ax = plt.subplots(2, 3, figsize=(15.0, 7.6), sharey=True)
    vals = {}
    for i, (tm, ks) in enumerate((("last", range(0, 5)), ("mid", range(1, 5)))):
        z = Z[tm]; k = z["sym_k"].astype(int)
        mo = np.array(["static" if c_.startswith("static") else ("flat" if "flat" in c_ else "ramp") for c_ in z["condition"]])
        combo = np.array([S["shapes"].index(s) * 8 + S["colors"].index(c_) if s in S["shapes"] else -1 for s, c_ in zip(z["shape_pre"], z["color_pre"])])
        okp = z["p_prob"][:, -8:].argmax(-1) == combo[:, None]; say = z["p"][:, -8:, 2] > 0
        okh = z["h_prob"][:, -8:].argmax(-1) == combo[:, None]; sayh = z["h"][:, -8:, 2] > 0
        for j, m in enumerate(("static", "flat", "ramp")):
            a = ax[i, j]; tt = np.arange(1, 7)
            for kk in ks:
                g = z["obj"] & (mo == m) & (k == kk)
                acc = [okp[g, t][say[g, t]].mean() if say[g, t].any() else np.nan for t in tt]
                col_ = "#222222" if kk == 0 else plt.get_cmap("Oranges")(0.3 + 0.17 * kk)
                a.plot(tt, 100 * np.array(acc), color=col_, lw=1.8, marker="o", ms=3, label="no occlusion" if kk == 0 else f"k={kk}")
                acch = [okh[g, t][sayh[g, t]].mean() if sayh[g, t].any() else np.nan for t in tt]
                if kk == (0 if tm == "last" else 1):
                    a.plot(tt, 100 * np.array(acch), color=COL["h"], lw=1.2, ls="--", label="target encoder h")
                vals[f"{tm}|{m}|k{kk}"] = dict(p_combo_given_present=[None if np.isnan(x) else round(float(x), 3) for x in acc],
                                               p_present=[round(float(say[g, t].mean()), 3) for t in tt])
            a.axhline(100 / 56, color="#999999", lw=0.8, ls="-.")
            a.set_ylim(0, 102); a.set_xticks(tt); a.set_xticklabels([f"t{t}" for t in tt])
            a.set_title(f"{m} · {'occluded at context END' if tm == 'last' else 'occluded MID-context (reappears in context)'}", fontsize=9, fontweight="bold")
            if j == 0:
                a.set_ylabel("combo correct among 'present' (%)")
            a.spines[["top", "right"]].set_visible(False)
            a.legend(fontsize=7, frameon=False, loc="lower left")
    f.suptitle(f"IntPhysGen v11 — when p's future has an object, is it the SAME object (shape × colour)?   chance 1.8% (dash-dot) · t7 left out (object leaving frame)   {DEC}", fontsize=10)
    f.subplots_adjust(left=0.06, right=0.99, top=0.9, bottom=0.07, hspace=0.35, wspace=0.08)
    f.savefig(OUT / "detail_identity_v11.png", dpi=150); plt.close(f)
    return vals


# ------------------------------------------------------------------ 논문 판 (1_ … 4_) — 진짜 궤적 (점선) 과 p 궤적 (실선) 을 겹친다
# 2026-09-27 사용자 지시: 범례 없는 선을 모두 표기, 논문 스타일 (CLAUDE.md §8-4: Nimbus Roman · pdf.fonttype 42 · 7.0 in · 패널 라벨 아래 가운데 ·
# 서술 문장은 그림에 넣지 않고 캡션 = 폴더 README 로). 색은 검증된 slot 1–3 만, 넷째부터는 잉크색 + 표식으로 구분한다.

PAPER = {"font.family": "Nimbus Roman", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 6.8,
         "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.6,
         "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
         "lines.linewidth": 1.4, "legend.frameon": False, "legend.handlelength": 2.4, "legend.borderaxespad": 0.3,
         "axes.spines.top": False, "axes.spines.right": False}
S1, S2, S3, INK, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#3a3a3a", "#8c8c8c"
GT = (0, (3.5, 2.0))                                                        # 진짜 = 점선
XT, XLAB = [1, 4, 8, 12, 16], "Tubelets after context end"
REAL_COLOR = {"blue": "#1f5fd6", "cyan": "#12a9bd", "green": "#2e9e2e", "magenta": "#c42ab8", "orange": "#f07f12",
              "purple": "#6a3cb0", "red": "#d62728", "yellow": "#c9a800"}   # 색 패널: 선 색 = 그 물체의 색 (색이 곧 대상)
SHAPE_MK = {"capsule": "o", "cone": "^", "cube": "s", "cylinder": "D", "pyramid": "v", "sphere": "h", "torus": "X"}
ENV_NAME = {"checker_steel": "checker steel", "concrete_brick": "concrete brick", "grass_hewn": "grass", "hex_rust": "hex rust"}


def plabel(a, s, y=-0.30):
    a.text(0.5, y, s, transform=a.transAxes, ha="center", va="top", fontsize=9)


def save(f, name):
    f.savefig(OUT / f"{name}.png", dpi=300); f.savefig(OUT / f"{name}.pdf"); plt.close(f)


def style_handles():
    return [plt.Line2D([], [], color=INK, ls=GT, lw=1.2, label="ground truth"),
            plt.Line2D([], [], color=INK, lw=1.6, label="predictor p (decoded)")]


def simple_data(V, c, p):
    w = window(V, c, p, "p"); law = tfirst(V, V["law"]); ok = ~np.isin(law, list(EVENT))
    return dict(w=w, law=law, ok=ok, D=tfirst(V, w["D"]), Dh=traj_mean(V, w["Dh"]), sp=tfirst(V, w["sp"]), x0=tfirst(V, w["x0"]),
                tt=np.arange(1, w["tp"] + 1))


def speed_groups(V):
    """문맥 속도 세 무리 — 주 창 (문맥 16 프레임) 의 문맥 끝 속도 3 분위로 **궤적마다 한 번** 정한다 (모든 창에서 같은 궤적이 같은 무리)."""
    d = simple_data(V, *MAIN); ok = d["ok"]; lo, hi = np.percentile(d["sp"][ok], [33.3, 66.7])
    g = [("slow", ok & (d["sp"] < lo), S1), ("medium", ok & (d["sp"] >= lo) & (d["sp"] <= hi), S3), ("fast", ok & (d["sp"] > hi), S2)]
    lab = {"slow": f"slow  (< {lo:.1f} px/tubelet)", "medium": f"medium  ({lo:.1f}–{hi:.1f})", "fast": f"fast  (> {hi:.1f})"}
    return [(lab[n] + f", n = {m.sum()}", m, c_) for n, m, c_ in g]


def ratio(w, mm):
    """같은 clip 들에서 p 가 간 거리 합 ÷ 진짜 간 거리 합 (%) — '있음' 칸만."""
    num = np.array([np.nansum(w["Dh"][mm, t]) for t in range(w["tp"])])
    den = np.array([w["D"][mm, t][np.isfinite(w["Dh"][mm, t])].sum() for t in range(w["tp"])])
    return 100 * num / den


def fig_simple_motion(V):
    """1_motion. 곡선 하나 = 무리 안 궤적들의 평균 변위 (clip → 궤적 평균 → 무리 평균, 2 단계). (c) 만 clip 합의 비율.
    범례는 패널 위 (패널 주제 = 범례 제목) — 선과 겹치지 않게 (2026-09-27)."""
    c, p = MAIN; d = simple_data(V, c, p); tt = d["tt"]; SG = speed_groups(V)
    with plt.rc_context(PAPER):
        f, ax = plt.subplots(1, 3, figsize=(7.0, 3.15))
        leg = dict(loc="lower left", bbox_to_anchor=(0.0, 1.02), borderaxespad=0.0, title_fontsize=7.4, alignment="left")
        a = ax[0]
        for lab, m, col in SG:
            a.plot(tt, np.nanmean(d["D"][m], 0), color=col, ls=GT, lw=1.1)
            a.plot(tt, np.nanmean(d["Dh"][m], 0), color=col, lw=1.6, label=lab)
        a.set_ylabel("Displacement from last observed position (px)")
        a.legend(title="Context speed (terciles, 84 trajectories)", **leg)
        a = ax[1]; band = (d["sp"] >= 6.5) & (d["sp"] <= 9.5)
        for lw, col, name in (("flat_a", S2, "accelerating"), ("flat_v", S3, "constant speed"), ("flat_d", S1, "decelerating")):
            m = (d["law"] == lw) & band
            a.plot(tt, np.nanmean(d["D"][m], 0), color=col, ls=GT, lw=1.1)
            a.plot(tt, np.nanmean(d["Dh"][m], 0), color=col, lw=1.6, label=f"{name}, n = {m.sum()}")
        a.legend(title="Context acceleration (flat floor,\nspeed matched 6.5–9.5 px/tubelet)", **leg)
        for a_ in ax[:2]:
            a_.set_ylim(0, 165)
        a = ax[2]; w = d["w"]; okc = ~np.isin(V["law"], list(EVENT))
        q1, q2 = np.percentile(d["x0"][d["ok"]], [33.3, 66.7]); x0c = w["x0"]
        for name, mm, col in (("left third", okc & (x0c < q1), S1), ("middle third", okc & (x0c >= q1) & (x0c <= q2), S3),
                              ("right third", okc & (x0c > q2), S2)):
            a.plot(tt, ratio(w, mm), color=col, lw=1.6, label=name)
        a.axhline(100, color=INK, ls=GT, lw=1.1)
        a.set_ylim(0, 165); a.set_ylabel("Predicted ÷ true displacement (%)")
        a.legend(title="Screen position at context end", **leg)
        for k, a_ in enumerate(ax):
            a_.set_xticks(XT); a_.set_xlim(0.5, 16.5); a_.set_xlabel(XLAB); plabel(a_, f"({'abc'[k]})", y=-0.24)
        f.legend(handles=style_handles(), loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
        f.subplots_adjust(left=0.07, right=0.99, top=0.66, bottom=0.2, wspace=0.34)
        save(f, "1_motion")


def fig_simple_look(V):
    c, p = MAIN; w = window(V, c, p, "p"); tt = np.arange(1, w["tp"] + 1); okc = ~np.isin(V["law"], list(EVENT))
    with plt.rc_context(PAPER):
        f, ax = plt.subplots(1, 3, figsize=(7.0, 2.7), sharey=True)
        for g, col in zip(sorted(np.unique(V["env"])), (S1, S2, S3, INK)):
            ax[0].plot(tt, ratio(w, okc & (V["env"] == g)), color=col, lw=1.4, label=ENV_NAME.get(g, g))
        for g in sorted(np.unique(V["shape"])):
            ax[1].plot(tt, ratio(w, okc & (V["shape"] == g)), color=INK, lw=0.9, marker=SHAPE_MK[g], ms=3.2, markevery=3,
                       mfc="white", mew=0.7, label=g)
        for g in sorted(np.unique(V["color"])):
            ax[2].plot(tt, ratio(w, okc & (V["color"] == g)), color=REAL_COLOR[g], lw=1.2, label=g)
        for k, (a, ttl, nc) in enumerate(zip(ax, ("Background", "Object shape", "Object colour"), (1, 2, 2))):
            a.axhline(100, color=INK, ls=GT, lw=1.1, label="ground truth (100%)")
            a.set_title(ttl); a.set_ylim(0, 175); a.set_xticks(XT); a.set_xlim(0.5, 16.5); a.set_xlabel(XLAB)
            a.legend(loc="lower left", ncol=nc, columnspacing=0.8, handlelength=1.8); plabel(a, f"({'abc'[k]})")
        ax[0].set_ylabel("Predicted ÷ true displacement (%)")
        f.subplots_adjust(left=0.07, right=0.99, top=0.9, bottom=0.27, wspace=0.08)
        save(f, "2_look")


def fig_simple_identity(V):
    c, p = MAIN; Ip, Iz = id_v3(V, c, p, "p"), id_v3(V, c, p, REF); tt = np.arange(1, p // 2 + 1); S = V["S"]
    with plt.rc_context(PAPER):
        f = plt.figure(figsize=(7.0, 3.1))
        gs = f.add_gridspec(1, 5, width_ratios=[1.3, 0.2, 1, 1, 1], wspace=0.1, left=0.07, right=0.99, top=0.74, bottom=0.21)
        a = f.add_subplot(gs[0]); hA = []
        for k, col, name in (("shape", S1, "shape"), ("color", S3, "colour"), ("combo", S2, "shape + colour")):
            hA += a.plot(tt, 100 * Ip[k], color=col, lw=1.6, label=f"p: {name}")
        hA += a.plot(tt, 100 * Iz["combo"], color=GREY, ls=GT, lw=1.1, label=f"{REF_LABEL}: shape + colour")
        hA.append(a.axhline(100 / 56, color=GREY, ls=":", lw=0.8, label="chance (1/56)"))
        a.legend(handles=hA, loc="lower left", handlelength=1.8)
        a.set_ylim(0, 102); a.set_xticks(XT); a.set_xlim(0.5, 16.5); a.set_xlabel(XLAB); a.set_ylabel("Correct identity (% of clips)")
        a.set_title("RollOut v3, no occlusion"); plabel(a, "(a)", y=-0.25)
        Z = {"last": np.load(V11 / "readings.npz", allow_pickle=True), "mid": np.load(V11_MID / "readings.npz", allow_pickle=True)}
        t6 = np.arange(1, 7); axs = [f.add_subplot(gs[j + 2]) for j in range(3)]; hB = []
        for j, (m, a) in enumerate(zip(("static", "flat", "ramp"), axs)):
            for tm, ks, col, name in (("last", [0], INK, "never occluded"), ("mid", [1, 2, 3, 4], S1, "occluded mid-context"),
                                       ("last", [1, 2, 3, 4], S2, "occluded at context end")):
                z = Z[tm]; k = z["sym_k"].astype(int)
                mo = np.array(["static" if c_.startswith("static") else ("flat" if "flat" in c_ else "ramp") for c_ in z["condition"]])
                combo = np.array([S["shapes"].index(s_) * 8 + S["colors"].index(c_) if s_ in S["shapes"] else -1
                                  for s_, c_ in zip(z["shape_pre"], z["color_pre"])])
                g = z["obj"] & (mo == m) & np.isin(k, ks)
                ok = z["p_prob"][g, -8:].argmax(-1) == combo[g, None]; say = z["p"][g, -8:, 2] > 0
                h_ = a.plot(t6, [100 * ok[:, t][say[:, t]].mean() for t in t6], color=col, lw=1.6, label=f"p: {name}")
                if ks == [0]:
                    okh = z["h_prob"][g, -8:].argmax(-1) == combo[g, None]; sh = z["h"][g, -8:, 2] > 0
                    h2 = a.plot(t6, [100 * okh[:, t][sh[:, t]].mean() for t in t6], color=GREY, ls=GT, lw=1.1,
                                label="target encoder h (never occluded)")
                if j == 0:
                    hB += h_ + (h2 if ks == [0] else [])
            hc = a.axhline(100 / 56, color=GREY, ls=":", lw=0.8, label="chance (1 of 56)")
            a.set_ylim(0, 102); a.set_xticks(t6); a.set_xlim(0.7, 6.3); a.set_xlabel(XLAB); a.set_title(f"IntPhysGen v11, {m}")
            plabel(a, f"({'bcd'[j]})", y=-0.25)
            if j == 0:
                a.set_ylabel("Correct identity (% of 'present' readouts)")
            else:
                a.set_yticklabels([])
        hB.append(hc)
        pa, pb, pd = a.get_position(), axs[0].get_position(), axs[2].get_position()
        pa = f.axes[0].get_position()
        f.legend(handles=hB, loc="lower center", ncol=3, bbox_to_anchor=((pb.x0 + pd.x1) / 2, 0.84), columnspacing=1.0)
        save(f, "3_identity")


def v3_top(V, c, p, rep):
    """v3 57-way argmax (clip, 튜블릿). 정답 = 문맥 물체 조합 (물체가 내내 보여 늘 56 조합 중 하나)."""
    return V["z"][f"C{c}_P{p}_{rep}_prob"][V["POS"]].argmax(-1)


def traj_ci(V, x):
    """clip 값 (n, tp) → (궤적 평균의 평균, 하한, 상한) — 법칙 안에서 궤적 재표집 (NB 회)."""
    tr = traj_mean(V, x.astype(float)); law = tfirst(V, V["law"]); groups = [np.where(law == g)[0] for g in np.unique(law)]
    bs = np.array([np.nanmean(tr[np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups])], 0) for _ in range(NB)])
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    return np.nanmean(tr, 0), lo, hi


def ci_panel(a, tt, m, lo, hi, col, label):
    a.fill_between(tt, 100 * lo, 100 * hi, color=col, alpha=0.2, lw=0)
    a.plot(tt, 100 * m, color=col, lw=1.6, marker="o", ms=2.6, label=label)


def fig_acc57_v3(V):
    """5_acc57_v3 (2026-09-28 사용자 지시): v3, x = 예측 튜블릿. 분모 = 가능 clip 전부, 띠 = 궤적 bootstrap 95 % (112 궤적).
    (a) 57-class 정확도 = 57-way argmax 가 문맥 물체 조합 ('없음' 이면 틀림) — 주 창 C16/P32, p 와 실제 미래 프레임 (REF)
    (b) P32 에서 문맥 길이 C = 4 · 8 · 16 · 32 의 p
    (c) '없음' 을 고른 비율 — 주 창, p 와 실제 미래 프레임"""
    none = V["S"]["none_index"]; c0, p0 = MAIN; tt = np.arange(1, p0 // 2 + 1)
    tp_, tr_ = v3_top(V, c0, p0, "p"), v3_top(V, c0, p0, REF); cmb = V["combo"][:, None]
    A = traj_ci(V, tp_ == cmb); Ar = traj_ci(V, tr_ == cmb)
    N = traj_ci(V, tp_ == none); Nr = traj_ci(V, tr_ == none)
    byC = {c: traj_mean(V, (v3_top(V, c, 32, "p") == cmb).astype(float)) for c in CS}
    with plt.rc_context(PAPER):
        f, ax = plt.subplots(1, 3, figsize=(7.0, 2.55))
        a = ax[0]
        ci_panel(a, tt, *A, S2, "predictor p")
        a.plot(tt, 100 * Ar[0], color=GREY, ls=GT, lw=1.2, label=REF_LABEL)
        a.axhline(100 / 57, color=GREY, ls=":", lw=0.8, label="chance (1/57)")
        a.set_title("57-class accuracy"); a.set_ylabel("% of clips"); a.legend(loc="lower left")
        a = ax[1]
        for c in CS:
            a.plot(tt, 100 * np.nanmean(byC[c], 0), color=CCOL[c], lw=1.4, marker="o", ms=2.0, label=f"context {c} frames")
        a.axhline(100 / 57, color=GREY, ls=":", lw=0.8)
        a.set_title("57-class accuracy, context length varied"); a.legend(loc="lower left", title="predictor p", title_fontsize=6.8)
        a = ax[2]
        ci_panel(a, tt, *N, S2, "predictor p")
        a.plot(tt, 100 * Nr[0], color=GREY, ls=GT, lw=1.2, label=REF_LABEL)
        a.set_title("'None' chosen"); a.legend(loc="upper left")
        for k, a in enumerate(ax):
            a.set_ylim(0, 100); a.set_xticks(XT); a.set_xlim(0.5, 16.5); a.set_xlabel("Prediction tubelet")
            plabel(a, f"({'abc'[k]})", y=-0.26)
            if k:
                a.set_yticklabels([])
        f.subplots_adjust(left=0.07, right=0.99, top=0.9, bottom=0.25, wspace=0.08)
        save(f, "5_acc57_v3")
    r = lambda v: [round(float(x), 4) for x in v]
    return dict(acc57_p=r(A[0]), acc57_p_ci=[r(A[1]), r(A[2])], acc57_ref=r(Ar[0]), none_p=r(N[0]), none_p_ci=[r(N[1]), r(N[2])],
                none_ref=r(Nr[0]), acc57_by_C={c: r(np.nanmean(byC[c], 0)) for c in CS})


def fig_shape_colour_v3(V):
    """6_shape_colour_v3 (2026-09-28 사용자 지시): 57-way argmax 로 고른 조합의 (a) 모양 · (b) 색이 문맥 물체와 같은 비율.
    '없음' 을 고르면 틀림 (5_acc57_v3 와 같은 규칙). 분모 = 가능 clip 전부, 띠 = 궤적 bootstrap 95 %.
    ⚠️ 3_identity (a) 는 '있음' 조건부 주변 확률 argmax 라 정의가 다르다."""
    none = V["S"]["none_index"]; c0, p0 = MAIN; tt = np.arange(1, p0 // 2 + 1)
    tp_, tr_ = v3_top(V, c0, p0, "p"), v3_top(V, c0, p0, REF)
    s_t, c_t = (V["combo"] // 8)[:, None], (V["combo"] % 8)[:, None]
    out = {}
    with plt.rc_context(PAPER):
        f, ax = plt.subplots(1, 2, figsize=(7.0, 2.55), sharey=True)
        for k, (a, name, truth, div, nk) in enumerate(((ax[0], "Shape accuracy (7 shapes)", s_t, lambda t: t // 8, 7),
                                                       (ax[1], "Colour accuracy (8 colours)", c_t, lambda t: t % 8, 8))):
            M = traj_ci(V, (tp_ != none) & (div(tp_) == truth)); Mr = traj_ci(V, (tr_ != none) & (div(tr_) == truth))
            ci_panel(a, tt, *M, S2, "predictor p")
            a.plot(tt, 100 * Mr[0], color=GREY, ls=GT, lw=1.2, label=REF_LABEL)
            a.axhline(100 / nk, color=GREY, ls=":", lw=0.8, label=f"chance (1/{nk})")
            a.set_title(name); a.set_ylim(0, 100); a.set_xticks(XT); a.set_xlim(0.5, 16.5); a.set_xlabel("Prediction tubelet")
            a.legend(loc="center left"); plabel(a, f"({'ab'[k]})", y=-0.26)
            out["shape" if k == 0 else "colour"] = dict(p=[round(float(x), 4) for x in M[0]], ref=[round(float(x), 4) for x in Mr[0]])
        ax[0].set_ylabel("% of clips")
        f.subplots_adjust(left=0.07, right=0.99, top=0.9, bottom=0.25, wspace=0.06)
        save(f, "6_shape_colour_v3")
    return out


def fig_simple_windows(V):
    SG = speed_groups(V)
    with plt.rc_context(PAPER):
        f, ax = plt.subplots(len(CS), len(PS), figsize=(7.0, 6.4), sharex="col")
        for i, c in enumerate(CS):
            for j, p in enumerate(PS):
                a = ax[i, j]; d = simple_data(V, c, p); tt = d["tt"]
                for lab, m, col in SG:
                    a.plot(tt, np.nanmean(d["D"][m], 0), color=col, ls=GT, lw=0.9)
                    a.plot(tt, np.nanmean(d["Dh"][m], 0), color=col, lw=1.3)
                a.set_xlim(0.5, tt[-1] + 0.5); a.set_xticks([t for t in XT if t <= tt[-1]] if tt[-1] > 2 else [1, 2])
                a.set_ylim(bottom=0)
                if i == 0:
                    a.set_title(f"predict P = {p} frames")
                if j == 0:
                    a.set_ylabel(f"context C = {c} frames\nDisplacement (px)")
                if i == len(CS) - 1:
                    a.set_xlabel(XLAB)
        hs = [plt.Line2D([], [], color=col, lw=1.6, label=lab) for lab, _, col in SG] + style_handles()
        f.legend(handles=hs, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0), title="context speed (fixed per trajectory from C = 16)", title_fontsize=7)
        f.subplots_adjust(left=0.1, right=0.99, top=0.88, bottom=0.07, hspace=0.3, wspace=0.26)
        save(f, "4_motion_windows")


# ------------------------------------------------------------------ main

def main():
    if not is_identity(PRES):
        raise SystemExit("정체 자 (identity) 가 아니다 — R3_DECODER=identity_r8 R3_OUT=identity_r8")
    OUT.mkdir(parents=True, exist_ok=True)
    V = load_v3()
    print(f"v3: 가능 clip {V['POS'].sum()} · 궤적 {V['nt']}", flush=True)
    W = {}
    for c in CS:
        for p in PS:
            W[(c, p)] = motion_coefs(V, c, p)
    print("운동 계수 끝", flush=True)
    fig_motion_main(V, *W[MAIN]); fig_motion_windows(W)
    A = {}
    for c in CS:
        for p in PS:
            for rep in ("p", REF):
                A[(c, p, rep)] = appearance(V, c, p, rep, nperm=200 if (c, p) == MAIN else 50)
    print("외형 끝", flush=True)
    fig_appearance_main(A); fig_appearance_windows(A)
    I = fig_identity_v3(V); v11 = {} if NO_V11 else fig_identity_v11()
    fig_simple_motion(V); fig_simple_look(V); fig_simple_windows(V)
    acc57 = fig_acc57_v3(V)
    shcol = fig_shape_colour_v3(V)
    if not NO_V11:
        fig_simple_identity(V)

    rnd = lambda x: None if x is None or not np.isfinite(x) else round(float(x), 3)
    vals = dict(decoder=PRES.name, n_clip=int(V["POS"].sum()), n_traj=int(V["nt"]),
                motion={f"C{c}_P{p}": {name: [{k: [rnd(v) for v in r[k]] for k in r} for r in rows] for name, rows in W[(c, p)][0].items()}
                        for c in CS for p in PS},
                appearance={f"C{c}_P{p}_{rep}": dict(sd=[rnd(v) for v in A[(c, p, rep)]["sd"]],
                                                     eta={k: [rnd(v) for v in A[(c, p, rep)]["eta"][k]] for k in A[(c, p, rep)]["eta"]},
                                                     null95={k: [rnd(v) for v in A[(c, p, rep)]["null"][k]] for k in A[(c, p, rep)]["null"]})
                            for c in CS for p in PS for rep in ("p", REF)},
                identity_v3={rep: {k: [rnd(v) for v in I[rep][k]] for k in ("combo", "shape", "color")} for rep in I},
                identity_v11=v11, acc57_v3=acc57, shape_colour_v3=shcol)
    (OUT / "values.json").write_text(json.dumps(vals, indent=1))
    stamp(OUT, PRES, [WIN / "readings.npz"] + ([] if NO_V11 else [V11 / "readings.npz", V11_MID / "readings.npz"]))
    # 한 줄 요약 (주 창)
    co = W[MAIN][0]
    for name in ("truth", REF, "p"):
        s = " · ".join(f"t{t} v{co[name][t]['bv'][0]:.2f} a{co[name][t]['ba'][0]:.2f} x{co[name][t]['bx'][0]:+.1f}" for t in (1, 4, 8, 15))
        print(f"C16/P32 {name:5s} {s}")
    P_, Z_ = A[(16, 32, "p")], A[(16, 32, REF)]
    print("외형 SD p", np.round(P_["sd"][[1, 4, 8, 15]], 1), " z", np.round(Z_["sd"][[1, 4, 8, 15]], 1))
    print("η² p shape/color/env t8", [round(100 * P_["eta"][k][8], 1) for k in ("shape", "color", "env")], " null", [round(100 * P_["null"][k][8], 1) for k in ("shape", "color", "env")])
    print(OUT)


if __name__ == "__main__":
    main()
