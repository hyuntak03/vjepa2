#!/usr/bin/env python3
"""IntPhysGen v11 (visible · 늦은 가림) — p 의 미래에 **물체가 있나 · 어느 방향으로 가나 · 가속하나**.

입력: `RollOutV3/exp_results/v11/readings.npz` (`v11_readout_online.py`). 미래 튜블릿 t = 0..7 (샘플 16+2t, 17+2t).
기준점 = **마지막으로 본 자리** x_seen: 문맥 샘플 중 가려지지 않은 마지막 것 (visible 15, late k 는 15−k).

  ① 있음      p 가 '있다' 고 한 clip 비율 (물체 있는 clip 전부). 빈 장면 clip 의 '있다' = 자의 오탐 바닥
  정의는 v3 의 `plot_direction_r2.py` 와 같다 — **방향만** (단위벡터), 2 축, p 가 '있다' 고 한 clip 만.
  ② 속도 방향  flat · ramp 각각 (v11 은 한 조건 안에 좌→우 · 우→좌가 반반이라 조건별로 정의된다):
                y = unit(x̂(t) − x_seen)     u = unit(V_seen)      V_seen = 마지막으로 본 두 샘플 사이 속도
  ③ 가속 방향  ramp 만 (flat 은 가속이 0 이라 방향이 없다). ramp 는 전부 내리막이라 **진행 방향으로 빨라진다** (문맥 끝 → 미래 1.5 배):
                y = unit(x̂(t) − (x_seen + V_seen·경과))   u = unit(A_seen)
                ⚠️ ramp 의 A 는 진행 방향과 같아서, p 가 **다 같이 덜 가면** (지연) R² 가 높고 β < 0 (⁻) 으로 나온다
  y ≈ β·u + b (β 하나, 절편 축마다),  R² = 1 − Σ‖y − ŷ‖² / Σ‖y − ȳ‖².  ⁻ = β < 0 (반대 방향)

  자 검사: 같은 계산을 `h` (target encoder, 창 전체를 본다) 에 한다 — `h` 의 R² 가 낮은 칸은 자가 그 차이를 못 가른다.
  `h` 는 상한이 아니라 **자 검사** 다.

⚠️ 가림 편향: 이 자는 가림 음성을 배운 적이 없다. 가려진 샘플에서 `h` 가 '있다' 고 하는 비율을 먼저 본다 (V11_READOUT.md §0).
⚠️ 신뢰구간은 block 단위 재표집 (같은 block 의 clip 은 궤적이 같다).

  $P z_research/scripts/figures/plot_v11_readout.py   → figures/v11/fig_v11_readout.png · exp_results/v11/V11_READOUT.md
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
sys.path.insert(0, str(ROOT / "z_research/scripts/analysis"))
from rollout3_paths import PRES, EXP, fig, check       # noqa: E402
from plot_direction_r2 import fit, unit                # noqa: E402  (정의를 v3 와 공유)

RES, CS, TP, NB, TC, EDGE, W = 144.0, 16, 8, 200, 8, 36.0, 288.0
# EDGE = 2 칸. v11 visible 에서 h 의 '있다' 가 가장자리까지 x 거리 0~18 px 0 % · 18~36 px 34 % · 36~54 px 95 % · 54 px~ 98~100 % (2026-09-24 실측)
BLUE, ORANGE, GREEN, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#888888"
MOT = {"static_visible": "static", "static_occlusion": "static", "moving_visible_flat": "flat",
       "moving_occlusion_flat": "flat", "moving_visible": "ramp", "moving_occlusion": "ramp"}
rng = np.random.default_rng(0)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5})
FX = 6 * (np.arange(TP) + 1)                          # 문맥 끝 (raw 45) 뒤 raw 프레임 — 튜블릿 t 의 끝


def load():
    z = np.load(EXP / "v11/readings.npz", allow_pickle=True); check(z, PRES)
    S = json.loads((PRES / "summary.json").read_text())
    thr = {r: S["reps"][r]["attn"]["thr_val_fpr5"] for r in S["reps"]}
    bias = json.loads((PRES / "attn_bias_px.json").read_text())
    n = len(z["video_id"]); k = z["sym_k"].astype(int)
    mot = np.array([MOT[c] for c in z["condition"]])
    L = z["truth"] * RES                                                  # (n, 32, 2) px − 144
    s_seen = np.where(k > 0, CS - 1 - k, CS - 1)                          # 마지막으로 본 샘플
    ar = np.arange(n)
    x_seen = L[ar, s_seen]                                                # (n, 2)
    v_seen = (L[ar, s_seen] - L[ar, s_seen - 2]) / 2                      # px / 샘플
    a_seen = L[ar, s_seen] - 2 * L[ar, s_seen - 2] + L[ar, s_seen - 4]    # 방향만 쓴다
    # 튜블릿마다 진짜 보이고 **화면 가장자리에서 EDGE 이상 안쪽**인가. v11 물체는 화면을 끝까지 가로질러 t5~t7 에 가장자리에 붙고,
    # 거기서는 보이는데도 h 가 '있다' 를 못 한다 — 자의 가장자리 효과라 뺀다
    Tabs = L.reshape(n, 16, 2, 2).mean(2) + RES
    inside = ((Tabs >= EDGE) & (Tabs <= W - EDGE)).all(-1)
    tvis = (z["in_frame"] & ~z["hidden"]).reshape(n, 16, 2).all(2) & inside & z["obj"][:, None]
    R = {}
    for r in ("p", "h"):
        v = z[r][:, -TP:]                                                 # h 는 미래 8 튜블릿만 잘라 p 와 맞춘다
        R[r] = dict(xy=v[..., 0:2] * RES - np.array(bias[r], np.float32), say=v[..., 2] > thr[r])
    hall = dict(say=z["h"][..., 2] > thr["h"], xy=z["h"][..., 0:2] * RES - np.array(bias["h"]))
    return z, dict(n=n, k=k, mot=mot, obj=z["obj"], dir=z["travel_dir"].astype(float), blk=z["block_id"],
                   s_seen=s_seen, x_seen=x_seen, v_seen=v_seen, a_seen=a_seen, tvis=tvis, L=L, R=R, hall=hall)


def r2(y, x):
    X = np.stack([x, np.ones_like(x)], 1)
    b, *_ = np.linalg.lstsq(X, y, rcond=None); res = y - X @ b; tot = y - y.mean()
    return 1 - (res @ res) / max(tot @ tot, 1e-9), b[0]


def boot(idx, blk, f):
    """block 단위 재표집 95 % 구간. f(idx) → 값 (nan 가능)."""
    ub = np.unique(blk[idx]); by = {b: idx[blk[idx] == b] for b in ub}; out = []
    for _ in range(NB):
        pick = rng.choice(ub, len(ub)); out.append(f(np.concatenate([by[b] for b in pick])))
    out = np.array(out, float); out = out[np.isfinite(out)]
    return (np.percentile(out, 2.5), np.percentile(out, 97.5)) if len(out) > NB // 2 else (np.nan, np.nan)


def presence(D, rep, sel, t, need_vis=True):
    if need_vis: sel = sel & D["tvis"][:, TC + t]           # 물체가 진짜 보이는 칸만 (빈 장면은 전부)
    return D["R"][rep]["say"][sel, t].mean() * 100 if sel.sum() else np.nan


def sgn(v):
    """(R², β) → 부호 붙은 R². 음수 = 반대 방향."""
    return v[0] * np.sign(v[1]) if np.isfinite(v[0]) else np.nan


def dir_r2(D, rep, idx, t, min_n=30):
    m = idx[D["R"][rep]["say"][idx, t] & D["tvis"][idx, TC + t]]
    if len(m) < min_n: return np.nan, np.nan
    return fit(unit(D["R"][rep]["xy"][m, t] - D["x_seen"][m]), unit(D["v_seen"][m]))


def acc_r2(D, rep, idx, t, min_n=30):
    m = idx[D["R"][rep]["say"][idx, t] & D["tvis"][idx, TC + t]]
    if len(m) < min_n: return np.nan, np.nan
    el = (CS + 2 * t + 0.5) - D["s_seen"][m]
    c = D["x_seen"][m] + D["v_seen"][m] * el[:, None]
    return fit(unit(D["R"][rep]["xy"][m, t] - c), unit(D["a_seen"][m]))


def curves(D):
    k, mot, obj = D["k"], D["mot"], D["obj"]
    G = {"visible": k == 0, "late": k > 0}
    C = {}
    for g, gm in G.items():
        for mo in ("static", "flat", "ramp"):
            sel = obj & gm & (mot == mo)
            for rep in ("p", "h"):
                C[("pres", g, mo, rep)] = np.array([presence(D, rep, sel, t) for t in range(TP)])
            C[("n", g, mo)] = D["tvis"][sel][:, TC:].sum(0)
        C[("pres", g, "empty", "p")] = np.array([presence(D, "p", ~obj & gm, t, need_vis=False) for t in range(TP)])
        for mo in ("flat", "ramp"):
            idx = np.where(obj & gm & (mot == mo))[0]
            for rep in ("p", "h"):
                v = [dir_r2(D, rep, idx, t) for t in range(TP)]
                C[("dir", g, mo, rep)] = np.array([a for a, _ in v]); C[("dirb", g, mo, rep)] = np.array([b for _, b in v])
                C[("dirs", g, mo, rep)] = np.array([sgn(x) for x in v])
                C[("dirci", g, mo, rep)] = np.array([boot(idx, D["blk"], lambda ii, t=t: sgn(dir_r2(D, rep, ii, t))) for t in range(TP)])
        idx = np.where(obj & gm & (mot == "ramp"))[0]
        for rep in ("p", "h"):
            v = [acc_r2(D, rep, idx, t) for t in range(TP)]
            C[("acc", g, rep)] = np.array([a for a, _ in v]); C[("accb", g, rep)] = np.array([b for _, b in v])
            C[("accs", g, rep)] = np.array([sgn(x) for x in v])
            C[("accci", g, rep)] = np.array([boot(idx, D["blk"], lambda ii, t=t: sgn(acc_r2(D, rep, ii, t))) for t in range(TP)])
    return C


def figure(C):
    f_, ax = plt.subplots(2, 3, figsize=(12.6, 7.4))
    for j, mo in enumerate(("static", "flat", "ramp")):
        a = ax[0, j]
        a.plot(FX, C[("pres", "visible", mo, "p")], color=BLUE, lw=2, marker="o", ms=3, label="p, visible (k=0)")
        a.plot(FX, C[("pres", "late", mo, "p")], color=ORANGE, lw=2, marker="o", ms=3, label="p, occluded at context end (k=1-4)")
        a.plot(FX, C[("pres", "late", mo, "h")], color=GREEN, lw=1.2, ls="--", label="h, occluded (readout check)")
        a.plot(FX, C[("pres", "late", "empty", "p")], color=GRAY, lw=1, ls="-.", label="p, empty scene with occluder (false alarm)")
        a.set_ylim(-3, 103); a.set_title(f"object present? — {mo}", fontsize=9.5, fontweight="bold")
        if j == 0: a.set_ylabel("% read as present\n(tubelets where the object is visible)")
    for j, mo in enumerate(("flat", "ramp")):
        a = ax[1, j]
        for g, col in (("visible", BLUE), ("late", ORANGE)):
            lo, hi = C[("dirci", g, mo, "p")].T
            a.fill_between(FX, lo, hi, color=col, alpha=0.15, lw=0)
            a.plot(FX, C[("dirs", g, mo, "p")], color=col, lw=2, marker="o", ms=3)
            a.plot(FX, C[("dirs", g, mo, "h")], color=col, lw=1, ls="--")
        a.set_ylim(-1.05, 1.05); a.axhline(0, color="#aaaaaa", lw=0.6); a.set_title(f"velocity direction R² — {mo}", fontsize=9.5, fontweight="bold")
        if j == 0: a.set_ylabel("signed R²  (+ = seen direction,\n− = opposite)")
    a = ax[1, 2]
    for g, col in (("visible", BLUE), ("late", ORANGE)):
        lo, hi = C[("accci", g, "p")].T
        a.fill_between(FX, lo, hi, color=col, alpha=0.15, lw=0)
        a.plot(FX, C[("accs", g, "p")], color=col, lw=2, marker="o", ms=3)
        a.plot(FX, C[("accs", g, "h")], color=col, lw=1, ls="--")
    a.set_ylim(-1.05, 1.05); a.axhline(0, color="#aaaaaa", lw=0.6); a.set_title("acceleration direction R² — ramp", fontsize=9.5, fontweight="bold")
    a.plot([], [], color="k", lw=2, label="p (bands: 95% CI, block bootstrap)"); a.plot([], [], color="k", lw=1, ls="--", label="h (readout check)")
    a.legend(frameon=False, fontsize=7, loc="lower left")
    for a in ax.flat:
        a.set_xticks(FX); a.set_xlim(3, 51); a.spines[["top", "right"]].set_visible(False); a.grid(axis="y", lw=0.3, alpha=0.4)
    for a in ax[1]: a.set_xlabel("raw frames after context end")
    ax[0, 0].legend(frameon=False, fontsize=6.8, loc="lower left")
    f_.tight_layout()
    d = fig("v11"); d.mkdir(parents=True, exist_ok=True); p = d / "fig_v11_readout.png"
    f_.savefig(p, dpi=170); plt.close(f_); return p


def checks(D, z):
    """자 검사 — 가려진 샘플·빈 장면·보이는 물체에서 h·z 가 뭐라고 하나 (16 튜블릿 전부)."""
    out = []
    n = D["n"]; hid_t = z["hidden"].reshape(n, 16, 2).all(2) & z["obj"][:, None]
    vis_t = D["tvis"]; emp = ~z["obj"]
    T = D["L"].reshape(n, 16, 2, 2).mean(2)
    for r in ("z", "h"):
        v = z[r]; S = json.loads((PRES / "summary.json").read_text())
        say = v[..., 2] > S["reps"][r]["attn"]["thr_val_fpr5"]
        bias = json.loads((PRES / "attn_bias_px.json").read_text())[r]
        err = np.linalg.norm(v[..., 0:2] * RES - np.array(bias) - T, axis=-1)
        out.append((r, say[vis_t].mean() * 100, np.median(err[vis_t]), say[hid_t].mean() * 100, say[emp].mean() * 100))
    return out


def report(D, C, z, figp):
    L = ["# IntPhysGen v11 — visible · 늦은 가림에서 p 의 미래 (RollOutV3 자, 2026-09-24)", "",
         f"정의: `z_research/scripts/figures/plot_v11_readout.py` 머리말. 그림 `{figp.relative_to(ROOT)}`.",
         f"clip: 물체 {int(z['obj'].sum()):,} · 빈 장면 {int((~z['obj']).sum()):,} (가능 변이만). 자 지문 `{z['decoder_fp']}`.", "",
         "## 0. 자 검사 (16 튜블릿 전부, 창 전체를 보는 표현)", "",
         "| 표현 | 보이는 물체 '있다' % | 위치 L2 중앙 px | **가려진 물체 '있다' %** | 빈 장면 '있다' % |", "|---|---|---|---|---|"]
    for r, a, e, h, em in checks(D, z):
        L.append(f"| `{r}` | {a:.1f} | {e:.1f} | **{h:.1f}** | {em:.1f} |")
    L += ["", "가려진 물체 '있다' % 가 높을수록 이 자는 '가려짐' 을 '있음' 으로 읽는다 (가림 음성을 배운 적이 없다).", ""]

    L += ["## 1. 있음 — p 가 '있다' 고 한 % (물체가 **진짜 보이고 화면 가장자리에서 2 칸 (36 px) 이상 안쪽**인 칸만)", "",
          "late 의 t0 (k≥2 면 전부 가려짐) 처럼 보이는 clip 이 없는 칸은 nan. 빈 장면은 전부.", "",
          "| 운동 | 무리 | " + " | ".join(f"t{t}" for t in range(TP)) + " |", "|---|---|" + "---|" * TP]
    for mo in ("static", "flat", "ramp"):
        for g in ("visible", "late"):
            L.append(f"| {mo} | p · {g} | " + " | ".join(f"{v:.0f}" for v in C[("pres", g, mo, "p")]) + " |")
        L.append(f"| {mo} | h · late (자 검사) | " + " | ".join(f"{v:.0f}" for v in C[("pres", "late", mo, "h")]) + " |")
        L.append(f"| {mo} | clip 수 (보이고 가장자리 밖) · visible / late | " + " | ".join(f"{a}/{b}" for a, b in zip(C[("n", "visible", mo)], C[("n", "late", mo)])) + " |")
    for g in ("visible", "late"):
        L.append(f"| 빈 장면 | p · {g} (오탐) | " + " | ".join(f"{v:.0f}" for v in C[("pres", g, "empty", "p")]) + " |")

    def fmt(v, b):
        return "·" if not np.isfinite(v) else (f"{v:.2f}" + ("⁻" if b < 0 else ""))
    L += ["", "## 2. 속도 방향 R² (p 가 '있다' 고 한 clip 만, ⁻ = β < 0 반대 방향)", "",
          "| 운동 | 무리 | " + " | ".join(f"t{t}" for t in range(TP)) + " |", "|---|---|" + "---|" * TP]
    for mo in ("flat", "ramp"):
        for g in ("visible", "late"):
            for rep in ("p", "h"):
                L.append(f"| {mo} | {rep} · {g} | " + " | ".join(fmt(v, b) for v, b in zip(C[("dir", g, mo, rep)], C[("dirb", g, mo, rep)])) + " |")
    L += ["", "## 3. 가속 방향 R² (ramp 만, ⁻ = β < 0 — 가속 반대쪽 = 본 속도보다 덜 감)", "",
          "| 무리 | " + " | ".join(f"t{t}" for t in range(TP)) + " |", "|---|" + "---|" * TP]
    for g in ("visible", "late"):
        for rep in ("p", "h"):
            L.append(f"| {rep} · {g} | " + " | ".join(fmt(v, b) for v, b in zip(C[("acc", g, rep)], C[("accb", g, rep)])) + " |")
    L += ["", "## 재현", "", "```bash", "cd /data/hyuntak/project/2026/2027_cvpr/vjepa2", "P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python",
          "$P z_research/scripts/analysis/v11_readout_online.py --gpus 8 --token-budget 196608 --max-bs 48   # 읽은 값",
          "$P z_research/scripts/figures/plot_v11_readout.py                                               # 이 문서 + 그림", "```", ""]
    p = EXP / "v11/V11_READOUT.md"; p.write_text("\n".join(L)); return p


def main():
    z, D = load()
    C = curves(D)
    figp = figure(C)
    print(report(D, C, z, figp)); print(figp)


if __name__ == "__main__":
    main()
