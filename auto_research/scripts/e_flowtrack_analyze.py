#!/usr/bin/env python3
"""SC2v2 분석 — flow 유사 정답 지평 곡선. python e_flowtrack_analyze.py release=<dir> ariel=<dir> ...  (첫 태그가 짝 비교 기준)

칸 = 256 격자의 16 px.  s0 = 출발 칸 (질의 토큰 q),  f_i = flow 추적 칸 (슬롯 i),  v3 모드면 g_i = GT 칸.
움직임 선별 (인과): 문맥 끝 속도 |v| ≥ VMIN px/프레임 (기본 2 → 등속이면 미래 16 장에 2 칸).
판독 (argmax, 256 토큰): enc = h_{8+i}·Tc (encoder 천장, 미래를 봄) · p = p_i·Tc (predictor) ; 표준 템플릿 T 판도 같이.
  hit  = 1[‖X_i − f_i‖∞ ≤ 1]      copy = 1[‖s0 − f_i‖∞ ≤ 1] (복사 기준선)      cv = 1[‖s0 + v·Δ − f_i‖∞ ≤ 1] (문맥 끝 속도 등속 오라클)
  null = 1[‖X_i(a) − f_i(b)‖∞ ≤ 1] (clip 순열 20 회)
  prog = (X_i − s0)·(f_i − s0)/|f_i − s0|²  (|f_i − s0| ≥ 1 칸인 clip; 평균은 [−1, 2] 로 자름, 중앙값 같이)
  soft = cos_X[f_i] − cos_X[s0]  (‖f_i − s0‖∞ ≥ 2 인 clip) — argmax 없이 "새 자리를 옛 자리보다 더 닮게 두나"
CI = clip bootstrap (v3 는 궤적 cluster) B = 1000.  짝 차이 = 같은 clip 의 (태그 − 첫 태그).
도달 거리 (by_disp): (clip, 슬롯) 쌍을 유사 정답 변위 구간으로 묶어 이른/늦은 슬롯 비교 + 회귀 (hit_p − copy) ~ 1 + slot + cell (SC1 의 거리 법칙을 실영상에서).
"""
import json, os, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "auto_research/exp_results/ssv2"; OUT.mkdir(parents=True, exist_ok=True)
VMIN = float(os.environ.get("VMIN", 2.0)); PX, G = 16.0, 16
rng = np.random.default_rng(0)


def cell(xy):
    return np.clip(np.floor(xy / PX), 0, G - 1)


def boot(v, cl, B=1000):
    ok = np.isfinite(v); v, cl = v[ok], cl[ok]
    if len(v) == 0: return [np.nan] * 3
    uc = np.unique(cl); ix = {c: np.where(cl == c)[0] for c in uc}
    bs = [np.mean(v[np.concatenate([ix[c] for c in rng.choice(uc, len(uc))])]) for _ in range(B)]
    return [round(float(v.mean()), 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]


def load(d):
    d = Path(d); meta = json.load(open(d / "meta.json"))
    r = dict(meta=meta, cos=np.load(d / "cos.npy").astype(np.float32), flow=np.load(d / "flow.npy"), q=np.load(d / "q.npy"))
    if meta["mode"] == "v3": r["gt"] = np.load(d / "gt.npy")
    return r


def per_clip(r, tgt):
    """clip × 슬롯 지표 dict (판독 넷)."""
    q = r["q"]; n = len(q); cos = r["cos"]
    s0 = np.stack([q[:, 1], q[:, 0]], -1)                                        # (n, 2) x, y 칸
    am = cos.argmax(-1); A = np.stack([am % G, am // G], -1).astype(float)      # (n, 4, 8, 2)
    T = cell(tgt)                                                               # (n, 8, 2)
    v = np.stack([q[:, 6], q[:, 7]], -1); st = np.stack([q[:, 4], q[:, 5]], -1)
    dt = np.array([1.5 + 2 * i for i in range(8)])                               # 프레임 15 → 슬롯 i (16+2i, 17+2i) 평균 시각
    CV = cell(st[:, None] + v[:, None] * dt[None, :, None])
    dv = T - s0[:, None]; dd = (dv ** 2).sum(-1); far = np.abs(dv).max(-1)
    out = dict(copy=(np.abs(s0[:, None] - T).max(-1) <= 1).astype(float), cv=(np.abs(CV - T).max(-1) <= 1).astype(float), disp=np.sqrt(dd), cheb=far)
    for k, nm in enumerate(("enc", "p", "encT", "pT")):
        X = A[:, k]
        out[f"hit_{nm}"] = (np.abs(X - T).max(-1) <= 1).astype(float)
        out[f"null_{nm}"] = np.mean([(np.abs(X - T[rng.permutation(n)]).max(-1) <= 1) for _ in range(20)], 0)
        pr = np.where(dd >= 1, ((X - s0[:, None]) * dv).sum(-1) / np.maximum(dd, 1e-9), np.nan)
        out[f"prog_{nm}"] = pr
        ti = (T[..., 1] * G + T[..., 0]).astype(int); si = (s0[:, 1] * G + s0[:, 0]).astype(int)
        c = cos[:, k]; ct = np.take_along_axis(c, ti[..., None], -1)[..., 0]; cs = c[np.arange(n)[:, None], np.arange(8)[None], si[:, None]]
        out[f"soft_{nm}"] = np.where(far >= 2, ct - cs, np.nan)
    return out


def summarize(pc, m, cl):
    rows = []
    for i in range(8):
        rw = dict(slot=i, disp=round(float(np.nanmean(pc["disp"][m, i])), 2), copy=boot(pc["copy"][m, i], cl[m]), cv=boot(pc["cv"][m, i], cl[m]))
        for nm in ("enc", "p", "encT", "pT"):
            rw[f"hit_{nm}"] = boot(pc[f"hit_{nm}"][m, i], cl[m]); rw[f"hitnull_{nm}"] = boot((pc[f"hit_{nm}"] - pc[f"null_{nm}"])[m, i], cl[m])
            rw[f"hitcopy_{nm}"] = boot((pc[f"hit_{nm}"] - pc["copy"])[m, i], cl[m])
            pr = pc[f"prog_{nm}"][m, i]; rw[f"prog_{nm}"] = boot(np.clip(pr, -1, 2), cl[m]); rw[f"progmed_{nm}"] = round(float(np.nanmedian(pr)), 3) if np.isfinite(pr).any() else np.nan
            rw[f"soft_{nm}"] = boot(pc[f"soft_{nm}"][m, i], cl[m])
        rows.append(rw)
    return rows


BINS = [0, 1, 2, 3, 4, 6, 9, 99]


def by_disp(pc, m, cl, n_slot=8):
    """도달 거리 (reach) 검정: (clip, 슬롯) 쌍을 유사 정답 변위 ‖f_i − s0‖∞ (칸) 구간으로 묶는다 (슬롯 풀링).
    거리 법칙이면 같은 변위에서 이른 슬롯 (0–3) · 늦은 슬롯 (4–7) 이 같고, 슬롯 법칙이면 같은 슬롯에서 변위가 무관하다.
    회귀 (hit_p − copy) ~ 1 + slot + disp, (clip, 슬롯) 쌍, clip bootstrap."""
    dsp = pc["cheb"]; sl = np.broadcast_to(np.arange(n_slot)[None], dsp.shape)
    out = dict(bins=[])
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        for sname, sm in (("all", sl >= 0), ("early", sl <= 3), ("late", sl >= 4)):
            k = m[:, None] & (dsp >= lo) & (dsp < hi) & sm
            if k.sum() < 10: continue
            ci = np.nonzero(k)[0]
            row = dict(lo=lo, hi=hi, slots=sname, n=int(k.sum()))
            for key in ("copy", "hit_p", "hit_enc", "hit_pT"):
                row[key] = boot(pc[key][k], cl[ci])
            row["hitcopy_p"] = boot((pc["hit_p"] - pc["copy"])[k], cl[ci]); row["prog_p"] = boot(np.clip(pc["prog_p"][k], -1, 2), cl[ci])
            row["prog_enc"] = boot(np.clip(pc["prog_enc"][k], -1, 2), cl[ci]); row["soft_p"] = boot(pc["soft_p"][k], cl[ci])
            out["bins"].append(row)
    k = m[:, None] & np.isfinite(dsp)
    y = (pc["hit_p"] - pc["copy"])[k]; X = np.stack([np.ones(k.sum()), sl[k], dsp[k]], 1); ci = np.nonzero(k)[0]
    uc = np.unique(cl[ci]); ix = {c: np.where(cl[ci] == c)[0] for c in uc}
    bs = []
    for _ in range(300):
        s_ = np.concatenate([ix[c] for c in rng.choice(uc, len(uc))]); bs.append(np.linalg.lstsq(X[s_], y[s_], rcond=None)[0])
    bs = np.array(bs); b0 = np.linalg.lstsq(X, y, rcond=None)[0]
    out["regress_hitcopy_p"] = {nm: [round(float(b0[j]), 4), round(float(np.percentile(bs[:, j], 2.5)), 4), round(float(np.percentile(bs[:, j], 97.5)), 4)] for j, nm in enumerate(("intercept", "per_slot", "per_cell"))}
    return out


def by_disp_valid(pc, m, cl, n_slot=8):
    """유사 정답 두 개가 합의한 (clip, 슬롯) 쌍만: encoder 판독 (h·Tc argmax) 이 flow 칸 1 칸 안 (= 물체가 거기 있다는 독립 두 증거).
    그 위에서 P(hit_p) 를 변위 구간 × 이른/늦은 슬롯으로. 회귀 hit_p ~ 1 + slot + cell (clip bootstrap).
    (슬롯이 늦을수록 encoder 천장 자체가 낮아지는 교란을, 천장이 맞은 쌍만 남겨 제거한다.) valid 는 predictor 와 무관하므로 predictor 사이 짝 비교가 된다."""
    dsp = pc["cheb"]; sl = np.broadcast_to(np.arange(n_slot)[None], dsp.shape); val = m[:, None] & (pc["hit_enc"] > 0.5)
    out = dict(n_valid=int(val.sum()), frac_valid=round(float(val.sum() / max((m[:, None] & np.isfinite(dsp)).sum(), 1)), 3), bins=[])
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        for sname, sm in (("all", sl >= 0), ("early", sl <= 3), ("late", sl >= 4)):
            k = val & (dsp >= lo) & (dsp < hi) & sm
            if k.sum() < 10: continue
            ci = np.nonzero(k)[0]
            out["bins"].append(dict(lo=lo, hi=hi, slots=sname, n=int(k.sum()), hit_p=boot(pc["hit_p"][k], cl[ci]), hit_pT=boot(pc["hit_pT"][k], cl[ci]),
                                    prog_p=boot(np.clip(pc["prog_p"][k], -1, 2), cl[ci]), copy=boot(pc["copy"][k], cl[ci]), cv=boot(pc["cv"][k], cl[ci])))
    k = val & np.isfinite(dsp); y = pc["hit_p"][k]; X = np.stack([np.ones(k.sum()), sl[k], dsp[k]], 1); ci = np.nonzero(k)[0]
    uc = np.unique(cl[ci]); ix = {c: np.where(cl[ci] == c)[0] for c in uc}; bs = []
    for _ in range(300):
        s_ = np.concatenate([ix[c] for c in rng.choice(uc, len(uc))]); bs.append(np.linalg.lstsq(X[s_], y[s_], rcond=None)[0])
    bs = np.array(bs); b0 = np.linalg.lstsq(X, y, rcond=None)[0]
    out["regress_hit_p"] = {nm: [round(float(b0[j]), 4), round(float(np.percentile(bs[:, j], 2.5)), 4), round(float(np.percentile(bs[:, j], 97.5)), 4)] for j, nm in enumerate(("intercept", "per_slot", "per_cell"))}
    out["_valid_mask"] = val
    return out


res = {}; base = None
for arg in sys.argv[1:]:
    tag, d = arg.split("=", 1); r = load(d); mode = r["meta"]["mode"]
    cl = np.array(["|".join(map(str, t)) for t in r["meta"]["traj"]]) if mode == "v3" else np.arange(len(r["q"])).astype(str)
    m = r["q"][:, 3] >= VMIN
    rec = dict(mode=mode, n=int(len(m)), n_moving=int(m.sum()), vmin=VMIN, predictor=r["meta"]["predictor"])
    pcf = per_clip(r, r["flow"]); rec["flow"] = summarize(pcf, m, cl); rec["flow_by_disp"] = by_disp(pcf, m, cl); rec["flow_valid"] = by_disp_valid(pcf, m, cl)
    VAL = rec["flow_valid"].pop("_valid_mask")
    if mode == "v3":
        pcg = per_clip(r, r["gt"]); rec["gt"] = summarize(pcg, m, cl); rec["gt_by_disp"] = by_disp(pcg, m, cl); rec["gt_valid"] = by_disp_valid(pcg, m, cl); rec["gt_valid"].pop("_valid_mask")
        fg = (np.abs(cell(r["flow"]) - cell(r["gt"])).max(-1) <= 1).astype(float)
        rec["flow_vs_gt"] = [boot(fg[m, i], cl[m]) for i in range(8)]
    if base is None:
        base = (tag, pcf, m, cl, VAL)
    else:
        bt, bp, bm, bcl, bval = base
        if bval.shape == VAL.shape:
            vv = bval & VAL; dsp = pcf["cheb"]
            rec["paired_valid_vs_" + bt] = [dict(lo=lo, hi=hi, n=int((vv & (dsp >= lo) & (dsp < hi)).sum()),
                                                 d_hit_p=boot((pcf["hit_p"] - bp["hit_p"])[vv & (dsp >= lo) & (dsp < hi)], cl[np.nonzero(vv & (dsp >= lo) & (dsp < hi))[0]]))
                                            for lo, hi in zip(BINS[:-1], BINS[1:]) if (vv & (dsp >= lo) & (dsp < hi)).sum() >= 10]
        if len(bm) == len(m) and np.array_equal(bm, m):
            rec["paired_vs_" + bt] = {f"{k}_{nm}": [boot((pcf[f"{k}_{nm}"] - bp[f"{k}_{nm}"])[m, i] if k != "prog" else (np.clip(pcf[f"{k}_{nm}"], -1, 2) - np.clip(bp[f"{k}_{nm}"], -1, 2))[m, i], cl[m]) for i in range(8)]
                                      for k in ("hit", "prog", "soft") for nm in ("p", "pT")}
        else:
            rec["paired_vs_" + bt] = "clip 선택이 달라 짝 비교 불가"
    res[tag] = rec

json.dump(res, open(OUT / f"flowtrack{os.environ.get('OUT_TAG', '')}.json", "w"), indent=1, default=float)
f2 = lambda xs: " ".join(f"{x[0]:+.2f}" if isinstance(x, list) else f"{x:+.2f}" for x in xs)
for tag, rec in res.items():
    for tgt in ("flow", "gt"):
        if tgt not in rec: continue
        R = rec[tgt]; print(f"\n== {tag} [{rec['mode']}] 유사정답={tgt}  움직임 clip {rec['n_moving']}/{rec['n']} (|v|≥{VMIN})   슬롯 0..7")
        print("  변위(칸)      " + " ".join(f"{x['disp']:5.2f}" for x in R))
        print("  copy          " + f2([x["copy"] for x in R]) + "   | cv 오라클 " + f2([x["cv"] for x in R]))
        for nm in ("enc", "p", "pT"):
            print(f"  hit {nm:4s}      " + f2([x[f'hit_{nm}'] for x in R]) + "   | −copy " + f2([x[f'hitcopy_{nm}'] for x in R]) + "   | −null " + f2([x[f'hitnull_{nm}'] for x in R]))
            print(f"  prog {nm:4s}     " + f2([x[f'prog_{nm}'] for x in R]) + "   | 중앙 " + f2([x[f'progmed_{nm}'] for x in R]) + "   | soft " + " ".join(f"{x[f'soft_{nm}'][0]:+.3f}" for x in R))
    if "flow_vs_gt" in rec: print("  flow vs GT ≤1칸 " + " ".join(f"{x[0]:.2f}" for x in rec["flow_vs_gt"]))
    for bk in ("flow_by_disp", "gt_by_disp"):
        if bk not in rec: continue
        print(f"  [{bk}] 변위 구간 (칸, Chebyshev) · 슬롯   n    copy   hit_p  hit_p−copy   prog_p  prog_enc  hit_enc")
        for b in rec[bk]["bins"]:
            print(f"    [{b['lo']},{b['hi']:>2}) {b['slots']:5s} {b['n']:5d}  {b['copy'][0]:.2f}  {b['hit_p'][0]:.2f}  {b['hitcopy_p'][0]:+.2f} [{b['hitcopy_p'][1]:+.2f},{b['hitcopy_p'][2]:+.2f}]  {b['prog_p'][0]:+.2f}  {b['prog_enc'][0]:+.2f}  {b['hit_enc'][0]:.2f}")
        print("    회귀 hit_p−copy ~ 1 + slot + cell: " + json.dumps(rec[bk]["regress_hitcopy_p"]))
    for bk in ("flow_valid", "gt_valid"):
        if bk not in rec: continue
        V = rec[bk]; print(f"  [{bk}] 두 유사정답 합의 쌍 {V['n_valid']} ({V['frac_valid']:.0%})   변위 구간 · 슬롯   n   P(hit_p)  P(hit_pT)  prog_p   cv 오라클")
        for b in V["bins"]:
            print(f"    [{b['lo']},{b['hi']:>2}) {b['slots']:5s} {b['n']:5d}  {b['hit_p'][0]:.2f} [{b['hit_p'][1]:.2f},{b['hit_p'][2]:.2f}]  {b['hit_pT'][0]:.2f}  {b['prog_p'][0]:+.2f}   {b['cv'][0]:.2f}")
        print("    회귀 hit_p ~ 1 + slot + cell: " + json.dumps(V["regress_hit_p"]))
    for k, v in rec.items():
        if k.startswith("paired_valid_vs_"):
            print(f"  Δhit_p ({k[16:]} 대비, 합의 쌍) " + " ".join(f"[{b['lo']},{b['hi']}) {b['d_hit_p'][0]:+.3f}[{b['d_hit_p'][1]:+.2f},{b['d_hit_p'][2]:+.2f}] n{b['n']}" for b in v))
    for k, v in rec.items():
        if k.startswith("paired_vs_") and isinstance(v, dict):
            for kk in ("hit_p", "prog_p", "soft_p"):
                print(f"  Δ{kk} ({k[10:]} 대비) " + " ".join(f"{x[0]:+.3f}[{x[1]:+.2f},{x[2]:+.2f}]" for x in v[kk]))
