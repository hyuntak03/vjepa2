#!/usr/bin/env python3
"""AR ep40 vs 릴리즈 — 각자의 정체 자 (identity_ar40 · identity_r8) 로 읽은 RollOut_v3 16 창 · v11 late 를 나란히 (2026-09-30).

질문 (사용자): AR 이 중력 (가속) 을 이해했나. 자는 각 predictor 의 p 로 따로 학습했고 (학습셋 test: AR p 91.1 / 릴리즈 p 84.0), 판독 규약은 RollOutV3 와 같다:
  '있음' = 57-way argmax ≠ 없음(56), 위치 px = xy·144 − 치우침 (attn_bias_px), 진실 t = 샘플 (32+2t, 33+2t) 평균 (v3) / (16+2t, 17+2t) (v11)
  복사 = 문맥 마지막 튜블릿 (샘플 30·31) 의 진실 자리. 등속 외삽 cv_t = 마지막 자리 + v_ctx · (2t+1.5) (v_ctx = 샘플 28→31 평균 속도 / 프레임).
  가속 반영 α = ((pos − cv)·(truth − cv)) / |truth − cv|²  — 0 = 등속 외삽, 1 = 진실 (가속 반영). |truth − cv| ≥ 9 px 인 칸만.
출력: exp_results/scene/ar40_readout_compare.json + 표 (시나리오 × 튜블릿 구간).
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2"); EXP = ROOT / "z_research/RollOutV3/exp_results"; OUT = ROOT / "auto_research/exp_results/scene"; OUT.mkdir(parents=True, exist_ok=True)
R, SPLIT, NONE = 144.0, 32, 56
DEC = {"release": "identity_r8", "AR40": "identity_ar40"}
rng = np.random.default_rng(0)


def load(dec, win):
    z = np.load(EXP / f"windows_{dec}" / "readings.npz", allow_pickle=True); b = json.load(open(EXP / dec / "attn_bias_px.json"))
    return z, b


def bootci(v, blk, B=500):
    ok = np.isfinite(v); v, blk = v[ok], blk[ok]
    if len(v) == 0: return [np.nan] * 3
    ub = np.unique(blk); ix = {u: np.where(blk == u)[0] for u in ub}
    bs = [v[np.concatenate([ix[u] for u in rng.choice(ub, len(ub))])].mean() for _ in range(B)]
    return [round(float(v.mean()), 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]


res = {"v3": {}, "v11": {}}
for tag, dec in DEC.items():
    for win in ("C16_P32", "C16_P16"):
        z, b = load(dec, win); P = int(win.split("_P")[1]); tp = P // 2
        pos_ok = z["role"] == "roll"; L = z["truth"] * R; blk = z["block_id"]
        gt = L[:, SPLIT:SPLIT + P].reshape(len(L), tp, 2, 2).mean(2)                       # (n, tp, 2)
        last = L[:, 30:32].mean(1); vctx = (L[:, 31] - L[:, 28]) / 3.0                       # px/frame
        dt = (2 * np.arange(tp) + 1.5)[None, :, None]
        cv = last[:, None] + vctx[:, None] * dt
        for rep in ("p", "h", "z"):
            if f"{win}_{rep}" not in z.files: continue
            v = z[f"{win}_{rep}"]; pr = z[f"{win}_{rep}_prob"].astype(np.float32)
            present = pr.argmax(-1) != NONE
            pos = v[..., :2] * R - np.array(b[rep], np.float32)
            e_t = np.linalg.norm(pos - gt, axis=-1); e_cv = np.linalg.norm(pos - cv, axis=-1); e_copy = np.linalg.norm(pos - last[:, None], axis=-1)
            d = gt - cv; dn = (d ** 2).sum(-1); alpha = ((pos - cv) * d).sum(-1) / np.maximum(dn, 1e-6); sep = np.sqrt(dn) >= 9
            for sc in sorted(set(z["scenario"])):
                m = pos_ok & (z["scenario"] == sc)
                row = {}
                for nm, sl in (("t0-3", slice(0, 4)), ("t4-7", slice(4, 8)), ("t8-15", slice(8, tp))):
                    if sl.start >= tp: continue
                    mm = m[:, None] & np.ones((1, tp), bool); mm[:, :] = False; mm[m, sl] = True
                    pm = mm & present; am = pm & sep
                    B_ = np.broadcast_to(blk[:, None], mm.shape)
                    row[nm] = dict(present=bootci(present[mm].astype(float), B_[mm]), err_truth=bootci(e_t[pm], B_[pm]), err_cv=bootci(e_cv[pm], B_[pm]),
                                   err_copy=bootci(e_copy[pm], B_[pm]), alpha=bootci(np.clip(alpha[am], -1, 2), B_[am]), n_alpha=int(am.sum()),
                                   truth_minus_cv_px=round(float(np.sqrt(dn)[mm].mean()), 1))
                res["v3"].setdefault(win, {}).setdefault(sc, {})[f"{tag}:{rep}"] = row
# v11 late
for tag, dec in DEC.items():
    z = np.load(EXP / f"v11_{dec}" / "readings.npz", allow_pickle=True); b = json.load(open(EXP / dec / "attn_bias_px.json"))
    L = z["truth"] * R; k = z["sym_k"].astype(int); obj = z["obj"]; cond = z["condition"]; blk = z["block_id"]
    gt = L[:, 16:32].reshape(len(L), 8, 2, 2).mean(2)
    for rep in ("p", "h"):
        v = z[rep][:, -8:] if rep == "h" else z[rep]; pr = (z[f"{rep}_prob"][:, -8:] if rep == "h" else z[f"{rep}_prob"]).astype(np.float32)
        present = pr.argmax(-1) != NONE; pos = v[..., :2] * R - np.array(b[rep], np.float32); e_t = np.linalg.norm(pos - gt, axis=-1)
        for motion in ("static", "moving_flat", "moving_ramp"):
            cm = {"static": ["static_visible", "static_occlusion"], "moving_flat": ["moving_visible_flat", "moving_occlusion_flat"], "moving_ramp": ["moving_visible", "moving_occlusion"]}[motion]
            for kk in range(5):
                m = obj & np.isin(cond, cm) & (k == kk)
                if m.sum() == 0: continue
                B_ = np.broadcast_to(blk[:, None], (len(L), 8))
                mm = np.zeros((len(L), 8), bool); mm[m] = True; pm = mm & present
                e_pm = mm.copy(); e_pm[:, :] = False; e_pm[m, :] = True
                res["v11"].setdefault(f"{motion}", {}).setdefault(f"k{kk}", {})[f"{tag}:{rep}"] = dict(
                    n=int(m.sum()), present_by_t=[round(float(present[m, t].mean()), 2) for t in range(8)], present_all=bootci(present[mm].astype(float), B_[mm]),
                    err_truth_present=bootci(e_t[pm], B_[pm]), err_truth_t3_7=bootci(e_t[pm & (np.arange(8)[None] >= 3)], B_[pm & (np.arange(8)[None] >= 3)]))
            # 빈 장면 오탐 (empty)
        m = (~obj) & np.isin(cond, ["static_occlusion", "moving_occlusion_flat", "moving_occlusion"])
        res["v11"].setdefault("empty_false_alarm", {})[f"{tag}:{rep}"] = round(float(present[m].mean()), 4) if m.sum() else None
json.dump(res, open(OUT / "ar40_readout_compare.json", "w"), indent=1, default=float)

print("=== v3 C16/P32 — 시나리오 × 구간: 있음 % | 진실까지 px | 등속외삽까지 px | 복사까지 px | α (0 등속, 1 진실) [n]")
for sc, rows in res["v3"]["C16_P32"].items():
    print(f"-- {sc}")
    for key in ("release:p", "AR40:p", "AR40:h", "release:z"):
        if key not in rows: continue
        cells = []
        for nm in ("t0-3", "t4-7", "t8-15"):
            r = rows[key].get(nm)
            if not r: continue
            cells.append(f"{nm}: {100*r['present'][0]:.0f}% | {r['err_truth'][0]:.0f} | {r['err_cv'][0]:.0f} | {r['err_copy'][0]:.0f} | α {r['alpha'][0]:+.2f} [{r['n_alpha']}] (Δ {r['truth_minus_cv_px']})")
        print(f"   {key:10s} " + "  ||  ".join(cells))
print("\n=== v11 late (obj) — 있음 % t0..t7 · 진실까지 px (t3-7, 있음 칸)")
for motion, ks in res["v11"].items():
    if motion == "empty_false_alarm": print("empty false alarm", ks); continue
    for kk, rows in ks.items():
        for key in ("release:p", "AR40:p", "AR40:h"):
            if key in rows:
                r = rows[key]; print(f"   {motion:12s} {kk} {key:10s} 있음 {r['present_by_t']}  err t3-7 {r['err_truth_t3_7'][0]:.0f} px  n {r['n']}")
