#!/usr/bin/env python3
"""RollOut_v2 — 운동 법칙 7종에서 미래 8슬롯의 물체 위치 (x, y) 를 읽고, 두 미래(ledge/wall) 중 어느 쪽을 따르는지 본다.

자(readout): 슬롯 256토큰 공간평균 → (8, 1280) → 좌표(화면 x, y)마다 **w 하나를 8슬롯이 공유**하는 최소제곱.
자를 정하는 곳 (전부 30% 학습 / 70% held-out, 셀 층화; plan 의 holdout primary 레벨은 평가 전용):
  p_A   p 로 학습, 7 시나리오 전부 섞음            ← 주. 시나리오를 섞으면 선형 w 는 장면별 기울기 속임수를 못 만든다 (가산성)
  p_B   p 로 학습, ledge·wall 제외                  ← ablation. A 와 판정이 같아야 "라벨 흡수" 가 없었다는 뜻
  h     h 로 학습 (ledge/wall 의 imp 클립도 포함 — h 는 실제 미래를 봤다)   → h→p 이식, h self 는 천장
  z     z(문맥 튜블릿 8개, 문맥 시점 라벨) 로 학습                            → z→p 이식, z self 는 천장
두 미래 판정 (ledge: 낙하 vs 부유, wall: 정지 vs 통과): 슬롯 0 제외, 불가능 궤적이 화면 안인 슬롯만, (x̂,ŷ) 평면 거리 + x 기울기.
null 3종 (튜블릿 셔플 / 8칸 동일 / 라벨 셔플, p_A 학습 집합) → β = 0.

  python z_research/scripts/analysis/rollout2_position.py
"""
from __future__ import annotations
import argparse, csv, json, re, sys, time
from pathlib import Path
import numpy as np

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
CACHE = Path("/local_datasets/world/world_analysis/cache/rollout_v2_vith")
INDEX = ROOT / "data_csv/rollout_v2/index_probe.csv"
OUT = ROOT / "z_research/RollOutV2/exp_results/position_regression.json"
POOLED = CACHE / "_pooled_zph8.npz"
S, T, D, CTX = 256, 8, 1280, 16
FPS, STRIDE = 16.0, 3
SCEN = ["flat_v", "flat_a", "ramp_a", "arc", "fall", "ledge", "wall"]
MOVING = {"flat_v": "x", "flat_a": "x", "ramp_a": "x", "arc": "xy", "fall": "y", "ledge": "xy", "wall": "x"}
TRAIN_FRAC = 0.3


def die(m):
    print(f"\nERROR: {m}\n", file=sys.stderr); sys.exit(1)


def arr(s):
    return np.array([float(v) for v in re.split(r"[;,\s|]+", str(s).strip().strip("[]")) if v], float)


LAM = 1e-3          # ridge: λ = LAM × 행 수. 30% 학습(시나리오당 170 clip) 에서 OLS 가 붕괴해 넣었다 (V1 은 3,696 clip 이라 불필요했다).
                    # flat_v 실측: OLS −0.86 → ridge 0.83 (70% 학습 OLS 0.82 와 같다). 풀링 fit 에서는 ridge 가 0.06 깎으므로 λ 를 작게 둔다


def fit_lstsq(X, y, lam=LAM):
    """중심화 ridge. w 의 마지막 원소가 bias (apply_w 규약 유지). lam=0 이면 순수 최소제곱."""
    mu = X.mean(0); ym = y.mean(); Xc = X - mu
    if lam == 0:
        w = np.linalg.lstsq(Xc, y - ym, rcond=None)[0]
    else:
        w = np.linalg.solve(Xc.T @ Xc + lam * len(X) * np.eye(X.shape[1]), Xc.T @ (y - ym))
    return np.append(w, ym - mu @ w)


def apply_w(X, w):
    return X @ w[:-1] + w[-1]


def slope(x):
    t = np.arange(T, dtype=np.float64); tc = t - t.mean(); return (x * tc).sum(1) / (tc ** 2).sum()


def quad(x):
    t = np.arange(T, dtype=np.float64); A = np.vstack([np.ones_like(t), t, t * t]).T
    return np.linalg.lstsq(A, x.T, rcond=None)[0][2]


def r2_within(x, y):
    yc = y - y.mean(1, keepdims=True); xc = x - x.mean(1, keepdims=True); d = (yc ** 2).sum()
    return float(1 - ((yc - xc) ** 2).sum() / d) if d > 0 else None


def beta_r2(b_hat, b_true):
    if len(np.unique(np.round(b_true, 6))) < 3:
        return None, None
    A = np.vstack([b_true, np.ones_like(b_true)]).T; beta, c0 = np.linalg.lstsq(A, b_hat, rcond=None)[0]
    return float(beta), float(1 - ((b_hat - A @ [beta, c0]) ** 2).sum() / ((b_hat - b_hat.mean()) ** 2).sum())


def pool(rows_needed, bs=64):
    if POOLED.is_file():
        z = np.load(POOLED)
        if np.array_equal(z["rows"], rows_needed):
            print(f"    풀링 캐시 사용 {POOLED}"); return {"z": z["z"], "p": z["p"], "h": z["h"]}
    meta = json.loads((CACHE / "meta.json").read_text()); bc = meta["base_counts"]
    for b, n in (("ctx_masked", T * S), ("predictor", T * S), ("target", 2 * T * S)):
        if bc.get(b) != n:
            die(f"캐시 base '{b}' 토큰 수 {bc.get(b)} != {n}")
    Z = np.load(CACHE / "ctx_masked.npy", mmap_mode="r"); P = np.load(CACHE / "predictor.npy", mmap_mode="r"); H = np.load(CACHE / "target.npy", mmap_mode="r")
    n = len(rows_needed); out = {k: np.empty((n, T, D), np.float32) for k in ("z", "p", "h")}
    order = np.argsort(rows_needed); t0 = time.time()
    for k in range(0, n, bs):
        sel = order[k:k + bs]; r = rows_needed[sel]
        out["z"][sel] = np.asarray(Z[r]).reshape(-1, T, S, D).mean(2)
        out["p"][sel] = np.asarray(P[r]).reshape(-1, T, S, D).mean(2)
        out["h"][sel] = np.asarray(H[r, T * S:]).reshape(-1, T, S, D).mean(2)
        if k % (bs * 20) == 0:
            print(f"    풀링 {k}/{n}  {time.time()-t0:.0f}s", flush=True)
    print(f"    풀링 완료 {n}개  {time.time()-t0:.0f}s"); np.savez(POOLED, rows=rows_needed, **out); return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0); ap.add_argument("-o", "--out", type=Path, default=OUT)
    a = ap.parse_args()
    idx = list(csv.DictReader(INDEX.open())); vids = [r["video_id"] for r in idx]
    meta = json.loads((CACHE / "meta.json").read_text()); row_of = {v: i for i, v in enumerate(meta["video_ids"])}
    if any(v not in row_of for v in vids):
        die("캐시에 없는 video")
    F = pool(np.array([row_of[v] for v in vids]))
    res_px = float(idx[0]["resolution"]) / 2
    px = np.stack([arr(r["px_x_by_sample"]) for r in idx]); py = np.stack([arr(r["px_y_by_sample"]) for r in idx])
    inf = np.stack([arr(r["in_frame_by_sample"]) for r in idx]) > 0
    Lx = px.reshape(-1, 16, 2).mean(2) / res_px - 1; Ly = py.reshape(-1, 16, 2).mean(2) / res_px - 1    # (n, 16 튜블릿): 0..7 문맥, 8..15 미래
    INF = inf.reshape(-1, 16, 2).all(2)
    zcm = np.stack([arr(r["z_cm_by_sample"]) for r in idx]); hw = float(idx[0]["frame_half_width_cm"])
    ay = np.polyfit(zcm.ravel(), py.ravel(), 1)[0]; cm_per = {"x": hw, "y": res_px / abs(ay)}; dt = STRIDE * 2 / FPS
    sc = np.array([r["scenario"] for r in idx]); plaus = np.array([int(r["plausible"]) for r in idx])
    prim = np.array([float(r["primary"]) for r in idx]); sec = np.array([float(r["secondary"]) for r in idx])
    hold = np.array([r["holdout"] in ("1", "True", "true") for r in idx]); blk = np.array([r["block_id"] for r in idx])
    imp_of = {blk[i]: i for i in np.where(plaus == 0)[0]}

    # ── split: 가능 클립, 셀(시나리오×primary×secondary) 층화 30/70; holdout 레벨은 평가 전용 ──
    rng = np.random.RandomState(a.seed); tr, va, ho = [], [], []
    for scn in SCEN:
        ids = np.where((sc == scn) & (plaus == 1))[0]
        ho += list(ids[hold[ids]]); fit = ids[~hold[ids]]
        cell = np.array([f"{prim[i]}|{sec[i]}" for i in fit])
        for c in np.unique(cell):
            i = fit[cell == c]; rng.shuffle(i); k = int(round(TRAIN_FRAC * len(i))); tr += list(i[:k]); va += list(i[k:])
    tr, va, ho = (np.array(sorted(v)) for v in (tr, va, ho))
    tr_B = tr[~np.isin(sc[tr], ["ledge", "wall"])]
    tr_h = np.concatenate([tr, np.array([imp_of[blk[i]] for i in tr if blk[i] in imp_of])])   # h 는 imp 클립도 (실제 미래를 봤다)
    print(f"split: train {len(tr)} (B: {len(tr_B)}, h: {len(tr_h)}) / val {len(va)} / holdout-level {len(ho)}   행 = ×8")
    fut = lambda s_, ids: F[s_][ids].reshape(-1, D).astype(np.float64)
    Lf = {"x": Lx[:, 8:], "y": Ly[:, 8:]}; Lc = {"x": Lx[:, :8], "y": Ly[:, :8]}
    # z 학습: 문맥 튜블릿 중 화면 안인 것만
    zm = INF[tr][:, :8].ravel()
    W = {"p_A": {c: fit_lstsq(fut("p", tr), Lf[c][tr].ravel()) for c in "xy"},
         "p_B": {c: fit_lstsq(fut("p", tr_B), Lf[c][tr_B].ravel()) for c in "xy"},
         "h": {c: fit_lstsq(fut("h", tr_h), Lf[c][tr_h].ravel()) for c in "xy"},
         "z": {c: fit_lstsq(fut("z", tr)[zm], Lc[c][tr].ravel()[zm]) for c in "xy"}}
    # 시나리오별 자 — "위치를 얼마나 잘 읽나" 의 주 수치 (7 시나리오 풀링은 p 에서 R² 0.1 을 깎는다)
    for scn in SCEN:
        t_ = tr[sc[tr] == scn]; th_ = tr_h[sc[tr_h] == scn]
        W[f"p_{scn}"] = {c: fit_lstsq(fut("p", t_), Lf[c][t_].ravel()) for c in "xy"}
        W[f"h_{scn}"] = {c: fit_lstsq(fut("h", th_), Lf[c][th_].ravel()) for c in "xy"}
    # 자(w) × 적용 소스
    RUNS = {"p_self_scn": ("p_scn", "p"), "h_self_scn": ("h_scn", "h"),
            "p_self_A": ("p_A", "p"), "p_self_B": ("p_B", "p"), "h_self": ("h", "h"), "h_to_p": ("h", "p"),
            "z_to_p": ("z", "p"), "z_self_ctx": ("z", "z")}

    def read(run, ids):
        wname, src = RUNS[run]; X = fut(src, ids)
        if wname.endswith("_scn"):
            wname = wname[:2] + str(sc[ids[0]])
        return {c: apply_w(X, W[wname][c]).reshape(len(ids), T) for c in "xy"}

    def metrics(xh, ids, ctx=False):
        L = Lc if ctx else Lf; rec = {"n": int(len(ids))}
        for c in "xy":
            x, y = xh[c], L[c][ids]; b, bl = slope(x), slope(y); beta, r2v = beta_r2(b, bl)
            rec[c] = dict(r2_within=r2_within(x, y), mae_px=float(np.abs(x - y).mean() * res_px), offset_px=float((x - y).mean() * res_px),
                          beta=beta, r2_speed=r2v, vel_mae_cm_s=float(np.abs(b - bl).mean() * cm_per[c] / dt),
                          vel_true_range=[float(bl.min() * cm_per[c] / dt), float(bl.max() * cm_per[c] / dt)],
                          acc_hat=float(quad(x).mean() * 2 * cm_per[c] / dt ** 2), acc_true=float(quad(y).mean() * 2 * cm_per[c] / dt ** 2))
        return rec

    R = {"meta": dict(cache=str(CACHE), n_clip=len(vids), seed=a.seed, train_frac=TRAIN_FRAC, n_train=int(len(tr)), n_val=int(len(va)), n_holdout_level=int(len(ho)),
                      label="screen-normalised (px/144 − 1); x, y separately; one shared w per coordinate across 8 tubelets",
                      ridge_lambda_per_row=LAM,
                      readouts={"p_scn/h_scn": "per scenario (primary self numbers)", "p_A": "p, all 7 scenarios", "p_B": "p, without ledge/wall", "h": "h incl. ledge/wall imp clips", "z": "context tubelets of z with context-time labels"},
                      cm_per_img=cm_per), "by_scenario": {}, "two_futures": {}, "nulls": {}}
    print("\n[위치 readout — 시나리오 × 평가셋 × 자]   R² = 클립내부, MAE px, β = 읽은 기울기 vs 정답 기울기 (7 레벨)")
    for scn in SCEN:
        R["by_scenario"][scn] = {}
        for ev, pool_ids in (("val", va), ("holdout", ho)):
            ids = pool_ids[sc[pool_ids] == scn]; R["by_scenario"][scn][ev] = {}
            for run in RUNS:
                xh = read(run, ids); rec = metrics(xh, ids, ctx=(run == "z_self_ctx")); R["by_scenario"][scn][ev][run] = rec
            cols = ["x", "y"] if MOVING[scn] == "xy" else [MOVING[scn]]
            for run in RUNS:
                rec = R["by_scenario"][scn][ev][run]
                f3 = lambda v: " n/a " if v is None else f"{v:.3f}"
                print(f"  {scn:<7} {ev:<7} {run:<10} n={rec['n']:>4}  " + "  ".join(
                    f"[{c}] R²={f3(rec[c]['r2_within'])} MAE={rec[c]['mae_px']:5.1f} off={rec[c]['offset_px']:+6.1f} β={f3(rec[c]['beta'])} vMAE={rec[c]['vel_mae_cm_s']:4.0f} acc={rec[c]['acc_hat']:5.0f}/{rec[c]['acc_true']:5.0f}" for c in cols))
    # ── 두 미래 (ledge / wall) ──
    print("\n[두 미래 판정 — pos 클립 (val+holdout). 슬롯 0 제외, 불가능 궤적이 화면 안인 슬롯만]")
    for scn in ("ledge", "wall"):
        ids = np.concatenate([va[sc[va] == scn], ho[sc[ho] == scn]]); imp_ids = np.array([imp_of[blk[i]] for i in ids])
        use = INF[imp_ids][:, 8:].copy(); use[:, 0] = False
        for k in range(len(ids)):
            if not use[k].any():
                use[k, 1] = True
        R["two_futures"][scn] = {}
        for run in ("p_self_A", "p_self_B", "h_to_p", "z_to_p", "h_self"):
            xh = read(run, ids)
            d_pos = np.array([np.hypot(xh["x"][k] - Lf["x"][i], xh["y"][k] - Lf["y"][i])[use[k]].mean() for k, i in enumerate(ids)])
            d_imp = np.array([np.hypot(xh["x"][k] - Lf["x"][j], xh["y"][k] - Lf["y"][j])[use[k]].mean() for k, j in enumerate(imp_ids)])
            bx = slope(xh["x"]); bp = slope(Lf["x"][ids]); bi = slope(Lf["x"][imp_ids]); by = quad(xh["y"]); qp = quad(Lf["y"][ids]); qi = quad(Lf["y"][imp_ids])
            rec = dict(n=int(len(ids)), frac_possible_by_distance=float((d_pos < d_imp).mean()),
                       dist_px=dict(possible=float(d_pos.mean() * res_px), impossible=float(d_imp.mean() * res_px)),
                       x_slope_cm_s=dict(read=float(bx.mean() * -cm_per["x"] / dt), possible=float(bp.mean() * -cm_per["x"] / dt), impossible=float(bi.mean() * -cm_per["x"] / dt)),
                       y_acc_cm_s2=dict(read=float(by.mean() * 2 * cm_per["y"] / dt ** 2), possible=float(qp.mean() * 2 * cm_per["y"] / dt ** 2), impossible=float(qi.mean() * 2 * cm_per["y"] / dt ** 2)),
                       slots_used_mean=float(use.sum(1).mean()))
            if run == "h_self":       # 천장: imp 클립의 h 는 실제 imp 미래를 봤다 → impossible 쪽이어야
                xi = read(run, imp_ids)
                dp = np.array([np.hypot(xi["x"][k] - Lf["x"][i], xi["y"][k] - Lf["y"][i])[use[k]].mean() for k, i in enumerate(ids)])
                di = np.array([np.hypot(xi["x"][k] - Lf["x"][j], xi["y"][k] - Lf["y"][j])[use[k]].mean() for k, j in enumerate(imp_ids)])
                rec["on_imp_clips_frac_impossible"] = float((di < dp).mean())
                rec["on_imp_clips_x_slope_cm_s"] = float(slope(xi["x"]).mean() * -cm_per["x"] / dt); rec["on_imp_clips_y_acc"] = float(quad(xi["y"]).mean() * 2 * cm_per["y"] / dt ** 2)
            R["two_futures"][scn][run] = rec
            extra = f" | imp 클립의 h: impossible 쪽 {100*rec['on_imp_clips_frac_impossible']:.0f}%, x기울기 {rec['on_imp_clips_x_slope_cm_s']:.0f}, y가속 {rec['on_imp_clips_y_acc']:.0f}" if run == "h_self" else ""
            print(f"  {scn:<6} {run:<9} possible 쪽 {100*rec['frac_possible_by_distance']:5.1f}%  dist {rec['dist_px']['possible']:5.1f} vs {rec['dist_px']['impossible']:5.1f} px | x기울기 읽음 {rec['x_slope_cm_s']['read']:4.0f} (pos {rec['x_slope_cm_s']['possible']:.0f} / imp {rec['x_slope_cm_s']['impossible']:.0f}) | y가속 읽음 {rec['y_acc_cm_s2']['read']:4.0f} (pos {rec['y_acc_cm_s2']['possible']:.0f} / imp {rec['y_acc_cm_s2']['impossible']:.0f}) [슬롯 {rec['slots_used_mean']:.1f}]{extra}")
    # ── null (p_A 학습 집합) ──
    print("\n[null — p_A, 움직이는 좌표, val]")
    rs = np.random.RandomState(123); Fp = F["p"]
    Fs = Fp.copy()
    for i in range(len(Fs)):
        Fs[i] = Fs[i][rs.permutation(T)]
    Fc = Fp.copy(); Fc[:] = Fc.mean(1, keepdims=True)
    for c in "xy":
        Lsh = Lf[c].copy(); Lsh[tr] = Lf[c][rs.permutation(tr)]
        for name, FF, LL in (("tubelet_shuffle", Fs, Lf[c]), ("slots_identical", Fc, Lf[c]), ("label_shuffle", Fp, Lsh)):
            ww = fit_lstsq(FF[tr].reshape(-1, D).astype(np.float64), LL[tr].ravel())
            for scn in SCEN:
                if c not in MOVING[scn]:
                    continue
                ids = va[sc[va] == scn]; xh = apply_w(FF[ids].reshape(-1, D).astype(np.float64), ww).reshape(len(ids), T)
                bt, _ = beta_r2(slope(xh), slope(Lf[c][ids])); R["nulls"].setdefault(scn, {})[f"{name}_{c}"] = bt
    for scn in SCEN:
        print(f"  {scn:<7} " + "  ".join(f"{k}: β={'n/a' if v is None else round(v, 3)}" for k, v in R["nulls"].get(scn, {}).items()))
    a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(R, indent=1, ensure_ascii=False)); print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
