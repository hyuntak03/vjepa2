#!/usr/bin/env python3
"""v11_full — predictor 가 만든 미래 8슬롯에서 물체 위치가 읽히는가, 그 기울기가 속도인가.
RollOut_v1 의 측정(`rollout_position.py`, METHOD_POSITION_READOUT §9)을 v11 로 옮긴 것.

  x̂(t) = a + b·t (+ c·t²)   t = 0..7 미래 튜블릿
  b̂ ≈ v  ->  슬롯마다 그 시점 위치가 있다      b̂ ≈ 0 -> 마지막 관측을 8칸에 들고 있을 뿐

v11 이 RollOut 과 다른 점 (설계상 중요)
  - 조건마다 궤적이 **하나**다: static 0 / flat 260.12 cm/s / ramp 115.85 + 49.11·t (2차).
    속도 11단계 대신 **세 운동을 한 회귀에 묶는다** — w 하나가 세 조건에서 각각 0 / 260 / 가속
    기울기를 내야 한다. p_t 에 "몇 번째 슬롯인가" 만 남아 있으면(가산적 t-index) 기울기가 조건에
    무관하게 같아진다. 방향(±x)도 같은 논리로 배제한다.
  - 가림 조건(late k)은 **미래 첫 k 프레임도 가려져 있다**. 그 슬롯의 라벨은 해석식이지만 h 도
    가림막을 본다. h 는 32프레임을 한 번에 보는 양방향 인코더라 보간할 수 있다 — 천장으로만 읽을 것.
  - 라벨은 metadata 의 `object_x_cm_by_sample` (32 샘플 = 프로토콜 프레임 0,3,…,93 과 동일).
    픽셀 좌표 `object_px_x_by_sample` 로 선형성(잔차 < 1px)을 검사하고, 실제 PNG 에서 물체 열을
    잡아 부호·스케일을 대조한다 (--pixel-check).

  python z_research/scripts/analysis/v11_position.py            # 전수 obj 18,816 clip, GPU 불필요
"""
from __future__ import annotations
import argparse, csv, json, re, sys, time
from pathlib import Path
import numpy as np

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
CACHE = Path("/local_datasets/world/world_analysis/cache/v11_full_vith")
FRAMES = Path("/local_datasets/world/world_analysis/IntPhysGen_v11")
META = FRAMES / "metadata.csv"
INDEX = ROOT / "data_csv/intphysgen_v11_full/index_probe.csv"
OUT = ROOT / "z_research/v11_roll_out/exp_results/position_regression.json"
S, T_FUT, T_ALL, D = 256, 8, 16, 1280
FPS, STRIDE, CTX = 16.0, 3, 16          # 샘플 프레임 = raw 0,3,…,93 ; 문맥 = 앞 16장


def die(m):
    print(f"\nERROR: {m}\n", file=sys.stderr); sys.exit(1)


def arr(x):
    return np.array([float(v) for v in re.split(r"[;,\s|]+", str(x).strip().strip("[]")) if v], float)


def motion_of(c):
    return "static" if c.startswith("static") else ("flat" if "flat" in c else "ramp")


def timing_of(c):
    return "visible" if "visible" in c else ("early" if c.endswith("_early") else ("mid" if c.endswith("_mid") else "late"))


def fit_lstsq(X, y):
    Xb = np.hstack([X, np.ones((len(X), 1), X.dtype)])
    return np.linalg.lstsq(Xb, y, rcond=None)[0]


def apply_w(X, w):
    return X @ w[:-1] + w[-1]


def slope(x):
    t = np.arange(T_FUT, dtype=np.float64); tc = t - t.mean()
    b = (x * tc).sum(1) / (tc ** 2).sum()
    return b, x.mean(1) - b * t.mean()


def quad(x):
    """x: (N,8) -> (b, c) of x = a + b·t + c·t²  (t = 0..7)."""
    t = np.arange(T_FUT, dtype=np.float64)
    A = np.vstack([np.ones_like(t), t, t * t]).T
    coef = np.linalg.lstsq(A, x.T, rcond=None)[0]        # (3, N)
    return coef[1], coef[2]


def load_labels(vids):
    import pandas as pd
    m = pd.read_csv(META, low_memory=False).set_index("name")
    miss = [v for v in vids if v not in m.index]
    if miss:
        die(f"metadata 에 없는 video {len(miss)}개 (예 {miss[:3]})")
    m = m.loc[vids]
    sf0 = arr(m.sample_frames.iloc[0])
    want = np.arange(0, 32 * STRIDE, STRIDE, dtype=float)
    if len(sf0) != 32 or not np.allclose(sf0, want):
        die(f"sample_frames 가 프로토콜 프레임과 다르다: {sf0[:5]}…")
    X = np.stack([arr(v) for v in m.object_x_cm_by_sample])          # (N, 32) cm
    PX = np.stack([arr(v) for v in m.object_px_x_by_sample])         # (N, 32) px (288 기준)
    if not (np.isfinite(X).all() and np.isfinite(PX).all()):
        die("위치 배열에 NaN 이 있다")
    res = float(m.resolution.iloc[0])
    # 픽셀 <- cm 선형성: 조건 불문 한 직선이어야 라벨 단위가 성립한다
    a, b = np.polyfit(X.ravel(), PX.ravel(), 1)
    resid = np.abs(a * X + b - PX).max()
    hw = (res / 2) / abs(a)                                          # 화면 반폭 (cm, 물체 깊이에서)
    lab = {"px_per_cm": float(a), "px_intercept": float(b), "px_resid_max": float(resid),
           "frame_half_width_cm": float(hw), "world_x_sign_on_screen": "left" if a < 0 else "right"}
    # 미래 튜블릿 t = 샘플 16+2t, 17+2t 의 평균
    fut = X[:, CTX:].reshape(-1, T_FUT, 2).mean(2)                  # (N, 8) cm
    Y = fut / hw
    cols = dict(condition=m.condition.values.astype(str), k=m.sym_k.fillna(0).astype(int).values,
                v0=m.v0_cm_s.astype(float).values, acc=m.ramp_acc_cm_s2.astype(float).values,
                n_ctx_hidden=m.n_context_hidden.fillna(0).astype(int).values,
                n_pred_hidden=m.n_predict_hidden.fillna(0).astype(int).values,
                x_ctx_end=X[:, CTX - 1], x_start=X[:, 0])
    return Y, cols, lab, X


POOLED = CACHE / "_pooled_obj_fut8.npz"      # 풀링 결과 캐시 (1.5 GB, 캐시 폴더 옆 = 로컬 디스크)


def pool(rows_needed, bs=64):
    """캐시에서 필요한 행만 읽어 (n, 8, 1280) 공간평균. h 는 target 의 미래 절반."""
    if POOLED.is_file():
        z = np.load(POOLED)
        if np.array_equal(z["rows"], rows_needed):
            print(f"    풀링 캐시 사용 {POOLED}")
            return {"p": z["p"], "h": z["h"]}
    meta = json.loads((CACHE / "meta.json").read_text())
    bc = meta["base_counts"]
    for b, n in (("predictor", T_FUT * S), ("target", T_ALL * S)):
        if bc.get(b) != n:
            die(f"캐시 base '{b}' 토큰 수 {bc.get(b)} != {n}")
    P = np.load(CACHE / "predictor.npy", mmap_mode="r")
    H = np.load(CACHE / "target.npy", mmap_mode="r")
    n = len(rows_needed)
    out = {"p": np.empty((n, T_FUT, D), np.float32), "h": np.empty((n, T_FUT, D), np.float32)}
    order = np.argsort(rows_needed)                                  # 디스크는 오름차순으로
    t0 = time.time()
    for k in range(0, n, bs):
        sel = order[k:k + bs]; r = rows_needed[sel]
        out["p"][sel] = np.asarray(P[r]).reshape(-1, T_FUT, S, D).mean(2)
        out["h"][sel] = np.asarray(H[r, T_FUT * S:]).reshape(-1, T_FUT, S, D).mean(2)
        if k % (bs * 25) == 0:
            print(f"    풀링 {k}/{n}  {time.time()-t0:.0f}s", flush=True)
    print(f"    풀링 완료 {n}개  {time.time()-t0:.0f}s")
    np.savez(POOLED, rows=rows_needed, p=out["p"], h=out["h"])
    return out


def pixel_check(vids, X, PXlab, cols, n=24, seed=0):
    """실제 PNG 에서 물체 열을 시간중앙값 배경차분으로 잡아 metadata 픽셀 x 와 회귀한다."""
    from PIL import Image
    rng = np.random.RandomState(seed)
    cand = [i for i, c in enumerate(cols["condition"]) if "visible" in c and not c.startswith("static")]
    pick = rng.choice(cand, size=min(n, len(cand)), replace=False)
    idx = {r["video_id"]: r for r in csv.DictReader(INDEX.open())}
    got, want = [], []
    for i in pick:
        fdir = FRAMES / idx[vids[i]]["file_name"]
        frames = [np.asarray(Image.open(fdir / f"{f:06d}.png").convert("L"), np.float32) for f in range(0, 96, 3)]
        stack = np.stack(frames); bg = np.median(stack, 0)
        for j in (0, 8, 15, 20, 28):
            d = np.abs(stack[j] - bg); thr = d.mean() + 4 * d.std()
            ys, xs = np.nonzero(d > thr)
            if len(xs) < 20:
                continue
            got.append(np.median(xs)); want.append(PXlab[i, j])
    got, want = np.array(got), np.array(want)
    a, b = np.polyfit(want, got, 1)
    r2 = 1 - ((got - (a * want + b)) ** 2).sum() / ((got - got.mean()) ** 2).sum()
    return dict(n_points=int(len(got)), slope=float(a), intercept_px=float(b), r2=float(r2),
                resid_px_max=float(np.abs(got - (a * want + b)).max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pixel-check", type=int, default=24)
    ap.add_argument("-o", "--out", type=Path, default=OUT)
    a = ap.parse_args()

    idx = [r for r in csv.DictReader(INDEX.open()) if r["probe_type"] == "obj"]
    vids = [r["video_id"] for r in idx]
    meta = json.loads((CACHE / "meta.json").read_text())
    row_of = {v: i for i, v in enumerate(meta["video_ids"])}
    miss = [v for v in vids if v not in row_of]
    if miss:
        die(f"캐시에 없는 video {len(miss)}개")
    rows_needed = np.array([row_of[v] for v in vids])
    print(f"obj clip {len(vids)}  (probe_type == obj: 가능 + 물체 있음)")

    Y, C, lab, Xcm = load_labels(vids)
    hw = lab["frame_half_width_cm"]
    print(f"라벨: px = {lab['px_per_cm']:.4f}·x_cm + {lab['px_intercept']:.1f}  잔차 max {lab['px_resid_max']:.2f}px"
          f"  -> 화면 반폭 {hw:.1f} cm, world +x 는 화면 {lab['world_x_sign_on_screen']}")
    if lab["px_resid_max"] > 2.0:      # ramp 는 물체 깊이가 조금 변해 0.5% (≈1px) 어긋난다. 그 이상이면 라벨 단위가 틀린 것
        die("픽셀-cm 선형성이 깨졌다")
    if a.pixel_check:
        import pandas as pd
        PXlab = np.stack([arr(v) for v in pd.read_csv(META, low_memory=False).set_index("name").loc[vids].object_px_x_by_sample])
        lab["pixel_check"] = pixel_check(vids, Xcm, PXlab, C, n=a.pixel_check, seed=a.seed)
        pc = lab["pixel_check"]
        print(f"픽셀 대조 ({pc['n_points']}점): 측정 = {pc['slope']:.3f}·라벨 + {pc['intercept_px']:.1f}  R² {pc['r2']:.4f}  잔차max {pc['resid_px_max']:.1f}px")

    motion = np.array([motion_of(c) for c in C["condition"]])
    timing = np.array([timing_of(c) for c in C["condition"]])
    kk = np.where(timing == "visible", 0, C["k"])
    direction = np.sign(Y[:, -1] - Y[:, 0]); direction[motion == "static"] = 0
    cm_per_unit = hw / (STRIDE * 2 / FPS)                            # 칸당 정규화x -> cm/s
    b_lab, _ = slope(Y); bq_lab, cq_lab = quad(Y)

    print("풀링 (캐시에서 obj 행만)")
    F = pool(rows_needed)

    # split — (condition, k, 방향) 층화 50/50
    rng = np.random.RandomState(a.seed)
    cell = np.array([f"{c}|{k}|{d:+.0f}" for c, k, d in zip(C["condition"], kk, direction)])
    tr, va = [], []
    for c in np.unique(cell):
        i = np.where(cell == c)[0]; rng.shuffle(i); h = len(i) // 2
        tr += list(i[:h]); va += list(i[h:])
    tr, va = np.array(sorted(tr)), np.array(sorted(va))
    print(f"split  train {len(tr)} / val {len(va)}  ({len(np.unique(cell))} 셀 층화)")

    flat = lambda s, ids: F[s][ids].reshape(-1, D).astype(np.float64)
    al = np.where(direction == 0, 1.0, direction)     # 정답 진행 방향에 정렬 (±x 가 섞여 평균이 상쇄되지 않게)
    s2 = cm_per_unit / (STRIDE * 2 / FPS) * 2          # 칸² 계수 -> cm/s²

    def metrics(ids, xh_ids):
        """ids: 전역 인덱스, xh_ids: 그 클립들의 (n, 8) 예측."""
        y = Y[ids]; x = xh_ids
        yc = y - y.mean(1, keepdims=True); xc = x - x.mean(1, keepdims=True)
        den = (yc ** 2).sum()
        r2 = float(1 - ((yc - xc) ** 2).sum() / den) if den > 0 else None
        den_all = ((y - y.mean()) ** 2).sum()                       # 클립 간 변이 포함 (static 은 이것만 정의된다)
        r2_all = float(1 - ((y - x) ** 2).sum() / den_all) if den_all > 0 else None
        b, _ = slope(x); _, cq = quad(x)
        b = b * al[ids]; cq = cq * al[ids]
        bl = b_lab[ids] * al[ids]; cl = cq_lab[ids] * al[ids]
        xa = x * al[ids][:, None]; ya = y * al[ids][:, None]      # 방향 정렬한 슬롯별 평균 궤적 (cm)
        return dict(n=int(len(ids)), r2_pos_within_clip=r2, r2_pos_across_clip=r2_all,
                    x_hat_mean_by_slot_cm=[float(v) for v in xa.mean(0) * hw],
                    x_hat_sd_by_slot_cm=[float(v) for v in xa.std(0) * hw],
                    x_true_mean_by_slot_cm=[float(v) for v in ya.mean(0) * hw],
                    mae_cm=float(np.abs(x - y).mean() * hw),
                    mae_cm_by_slot=[float(v) for v in np.abs(x - y).mean(0) * hw],
                    b_hat_cm_s=float(b.mean() * cm_per_unit), b_se_cm_s=float(b.std(ddof=1) / np.sqrt(len(b)) * cm_per_unit),
                    b_true_cm_s=float(bl.mean() * cm_per_unit),
                    c_hat_cm_s2=float(cq.mean() * s2), c_true_cm_s2=float(cl.mean() * s2))

    def speed_r2(ids, xh_ids):
        """(읽은 기울기, 정답 기울기) 회귀 — 정답 기울기가 두 값 이상일 때만 뜻이 있다."""
        b, _ = slope(xh_ids); bl = b_lab[ids]
        if len(np.unique(np.round(bl, 6))) < 2:
            return None, None
        A = np.vstack([bl, np.ones_like(bl)]).T
        beta, c0 = np.linalg.lstsq(A, b, rcond=None)[0]
        return float(beta), float(1 - ((b - A @ [beta, c0]) ** 2).sum() / ((b - b.mean()) ** 2).sum())

    # fit 단위. condition = "정보가 있는가"(self), from_visible = "같은 자리인가"(visible readout 을 가림에 이식),
    # motion / all 은 슬롯 번호 지름길 배제용 대조군 (운동마다 기울기가 달라야 한다)
    fit_groups = {"all": {"all": np.ones(len(vids), bool)},
                  "motion": {mo: motion == mo for mo in ("static", "flat", "ramp")},
                  "condition": {c: C["condition"] == c for c in np.unique(C["condition"])},
                  "from_visible": {mo: motion == mo for mo in ("static", "flat", "ramp")}}
    res = {}
    for fb, groups in fit_groups.items():
        res[fb] = {"h": {"groups": {}, "by_cell": {}}, "p": {"groups": {}, "by_cell": {}}}
        for g, mask in groups.items():
            g_tr = tr[mask[tr]]; g_va = va[mask[va]]
            if fb == "from_visible":                       # 학습은 그 운동의 visible 만, 평가는 그 운동 전부
                g_tr = g_tr[timing[g_tr] == "visible"]
            for s in ("h", "p"):
                w = fit_lstsq(flat(s, g_tr), Y[g_tr].ravel())
                xh = apply_w(flat(s, g_va), w).reshape(len(g_va), T_FUT)
                beta, r2v = speed_r2(g_va, xh)
                res[fb][s]["groups"][g] = dict(n_train=int(len(g_tr)), beta_vs_label=beta, r2_speed=r2v, **metrics(g_va, xh))
                for mo in ("static", "flat", "ramp"):
                    for tm in ("visible", "early", "mid", "late"):
                        for k in range(5):
                            sel = (motion[g_va] == mo) & (timing[g_va] == tm) & (kk[g_va] == k)
                            if sel.any():
                                res[fb][s]["by_cell"][f"{mo}|{tm}|{k}"] = metrics(g_va[sel], xh[sel])
        print(f"\n[fit = {fb}]")
        if fb == "condition":                                   # 같은 조건 안에서 h -> p / p -> h 이식
            for src, dst in (("h", "p"), ("p", "h")):
                key = f"xfer_{src}_to_{dst}"
                res[key] = {dst: {"groups": {}, "by_cell": {}}}
                for g, mask in groups.items():
                    g_tr = tr[mask[tr]]; g_va = va[mask[va]]
                    w = fit_lstsq(flat(src, g_tr), Y[g_tr].ravel())
                    xh = apply_w(flat(dst, g_va), w).reshape(len(g_va), T_FUT)
                    beta, r2v = speed_r2(g_va, xh)
                    res[key][dst]["groups"][g] = dict(n_train=int(len(g_tr)), beta_vs_label=beta, r2_speed=r2v, **metrics(g_va, xh))
                    for mo in ("static", "flat", "ramp"):
                        for tm in ("visible", "early", "mid", "late"):
                            for k in range(5):
                                sel = (motion[g_va] == mo) & (timing[g_va] == tm) & (kk[g_va] == k)
                                if sel.any():
                                    res[key][dst]["by_cell"][f"{mo}|{tm}|{k}"] = metrics(g_va[sel], xh[sel])
        for g in groups:
            for s in ("h", "p"):
                m_ = res[fb][s]["groups"][g]
                r2p = "  n/a " if m_["r2_pos_within_clip"] is None else f"{m_['r2_pos_within_clip']:.3f}"
                r2v = "  n/a " if m_["r2_speed"] is None else f"{m_['r2_speed']:.3f}"
                print(f"  {g:<28} {s}  train {m_['n_train']:>5}  위치R²(클립내부) {r2p}  속도R² {r2v}  MAE {m_['mae_cm']:5.1f}cm  "
                      f"b̂ {m_['b_hat_cm_s']:+7.1f}±{m_['b_se_cm_s']:.1f} (true {m_['b_true_cm_s']:+7.1f})  ĉ {m_['c_hat_cm_s2']:+6.1f} (true {m_['c_true_cm_s2']:+6.1f})")

    # ── null ──────────────────────────────────────────────────────────────
    print("\n  null 대조군 (β≈0 이어야)")
    nulls = {}
    Fp = F["p"]; rs = np.random.RandomState(123)
    Fs = Fp.copy()
    for i in range(len(Fs)):
        Fs[i] = Fs[i][rs.permutation(T_FUT)]
    Fc = Fp.copy(); Fc[:] = Fc.mean(1, keepdims=True)
    Ysh = Y[rs.permutation(len(Y))]
    for name, FF, YY in (("튜블릿 순서 셔플", Fs, Y), ("8칸 동일 (복사기)", Fc, Y), ("라벨 클립간 셔플", Fp, Ysh)):
        ww = fit_lstsq(FF[tr].reshape(-1, D).astype(np.float64), YY[tr].ravel())
        xh = apply_w(FF[va].reshape(-1, D).astype(np.float64), ww).reshape(len(va), T_FUT)
        bb, _ = slope(xh); bl, _ = slope(YY[va])
        A_ = np.vstack([bl, np.ones_like(bl)]).T
        bt, c_ = np.linalg.lstsq(A_, bb, rcond=None)[0]
        r2_ = 1 - ((bb - A_ @ [bt, c_]) ** 2).sum() / ((bb - bb.mean()) ** 2).sum()
        nulls[name] = dict(beta=float(bt), r2=float(r2_))
        print(f"    {name:20s} β={bt:+.4f}  R²={r2_:.4f}")
    del Fs, Fc

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(dict(
        meta=dict(cache=str(CACHE), index=str(INDEX), n_clip=len(vids), n_train=len(tr), n_val=len(va),
                  readout="mean-pool 256 spatial tokens + shared OLS (no standardisation) across 8 future tubelets; w per fit group (all / motion / condition)",
                  fit_by=list(fit_groups), fit_by_meaning={"condition": "self — 같은 조건 held-out (정보가 있는가)",
                      "from_visible": "운동별 visible 로 학습한 readout 을 early/mid/late 에 적용 (같은 자리인가)",
                      "motion": "운동별 풀링 (대조군)", "all": "12조건 풀링 (대조군)"}, slope_sign="b, c aligned to the true direction of travel (+); static keeps raw sign",
                  seed=a.seed, cm_per_unit_slope=cm_per_unit, label=lab,
                  trajectories={"static": "x = const (5 positions)", "flat": "x = ∓664.1 ± 260.12·t", "ramp": "x = ∓664.1 ± (115.85·t + 24.56·t²)"},
                  note="late k: 미래 첫 k 프레임도 가려져 있다. h 는 32프레임을 한 번에 보므로 보간할 수 있다"),
        results=res, nulls=nulls), indent=1, ensure_ascii=False))
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
