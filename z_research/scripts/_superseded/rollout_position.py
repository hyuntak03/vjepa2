#!/usr/bin/env python3
"""RollOut_v1 — 미래 토큰에서 물체 위치를 튜블릿별로 읽고, 그 기울기가 속도인지 본다.

`PAPER_STORY_2026-09-06.md` beat 4(a) 의 측정이다.

    x̂(t) = a + b·t   (t = 0..7 튜블릿)
      b̂ ≈ v  ->  predictor 가 상태를 앞으로 나른다
      b̂ ≈ 0  ->  조회기. 마지막 관측 위치를 8칸에 들고 있을 뿐이다

읽는 방법 — **튜블릿 안 256 토큰 평균 -> 최소제곱**. 닫힌 해라 epoch·lr·수렴이 없다.
표준화는 하지 않는다 (OLS 는 스케일 불변이고, 이식에서는 오히려 교란이 된다 — `fit_lstsq`).

⚠️ **프로브는 8칸 전부에 같은 가중치를 쓴다.** 이게 이 측정의 유일한 방어선이다 —
   `p_t` 가 t 에 따라 안 변하면 (문맥을 복사해 8칸에 넣었으면) 공유 프로브는 모든 칸에
   같은 값을 내고 **b̂ = 0** 이 된다. 칸별로 따로 학습하면 이 성질이 깨진다.

⚠️ **GT 속도는 프로브 입력에 안 들어간다.** 라벨을 만들 때와 마지막에 b̂ 와 비교할 때만 쓴다.
   `p` 에서 속도를 직접 회귀하면 안 된다 — v 는 문맥에 이미 있고 p = predictor(context) 라
   **문맥을 복사만 해도 통과**한다 (`SYNTHESIS_2026-08-30.md` §3-4 의 문맥 잔상 함정).

두 칸만 낸다
    h->h  도구 대조군 (천장). 이게 안 나오면 도구가 무력하다
    p->p  **핵심**. p 자체 basis 에서 진전이 읽히는가
⚠️ 이식(h->p / p->h)은 "같은 좌표계를 쓰는가" 라는 **다른 질문**이라 여기서 내지 않는다.

대조군
    v=0 셀        b̂ ≈ 0 이어야    프로브가 무조건 진전을 만들어내는 것을 배제
    intercept â   anchor 를 따라가야  <- â 는 맞는데 b̂≈0 이면 **조회기 확정**
    부호          h 에서 β<0 이면 이미지 x 축과 world x 축이 반대. 자동 감지해 보고한다

GPU 불필요 (numpy 만). 비용은 캐시 72 GiB 를 한 번 읽어 풀링하는 I/O 다.

  python z_research/scripts/analysis/rollout_position.py
  python z_research/scripts/analysis/rollout_position.py --max-clips 1320   # 빠른 확인
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
CACHE = Path("/local_datasets/world/world_analysis/cache/rollout_v1_vith")
INDEX = ROOT / "data_csv/rollout_v1/index_probe.csv"
OUT = ROOT / "z_research/RollOutV1/exp_results/position_regression.json"

G, S, T_FUT, T_ALL = 16, 256, 8, 16


def die(m):
    print(f"\nERROR: {m}\n", file=sys.stderr)
    sys.exit(1)


def labels_from_index(rows):
    """튜블릿 t 의 정규화 x. 렌더 메타데이터만 쓰고 픽셀 측정이 없다.

    x_cm(f) = anchor + v·(f-45)/fps,  튜블릿 t 는 raw {48+6t, 51+6t}
    두 프레임 평균이므로 (f-45) 의 평균이 4.5+6t
    """
    v = np.array([float(r["flat_v_cm_s"]) for r in rows])
    a = np.array([float(r["flat_anchor_x_cm"]) for r in rows])
    hw, fps = float(rows[0]["frame_half_width_cm"]), float(rows[0]["fps"])
    dt = (4.5 + 6.0 * np.arange(T_FUT)) / fps
    return ((a[:, None] + v[:, None] * dt[None, :]) / hw), v, a / hw, hw, fps


def pool(bs=64):
    """캐시를 한 번 훑어 (N, 8, 1280) 로 공간평균. h 는 target 의 **미래 절반**."""
    meta = json.loads((CACHE / "meta.json").read_text())
    vids, bc = meta["video_ids"], meta["base_counts"]
    for b, n in (("predictor", T_FUT * S), ("target", T_ALL * S)):
        if bc.get(b) != n:
            die(f"캐시 base '{b}' 토큰 수 {bc.get(b)} != {n}")
    P = np.load(CACHE / "predictor.npy", mmap_mode="r")
    H = np.load(CACHE / "target.npy", mmap_mode="r")
    N, D = len(vids), meta["embed_dim"]
    out = {"p": np.empty((N, T_FUT, D), np.float32), "h": np.empty((N, T_FUT, D), np.float32)}
    t0 = time.time()
    for k in range(0, N, bs):
        s = slice(k, min(k + bs, N))
        out["p"][s] = np.asarray(P[s]).reshape(-1, T_FUT, S, D).mean(2)
        out["h"][s] = np.asarray(H[s, T_FUT * S:]).reshape(-1, T_FUT, S, D).mean(2)
        if k % (bs * 20) == 0:
            print(f"    풀링 {k}/{N}  {time.time()-t0:.0f}s", flush=True)
    print(f"    풀링 완료 {N}개  {time.time()-t0:.0f}s")
    return vids, out


def fit_lstsq(X, y):
    """표준화 없는 순수 최소제곱. bias 열만 붙인다.

    ⚠️ 표준화를 쓰지 않는 이유:
      - OLS 는 특징 스케일에 **불변**이다 (가역 아핀 재매개화라 예측값이 동일).
      - 반대로 이식(h->p)에서는 **해롭다** — 학습 소스의 mu/sd 로 다른 소스를
        표준화하면 스케일 불일치가 그것만으로 readout 을 깨서, "좌표가 다르다" 와
        "스케일이 다르다" 가 섞인다.
    SVD 기반 lstsq 라 조건수가 나빠도 안전하다 (작은 특이값은 rcond 로 잘린다).
    """
    Xb = np.hstack([X, np.ones((len(X), 1), X.dtype)])
    return np.linalg.lstsq(Xb, y, rcond=None)[0]


def apply_w(X, w):
    return X @ w[:-1] + w[-1]


def slope(x):
    """x: (N, 8) -> 칸 인덱스에 대한 최소제곱 기울기와 절편(t=0 에서의 값)."""
    t = np.arange(T_FUT, dtype=np.float64)
    tc = t - t.mean()
    b = (x * tc).sum(1) / (tc ** 2).sum()
    return b, x.mean(1) - b * t.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-clips", type=int, default=0, help="0 = 전수")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--examples", type=float, nargs="*", default=[-135., -60., 0., 110.],
                    help="칸별 예측을 그대로 찍어볼 속도들")
    ap.add_argument("-o", "--out", type=Path, default=OUT)
    a = ap.parse_args()

    if not (CACHE / "meta.json").is_file():
        die(f"캐시가 없다 -> {CACHE}")
    idx_rows = {r["video_id"]: r for r in csv.DictReader(INDEX.open())}
    print("풀링 (캐시 72 GiB 한 번 읽기)")
    vids, F = pool()
    if set(vids) - set(idx_rows):
        die("캐시 video_ids 가 index 에 없다 (다른 세트의 캐시인가?)")
    rows = [idx_rows[v] for v in vids]

    Y, v_true, anchor_n, hw, fps = labels_from_index(rows)
    cond = np.array([r["condition"] for r in rows])
    cm_per_unit = hw / (6.0 / fps)                     # 칸당 정규화x -> cm/s

    # split — (speed, anchor) 33셀 층화 50/50. --max-clips 는 셀 안에서 자른다
    rng = np.random.RandomState(a.seed)
    conds = np.unique(cond)
    cap = max(2, a.max_clips // (2 * len(conds))) if a.max_clips else None
    tr, va = [], []
    for c in conds:
        i = np.where(cond == c)[0]; rng.shuffle(i)
        h = len(i) // 2
        t_i, v_i = i[:h], i[h:]
        if cap:
            t_i, v_i = t_i[:cap], v_i[:cap]
        tr += list(t_i); va += list(v_i)
    tr, va = np.array(sorted(tr)), np.array(sorted(va))
    nv = len(np.unique(v_true[va]))
    print(f"split  train {len(tr)} / val {len(va)}  (33셀 층화, val 속도 {nv}/11 레벨)")
    if nv < 11:
        die("val 에 속도 레벨이 다 없다 — --max-clips 를 키울 것")

    def flat(src, ids):
        return F[src][ids].reshape(-1, F[src].shape[-1]).astype(np.float64), Y[ids].ravel()

    res, W = {}, {}
    for src in ("h", "p"):
        Xt, yt = flat(src, tr)
        W[src] = fit_lstsq(Xt, yt)
        print(f"  least squares on {src}:  {Xt.shape[0]} 식 x {Xt.shape[1]+1} 미지수")

    print()
    for trs in ("h", "p"):
        w = W[trs]
        for evs in (trs,):                 # self 만 — 이식(h->p, p->h)은 다른 질문이라 안 낸다
            Xe, _ = flat(evs, va)
            xh = apply_w(Xe, w).reshape(len(va), T_FUT)
            b, a0 = slope(xh)
            b_lab, _ = slope(Y[va])
            A = np.vstack([b_lab, np.ones_like(b_lab)]).T
            beta, c0 = np.linalg.lstsq(A, b, rcond=None)[0]
            r2 = 1 - ((b - A @ [beta, c0]) ** 2).sum() / ((b - b.mean()) ** 2).sum()
            per_v = {}
            for vv in np.unique(v_true[va]):
                m = v_true[va] == vv
                per_v[float(vv)] = dict(
                    b_hat_cm_s=float(b[m].mean() * cm_per_unit),
                    se=float(b[m].std(ddof=1) / np.sqrt(m.sum()) * cm_per_unit),
                    n=int(m.sum()))
            yc = Y[va] - Y[va].mean(1, keepdims=True)
            xc = xh - xh.mean(1, keepdims=True)
            r2_in = 1 - ((yc - xc) ** 2).sum() / (yc ** 2).sum()
            key = f"{trs}->{evs}"
            res[key] = dict(beta_vs_label=float(beta), r2_speed=float(r2),
                            r2_pos_within_clip=float(r2_in),
                            mae_x=float(np.abs(xh - Y[va]).mean()),
                            intercept_vs_anchor_r=float(np.corrcoef(a0, anchor_n[va])[0, 1]),
                            b_hat_at_v0_cm_s=per_v.get(0.0, {}).get("b_hat_cm_s"),
                            per_speed=per_v)
            b0 = res[key]["b_hat_at_v0_cm_s"]
            print(f"  {key:8s} 위치R²(클립내부)={r2_in:.4f}  속도R²={r2:.4f}  β={beta:+.3f}  "
                  f"위치MAE={res[key]['mae_x']*hw:.1f}cm  "
                  f"b̂(v=0)={'%+.1f' % b0 if b0 is not None else 'n/a'} cm/s")

    print("\n  속도별 b̂ (cm/s) — 완벽하면 v_true 와 같다")
    print(f"  {'v_true':>7} " + " ".join(f"{k:>10}" for k in res))
    for s in sorted(res["h->h"]["per_speed"], key=float):
        print(f"  {float(s):>+7.0f} " +
              " ".join(f"{res[k]['per_speed'][s]['b_hat_cm_s']:>+10.1f}" for k in res))

    # ── 예시: 8칸의 예측 위치를 정답과 나란히 (p->p) ─────────────────────────
    if a.examples:
        print("\n  예시 — p 의 8칸에서 읽은 위치 vs 정답 (held-out, w 는 얼린 것)")
        xh_pp = apply_w(flat("p", va)[0], W["p"]).reshape(len(va), T_FUT)
        for tv in a.examples:
            k = np.where(v_true[va] == tv)[0]
            if not len(k):
                continue
            k = k[0]
            gt, pr = Y[va][k], xh_pp[k]
            print(f"    v={tv:+7.0f} cm/s  anchor={anchor_n[va][k]*hw:+7.1f} cm")
            print("      t      " + " ".join(f"{t:>8d}" for t in range(T_FUT)))
            print("      정답   " + " ".join(f"{x:>8.3f}" for x in gt))
            print("      읽음   " + " ".join(f"{x:>8.3f}" for x in pr))
            print("      오차   " + " ".join(f"{x:>+8.3f}" for x in pr - gt)
                  + f"   |평균| {np.abs(pr-gt).mean():.3f} (격자 {np.abs(pr-gt).mean()*8:.2f}칸)")

    # ── null 대조군 — 우연히 나온 값이 아님을 보인다 ──────────────────────────
    print("\n  null 대조군 (전부 β≈0 이어야 한다)")
    nulls = {}
    Fp = F["p"]
    # (1) 튜블릿 순서 셔플 — 클립마다 8칸을 무작위 치환, 라벨은 그대로
    rs = np.random.RandomState(123)
    Fs = Fp.copy()
    for i in range(len(Fs)):
        Fs[i] = Fs[i][rs.permutation(T_FUT)]
    # (2) 8칸을 전부 같게 — 복사기 모사
    Fc = Fp.copy(); Fc[:] = Fc.mean(1, keepdims=True)
    # (3) 라벨을 클립 간에 섞음 — 특징과 정답의 대응을 끊는다
    Ysh = Y[rs.permutation(len(Y))]
    for name, FF, YY in (("튜블릿 순서 셔플", Fs, Y),
                         ("8칸 동일 (복사기)", Fc, Y),
                         ("라벨 클립간 셔플", Fp, Ysh)):
        Xt = FF[tr].reshape(-1, FF.shape[-1]).astype(np.float64)
        ww = fit_lstsq(Xt, YY[tr].ravel())
        xh = apply_w(FF[va].reshape(-1, FF.shape[-1]).astype(np.float64), ww).reshape(len(va), T_FUT)
        bb, _ = slope(xh)
        bl, _ = slope(YY[va])
        A_ = np.vstack([bl, np.ones_like(bl)]).T
        bt, c_ = np.linalg.lstsq(A_, bb, rcond=None)[0]
        r2_ = 1 - ((bb - A_ @ [bt, c_]) ** 2).sum() / ((bb - bb.mean()) ** 2).sum()
        nulls[name] = dict(beta=float(bt), r2=float(r2_))
        print(f"    {name:20s} β={bt:+.4f}  R²={r2_:.4f}")
    del Fs, Fc

    print("\n판정:")
    ctrl, ok = res["h->h"], True

    def chk(n, c, d=""):
        nonlocal ok
        print(f"  {'OK  ' if c else '실패'} {n}" + (f"   {d}" if d else "")); ok = ok and c

    chk("① 도구 대조군 h->h (위치R²>0.8, 속도R²>0.8)",
        ctrl["r2_pos_within_clip"] > 0.8 and ctrl["r2_speed"] > 0.8,
        f"위치 {ctrl['r2_pos_within_clip']:.3f} / 속도 {ctrl['r2_speed']:.3f}")
    if ctrl["beta_vs_label"] < 0:
        print("     ⚠️ β<0 — 이미지 x 축과 world x 축이 반대. 부호 규약만 뒤집으면 된다")
    chk("대조군 v=0 에서 h->h 기울기 ≈ 0 (|b̂| < 15 cm/s)",
        abs(ctrl["b_hat_at_v0_cm_s"] or 0) < 15, f"{ctrl['b_hat_at_v0_cm_s']:+.1f} cm/s")
    if not ok:
        print("\n  ⚠️ 대조군이 깨졌다. p 쪽 수치를 해석하지 말 것")

    core = res["p->p"]
    print(f"\n핵심 p->p :  위치R²(클립내부) {core['r2_pos_within_clip']:.4f}  "
          f"속도R² {core['r2_speed']:.4f}")
    print(f"  천장(h->h) 대비:  위치 {core['r2_pos_within_clip']/ctrl['r2_pos_within_clip']:.2f}  "
          f"속도 {core['r2_speed']/ctrl['r2_speed']:.2f}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(dict(
        meta=dict(cache=str(CACHE), n_clip=len(vids), n_train=len(tr), n_val=len(va),
                  readout="mean-pool over 256 spatial tokens + shared OLS (no standardisation) across 8 tubelets",
                  seed=a.seed, cm_per_unit_slope=cm_per_unit, controls_ok=bool(ok)),
        results=res, nulls=nulls), indent=1, ensure_ascii=False))
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
