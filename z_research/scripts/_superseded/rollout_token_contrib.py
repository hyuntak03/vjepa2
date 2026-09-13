#!/usr/bin/env python3
"""읽힌 위치가 **실제 물체 토큰**에서 나오는가 — 선형 readout 의 토큰별 분해.

`rollout_position.py` 의 readout 은 공간평균 + 선형이라 예측이 토큰별 기여의 합으로
**정확히** 분해된다 (학습한 attention 이 아니라 항등식):

    x̂_t = (1/256) Σ_ij s_ij,t + b,      s_ij = ((f_ij − mu)/sd) · w

물체가 있는 열 j*(t) 는 라벨에서 해석적으로 안다. 그래서 두 가지를 잰다:

  1) **기여 질량** — |s| 의 몇 %가 물체 열 ±1 (3/16 = 18.75% 가 균등 기대값) 에 있나
  2) **제한 readout** — 물체 창 토큰만 / 물체에서 먼 토큰만으로 다시 예측했을 때
     기울기 β 가 살아남는가.  **이게 결정적이다**
         β_obj ≈ β_full, β_far ≈ 0   ->  물체 토큰이 나른다
         β_far ≈ β_full              ->  배경/전역이 나른다

⚠️ 이건 "프로브가 물체를 찾게 학습" 하는 게 아니다 (그 시도는 폐기했다 —
   `RollOutV1/README.md` §4 의 기록). 이미 학습된 readout 을 **사후 분해**할 뿐이다.

GPU 불필요. 캐시를 두 번 읽는다 (풀링 + 토큰별 점수).

  python z_research/scripts/analysis/rollout_token_contrib.py
"""
from __future__ import annotations
import argparse, csv, json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rollout_position as rp

OUT = rp.ROOT / "z_research/RollOutV1/exp_results/token_contribution.json"
G, S, T = rp.G, rp.S, rp.T_FUT


def token_scores(src, w, mu, sd, bs=64):
    """s_ij = ((f_ij − mu)/sd)·w  ->  (N, 8, 16, 16).  x̂ = s.mean((2,3)) + b 가 성립한다."""
    meta = json.loads((rp.CACHE / "meta.json").read_text())
    N, D = len(meta["video_ids"]), meta["embed_dim"]
    A = np.load(rp.CACHE / ("predictor.npy" if src == "p" else "target.npy"), mmap_mode="r")
    co = w[:-1] / sd                      # ((f-mu)/sd)·w = f·(w/sd) − (mu/sd)·w
    off = float((mu / sd) @ w[:-1])
    out = np.empty((N, T, G, G), np.float32)
    t0 = time.time()
    for k in range(0, N, bs):
        s = slice(k, min(k + bs, N))
        f = np.asarray(A[s] if src == "p" else A[s, T * S:]).reshape(-1, T, S, D)
        out[s] = (f.astype(np.float32) @ co.astype(np.float32) - off).reshape(-1, T, G, G)
        if k % (bs * 20) == 0:
            print(f"    {src} 점수 {k}/{N}  {time.time()-t0:.0f}s", flush=True)
    print(f"    {src} 완료  {time.time()-t0:.0f}s")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--win", type=int, default=1, help="물체 창 반폭 (열 단위)")
    ap.add_argument("--far", type=int, default=3, help="'먼' 토큰의 최소 거리")
    ap.add_argument("-o", "--out", type=Path, default=OUT)
    a = ap.parse_args()

    idx_rows = {r["video_id"]: r for r in csv.DictReader(rp.INDEX.open())}
    print("풀링 + ridge 적합 (rollout_position 재사용)")
    vids, F = rp.pool()
    rows = [idx_rows[v] for v in vids]
    Y, v_true, anchor_n, hw, fps = rp.labels_from_index(rows)
    cond = np.array([r["condition"] for r in rows])
    cm = hw / (6.0 / fps)

    rng = np.random.RandomState(0)
    tr, va = [], []
    for c in np.unique(cond):
        i = np.where(cond == c)[0]; rng.shuffle(i); h = len(i) // 2
        tr += list(i[:h]); va += list(i[h:])
    tr, va = np.array(sorted(tr)), np.array(sorted(va))

    W = {}
    for src in ("h", "p"):
        X = F[src][tr].reshape(-1, F[src].shape[-1]).astype(np.float64)
        mu, sd = X.mean(0), X.std(0) + 1e-8
        W[src] = (rp.ridge((X - mu) / sd, Y[tr].ravel(), 1.0), mu, sd)
    del F

    # 물체가 있는 열 (라벨에서 해석적으로).  j = (x+1)/2*G − 0.5
    jstar = np.clip(np.round((Y + 1) / 2 * G - 0.5), 0, G - 1).astype(int)   # (N, 8)
    cols = np.arange(G)[None, None, :]
    d = np.abs(cols - jstar[:, :, None])                                     # (N, 8, 16)
    m_obj, m_far = d <= a.win, d >= a.far

    res = {}
    for src in ("h", "p"):
        w, mu, sd = W[src]
        Sc = token_scores(src, w, mu, sd)                    # (N, 8, 16, 16)
        b = float(w[-1])
        col = Sc.mean(2)                                     # 행 평균 -> (N, 8, 16) 열 프로파일
        full = col.mean(2) + b                               # == x̂ (항등식)
        # 기여 질량 (|s| 를 열별로)
        aw = np.abs(col - col.mean(2, keepdims=True))
        mass = float((aw * m_obj).sum() / aw.sum())
        # 제한 readout
        out = {}
        for name, m in (("full", np.ones_like(m_obj)), ("obj", m_obj), ("far", m_far)):
            xh = (col * m).sum(2) / np.maximum(m.sum(2), 1) + b
            bb, _ = rp.slope(xh[va].astype(np.float64))
            bl, _ = rp.slope(Y[va])
            A_ = np.vstack([bl, np.ones_like(bl)]).T
            beta, c0 = np.linalg.lstsq(A_, bb, rcond=None)[0]
            r2 = 1 - ((bb - A_ @ [beta, c0]) ** 2).sum() / ((bb - bb.mean()) ** 2).sum()
            out[name] = dict(beta=float(beta), r2=float(r2),
                             b_hat_span_cm_s=float((bb.max() - bb.min()) * cm))
        res[src] = dict(obj_mass_frac=mass,
                        uniform_expect=float((2 * a.win + 1) / G), restricted=out)
        print(f"\n  [{src}]  물체 열 ±{a.win} 기여 질량 {mass*100:.1f}% "
              f"(균등 {100*(2*a.win+1)/G:.1f}%)")
        for n in ("full", "obj", "far"):
            o = out[n]
            print(f"    {n:5s} β={o['beta']:+.3f}  R²={o['r2']:.3f}")
        del Sc

    print("\n검증 — full β 는 rollout_position.py 의 self 칸과 같아야 한다")
    ref = {"h": 0.987, "p": 0.933}
    for src in ("h", "p"):
        got = res[src]["restricted"]["full"]["beta"]
        ok = abs(got - ref[src]) < 0.02
        print(f"  {'OK  ' if ok else '불일치'} {src}->{src}  full β={got:+.3f}  (기대 {ref[src]:+.3f})")

    print("\n판정")
    for src in ("h", "p"):
        o = res[src]["restricted"]
        ratio_o = o["obj"]["beta"] / o["full"]["beta"] if o["full"]["beta"] else float("nan")
        ratio_f = o["far"]["beta"] / o["full"]["beta"] if o["full"]["beta"] else float("nan")
        print(f"  {src}:  물체창만 {ratio_o:+.2f}×full   먼토큰만 {ratio_f:+.2f}×full")
    print("  물체창≈1, 먼토큰≈0 이면 물체 토큰이 나른다 / 둘 다 ≈1 이면 전역 분산")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(dict(
        meta=dict(win=a.win, far=a.far, n_val=len(va), cm_per_unit_slope=cm,
                  note="linear readout 의 사후 토큰 분해. 학습된 attention 이 아니다"),
        results=res), indent=1, ensure_ascii=False))
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
