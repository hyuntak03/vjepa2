#!/usr/bin/env python3
"""H5b — 되먹임 팔은 '다음 칸으로 외삽' 하는가, '되먹인 칸을 복사' 하는가 (CPU, 파라미터 0, 새 GPU 실행 없음).

왜: H5 초판 (2026-09-24) 의 적대 검증이 (i) 위치 귀무 없음 (ii) 복사 기준선 없음 (iii) 어댑터 검사가 p 가 실제로 지나는 길
(target/p 공간 → 문맥 자리) 을 안 잰다 (iv) 물체다움 문턱 하나 (p10) (v) '느린 걸음의 복리' 기제가 H5 자기 데이터와 어긋난다
를 짚었다. 여기서 h5.npy (h5_autoregress_v3.py 산출물) 만으로 다시 잰다.

입력: <inp>/h5.npy (n, 2 C, 8 j, 6 arm, 9 col), meta.json, adapter_r*.json;  index data_csv/rollout_v3/index_probe.csv.
대상: 자유 운동 6 법칙 가능 clip (2,352). 칸 = 18 px, 16×16 격자. a_j = 팔의 argmax 칸 (템플릿 = 진실 시각 h 의 물체 칸 토큰).

기호·공식 (T_j = 미래 튜블릿 j (프레임 32+2j, 33+2j) 의 진실 칸, L = 문맥 마지막 튜블릿 (프레임 30, 31) 의 칸):
  hit_j        = 1[ max|a_j − T_j| ≤ 1 ]                                  (Chebyshev ≤ 1 칸)
  null_j       = 1[ max|a_j − T_j(π)| ≤ 1 ]  π = 같은 시나리오 · 같은 j 에서 ok 인 clip 끼리 섞기, 20 번 평균  (위치 사전확률)
  oracles      stay: a_j = L · copy: a_j = T_{j−1} (T_{−1} = L) · cv: 문맥 끝 속도 v = xy(30,31) − xy(28,29) 로 등속 외삽 xy_L + (j+1)v 의 칸
  landing_j    T_j ≠ T_{j−1} 인 clip 에서 P(a_j = T_j) vs P(a_j = T_{j−1})  (정확히 같은 칸)
  self-copy_j  P(a_j = a_{j−1})  (ar_p 에서 a_{j−1} 은 그 자신이 되먹인 토큰의 물체 칸)  vs 진실 P(T_j = T_{j−1})
  along_j      (a_j − L)·u,  u = (xy_{T7} − xy_L)/|·|  (칸 단위; 진실 경로 현 방향 성분)   — 모든 j 에서 ok, |xy_{T7} − xy_L| ≥ 1 칸 인 clip
  slope        clip 마다 j = 3..7 에서 along_j 의 최소제곱 기울기 (칸/튜블릿), clip 평균
  progress_j   ((a_j − L)·d)/|d|²,  d = T_j − L, |d| ≥ 3 칸 (H4 와 같은 식) — 표본 3 종: 팔별 물체다움 p10 (초판) · 공통 (물체다움 거르지 않음) · 공통 물체다움
  objlike_q    contrast (max − median cos) ≥ os 팔 j ≤ 1 contrast 의 q 백분위, q ∈ {10, 25, 50}
  속도 3 분위  문맥 끝 속도 |v| (px/튜블릿) — 멈춤이 '같은 거리' 에서 오나 (거리 한계) '같은 j' 에서 오나 (되먹임 걸음 수) 를 가른다
  CI           clip 단위 bootstrap 1000 번 (seed 0), 95 % 백분위 구간. 중앙값은 300 번.

  auto_research/scripts/srun6.sh 4 32G /data/hyuntak/anaconda3/envs/vjepa2/bin/python auto_research/scripts/h5b_copy_vs_extrapolate.py
"""
from __future__ import annotations
import argparse, glob, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from h23_analyze_v3 import load_index, tub_xy, cell, FREE, CELL  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
IN_DEFAULT = "/data2/local_datasets/world/world_analysis/cache/auto_research/v3_h5"
OUT = ROOT / "auto_research/exp_results/h5"
NB, NBM, NPERM, J = 1000, 300, 20, 8
SHOW = ("os", "single", "ar_p", "ar_z", "ar_h")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default=IN_DEFAULT)
    a_ = ap.parse_args()
    I = Path(a_.inp)
    meta = json.load(open(I / "meta.json")); ids = meta["video_ids"]; arms = meta["arms"]
    H = np.load(I / "h5.npy")
    R, X, Y, OK = load_index(ids)
    scen = np.array([r["scenario"] for r in R]); pl = np.array([r["plausible"] == "1" for r in R])
    free = np.isin(scen, FREE) & pl
    H, X, Y, OK, scen = H[free], X[free], Y[free], OK[free], scen[free]
    n = len(scen)
    rng = np.random.default_rng(0)
    BI = rng.integers(0, n, size=(NB, n))

    def bmean(x, m):
        x = np.where(m, np.nan_to_num(x.astype(float)), 0.0); w = m.astype(float)
        est = x.sum() / max(w.sum(), 1); bs = x[BI].sum(1) / np.maximum(w[BI].sum(1), 1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        return dict(v=round(float(est), 4), lo=round(float(lo), 4), hi=round(float(hi), 4), n=int(w.sum()))

    def bmed(x, m):
        if m.sum() < 20:
            return dict(v=None, lo=None, hi=None, n=int(m.sum()))
        bs = [np.median(x[BI[b]][m[BI[b]]]) for b in range(NBM)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        return dict(v=round(float(np.median(x[m])), 3), lo=round(float(lo), 3), hi=round(float(hi), 3), n=int(m.sum()))

    fr = lambda j: 30 if j < 0 else 32 + 2 * j
    Txy = {j: tub_xy(X, Y, fr(j)) for j in range(-1, J)}
    T = {j: cell(Txy[j]).astype(float) for j in range(-1, J)}
    OKj = {j: OK[:, fr(j)] & OK[:, fr(j) + 1] for j in range(-1, J)}
    L, Lxy = T[-1], Txy[-1]
    v = Lxy - tub_xy(X, Y, 28)
    speed = np.linalg.norm(v, axis=1)
    orc = {"stay": {j: L for j in range(J)}, "copy": {j: T[j - 1] for j in range(J)},
           "cv": {j: cell(Lxy + (j + 1) * v).astype(float) for j in range(J)}}
    chord = Txy[7] - Lxy; cl = np.linalg.norm(chord, axis=1); u = chord / np.maximum(cl, 1e-6)[:, None]
    okall = np.all([OKj[j] for j in range(-1, J)], 0) & OK[:, 28] & OK[:, 29] & (cl >= CELL)
    tert = np.quantile(speed[okall], [1 / 3, 2 / 3]); sbin = np.digitize(speed, tert)
    perms = []
    for j in range(J):
        pj = []
        for _ in range(NPERM):
            p = np.arange(n)
            for s in FREE:
                ix = np.where((scen == s) & OKj[j])[0]; p[ix] = rng.permutation(ix)
            pj.append(p)
        perms.append(pj)
    along = lambda a: ((a - L) * u).sum(1)
    res = dict(n_free=n, n_okall=int(okall.sum()), speed_tertiles_px=np.round(tert, 2).tolist(), C={})
    ad = [json.load(open(f))["adapter_rel_l1_mean"] for f in sorted(glob.glob(str(I / "adapter_r*.json")))]
    res["adapter_recon_rel_l1_first_fed_tubelet"] = dict(per_rank=[round(x, 5) for x in ad], mean=round(float(np.mean(ad)), 5) if ad else None)
    lines = [f"n_free {n}  okall {okall.sum()}  speed tertiles px/tub {tert.round(2)}",
             f"adapter recon rel L1 |A(LN(z*_0))-z*_0|/|z*_0| per rank {np.round(ad, 4)} mean {np.mean(ad):.4f}" if ad else "adapter json 없음"]
    for ci, C in enumerate(meta["Cs"]):
        con = H[:, ci, :, :, 2] - H[:, ci, :, :, 3]
        A = {arm: {j: H[:, ci, j, arms.index(arm), :2][:, ::-1] for j in range(J)} for arm in arms}
        refs = {q: float(np.percentile(con[:, :2, arms.index("os")], q)) for q in (10, 25, 50)}
        rc = dict(obj_ref=refs); lines.append(f"\n==== C{C}  objectness refs " + " ".join(f"p{q} {r:.3f}" for q, r in refs.items()))
        # --- hit / null / oracles
        tab = {}
        for arm in list(SHOW) + ["stay", "copy", "cv"]:
            src = A[arm] if arm in A else orc[arm]; row = dict(hit=[], null=[], hmn=[])
            for j in range(J):
                a = src[j]; ok = OKj[j]
                h = (np.abs(a - T[j]).max(1) <= 1).astype(float)
                nl = np.mean([(np.abs(a - T[j][p]).max(1) <= 1) for p in perms[j]], 0)
                row["hit"].append(bmean(h, ok)); row["null"].append(bmean(nl, ok)); row["hmn"].append(bmean(h - nl, ok))
            tab[arm] = row
        rc["hit_null"] = tab
        lines.append("-- hit / null  (j 0..7)   [hit−null 95% CI at j7]")
        for arm, row in tab.items():
            lines.append(f"  {arm:6s} " + " ".join(f"{row['hit'][j]['v']:.2f}/{row['null'][j]['v']:.2f}" for j in range(J))
                         + f"   j7 hit−null {row['hmn'][7]['v']:+.3f} [{row['hmn'][7]['lo']:+.3f},{row['hmn'][7]['hi']:+.3f}] n{row['hmn'][7]['n']}")
        # --- landing test
        land = {}
        lines.append("-- landing: T_j≠T_{j−1} 일 때 P(a_j=T_j) / P(a_j=T_{j−1})  (j 1..7)")
        for arm in SHOW:
            row = dict(onT=[], onPrev=[], n=[])
            for j in range(1, J):
                ch = OKj[j] & OKj[j - 1] & (np.abs(T[j] - T[j - 1]).max(1) >= 1)
                a = A[arm][j]
                row["onT"].append(bmean((a == T[j]).all(1).astype(float), ch)); row["onPrev"].append(bmean((a == T[j - 1]).all(1).astype(float), ch))
                row["n"].append(int(ch.sum()))
            land[arm] = row
            lines.append(f"  {arm:6s} " + " ".join(f"j{j}:{row['onT'][j-1]['v']:.2f}/{row['onPrev'][j-1]['v']:.2f}" for j in range(1, J))
                         + f"   (n j1..7 {row['n']})")
        rc["landing"] = land
        # --- self-copy
        sc = {}
        lines.append("-- self-copy P(a_j = a_{j−1}) (j 1..7); truth P(T_j=T_{j−1}); ar_p obj-both = 두 걸음 모두 물체다움 p10")
        tr = [bmean((T[j] == T[j - 1]).all(1).astype(float), OKj[j] & OKj[j - 1]) for j in range(1, J)]
        sc["truth"] = tr; lines.append("  truth  " + " ".join(f"{x['v']:.2f}" for x in tr))
        for arm in SHOW:
            r_ = []; r_obj = []
            for j in range(1, J):
                ok = OKj[j] & OKj[j - 1]; same = (A[arm][j] == A[arm][j - 1]).all(1).astype(float)
                r_.append(bmean(same, ok))
                ob = ok & (con[:, j, arms.index(arm)] >= refs[10]) & (con[:, j - 1, arms.index(arm)] >= refs[10])
                r_obj.append(bmean(same, ob))
            sc[arm] = dict(all=r_, objboth=r_obj)
            lines.append(f"  {arm:6s} " + " ".join(f"{x['v']:.2f}" for x in r_) + "  | obj-both " + " ".join(f"{x['v']:.2f}" for x in r_obj))
        rc["self_copy"] = sc
        # --- along-track, per-step, slope
        al = {}; lines.append(f"-- along-track mean (칸, okall n{okall.sum()}) j 0..7  [per-step j3..7]  slope j3..7 (칸/튜블릿, 95% CI)")
        srcs = dict(truth={j: T[j] for j in range(J)}, **{k: A[k] for k in SHOW}, **orc)
        jj = np.arange(3, 8) - 5.0
        for nm, src in srcs.items():
            ys = np.stack([along(src[j]) for j in range(J)], 1)
            slope = (ys[:, 3:8] * jj).sum(1) / (jj ** 2).sum()
            al[nm] = dict(mean=[bmean(ys[:, j], okall) for j in range(J)], slope=bmean(slope, okall),
                          step=[round(float((ys[okall, j] - ys[okall, j - 1]).mean()), 3) for j in range(1, J)])
            lines.append(f"  {nm:6s} " + " ".join(f"{x['v']:.2f}" for x in al[nm]["mean"]) + "  step " + " ".join(f"{x:+.2f}" for x in al[nm]["step"])
                         + f"  slope {al[nm]['slope']['v']:+.3f} [{al[nm]['slope']['lo']:+.3f},{al[nm]['slope']['hi']:+.3f}]")
        rc["along"] = al
        # --- step direction: 한 걸음 along 변화가 앞 (> 0.5 칸) / 제자리 (|·| ≤ 0.5) /뒤 (< −0.5)
        sd = {}; lines.append("-- 걸음 방향 앞/제자리/뒤 (j 1..7; 진실은 뒤 0)")
        for nm in ("truth",) + SHOW:
            src = srcs[nm]; r_ = []
            for j in range(1, J):
                ok = OKj[j] & OKj[j - 1]; stp = ((src[j] - src[j - 1]) * u).sum(1)
                r_.append(dict(fwd=round(float((stp[ok] > 0.5).mean()), 3), stay=round(float((np.abs(stp[ok]) <= 0.5).mean()), 3),
                               back=round(float((stp[ok] < -0.5).mean()), 3)))
            sd[nm] = r_
            lines.append(f"  {nm:6s} " + " ".join(f"{x['fwd']:.2f}/{x['stay']:.2f}/{x['back']:.2f}" for x in r_))
        rc["step_direction"] = sd
        # --- speed tertiles
        st = {}; lines.append("-- 속도 3 분위별 along-track mean (칸) j 0..7   (느림 / 중간 / 빠름);  plateau = clip 별 j3..7 평균 [CI];  p25 = 그 j 에서 물체다움 p25 인 clip 만")
        for b in range(3):
            m = okall & (sbin == b); st[b] = dict(n=int(m.sum()), speed_px=round(float(speed[m].mean()), 2))
            for nm in ("truth", "os", "single", "ar_p", "ar_h", "ar_z"):
                ys = np.stack([along(srcs[nm][j]) for j in range(J)], 1)
                st[b][nm] = [round(float(ys[m, j].mean()), 3) for j in range(J)]
                pl_ = bmean(ys[:, 3:8].mean(1), m); st[b][nm + "_plateau_j3to7"] = pl_
                extra = ""
                if nm in A:
                    c_ = con[:, :, arms.index(nm)]
                    st[b][nm + "_p25"] = [dict(v=round(float(ys[m & (c_[:, j] >= refs[25]), j].mean()), 3) if (m & (c_[:, j] >= refs[25])).sum() >= 20 else None,
                                               n=int((m & (c_[:, j] >= refs[25])).sum())) for j in range(J)]
                    extra = " | p25 " + " ".join("  – " if x["v"] is None else f"{x['v']:.2f}" for x in st[b][nm + "_p25"])
                lines.append(f"  b{b} n{m.sum()} |v|{speed[m].mean():5.1f}px {nm:6s} " + " ".join(f"{x:.2f}" for x in st[b][nm])
                             + f"  plateau {pl_['v']:.2f} [{pl_['lo']:.2f},{pl_['hi']:.2f}]" + extra)
        rc["speed_tertiles"] = st
        # --- progress
        pg = {}; lines.append("-- progress median (j 4..7): 팔별 p10 (초판) | 공통 (거르지 않음) [CI] | 공통 물체다움 p10 | 팔별 p25")
        for j in range(4, J):
            d = T[j] - L; dd = (d ** 2).sum(1); mv = OKj[j] & (np.sqrt(dd) >= 3)
            objall = np.all([con[:, j, arms.index(x)] >= refs[10] for x in SHOW], 0)
            pj = {}
            for nm in list(SHOW) + ["stay", "copy", "cv"]:
                a = A[nm][j] if nm in A else orc[nm][j]
                pr = ((a - L) * d).sum(1) / np.maximum(dd, 1)
                e = dict(shared=bmed(pr, mv), common_obj=bmed(pr, mv & objall))
                if nm in A:
                    c_ = con[:, j, arms.index(nm)]
                    e["arm_p10"] = bmed(pr, mv & (c_ >= refs[10])); e["arm_p25"] = bmed(pr, mv & (c_ >= refs[25]))
                pj[nm] = e
            pg[j] = pj
            f = lambda e, k: "  – " if e.get(k) is None or e[k]["v"] is None else f"{e[k]['v']:.2f}"
            lines.append(f"  j{j} n_shared {mv.sum()} n_common {(mv & objall).sum()}  " + "; ".join(
                f"{nm} {f(e, 'arm_p10') if 'arm_p10' in e else '  – '}|{e['shared']['v']:.2f}[{e['shared']['lo']:.2f},{e['shared']['hi']:.2f}]|{f(e, 'common_obj')}|{f(e, 'arm_p25') if 'arm_p25' in e else '  – '}"
                for nm, e in pj.items()))
        rc["progress"] = pg
        # --- objectness thresholds
        ob = {}; lines.append("-- 물체다움 (j 0..7) at p10 / p25 / p50 ;  contrast 중앙값")
        for nm in SHOW:
            ob[nm] = {}
            for q in (10, 25, 50):
                ob[nm][f"p{q}"] = [bmean((con[:, j, arms.index(nm)] >= refs[q]).astype(float), OKj[j]) for j in range(J)]
            ob[nm]["median_contrast"] = [round(float(np.median(con[OKj[j], j, arms.index(nm)])), 3) for j in range(J)]
            lines.append(f"  {nm:6s} " + " | ".join(f"p{q} " + " ".join(f"{x['v']:.2f}" for x in ob[nm][f'p{q}']) for q in (10, 25, 50))
                         + " | med " + " ".join(f"{x:.3f}" for x in ob[nm]["median_contrast"]))
        rc["objlike"] = ob
        # --- per scenario j7
        ps = {}; lines.append("-- 시나리오별 j7: hit/null · along (칸) · 진실 along")
        for s in FREE:
            m7 = OKj[7] & (scen == s); ma = okall & (scen == s); ps[s] = dict(n=int(m7.sum()), n_along=int(ma.sum()))
            ps[s]["truth_along"] = round(float(along(T[7])[ma].mean()), 2)
            for nm in ("os", "ar_p", "ar_z", "ar_h"):
                a = A[nm][7]; h = (np.abs(a - T[7]).max(1) <= 1)
                nl = np.mean([(np.abs(a - T[7][p]).max(1) <= 1) for p in perms[7]], 0)
                ps[s][nm] = dict(hit=round(float(h[m7].mean()), 3), null=round(float(nl[m7].mean()), 3), along=round(float(along(a)[ma].mean()), 2))
            lines.append(f"  {s:7s} n{m7.sum()} truth {ps[s]['truth_along']:.2f}  " + "  ".join(
                f"{nm} {ps[s][nm]['hit']:.2f}/{ps[s][nm]['null']:.2f} al {ps[s][nm]['along']:.2f}" for nm in ("os", "ar_p", "ar_z", "ar_h")))
        rc["per_scenario_j7"] = ps
        # --- adapter arm difference, L1
        iz, iza = arms.index("ar_z"), arms.index("ar_zA")
        dz = np.abs(H[:, ci, :, iz] - H[:, ci, :, iza])
        rc["ar_z_vs_ar_zA"] = dict(argmax_differs=round(float(np.mean(np.any(dz[..., :2] > 0, -1))), 5),
                                   rel_l1_diff=round(float(dz[..., 8].mean() / H[:, ci, :, iz, 8].mean()), 6))
        lines.append(f"-- ar_z vs ar_zA: argmax 칸 다름 {rc['ar_z_vs_ar_zA']['argmax_differs']:.4f}, 채점 L1 상대차 {rc['ar_z_vs_ar_zA']['rel_l1_diff']:.6f}")
        rc["l1"] = {nm: [round(float(H[:, ci, j, arms.index(nm), 8].mean()), 4) for j in range(J)] for nm in SHOW}
        for nm in SHOW:
            lines.append(f"   L1 {nm:6s} " + " ".join(f"{x:.3f}" for x in rc["l1"][nm]))
        res["C"][f"C{C}"] = rc
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / "h5b_copy_vs_extrapolate.json", "w"), indent=1, ensure_ascii=False)
    (OUT / "h5b_summary.txt").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
