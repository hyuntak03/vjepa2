#!/usr/bin/env python3
"""G19 · G20 · G21 요약 (진행자 1, 2026-10-22, CPU) — carry_coef 산출물에서만 읽는다. 판정 규칙 = EVIDENCE §3-2a (결과 전에 적음).

    python paper_lab/scripts/g19_summary.py            # markdown 표 + paper_lab/_scratch/carry/g19_summary.json
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import carry_coef as CC  # noqa: E402

OUT = CC.OUT
ENCS = (("", "V-JEPA 2 연속 2 장"), ("_dup", "V-JEPA 2 한 장 복제"), ("_dinov2", "DINOv2 한 장"), ("_dinov2avg", "DINOv2 두 장 평균 (보조)"))
NAT_TRAINED = ("vast70", "ariel_ar40", "pilotA0_e10", "pilotAX_e10", "pilotA2c_e10", "pilotA4_e10", "shortctx_c8", "shortctx_c2")
TWO_D = ("state_mix", "state_nosup", "state_w005", "state_sup_v1")
B = 1000


def load(setname, name):
    p = OUT / setname / f"{name}.npz"
    return CC._load_stats(setname, name) if p.exists() else (None, None, None)


def kappa(st, key, seed=0):
    """(c 점추정, 영상 bootstrap 표본 (B,), 쓴 영상 수)."""
    if st is None or key not in st:
        return None
    x = st[key]
    _, c, _ = CC.solve(x.sum(0))
    idx = np.random.default_rng(seed).integers(0, len(x), size=(B, len(x)))
    return float(c), CC.solve(x[idx].sum(1))[1], int((x[:, 6] > 0).sum())


def fmt(k):
    if k is None:
        return "—"
    lo, hi = np.nanpercentile(k[1], [2.5, 97.5])
    return f"{k[0]:+.3f} [{lo:+.3f}, {hi:+.3f}] (n {k[2]})"


def ratio(kn, kr):
    if kn is None or kr is None:
        return None
    r = kn[1] / kr[1]
    return float(kn[0] / kr[0]), np.nanpercentile(r, [2.5, 97.5])


def main():
    res = {"kappa": {}, "tool": {}, "ratio": {}, "law": {}, "encprops": None, "velprobe": None}
    lines = ["**κ (y = 실제 다음 블록) · 도구 검사 (y = 복사 → |c| ≤ 0.10, y = 오라클 · (b) 블록 → c ≥ 0.8)**", "",
             "| encoder | 판 | 운반 연산 | 칸 | κ [95 %] | c(복사) | c(오라클) | c((b) 블록) |", "|---|---|---|---|---|---|---|---|"]
    for tag, lab in ENCS:
        for setname in ("natural", "vobench", "phys2d"):
            st, _, _ = load(setname, f"_data{tag}")
            if st is None:
                continue
            for s in ("a", "a1", "pix"):
                for cells in (("all", "cons") if setname == "natural" else ("all", "wide")):
                    k = kappa(st, (s, cells, "true"))
                    if k is None:
                        continue
                    kc = kappa(st, (s, cells, "copy")); ko = kappa(st, (s, cells, "oracle")); kb = kappa(st, (s, cells, "s_pix"))
                    res["kappa"][f"{tag}|{setname}|{s}|{cells}"] = {"k": k[0], "ci": list(np.nanpercentile(k[1], [2.5, 97.5])), "n": k[2]}
                    res["tool"][f"{tag}|{setname}|{s}|{cells}"] = {"copy": kc[0] if kc else None, "oracle": ko[0] if ko else None, "s_pix": kb[0] if kb else None}
                    v_ = lambda q: "—" if q is None else f"{q[0]:+.3f}"
                    lines.append(f"| {lab} | {setname} | {s} | {cells} | {fmt(k)} | {v_(kc)} | {v_(ko)} | {v_(kb)} |")
    lines += ["", "**자연 / 합성 κ 비** (전체 칸, 영상 bootstrap 독립 재표집)", "", "| encoder | 운반 연산 | 자연 / VOBench | 자연 / phys2d |", "|---|---|---|---|"]
    for tag, lab in ENCS:
        stn, _, _ = load("natural", f"_data{tag}"); stv, _, _ = load("vobench", f"_data{tag}"); stp, _, _ = load("phys2d", f"_data{tag}")
        for s in ("a", "a1", "pix"):
            kn = kappa(stn, (s, "all", "true"), 1) if stn else None
            rv = ratio(kn, kappa(stv, (s, "all", "true"), 2) if stv else None)
            rp = ratio(kn, kappa(stp, (s, "all", "true"), 3) if stp else None)
            if rv is None and rp is None:
                continue
            res["ratio"][f"{tag}|{s}"] = {"vobench": [rv[0], *rv[1]] if rv else None, "phys2d": [rp[0], *rp[1]] if rp else None}
            f_ = lambda r: "—" if r is None else f"{r[0]:.3f} [{r[1][0]:.3f}, {r[1][1]:.3f}]"
            lines.append(f"| {lab} | {s} | {f_(rv)} | {f_(rp)} |")
    # law test: c of models, paired vs vast70 (same cached videos)
    lines += ["", "**옮긴 계수 c (y = 모델 첫 걸음) · 법칙 검정** (운반 연산 a, 전체 칸; 짝지은 차이 = 같은 영상 bootstrap)", "",
              "| 모델 (학습 판) | 자연 c | VOBench c | phys2d c | VOBench c − 기준 블록 AR [95 %] | phys2d c − 기준 [95 %] |", "|---|---|---|---|---|---|"]
    for nm in NAT_TRAINED + TWO_D:
        cs, ds = [], []
        for setname in ("natural", "vobench", "phys2d"):
            st, _, _ = load(setname, nm)
            cs.append(kappa(st, ("a", "all", "model")) if st else None)
        for setname in ("vobench", "phys2d"):
            try:
                r = CC.compare("vast70", nm, setname, "a", "all") if nm != "vast70" else None
            except Exception:
                r = None
            ds.append(r)
        res["law"][nm] = {"c": [c[0] if c else None for c in cs], "diff_vs_vast70": ds}
        f2 = lambda r: "—" if r is None else f"{r['diff']:+.3f} [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]"
        grp = "2D 합성 이어 학습" if nm in TWO_D else "자연"
        lines.append(f"| {nm} ({grp}) | {fmt(cs[0])} | {fmt(cs[1])} | {fmt(cs[2])} | {f2(ds[0])} | {f2(ds[1])} |")
    for f, key in (("encprops_natural.json", "encprops_natural"), ("encprops_vobench.json", "encprops_vobench"), ("velprobe_vobench.json", "velprobe")):
        p = OUT / f
        if p.exists():
            res[key] = json.load(open(p))
    if res.get("encprops_natural"):
        lines += ["", "**떨림 · 칸 미만 매끄러움 (블록 7, 같은 영상)**", "", "| encoder | 판 | 떨림 (±1/255 / 다른 영상 같은 칸) | 반 칸 이동 vs 이웃 평균 (/ 이웃 차) | 한 칸 이동 vs 왼쪽 이웃 (대조) |", "|---|---|---|---|---|"]
        for key in ("encprops_natural", "encprops_vobench"):
            for e, v in (res.get(key) or {}).items():
                lines.append(f"| {e} | {key.split('_')[1]} | {v['jitter_over_other_video']:.3f} | {v['half_shift_vs_neighbour_avg_over_neighbour_diff']:.3f} | {v['full_shift_vs_left_neighbour_over_neighbour_diff']:.3f} |")
    if res.get("velprobe"):
        v = res["velprobe"]
        lines += ["", f"**블록 안 변위 판독 (VOBench 보이는 이동, 표본 {v['n_samples']} · 묶음 {v['n_groups']}, 묶음 반 보류)**", "",
                  "| 특징 | 보류 R² | 라벨 섞은 대조 R² |", "|---|---|---|"]
        for e in ("vjepa2", "dup", "dup2", "dinov2", "dinov2_2"):
            if e in v:
                lines.append(f"| {e} | {v[e]['heldout_R2']:.3f} | {v[e]['shuffled_label_R2']:.3f} |")
    json.dump(res, open(OUT / "g19_summary.json", "w"), indent=1, default=float)
    md = "\n".join(lines)
    (OUT / "g19_summary.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
