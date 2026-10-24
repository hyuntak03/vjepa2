"""근거표 v3 의 파일럿 표 (진행자 1, 2026-10-22). 하네스 산출물에서만 읽는다 — 새 계산은 자연 영상 L1 짝지은 bootstrap 하나.

    python paper_lab/scripts/pilot_v3_table.py            # 표 셋을 markdown 으로 출력 + paper_lab/_scratch/pilot_v3_table.json

읽는 것: paper_lab/eval/results/<name>/passline_all.json (cells), compare_*.json (짝지은 묶음 bootstrap, passline.py compare),
         <name>/natural_r0.npz (L_model, 영상별 걸음 L1 — B3 짝지은 구간).
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
R = os.path.join(ROOT, "paper_lab/eval/results")

ARMS = [("vast70", "기준 블록 AR (출발점, 전체 판)"),
        ("pilotA0_e5", "A0 e5"), ("pilotA0_e10", "A0 e10"),
        ("pilotAX_e5", "AX e5"), ("pilotAX_e10", "AX e10"),
        ("pilotA1_e10", "A1 e10"), ("pilotA1c_e10", "A1′ e10"),
        ("pilotA2c_e10", "A2c e10 (가림막 + 문맥 판 가중, 원래 공간)"), ("pilotA4_e10", "A4 e10 (움직이는 칸 위 가림막)"),
        ("ext_ar_vw_shortctx_e30_c8", "짧은 문맥 e30 · 문맥 8"), ("ext_ar_vw_shortctx_e30_c2", "짧은 문맥 e30 · 문맥 2")]
TSP = [("pilotA1_e10_tsp", "A1 e10 · 학습 공간"), ("pilotA1c_e10_tsp", "A1′ e10 · 학습 공간")]


def J(p):
    return json.load(open(p))


def cells(name):
    p = os.path.join(R, name, "passline_all.json")
    return J(p)["cells"] if os.path.exists(p) else None


def pt(c, cell, key="model"):
    return c[cell][key][0] if c and cell in c else float("nan")


def mean8(c, cell, key="model"):
    return float(np.mean(c[cell][key])) if c and cell in c else float("nan")


def cmp(a, b, cell="late_moving"):
    p = os.path.join(R, f"compare_{a}__{b}_all_{cell}.json")
    if not os.path.exists(p):
        return None
    r = J(p)["rows"]["all"]
    return r["diff"], r["ci95"][0], r["ci95"][1]


def cmpm(a, b, cell="late_moving"):
    """continuous object-region margin change, % of mean |oracle margin| (R2 round-2 definition) + discordant pairs"""
    p = os.path.join(R, f"compare_{a}__{b}_all_{cell}.json")
    if not os.path.exists(p):
        return None, None
    r = J(p)["rows"]["all"]
    if "margin_share_pct" not in r:
        return None, None
    m = r["margin_share_pct"]
    return (m["diff"], m["ci95"][0], m["ci95"][1]), r.get("n_discordant")


def did(a1, b1, a2, b2, cell="late_moving"):
    p = os.path.join(R, f"compare_{a1}__{b1}__{a2}__{b2}_all_{cell}.json")
    if not os.path.exists(p):
        return None
    r = J(p)["rows"]["all"]
    return r["diff"], r["ci95"][0], r["ci95"][1]


def nat_paired(a, b, B=1000):
    """B3: (L(b) − L(a)) / L(a) in %, step mean, paired over the same 200 K400 val videos, bootstrap over videos."""
    za, zb = np.load(os.path.join(R, a, "natural_r0.npz")), np.load(os.path.join(R, b, "natural_r0.npz"))
    assert (za["video"] == zb["video"]).all()
    la, lb = za["L_model"].mean(1), zb["L_model"].mean(1)
    rng = np.random.default_rng(0)
    bs = [100 * (lb[i].mean() - la[i].mean()) / la[i].mean() for i in (rng.integers(0, len(la), len(la)) for _ in range(B))]
    return 100 * (lb.mean() - la.mean()) / la.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def f(x, d=1):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def fi(t, d=1):
    return "—" if t is None else f"{t[0]:+.{d}f} [{t[1]:+.{d}f}, {t[2]:+.{d}f}]"


def main():
    out = {}
    rows = []
    hdr = ("| 팔 | 주 지표: 끝 가림 이동 물체 자리 (360) | B1 끝 가림 정지 물체 자리 (72) | B2 거짓 운반 정지 · 먼 칸 | B3 자연 영상 L1 | "
           "보이는 이동 물체 자리 (120) · 운반 이득 | IntPhys 1 끝 가림 (180) · 이동 가림 (60) | 포물선 · 선반 (물체 자리) |")
    rows.append(hdr)
    rows.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, lab in ARMS + TSP:
        c = cells(name)
        if c is None:
            rows.append(f"| {lab} | (아직 없음) | | | | | | |")
            continue
        v = dict(main=pt(c, "object-region late/moving"), b1=pt(c, "object-region late/static"),
                 b2s=mean8(c, "false carry static clips"), b2f=mean8(c, "false carry far cells (moving clips)"),
                 b3=mean8(c, "natural L1"), vis=pt(c, "object-region visible/moving"), carry=pt(c, "carry gain all"),
                 ip=pt(c, "all-token ip1late all"), ipm=pt(c, "all-token ip1late moving/occluded"),
                 arc=pt(c, "object-region gravity arc"), ledge=pt(c, "object-region gravity ledge"))
        out[name] = v
        rows.append(f"| {lab} | {f(v['main'])} | {f(v['b1'])} | {f(v['b2s'], 3)} · {f(v['b2f'], 3)} | {f(v['b3'], 3)} | "
                    f"{f(v['vis'])} · {f(v['carry'], 2)} | {f(v['ip'])} · {f(v['ipm'])} | {f(v['arc'])} · {f(v['ledge'])} |")
    base = cells("vast70")
    rows.append(f"| 복사 · 문맥 평균 · 오라클 (원래 공간) | {f(pt(base, 'object-region late/moving', 'copy'))} · {f(pt(base, 'object-region late/moving', 'mean8'))} · {f(pt(base, 'object-region late/moving', 'oracle'))} | "
                f"{f(pt(base, 'object-region late/static', 'copy'))} · {f(pt(base, 'object-region late/static', 'mean8'))} · {f(pt(base, 'object-region late/static', 'oracle'))} | "
                f"복사 {f(mean8(base, 'false carry static clips', 'copy'), 3)} · {f(mean8(base, 'false carry far cells (moving clips)', 'copy'), 3)} | "
                f"{f(mean8(base, 'natural L1', 'copy'), 3)} · {f(mean8(base, 'natural L1', 'mean8'), 3)} | "
                f"{f(pt(base, 'object-region visible/moving', 'copy'))} · {f(pt(base, 'object-region visible/moving', 'mean8'))} · {f(pt(base, 'object-region visible/moving', 'oracle'))}; 상한 {f(pt(base, 'carry gain all', 'truth'), 2)} | "
                f"{f(pt(base, 'all-token ip1late all', 'copy'))} · {f(pt(base, 'all-token ip1late moving/occluded', 'copy'))} (복사) | "
                f"{f(pt(base, 'object-region gravity arc', 'mean8'))} · {f(pt(base, 'object-region gravity ledge', 'mean8'))} (문맥 평균; 우연 11.1 · 50) |")
    for name, lab in TSP:
        c = cells(name)
        if c:
            rows.append(f"| ↳ {lab} 의 기준선 (복사 · 문맥 평균 · 오라클, 사영 공간) | {f(pt(c, 'object-region late/moving', 'copy'))} · {f(pt(c, 'object-region late/moving', 'mean8'))} · {f(pt(c, 'object-region late/moving', 'oracle'))} | "
                        f"{f(pt(c, 'object-region late/static', 'copy'))} · {f(pt(c, 'object-region late/static', 'mean8'))} · {f(pt(c, 'object-region late/static', 'oracle'))} | "
                        f"복사 {f(mean8(c, 'false carry static clips', 'copy'), 3)} · {f(mean8(c, 'false carry far cells (moving clips)', 'copy'), 3)} | "
                        f"{f(mean8(c, 'natural L1', 'copy'), 3)} · {f(mean8(c, 'natural L1', 'mean8'), 3)} | "
                        f"{f(pt(c, 'object-region visible/moving', 'copy'))} · {f(pt(c, 'object-region visible/moving', 'mean8'))} · {f(pt(c, 'object-region visible/moving', 'oracle'))}; 상한 {f(pt(c, 'carry gain all', 'truth'), 2)} | "
                        f"{f(pt(c, 'all-token ip1late all', 'copy'))} · {f(pt(c, 'all-token ip1late moving/occluded', 'copy'))} (복사) | "
                        f"{f(pt(c, 'object-region gravity arc', 'mean8'))} · {f(pt(c, 'object-region gravity ledge', 'mean8'))} |")
    print("\n".join(rows))

    print("\n| 짝지은 차이 (묶음 bootstrap 1,000, 95 %) | 주 지표 (360) | 불일치 쌍 | 주 지표 연속: 자리 여유 변화 (오라클 거리 몫 %) | B1 정지 (72) | B1 연속 (오라클 거리 몫 %) | B3 자연 영상 L1 변화 % (영상 200) |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    pairs = [("pilotA0_e10", "pilotAX_e10", "AX − A0 (e10)"), ("pilotA0_e5", "pilotAX_e5", "AX − A0 (e5)"),
             ("pilotA0_e10", "pilotA1_e10", "A1 − A0 (e10)"), ("pilotA0_e10", "pilotA1c_e10", "A1′ − A0 (e10)"),
             ("pilotA0_e10", "pilotA2c_e10", "A2c − A0 (e10)"), ("pilotA0_e10", "pilotA4_e10", "A4 − A0 (e10)"),
             ("pilotAX_e10", "pilotA2c_e10", "A2c − AX (e10)"), ("pilotAX_e10", "pilotA4_e10", "A4 − AX (e10)"),
             ("pilotAX_e10", "pilotA1_e10", "A1 − AX (e10)"), ("pilotAX_e10", "pilotA1c_e10", "A1′ − AX (e10)"),
             ("vast70", "pilotA0_e10", "A0 e10 − 출발점 (이어 학습만)"),
             ("vast70", "ext_ar_vw_shortctx_e30_c8", "짧은 문맥 · 문맥 8 − 기준 블록 AR"),
             ("vast70", "ext_ar_vw_shortctx_e30_c2", "짧은 문맥 · 문맥 2 − 기준 블록 AR"),
             ("pilotA0_e10", "pilotA1_e10_tsp", "A1 (사영 공간 채점) − A0 (원래 공간) — 쌍마다 각자 학습 공간"),
             ("pilotA0_e10", "pilotA1c_e10_tsp", "A1′ (사영 공간 채점) − A0 (원래 공간) — 쌍마다 각자 학습 공간")]
    out["paired"] = {}
    for a, b, lab in pairs:
        m, s = cmp(a, b, "late_moving"), cmp(a, b, "late_static")
        try:
            n = None if b.endswith("_tsp") else nat_paired(a, b)          # L1 across two spaces is not comparable
        except Exception:
            n = None
        mm, nd = cmpm(a, b, "late_moving")
        ms, _ = cmpm(a, b, "late_static")
        out["paired"][lab] = dict(main=m, b1=s, b3=n, main_margin=mm, b1_margin=ms, discordant=nd)
        print(f"| {lab} | {fi(m)} | {nd if nd is not None else '—'} | {fi(mm, 2)} | {fi(s)} | {fi(ms, 2)} | {fi(n, 2)} |")
    d1, d2 = did("pilotA0_e5", "pilotAX_e5", "pilotA0_e10", "pilotAX_e10", "late_moving"), did("pilotA0_e5", "pilotAX_e5", "pilotA0_e10", "pilotAX_e10", "late_static")
    out["paired"]["DiD AX e5→e10"] = dict(main=d1, b1=d2)
    dm = None
    pdid = os.path.join(R, "compare_pilotA0_e5__pilotAX_e5__pilotA0_e10__pilotAX_e10_all_late_moving.json")
    if os.path.exists(pdid) and "margin_share_pct" in J(pdid)["rows"]["all"]:
        q = J(pdid)["rows"]["all"]["margin_share_pct"]; dm = (q["diff"], q["ci95"][0], q["ci95"][1])
    out["paired"]["DiD AX e5→e10"]["main_margin"] = dm
    print(f"| 학습 곡선 (AX − A0) e10 − e5 (차이의 차이) | {fi(d1)} | — | {fi(dm, 2)} | {fi(d2)} | — | — |")
    json.dump(out, open(os.path.join(ROOT, "paper_lab/_scratch/pilot_v3_table.json"), "w"), indent=1, ensure_ascii=False, default=float)


if __name__ == "__main__":
    sys.exit(main())
