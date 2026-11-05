#!/usr/bin/env python3
"""장면 단위 짝 검정 (2026-11-05, tak's_story 세션 요청 · 사용자 논문 Table 2/5). GPU 불필요 — 채점 산출물 per_block.json 만 읽는다.

장면 = 문맥이 다른 두 짝 (A · B) 을 가진 block.  m_A = S(imp_A) − S(pos_A),  m_B = S(imp_B) − S(pos_B)  (S = 영상별 surprise).
  영속 보드에서 A = 문맥에 물체 있음 (불가능 = 사라짐), B = 문맥 빔 (불가능 = 나타남).
  A · B   = 짝 정답률 (m > 0, 동점 0.5)
  C       = 1[m_A + m_B > 0] 의 평균 = 문맥 항 (결과 쪽 치우침이 상쇄된 판별) · Wilson 95 %
  J       = 1[m_A > 0 ∧ m_B > 0] 의 평균 (장면의 두 짝 모두 정답)
  d′, c   = 신호탐지: '빈 미래가 더 놀랍다' 반응. H = A 정답률, F = 1 − B 정답률, 로그-선형 보정 (k + 0.5) / (n + 1).
            d′ = z(H) − z(F),  c = −(z(H) + z(F)) / 2  (c > 0: 물체 있는 미래를 더 놀랍다고 보는 치우침 = "빈 미래가 싸다")
  결과 항 = (m_A − m_B) / 2 의 평균 (빈 미래 대 물체 미래의 surprise 차, 문맥과 무관한 몫)
McNemar = 같은 장면에서 C 정오가 대조 판과 갈린 수 (고침 / 깨짐, 정확 이항 양측).
문맥 의존성 (요청 2): 대조 → 처리 Δm_A · Δm_B (장면 평균, 장면 부트스트랩 95 %), 그리고 장면 평균 surprise 로 나눈 상대 margin.

  /venv/vjepa/bin/python z_research/scripts/analysis/scene_paired_tests.py --out z_analysis/results/scene_paired_20261105
"""
import argparse, csv, json, math, os, random
from collections import defaultdict
from statistics import NormalDist, median

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
RES = f"{REPO}/cvpr/results/eval"
Z = NormalDist().inv_cdf

MODELS = [  # (시트 행 이름, 채점 모델, 실제 영상 보드에서 보류 절반만 읽나)
    ("V-JEPA 2 (release)", "vith", False),
    ("release + v4 (2k)", "vith_release_v4vc", False),
    ("release + v4 + neg", "vith_release_v4vc_negimp", False),
    ("release + VOBench", "vith_release_vobpft_e20", False),
    ("release + real-perm", "vith_release_realbench", True),
    ("release + WROP B (SSv2)", "vast_permwrop_B", False),
    ("release + WROP C (SSv2+occ)", "vast_permwrop_C", False),
    ("release + WROP D (SSv2+non-perm)", "vast_permwrop_D", False),
    ("AR scratch", "vast_ar_vw_state", False),
    ("AR + v4 (2k)", "vast_ar_vw_state_v4vc", False),
    ("AR + v4 + neg", "vast_ar_vw_state_v4vc_negimp", False),
    ("AR + VOBench", "vast_ar_vw_state_vobpft", False),
    ("AR + real-perm", "vast_ar_vw_state_realbench", True),
]
CONTRASTS = [("vith", "vith_release_v4vc"), ("vith_release_v4vc", "vith_release_v4vc_negimp"), ("vith", "vith_release_vobpft_e20"),
             ("vith", "vith_release_realbench"), ("vith", "vast_permwrop_B"), ("vast_permwrop_D", "vast_permwrop_C"), ("vast_permwrop_B", "vast_permwrop_C"),
             ("vast_ar_vw_state", "vast_ar_vw_state_v4vc"), ("vast_ar_vw_state_v4vc", "vast_ar_vw_state_v4vc_negimp"),
             ("vast_ar_vw_state", "vast_ar_vw_state_vobpft"), ("vast_ar_vw_state", "vast_ar_vw_state_realbench")]


def _blocks(idx, keep=lambda r: True, a_of=None):
    """index → {block: {"A": (pos_id, imp_id), "B": (...)}}. a_of(rows of a pair) → True 면 그 짝이 A."""
    by = defaultdict(lambda: defaultdict(dict))
    for r in csv.DictReader(open(idx)):
        if keep(r):
            by[r["block_id"]][r["pair_id"]]["pos" if r["plausible"] == "1" else "imp"] = r
    out = {}
    for b, prs in by.items():
        if len(prs) != 2 or any(len(v) != 2 for v in prs.values()):
            continue
        ks = sorted(prs)
        if a_of is None:
            a = "A" if "A" in prs else ks[0]
        else:
            cand = [k for k in ks if a_of(prs[k])]
            if len(cand) != 1:
                continue
            a = cand[0]
        bb = [k for k in ks if k != a][0]
        out[b] = {"A": (prs[a]["pos"]["video_id"], prs[a]["imp"]["video_id"]), "B": (prs[bb]["pos"]["video_id"], prs[bb]["imp"]["video_id"])}
    return out


def boards():
    D = f"{REPO}/data_csv"
    vob = f"{D}/vobench_v3/index.csv"
    held = {r["video_id"] for r in csv.DictReader(open(f"{D}/real_world_permanence_uniform/index_test.csv"))}
    rwp = _blocks(f"{D}/real_world_permanence_uniform/index.csv")
    return {
        "VOBench v3 late · panel": ("testbed/vobench_v3_{m}", _blocks(vob, lambda r: r["occluder_kind"] == "panel")),
        "VOBench v3 late · moving occluder": ("testbed/vobench_v3_{m}", _blocks(vob, lambda r: r["occluder_kind"] == "mover")),
        "IntPhys1 Surrogate": ("testbed/intphys1_surrogate_{m}", _blocks(f"{D}/intphys1_surrogate/index.csv")),
        "real world (all 95)": ("testbed/real_world_permanence_uniform_{m}", rwp),
        "real world (held-out 48)": ("testbed/real_world_permanence_uniform_{m}", {b: v for b, v in rwp.items() if all(x in held for x in v["A"] + v["B"])}),
        "IntPhys1 late · O1 occluded": ("intphys1_late/intphys1_late_uni_{m}",
                                       _blocks(f"{D}/intphys1_late_uni/index.csv", lambda r: r["block_type"] == "O1" and r["group"].endswith("/occluded"),
                                               a_of=lambda pr: pr["pos"]["event_kind"] != "not_seen_before_d")),
        "IntPhys2 late · permanence (A=pair 1, B=pair 2)": ("testbed/intphys2_late_final_{m}",
                                                            _blocks(f"{D}/intphys2_late_final/index.csv", lambda r: r["condition"] == "permanence", a_of=lambda pr: pr["pos"]["pair_id"] == "1")),
    }


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def binom_p(k, n):
    if n == 0:
        return 1.0
    x = min(k, n - k)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(x + 1)) / 2 ** n)


def scene_stats(S, blk):
    rows = []
    for b, v in blk.items():
        (pa, ia), (pb, ib) = v["A"], v["B"]
        if any(x not in S for x in (pa, ia, pb, ib)):
            return None
        mA, mB = S[ia] - S[pa], S[ib] - S[pb]
        mean_s = (S[pa] + S[ia] + S[pb] + S[ib]) / 4
        rows.append((b, mA, mB, mean_s))
    return rows


def summarize(rows):
    n = len(rows); sg = lambda x: 1.0 if x > 0 else (0.5 if x == 0 else 0.0)
    A = sum(sg(r[1]) for r in rows); B = sum(sg(r[2]) for r in rows)
    C = sum(sg(r[1] + r[2]) for r in rows); J = sum(1 for r in rows if r[1] > 0 and r[2] > 0)
    H = (A + 0.5) / (n + 1); F = (n - B + 0.5) / (n + 1)
    return {"n": n, "A": 100 * A / n, "B": 100 * B / n, "C": 100 * C / n, "C_ci": wilson(C, n), "J": 100 * J / n,
            "dprime": Z(H) - Z(F), "c": -(Z(H) + Z(F)) / 2, "outcome": sum((r[1] - r[2]) / 2 for r in rows) / n}


def boot_ci(xs, it=4000, seed=0):
    rnd = random.Random(seed); n = len(xs)
    ms = sorted(sum(xs[rnd.randrange(n)] for _ in range(n)) / n for _ in range(it))
    return ms[int(0.025 * it)], ms[int(0.975 * it)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    B = boards(); md = []; js = {}
    lab = {m: l for l, m, _ in MODELS}
    md.append("# 장면 단위 짝 검정 · 문맥 항 (2026-11-05, vast, 채점 산출물에서 재계산)\n")
    md.append("정의는 `z_research/scripts/analysis/scene_paired_tests.py` 머리말. C = 1[m_A+m_B>0] (Wilson 95 %), J = 두 짝 모두 정답, "
              "c > 0 = '빈 미래가 싸다' 치우침, 결과 항 = mean (m_A − m_B)/2.\n")
    cache = {}
    for bname, (pat, blk) in B.items():
        md.append(f"\n## {bname}  (장면 {len(blk)})\n")
        md.append("| 행 | 모델 | n | A | B | **C** [95 %] | J | d′ | c | 결과 항 |\n|---|---|---:|---:|---:|---|---:|---:|---:|---:|")
        for l, m, held_only in MODELS:
            if bname.startswith("real world (all") and held_only:
                continue            # real bench 로 학습한 판은 보류 절반만
            f = f"{RES}/{pat.format(m=m)}/per_block.json"
            if not os.path.exists(f):
                md.append(f"| {l} | `{m}` | — | 채점 없음 | | | | | | |"); continue
            S = json.load(open(f))["per_video_surprise"]
            rows = scene_stats(S, blk)
            if rows is None:
                md.append(f"| {l} | `{m}` | — | 영상 빠짐 | | | | | | |"); continue
            s = summarize(rows); cache[(bname, m)] = rows; js[f"{bname}|{m}"] = s
            md.append(f"| {l} | `{m}` | {s['n']} | {s['A']:.1f} | {s['B']:.1f} | **{s['C']:.1f}** [{s['C_ci'][0]:.1f}, {s['C_ci'][1]:.1f}] | {s['J']:.1f} | "
                      f"{s['dprime']:+.2f} | {s['c']:+.2f} | {s['outcome']:+.5f} |")
        # McNemar (C) + 문맥 의존성 (Δm_A · Δm_B)
        md.append("\n| 대조 → 처리 | C 고침 / 깨짐 | McNemar p | ΔC | ΔJ | Δm_A 평균 [95 %] | Δm_B 평균 [95 %] | 중앙값 Δm_A / Δm_B | Δ상대 m_A / m_B | 장면 Δm_A>0 ∧ Δm_B<0 |\n"
                  "|---|---:|---:|---:|---:|---|---|---|---|---:|")
        for m0, m1 in CONTRASTS:
            r0, r1 = cache.get((bname, m0)), cache.get((bname, m1))
            if not r0 or not r1:
                continue
            d0 = {r[0]: r for r in r0}; d1 = {r[0]: r for r in r1}; ks = sorted(set(d0) & set(d1))
            c0 = {k: d0[k][1] + d0[k][2] > 0 for k in ks}; c1 = {k: d1[k][1] + d1[k][2] > 0 for k in ks}
            fx = sum(1 for k in ks if not c0[k] and c1[k]); bk = sum(1 for k in ks if c0[k] and not c1[k])
            j0 = sum(1 for k in ks if d0[k][1] > 0 and d0[k][2] > 0); j1 = sum(1 for k in ks if d1[k][1] > 0 and d1[k][2] > 0)
            dA = [d1[k][1] - d0[k][1] for k in ks]; dB = [d1[k][2] - d0[k][2] for k in ks]
            rA = [d1[k][1] / d1[k][3] - d0[k][1] / d0[k][3] for k in ks]; rB = [d1[k][2] / d1[k][3] - d0[k][2] / d0[k][3] for k in ks]
            both = sum(1 for x, y in zip(dA, dB) if x > 0 and y < 0)
            n = len(ks); ciA, ciB = boot_ci(dA), boot_ci(dB)
            md.append(f"| {lab[m0]} → {lab[m1]} | {fx} / {bk} | {binom_p(fx, fx + bk):.3g} | {100 * (sum(c1.values()) - sum(c0.values())) / n:+.1f} | "
                      f"{100 * (j1 - j0) / n:+.1f} | {sum(dA) / n:+.5f} [{ciA[0]:+.5f}, {ciA[1]:+.5f}] | {sum(dB) / n:+.5f} [{ciB[0]:+.5f}, {ciB[1]:+.5f}] | "
                      f"{median(dA):+.5f} / {median(dB):+.5f} | {100 * sum(rA) / n:+.2f} % / {100 * sum(rB) / n:+.2f} % | {both} / {n} |")
            js[f"{bname}|{m0}->{m1}"] = {"fix": fx, "break": bk, "p": binom_p(fx, fx + bk), "dmA": sum(dA) / n, "dmB": sum(dB) / n, "dmA_ci": ciA, "dmB_ci": ciB,
                                          "rel_dmA": sum(rA) / n, "rel_dmB": sum(rB) / n, "med_dmA": median(dA), "med_dmB": median(dB), "shift_both": both, "n": n}
    open(f"{a.out}/SCENE_PAIRED_TESTS.md", "w").write("\n".join(md) + "\n")
    json.dump(js, open(f"{a.out}/scene_paired_tests.json", "w"), indent=1, ensure_ascii=False)
    print("\n".join(md))


if __name__ == "__main__":
    main()
