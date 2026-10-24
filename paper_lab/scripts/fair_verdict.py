#!/usr/bin/env python3
"""공정한 시험 판정 (진행자 2, 2026-10-22) — PLAN §6 '승인과 실제 설정' 의 규칙을 그대로 기계적으로 적용한다. CPU, 읽기만.

  python paper_lab/scripts/fair_verdict.py            # 네 run 이 모두 가진 가장 늦은 체크포인트에서 주 판정 + 학습 곡선 + 떨어지지 않음
입력: paper_lab/eval/results/fair{AXs0,AXs1,B0s0,B0s1}_e{2,4,6,8}/ (하네스 빠른 판) · vast70 (기준 블록 AR).
규칙 (결과 전, 해시 사본 _scratch/prereg/PLAN_2026-10-22_fair_1538.*):
  성공 = 먼 칸 자리 여유 AX − B0 (seed 짝 평균) ≥ 오라클 거리 10 % · 구간 0 위 · seed 둘 부호 같음 · 이진 구간이 0 을 안 품음 (양)
  얼림만 = 자리 여유 구간 0 위 · 먼 칸 구간이 0 을 품음 / 작은 이동 = 먼 칸 0 위 · 10 % 미만 / 무변화 = 모두 0 을 품음 / seed 부호 다르면 결론 없음
"""
import json, os, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "paper_lab/eval"))
sys.argv = sys.argv[:1]
import passline as PL  # noqa: E402

RES = REPO / "paper_lab/eval/results"
ARMS = ("AXs0", "AXs1", "B0s0", "B0s1")
FAR = 48.0


def have(name):
    return (RES / name / "passline_all.json").exists() and (RES / name / "vobench_r0.npz").exists()


def boot_blocks(val, B=1000, seed=0):
    """val {(block, dir): x} → (mean, lo, hi) 묶음 bootstrap."""
    keys = sorted(val); blocks = sorted({k[0] for k in keys}); by = defaultdict(list)
    for k in keys:
        by[k[0]].append(val[k])
    rng = np.random.default_rng(seed); bs = []
    for _ in range(B):
        pick = rng.integers(0, len(blocks), len(blocks))
        bs.append(np.mean([v for j in pick for v in by[blocks[j]]]))
    return float(np.mean([val[k] for k in keys])), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), len(keys)


def pair_stats(ax, b0, occ, mot, far=None, scale_from="vast70"):
    """seed 짝 평균 차 (AX_s − B0_s 의 seed 평균) — 이진 · 자리 여유 (오라클 몫). seed 별 점추정도."""
    H = {n: PL._pair_hits(n, occ=occ, mot=mot, far=far) for n in ax + b0}
    M = {n: PL._pair_hits(n, occ=occ, mot=mot, margin=True, far=far) for n in ax + b0}
    MO = PL._pair_hits(scale_from, occ=occ, mot=mot, render="oracle", margin=True, far=far)
    keys = sorted(set.intersection(*[set(h) for h in H.values()]) & set(MO))
    scale = float(np.mean([abs(MO[k]) for k in keys]))
    out = {"n_pairs": len(keys), "oracle_scale": scale, "acc": {n: 100 * float(np.mean([H[n][k] for k in keys])) for n in H}}
    db = {k: np.mean([H[a][k] - H[b][k] for a, b in zip(ax, b0)]) for k in keys}
    dm = {k: np.mean([(M[a][k] - M[b][k]) / scale for a, b in zip(ax, b0)]) for k in keys}
    out["binary_diff"] = [100 * x for x in boot_blocks(db)[:3]]
    out["margin_diff_pct"] = [100 * x for x in boot_blocks(dm)[:3]]
    out["per_seed"] = {f"{a}-{b}": {"binary": 100 * float(np.mean([H[a][k] - H[b][k] for k in keys])),
                                    "margin_pct": 100 * float(np.mean([(M[a][k] - M[b][k]) / scale for k in keys]))} for a, b in zip(ax, b0)}
    return out


def cell(name, key, idx=0):
    c = json.load(open(RES / name / "passline_all.json"))["cells"].get(key)
    if not c:
        return float("nan")
    v = c["model"]
    return float(np.mean(v)) if key in ("natural L1", "false carry static clips", "false carry far cells (moving clips)") else float(v[idx])


def main():
    cks = [e for e in (8, 6, 4, 2) if all(have(f"fair{a}_e{e}") for a in ARMS)]
    if not cks:
        sys.exit("평가가 끝난 공통 체크포인트가 없다")
    main_e = cks[0]
    print(f"공통 체크포인트: {sorted(cks)} → 주 판정 e{main_e}")
    ax = [f"fairAXs0_e{main_e}", f"fairAXs1_e{main_e}"]; b0 = [f"fairB0s0_e{main_e}", f"fairB0s1_e{main_e}"]
    R = {"main_ckpt": f"e{main_e}"}
    R["late_moving_all"] = pair_stats(ax, b0, "late", "moving")
    R["late_moving_far48"] = pair_stats(ax, b0, "late", "moving", far=FAR)
    R["late_static_B1"] = pair_stats(ax, b0, "late", "static")
    # 떨어지지 않음 · 보조 칸 (하네스 표, seed 평균)
    keys = {"B2 false carry static": "false carry static clips", "B2 false carry far cells": "false carry far cells (moving clips)",
            "B3 natural L1": "natural L1", "IntPhys1 late moving/occluded": "all-token ip1late moving/occluded",
            "apex48 (all-token)": "all-token gravity arc/apex48", "ledge (object-region)": "object-region gravity ledge",
            "visible carry gain": "carry gain all", "visible moving object-region": "object-region visible/moving"}
    side = {}
    for lab, k in keys.items():
        side[lab] = {n: cell(n, k) for n in ax + b0 + ["vast70"]}
    R["side"] = side
    # 판정
    far = R["late_moving_far48"]; allm = R["late_moving_all"]; bi = allm["binary_diff"]
    seeds_far = [v["margin_pct"] for v in far["per_seed"].values()]
    same_sign = np.sign(seeds_far[0]) == np.sign(seeds_far[1])
    if not same_sign:
        verdict = "결론 없음 (seed 부호가 다름)"
    elif far["margin_diff_pct"][0] >= 10 and far["margin_diff_pct"][1] > 0 and bi[1] > 0:
        verdict = "성공"
    elif allm["margin_diff_pct"][1] > 0 and far["margin_diff_pct"][1] <= 0 <= far["margin_diff_pct"][2]:
        verdict = "얼림만"
    elif far["margin_diff_pct"][1] > 0:
        verdict = "작은 이동"
    elif allm["margin_diff_pct"][1] <= 0 <= allm["margin_diff_pct"][2] and far["margin_diff_pct"][1] <= 0 <= far["margin_diff_pct"][2] and bi[1] <= 0 <= bi[2]:
        verdict = "무변화"
    else:
        verdict = "그 밖 (표를 그대로 보고)"
    b1 = R["late_static_B1"]["binary_diff"]
    mean = lambda d, ns: float(np.mean([d[n] for n in ns]))
    checks = {
        "B1 가려진 정지 (구간 아래끝 ≥ −3)": b1[1] >= -3,
        "B2 거짓 운반 정지 (AX ≤ B0 × 1.05)": mean(side["B2 false carry static"], ax) <= 1.05 * mean(side["B2 false carry static"], b0),
        "B2 거짓 운반 먼 칸 (AX ≤ B0 × 1.05)": mean(side["B2 false carry far cells"], ax) <= 1.05 * mean(side["B2 false carry far cells"], b0),
        "B3 자연 영상 L1 (AX ≤ B0 × 1.01)": mean(side["B3 natural L1"], ax) <= 1.01 * mean(side["B3 natural L1"], b0),
        "IntPhys 1 끝 가림 (AX ≥ B0)": mean(side["IntPhys1 late moving/occluded"], ax) >= mean(side["IntPhys1 late moving/occluded"], b0),
        "포물선 48 (AX ≥ B0)": mean(side["apex48 (all-token)"], ax) >= mean(side["apex48 (all-token)"], b0),
        "선반 물체 자리 (AX ≥ B0)": mean(side["ledge (object-region)"], ax) >= mean(side["ledge (object-region)"], b0),
    }
    R["verdict"] = verdict; R["checks"] = {k: bool(v) for k, v in checks.items()}
    # 학습 곡선 (먼 칸 자리 여유 AX − B0)
    curve = {}
    for e in sorted(cks):
        st = pair_stats([f"fairAXs0_e{e}", f"fairAXs1_e{e}"], [f"fairB0s0_e{e}", f"fairB0s1_e{e}"], "late", "moving", far=FAR)
        curve[f"e{e}"] = st["margin_diff_pct"]
    R["curve_far_margin"] = curve
    # B0 가 기준 블록 AR 에서 얼마나 움직였나
    R["B0_vs_vast70"] = {s: PL.compare(["vast70", f"fairB0{s}_e{main_e}"], "all")["rows"]["all"] for s in ("s0", "s1")}
    print(json.dumps({k: R[k] for k in ("main_ckpt", "verdict", "checks")}, ensure_ascii=False, indent=1))
    for k in ("late_moving_all", "late_moving_far48", "late_static_B1"):
        r = R[k]
        print(f"{k:20s} n {r['n_pairs']}  binary {r['binary_diff'][0]:+.1f} [{r['binary_diff'][1]:+.1f}, {r['binary_diff'][2]:+.1f}]  "
              f"margin {r['margin_diff_pct'][0]:+.2f}% [{r['margin_diff_pct'][1]:+.2f}, {r['margin_diff_pct'][2]:+.2f}]  per seed {r['per_seed']}  acc {r['acc']}")
    print("curve (far margin AX−B0):", {k: [round(x, 2) for x in v] for k, v in curve.items()})
    for lab, d in side.items():
        print(f"  {lab:32s} " + "  ".join(f"{n}: {v:.3f}" for n, v in d.items()))
    out = REPO / "paper_lab/_scratch/fair_verdict.json"
    json.dump(R, open(out, "w"), indent=1, ensure_ascii=False, default=float)
    print("→", out)


if __name__ == "__main__":
    main()
