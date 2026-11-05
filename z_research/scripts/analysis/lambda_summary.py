#!/usr/bin/env python3
"""negimp λ ablation 요약 (2026-11-05). GPU 불필요. λ 0 = L1 대조 (v4vc_*_postft), 0.5 · 1 · 2 = negimp (weight).
보드마다 장면 단위: A (사라짐) · B (나타남) · D = 1[m_A+m_B>0] · c (치우침) · 연속 문맥 margin g = (m_A+m_B)/장면 평균 S (%).
+ 가능 영상 예측 오차 = 학습 중 검증 (VOBench v1 late 432 쌍, e20) 의 가능 영상 surprise.
  /venv/vjepa/bin/python z_research/scripts/analysis/lambda_summary.py > z_analysis/results/scene_paired_20261105/LAMBDA_ABLATION.md
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from scene_paired_tests import boards, scene_stats, summarize, binom_p, RES

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
FAM = {"released": [("0 (L1)", "vith_release_v4vc", "v4vc_release_postft"), ("0.5", "vith_release_v4vc_negimp_w05", "v4vc_release_negimp_w05"),
                    ("1", "vith_release_v4vc_negimp", "v4vc_release_negimp"), ("2", "vith_release_v4vc_negimp_w2", "v4vc_release_negimp_w2")],
       "AR": [("0 (L1)", "vast_ar_vw_state_v4vc", "v4vc_ar_vw_state_postft"), ("0.5", "vast_ar_vw_state_v4vc_negimp_w05", "v4vc_ar_vw_state_negimp_w05"),
              ("1", "vast_ar_vw_state_v4vc_negimp", "v4vc_ar_vw_state_negimp"), ("2", "vast_ar_vw_state_v4vc_negimp_w2", "v4vc_ar_vw_state_negimp_w2"),
              ("1 짝 손실 · 1:1:1", "vast_ar_vw_state_v4vc_negpair", "v4vc_ar_vw_state_negpair")]}
SETS = ["VOBench v3 late · panel", "VOBench v3 late · moving occluder", "IntPhys1 Surrogate", "real world (all 95)",
        "IntPhys1 late · O1 occluded", "IntPhys2 late · permanence (A=pair 1, B=pair 2)"]


def val_pos(run):
    f = f"{REPO}/z_training/runs/{run}/metrics.jsonl"
    if not os.path.exists(f):
        return None
    d = [json.loads(l) for l in open(f)][-1]
    return d["epoch"], d["val"]["surprise_pos"], d["val"]["acc"], (d.get("neg") or {}).get("pos_l1"), (d.get("neg") or {}).get("imp_l1")


def main():
    B = boards()
    print("# negimp λ ablation — 장면 단위 요약 (2026-11-05, vast)\n")
    print("λ = 불가능 쪽 손실 가중 (λ 0 = L1 만, 같은 2,000 step · batch 48 · seed 0). D = 장면 판별, c > 0 = '빈 미래가 싸다' 치우침, g = 연속 문맥 margin (%).\n")
    for fam, rows in FAM.items():
        print(f"\n## {fam}\n")
        print("| λ | 가능 영상 예측 오차 (VOB v1 late val) | val 정답률 | 학습 가능 L1 / 불가능 L1 (e20) |\n|---|---:|---:|---|")
        for lam, m, run in rows:
            v = val_pos(run)
            if v:
                e, sp, acc, pl, il = v
                print(f"| {lam} | {sp:.4f} | {100 * acc:.1f} | {('%.3f / %.3f' % (pl, il)) if pl is not None else '—'} |")
        for sname in SETS:
            pat, blk = B[sname]
            print(f"\n**{sname}** (장면 {len(blk)})\n\n| λ | A | B | D | J | c | g (%) | D 고침/깨짐 vs λ0 (p) |\n|---|---:|---:|---:|---:|---:|---:|---|")
            base = None
            for lam, m, _ in rows:
                f = f"{RES}/{pat.format(m=m)}/per_block.json"
                if not os.path.exists(f):
                    print(f"| {lam} | — |"); continue
                S = json.load(open(f))["per_video_surprise"]
                S = {k: (v["all"] if isinstance(v, dict) else v) for k, v in S.items()}
                r = scene_stats(S, blk)
                if r is None:
                    print(f"| {lam} | 영상 빠짐 |"); continue
                s = summarize(r); g = sum(100 * (x[1] + x[2]) / x[3] for x in r) / len(r)
                fb = "—"
                if base is None:
                    base = {x[0]: x for x in r}
                else:
                    ks = [x[0] for x in r if x[0] in base]; d1 = {x[0]: x for x in r}
                    fx = sum(1 for k in ks if base[k][1] + base[k][2] <= 0 < d1[k][1] + d1[k][2]); bk = sum(1 for k in ks if d1[k][1] + d1[k][2] <= 0 < base[k][1] + base[k][2])
                    fb = f"{fx}/{bk} ({binom_p(fx, fx + bk):.2g})"
                print(f"| {lam} | {s['A']:.1f} | {s['B']:.1f} | {s['C']:.1f} | {s['J']:.1f} | {s['c']:+.2f} | {g:.2f} | {fb} |")


if __name__ == "__main__":
    main()
