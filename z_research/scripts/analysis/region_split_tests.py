#!/usr/bin/env python3
"""물체 영역 vs 먼 배경으로 나눈 장면 단위 검정 (2026-11-05, tak's_story 요청 ①). GPU 불필요.
입력: cvpr/results/eval/testbed/<보드>_<모델>_region/per_block.json — per_video_surprise[vid] = {all, object, far_bg}
  (채점기 token_subset: object = 미래 블록마다 물체 투영 중심 3×3, far_bg = 중심에서 Chebyshev 3 칸 이상. frames.csv = build_testbed_frames_csv.py)
정의는 scene_paired_tests.py 와 같다 (A · B · C · J · c, McNemar).
  /venv/vjepa/bin/python z_research/scripts/analysis/region_split_tests.py --out z_analysis/results/scene_paired_20261105
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from scene_paired_tests import boards, scene_stats, summarize, binom_p, RES

PAIRS = [("release + v4 (2k)", "vith_release_v4vc", "release + v4 + neg", "vith_release_v4vc_negimp"),
         ("AR + v4 (2k)", "vast_ar_vw_state_v4vc", "AR + v4 + neg", "vast_ar_vw_state_v4vc_negimp")]
BOARDS = {"IntPhys1 Surrogate": "intphys1_surrogate", "VOBench v3 late · panel": "vobench_v3", "VOBench v3 late · moving occluder": "vobench_v3"}
MODES = [("all", "전체 토큰"), ("object", "물체 영역 3×3"), ("far_bg", "먼 배경 (≥3 칸)")]


def load(board, m, mode):
    f = f"{RES}/testbed/{board}_{m}_region/per_block.json"
    if not os.path.exists(f):
        return None
    return {k: v[mode] for k, v in json.load(open(f))["per_video_surprise"].items()}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    B = boards(); md = ["# 물체 영역 vs 먼 배경 — negimp 와 2k 대조 (2026-11-05, vast)\n",
                       "object = 미래 블록마다 물체 투영 중심 (가림막 뒤여도) 의 3×3 패치, far_bg = 그 중심에서 3 패치 이상. 블록 안 네 변이에 같은 마스크.\n"]
    for bname, bfile in BOARDS.items():
        blk = B[bname][1]
        md.append(f"\n## {bname} (장면 {len(blk)})\n")
        md.append("| 모델 | 영역 | A 사라짐 | B 나타남 | C [95 %] | J | c |\n|---|---|---:|---:|---|---:|---:|")
        rows = {}
        for _, m0, _, m1 in PAIRS:
            for m in (m0, m1):
                for mode, ml in MODES:
                    S = load(bfile, m, mode)
                    if S is None:
                        continue
                    r = scene_stats(S, blk); s = summarize(r); rows[(m, mode)] = r
                    md.append(f"| `{m}` | {ml} | {s['A']:.1f} | {s['B']:.1f} | {s['C']:.1f} [{s['C_ci'][0]:.1f}, {s['C_ci'][1]:.1f}] | {s['J']:.1f} | {s['c']:+.2f} |")
        md.append("\n| 대조 → negimp | 영역 | A 고침 / 깨짐 (p) | B 고침 / 깨짐 (p) | C 고침 / 깨짐 (p) |\n|---|---|---|---|---|")
        for l0, m0, l1, m1 in PAIRS:
            for mode, ml in MODES:
                r0, r1 = rows.get((m0, mode)), rows.get((m1, mode))
                if not r0 or not r1:
                    continue
                d0 = {x[0]: x for x in r0}; d1 = {x[0]: x for x in r1}; ks = sorted(set(d0) & set(d1))
                def fb(fn):
                    f_ = sum(1 for k in ks if not fn(d0[k]) and fn(d1[k])); b_ = sum(1 for k in ks if fn(d0[k]) and not fn(d1[k]))
                    return f"{f_} / {b_} ({binom_p(f_, f_ + b_):.2g})"
                md.append(f"| {l0} → {l1} | {ml} | {fb(lambda x: x[1] > 0)} | {fb(lambda x: x[2] > 0)} | {fb(lambda x: x[1] + x[2] > 0)} |")
    open(f"{a.out}/REGION_SPLIT.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
