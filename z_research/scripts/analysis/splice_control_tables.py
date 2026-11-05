#!/usr/bin/env python3
"""splice 서명 대조 표 (2026-11-05, tak's_story 리뷰어 요청). GPU 불필요.
가설: CF (negimp) 손실이 '창 전체 정답' 에 보이는 이음 (raw 48) 의 흔적에서 멀어지는 법을 배웠을 수 있다.
검사: 릴리즈 계열을 **미래만 정답** (target encoder 가 미래 16 장 = raw 48–93 만 본다 → 이음 안 보임) 으로 다시 채점.
  AR 판은 채점 정답이 원래 블록별 (블록 = 자기 2 장, 첫 미래 블록 = raw 48 · 51) 이라 이미 이음을 안 본다 → 기존 채점 그대로.
  복사 기준선 = 문맥 16 장만 넣은 pass 의 마지막 tubelet 을 미래 8 tubelet 에 (미래만 정답 공간).
D = 장면 1[m_A + m_B > 0] (= scene_paired_tests 의 C), J = 두 짝 모두 정답, c = 치우침 (양수 = 빈 미래가 싸다).
물체 영역 = 미래 블록마다 물체 투영 중심 3×3 (region_split_tests 와 같은 정의).
  /venv/vjepa/bin/python z_research/scripts/analysis/splice_control_tables.py > z_analysis/results/scene_paired_20261105/SPLICE_CONTROL_FUTONLY.md
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from scene_paired_tests import boards, scene_stats, summarize, binom_p, RES

T = f"{RES}/testbed"
SETS = [("LateOcc-IntPhys (Surrogate)", "IntPhys1 Surrogate", "intphys1_surrogate", True),
        ("LateOcc-Syn panel (VOB v3)", "VOBench v3 late · panel", "vobench_v3", True),
        ("LateOcc-Syn mover (VOB v3)", "VOBench v3 late · moving occluder", "vobench_v3", True),
        ("LateOcc-Real (95)", "real world (all 95)", "real_world_permanence_uniform", False)]
# (행 이름, 모델, 결과 suffix, key) — key 는 per_video_surprise 가 dict 일 때
ROWS = [("released (untouched) · future-only", "vith", "_futonly", ""),
        ("released + L1 2k · future-only", "vith_release_v4vc", "_futonly", ""),
        ("released + CF 2k · future-only", "vith_release_v4vc_negimp", "_futonly", ""),
        ("copy baseline · future-only", "vith", "_futonly", "copy_"),
        ("released + L1 2k · full-window (ref)", "vith_release_v4vc", "", ""),
        ("released + CF 2k · full-window (ref)", "vith_release_v4vc_negimp", "", ""),
        ("AR + L1 2k · block target (splice-blind)", "vast_ar_vw_state_v4vc", "", ""),
        ("AR + CF 2k · block target (splice-blind)", "vast_ar_vw_state_v4vc_negimp", "", "")]
CONTRASTS = [("released L1 → CF (future-only)", 1, 2), ("released L1 → CF (full-window ref)", 4, 5), ("AR L1 → CF (block target)", 6, 7)]


def load(board, m, suffix, pre, mode):
    # full-window 물체 영역은 _region 실행에, future-only 는 _futonly 실행에 같이 있다
    cands = [f"{T}/{board}_{m}{suffix}/per_block.json"] + ([f"{T}/{board}_{m}_region/per_block.json"] if suffix == "" else [])
    for f in cands:
        if not os.path.exists(f):
            continue
        S = json.load(open(f))["per_video_surprise"]
        v0 = next(iter(S.values()))
        if not isinstance(v0, dict):
            if mode == "all" and pre == "":
                return S
            continue
        k = pre + mode
        if k in v0:
            return {vid: v[k] for vid, v in S.items()}
    return None


def moving_A(rows_by_scene, blk_moving):
    sel = [r for r in rows_by_scene if r[0] in blk_moving]
    return 100 * sum(1.0 if r[1] > 0 else 0.5 if r[1] == 0 else 0.0 for r in sel) / len(sel), len(sel)


def main():
    import csv
    B = boards(); D = f"{os.path.dirname(RES)}/../../data_csv"
    mov = {}
    for _, _, board, _ in SETS:
        if board in mov:
            continue
        idx = f"/workspace/world_model/vjepa2/data_csv/{board}/index.csv"
        mov[board] = {r["block_id"] for r in csv.DictReader(open(idx)) if r.get("motion") not in ("flat_static",)}
    print("# Splice-signature control: future-only target (released family) · block target (AR)\n")
    print(__doc__.split("\n  /venv")[0].replace('"""', ""))
    for sname, bname, board, has_reg in SETS:
        blk = B[bname][1]
        print(f"\n## {sname} — {len(blk)} scenes\n")
        print("| row | A vanish | B appear | c | **D** [95 %] | J | A moving | obj-region A moving | obj-region D |\n|---|---:|---:|---:|---|---:|---:|---:|---:|")
        cache = {}
        for i, (lab, m, suf, pre) in enumerate(ROWS):
            S = load(board, m, suf, pre, "all")
            if S is None:
                print(f"| {lab} | — |"); continue
            r = scene_stats(S, blk); s = summarize(r); cache[(i, "all")] = r
            am, nm = moving_A(r, mov[board])
            oa = od = "—"
            if has_reg:
                So = load(board, m, suf, pre, "object")
                if So is not None:
                    ro = scene_stats(So, blk); so = summarize(ro); cache[(i, "object")] = ro
                    oa = f"{moving_A(ro, mov[board])[0]:.1f}"; od = f"{so['C']:.1f}"
            print(f"| {lab} | {s['A']:.1f} | {s['B']:.1f} | {s['c']:+.2f} | **{s['C']:.1f}** [{s['C_ci'][0]:.1f}, {s['C_ci'][1]:.1f}] | {s['J']:.1f} | {am:.1f} (n {nm}) | {oa} | {od} |")
        print("\n| control → CF | A fix/brk (p) | D fix/brk (p) | obj-region A fix/brk (p) | obj-region D fix/brk (p) |\n|---|---|---|---|---|")
        for lab, i0, i1 in CONTRASTS:
            out = [lab]
            for mode in ("all", "object"):
                r0, r1 = cache.get((i0, mode)), cache.get((i1, mode))
                if not r0 or not r1:
                    out += ["—", "—"]; continue
                d0 = {x[0]: x for x in r0}; d1 = {x[0]: x for x in r1}; ks = sorted(set(d0) & set(d1))
                for fn in (lambda x: x[1] > 0, lambda x: x[1] + x[2] > 0):
                    f_ = sum(1 for k in ks if not fn(d0[k]) and fn(d1[k])); b_ = sum(1 for k in ks if fn(d0[k]) and not fn(d1[k]))
                    out.append(f"{f_}/{b_} ({binom_p(f_, f_ + b_):.2g})")
            print("| " + " | ".join(out) + " |")


if __name__ == "__main__":
    main()
