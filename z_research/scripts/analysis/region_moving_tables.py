#!/usr/bin/env python3
"""움직임 장면만, 영역별 (물체 3×3 · 먼 배경) A · B · D · c 와 McNemar — 같은 장면 집합에서 (2026-11-05, tak's_story 리뷰어 요청). GPU 불필요.
정정: splice_control_tables.py 의 고침 / 깨짐은 **보드 전체 장면** (움직임 + 정지) 에서 셌다. 정답률 칸 'A moving' 은 움직임 장면만이다. 이 표는 둘 다 움직임 장면.
공간: FO = 미래만 정답 (*_futonly), FW = 창 전체 정답 (*_region; 복사는 Surrogate 만 *_copychk 에 있다).
복사 기준선은 predictor 를 안 쓰므로 체크포인트와 무관하다 — 세 판의 copy_* 가 같은지 확인해서 적는다.
  /venv/vjepa/bin/python z_research/scripts/analysis/region_moving_tables.py > z_analysis/results/scene_paired_20261105/REGION_MOVING_BOTHDIR.md
"""
import csv, json, os, random, sys
sys.path.insert(0, os.path.dirname(__file__))
from scene_paired_tests import boards, scene_stats, summarize, binom_p, RES, Z

T = f"{RES}/testbed"
SETS = [("LateOcc-IntPhys moving (Surrogate ip1_throw)", "IntPhys1 Surrogate", "intphys1_surrogate"),
        ("LateOcc-Syn panel moving (VOB v3)", "VOBench v3 late · panel", "vobench_v3"),
        ("LateOcc-Syn mover moving (VOB v3)", "VOBench v3 late · moving occluder", "vobench_v3")]
L1, CF = "vith_release_v4vc", "vith_release_v4vc_negimp"


def pv(board, m, suffix):
    f = f"{T}/{board}_{m}{suffix}/per_block.json"
    return json.load(open(f))["per_video_surprise"] if os.path.exists(f) else None


def sel(S, key):
    return None if S is None else {k: v[key] for k, v in S.items()}


def fb(r0, r1, fn):
    d0 = {x[0]: x for x in r0}; d1 = {x[0]: x for x in r1}; ks = sorted(set(d0) & set(d1))
    f_ = sum(1 for k in ks if not fn(d0[k]) and fn(d1[k])); b_ = sum(1 for k in ks if fn(d0[k]) and not fn(d1[k]))
    return f"{f_}/{b_} ({binom_p(f_, f_ + b_):.2g})", f_ - b_


def main():
    B = boards()
    print("# 움직임 장면 · 영역별 양방향 (released L1 2k vs + CF) — 같은 장면 집합\n")
    print(__doc__.split("  /venv")[0].replace('"""', ""))
    # 복사 동일성 확인 (FO)
    for board in ("intphys1_surrogate", "vobench_v3", "real_world_permanence_uniform"):
        Ss = [pv(board, m, "_futonly") for m in ("vith", L1, CF)]
        if all(Ss):
            d = max(abs(Ss[0][k]["copy_all"] - s[k]["copy_all"]) for s in Ss[1:] for k in Ss[0])
            print(f"- copy 동일성 ({board}, FO): 세 체크포인트 사이 copy_all 최대 차 {d:.2e}")
    for sname, bname, board in SETS:
        blk_all = B[bname][1]
        idx = list(csv.DictReader(open(f"/workspace/world_model/vjepa2/data_csv/{board}/index.csv")))
        movb = {r["block_id"] for r in idx if r["motion"] != "flat_static"}
        blk = {b: v for b, v in blk_all.items() if b in movb}
        print(f"\n## {sname} — {len(blk)} scenes (A pairs {len(blk)}, B pairs {len(blk)})\n")
        print("| space | region | row | A | B | D | J | c |\n|---|---|---|---:|---:|---:|---:|---:|")
        res = {}
        spaces = [("FO", "_futonly", "_futonly", "vith"), ("FW", "_region", "_copychk", "vith")]
        for sp, suf, csuf, cm in spaces:
            for reg in ("object", "far_bg", "all"):
                rows = [("L1", sel(pv(board, L1, suf), reg)), ("CF", sel(pv(board, CF, suf), reg)),
                        ("copy", sel(pv(board, cm, csuf), "copy_" + reg))]
                for lab, S in rows:
                    if S is None:
                        print(f"| {sp} | {reg} | {lab} | — (없음) | | | | |"); continue
                    r = scene_stats(S, blk); s = summarize(r); res[(sp, reg, lab)] = r
                    print(f"| {sp} | {reg} | {lab} | {s['A']:.1f} | {s['B']:.1f} | {s['C']:.1f} | {s['J']:.1f} | {s['c']:+.2f} |")
        print("\n| space | region | L1 → CF: A fix/brk (p) | net A pts | B fix/brk (p) | D fix/brk (p) |\n|---|---|---|---:|---|---|")
        n = len(blk)
        for sp in ("FO", "FW"):
            for reg in ("object", "far_bg", "all"):
                r0, r1 = res.get((sp, reg, "L1")), res.get((sp, reg, "CF"))
                if not r0 or not r1:
                    continue
                a, na = fb(r0, r1, lambda x: x[1] > 0); b, _ = fb(r0, r1, lambda x: x[2] > 0); d, _ = fb(r0, r1, lambda x: x[1] + x[2] > 0)
                print(f"| {sp} | {reg} | {a} | {100 * na / n:+.1f} | {b} | {d} |")
    # 3) real: c 와 복사 c
    print("\n## LateOcc-Real (95, 전체 토큰): c 와 복사 c\n")
    pat, blk = B["real world (all 95)"]
    rows = {}
    for sp, suf in (("FW", ""), ("FO", "_futonly")):
        for lab, m in (("L1", L1), ("CF", CF)):
            S = pv("real_world_permanence_uniform", m, suf)
            if S is None:
                continue
            S = {k: (v["all"] if isinstance(v, dict) else v) for k, v in S.items()}
            rows[(sp, lab)] = sorted(scene_stats(S, blk))
        S = pv("real_world_permanence_uniform", "vith", suf) if sp == "FO" else None
        if S is not None:
            rows[(sp, "copy")] = sorted(scene_stats({k: v["copy_all"] for k, v in S.items()}, blk))
    def crit(r, ix):
        nn = len(ix); A = sum(1.0 if r[i][1] > 0 else 0.5 if r[i][1] == 0 else 0.0 for i in ix); Bb = sum(1.0 if r[i][2] > 0 else 0.5 if r[i][2] == 0 else 0.0 for i in ix)
        return -(Z((A + 0.5) / (nn + 1)) + Z((nn - Bb + 0.5) / (nn + 1))) / 2
    print("| space | c L1 | c CF | c copy | c L1 − c copy | c CF − c copy | shift Δc (paired bootstrap 95 %) |\n|---|---:|---:|---:|---:|---:|---|")
    for sp in ("FW", "FO"):
        r0, r1, rc = rows.get((sp, "L1")), rows.get((sp, "CF")), rows.get((sp, "copy"))
        n = len(r0); full = list(range(n)); rnd = random.Random(0); dd = []
        for _ in range(10000):
            ix = [rnd.randrange(n) for _ in range(n)]
            dd.append(crit(r1, ix) - crit(r0, ix))
        dd.sort(); p = 2 * min(sum(1 for x in dd if x <= 0), sum(1 for x in dd if x >= 0)) / 10000
        cc = crit(rc, full) if rc else None
        cs = f"{cc:+.2f}" if cc is not None else "— (FW 복사 미채점)"
        print(f"| {sp} | {crit(r0, full):+.2f} | {crit(r1, full):+.2f} | {cs} | "
              f"{(crit(r0, full) - cc):+.2f} | {(crit(r1, full) - cc):+.2f} | {sum(dd) / 1e4:+.2f} [{dd[250]:+.2f}, {dd[9750]:+.2f}], p {min(1, p):.4f} |"
              if cc is not None else
              f"| {sp} | {crit(r0, full):+.2f} | {crit(r1, full):+.2f} | {cs} | — | — | {sum(dd) / 1e4:+.2f} [{dd[250]:+.2f}, {dd[9750]:+.2f}], p {min(1, p):.4f} |")


if __name__ == "__main__":
    main()
