#!/usr/bin/env python3
"""v3 축 전이 표 — 사라짐 (A 짝) 기준 (2026-11-02, 사용자 요청). 행 = 한 축 수준만으로 학습한 팔, 열 = v3 시험의 그 축 수준.

판정 = table_v3 와 같다 (s = 미래 8 블록 평균 L1, 정답 = s(사라짐) > s(가능), 같으면 0.5). 물체 움직임 (움직임 · 정지) 으로 나눠 낸다.
전이율 = (학습 팔 − release) / (시험 수준과 같은 수준으로 학습한 팔 − release), 움직이는 물체 기준.
출력 predictor_lab/results/v3_ablation/sheet_axis/axis_transfer_vanish.json
  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python; $P predictor_lab/v3_ablation/axis_transfer_vanish.py
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import table_v3 as T  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
AXES = [  # (축 이름, 시험 열, [(수준 이름, 팔, 열 값)])
    ("물체 종류", "objset", [("primitive", "v3_prim_s0", "primitive"), ("realistic", "v3_real_s0", "realistic")]),
    ("카메라", "camera", [("정면", "v3_front_s0", "front"), ("측면", "v3_side_s0", "side"), ("위 (bird-eye)", "v3_bird_s0", "bird")]),
    ("가림 종류", "occ_type", [("가림막", "v3_barrier_s0", "barrier"), ("다른 물체", "v3_object_s0", "object")]),
    ("가림 움직임", "occ_motion", [("서 있음", "v3_occstatic_s0", "static"), ("움직임", "v3_occmoving_s0", "moving")]),
    ("장소", "place", [("평면", "v3_flat_s0", "flat"), ("빗면", "v3_ramp_s0", "ramp")]),
    ("가림 길이", "f_hide", [("짧게", "v3_short_s0", "short"), ("길게", "v3_long_s0", "long")]),
    ("물체 움직임", "target_mode", [("움직임", "v3_tgtmoving_s0", "moving"), ("정지", "v3_tgtstatic_s0", "static")]),
]
EXTRA = [("원래 (release, 학습 전)", "release"), ("복사 (예측 없음)", "copy"), ("모든 수준 섞음 (같은 장면 수 1,600)", "v3_mix1600_s0")]


def main():
    S, P = T.panel_scores("v3test")
    out = {"def": __doc__.split("\n\n")[1], "axes": []}
    for name, colname, lv in AXES:
        subsets = ["all"] if colname == "target_mode" else ["moving", "static"]
        blk = {"axis": name, "levels": [l[0] for l in lv], "subsets": subsets, "rows": []}
        n = {}
        for sub in subsets:
            for lname, _, val in lv:
                sel = [(a, b) for a, b, r in P if r["pair_id"] == "A" and r[colname] == val and (sub == "all" or r["target_mode"] == sub)]
                n[(sub, lname)] = sel
        rows = [(l[0] + "만", l[1]) for l in lv] + EXTRA
        acc = {}
        for rname, arm in rows:
            for sub in subsets:
                for lname, _, _ in lv:
                    sel = n[(sub, lname)]
                    acc[(arm, sub, lname)] = round(100 * float(T._judge(S[arm], sel).mean()), 1) if sel else None
        for rname, arm in rows:
            row = {"name": rname, "arm": arm, "acc": {f"{sub}|{l}": acc[(arm, sub, l)] for sub in subsets for l in blk["levels"]}}
            if arm not in ("release", "copy"):
                tr = {}
                for lname, larm, _ in lv:
                    sub = subsets[0] if colname == "target_mode" else "moving"
                    base, own, me = acc[("release", sub, lname)], acc[(larm, sub, lname)], acc[(arm, sub, lname)]
                    tr[lname] = None if None in (base, own, me) or own - base < 5 else round(100 * (me - base) / (own - base))
                row["transfer"] = tr
            blk["rows"].append(row)
        blk["n"] = {f"{sub}|{l}": len(v) for (sub, l), v in n.items()}
        out["axes"].append(blk)
    p = REPO / "predictor_lab/results/v3_ablation/sheet_axis/axis_transfer_vanish.json"
    json.dump(out, open(p, "w"), ensure_ascii=False, indent=1)
    for b in out["axes"]:
        print("==", b["axis"], b["n"])
        for r in b["rows"]:
            print(f"  {r['name']:28s}", r["acc"], r.get("transfer", ""))


if __name__ == "__main__":
    main()
