#!/usr/bin/env python3
"""BlenderTrack v3 ablation 채점표 — eval_v3.sh 산출물 (te/<패널>_<배치>/) 을 시트 trianing_ablation 의 칸 값으로 (2026-11-01).

판정 (기존 표와 같다, transfer_gap_r4_matrix.judge): 같은 (block_id, pair_id) 의 가능 · 불가능 두 영상에서
  s = 미래 8 블록 평균 L1, 정답 = s(불가능) > s(가능), 같으면 0.5. 정답률 % = 짝 평균 × 100. 복사 기준선도 같은 판정.
배치가 여럿이면 predictor 이름 (tag) 마다 처음 나온 배치 값을 쓴다 (복사 · 짝 목록은 배치끼리 같아야 한다 → 검사).
출력: predictor_lab/results/v3_ablation/table.json ({패널: {부분집합: {tag: [정답률, 짝 수]}}})
      predictor_lab/results/v3_ablation/cells.json (시트 칸 → 값; 표 1 · 2 의 H–Z 열)
  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python; $P predictor_lab/v3_ablation/table_v3.py
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
TE = REPO / "predictor_lab/results/v3_ablation/te"
OUT = REPO / "predictor_lab/results/v3_ablation"

# 패널마다 부분집합 (시트 열 순서). f(r) = 가능 쪽 index 행 r 로 판정
PANELS = {
    "vob_late": [("all", lambda r: r["violation_type"] == "permanence"),
                 ("primitive", lambda r: r["violation_type"] == "permanence" and r["objset"] == "primitive"),
                 ("realistic", lambda r: r["violation_type"] == "permanence" and r["objset"] == "realistic"),
                 ("A", lambda r: r["violation_type"] == "permanence" and r["pair_id"] == "A"),
                 ("B", lambda r: r["violation_type"] == "permanence" and r["pair_id"] == "B")],
    # 11-01 사용자: IntPhys 1 · 2 도 object permanence 만 (IntPhys 1 = O1 블록, IntPhys 2 = permanence 조건), CLEVRER 식은 static 뺌 (쉬워서 평균을 올림)
    "ip1uni": [("all", lambda r: r["block_type"] == "O1" and r["group"].endswith("/occluded")),
               ("moving", lambda r: r["block_type"] == "O1" and r["group"] == "moving/occluded"),
               ("static", lambda r: r["block_type"] == "O1" and r["group"] == "static/occluded")],
    "ip2uni": [("all", lambda r: r["condition"] == "permanence"),
               ("occluder", lambda r: r["condition"] == "permanence" and r["occluder"] not in ("", "None"))],
    "rubric": [("all", lambda r: True), ("occlusion", lambda r: r["condition"] == "occlusion"),
               ("containment", lambda r: r["condition"] == "containment")],
    "clevrer": [("all", lambda r: r["motion"] != "static")] + [(m, (lambda m_: lambda r: r["motion"] == m_)(m)) for m in ("behind", "cross", "overtake")],
    "vomo": [("all", lambda r: r["violation_type"] == "permanence"),
             ("A", lambda r: r["violation_type"] == "permanence" and r["pair_id"] == "A"),
             ("B", lambda r: r["violation_type"] == "permanence" and r["pair_id"] == "B")],
    "v3ttest": [("all", lambda r: True), ("A", lambda r: r["pair_id"] == "A"), ("B", lambda r: r["pair_id"] == "B"),
                ("flat", lambda r: r["place"] == "flat"), ("ramp", lambda r: r["place"] == "ramp"),
                ("tgt_moving", lambda r: r["target_mode"] == "moving"), ("tgt_static", lambda r: r["target_mode"] == "static"),
                ("front", lambda r: r["camera"] == "front"), ("side", lambda r: r["camera"] == "side")],
    "v3test": [("all", lambda r: True),
               ("primitive", lambda r: r["objset"] == "primitive"), ("realistic", lambda r: r["objset"] == "realistic"),
               ("front", lambda r: r["camera"] == "front"), ("side", lambda r: r["camera"] == "side"), ("bird", lambda r: r["camera"] == "bird"),
               ("barrier", lambda r: r["occ_type"] == "barrier"), ("object", lambda r: r["occ_type"] == "object"),
               ("occ_static", lambda r: r["occ_motion"] == "static"), ("occ_moving", lambda r: r["occ_motion"] == "moving"),
               ("tgt_moving", lambda r: r["target_mode"] == "moving"), ("tgt_static", lambda r: r["target_mode"] == "static"),
               ("flat", lambda r: r["place"] == "flat"), ("ramp", lambda r: r["place"] == "ramp"),
               ("short", lambda r: r["f_hide"] == "short"), ("long", lambda r: r["f_hide"] == "long"),
               ("vob_like", lambda r: r["camera"] == "front" and r["occ_type"] == "barrier" and r["occ_motion"] == "static"),
               ("A", lambda r: r["pair_id"] == "A"), ("B", lambda r: r["pair_id"] == "B")],
}
# 시트 표 1 (외부) 열 H… 순서 = (패널, 부분집합)
T1_COLS = [("vob_late", s) for s in ("all", "primitive", "realistic", "A", "B")] + [("ip1uni", s) for s in ("all", "moving", "static")] \
    + [("ip2uni", s) for s in ("all", "occluder")] + [("rubric", s) for s in ("all", "occlusion", "containment")] \
    + [("clevrer", s) for s in ("all", "behind", "cross", "overtake")]
T2_COLS = [("v3test", s) for s, _ in PANELS["v3test"]]
# tag → 표 1 행 (표 2 는 +42)
ROW = {"copy": 14, "release": 15, "vobftg": 16, "v3_mix1600_s0": 18, "v3_mix680_s0": 19, "v3_full_s0": 20,
       "v3_prim_s0": 22, "v3_real_s0": 23, "v3_front_s0": 25, "v3_side_s0": 26, "v3_bird_s0": 27,
       "v3_barrier_s0": 29, "v3_object_s0": 30, "v3_occstatic_s0": 32, "v3_occmoving_s0": 33,
       "v3_tgtmoving_s0": 35, "v3_tgtstatic_s0": 36, "v3_flat_s0": 38, "v3_ramp_s0": 39,
       "v3_short_s0": 41, "v3_long_s0": 42, "v3_aonly_s0": 44, "v3_frontstatic_s0": 46}


def col(i):
    return "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[i]


def panel_scores(pn):
    """배치들을 합쳐 tag → s (n,) 과 짝 목록."""
    # 만든 시각 순. 복사 기준선 · 짝 목록은 첫 배치 (vll1 b1) 것, predictor 값은 그 tag 가 나온 **마지막** 배치 것
    # (다시 채점한 배치가 이긴다). 노드가 다르면 (vll1 vs vll7 Blackwell) 수치가 상대 ~2e-4 달라 짝 몇 개가 뒤집힌다 → 경고만.
    dirs = sorted((d for d in TE.glob(f"{pn}_*") if (d / "done.npy").exists()), key=lambda d: (d / "meta.json").stat().st_mtime)
    S, vids, pairs = {}, None, None
    SRC.setdefault(pn, {})
    for d in dirs:
        if not np.load(d / "done.npy").all():
            print(f"  (덜 끝남, 건너뜀) {d.name}"); continue
        m = json.load(open(d / "meta.json"))
        l1 = np.load(d / "l1.npy").mean(-1); cp = np.load(d / "copy.npy").mean(-1)
        if vids is None:
            vids = list(m["video_id"]); S["copy"] = cp
            idx = {r["video_id"]: r for r in csv.DictReader(open(m["index_csv"]))}
            by = defaultdict(dict)
            for i, v in enumerate(vids):
                r = idx[v]; by[(r["block_id"], r["pair_id"])]["pos" if r["plausible"] == "1" else "imp"] = i
            pairs = [(v["pos"], v["imp"], idx[vids[v["pos"]]]) for v in by.values()]
        else:
            assert list(m["video_id"]) == vids, f"{d}: 영상 순서가 다른 배치와 다르다"
            if not np.allclose(cp, S["copy"], rtol=1e-5, atol=0):
                rel = np.abs(cp - S["copy"]) / np.abs(S["copy"])
                print(f"  ⚠️ {d.name}: 복사 기준선이 앞 배치와 상대 중앙 {np.median(rel):.1e} 다르다 (다른 노드 · 환경)")
            S["copy"] = cp; SRC[pn]["copy"] = d.name          # 복사도 마지막 배치 값 (최종 패스가 모든 값을 한 노드로 맞춘다)
        for k, t in enumerate(m["pred_tags"]):
            S[t] = l1[:, k]; SRC[pn][t] = d.name
    return S, pairs


SRC = {}   # 패널 → tag → 값을 가져온 배치 폴더


# 11-01 사용자: v3t 를 따로 두지 말고 v3 하나로 — v3 시험 (3,438 짝) + trapdoor 시험 (892 짝) 을 합친 "v3all" 패널.
# trapdoor 는 가림 종류의 세 번째 수준 (가림막 · 물체 · trapdoor). 두 패널을 다 채점한 predictor 만 낸다.
V3ALL = [("all", lambda r: True),
         ("primitive", lambda r: r["objset"] == "primitive"), ("realistic", lambda r: r["objset"] == "realistic"),
         ("front", lambda r: r["camera"] == "front"), ("side", lambda r: r["camera"] == "side"), ("bird", lambda r: r["camera"] == "bird"),
         ("barrier", lambda r: r["occ_type"] == "barrier" and r["occ_motion"] != "trapdoor"), ("object", lambda r: r["occ_type"] == "object"),
         ("trapdoor", lambda r: r["occ_motion"] == "trapdoor"),
         ("occ_static", lambda r: r["occ_motion"] == "static"), ("occ_moving", lambda r: r["occ_motion"] == "moving"),
         ("tgt_moving", lambda r: r["target_mode"] == "moving"), ("tgt_static", lambda r: r["target_mode"] == "static"),
         ("flat", lambda r: r["place"] == "flat"), ("ramp", lambda r: r["place"] == "ramp"),
         ("short", lambda r: r["f_hide"] == "short"), ("long", lambda r: r["f_hide"] == "long"),
         ("vob_form", lambda r: r["camera"] == "front" and r["occ_motion"] == "trapdoor"),
         ("A", lambda r: r["pair_id"] == "A"), ("B", lambda r: r["pair_id"] == "B")]


def _judge(s, sel):
    return np.array([1.0 if s[b] > s[a] else 0.5 if s[b] == s[a] else 0.0 for a, b in sel])


def main():
    table, data = {}, {}
    for pn, subs in PANELS.items():
        S, P = panel_scores(pn)
        if not P:
            print(f"{pn}: 결과 없음"); continue
        data[pn] = (S, P)
        table[pn] = {}
        for sname, f in subs:
            sel = [(a, b) for a, b, r in P if f(r)]
            table[pn][sname] = {t: [round(100 * float(_judge(s, sel).mean()), 2), len(sel)] for t, s in S.items()}
        print(f"{pn}: " + " · ".join(f"{t} {table[pn]['all'][t][0]:.1f}" for t in S) + f"  (짝 {table[pn]['all']['copy'][1]})")
    if "v3test" in data and "v3ttest" in data:
        (S1, P1), (S2, P2) = data["v3test"], data["v3ttest"]
        table["v3all"] = {}
        for sname, f in V3ALL:
            s1 = [(a, b) for a, b, r in P1 if f(r)]; s2 = [(a, b) for a, b, r in P2 if f(r)]
            table["v3all"][sname] = {}
            for t in set(S1) & set(S2):
                j = np.concatenate([_judge(S1[t], s1), _judge(S2[t], s2)])
                if len(j):
                    table["v3all"][sname][t] = [round(100 * float(j.mean()), 2), len(j)]
        print("v3all: " + " · ".join(f"{t} {v[0]:.1f}" for t, v in sorted(table["v3all"]["all"].items())) + f"  (짝 {table['v3all']['all']['copy'][1]})")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(table, open(OUT / "table.json", "w"), indent=1, ensure_ascii=False)
    json.dump(SRC, open(OUT / "sources.json", "w"), indent=1)
    print(f"→ {OUT}/table.json (시트 칸 배치는 build_sheet.py)")


if __name__ == "__main__":
    sys.exit(main())
