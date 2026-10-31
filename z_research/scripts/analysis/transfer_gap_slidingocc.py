#!/usr/bin/env python3
"""transfer_gap Q3 · E30 — IntPhys 1 Garrido sliding 창 가운데 '문맥 끝이 가림 안' 인 창만 골라 릴리즈 vs VOBench PFT 를 비교한다. 2026-10-30, [X2] R2.

왜: VOBench PFT 의 핵심 효과는 문맥 끝에 물체가 가려진 창에서 생긴다 (VOBench late, IntPhys 2 라벨 −10 판). 표준 IntPhys 1 채점
    (sliding · 창 평균) 이 그런 창을 거의 안 뽑으면 효과가 희석된다 (H3-E). 그런 창에서만 이득이 있으면 H3-W.
정의 (CPU, 기존 per_window.json 만 읽는다)
  창 w = (조합 skipS_wW, 시작 원프레임 st, 문맥 길이 C). 문맥 끝 원프레임 e = st + S·(C − 1), 첫 미래 원프레임 f = st + S·C.
  가림 구간 = OcclusionStats gaps.csv (가능 영상 마스크에서 잰, 물체가 보이다가 가려졌다 다시 보이는 원프레임 [start, end]).
  가림 창 (occ_end) = e ∈ [start, end] 이고 end ≥ f (가림이 첫 미래 샘플까지 이어짐).  느슨판 (occ_end_loose) = e ∈ [start, end] 이고 end > e.
  미래 가림 창 (occ_future) = 가림 창이 아니면서 가림 구간이 미래 샘플 하나 이상을 덮음.  나머지 = none.
  쌍 = index 의 (block_id, pair_id) 가능 · 불가능. 분류는 가능 영상의 가림 구간으로 (불가능 영상은 같은 장면 · 문맥).
  창 단위 판정 = s_w(imp) > s_w(pos).  창 정확도 = 그 분류의 (쌍, 창) 평균.  이득 G = PFT − 릴리즈 (같은 쌍 · 같은 창).
  Garrido 집계 (영상 점수 = 시작점 t 마다 C 최소 → t 평균) 를 분류 안 창만으로 다시 한 판도 낸다 (그 분류의 창이 있는 쌍만).
  비율 = 전체 창 중 가림 창 수 · 시작점 t 중 'C 최소가 가림 창' 인 비율 (= Garrido 집계에서 가림 창이 차지하는 가중).
  CI = 장면 (사분 묶음, 예 O1_01) bootstrap 4,000.
출력: z_analysis/transfer_gap/results/q3_r2/E30_slidingocc.json

  /data/hyuntak/anaconda3/envs/vjepa2/bin/python z_research/scripts/analysis/transfer_gap_slidingocc.py
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
RES = REPO / "cvpr/results/eval/intphys1"
OUT = REPO / "z_analysis/transfer_gap/results/q3_r2/E30_slidingocc.json"
IDX = Path("/local_datasets/world/world_analysis/IntPhys1_dev_videos/index.csv")
OCC = REPO / "z_research/previous/OcclusionStats/intphys1"
MODELS = {"release": "intphys1_dev_vith_wgarrido_w{w}", "vob_pft": "intphys1_dev_vith_release_vobftg_wgarrido_w{w}"}
NB = 4000


def main():
    rng = np.random.default_rng(0)
    rows = list(csv.DictReader(open(IDX)))
    by = defaultdict(dict)
    for r in rows:
        by[(r["block_id"], r["pair_id"])]["pos" if r["plausible"] == "1" else "imp"] = r
    pairs = [(v["pos"]["video_id"], v["imp"]["video_id"], v["pos"]["block"] + "_" + v["pos"]["quadruplet"])
             for v in by.values() if "pos" in v and "imp" in v]
    gaps = defaultdict(list)
    for g in csv.DictReader(open(OCC / "gaps.csv")):
        gaps[g["video"]].append((int(g["start"]), int(g["end"]), g["event_gap"] == "1"))
    vids = json.load(open(OCC / "videos.json"))
    group = {v: vids[v]["group"] for v in vids}

    def classify(pos, combo, st, C, loose=False):
        S = int(combo.split("_")[0][4:]); W = int(combo.split("_w")[1])
        e, f, last = st + S * (C - 1), st + S * C, st + S * (W - 1)
        for a, b, ev in gaps.get(pos, []):
            if a <= e <= b and (b > e if loose else b >= f):
                return "occ_end_event" if ev else "occ_end_other"
        for a, b, ev in gaps.get(pos, []):
            fut = range(f, last + 1, S)
            if any(a <= x <= b for x in fut):
                return "occ_future"
        return "none"

    res = {"definition": __doc__.split("정의")[1].split("출력:")[0].strip(), "n_pairs": len(pairs), "combos": {}}
    for w in (16, 32):
        PW = {m: json.load(open(RES / (p.format(w=w)) / "per_window.json"))["windows"] for m, p in MODELS.items()}
        combos = sorted(set(x[0] for x in next(iter(PW["release"].values()))))
        for combo in combos:
            for loose in (False, True):
                key = f"{combo}{'_loose' if loose else ''}"
                recs = []   # (pair_idx, scene, cls, st, C, judge_rel, judge_pft)
                sw = {m: {} for m in PW}
                for m in PW:
                    for v, lst in PW[m].items():
                        for c_, st, C, s in lst:
                            if c_ == combo:
                                sw[m][(v, st, C)] = s
                wins = sorted({(st, C) for (v, st, C) in sw["release"] if v == pairs[0][0]})
                for pi, (pos, imp, scene) in enumerate(pairs):
                    for st, C in wins:
                        k1, k2 = (pos, st, C), (imp, st, C)
                        if not all(k in sw[m] for m in sw for k in (k1, k2)):
                            continue
                        cls = classify(pos, combo, st, C, loose)
                        j = {m: (1.0 if sw[m][k2] > sw[m][k1] else 0.5 if sw[m][k2] == sw[m][k1] else 0.0) for m in sw}
                        recs.append((pi, scene, cls, st, C, j["release"], j["vob_pft"], group.get(pos, "?")))
                if not recs:
                    continue
                A = np.array([(r[5], r[6]) for r in recs])
                cls = np.array([r[2] for r in recs]); scn = np.array([r[1] for r in recs]); grp = np.array([r[7] for r in recs])
                out = {"n_windows": len(recs), "class_share": {}, "window_acc": {}, "garrido_within_class": {}}
                for c in ("occ_end_event", "occ_end_other", "occ_future", "none"):
                    out["class_share"][c] = round(float((cls == c).mean()), 4)
                cats = {"occ_end": np.isin(cls, ["occ_end_event", "occ_end_other"]), "occ_end_event": cls == "occ_end_event",
                        "occ_future": cls == "occ_future", "none": cls == "none", "not_occ_end": ~np.isin(cls, ["occ_end_event", "occ_end_other"])}
                for g in ("moving/occluded", "static/occluded"):
                    cats[f"occ_end|{g}"] = cats["occ_end"] & (grp == g)
                    cats[f"not_occ_end|{g}"] = cats["not_occ_end"] & (grp == g)
                for c, m in cats.items():
                    if m.sum() == 0:
                        out["window_acc"][c] = None
                        continue
                    d = A[m, 1] - A[m, 0]
                    sc = scn[m]
                    u = np.unique(sc); ix = {s_: np.flatnonzero(sc == s_) for s_ in u}
                    bs = [d[np.concatenate([ix[u[k]] for k in rng.choice(len(u), len(u))])].mean() for _ in range(NB)]
                    out["window_acc"][c] = {"n_windows": int(m.sum()), "n_pairs": int(len(set(np.array([r[0] for r in recs])[m]))),
                                            "release": round(100 * float(A[m, 0].mean()), 2), "vob_pft": round(100 * float(A[m, 1].mean()), 2),
                                            "G": round(100 * float(d.mean()), 2),
                                            "G_ci": [round(100 * float(np.percentile(bs, q)), 2) for q in (2.5, 97.5)]}
                # Garrido 집계를 분류 안에서: 영상 점수 = mean_t min_C s (그 분류 창만)
                for c, cset in (("occ_end", {"occ_end_event", "occ_end_other"}), ("not_occ_end", {"occ_future", "none"}), ("all", None)):
                    hits = {m: [] for m in sw}; scs = []
                    for pi, (pos, imp, scene) in enumerate(pairs):
                        keep = [(st, C) for st, C in wins if cset is None or classify(pos, combo, st, C, loose) in cset]
                        if not keep:
                            continue
                        ok = True
                        for m in sw:
                            sv = []
                            for v in (pos, imp):
                                per_t = defaultdict(list)
                                for st, C in keep:
                                    if (v, st, C) in sw[m]:
                                        per_t[st].append(sw[m][(v, st, C)])
                                if not per_t:
                                    ok = False
                                    break
                                sv.append(np.mean([min(x) for x in per_t.values()]))
                            if not ok:
                                break
                            hits[m].append(1.0 if sv[1] > sv[0] else 0.5 if sv[1] == sv[0] else 0.0)
                        if ok:
                            scs.append(scene)
                    if scs:
                        d = np.array(hits["vob_pft"]) - np.array(hits["release"]); sc = np.array(scs)
                        u = np.unique(sc); ix = {s_: np.flatnonzero(sc == s_) for s_ in u}
                        bs = [d[np.concatenate([ix[u[k]] for k in rng.choice(len(u), len(u))])].mean() for _ in range(NB)]
                        out["garrido_within_class"][c] = {"n_pairs": len(scs), "release": round(100 * float(np.mean(hits["release"])), 2),
                                                          "vob_pft": round(100 * float(np.mean(hits["vob_pft"])), 2), "G": round(100 * float(d.mean()), 2),
                                                          "G_ci": [round(100 * float(np.percentile(bs, q)), 2) for q in (2.5, 97.5)]}
                # 가중: 시작점 t 중 C 최소 창이 가림 창인 비율 (릴리즈 · 가능 영상 기준)
                wt, nt = 0, 0
                for pos, imp, scene in pairs:
                    per_t = defaultdict(list)
                    for st, C in wins:
                        if (pos, st, C) in sw["release"]:
                            per_t[st].append((sw["release"][(pos, st, C)], C))
                    for st, lst in per_t.items():
                        C_min = min(lst)[1]
                        nt += 1
                        wt += classify(pos, combo, st, C_min, loose).startswith("occ_end")
                out["start_share_min_is_occ_end"] = round(wt / max(nt, 1), 4)
                res["combos"][key] = out
                print(key, out["class_share"], {c: (v["G"], v["G_ci"], v["n_windows"]) for c, v in out["window_acc"].items() if v and c in ("occ_end", "not_occ_end", "occ_end|moving/occluded", "occ_end|static/occluded")},
                      out["garrido_within_class"], out["start_share_min_is_occ_end"], flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
    print("→", OUT)


if __name__ == "__main__":
    main()
