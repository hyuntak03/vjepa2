#!/usr/bin/env python3
"""IntPhys 1 dev — 사건 물체의 가림 통계 (얼마나 오래 · 가려진 동안 움직이나 · 얼마나 멀리) 와 V-JEPA 2 정답의 관계.

사건 물체 추적 (파라미터 1 개 = 잡음 문턱 T 픽셀):
  한 scene (4 중항) 의 possible 영상 두 개 A, B 는 사건 물체만 다르다
  (O1: 물체 있음 / 없음 · O2: 같은 궤적의 다른 모양 · O3: 같은 물체의 다른 타이밍).
  프레임 f 에서 두 mask 를 픽셀 단위로 맞댄다. mask 값은 프레임마다 무작위로 다시 매겨지므로, (A 라벨, B 라벨) 조합 중
  **서로 최다 겹침인 쌍** (mutual best) 만 '같은 조각' 으로 보고 나머지 픽셀을 차이 영역 D_f 로 둔다.
  원리마다 두 영상의 관계가 달라 영상별 물체 픽셀을 이렇게 정한다 (IntPhys 1 설계):
    O1 (있음 / 없음)          D 전체가 '물체 있는 영상' 의 물체. 없는 영상 = 영상 전체에서 짝 없는 조각 픽셀이 적은 쪽 → 가시성 0
    O2 (같은 궤적 · 다른 모양)  D 전체 (두 모양의 대칭차) 를 두 영상 공통의 가시성으로 쓴다
    O3 (같은 물체 · 다른 타이밍) 픽셀마다 짝 (mutual best) 이 없는 조각 쪽의 물체로 나눈다 (둘 다 짝이 있거나 없으면 작은 조각 쪽)
    보임_X(f) = |X 의 물체 픽셀| ≥ T      (T 기본 20 px = 288² 의 0.024 %, 민감도 1 · 50 px 병기)
    위치_X(f) = 그 픽셀의 무게중심 (px, 288×288 원본)       크기_X(f) = 그 픽셀이 가장 많이 속한 조각의 면적 (A·B 중 작은 쪽) → 폭 = √면적
  ⚠️ 2026-09-26 정정 (두 번) —
     v1: D 전체를 모든 원리에서 한 물체로 봤다 → O3 는 두 영상의 물체 타이밍이 달라 '둘 다 가려진 순간' 만 가림으로 잡혀
         가짜 1 프레임 가림 5 개 (O3_04 · 09 · 14 · 20 · 29).
     v2: 모든 원리에서 '작은 조각 쪽' 으로 영상별 분리 → O2 는 비슷한 두 모양의 대칭차가 반으로 쪼개져 20 px 아래로 떨어져
         보이는 물체를 가림으로 잡았다 (이동+보임에서 1–1.6 초 가짜 가림 4 개). O1 은 사건 물체가 다른 물체를 가리는 프레임에서
         없는 영상에 가짜 가시성이 생겼다 (O1_04_4 등).  → 지금 판 (원리별). v1 · v2 산출물은 intphys1/_superseded/.
가림 (gap) = 한 영상에서 '보임' 사이에 끼인 '안 보임' 연속 구간 (앞 = 등장 전, 뒤 = 퇴장 뒤는 가림이 아니다).
사건 가림 = **그 쌍의 possible 영상** 에서 pos/imp 첫 픽셀 차 d
  (0-기준; PNG 직접 비교 = pairs.csv first_div_sensitive − 1) 를 포함하거나, d 가 그 구간 끝 뒤 LAG=3 프레임 안인 가림
  (위반은 가림 중에 일어나고 픽셀은 재등장 순간에 갈리는 경우).
  d 전에 물체가 그 영상에서 한 번도 안 보였으면 'not_seen_before_d' — 모델이 context 에서 물체를 본 적이 없어 운반할 것이 없다
  (O1 의 '없음 영상 vs 나타남' 쌍, O3 의 가림막 뒤에서 처음 나오는 쌍).
가림마다: 길이 (raw 프레임 · 초, 15 fps) · skip 2 / skip 5 격자에 드는 프레임 수 · 가림 앞뒤 위치 변위 (px, 물체 폭 단위) ·
  가림 직전 속도 (마지막 3 보임 프레임의 평균 |Δ위치|).
V-JEPA 2 정답: 공식 88.89 실행 (skip2_w32) 의 창별 surprise → Filtered (시작점마다 min over C → 평균) 쌍 비교.
  창 (skip2_w32) 마다 context 마지막 프레임에서 물체가 가려져 있었는지도 센다 (v11 의 '경계 가림' 과 대조).
통계 단위: 사건 가림은 **고유한 (영상, 가림)** 하나씩 (O1 의 두 쌍은 같은 영상의 같은 가림을 공유한다). 정답만 쌍 단위.

  python z_research/scripts/analysis/occlusion_intphys1.py            (CPU 약 3 분, mask 1.8 만 장)
  → z_research/OcclusionStats/intphys1/{summary.json, videos.json, gaps.csv, pairs.csv}
"""
from __future__ import annotations
import collections, csv, json, sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from evals.world_model_analysis.eval import _intphys1_windows  # noqa: E402

AUX = ROOT / "auto_research/_stage/IntPhys1_dev_by_scene"
FR = Path("/local_datasets/world/world_analysis/IntPhys1_dev_frame_png")
PER_WIN = ROOT / "z_research/Benchmarks/exp_results/intphys1_sliding__intphys1_dev_vith_w32/per_window.json"
OUT = ROOT / "z_research/OcclusionStats/intphys1"
FPS, NF, LAG, T_DEF, T_SENS = 15.0, 100, 3, 20, (1, 20, 50)
MO = {"정지": "static", "이동": "moving"}; VI = {"눈앞": "visible", "가려짐": "occluded"}


def mask(vid, f):
    b, q, r = vid.split("_")
    return np.asarray(Image.open(FR / b / q / r / "masks" / f"masks_{f + 1:03d}.png")).astype(np.int32)


def split_diff(a, b):
    """→ (D, A 의 물체 픽셀, B 의 물체 픽셀). 영상별 분리는 짝 없는 조각 우선, 그다음 작은 조각."""
    code = a * 256 + b
    u, inv, c = np.unique(code.ravel(), return_inverse=True, return_counts=True)
    la, lb = u // 256, u % 256
    best_a, best_b = {}, {}
    for i in range(len(u)):
        if la[i] not in best_a or c[i] > c[best_a[la[i]]]:
            best_a[la[i]] = i
        if lb[i] not in best_b or c[i] > c[best_b[lb[i]]]:
            best_b[lb[i]] = i
    matched = np.array([best_a[la[i]] == i and best_b[lb[i]] == i for i in range(len(u))])
    D = ~matched[inv].reshape(a.shape)
    mat_a = np.zeros(256, bool); mat_b = np.zeros(256, bool)
    mat_a[la[matched]] = True; mat_b[lb[matched]] = True
    ua, ub = ~mat_a[a], ~mat_b[b]                                  # 픽셀의 조각이 짝이 없나
    area_a = np.bincount(a.ravel(), minlength=256)[a]; area_b = np.bincount(b.ravel(), minlength=256)[b]
    to_a = D & ((ua & ~ub) | ((ua == ub) & (area_a < area_b)))
    to_b = D & ((ub & ~ua) | ((ua == ub) & (area_b < area_a)))
    return D, to_a, to_b


def _put(t, o, a, b):
    k = int(o.sum()); t["px"].append(k)
    if k:
        ys, xs = np.nonzero(o); t["cx"].append(round(float(xs.mean()), 2)); t["cy"].append(round(float(ys.mean()), 2))
        la_, lb_ = np.bincount(a[o]).argmax(), np.bincount(b[o]).argmax()
        t["size_px"].append(int(min((a == la_).sum(), (b == lb_).sum())))
    else:
        t["cx"].append(None); t["cy"].append(None); t["size_px"].append(0)


def track_pair(A, B, principle):
    new = lambda: {"px": [], "cx": [], "cy": [], "size_px": []}
    tot, sa, sb = new(), new(), new()
    for f in range(NF):
        a, b = mask(A, f), mask(B, f)
        D, oa, ob = split_diff(a, b)
        _put(tot, D, a, b); _put(sa, oa, a, b); _put(sb, ob, a, b)
    if principle == "O3":
        return {A: {**sa, "method": "split"}, B: {**sb, "method": "split"}}
    if principle == "O2":
        return {A: {**tot, "method": "total"}, B: {**tot, "method": "total"}}
    full, empty = (A, B) if sum(sa["px"]) >= sum(sb["px"]) else (B, A)     # O1: 짝 없는 픽셀이 많은 쪽이 물체 있는 영상
    none = {"px": [0] * NF, "cx": [None] * NF, "cy": [None] * NF, "size_px": [0] * NF}
    return {full: {**tot, "method": "total"}, empty: {**none, "method": "empty"}}


def runs(v):
    out, s = [], 0
    for i in range(1, len(v) + 1):
        if i == len(v) or v[i] != v[s]:
            out.append((int(v[s]), s, i - s)); s = i
    return out


def gaps_of(vis):
    r = runs(vis)
    return [(s, n) for k, (val, s, n) in enumerate(r) if val == 0 and 0 < k < len(r) - 1]


def event_gap(vis, d):
    r = runs(vis)
    k = next(i for i, (val, s, n) in enumerate(r) if s <= d < s + n)
    val, s, n = r[k]
    if val == 0 and 0 < k < len(r) - 1:
        return (s, n), "gap_contains_d"
    if val == 1 and k >= 2 and r[k - 1][0] == 0 and d - (r[k - 1][1] + r[k - 1][2]) < LAG:
        return (r[k - 1][1], r[k - 1][2]), "gap_ends_just_before_d"
    return None, ("not_yet_visible" if val == 0 and k == 0 else "hidden_to_end" if val == 0 else "visible_at_d")


def on_grid(s, n, step):
    return sum(1 for f in range(s, s + n) if f % step == 0)


def gap_geometry(s, n, t):
    cx, cy, size = t["cx"], t["cy"], t["size_px"]
    a, b = s - 1, s + n
    disp = float(np.hypot(cx[b] - cx[a], cy[b] - cy[a]))
    w = float(np.sqrt(max(size[a], size[b])))
    prev = [f for f in range(a - 3, a + 1) if f >= 0 and cx[f] is not None]
    sp = (float(np.mean([np.hypot(cx[prev[i + 1]] - cx[prev[i]], cy[prev[i + 1]] - cy[prev[i]]) for i in range(len(prev) - 1)]))
          if len(prev) >= 2 and prev == list(range(prev[0], prev[-1] + 1)) else None)
    return {"disp_px": round(disp, 1), "width_px": round(w, 1), "disp_widths": round(disp / w, 2) if w else None,
            "speed_before_px_per_frame": round(sp, 2) if sp is not None else None}


def summ(x):
    x = np.array([v for v in x if v is not None], float)
    if not len(x):
        return None
    return {"n": int(len(x)), "mean": round(float(x.mean()), 3), "median": round(float(np.median(x)), 3),
            "min": round(float(x.min()), 3), "max": round(float(x.max()), 3),
            "p25": round(float(np.percentile(x, 25)), 3), "p75": round(float(np.percentile(x, 75)), 3)}


def pair_scores():
    S = collections.defaultdict(lambda: collections.defaultdict(list))
    for vid, rows in json.load(open(PER_WIN))["windows"].items():
        for combo, s, C, sur in rows:
            if combo == "skip2_w32":
                S[vid][int(s)].append(float(sur))
    return {v: float(np.mean([min(c) for c in st.values()])) for v, st in S.items()}


def main():
    V = {r["video_id"]: r for r in csv.DictReader((AUX / "videos.csv").open())}
    P = list(csv.DictReader((AUX / "pairs.csv").open()))
    scenes = sorted({v["scene"] for v in V.values()})
    lab = {sc: next(f'{MO[v["label_motion"]]}/{VI[v["label_vis"]]}' for v in V.values() if v["scene"] == sc) for sc in scenes}
    VT = {}
    for sc in scenes:
        A, B = sorted(k for k, v in V.items() if v["scene"] == sc and v["plausible"] == "1")
        for v, t in track_pair(A, B, sc[:2]).items():
            VT[v] = {"scene": sc, "group": lab[sc], "principle": sc[:2], **t}
    G = pair_scores()
    wins = _intphys1_windows(NF, 2, 32, [4, 8, 12, 16, 20], 2, 2, "official")
    summary = {"_doc": __doc__.split("\n\n")[1], "fps": FPS, "threshold_px": T_DEF, "n_scenes": len(scenes), "n_pairs": len(P),
               "threshold_sensitivity": {}, "by_group": {}}
    # ── 문턱 민감도 (사건 가림 길이) ──
    for T in T_SENS:
        ev = {}
        for p in P:
            sv = p["pos"]
            vis = [int(k >= T) for k in VT[sv]["px"]]
            g, _ = event_gap(vis, int(p["first_div_sensitive"]) - 1)
            if g:
                ev[(sv, g)] = lab[p["scene"]]
        by = collections.defaultdict(list)
        for (sv, g), grp in ev.items():
            by[grp].append(g[1])
        summary["threshold_sensitivity"][f"T={T}px"] = {grp: summ(x) for grp, x in sorted(by.items())}
    # ── 기본 문턱 ──
    for v, t in VT.items():
        vis = [int(k >= T_DEF) for k in t["px"]]; t["visible"] = "".join(map(str, vis))
        t["has_object"] = int(sum(vis) > 0)
        hidden = set(f for s, n in gaps_of(vis) for f in range(s, s + n))
        t["windows_ctx_end_hidden_skip2_w32"] = sum(1 for C, cf, tf in wins if cf[-1] in hidden)
    gap_rows = []
    for v, t in sorted(VT.items()):
        vis = [int(c) for c in t["visible"]]
        for s, n in gaps_of(vis):
            gap_rows.append({"video": v, "scene": t["scene"], "group": t["group"], "principle": t["principle"], "event_gap": 0,
                             "start": s, "end": s + n - 1, "raw_frames": n, "seconds": round(n / FPS, 3),
                             "skip2_frames": on_grid(s, n, 2), "skip5_frames": on_grid(s, n, 5), **gap_geometry(s, n, t)})
    pair_rows = []
    for p in P:
        d = int(p["first_div_sensitive"]) - 1; sc = p["scene"]
        sv = p["pos"]
        vis = [int(c) for c in VT[sv]["visible"]]
        g, kind = event_gap(vis, d)
        if g is None and sum(vis[:d]) == 0:
            kind = "not_seen_before_d"
        gi, gp = G[p["imp"]], G[p["pos"]]
        row = {"pos": p["pos"], "imp": p["imp"], "scene": sc, "group": lab[sc], "principle": sc[:2], "d": d,
               "event_kind": kind, "gap_start": g[0] if g else None, "gap_raw": g[1] if g else None,
               "gap_skip2": on_grid(*g, 2) if g else None, "disp_widths": None,
               "correct_skip2_w32": 1.0 if gi > gp else (0.5 if gi == gp else 0.0)}
        if g:
            gr = next(x for x in gap_rows if x["video"] == sv and x["start"] == g[0])
            gr["event_gap"] = 1; row["disp_widths"] = gr["disp_widths"]
        pair_rows.append(row)
    for grp in sorted(set(lab.values())):
        ev = [x for x in gap_rows if x["group"] == grp and x["event_gap"]]
        vids = [v for v, t in VT.items() if t["group"] == grp and t["has_object"]]
        allg = [x for x in gap_rows if x["group"] == grp]
        pr = [x for x in pair_rows if x["group"] == grp]
        acc_by = collections.defaultdict(list)
        for x in pr:
            acc_by[str(x["gap_skip2"]) if x["gap_skip2"] is not None else "no_event_gap"].append(x["correct_skip2_w32"])
        summary["by_group"][grp] = {
            "n_scenes": sum(1 for sc in scenes if lab[sc] == grp), "n_pairs": len(pr),
            "event_kind": dict(collections.Counter(x["event_kind"] for x in pr)),
            "n_event_gaps_unique": len(ev), "scenes_with_event_gap": len({x["scene"] for x in ev}),
            "event_gap_raw_frames": summ([x["raw_frames"] for x in ev]), "event_gap_seconds": summ([x["seconds"] for x in ev]),
            "event_gap_skip2_frames": summ([x["skip2_frames"] for x in ev]), "event_gap_skip5_frames": summ([x["skip5_frames"] for x in ev]),
            "event_gap_disp_widths": summ([x["disp_widths"] for x in ev]), "event_gap_disp_px": summ([x["disp_px"] for x in ev]),
            "event_gap_speed_before": summ([x["speed_before_px_per_frame"] for x in ev]),
            "hist_event_gap_raw": dict(sorted(collections.Counter(x["raw_frames"] for x in ev).items())),
            "hist_event_gap_skip2": dict(sorted(collections.Counter(x["skip2_frames"] for x in ev).items())),
            "videos_with_object": len(vids),
            "gaps_per_video": summ([sum(1 for x in allg if x["video"] == v) for v in vids]),
            "all_gaps_raw_frames": summ([x["raw_frames"] for x in allg]),
            "visible_frames_per_video": summ([sum(int(c) for c in VT[v]["visible"]) for v in vids]),
            "windows_ctx_end_hidden_frac_skip2_w32": round(sum(VT[v]["windows_ctx_end_hidden_skip2_w32"] for v in vids) / (len(wins) * max(len(vids), 1)), 4),
            "acc_skip2_w32": round(100 * float(np.mean([x["correct_skip2_w32"] for x in pr])), 2),
            "acc_by_event_gap_skip2_frames": {k: [round(100 * float(np.mean(v)), 1), len(v)] for k, v in sorted(acc_by.items())}}
    summary["verify"] = {"acc_skip2_w32_all_pairs": round(100 * float(np.mean([x["correct_skip2_w32"] for x in pair_rows])), 2),
                         "note": "공식 88.89 는 principle macro. principle 마다 60 쌍이라 단순 평균과 같다."}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    (OUT / "videos.json").write_text(json.dumps(VT, ensure_ascii=False))
    old = OUT / "scenes.json"                              # 첫 판 (D 전체 = 한 물체) 산출물 → _superseded 로 내린다
    if old.exists():
        (OUT / "_superseded").mkdir(exist_ok=True); old.rename(OUT / "_superseded" / "scenes_v1_diff_as_one_object.json")
    for name, rows in (("gaps.csv", gap_rows), ("pairs.csv", pair_rows)):
        with open(OUT / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(json.dumps({k: v for k, v in summary.items() if k != "_doc"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
