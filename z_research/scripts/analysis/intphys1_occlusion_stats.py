#!/usr/bin/env python3
"""⚠️ 대체됨 (2026-09-26) → z_research/scripts/analysis/occlusion_intphys1.py (O3 영상별 분리 · 쌍 단위 · 가려진 동안 이동 거리). 기록으로 둔다.

IntPhys 1 dev — 사건 물체가 몇 프레임 가려지나, 그리고 Garrido 88.89 칸 (skip 2) 에서 모델이 받는 프레임으로는 몇 장인가.

사건 물체의 가시성 (파라미터 없음):
  한 scene (4 중항) 의 possible 영상 두 개는 **사건 물체만** 다르다 (O1: 물체 있음 / 없음, O2: 모양 A / B, O3: 궤적).
  프레임 f 에서 두 영상의 mask 조각 면적 목록 (정렬한 픽셀 수) 이 **같으면** 사건 물체는 그 프레임에서 안 보인다 (가려졌거나 아직 화면에 없다).
  mask 값은 프레임마다 무작위로 다시 매겨지므로 값이 아니라 면적 목록을 비교한다.
  ⚠️ `auto_research/_stage/IntPhys1_dev_by_scene/obj_vis.npz` 는 0/1 가시성이 아니라 **보이는 물체 수** (0–3) 라 여러 물체 scene 에서 사건 물체의 가림을 못 가른다 — 쓰지 않는다.
사건 가림 (event gap) = pos/imp 첫 픽셀 차 d (0-기준) 를 포함하는 '안 보임' 연속 구간 중 **양쪽이 '보임' 으로 막힌 것**
  (앞이 막히지 않으면 물체가 아직 등장 전 → 가림이 아니다; 뒤가 막히지 않으면 영상 끝까지 안 보임).
  d 가 '보임' 프레임이어도 바로 앞 구간이 양쪽이 막힌 가림이고 d 가 그 끝 뒤 3 프레임 안이면 그 가림을 사건 가림으로 센다
  (위반은 가림 중에 일어나고 pos/imp 픽셀은 물체가 다시 나타나는 순간에야 갈린다). 그 밖은 event gap 없음.
  skip 2 격자 (raw 짝수) 와 skip 5 격자 (raw 5 의 배수) 에서 그 구간에 드는 프레임 수 = 모델이 받는 '가려진 프레임' 수.

  python z_research/scripts/analysis/intphys1_occlusion_stats.py
  → z_research/Benchmarks/exp_results/intphys1_event_position/occlusion_stats.json
"""
from __future__ import annotations
import collections, csv, json
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
AUX = ROOT / "auto_research/_stage/IntPhys1_dev_by_scene"
FR = Path("/local_datasets/world/world_analysis/IntPhys1_dev_frame_png")
OUT = ROOT / "z_research/Benchmarks/exp_results/intphys1_event_position"
MO = {"정지": "static", "이동": "moving"}; VI = {"눈앞": "visible", "가려짐": "occluded"}
LAG = 3          # d 가 가림 끝 뒤 몇 프레임 안이면 그 가림을 사건 가림으로 본다 (재등장 순간에 픽셀이 갈리는 경우)


def areas(vid, f):
    b, q, r = vid.split("_")
    m = np.asarray(Image.open(FR / b / q / r / "masks" / f"masks_{f + 1:03d}.png"))
    return tuple(sorted(np.unique(m, return_counts=True)[1].tolist()))


def event_visibility(pa, pb):
    """1 = 사건 물체가 보인다 (두 possible 영상의 mask 면적 목록이 다르다)."""
    return np.array([int(areas(pa, f) != areas(pb, f)) for f in range(100)], np.int8)


def runs(v):
    out, s = [], 0
    for i in range(1, len(v) + 1):
        if i == len(v) or v[i] != v[s]:
            out.append((int(v[s]), s, i - s)); s = i
    return out


def on_grid(s, n, step):
    return sum(1 for f in range(s, s + n) if f % step == 0)


def summ(x):
    x = np.array(x, float)
    if not len(x):
        return None
    return {"n": int(len(x)), "mean": round(float(x.mean()), 2), "median": float(np.median(x)), "min": float(x.min()), "max": float(x.max()),
            "p25": float(np.percentile(x, 25)), "p75": float(np.percentile(x, 75))}


def main():
    V = {r["video_id"]: r for r in csv.DictReader((AUX / "videos.csv").open())}
    P = list(csv.DictReader((AUX / "pairs.csv").open()))
    vis = {}
    for scene in sorted({v["scene"] for v in V.values()}):
        pos = sorted(k for k, v in V.items() if v["scene"] == scene and v["plausible"] == "1")
        assert len(pos) == 2, (scene, pos)
        vis[scene] = event_visibility(*pos)
    rows = []
    for p in P:
        d = int(p["first_div_sensitive"]) - 1
        v = vis[p["scene"]]
        rr = runs(v)
        k = next(i for i, (val, s, n) in enumerate(rr) if s <= d < s + n)
        val, s, n = rr[k]
        bounded = val == 0 and 0 < k < len(rr) - 1
        kind = ("gap" if bounded else "not_yet_visible" if val == 0 and k == 0 else
                "hidden_to_end" if val == 0 else "visible_at_d")
        if kind == "visible_at_d" and k >= 2:
            # 재등장 순간에 갈리는 경우: 바로 앞 구간이 양쪽이 막힌 가림이고 d 가 그 끝 뒤 LAG 프레임 안
            pv, ps, pn = rr[k - 1]
            if pv == 0 and d - (ps + pn) < LAG:
                kind, bounded, s, n = "gap_ends_just_before_d", True, ps, pn
        rows.append({"pos": p["pos"], "imp": p["imp"], "scene": p["scene"], "principle": p["principle"][:2],
                     "group": f'{MO[p["label_motion"]]}/{VI[p["label_vis"]]}', "d": d, "kind_at_d": kind,
                     "event_gap": [s, n] if bounded else None, "vis": "".join(map(str, v))})
    out = {"_doc": __doc__.split("\n\n")[1], "by_group": {}}
    groups = collections.defaultdict(list)
    for r in rows:
        groups[r["group"]].append(r)
    for g, rs in sorted(groups.items()):
        eg = [r["event_gap"] for r in rs if r["event_gap"]]
        out["by_group"][g] = {
            "n_pairs": len(rs), "kind_at_d": dict(collections.Counter(r["kind_at_d"] for r in rs)),
            "event_gap_raw_frames": summ([n for _, n in eg]),
            "event_gap_skip2_frames": summ([on_grid(s, n, 2) for s, n in eg]),
            "event_gap_skip5_frames": summ([on_grid(s, n, 5) for s, n in eg]),
            "event_gap_hist_raw": dict(sorted(collections.Counter(n for _, n in eg).items())),
            "event_gap_hist_skip2": dict(sorted(collections.Counter(on_grid(s, n, 2) for s, n in eg).items())),
            "by_principle_raw": {pr: summ([r["event_gap"][1] for r in rs if r["event_gap"] and r["principle"] == pr]) for pr in ("O1", "O2", "O3")}}
    ex = next(r for r in rows if r["pos"] == "O1_01_2")
    out["example_O1_01"] = {k: ex[k] for k in ("d", "kind_at_d", "event_gap", "vis")}
    out["pairs"] = rows
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "occlusion_stats.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("pairs", "_doc")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
