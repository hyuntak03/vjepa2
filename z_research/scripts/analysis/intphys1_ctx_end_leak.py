#!/usr/bin/env python3
"""IntPhys 1 dev — 문맥 끝이 가려진 창에서 물체가 새어 보이는가 (2026-10-17, 사용자 지시: "가림을 context 마지막에 넣을 때 물체 실루엣이나 이런 거 노출됐는지").

창 = Garrido skip2_w32 (C ∈ {4..20}, z_research/previous/Benchmarks 의 88.89 칸). 문맥 끝 프레임 cf[-1] 이 사건 물체의 가림 구간 (OcclusionStats videos.json 의 gap,
보임 = 물체 mask ≥ 20 px) 에 드는 창만 본다. 그 프레임 (과 같은 튜블릿의 앞 프레임) 에서 둘을 잰다 — 학습·문턱 외 파라미터 없음.
  (1) 실루엣: 물체 mask 픽셀 수 (videos.json px). 1–19 px = 보임 문턱 아래로 남은 조각.
  (2) 그 밖의 흔적 (그림자 · 반사 · 가림막 너머 비침): 같은 scene 의 다른 possible 영상과의 RGB 차이 — 그 프레임에서 짝 영상도 물체 mask 0 일 때만.
      차이 픽셀 = 세 채널 중 최대 |Δ| > 10 (0–255). 기준선 = 둘 다 물체가 아직 화면에 들어오지 않은 앞쪽 프레임의 같은 값 (렌더 잡음 · 다른 물체 차이).
  python z_research/scripts/analysis/intphys1_ctx_end_leak.py   → 표 출력 + z_research/previous/OcclusionStats/intphys1/ctx_end_leak.json
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from evals.world_model_analysis.eval import _intphys1_windows  # noqa: E402

AUX = ROOT / "auto_research/_stage/IntPhys1_dev_by_scene"
FR = Path("/local_datasets/world/world_analysis/IntPhys1_dev_frame_png")
OUT = ROOT / "z_research/previous/OcclusionStats/intphys1"
DIFF = 10


def rgb(vid, f):
    b, q, r = vid.split("_")
    return np.asarray(Image.open(FR / b / q / r / "scene" / f"scene_{f + 1:03d}.png").convert("RGB")).astype(np.int16)


def runs(v):
    out, s = [], 0
    for i in range(1, len(v) + 1):
        if i == len(v) or v[i] != v[s]:
            out.append((v[s], s, i - s)); s = i
    return out


def main():
    VT = json.load(open(OUT / "videos.json"))
    V = {r["video_id"]: r for r in csv.DictReader((AUX / "videos.csv").open())}
    wins = _intphys1_windows(100, 2, 32, [4, 8, 12, 16, 20], 2, 2, "official")
    rows = []
    for vid, t in sorted(VT.items()):
        if "occluded" not in t["group"] or not t["has_object"]:
            continue
        vis = [int(c) for c in t["visible"]]; r = runs(vis)
        hidden = {f for k, (val, s, n) in enumerate(r) if val == 0 and 0 < k < len(r) - 1 for f in range(s, s + n)}
        first = next(i for i, x in enumerate(vis) if x)
        mate = next(k for k, v in V.items() if v["scene"] == t["scene"] and v["plausible"] == "1" and k != vid)
        mpx = VT[mate]["px"]; mfirst = next((i for i, x in enumerate(VT[mate]["visible"]) if x == "1"), 100)
        base_f = [f for f in range(0, min(first, mfirst) - 2, 3)][:6]
        base = [int((np.abs(rgb(vid, f) - rgb(mate, f)).max(-1) > DIFF).sum()) for f in base_f]
        ends = sorted({cf[-1] for C, cf, tf in wins if cf[-1] in hidden})
        for f in ends:
            fr = [f - 2, f]                                       # 문맥 마지막 튜블릿의 두 프레임 (skip 2 → 원 프레임 f-2, f)
            ok = all(mpx[x] == 0 for x in fr)
            dpx = [int((np.abs(rgb(vid, x) - rgb(mate, x)).max(-1) > DIFF).sum()) for x in fr] if ok else None
            rows.append(dict(video=vid, scene=t["scene"], group=t["group"], frame=f, n_windows=sum(1 for C, cf, tf in wins if cf[-1] == f),
                             obj_px=[int(t["px"][x]) for x in fr], mate=mate, mate_obj_px=[int(mpx[x]) for x in fr],
                             diff_px=dpx, base_diff_px=base))
    json.dump(rows, open(OUT / "ctx_end_leak.json", "w"), indent=1)
    print(f"문맥 끝이 가려진 (영상, 끝 프레임) {len(rows)} 개 · 영상 {len({r['video'] for r in rows})} 개")
    for g in sorted({r["group"] for r in rows}):
        R = [r for r in rows if r["group"] == g]
        sil = [r["obj_px"][1] for r in R]; sil2 = [r["obj_px"][0] for r in R]
        D = [r for r in R if r["diff_px"] is not None]
        lk = [max(r["diff_px"]) - max(r["base_diff_px"] or [0]) for r in D]
        print(f"\n[{g}] {len(R)} 개")
        print(f"  (1) 실루엣 mask px — 끝 프레임 f: 0 → {sum(s == 0 for s in sil)} · 1–19 → {sum(0 < s < 20 for s in sil)} · 최대 {max(sil)}"
              f"  | 튜블릿 앞 프레임 f-2: 0 → {sum(s == 0 for s in sil2)} · 1–19 → {sum(0 < s < 20 for s in sil2)} · ≥20 (보임) → {sum(s >= 20 for s in sil2)}")
        print(f"  (2) 짝 영상 대비 RGB 차이 픽셀 (기준선 뺀 값, 짝도 가려진 {len(D)} 개): 0 이하 {sum(x <= 0 for x in lk)} · 1–50 {sum(0 < x <= 50 for x in lk)} · >50 {sum(x > 50 for x in lk)}"
              + (f" · 중앙 {int(np.median(lk))} · 최대 {max(lk)}" if lk else ""))
        for r in sorted(D, key=lambda r: -max(r["diff_px"]))[:4]:
            print(f"     {r['video']} f{r['frame']}: diff {r['diff_px']} (기준선 {r['base_diff_px']}), 물체 px {r['obj_px']}, 창 {r['n_windows']}")


if __name__ == "__main__":
    main()
