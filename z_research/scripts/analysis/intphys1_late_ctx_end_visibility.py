#!/usr/bin/env python3
"""intphys1_late (문맥 끝 = 위반 직전 d−1, skip 2, 문맥 16 + 미래 16) — 문맥 마지막 토큰에 사건 물체가 보이는가 (2026-10-17, 사용자 지시).
쌍마다 가능 영상 (pos) 의 물체 mask 픽셀 수 (OcclusionStats videos.json px) 를 문맥 마지막 튜블릿 (문맥 15·16 번째 장) 과 그 앞 튜블릿에서 본다.
mask 가 놓치는 조각을 잡으려고 같은 scene 의 다른 가능 영상 (짝) 과 RGB 차이 픽셀 (max 채널 |Δ| > 10) 도 센다 — 짝도 그 장에서 물체 mask 0 일 때만.
  python z_research/scripts/analysis/intphys1_late_ctx_end_visibility.py → 표 + cvpr/results/eval/intphys1_late/intphys1_late_vith/ctx_end_visibility.json
"""
import csv, json, collections
from pathlib import Path
import numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parents[3]
FR = Path("/local_datasets/world/world_analysis/IntPhys1_dev_frame_png")
RES = ROOT / "cvpr/results/eval/intphys1_late/intphys1_late_vith"
VT = json.load(open(ROOT / "z_research/previous/OcclusionStats/intphys1/videos.json"))
V = {r["video_id"]: r for r in csv.DictReader(open(ROOT / "auto_research/_stage/IntPhys1_dev_by_scene/videos.csv"))}
IDX = {r["video_id"]: r for r in csv.DictReader(open(ROOT / "data_csv/intphys1_late/index.csv"))}
PB = json.load(open(RES / "per_block.json"))["per_video_surprise"]
def rgb(v, f):
    b, q, r = v.split("_"); return np.asarray(Image.open(FR / b / q / r / "scene" / f"scene_{f + 1:03d}.png").convert("RGB")).astype(np.int16)
rows = []
for p in csv.DictReader(open(ROOT / "z_research/previous/OcclusionStats/intphys1/pairs.csv")):
    pos, imp = p["pos"], p["imp"]
    if pos not in IDX: continue                                   # 가림 쌍 (moving/static occluded) 만
    fr = [int(x) for x in IDX[pos]["frames"].split()][:16]
    mate = next(k for k, v in V.items() if v["scene"] == p["scene"] and v["plausible"] == "1" and k != pos)
    px = VT[pos]["px"]; mpx = VT[mate]["px"]
    last, prev = fr[14:16], fr[12:14]
    def trace(fs):
        return [int((np.abs(rgb(pos, f) - rgb(mate, f)).max(-1) > 10).sum()) if mpx[f] == 0 else None for f in fs]
    rows.append(dict(pos=pos, imp=imp, group=p["group"], principle=p["principle"], event_kind=p["event_kind"], d=int(p["d"]), ctx_end=fr[-1],
                     px_last=[int(px[f]) for f in last], px_prev=[int(px[f]) for f in prev], rgb_last=trace(last),
                     correct=int(PB[imp] > PB[pos])))
json.dump(rows, open(RES / "ctx_end_visibility.json", "w"), indent=1)
def cls(r):
    a = max(r["px_last"]); t = [x for x in r["rgb_last"] if x is not None]
    if a >= 20: return "보임 (마지막 토큰 mask ≥ 20 px)"
    if a > 0 or (t and max(t) > 30): return "조각 (mask 1–19 px 또는 RGB 흔적 > 30 px)"
    if max(r["px_prev"]) >= 20: return "마지막 토큰은 가림, 그 앞 토큰에 보임"
    return "마지막 두 토큰 모두 가림"
for g in ("moving/occluded", "static/occluded"):
    R = [r for r in rows if r["group"] == g]; c = collections.defaultdict(list)
    for r in R: c[cls(r)].append(r["correct"])
    print(f"\n[{g}] {len(R)} 쌍 · release 정답 {100*np.mean([r['correct'] for r in R]):.1f}")
    for k in ("보임 (마지막 토큰 mask ≥ 20 px)", "조각 (mask 1–19 px 또는 RGB 흔적 > 30 px)", "마지막 토큰은 가림, 그 앞 토큰에 보임", "마지막 두 토큰 모두 가림"):
        if k in c: print(f"  {k}: {len(c[k])} 쌍 · release 정답 {100*np.mean(c[k]):.1f}")
    ek = collections.defaultdict(list)
    for r in R: ek[(r["event_kind"], cls(r)[:6])].append(r["correct"])
    print("  event_kind × 분류:", {f"{a}|{b}": f"{len(v)}쌍 {100*np.mean(v):.0f}" for (a, b), v in sorted(ek.items())})
