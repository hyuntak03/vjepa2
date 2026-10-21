#!/usr/bin/env python3
"""intphys1_late 데이터 만들기 (2026-10-17, 사용자 지시) — IntPhys 1 dev 의 **가림 샘플만**, 위반 시점 직전이 문맥 끝이 되도록 창을 쌍마다 맞춘다.

위반 시점 d = 가능 · 불가능 영상이 처음 픽셀로 달라지는 raw 프레임 (0-based, 15 fps).
  출처 z_research/previous/OcclusionStats/intphys1/pairs.csv (정본 Archive/INTPHYS1_OCCLUSION_2026-09-26.md, 180 쌍 모두 first_div_sensitive − 1).
대상 = group ∈ {moving/occluded, static/occluded} = 90 쌍 (O1 · O2 · O3 각 30, 180 영상).
창 = 32 장, 간격 --skip (기본 2 = 7.5 fps, Garrido IntPhys 1 최고 칸 skip2 와 같은 간격):
  문맥 16 장 = 마지막 문맥 프레임 e = d + ctx_end_offset (기본 −1 = pos · imp 가 같은 마지막 프레임) 에서 거꾸로 e − skip·15 … e
  미래 16 장 = e + skip · 1 … e + skip · 16                   (미래 첫 장이 d 다: 기본값에서 e + 2 = d + 1 … 이 아니라 아래 주의)
  ⚠️ skip 2 · offset −1 이면 미래 첫 장 = d + 1 이고 d 자체는 창에 없다 (짝수 간격). offset 0 이면 d 가 문맥 끝 (pos · imp 문맥이 그 한 장에서 갈린다).
  영상 범위 [0, 99] 밖은 끝 프레임으로 채운다 (clamp) — index 의 clamp_lo / clamp_hi 열에 몇 장인지 적는다.
산출
  프레임: <frames_out>/<video_id>/000000.png … 000031.png (원본 PNG 의 심볼릭 링크)
  index : data_csv/intphys1_late[_s<skip>][_o<offset>]/index.csv (IntPhys 1 index 열 + d · ctx_end · clamp_lo · clamp_hi · group · event_kind)
  감사  : 같은 폴더 context_integrity.json — 쌍마다 문맥 16 장이 픽셀 단위로 같은가 (offset −1 이면 0 mismatch 여야 한다)

  python cvpr/eval/intphys1_late/build.py [--skip 2] [--ctx-end-offset -1]

--hidden-end (2026-10-17, 사용자 지시 "못 보게 설정하자"): 문맥 끝 e 를 쌍마다 다시 고른다 — d−1 에서는 이동 물체 60 쌍 중 40 쌍이 마지막 토큰에 물체가 보였다
  (z_research/scripts/analysis/intphys1_late_ctx_end_visibility.py). 조건: 마지막 문맥 튜블릿 두 장 (e − skip, e) 에서
  (a) 가능 영상의 물체 mask 0 px (OcclusionStats videos.json), (b) 가능 · 불가능 RGB 차이 픽셀 (max 채널 |Δ| > 10) ≤ HID_DIFF (= 불가능 영상에도 안 보임),
  (c) 채점에 들어가는 문맥 장 (마지막 튜블릿 제외) 중 물체가 보이는 장이 있음 (mask ≥ 20 px). ⚠️ 2026-10-18 정정: 전에는 영상 처음부터 검사해 정지 쌍이 문맥 안에서 물체를 한 번도 안 보는데도 남았다. e ∈ [d − 12, d + 8] 중 d 에 가장 가까운 것 (d 뒤 문맥 장도 전부 (b) 를 만족). 없는 쌍은 뺀다.
  e ≥ d 면 문맥 몇 장이 d 뒤라 pos · imp 가 수 px 다를 수 있다 → 불가능 영상의 문맥은 가능 영상 프레임으로 링크해 문맥을 비트 단위로 맞춘다
  (바뀐 장 수 = index 의 ctx_from_pos 열, 그 장들의 원래 차이 픽셀 = context_integrity.json).

--reveal-end (2026-10-18, 사용자 지시 "future 에 가림막이 다시 안 내려가는데 고려해서 다시"): --hidden-end 조건을 그대로 두고 e 를 '결과가 드러나기 직전' 으로.
  r = d 이후 가능 · 불가능 RGB 차이 픽셀 (max 채널 |Δ| > 10) 이 50 px 를 처음 넘는 프레임 (물체의 있음 / 없음 · 모양 · 위치가 눈에 보이는 순간).
  e = 조건 (a)(b)(c) 를 만족하는 e < r 중 가장 늦은 것 → 미래 첫 장 (e + skip) 에 결과가 이미 드러나 있다. --hidden-end 판은 정지 25 쌍 중 17 쌍이 미래 창 안에서 결과가 안 드러났다.
--denoise T (2026-10-18): IntPhys 1 dev 불가능 영상은 d 부터 화면 전체에 ±1 렌더 차이가 생긴다 (하늘 띠 픽셀의 58 %, |Δ| ≥ 2 는 0 %; d 전은 0).
  불가능 영상 미래 장의 픽셀 중 가능 영상과 max 채널 |Δ| ≤ T 인 것은 가능 영상 값으로 바꿔 PNG 로 쓴다 (링크 대신). 문맥은 --hidden-end 와 같이 가능 영상 프레임.
  ⚠️ 가능 영상을 기준으로 불가능 영상을 고치는 것이라 라벨 정보를 쓴다 — 원인 진단 (반사실) 전용, 평가 규약으로 쓰지 않는다.
--auto-skip 2,3,4,5,6 (2026-10-18, 사용자 지시 "frame stride 를 잘 해서 미래에 잘 보이도록, 잡음은 건드리지 말고"): --hidden-end 규칙 (e 는 d 에 가장 가까운 '마지막 튜블릿 가림' 지점)
  을 stride s 마다 다시 풀고, 결과가 드러나는 프레임 r 이 미래 앞쪽 --reveal-within 장 (기본 8) 안에 드는 (e < r ≤ e + 8·s) 가장 작은 s 를 쌍마다 고른다. 잡음 · 프레임은 건드리지 않는다.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[3]
PAIRS = REPO / "z_research/previous/OcclusionStats/intphys1/pairs.csv"
SRC_INDEX = Path("/local_datasets/world/world_analysis/IntPhys1_dev_videos/index.csv")
SRC_FR = Path("/local_datasets/world/world_analysis/IntPhys1_dev_frame_png")
N_RAW, N_CTX, N_FUT = 100, 16, 16
VIDEOS_JSON = REPO / "z_research/previous/OcclusionStats/intphys1/videos.json"
HID_DIFF = 20


def rgb(vid, t):
    return np.asarray(Image.open(src(vid, t)).convert("RGB")).astype(np.int16)


def ndiff(a, b, t):
    return int((np.abs(rgb(a, t) - rgb(b, t)).max(-1) > 10).sum())


def reveal_frame(pos, imp, d):
    return next((f for f in range(d, N_RAW) if ndiff(pos, imp, f) > 50), None)


def reveal_end(pos, imp, d, skip, px):
    r = reveal_frame(pos, imp, d)
    if r is None:
        return None, None
    for e in range(r - 1, skip - 1, -1):
        fs = [e - skip, e]
        after = [e - skip * k for k in range(N_CTX) if e - skip * k >= d]
        if all(px[f] == 0 for f in fs) and any(x >= 20 for x in px[: e - skip]) and all(ndiff(pos, imp, f) <= HID_DIFF for f in set(fs) | set(after)):
            return e, r
    return None, r


def fit_end(pos, imp, d, skip, px, r, within):
    """--fit (2026-10-18): stride skip 에서 (a) 마지막 튜블릿 (e−skip, e) 물체 mask 0 · (b) 그 두 장과 d 뒤 문맥 장의 가능 · 불가능 차이 ≤ HID_DIFF
    (c) 마지막 튜블릿을 뺀 문맥 장 중 물체가 보이는 장 (mask ≥ 20) 이 있음 · (d) 결과 프레임 r 이 미래 within 장 안 (e < r ≤ e + within·skip)
    을 모두 만족하는 e 중 d 에 가장 가까운 것 (같은 거리면 d 앞쪽). e 범위는 [skip, r − 1] 전체 — 정지 쌍은 d 가 안 보이는 시점이라 가림 구간 어디든."""
    cand = sorted(range(skip, r), key=lambda e: (abs(e - d), e > d))
    for e in cand:
        if r > e + within * skip:
            continue
        fs = [e - skip, e]
        if not all(px[f] == 0 for f in fs):
            continue
        ctx = [min(max(e - skip * k, 0), N_RAW - 1) for k in range(2, N_CTX)]
        if not any(px[f] >= 20 for f in ctx):
            continue
        after = [e - skip * k for k in range(N_CTX) if e - skip * k >= d]
        if all(ndiff(pos, imp, f) <= HID_DIFF for f in set(fs) | set(after)):
            return e
    return None


def hidden_end(pos, imp, d, skip, px):
    """d 에 가장 가까운 e (같은 거리면 d 앞쪽). d 뒤에 들어가는 문맥 장도 전부 pos · imp 차이 ≤ HID_DIFF."""
    cand = sorted(range(max(skip, d - 12), min(N_RAW - 1, d + 8) + 1), key=lambda e: (abs(e - d), e > d))
    for e in cand:
        fs = [e - skip, e]
        after = [e - skip * k for k in range(N_CTX) if e - skip * k >= d]
        ctx = [min(max(e - skip * k, 0), N_RAW - 1) for k in range(2, N_CTX)]       # 마지막 튜블릿을 뺀 문맥 장 (clamp 포함)
        if all(px[f] == 0 for f in fs) and any(px[f] >= 20 for f in ctx) and all(ndiff(pos, imp, f) <= HID_DIFF for f in set(fs) | set(after)):
            return e
    return None


def src(vid, t):
    p, s, r = vid.split("_")
    return SRC_FR / p / s / r / "scene" / f"scene_{t + 1:03d}.png"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", type=int, default=2); ap.add_argument("--ctx-end-offset", type=int, default=-1)
    ap.add_argument("--hidden-end", action="store_true", help="문맥 마지막 튜블릿에 물체가 안 보이는 e 를 쌍마다 고른다 (docstring)")
    ap.add_argument("--reveal-end", action="store_true", help="문맥 끝 = 결과가 드러나기 직전의 가려진 지점 (docstring)")
    ap.add_argument("--denoise", type=int, default=-1, help="불가능 영상 미래의 |Δ| ≤ T 픽셀을 가능 영상 값으로 (-1 = 안 함)")
    ap.add_argument("--auto-skip", default="", help="쌍마다 고를 stride 후보 (예: 2,3,4,5,6). docstring")
    ap.add_argument("--reveal-within", type=int, default=8)
    ap.add_argument("--fit", action="store_true", help="--auto-skip 과 같이: e 를 가림 구간 어디서든 (fit_end docstring)")
    ap.add_argument("--uniform-d", action="store_true",
                    help="2026-10-20 사용자: 문맥 16 장 = [0, d−1] 균일, 미래 16 장 = [d, 99] 균일 (쌍마다 문맥 · 미래 간격이 다르다). IntPhys 1 dev 180 쌍 전부")
    ap.add_argument("--frames-out", default="/local_datasets/world/world_analysis/IntPhys1_late")
    a = ap.parse_args()
    auto = [int(x) for x in a.auto_skip.split(",") if x]
    a.hidden_end = a.hidden_end or a.reveal_end or bool(auto)
    name = "intphys1_late_uni" if a.uniform_d else "intphys1_late" + ("" if a.skip == 2 else f"_s{a.skip}") + (f"_auto{'-'.join(map(str, [min(auto), max(auto)])) if len(auto) > 1 else auto[0]}" + ("" if a.reveal_within == 8 else f"r{a.reveal_within}") + ("fit" if a.fit else "") if auto else "_rev" if a.reveal_end else "_hid" if a.hidden_end else ("" if a.ctx_end_offset == -1 else f"_o{a.ctx_end_offset:+d}".replace("+", "p").replace("-", "m")))
    name += f"_dn{a.denoise}" if a.denoise >= 0 else ""
    VT = json.load(open(VIDEOS_JSON)) if a.hidden_end else None
    out_csv = REPO / "data_csv" / name; out_csv.mkdir(parents=True, exist_ok=True)
    fr_root = Path(a.frames_out + name[len("intphys1_late"):]); fr_root.mkdir(parents=True, exist_ok=True)

    pairs = [r for r in csv.DictReader(open(PAIRS)) if a.uniform_d or r["group"] in ("moving/occluded", "static/occluded")]
    idx = {r["video_id"]: r for r in csv.DictReader(open(SRC_INDEX))}
    rows, integ, dropped = [], [], []
    for p in pairs:
        d = int(p["d"]); e = d + a.ctx_end_offset
        r = None; SK = a.skip
        if auto:
            r = reveal_frame(p["pos"], p["imp"], d); e = None
            for sk in auto:
                if a.fit:
                    es = fit_end(p["pos"], p["imp"], d, sk, VT[p["pos"]]["px"], r, a.reveal_within) if r is not None else None
                else:
                    es = hidden_end(p["pos"], p["imp"], d, sk, VT[p["pos"]]["px"])
                if es is not None and r is not None and es < r <= es + a.reveal_within * sk:
                    SK, e = sk, es; break
            if e is None:
                dropped.append(dict(pos=p["pos"], imp=p["imp"], group=p["group"], event_kind=p["event_kind"], reveal=r)); continue
        elif a.hidden_end:
            if a.reveal_end:
                e, r = reveal_end(p["pos"], p["imp"], d, a.skip, VT[p["pos"]]["px"])
            else:
                e = hidden_end(p["pos"], p["imp"], d, a.skip, VT[p["pos"]]["px"])
            if e is None:
                dropped.append(dict(pos=p["pos"], imp=p["imp"], group=p["group"], event_kind=p["event_kind"])); continue
        ts = [e - SK * (N_CTX - 1 - k) for k in range(N_CTX)] + [e + SK * (k + 1) for k in range(N_FUT)]
        if a.uniform_d:                                       # 문맥 = [0, d−1] 균일 16 장, 미래 = [d, 99] 균일 16 장
            e = d - 1; SK = 0
            ts = [int(round(x)) for x in np.linspace(0, d - 1, N_CTX)] + [int(round(x)) for x in np.linspace(d, N_RAW - 1, N_FUT)]
        lo = sum(t < 0 for t in ts); hi = sum(t > N_RAW - 1 for t in ts); ts = [min(max(t, 0), N_RAW - 1) for t in ts]
        from_pos = [k for k in range(N_CTX) if a.hidden_end and ts[k] >= d]       # d 뒤 문맥 장: 불가능 영상도 가능 영상 프레임을 쓴다
        pre_diff = {k: ndiff(p["pos"], p["imp"], ts[k]) for k in from_pos}
        for vid in (p["pos"], p["imp"]):
            vd = fr_root / vid; vd.mkdir(exist_ok=True)
            for k, t in enumerate(ts):
                l = vd / f"{k:06d}.png"
                if l.is_symlink() or l.exists(): l.unlink()
                if a.denoise >= 0 and vid == p["imp"] and k >= N_CTX:      # 불가능 미래: 가능 영상과 |Δ| ≤ T 픽셀을 가능 값으로
                    A = rgb(p["imp"], t); B = rgb(p["pos"], t); m = np.abs(A - B).max(-1) <= a.denoise
                    A[m] = B[m]; Image.fromarray(A.astype(np.uint8)).save(l)
                else:
                    os.symlink(src(p["pos"] if k in from_pos else vid, t), l)
            row = dict(idx[vid]); row.update(d=d, reveal=r, ctx_end=e, ctx_end_minus_d=e - d, ctx_from_pos=len(from_pos), skip=SK, frames=" ".join(map(str, ts)), clamp_lo=lo, clamp_hi=hi,
                                         group=p["group"], event_kind=p["event_kind"], principle=p["principle"])
            rows.append(row)
        lk = lambda vid, k: np.asarray(Image.open(fr_root / vid / f"{k:06d}.png"))
        mism = [k for k in range(N_CTX) if not np.array_equal(lk(p["pos"], k), lk(p["imp"], k))]
        integ.append(dict(pos=p["pos"], imp=p["imp"], d=d, reveal=r, skip=SK, ctx_end=e, n_ctx_mismatch=len(mism), mismatch_slots=mism, clamp_lo=lo, clamp_hi=hi,
                          ctx_from_pos_slots=from_pos, ctx_from_pos_orig_diff_px=pre_diff))
    cols = list(rows[0].keys())
    with open(out_csv / "index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    n_bad = sum(i["n_ctx_mismatch"] > 0 for i in integ)
    json.dump(dict(n_pairs=len(integ), n_pairs_ctx_mismatch=n_bad, skip=a.skip, ctx_end_offset=a.ctx_end_offset, hidden_end=a.hidden_end, dropped=dropped,
                   n_pairs_clamped=sum((i["clamp_lo"] + i["clamp_hi"]) > 0 for i in integ), pairs=integ),
              open(out_csv / "context_integrity.json", "w"), indent=1)
    if a.hidden_end:
        import collections
        print("뺀 쌍:", len(dropped), dict(collections.Counter((x["group"], x["event_kind"]) for x in dropped)))
        if auto: print("stride (쌍 수):", dict(collections.Counter((i["skip"], i["pos"][:2]) for i in integ)), "· r 이 미래 몇 번째 장:", dict(collections.Counter(-(-(i["reveal"] - i["ctx_end"]) // i["skip"]) for i in integ)))
        if a.reveal_end: print("r − e (미래 첫 장 = e + skip):", dict(collections.Counter(i["reveal"] - i["ctx_end"] for i in integ)))
        print("e − d:", dict(collections.Counter(i["ctx_end"] - i["d"] for i in integ)), "· 가능 영상 프레임으로 바꾼 문맥 장 (쌍 수):", sum(bool(i["ctx_from_pos_slots"]) for i in integ),
              "· 원래 차이 px 최대", max([max(i["ctx_from_pos_orig_diff_px"].values()) for i in integ if i["ctx_from_pos_slots"]] or [0]))
    print(f"{name}: 쌍 {len(integ)} · 영상 {len(rows)} · 문맥 불일치 쌍 {n_bad} · clamp 쌍 {sum((i['clamp_lo'] + i['clamp_hi']) > 0 for i in integ)}")
    print(" principle:", {k: sum(i['pos'][:2] == k for i in integ) for k in ('O1', 'O2', 'O3')}, "· index", out_csv / "index.csv", "· 프레임", fr_root)


if __name__ == "__main__":
    main()
