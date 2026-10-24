#!/usr/bin/env python
"""phys2d_sup → '전환이 예측 구간에만 있는' 고정 창 인덱스 (T 장, 문맥 C 장, stride s).

사용자 결정 (2026-10-18):
  "충돌이면 충돌 직전까지만 context 에 보여주고 그 다음 충돌하고 멈추거나 그런 걸 예측해야지. 아니면 딱 충돌 시점까지만 context"
  "permanence 는 context 에서 가려져도 되고 다양하게 여러 번 가려져도 돼"

⚠️ 정정 (2026-10-18, vast) — 옛 판은 사건을 `boundary.tsv` 에서만 읽었다. 그 파일은 **행으로 고른 사건만** 담는다
   (충돌은 영상당 2 개까지). json `events` 전체로 다시 세면 "문맥에 사건 없음" 창의 56.6 % 문맥에 사건이 있었고
   (dv ≥ 60 충돌 48 %), 예측 구간에 화면 밖 물체가 처음 보이는 창이 34 % 였다 (시나리오당 10 편 표본).
   지금은 clip json `events` 전체와 `_states.npz` 를 읽는다.

창 규칙 (전환 뒤 첫 장 = post. 전환은 post−1 과 post 사이에 일어난다)
  문맥 = [start, start + C·s) 의 장들, 예측 = [start + C·s, start + T·s)
  ① 문맥 안에 **충돌 (dv ≥ DV_ROW, 방해 물체 제외) · 받침 이탈** 전환이 없다:
     post−1 과 post 가 둘 다 문맥 장이면 위반. 단 문맥 마지막 장이 딱 전환 순간 (post = 문맥 마지막 장) 인 것은 허용
     ("딱 충돌 시점까지만 context"). 가림 · 다시 보임은 문맥에 있어도 된다.
  ② 예측 구간 안에 전환 (충돌 · 이탈 · 가림 · 다시 보임) 이 하나 이상
  ③ 예측 구간에 화면 밖에서 생긴 물체가 처음 보이는 장이 없다 (문맥으로 예측 못 한다)
  ④ 0 <= start,  start + T·s <= video_frames

출력: CSV `path,start,stride,n_frames,scenario,category,n_future_events,first_future_event,first_kind`
"""
from __future__ import annotations

import argparse, collections, csv, json, os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

DV_ROW = 60.0
HIDDEN = 0.02
GATE = ("impact", "leave")
TARGET = ("impact", "leave", "occlude", "reappear")


def clip_info(mp4):
    stem = mp4[:-4]
    js = json.load(open(stem + ".json"))
    role = [o["role"] for o in js["objects"]]
    ev = [e for e in js["events"] if role[e["obj"]] != "distractor" and (e["kind"] != "impact" or e["dv"] >= DV_ROW)]
    gate = sorted({e["frame"] for e in ev if e["kind"] in GATE})
    tgt = sorted((e["frame"], e["kind"]) for e in ev if e["kind"] in TARGET)
    z = np.load(stem + "_states.npz")
    vis = z["vis_frac"]
    entry = []
    for i, r in enumerate(role):
        if r == "spawn":
            s = np.where(vis[:, i] > HIDDEN)[0]
            if len(s):
                entry.append(int(s[0]))
    return mp4, js["n_frames"], js["scenario"], js["category"], gate, tgt, entry


def windows(gate, tgt, entry, vf, T, C, s):
    out = []
    for start in range(0, vf - T * s + 1):
        ctx = set(range(start, start + C * s, s))
        last = start + (C - 1) * s
        c1, f1 = start + C * s, start + T * s
        if any(p in ctx and (p - 1) in ctx and p != last for p in gate):
            continue                                   # ① 문맥 안에 충돌 · 이탈 전환
        if s > 1 and any(start < p <= last and p not in ctx for p in gate):
            continue                                   # stride > 1: 두 문맥 장 사이에 낀 전환도 문맥이다
        # 예측 구간의 전환 + 문맥 마지막 장이 딱 충돌 · 이탈 순간인 것 (그 결과는 예측 구간에 있다)
        fut = [(p, k) for p, k in tgt if c1 <= p < f1 or (p == last and k in GATE)]
        if not fut:
            continue                                   # ② 예측 구간에 전환 없음
        if any(c1 <= t < f1 for t in entry):
            continue                                   # ③ 예측 구간에 화면 밖 물체 첫 등장
        out.append((start, len(fut), fut[0][0], fut[0][1]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="생성기 --out (index.csv 가 있는 곳)")
    ap.add_argument("--n-frames", type=int, default=32)
    ap.add_argument("--context", type=int, default=16)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-per-clip", type=int, default=4, help="clip 당 창 상한 (0=전부). 고르게 솎는다")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    mp4s = [l.split()[0] for l in open(os.path.join(a.root, "index.csv")) if l.strip()]
    with ProcessPoolExecutor(a.workers) as ex:
        infos = list(ex.map(clip_info, mp4s, chunksize=64))
    kept, per_scen, per_kind, no_win, tot = [], collections.Counter(), collections.Counter(), collections.Counter(), 0
    for mp4, vf, scen, cat, gate, tgt, entry in infos:
        w = windows(gate, tgt, entry, vf, a.n_frames, a.context, a.stride)
        tot += len(w)
        if not w:
            no_win[scen] += 1
            continue
        if a.max_per_clip and len(w) > a.max_per_clip:          # 고르게 솎기 (시작점 편향 방지)
            step = (len(w) - 1) / (a.max_per_clip - 1) if a.max_per_clip > 1 else 0
            w = [w[round(i * step)] for i in range(a.max_per_clip)]
        per_scen[scen] += len(w)
        for start, nfut, first, kind in w:
            per_kind[kind] += 1
            kept.append((mp4, start, a.stride, a.n_frames, scen, cat, nfut, first, kind))
    n_ok = len(infos) - sum(no_win.values())
    print(f"T={a.n_frames} C={a.context} stride={a.stride}  clip {n_ok}/{len(infos)} ({n_ok / max(1, len(infos)):.1%}) 가 창을 갖는다, "
          f"유효 창 {tot:,} → 솎은 뒤 {len(kept):,}")
    print("  첫 전환 종류:", dict(per_kind.most_common()))
    if no_win:
        print("  창 0 인 clip (시나리오별):", dict(no_win.most_common()))
    out = a.out or os.path.join(a.root, f"windows_T{a.n_frames}C{a.context}s{a.stride}.csv")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["path", "start", "stride", "n_frames", "scenario", "category", "n_future_events", "first_future_event", "first_kind"])
        wr.writerows(kept)
    print(f"  → {out}  ({len(kept):,} 행)")
    print("  시나리오별:", dict(per_scen.most_common()))


if __name__ == "__main__":
    main()
