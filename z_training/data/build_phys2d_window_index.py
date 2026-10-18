#!/usr/bin/env python
"""phys2d_v1 → '사건이 예측 구간에만 있는' 창 인덱스를 만든다.

왜 필요한가 (2026-10-18 사용자 지시):
  "어떤 물리적 특성이 context 에 들어가면 안되고 미래 예측부분에 들어가도록
   — event transition 이 context 에 있으면 안돼"

⚠️ 세트가 들고 있는 `boundary.tsv:boundary_frame` 을 **쓰지 않는다.**
   실측해 보면 `boundary_frame = event_frame + 1~3` 이다 (δ 분포 1:2:3 ≈ 균등, 전수 150,381 행).
   즉 그 컬럼대로 자르면 사건이 문맥의 **마지막 1~3 프레임 안에** 들어온다 — 지시의 반대다.
   대신 `event_frame` 을 쓴다. 그 컬럼이 실제로 영상의 운동 불연속을 가리키는지는 따로 쟀다:
   단일 사건 clip 60 개에서 event_frame 의 가속도 백분위 중앙값 **0.92** (무작위 프레임 대조군 0.74),
   백분위 > 0.9 비율 **0.57** (대조군 0.20). → event_frame 은 실재하는 사건 시점이다.

창 규칙 (stride s, 길이 T, 문맥 C 프레임)
  문맥 = [start, start + C*s)      예측 = [start + C*s, start + T*s)
  조건 ① 문맥 안에 **어떤 사건도** 없다   (clip 의 모든 event_frame 에 대해)
       ② 예측 구간 안에 사건이 **하나 이상** 있다
       ③ 0 <= start,  start + T*s <= video_frames
  ①은 "clip 의 모든 사건" 으로 건다 — 겨냥한 사건 하나만 빼는 게 아니다.

출력: CSV `path,start,stride,n_frames,scenario,category,n_future_events,first_future_event`
     (학습 로더 `video_window` 가 이 컬럼을 읽는다)
"""
from __future__ import annotations

import argparse, collections, csv, os, sys

SRC_PREFIX = "/workspace/dataset/training_data/phys2d_v1"


def load_boundary(split_dir: str, dst_prefix: str):
    """clip -> (video_frames, 정렬된 event_frame 집합, scenario, category)"""
    p = os.path.join(split_dir, "boundary.tsv")
    ev, meta = collections.defaultdict(set), {}
    with open(p, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            path = r["path"].replace(SRC_PREFIX, dst_prefix)
            ev[path].add(int(r["event_frame"]))
            meta[path] = (int(r["video_frames"]), r["scenario"], r["category"])
    return {k: (sorted(v),) + meta[k] for k, v in ev.items()}


def windows(events, video_frames, T, C, s):
    """조건 ①②③ 을 만족하는 start 목록."""
    out = []
    ctx, span = C * s, T * s
    for start in range(0, video_frames - span + 1):
        c0, c1 = start, start + ctx            # 문맥 [c0, c1)
        f1 = start + span                      # 예측 [c1, f1)
        if any(c0 <= e < c1 for e in events):
            continue                           # ① 위반: 문맥에 사건
        fut = [e for e in events if c1 <= e < f1]
        if not fut:
            continue                           # ② 위반: 예측 구간에 사건 없음
        out.append((start, len(fut), fut[0]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/local_datasets/world/training_data/phys2d_v1")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n-frames", type=int, default=32)
    ap.add_argument("--context", type=int, default=16)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-per-clip", type=int, default=4, help="clip 당 창 상한 (0=전부). 고르게 솎는다")
    ap.add_argument("--out", default=None)
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()

    split_dir = os.path.join(a.root, a.split)
    clips = load_boundary(split_dir, a.root)
    tot_w, kept, per_scen = 0, [], collections.Counter()
    no_win = collections.Counter()
    for path, (events, vf, scen, cat) in sorted(clips.items()):
        w = windows(events, vf, a.n_frames, a.context, a.stride)
        tot_w += len(w)
        if not w:
            no_win[scen] += 1
            continue
        if a.max_per_clip and len(w) > a.max_per_clip:          # 고르게 솎기 (시작점 편향 방지)
            step = (len(w) - 1) / (a.max_per_clip - 1) if a.max_per_clip > 1 else 0
            w = [w[round(i * step)] for i in range(a.max_per_clip)]
        per_scen[scen] += len(w)
        for start, nfut, first in w:
            kept.append((path, start, a.stride, a.n_frames, scen, cat, nfut, first))

    n_clip_ok = len(clips) - sum(no_win.values())
    print(f"[{a.split}] T={a.n_frames} C={a.context} stride={a.stride}  "
          f"clip {n_clip_ok}/{len(clips)} ({n_clip_ok/len(clips):.1%}) 가 창을 갖는다, "
          f"유효 창 {tot_w:,} → 솎은 뒤 {len(kept):,}")
    if no_win:
        print("  창 0 인 clip (시나리오별):", dict(no_win.most_common()))
    if a.report_only:
        return
    out = a.out or os.path.join("data_csv", "phys2d_v1", f"{a.split}_T{a.n_frames}C{a.context}s{a.stride}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["path", "start", "stride", "n_frames", "scenario", "category",
                     "n_future_events", "first_future_event"])
        wr.writerows(kept)
    print(f"  → {out}  ({len(kept):,} 행)")
    print("  시나리오별:", dict(per_scen.most_common()))


if __name__ == "__main__":
    main()
