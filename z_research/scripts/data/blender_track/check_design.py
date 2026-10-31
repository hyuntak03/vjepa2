#!/usr/bin/env python3
"""BlenderTrack 렌더 전 검사 — 설계만으로 (렌더 없이, 몇 초) 미래 경로가 반복되지 않는지 · late 창이 충분한지 (2026-10-31).

VOBench 의 문제: late 미래 칸 경로가 21 가지뿐이라 설계 칸 split 의 test late 경로 66.7 % · 끝 칸 100 % 가 train 에 있었다.
여기서는 같은 잣대로 잰다.

창: 보폭 s ∈ {2, 3, 4} 프레임, 문맥 16 샘플 + 미래 16 샘플 (평가와 같은 꼴). 시작 프레임은 모두.
  블록 = 연속 2 샘플. 미래 경로 = 미래 8 블록의 target 중심 칸 (16×16 토큰 격자) 순서.
  late 창 = 문맥 마지막 샘플에서 target 이 가려져 있고 (가려진 비율 ≥ 0.9, 화면 밖 포함), 문맥 앞쪽에서 보였고,
            미래에서 다시 보인다 (< 0.3).
  겹침 = 짝수 idx clip 을 A, 홀수를 B 로 나눠 B 의 late 미래 경로가 A 의 (모든 창의) 경로 집합에 있는 비율.
가려짐은 설계 근사 (판 · 화면 밖; 물체끼리 가림은 안 셈). 실제 라벨은 렌더가 ray cast 로 낸다.

  B=/data/hyuntak/anaconda3/envs/blender/bin/python   (numpy 만 쓴다; vjepa2 env 도 된다)
  $B z_research/scripts/data/blender_track/check_design.py --n 4000 --out <json>
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import design as D  # noqa: E402

CTX = FUT = 16


def windows(ctr, hid, s, T):
    span = (CTX + FUT - 1) * s + 1
    out = []
    for f0 in range(0, T - span + 1):
        idx = f0 + s * np.arange(CTX + FUT)
        h = hid[idx]
        c = ctr[idx]
        blk = c[CTX:].reshape(FUT // 2, 2, 2).mean(1)
        cell = np.clip(np.floor(blk / 16), -1, 16).astype(int)
        path = tuple(map(tuple, cell))
        late = bool(h[CTX - 1] >= 0.9 and (h[:CTX - 1] < 0.3).any() and (h[CTX:] < 0.3).any())
        k = 0
        while k < CTX and h[CTX - 1 - k] >= 0.9:
            k += 1
        moving = len(set(path)) > 1
        out.append((f0, path, late, k, moving))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    t0 = time.time()
    specs = [D.sample_clip(a.seed, i) for i in range(a.n)]
    st = [D.target_state(sp) for sp in specs]
    T = specs[0]["T"]
    res = {"n_clips": a.n, "design_s": round(time.time() - t0, 1),
           "target_kind": dict(collections.Counter(sp["objects"][0]["motion"]["kind"] for sp in specs)),
           "event_panels": dict(collections.Counter(str(sum(p["role"] == "event" for p in sp["panels"])) for sp in specs))}
    # 가려진 구간 길이 (연속 가려짐 ≥ 0.9, 화면 안 · 밖 모두)
    runs = []
    for _, h in st:
        r = 0
        for x in list(h >= 0.9) + [False]:
            if x:
                r += 1
            elif r:
                runs.append(r); r = 0
    runs = np.array(runs) / D.FPS
    res["hidden_run_s"] = {"n": int(len(runs)), "p10": float(np.percentile(runs, 10)), "median": float(np.median(runs)),
                           "p90": float(np.percentile(runs, 90))}
    for s in (2, 3, 4):
        W = [windows(c, h, s, T) for c, h in st]
        A_paths = set(p for i, w in enumerate(W) if i % 2 == 0 for (_, p, _, _, _) in w)
        A_end = set(p[-1] for i, w in enumerate(W) if i % 2 == 0 for (_, p, _, _, _) in w)
        B_late = [(p, k) for i, w in enumerate(W) if i % 2 == 1 for (_, p, l, k, _) in w if l]
        B_late_mov = [(p, k) for p, k in B_late if len(set(p)) > 1]
        n_win = sum(len(w) for w in W)
        n_late = sum(l for w in W for (_, _, l, _, _) in w)
        clips_late = sum(any(l for (_, _, l, _, _) in w) for w in W)
        late_paths = [p for w in W for (_, p, l, _, _) in w if l]
        ks = collections.Counter(min(k, 6) for w in W for (_, _, l, k, _) in w if l)
        speed = []
        for c, h in st:
            v = np.linalg.norm(np.diff(c[::s], axis=0), axis=1)
            speed.append(np.median(v))
        res[f"stride{s}"] = {
            "windows": n_win, "late_windows": int(n_late), "late_frac": n_late / n_win,
            "clips_with_late": clips_late / a.n,
            "late_k_hist(samples hidden at ctx end, 6=6+)": {str(k): v for k, v in sorted(ks.items())},
            "late_paths_unique_frac": len(set(late_paths)) / max(len(late_paths), 1),
            "B_late_path_in_A": float(np.mean([p in A_paths for p, _ in B_late])) if B_late else None,
            "B_late_moving_path_in_A": float(np.mean([p in A_paths for p, _ in B_late_mov])) if B_late_mov else None,
            "B_late_end_cell_in_A": float(np.mean([p[-1] in A_end for p, _ in B_late])) if B_late else None,
            "distinct_end_cells_late": len(set(p[-1] for p in late_paths)),
            "median_speed_px_per_sample": float(np.median(speed)),
        }
    res["vobench_v2_reference"] = {"late_distinct_future_paths": 21, "test_late_path_in_train": 0.667, "test_late_end_cell_in_train": 1.0}
    print(json.dumps(res, indent=1, ensure_ascii=False))
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
