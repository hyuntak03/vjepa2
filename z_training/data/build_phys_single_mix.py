#!/usr/bin/env python3
"""phys{2,3}d_single 학습 manifest (train_mix.tsv) 만들기 + 볼 수 없는 창 거르기 (2026-10-20).

  python3 z_training/data/build_phys_single_mix.py --root <out>/train --write data_csv/phys3d_single_v1/train_mix.tsv \
      [--max-blind-run 39] [--min-vis-ctx 0] [--vis 0.3]

행 = boundary.tsv 의 impact · leave 행 + uniform.tsv 전부 (2D single 의 train_mix 와 같은 구성).
거르기 (사용자 2026-10-20 "처음부터 다 가려져 있는데 뭘 맞춰야 되는 거임 → 좀 걸러내봐"):
  --max-blind-run L (기본, 영상 단위): vis_frac ≤ --vis 가 L 장보다 길게 이어지는 영상을 통째로 뺀다.
    창 자리와 무관하게 빼지만 긴 가림이 있는 영상을 빼면 남은 영상의 가림이 줄고 앞쪽에 몰린다 — phys3d train L 39: 블록 자리 폭 4.1 pt
    (거르기 전 0.9), 정체별 영상 331~516, drop_behind 581 / 2,200. 값별 표는 PHYS3D_SINGLE_DATASET_2026-10-19.md §4-1.
  --min-vis-ctx N (창 단위, 기본 끔): 경계 b 앞 문맥 마지막 min(--min-vis-ctx, b − ctx_start) 장 안에 vis_frac > --vis 인 장이 하나도 없으면 뺀다.
    ⚠️ 창 단위는 블록 자리 편향을 만든다 (phys3d train, N 16: 문맥 블록 18 % · 예측 끝 블록 28 % 가림) — 문맥이 보이는 창만 남기 때문.
  (화면 밖도 '안 보임' 이다 — hidden 은 화면 안 가림만이라 vis_frac 로 잰다.) --min-vis-ctx 0 이면 거르지 않는다.
로더 (app/vjepa_frozen/var_window.py) 는 step 마다 문맥 C ∈ context_frames 를 고르고 b − ctx_start ≥ C 인 행에서 뽑는다.
그래서 C 별로 "문맥 C 장이 전부 안 보이는 창" 비율을 거르기 전후로 찍는다 (C 마다 행을 고르게 뽑는다고 보고).
"""
import argparse
import csv
import os

import numpy as np


def load_vis(root):
    vis = {}
    for m in csv.DictReader(open(os.path.join(root, "manifest.tsv")), delimiter="\t"):
        p = os.path.join(root, "clips", m["name"], f"{m['name']}_{int(m['k']):05d}")
        vis[os.path.abspath(p + ".mp4")] = np.load(p + "_states.npz")["vis_frac"][:, 0]
    return vis


def stats(rows, vis, thr, Cs, K):
    out = {}
    for C in Cs:
        n = blind = 0
        for r in rows:
            b, c0 = int(r["boundary_frame"]), int(r["ctx_start"] or 0)
            v = vis[os.path.abspath(r["path"])]
            if b - c0 < C or b + K > len(v):
                continue
            n += 1
            blind += not (v[b - C:b] > thr).any()
        out[C] = (n, blind)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--write")
    ap.add_argument("--max-blind-run", type=int, default=39)   # phys3d: 2D 의 C16 안 보이는 창 8.8 % 에 맞춘 값 (39 → 8.7 %)
    ap.add_argument("--min-vis-ctx", type=int, default=0)
    ap.add_argument("--vis", type=float, default=0.3)
    ap.add_argument("--C", default="4,8,16")
    ap.add_argument("--K", type=int, default=16)
    a = ap.parse_args()
    Cs = [int(c) for c in a.C.split(",")]
    vis = load_vis(a.root)
    rd = lambda f: list(csv.DictReader(open(os.path.join(a.root, f)), delimiter="\t"))
    rows = [r for r in rd("boundary.tsv") if r["event"] in ("impact", "leave")] + rd("uniform.tsv")
    def blind_run(v):
        best = cur = 0
        for x in v <= a.vis:
            cur = cur + 1 if x else 0
            best = max(best, cur)
        return best
    ok_clip = {p for p, v in vis.items() if a.max_blind_run <= 0 or blind_run(v) <= a.max_blind_run}
    keep = []
    for r in rows:
        if os.path.abspath(r["path"]) not in ok_clip:
            continue
        b, c0 = int(r["boundary_frame"]), int(r["ctx_start"] or 0)
        w = min(a.min_vis_ctx, b - c0)
        if a.min_vis_ctx == 0 or (vis[os.path.abspath(r["path"])][b - w:b] > a.vis).any():
            keep.append(r)
    print(f"{a.root}: 행 {len(rows):,} → {len(keep):,} (뺌 {100 * (1 - len(keep) / len(rows)):.1f} %) · "
          f"영상 {len({r['path'] for r in keep}):,} / {len(vis):,} · 남은 영상 장 중 안 보임 "
          f"{100 * np.concatenate([v for p, v in vis.items() if p in ok_clip]) .__le__(a.vis).mean():.1f} %")
    s0, s1 = stats(rows, vis, a.vis, Cs, a.K), stats(keep, vis, a.vis, Cs, a.K)
    for C in Cs:
        (n0, b0), (n1, b1) = s0[C], s1[C]
        print(f"  C{C}: 문맥 {C} 장 전부 안 보이는 창 {100 * b0 / max(n0, 1):.1f} % ({n0:,} 행) → {100 * b1 / max(n1, 1):.1f} % ({n1:,} 행)")
    if a.write:
        os.makedirs(os.path.dirname(a.write) or ".", exist_ok=True)
        with open(a.write, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t", lineterminator="\n")
            w.writeheader()
            w.writerows(keep)
        print(f"  → {a.write}")


if __name__ == "__main__":
    main()
