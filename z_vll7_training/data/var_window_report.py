#!/usr/bin/env python3
"""가변 창 (`data.var_window`) 의 **실제** 샘플링을 표로 — 돌리기 전에 비율을 눈으로 본다 (2026-10-16).

  python z_vll7_training/data/var_window_report.py --table k400=data_csv/vll7_k400_320/fps4_table.tsv \\
         --table ssv2=data_csv/vll7_ssv2/fps4_table.tsv --C 2,4,8,16,32 --K 2,4,8,16,32 --budget 950

왜: `var_window` 는 창 W = C+K 를 **패딩 없이** 채울 수 있는 데이터셋 중에서만 가중치로 고른다
(`min_eligible` 편 미만이면 그 칸에서 빠진다). 그래서 **명목 가중치 != 실제로 본 비율**이다.
vast 실측도 1:1:1 을 걸었는데 프레임 비율은 K710 38.3 / SSv2 20.0 / HowTo 41.8 이었다
(`git show origin/vast:z_training/RESULTS_2026-10-12.md` §3). 창 상한이 짧아지면 짧은 데이터셋 비중이 올라간다.

내는 것
  1. (C, K) 칸마다: 창 W, rank 당 batch B = clamp(budget // W, b_min, b_max), 데이터셋별 적격 편수
  2. 쓸 수 있는 조합 수와 **버려진 조합** (어떤 데이터셋도 못 채우는 칸)
  3. 조합을 균등하게 뽑았을 때의 **실제 기대 비율** — 데이터셋별 (step 비율 · clip 비율 · 프레임 비율)
"""
import argparse
import bisect
import collections

ap = argparse.ArgumentParser()
ap.add_argument("--table", action="append", required=True, metavar="NAME=PATH")
ap.add_argument("--weight", action="append", default=[], metavar="NAME=W", help="기본 1.0")
ap.add_argument("--C", default="2,4,8,16,32")
ap.add_argument("--K", default="2,4,8,16,32")
ap.add_argument("--budget", type=int, default=950)
ap.add_argument("--b-min", type=int, default=8)
ap.add_argument("--b-max", type=int, default=256)
ap.add_argument("--min-eligible", type=int, default=1000)
a = ap.parse_args()

W_OF = {}
for t in a.table:
    name, path = t.split("=", 1)
    n4 = sorted(int(ln.split("\t")[4]) for ln in open(path, encoding="utf-8") if int(ln.rstrip("\n").split("\t")[4]) > 0)
    W_OF[name] = n4
weights = {n: 1.0 for n in W_OF}
for w in a.weight:
    n, v = w.split("=", 1)
    weights[n] = float(v)

C_list = [int(x) for x in a.C.split(",")]
K_list = [int(x) for x in a.K.split(",")]


def n_ok(name, W):
    return len(W_OF[name]) - bisect.bisect_left(W_OF[name], W)


def batch(W):
    return max(a.b_min, min(a.b_max, a.budget // W))


print(f"표: " + ", ".join(f"{n} {len(v):,} 편" for n, v in W_OF.items()))
print(f"가중치: {weights} | token_budget {a.budget} (B = clamp(budget//W, {a.b_min}, {a.b_max})) | min_eligible {a.min_eligible:,}\n")
hdr = f"{'C':>3} {'K':>3} {'W':>3} {'B':>4}  " + "  ".join(f"{n:>10}" for n in W_OF)
print(hdr); print("-" * len(hdr))
combos, dropped = [], []
for C in C_list:
    for K in K_list:
        W = C + K
        el = [n for n in W_OF if n_ok(n, W) >= a.min_eligible]
        row = f"{C:>3} {K:>3} {W:>3} {batch(W):>4}  " + "  ".join(
            f"{n_ok(n, W):>10,}" + ("" if n in el else "*") for n in W_OF)
        if el:
            combos.append((C, K, W, el)); print(row)
        else:
            dropped.append((C, K)); print(row + "   <- 버림")
print(f"\n쓸 수 있는 조합 {len(combos)} / {len(C_list) * len(K_list)}" + (f", 버린 조합 {dropped}" if dropped else ""))
print("(* = 적격 편수가 min_eligible 미만이라 그 칸에서 그 데이터셋은 빠진다)")

# ── 균등 (C,K) 추첨에서의 기대 비율 ──────────────────────────────────────────
step = collections.Counter(); clip = collections.Counter(); frame = collections.Counter()
for C, K, W, el in combos:
    tot = sum(weights[n] for n in el)
    for n in el:
        pr = (weights[n] / tot) / len(combos)       # 이 칸이 뽑힐 확률 × 이 칸에서 n 이 뽑힐 확률
        step[n] += pr
        clip[n] += pr * batch(W)
        frame[n] += pr * batch(W) * W
print("\n균등 (C,K) 추첨에서 **실제** 기대 비율")
print(f"  {'데이터셋':<10} {'step':>8} {'clip':>8} {'frame':>8}")
for n in W_OF:
    print(f"  {n:<10} {100*step[n]:>7.1f}% {100*clip[n]/max(sum(clip.values()),1e-9):>7.1f}% "
          f"{100*frame[n]/max(sum(frame.values()),1e-9):>7.1f}%")
print(f"\n  step 당 평균 clip {sum(clip.values()):.1f} (rank 당) · 평균 창 {sum(frame.values())/max(sum(clip.values()),1e-9):.1f} 장")
