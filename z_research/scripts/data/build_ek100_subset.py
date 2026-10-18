"""EK100 train 의 1/N 부분집합 후보를 만들고, 채점 지표 관점의 대리 지표로 비교한다 (2026-10-15).

지표 = val mean class recall@5 — **val 과 full train 에 둘 다 있는 클래스** (action 1,114 · verb 73 · noun 201) 의 평균. 학습에 한 번도 안 나온
클래스는 맞힐 수 없으므로, 부분집합이 이 클래스를 몇 개 덮는지 · 클래스당 몇 개를 주는지가 성능 하한을 정한다 (1/3 영상 단위 자르기가 action 34 % 를 잃었다).

후보 (목표 = full 구간 수 / N):
  video      영상 단위 무작위 (옛 third 와 같은 방식)
  segment    구간 단위 무작위
  strat      action 클래스 층화 — 지표 클래스는 최소 1 개 보장, 나머지는 빈도 비례
  strat_min3 같은데 지표 클래스 최소 3 개 (그 클래스 표본이 3 개 미만이면 전부)
  capped     action 클래스당 상한 k (긴 꼬리 평탄화) — 지표 클래스 최소 1 개 보장, k 는 목표 크기에 맞춤
대리 지표: 지표 클래스 덮음 (≥1 · ≥3) · 클래스당 중앙 표본 · 영상 / 참가자 수 · verb · noun 빈도 분포의 total variation (full 대비).

  python z_research/scripts/data/build_ek100_subset.py --n 10            # → data_csv/ek100/EPIC_100_train_tenth_<후보>_seed0.csv + 비교표
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
ANN = Path("/data/hyuntak/project/2026/2027_cvpr/epic-kitchens-100-annotations")
OUT = REPO / "data_csv/ek100"


def metric_classes(tr, va):
    return {k: sorted(set(va[k]) & set(tr[k])) for k in ("verb_class", "noun_class", "action")}


def pick_video(tr, n, rng):
    vids = tr.video_id.unique().copy(); rng.shuffle(vids)
    out, tot = [], 0
    for v in vids:
        if tot >= n: break
        out.append(v); tot += (tr.video_id == v).sum()
    return tr[tr.video_id.isin(out)]


def pick_segment(tr, n, rng):
    return tr.sample(n=n, random_state=int(rng.integers(1 << 30)))


def pick_quota(tr, n, rng, need, cap=None, floor=1):
    """action 클래스마다 몫. cap 없으면 빈도 비례 (strat), 있으면 min(빈도, cap) (capped). 지표 클래스는 최소 floor (그 클래스 표본 수 한도 안)."""
    cnt = tr.action.value_counts()
    if cap is None:
        lo, hi = 0.0, 1.0
        for _ in range(40):                                   # 비율 r 을 이분 탐색 — 몫 = max(지표면 1, round(r · 빈도))
            r = (lo + hi) / 2
            q = np.maximum(np.where(cnt.index.isin(need), np.minimum(floor, cnt.values), 0), np.round(r * cnt.values)).astype(int)
            lo, hi = (r, hi) if q.sum() < n else (lo, r)
    else:
        q = np.minimum(cnt.values, cap)
        q = np.maximum(q, np.where(cnt.index.isin(need), 1, 0))
    q = pd.Series(q, index=cnt.index)
    parts = [g.sample(n=int(min(q[a], len(g))), random_state=int(rng.integers(1 << 30))) for a, g in tr.groupby("action") if q.get(a, 0) > 0]
    return pd.concat(parts)


def find_cap(tr, n, need):
    cnt = tr.action.value_counts().values
    for cap in range(1, 2000):
        if np.maximum(np.minimum(cnt, cap), 1).sum() >= n:      # 지표 클래스 최소 1 은 거의 안 바뀐다
            return cap
    return 2000


def tv(a, b):
    p = a.value_counts(normalize=True); q = b.value_counts(normalize=True)
    idx = p.index.union(q.index)
    return 0.5 * np.abs(p.reindex(idx, fill_value=0) - q.reindex(idx, fill_value=0)).sum()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=10); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
    tr = pd.read_csv(ANN / "EPIC_100_train.csv"); va = pd.read_csv(ANN / "EPIC_100_validation.csv")
    for d in (tr, va): d["action"] = d.verb_class.astype(str) + "_" + d.noun_class.astype(str)
    M = metric_classes(tr, va); n = round(len(tr) / a.n)
    print(f"full train {len(tr)} 구간 · 영상 {tr.video_id.nunique()} · 참가자 {tr.participant_id.nunique()} | 지표 클래스 action {len(M['action'])} · verb {len(M['verb_class'])} · noun {len(M['noun_class'])} | 목표 {n}")
    rng = np.random.default_rng(a.seed)
    cap = find_cap(tr, n, M["action"])
    C = {"video": pick_video(tr, n, rng), "segment": pick_segment(tr, n, rng),
         "strat": pick_quota(tr, n, rng, M["action"]), "strat_min3": pick_quota(tr, n, rng, M["action"], floor=3),
         "capped": pick_quota(tr, n, rng, M["action"], cap=cap)}
    rows = []
    for name, s in C.items():
        r = dict(후보=name, 구간=len(s), 영상=s.video_id.nunique(), 참가자=s.participant_id.nunique())
        for k, lab in (("action", "action"), ("verb_class", "verb"), ("noun_class", "noun")):
            vc = s[k].value_counts(); m = M[k]
            r[f"{lab} 덮음≥1"] = f"{100 * np.mean([c in vc.index for c in m]):.1f}"
            r[f"{lab} 덮음≥3"] = f"{100 * np.mean([vc.get(c, 0) >= 3 for c in m]):.1f}"
        r["verb 분포 TV"] = f"{tv(s.verb_class, tr.verb_class):.3f}"; r["noun 분포 TV"] = f"{tv(s.noun_class, tr.noun_class):.3f}"
        rows.append(r)
        tag = {1: "full", 3: "third", 10: "tenth"}.get(a.n, f"1of{a.n}")
        OUT.mkdir(parents=True, exist_ok=True)
        s.drop(columns=["action"]).sort_values(["video_id", "start_frame"]).to_csv(OUT / f"EPIC_100_train_{tag}_{name}_seed{a.seed}.csv", index=False)
    T = pd.DataFrame(rows)
    print(f"(capped 의 클래스당 상한 = {cap})")
    print(T.to_string(index=False))
    third = OUT / "EPIC_100_train_third_seed0.csv"
    if third.exists():
        s = pd.read_csv(third); s["action"] = s.verb_class.astype(str) + "_" + s.noun_class.astype(str)
        print(f"\n(참고) 옛 1/3 영상 단위: 구간 {len(s)} · action 덮음≥1 {100 * np.mean([c in set(s.action) for c in M['action']]):.1f} · "
              f"verb {100 * np.mean([c in set(s.verb_class) for c in M['verb_class']]):.1f} · noun {100 * np.mean([c in set(s.noun_class) for c in M['noun_class']]):.1f} → 실측 action 21.8 (full 35.3)")


if __name__ == "__main__":
    main()
