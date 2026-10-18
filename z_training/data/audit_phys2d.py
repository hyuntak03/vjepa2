#!/usr/bin/env python3
"""phys2d 학습셋 편향 점검 (2026-10-18). 생성기 `gen_phys2d.py` 산출물의 json · npz · boundary.tsv 만 읽는다 (영상 디코딩 없음).

  python z_training/data/audit_phys2d.py --root /workspace/dataset/training_data/phys2d_v1/train
  python z_training/data/audit_phys2d.py --root .../train --val .../val        # train/val seed 겹침도 검사

점검 항목 (결과를 markdown 으로 찍는다 — z_training/docs/PHYS2D_DATASET_2026-10-18.md 에 옮겨 적는다)
  A. 구성     시나리오 · 길이 · 재시도 (rejection) · 좌우 반전 · 배경 · 옆벽 · 들어오는 물체 · 방해 물체
  B. 변수     g · 반발 계수 · 물체 크기 · 모양 — 사전분포 대비 rejection 후 치우침
  C. 공간     물체 위치 (x, y) 분포 · 운동 방향 (좌/우) 균형
  D. 사건 행  종류 · 시나리오별 행 수 · b − 사건 장 offset · 사건 물체 모양 (모양 → 사건 지름길?)
  E. 예측창   행마다 예측 구간의 복사 오차 · 등속 외삽 오차 (사건 종류별) — 복사로 풀리는 행이 남았는가
  F. 가림막   가림막이 있는 클립 중 실제 가림 사건이 있는 비율 (가림막 존재 → 가림 지름길?)
  G. 누수     train/val seed 겹침, 영상 경로 중복
"""
import argparse
import collections
import csv
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

W = H = 256


def load(js):
    m = json.load(open(js))
    z = np.load(js[:-5] + "_states.npz")
    return m, z["pos"], z["visible"]


def per_clip(js):
    m, pos, vis = load(js)
    inf = (pos[..., 0] > -5) & (pos[..., 0] < W + 5) & (pos[..., 1] > -5) & (pos[..., 1] < H + 5)
    roles = [o["role"] for o in m["objects"]]
    sizes = np.array([o["size"] for o in m["objects"]], np.float32)
    main = [i for i, r in enumerate(roles) if r in ("main", "bob")]
    # C. 위치 · 운동 방향 (화면 안, 주 물체)
    xs = np.concatenate([pos[inf[:, i], i, 0] for i in main]) if main else np.zeros(0)
    ys = np.concatenate([pos[inf[:, i], i, 1] for i in main]) if main else np.zeros(0)
    vx = []
    for i in main:
        ok = inf[1:, i] & inf[:-1, i]
        d = (pos[1:, i, 0] - pos[:-1, i, 0])[ok]
        if len(d):
            vx.append(float(d.sum()))
    # 들어오는 물체의 입구 (위 / 옆)
    entry = []
    for i, r in enumerate(roles):
        if r == "spawn" and inf[:, i].any():
            p = pos[np.argmax(inf[:, i]), i]
            entry.append("top" if p[1] < 20 else "side")
    # E. 행마다 예측창 오차 (앞 8 장): 복사 = 문맥 마지막 장 그대로, 등속 = 마지막 두 장 속도로 외삽
    rows = []
    for r in m["event_rows"]:
        b, e = r["boundary"], min(r["end"], r["boundary"] + 8)
        T = np.arange(b, e)
        ok = inf[b - 1] & inf[b - 2] & inf[T].all(0) if b >= 2 else inf[b - 1] & inf[T].all(0)
        if not ok.any():
            continue
        cp = np.linalg.norm(pos[T] - pos[b - 1], axis=-1)[:, ok]                       # (t, obj)
        v = pos[b - 1] - pos[b - 2] if b >= 2 else np.zeros_like(pos[b - 1])
        ln = np.linalg.norm(pos[T] - (pos[b - 1] + v * (T - b + 1)[:, None, None]), axis=-1)[:, ok]
        obj = r["obj"]
        rows.append(dict(event=r["event"], off=b - r["event_frame"], kind=m["objects"][obj]["kind"], role=r["obj_role"],
                         copy=float(cp.max(1).mean()), lin=float(ln.max(1).mean()),
                         copy_rel=float((cp / sizes[ok]).max(1).mean()), lin_rel=float((ln / sizes[ok]).max(1).mean())))
    occl = [o["kind"] for o in m["occluders"]]
    return dict(scenario=m["scenario"], category=m["category"], n=m["n_frames"], attempt=m["attempt"], flipped=m["flipped_lr"],
                bg=m["background"], walls=m["params"].get("walls"), n_spawn=m["params"].get("n_spawn", 0),
                n_distr=m["params"].get("n_distractor", 0), g=m["g_px_s2"], seed=m["seed"],
                el=[o["elasticity"] for o, r in zip(m["objects"], roles) if r == "main"],
                size=[o["size"] for o, r in zip(m["objects"], roles) if r == "main"],
                kinds=[o["kind"] for o, r in zip(m["objects"], roles) if r == "main"],
                xs=np.histogram(xs, bins=8, range=(0, W))[0], ys=np.histogram(ys, bins=8, range=(0, H))[0],
                vx=vx, entry=entry, rows=rows, occluders=occl,
                has_occ_event=any(r["event"] == "occlude" for r in m["event_rows"]),
                decoy=m["params"].get("decoy_occluder"))


def pct(c):
    t = sum(c.values())
    return ", ".join(f"{k} {v / t * 100:.1f}%" for k, v in sorted(c.items(), key=lambda kv: -kv[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--val", default=None)
    ap.add_argument("--workers", type=int, default=128)
    a = ap.parse_args()
    js = sorted(glob.glob(os.path.join(a.root, "clips", "*", "*.json")))
    with ProcessPoolExecutor(a.workers) as ex:
        C = list(ex.map(per_clip, js, chunksize=64))
    print(f"# audit: {a.root} — {len(C):,} clip\n")

    print("## A. 구성")
    sc = collections.Counter(c["scenario"] for c in C)
    print(f"- 시나리오 {len(sc)} 종, 편수 min {min(sc.values())} / max {max(sc.values())}")
    n = np.array([c["n"] for c in C])
    print(f"- 길이 (장): min {n.min()} · p10 {np.percentile(n, 10):.0f} · 중앙 {np.median(n):.0f} · max {n.max()} · 128 장 (상한) {np.mean(n == 128) * 100:.1f}%")
    print(f"- 좌우 반전 {np.mean([c['flipped'] for c in C]) * 100:.1f}% · 옆벽 {np.mean([bool(c['walls']) for c in C]) * 100:.1f}%"
          f" · 들어오는 물체 있음 {np.mean([c['n_spawn'] > 0 for c in C]) * 100:.1f}% · 방해 물체 있음 {np.mean([c['n_distr'] > 0 for c in C]) * 100:.1f}%")
    print(f"- 배경: {pct(collections.Counter(c['bg'] for c in C))}")
    print("- 재시도 (통과까지 버린 시도 수, 평균) — 시나리오별:")
    att = collections.defaultdict(list)
    for c in C:
        att[c["scenario"]].append(c["attempt"])
    print("  " + " · ".join(f"{k} {np.mean(v):.1f}" for k, v in sorted(att.items(), key=lambda kv: -np.mean(kv[1]))))

    print("\n## B. 변수 (rejection 후)")
    g = np.array([c["g"] for c in C])
    q = np.histogram(g, bins=5, range=(350, 900))[0]
    print(f"- g (사전 균등 350–900): 5 등분 칸 비율 " + " / ".join(f"{v / len(g) * 100:.1f}" for v in q) + " % (균등이면 20 씩)")
    el = np.concatenate([c["el"] for c in C])
    print(f"- 주 물체 반발 계수: 중앙 {np.median(el):.2f}, 4 분위 " + " / ".join(f"{v:.2f}" for v in np.percentile(el, [25, 50, 75])))
    for cat in ["gravity", "permanence", "solidity"]:
        k = collections.Counter(x for c in C if c["category"] == cat for x in c["kinds"])
        print(f"- 주 물체 모양 ({cat}): {pct(k)}")

    print("\n## C. 공간")
    xs = sum(c["xs"] for c in C)
    ys = sum(c["ys"] for c in C)
    print("- 주 물체 x 분포 (8 칸, 왼→오, %): " + " / ".join(f"{v / xs.sum() * 100:.1f}" for v in xs))
    print("- 주 물체 y 분포 (8 칸, 위→아래, %): " + " / ".join(f"{v / ys.sum() * 100:.1f}" for v in ys))
    vx = np.concatenate([c["vx"] for c in C])
    print(f"- 주 물체 순 수평 이동: 오른쪽 {np.mean(vx > 1) * 100:.1f}% / 왼쪽 {np.mean(vx < -1) * 100:.1f}% / 거의 0 {np.mean(abs(vx) <= 1) * 100:.1f}%")
    for cat in ["gravity", "permanence", "solidity"]:
        v = np.concatenate([c["vx"] for c in C if c["category"] == cat])
        print(f"  - {cat}: 오른쪽 {np.mean(v > 1) * 100:.1f}% / 왼쪽 {np.mean(v < -1) * 100:.1f}%")
    print(f"- 들어오는 물체 입구: {pct(collections.Counter(e for c in C for e in c['entry']))}")

    print("\n## D. 사건 행")
    R = [dict(r, scenario=c["scenario"], category=c["category"]) for c in C for r in c["rows"]]
    print(f"- 행 {len(R):,} (예측 앞 8 장이 화면 안인 행만 셈) · 종류: {pct(collections.Counter(r['event'] for r in R))}")
    print(f"- b − 사건 장 (offset): {pct(collections.Counter(r['off'] for r in R))}")
    for ev in ["impact", "occlude", "leave"]:
        print(f"- {ev} 행의 사건 물체 모양: {pct(collections.Counter(r['kind'] for r in R if r['event'] == ev))}"
              f" · 역할: {pct(collections.Counter(r['role'] for r in R if r['event'] == ev))}")
    rs = collections.Counter(r["scenario"] for r in R)
    print("- 시나리오별 행 수: " + " · ".join(f"{k} {v:,}" for k, v in rs.most_common()))
    print("- 시나리오별 사건 종류 (가림 · 이탈 이 있는 곳):")
    for s in sorted(rs):
        cc = collections.Counter(r["event"] for r in R if r["scenario"] == s)
        if cc.get("occlude") or cc.get("leave"):
            print(f"  - {s}: {dict(cc)}")

    print("\n## E. 예측창 (앞 8 장) — 복사 · 등속 외삽 오차, 물체 반폭 단위 (1 = 반지름만큼 틀림)")
    print("| 사건 | 행 | 복사 중앙 | 등속 중앙 | 복사 < 0.5 반폭 행 | 등속 < 0.5 반폭 행 |")
    print("|---|---:|---:|---:|---:|---:|")
    for ev in ["impact", "occlude", "leave", "all"]:
        rr = [r for r in R if ev == "all" or r["event"] == ev]
        cp = np.array([r["copy_rel"] for r in rr])
        ln = np.array([r["lin_rel"] for r in rr])
        print(f"| {ev} | {len(rr):,} | {np.median(cp):.2f} | {np.median(ln):.2f} | {np.mean(cp < 0.5) * 100:.1f}% | {np.mean(ln < 0.5) * 100:.1f}% |")

    print("\n## F. 가림막 → 가림 지름길")
    for kind in ["screen", "cup"]:
        cs = [c for c in C if kind in c["occluders"]]
        if cs:
            print(f"- `{kind}` 가 있는 클립 {len(cs):,} 중 가림 행이 있는 비율 {np.mean([c['has_occ_event'] for c in cs]) * 100:.1f}%")
    cs = [c for c in C if c["scenario"] == "occ_pass"]
    if cs:
        d = [c for c in cs if c["decoy"]]
        nd = [c for c in cs if not c["decoy"]]
        print(f"- occ_pass 미끼 가림막 {len(d) / len(cs) * 100:.1f}% — 가림 행 비율: 미끼 {np.mean([c['has_occ_event'] for c in d]) * 100:.1f}% / 궤적 위 {np.mean([c['has_occ_event'] for c in nd]) * 100:.1f}%")
    no = [c for c in C if not c["occluders"]]
    print(f"- 가림막 없는 클립 {len(no):,} 중 가림 행 {np.mean([c['has_occ_event'] for c in no]) * 100:.1f}%")

    print("\n## G. 누수")
    seeds = [c["seed"] for c in C]
    print(f"- train seed 중복 {len(seeds) - len(set(seeds))}")
    if a.val:
        vs = [json.load(open(p))["seed"] for p in glob.glob(os.path.join(a.val, "clips", "*", "*.json"))]
        print(f"- val {len(vs):,} 편, train 과 seed 겹침 {len(set(vs) & set(seeds))}")


if __name__ == "__main__":
    main()
