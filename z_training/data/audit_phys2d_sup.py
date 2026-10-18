#!/usr/bin/env python3
"""phys2d_sup 학습셋 편향 · 라벨 점검 (2026-10-18). 생성기 `gen_phys2d_sup.py` 산출물 (json · _states.npz · _seg.npz) 만 읽는다.

  python z_training/data/audit_phys2d_sup.py --root /local_datasets/world/training_data/phys2d_sup_v1/train \
         --val /local_datasets/world/training_data/phys2d_sup_v1/val > audit.md

vast `audit_phys2d.py` 의 항목 (구성 · 변수 · 공간 · 사건 행 · 예측창 복사/등속 · 가림막 지름길 · 누수) 에 더해
  B. 정체 — 주 물체 · 전체 물체의 모양 × 색 균형, 정체 ↔ 시나리오 · 사건 상관
  E. 영속 — 가림 횟수 (영상 · 물체), 가려진 길이, 가려진 시각 (영상 앞 · 중간 · 뒤), 가려진 동안 움직이는 비율
  H. 라벨 정합 — 위치 라벨 vs 보이는 세그 무게중심, 속도 라벨 vs 위치 중앙 차분
"""
import argparse
import collections
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

W = H = 256
SHAPES = ("circle", "square", "rectangle", "triangle", "pentagon", "hexagon")
COLORS = ("red", "orange", "yellow", "green", "cyan", "blue", "purple", "magenta")
FPS = 12


def per_clip(js, check_labels=False):
    m = json.load(open(js))
    L = np.load(js[:-5] + "_states.npz")
    pos, vel, present, vis, inf_f, hidden = L["pos"], L["vel"], L["present"], L["vis_frac"], L["inframe_frac"], L["hidden"]
    T, n = present.shape
    objs = m["objects"]
    inf = present & (inf_f > 0.5)
    main = [i for i, o in enumerate(objs) if o["role"] in ("main", "bob")]
    xs = pos[..., 0][inf]
    ys = pos[..., 1][inf]
    sp = np.linalg.norm(vel, axis=-1)
    mv = inf & (sp > 0.5)                                              # 0.5 px/장 넘게 움직이는 물체·장
    head = np.arctan2(-vel[..., 1][mv], vel[..., 0][mv])               # 위쪽 + (화면 y 는 아래로)
    vx = []
    for i in main:
        ok = inf[1:, i] & inf[:-1, i]
        d = (pos[1:, i, 0] - pos[:-1, i, 0])[ok]
        if len(d):
            vx.append(float(d.sum()))
    # 영속: 가려진 구간 (완전 가림 run)
    eps = []
    for i in range(n):
        h = hidden[:, i]
        s = None
        for f in range(T + 1):
            v = h[f] if f < T else False
            if v and s is None:
                s = f
            elif not v and s is not None:
                seen_before = bool((vis[:s, i] > 0.3).any())
                back = f < T and bool((vis[f:, i] > 0.3).any())
                disp = float(np.linalg.norm(pos[f - 1, i] - pos[s, i])) if np.isfinite(pos[[s, f - 1], i]).all() else 0.0
                eps.append(dict(obj=i, shape=objs[i]["shape"], start=s, len=f - s, seen_before=seen_before, back=back, rel_t=s / T,
                                disp=disp, moving=bool((sp[s:f, i] > 0.5).any())))
                s = None
    full = [e for e in eps if e["seen_before"] and e["back"] and e["len"] >= 2]
    per_obj = collections.Counter(e["obj"] for e in full)
    ev = m["events"]
    ev_shape = [(e["kind"], objs[e["obj"]]["shape"], objs[e["obj"]]["color"]) for e in ev]
    rows = []
    for r in m["event_rows"]:
        b, e = r["boundary"], min(r["end"], r["boundary"] + 8)
        if b < 2:
            continue
        Tt = np.arange(b, e)
        ok = inf[b - 1] & inf[b - 2] & inf[Tt].all(0)
        if not ok.any():
            continue
        sizes = np.array([o["bound"] for o in objs], np.float32)
        cp = np.linalg.norm(pos[Tt] - pos[b - 1], axis=-1)[:, ok]
        ln = np.linalg.norm(pos[Tt] - (pos[b - 1] + vel[b - 1] * (Tt - b + 1)[:, None, None]), axis=-1)[:, ok]
        rows.append(dict(event=r["event"], off=b - r["event_frame"], shape=objs[r["obj"]]["shape"],
                         copy_rel=float((cp / sizes[ok]).max(1).mean()), lin_rel=float((ln / sizes[ok]).max(1).mean())))
    lab = None
    if check_labels:
        seg = np.load(js[:-5] + "_seg.npz")["seg"]
        ce = []
        for f in range(0, T, 4):
            for i in range(n):
                if vis[f, i] > 0.97 and inf_f[f, i] > 0.97:
                    yy, xx = np.nonzero(seg[f] == i + 1)
                    ce.append(float(np.hypot(xx.mean() + 0.5 - pos[f, i, 0], yy.mean() + 0.5 - pos[f, i, 1])))
        d = (pos[2:] - pos[:-2]) / 2
        okv = np.isfinite(d).all(-1) & np.isfinite(vel[1:-1]).all(-1) & (L["impact"][1:-1] == 0) & (L["impact"][2:] == 0) & (L["impact"][:-2] == 0)
        lab = dict(ce=ce, ve=list(np.linalg.norm(d - vel[1:-1], axis=-1)[okv]))
    return dict(scenario=m["scenario"], category=m["category"], n=T, attempt=m["attempt"], flipped=m["flipped_lr"], bg=m["background"],
                walls=m["params"].get("walls"), n_spawn=m["params"].get("n_spawn", 0), n_distr=m["params"].get("n_distractor", 0),
                g=m["g_px_s2"], seed=m["seed"], n_obj=n, main_ident=m["result"]["main_identity"],
                idents=[o["identity"] for o in objs], roles=[o["role"] for o in objs], shapes=[o["shape"] for o in objs],
                el=[(o["shape"], o["elasticity"]) for o in objs if o["role"] == "main"], sizes=[o["size"] for o in objs],
                xs=np.histogram(xs, bins=8, range=(0, W))[0], ys=np.histogram(ys, bins=8, range=(0, H))[0],
                head=np.histogram(head, bins=8, range=(-np.pi, np.pi))[0], speed=np.histogram(sp[mv], bins=[0.5, 1, 2, 4, 8, 16, 1e9])[0],
                vx=vx, n_moving=int(mv.sum()), n_inf=int(inf.sum()), n_hidden=int(hidden.sum()), n_present=int(present.sum()),
                n_unseen=int((present & ~L["seen"]).sum()),
                eps=full, n_eps=len(full), max_obj_eps=max(per_obj.values()) if per_obj else 0, n_obj_multi=sum(v >= 2 for v in per_obj.values()),
                ev=ev_shape, rows=rows, occluders=[(o["kind"], o["origin"]) for o in m["occluders"]], decoy=m["params"].get("decoy_occluder"),
                lab=lab)


def pct(c, top=None):
    t = sum(c.values())
    items = sorted(c.items(), key=lambda kv: -kv[1])[:top]
    return ", ".join(f"{k} {v / t * 100:.1f}%" for k, v in items)


def _pc(args):
    return per_clip(*args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--val", default=None)
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--label-every", type=int, default=20, help="라벨 정합 검사를 몇 편마다 한 편씩 (세그를 읽어 느리다)")
    a = ap.parse_args()
    js = sorted(glob.glob(os.path.join(a.root, "clips", "*", "*.json")))
    with ProcessPoolExecutor(a.workers) as ex:
        C = list(ex.map(_pc, [(j, k % a.label_every == 0) for k, j in enumerate(js)], chunksize=32))
    print(f"# audit: `{a.root}` — {len(C):,} clip\n")

    print("## A. 구성")
    sc = collections.Counter(c["scenario"] for c in C)
    print(f"- 시나리오 {len(sc)} 종, 편수 min {min(sc.values())} / max {max(sc.values())} · 범주: {pct(collections.Counter(c['category'] for c in C))}")
    fp = os.path.join(a.root, "failed.tsv")
    if os.path.exists(fp):
        fl = [l.split("\t") for l in open(fp).read().splitlines()[1:]]
        print(f"- 생성 실패 (재시도 200 회 안에 검사 미통과) {len(fl)} 편" + (f": {pct(collections.Counter(f[0] for f in fl))}" if fl else ""))
    n = np.array([c["n"] for c in C])
    print(f"- 길이 (장, 1 장 = 1/12 초): min {n.min()} · p10 {np.percentile(n, 10):.0f} · 중앙 {np.median(n):.0f} · max {n.max()} · 128 상한 {np.mean(n == 128) * 100:.1f}%")
    no = np.array([c["n_obj"] for c in C])
    print(f"- 물체 수: 평균 {no.mean():.2f} · 분포 {pct(collections.Counter(no.tolist()))}")
    print(f"- 좌우 반전 {np.mean([c['flipped'] for c in C]) * 100:.1f}% · 옆벽 {np.mean([bool(c['walls']) for c in C]) * 100:.1f}%"
          f" · 들어오는 물체 있음 {np.mean([c['n_spawn'] > 0 for c in C]) * 100:.1f}% · 방해 물체 있음 {np.mean([c['n_distr'] > 0 for c in C]) * 100:.1f}%")
    print(f"- 배경: {pct(collections.Counter(c['bg'] for c in C))}")
    att = collections.defaultdict(list)
    for c in C:
        att[c["scenario"]].append(c["attempt"])
    print("- 재시도 평균 (시나리오별): " + " · ".join(f"{k} {np.mean(v):.1f}" for k, v in sorted(att.items(), key=lambda kv: -np.mean(kv[1]))))

    print("\n## B. 정체 (모양 6 × 색 8 = 48)")
    mi = collections.Counter(c["main_ident"] for c in C)
    cnt = np.array([mi.get(i, 0) for i in range(48)])
    print(f"- 주 물체 정체 (층화): 48 칸 편수 min {cnt.min()} / max {cnt.max()} (평균 {cnt.mean():.1f})")
    ms = collections.Counter(SHAPES[c["main_ident"] // 8] for c in C)
    mc = collections.Counter(COLORS[c["main_ident"] % 8] for c in C)
    print(f"- 주 물체 모양: {pct(ms)}\n- 주 물체 색: {pct(mc)}")
    al = collections.Counter(i for c in C for i in c["idents"])
    cnt = np.array([al.get(i, 0) for i in range(48)])
    print(f"- 전체 물체 정체: 48 칸 min {cnt.min()} / max {cnt.max()} (평균 {cnt.mean():.1f}, max/min {cnt.max() / max(cnt.min(), 1):.2f})")
    print(f"- 전체 물체 모양: {pct(collections.Counter(s for c in C for s in c['shapes']))}")
    print("- 범주별 전체 물체 모양 (stack_topple 상자는 square · rectangle 고정):")
    for cat in ["gravity", "permanence", "solidity", "mixed"]:
        print(f"  - {cat}: {pct(collections.Counter(s for c in C if c['category'] == cat for s in c['shapes']))}")
    print("- 역할: " + pct(collections.Counter(r for c in C for r in c["roles"])))
    print("- 사건 ↔ 모양 (사건을 겪은 물체의 모양 비율 — 전체 물체 모양 비율과 비교):")
    for k in ["impact", "leave", "occlude", "reappear"]:
        cc = collections.Counter(s for c in C for (kk, s, _) in c["ev"] if kk == k)
        if cc:
            print(f"  - {k} ({sum(cc.values()):,}): {pct(cc)}")
    print("- 사건 ↔ 색:")
    for k in ["impact", "occlude"]:
        cc = collections.Counter(col for c in C for (kk, _, col) in c["ev"] if kk == k)
        if cc:
            print(f"  - {k}: {pct(cc)}")

    print("\n## C. 변수")
    g = np.array([c["g"] for c in C])
    q = np.histogram(g, bins=5, range=(350, 900))[0]
    print(f"- g (사전 균등 350–900 px/s²): 5 등분 칸 " + " / ".join(f"{v / len(g) * 100:.1f}" for v in q) + " % (균등이면 20)")
    sz = np.concatenate([c["sizes"] for c in C])
    print(f"- 물체 크기 (반지름 상당, px): 4 분위 " + " / ".join(f"{v:.1f}" for v in np.percentile(sz, [25, 50, 75])))
    el = collections.defaultdict(list)
    for c in C:
        for s, e in c["el"]:
            el[s].append(e)
    print("- 주 물체 반발 계수 중앙 (모양별): " + " · ".join(f"{s} {np.median(v):.2f}" for s, v in el.items()))

    print("\n## D. 공간 · 운동 (화면 안에 반 넘게 있는 모든 물체 · 장)")
    xs, ys = sum(c["xs"] for c in C), sum(c["ys"] for c in C)
    print("- x 분포 (8 칸, 왼→오, %): " + " / ".join(f"{v / xs.sum() * 100:.1f}" for v in xs))
    print("- y 분포 (8 칸, 위→아래, %): " + " / ".join(f"{v / ys.sum() * 100:.1f}" for v in ys))
    nm, ni = sum(c["n_moving"] for c in C), sum(c["n_inf"] for c in C)
    print(f"- 움직이는 (> 0.5 px/장) 물체·장 비율 {nm / ni * 100:.1f}%")
    hd = sum(c["head"] for c in C)
    names = ["←(왼)", "↙", "↓(아래)", "↘", "→(오)", "↗", "↑(위)", "↖"]
    print("- 운동 방향 (8 방위, %): " + " · ".join(f"{nm_} {v / hd.sum() * 100:.1f}" for nm_, v in zip(names, hd)))
    spd = sum(c["speed"] for c in C)
    print("- 속력 (px/장) 칸 0.5–1 / 1–2 / 2–4 / 4–8 / 8–16 / >16: " + " / ".join(f"{v / spd.sum() * 100:.1f}" for v in spd) + " %")
    vx = np.concatenate([c["vx"] for c in C])
    print(f"- 주 물체 순 수평 이동: 오른쪽 {np.mean(vx > 1) * 100:.1f}% / 왼쪽 {np.mean(vx < -1) * 100:.1f}%")

    print("\n## E. 영속 (완전 가림 구간 — 그 전에 본 적 있고 다시 보이는 것만, 2 장 이상)")
    ne = np.array([c["n_eps"] for c in C])
    print(f"- 영상당 가림 구간 수: 0 {np.mean(ne == 0) * 100:.1f}% · 1 {np.mean(ne == 1) * 100:.1f}% · 2 {np.mean(ne == 2) * 100:.1f}%"
          f" · 3+ {np.mean(ne >= 3) * 100:.1f}% (평균 {ne.mean():.2f}, 총 {ne.sum():,})")
    mo = np.array([c["max_obj_eps"] for c in C])
    print(f"- 한 물체가 2 번 이상 가려졌다 나온 영상 {np.mean(mo >= 2) * 100:.1f}% · 3 번 이상 {np.mean(mo >= 3) * 100:.1f}%")
    print("- 범주별 가림 구간 있는 영상: " + " · ".join(
        f"{cat} {np.mean([c['n_eps'] > 0 for c in C if c['category'] == cat]) * 100:.1f}%" for cat in ["gravity", "permanence", "solidity", "mixed"]))
    E = [e for c in C for e in c["eps"]]
    if E:
        ln = np.array([e["len"] for e in E])
        print(f"- 가려진 길이 (장): 4 분위 " + " / ".join(f"{v:.0f}" for v in np.percentile(ln, [25, 50, 75])) + f" · max {ln.max()}"
              f" · 분포 2–3 {np.mean(ln <= 3) * 100:.1f}% / 4–8 {np.mean((ln >= 4) & (ln <= 8)) * 100:.1f}% / 9–16 {np.mean((ln >= 9) & (ln <= 16)) * 100:.1f}% / >16 {np.mean(ln > 16) * 100:.1f}%")
        rt = np.array([e["rel_t"] for e in E])
        print("- 가려지기 시작한 시각 (영상 길이 5 등분, %): " + " / ".join(f"{v / len(rt) * 100:.1f}" for v in np.histogram(rt, bins=5, range=(0, 1))[0]))
        print(f"- 가려진 동안 움직인 구간 {np.mean([e['moving'] for e in E]) * 100:.1f}% · 가려진 동안 이동 거리 중앙 {np.median([e['disp'] for e in E]):.1f} px")
        print(f"- 가림 구간의 물체 모양: {pct(collections.Counter(e['shape'] for e in E))}")
    hp = sum(c["n_hidden"] for c in C) / sum(c["n_present"] for c in C)
    print(f"- 완전히 가려진 물체·장 비율 {hp * 100:.1f}% · 아직 한 번도 안 보인 물체·장 (seen=0) {sum(c['n_unseen'] for c in C) / sum(c['n_present'] for c in C) * 100:.1f}%")

    print("\n## F. 사건 · boundary 행")
    evc = collections.Counter(k for c in C for (k, _, _) in c["ev"])
    print(f"- 사건 (json `events`, 충돌 dv ≥ 30 px/s): {pct(evc)} (총 {sum(evc.values()):,})")
    R = [dict(r, scenario=c["scenario"]) for c in C for r in c["rows"]]
    print(f"- boundary 행 {len(R):,} (예측 앞 8 장이 화면 안인 행만): {pct(collections.Counter(r['event'] for r in R))}")
    print("| 사건 | 행 | 복사 중앙 (반폭) | 등속 중앙 | 복사 < 0.5 | 등속 < 0.5 |\n|---|---:|---:|---:|---:|---:|")
    for k in ["impact", "leave", "occlude", "reappear", "all"]:
        rr = [r for r in R if k == "all" or r["event"] == k]
        if not rr:
            continue
        cp, ln = np.array([r["copy_rel"] for r in rr]), np.array([r["lin_rel"] for r in rr])
        print(f"| {k} | {len(rr):,} | {np.median(cp):.2f} | {np.median(ln):.2f} | {np.mean(cp < 0.5) * 100:.1f}% | {np.mean(ln < 0.5) * 100:.1f}% |")

    print("\n## G. 가림막 → 가림 지름길")
    for orig in ["scenario", "auto", "extra", "drift"]:
        cs = [c for c in C if any(o == orig for _, o in c["occluders"])]
        if cs:
            print(f"- 가림막 출처 `{orig}` 있는 영상 {len(cs):,} 중 가림 구간 있는 비율 {np.mean([c['n_eps'] > 0 for c in cs]) * 100:.1f}%")
    no = [c for c in C if not c["occluders"]]
    print(f"- 가림막 없는 영상 {len(no):,} ({len(no) / len(C) * 100:.1f}%) 중 가림 구간 {np.mean([c['n_eps'] > 0 for c in no]) * 100 if no else 0:.1f}%")
    cs = [c for c in C if c["scenario"] == "occ_pass"]
    if cs:
        d = [c for c in cs if c["decoy"]]
        print(f"- occ_pass 미끼 {len(d) / len(cs) * 100:.1f}% — 가림 구간 있는 비율: 미끼 {np.mean([c['n_eps'] > 0 for c in d]) * 100 if d else 0:.1f}%"
              f" / 궤적 위 {np.mean([c['n_eps'] > 0 for c in cs if not c['decoy']]) * 100:.1f}%")

    print("\n## H. 라벨 정합 (표본)")
    ce = np.concatenate([c["lab"]["ce"] for c in C if c["lab"]])
    ve = np.concatenate([c["lab"]["ve"] for c in C if c["lab"]])
    print(f"- 위치 라벨 vs 완전히 보이는 물체의 세그 무게중심 (px): 중앙 {np.median(ce):.3f} · p99 {np.percentile(ce, 99):.3f} · max {ce.max():.2f} (n={len(ce):,})")
    print(f"- 속도 라벨 (순간) vs 위치 중앙 차분 (px/장, 충돌 앞뒤 장 제외): 중앙 {np.median(ve):.4f} · p95 {np.percentile(ve, 95):.3f} · p99 {np.percentile(ve, 99):.3f}")

    print("\n## I. 누수")
    seeds = [c["seed"] for c in C]
    print(f"- seed 중복 {len(seeds) - len(set(seeds))}")
    if a.val:
        vs = [json.load(open(p))["seed"] for p in glob.glob(os.path.join(a.val, "clips", "*", "*.json"))]
        print(f"- val {len(vs):,} 편, train 과 seed 겹침 {len(set(vs) & set(seeds))}")


if __name__ == "__main__":
    main()
