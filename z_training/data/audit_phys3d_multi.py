#!/usr/bin/env python3
"""phys3d_multi 점검 (2026-10-25) — C 별 공정성 · C × K · 시나리오 · 가림 · 배경. 문서 z_training/docs/PHYS3D_MULTI_DATASET_2026-10-25.md

    python3 z_training/data/audit_phys3d_multi.py --root <out>/val > <out>/val/audit.md

C 별로 같아야 하는 것 (원칙 §4 — 한 축이 C 와 얽히지 않게): 시나리오 구성 · 앵커 때 움직임 · 문맥 정지 · 크기 · 가림 종류 · 배경.
문맥 속력이 C 와 함께 줄어드는 것은 받아들였다 (사용자 2026-10-25) — 표에는 그대로 찍는다.
"""
import argparse
import collections
import csv
import json
import os

import numpy as np

CS = (2, 4, 8, 16, 32)
KS = (2, 4, 8, 16, 32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    man = list(csv.DictReader(open(os.path.join(a.root, "manifest.tsv")), delimiter="\t"))
    bnd = list(csv.DictReader(open(os.path.join(a.root, "boundary.tsv")), delimiter="\t"))
    fail = list(csv.DictReader(open(os.path.join(a.root, "failed.tsv")), delimiter="\t"))
    R = []
    for m in man:
        stem = os.path.join(a.root, "clips", m["name"], f"{m['name']}_{int(m['k']):05d}")
        z = np.load(stem + "_states.npz")
        C = int(m["C_star"])
        fps = float(json.load(open(stem + ".json"))["fps"])
        v0 = np.linalg.norm(z["vel3d"][:, 0], axis=-1) * fps
        va = np.linalg.norm(z["vel3d"], axis=-1) * fps
        R.append(dict(m=m, C=C, mov=float(v0[max(C - 1, 0)]) > 0.3, cstill=float((va[:C] < 0.05).all(1).mean()),
                      still=float((va < 0.05).all(1).mean()), sp=float(np.median(v0[:C])), size=float(z["size"][0]),
                      hid=float((z["vis_frac"][:, 0] <= 0.3).mean()), v2d=np.diff(z["pos"][:, 0].astype(np.float64), axis=0),
                      mvfrac=float((v0 >= 0.05).mean()), fps=fps))
    global CS
    CS = tuple(sorted({x["C"] for x in R}))         # perm 판은 C* ∈ {12..20}
    print(f"# phys3d_multi 점검 — `{a.root}`\n\n영상 {len(R):,} (실패 {len(fail):,})\n")
    print("## 1. C 별 (같아야 하는 것)\n")
    print("| C | 영상 | 실패 | event · occ_event · occ_any | 앵커 때 움직임 | 문맥 정지 | 전체 정지 | 문맥 속력 중앙 m/s | 주 물체 크기 중앙 | 안 보임 장 | 문맥 중 가림 | 들어온 물체 | open 배경 | 평균 시도 |")
    print("|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    fc = collections.Counter(int(f["C_star"]) for f in fail if f["C_star"])
    for C in CS:
        r = [x for x in R if x["C"] == C]
        if not r:
            continue
        an = collections.Counter(x["m"]["anchor"] for x in r)
        f = lambda key: np.mean([x[key] for x in r])
        print(f"| {C} | {len(r):,} | {fc[C]} | {an['event']} · {an['occ_event']} · {an['occ_any']} | {100 * f('mov'):.0f} % | {100 * f('cstill'):.0f} % | "
              f"{100 * f('still'):.0f} % | {np.median([x['sp'] for x in r]):.2f} | {np.median([x['size'] for x in r]):.2f} | {100 * f('hid'):.1f} % | "
              f"{100 * np.mean([x['m']['mid_occ'] == '1' for x in r]):.0f} % | {np.mean([int(x['m']['n_arrivals']) for x in r]):.2f} | "
              f"{100 * np.mean([x['m']['bg_mode'] == 'open' for x in r]):.0f} % | {np.mean([int(x['m']['attempt']) for x in r]):.1f} |")
    print("\n## 2. 시나리오 × C (영상 수)\n")
    names = sorted({x["m"]["name"] for x in R})
    print("| 시나리오 | " + " | ".join(f"C{c}" for c in CS) + " | 실패 |\n|---|" + "---:|" * (len(CS) + 1))
    fn = collections.Counter(f["name"] for f in fail)
    for n in names:
        cnt = collections.Counter(x["C"] for x in R if x["m"]["name"] == n)
        print(f"| {n} | " + " | ".join(str(cnt[c]) for c in CS) + f" | {fn[n]} |")
    print("\n## 3. 앵커 행 (C*, K) 적격 — 예측 K 장이 영상 안에 있고 (가림 행은) K ≥ pred_min\n")
    anc = [b for b in bnd if b["anchor"]]
    print("| 앵커 \\ K | " + " | ".join(f"K{k}" for k in KS) + " |\n|---|" + "---:|" * len(KS))
    for kind in ("event", "occ_event", "occ_any"):
        rs = [b for b in anc if b["anchor"] == kind]
        print(f"| {kind} ({len(rs):,}) | " + " | ".join(
            str(sum(int(b["n_frames"]) - int(b["boundary_frame"]) >= k and int(b["pred_min"]) <= k for b in rs)) for k in KS) + " |")
    print("\n## 4. 가림 · 배경\n")
    om = collections.Counter(o for x in R for o in x["m"]["occ_modes"].split(",") if o)
    print("가림 모드 (앵커 + 문맥 중): " + " · ".join(f"{k} {v:,}" for k, v in om.most_common()))
    bm = collections.Counter(x["m"]["bg_mode"] for x in R)
    print("\n배경: " + " · ".join(f"{k} {v:,}" for k, v in bm.most_common()) + " · HDRI 종류 " + str(len({x["m"]["hdri"] for x in R if x["m"]["hdri"]})))
    print("\n바닥: " + " · ".join(f"{k} {v}" for k, v in collections.Counter(x["m"]["floor_kind"] for x in R).most_common()))
    print("\n벽: " + " · ".join(f"{k} {v}" for k, v in collections.Counter(x["m"]["wall_kind"] for x in R).most_common()))
    print("\n주 물체 모양: " + " · ".join(f"{k} {v}" for k, v in collections.Counter(x["m"]["main_shape"] for x in R).most_common()))
    print("\n## 4-1. 지속률 (라벨, 주 물체 화면 속도) — 원칙 A · C (DATA_DESIGN_PRINCIPLES_2026-10-26)\n")
    print("지금 장 속도 v(t) (px / 저장 장) 중 L 초 뒤 v(t+L) 에 그대로 이어지는 몫 = (v(t+L) · v(t)) / |v(t)|², 움직이는 장 (≥ 8 px / 초) 에서 중앙값 · 아래 0 으로 자름.")
    print("자연 영상 (광학 흐름, 0.5 초 뒤): K710 0.20 · SSv2 0.10 · HowTo 0.06. 도구 검사: 등속 1 · 멈춤 0.\n")
    secs = (0.5, 1.0, 1.6, 2.0)                   # 1.6 초 = 예측 최대 (flow 판 8 블록 · 10 fps)
    print("| C | " + " | ".join(f"{t:g} 초" for t in secs) + " | 주 물체 움직이는 장 | 장면 정지 장 |\n|---|" + "---:|" * (len(secs) + 2))
    def persist(v, L, fps):
        sp = np.linalg.norm(v, axis=-1)
        t = np.nonzero(sp[:-L] >= 8.0 / fps)[0] if len(sp) > L else []   # 움직임 = 8 px / 초 이상
        if len(t) == 0:
            return []
        return list(np.maximum(0, (v[t + L] * v[t]).sum(-1) / sp[t] ** 2))
    for C in CS + ("all",):
        rs = [x for x in R if C == "all" or x["C"] == C]
        vals = {t: [] for t in secs}
        mv = st = 0.0
        for x in rs:
            vel = x["v2d"]
            for t_ in secs:
                vals[t_] += persist(vel, int(round(t_ * x["fps"])), x["fps"])
            mv += x["mvfrac"]
            st += x["still"]
        print(f"| {C} | " + " | ".join(f"{np.median(vals[t_]):.2f}" if vals[t_] else "-" for t_ in secs) +
              f" | {100 * mv / max(len(rs), 1):.0f} % | {100 * st / max(len(rs), 1):.0f} % |")
    print("\n## 5. 실패 이유\n")
    for k, v in collections.Counter(f["why"].split(";")[0] for f in fail).most_common(10):
        print(f"- {v} × {k}")
    print("\n## 재현\n\n```bash\npython3 z_training/data/audit_phys3d_multi.py --root " + a.root + "\n```")


if __name__ == "__main__":
    main()
