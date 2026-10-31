#!/usr/bin/env python3
"""binding 학습 평가 — 한 번 통과 채점 (transfer_gap_te) 결과를 판 × 모델 × 방향 표로 (2026-10-31, z_analysis/release/03_binding_train).

판 (열):
  v2lv  VOBench_v2 test (궤적 묶음 반대쪽) late + visible — 학습 생성기 안, 처음 보는 궤적. 조건 × 위반 × 방향 (A = 문맥 A, B)
  v11   v11 split test 가림 이동 사라짐 (224 block, 원본 화소 사본) — 반쯤 다른 생성기 (배경 · 메시 일부 공유)
  v11r  v11_realistic 가림 이동 사라짐 — 실물 메시
  ip1   IntPhys 1 late uni 이동 · 가림 (60 쌍, 쌍 정확도) — 진짜 다른 생성기
  ip2   IntPhys 2 late uni permanence (120 쌍, 쌍 정확도) — 진짜 다른 생성기
모델: release · a0 (L1, VOBench_v2 PFT) · a1 (+ binding) · a2 (+ binding + 운동 head) · vobftg (원 PFT, VOBench_v1 + gravity) · copy (마지막 문맥 복사)
판정 · CI: transfer_gap_r4_matrix 와 같다 (s(불가능) > s(가능), 장면 bootstrap 2,000). Δ = 같은 쌍 차이.
  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python; $P z_research/scripts/analysis/release_bind_te_table.py  → results/te_table.json
"""
import json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "z_research/scripts/analysis"))
import transfer_gap_r4_matrix as MX  # noqa: E402

TE = REPO / "z_analysis/release/03_binding_train/te"
OUT = REPO / "z_analysis/release/03_binding_train/results/te_table.json"
COLS = {"v2lv": "v2lv", "v11": "v11sub3", "v11r": "v11r", "ip1": "ip1", "ip2": "ip2"}


def v2_pairs(m):
    from collections import defaultdict
    by = defaultdict(dict)
    for i in range(len(m["video_id"])):
        by[(m["block_id"][i], m["pair_id"][i])]["pos" if str(m["plausible"][i]) == "1" else "imp"] = i
    out = []
    for (b, pid), v in by.items():
        if "pos" in v and "imp" in v:
            i = v["pos"]
            out.append(dict(pos=v["pos"], imp=v["imp"], scene=b, d="A" if m["variant"][i] == "pos_a" else "B",
                            occ=m["occlusion"][i], viol=m["violation"][i], k=m["k"][i]))
    return out


def main():
    res = {}
    for col, sub in COLS.items():
        d = TE / sub
        if not (d / "done.npy").exists() or not np.load(d / "done.npy").all():
            print("미완", col); continue
        m, l1, cp = MX.load(d)
        tags = list(m["pred_tags"])
        P = v2_pairs(m) if col == "v2lv" else MX.pairs(m, col)
        J = {t: MX.judge(l1[:, k], P) for k, t in enumerate(tags)}; J["copy"] = MX.judge(cp, P)
        sc = np.array([p["scene"] for p in P])
        groups = {}
        if col == "v2lv":
            for occ in ("late", "visible"):
                for viol in ("permanence", "shape", "color", "all"):
                    for di in ("A", "B", "AB"):
                        groups[f"{occ}|{viol}|{di}"] = np.array([p["occ"] == occ and (viol == "all" or p["viol"] == viol) and (di == "AB" or p["d"] == di) for p in P])
        else:
            for di in sorted({p["d"] for p in P}):
                groups[di] = np.array([p["d"] == di for p in P])
        out = {}
        for g, msk in groups.items():
            if not msk.any():
                continue
            r = {"n": int(msk.sum())}
            for t in tags + ["copy"]:
                r[t] = MX.boot(J[t][msk], sc[msk])
            for t in ("a1", "a2"):
                r[f"{t}-a0"] = MX.boot(J[t][msk], sc[msk], J["a0"][msk])
            for t in ("a0", "a1", "a2", "vobftg"):
                r[f"{t}-release"] = MX.boot(J[t][msk], sc[msk], J["release"][msk])
                r[f"{t}-copy"] = MX.boot(J[t][msk], sc[msk], J["copy"][msk])
            out[g] = r
        res[col] = out
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
    show = {"v2lv": ["late|permanence|A", "late|permanence|B", "late|shape|AB", "late|color|AB", "late|all|AB", "visible|all|AB"],
            "v11": ["A", "B"], "v11r": ["A", "B"], "ip1": ["P"], "ip2": ["P"]}
    for col, keys in show.items():
        if col not in res:
            continue
        for g in keys:
            if g not in res[col]:
                continue
            r = res[col][g]
            print(f"{col:5s} {g:20s} n {r['n']:4d} | " + " ".join(f"{t} {r[t][0]:5.1f}" for t in ("release", "a0", "a1", "a2", "vobftg", "copy"))
                  + f" | a1−a0 {r['a1-a0'][0]:+.1f} [{r['a1-a0'][1]:+.0f},{r['a1-a0'][2]:+.0f}] · a1−copy {r['a1-copy'][0]:+.1f} [{r['a1-copy'][1]:+.0f},{r['a1-copy'][2]:+.0f}]"
                  + f" · a1−rel {r['a1-release'][0]:+.1f} [{r['a1-release'][1]:+.0f},{r['a1-release'][2]:+.0f}]")


if __name__ == "__main__":
    main()
