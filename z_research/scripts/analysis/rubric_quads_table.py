#!/usr/bin/env python3
"""Rubric Office quads (실제 영상 object permanence) 채점표 — 창 (uniform · symmetric) × 모델 × 방향 (2026-10-31, z_analysis/release/04_rubric_office).

입력: z_analysis/release/04_rubric_office/te/<창>/ (transfer_gap_te 한 번 통과: l1 (n,K,8) · copy (n,8) · meta)
      + harness/<창>_{vast70,vast70g,vast70_copyar}/per_block.json (vast AR · vast AR + PFT · AR 공간 복사, 표준 하네스, 2026-10-31).
        AR 열은 블록별 LN(target) 공간이라 그 공간의 복사 (copy_ar) 와만 견준다 — 쌍 정확도만 비교
짝: 같은 영상 (block) · 같은 pair_id 의 (가능, 불가능). A = 문맥 '있음' (possible_present vs impossible_vanish = 사라짐) ·
    B = 문맥 '없음' (possible_absent vs impossible_appear = 생겨남). 판정 = s(불가능) > s(가능) (같으면 0.5), s = 미래 8 블록 평균 L1.
부분집합: all (20 영상) · clean (짝 문맥이 화소로 같은 16 영상 — handball1 · lightball4–6 제외) · 사건 종류별.
CI = 영상 재표집 2,000 (95 %).
  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python; $P z_research/scripts/analysis/rubric_quads_table.py → results/rubric_table.json
"""
import json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "z_research/scripts/analysis"))
import transfer_gap_r4_matrix as MX  # noqa: E402

BASE = REPO / "z_analysis/release/04_rubric_office"
DIRTY = {"6_handball/handball1_wave", "7_ballbounce/lightball4_occl1x", "7_ballbounce/lightball5_occl2x", "7_ballbounce/lightball6_occl4x"}


def pairs(m):
    from collections import defaultdict
    import csv
    idx = {r["video_id"]: r for r in csv.DictReader(open(m["index_csv"]))}
    by = defaultdict(dict)
    for i in range(len(m["video_id"])):
        by[(m["block_id"][i], m["pair_id"][i])]["pos" if str(m["plausible"][i]) == "1" else "imp"] = i
    out = []
    for (b, pid), v in by.items():
        r = idx[m["video_id"][v["pos"]]]
        out.append(dict(pos=v["pos"], imp=v["imp"], scene=r["source_block"], d=pid, event=r["condition"]))
    return out


def load_cols(d):
    """te 한 폴더 → (meta, {열: (n,) 미래 슬롯 평균 L1}). kind=ar 열은 블록별 LN(target) 공간이라 copy_ar (= copy_blk) 와만 견준다."""
    m = json.load(open(d / "meta.json"))
    assert np.load(d / "done.npy").all(), d
    l1 = np.load(d / "l1.npy").mean(-1)
    cols = {t: l1[:, k] for k, t in enumerate(m["pred_tags"])}
    cols["copy"] = np.load(d / "copy.npy").mean(-1)
    if (d / "copy_blk.npy").exists():
        cols["copy_ar"] = np.load(d / "copy_blk.npy").mean(-1)
    return m, cols


def main():
    res = {}
    for w in ("uniform", "symmetric"):
        d = BASE / "te" / w
        if not (d / "done.npy").exists():
            print("미완", w); continue
        m, cols = load_cols(d)
        P = pairs(m)
        J = {t: MX.judge(v, P) for t, v in cols.items()}
        # vast AR (ar_state) 은 한 번 통과 채점기가 못 읽어 표준 하네스로 돌렸다 (2026-10-31): harness/<창>_<열>/per_block.json
        #   vast70 · vast70g = predictor, vast70_copyar = 같은 AR 공간 (블록별 LN(target)) 의 복사 기준선 (surprise.ar_baseline=copy_last)
        for t in ("vast70", "vast70g", "vast70_copyar"):
            f = BASE / "harness" / f"{w}_{t}" / "per_block.json"
            if not f.exists():
                continue
            pv = json.load(open(f))["per_video_surprise"]
            v = np.array([float(pv[vid]) for vid in m["video_id"]])
            J["copy_ar" if t == "vast70_copyar" else t] = MX.judge(v, P)
        tags = [t for t in J if t not in ("copy", "copy_ar")]
        sc = np.array([p["scene"] for p in P])
        out = {}
        subsets = {"all": lambda p: True, "clean": lambda p: p["scene"] not in DIRTY}
        for ev in sorted({p["event"] for p in P}):
            subsets[ev] = (lambda e: (lambda p: p["event"] == e))(ev)
        for sname, f in subsets.items():
            for di in ("A", "B", "AB"):
                msk = np.array([f(p) and (di == "AB" or p["d"] == di) for p in P])
                if not msk.any():
                    continue
                r = {"n": int(msk.sum())}
                for t in tags + [c for c in ("copy", "copy_ar") if c in J]:
                    r[t] = MX.boot(J[t][msk], sc[msk])
                for t in tags:
                    if t != "release":
                        r[f"{t}-release"] = MX.boot(J[t][msk], sc[msk], J["release"][msk])
                    base = "copy_ar" if t.startswith("vast") else "copy"
                    r[f"{t}-{base}"] = MX.boot(J[t][msk], sc[msk], J[base][msk])
                if "vast70g" in J and "vast70" in J:
                    r["vast70g-vast70"] = MX.boot(J["vast70g"][msk], sc[msk], J["vast70"][msk])
                out[f"{sname}|{di}"] = r
        res[w] = out
        print(f"\n[{w}]")
        for key in ("all|A", "all|B", "all|AB", "clean|A", "clean|B", "clean|AB", "occlusion|AB", "containment|AB", "out_of_frame|AB"):
            if key in out:
                r = out[key]
                print(f"  {key:16s} n {r['n']:2d} | " + " ".join(f"{t} {r[t][0]:5.1f}" for t in tags + [c for c in ("copy", "copy_ar") if c in r])
                      + f" | rel−copy {r['release-copy'][0]:+.0f} [{r['release-copy'][1]:+.0f},{r['release-copy'][2]:+.0f}]"
                      + f" · a1−rel {r['a1-release'][0]:+.0f} [{r['a1-release'][1]:+.0f},{r['a1-release'][2]:+.0f}]")
    (BASE / "results").mkdir(parents=True, exist_ok=True)
    json.dump(res, open(BASE / "results/rubric_table.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
