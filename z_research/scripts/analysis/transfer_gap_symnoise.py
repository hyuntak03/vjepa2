#!/usr/bin/env python3
"""transfer_gap Q3 · E31c — 대칭 잡음 바닥: 확률적 반올림 seed 4 개에서 (PFT − 릴리즈) 와 뒤집힘 방향. 2026-10-30, [X2] R2.

왜: R1 의 잡음 바닥 (결정적 재양자화 한 판) 은 뒤집힘이 PFT 쪽으로만 몰려 대칭 잡음이 아니었다. 평균 0 잡음 (확률적 반올림, E[y] = v)
    을 seed 마다 걸어 (가) 보고된 IntPhys 1 하락이 잡음 바닥 안인가, (나) 뒤집힘이 여전히 한쪽으로 몰리나 (H3-L) 를 본다.
정의 (CPU, 산출물만 읽음)
  판: 원본 (로더가 실수로 크기 조정, 반올림 없음) · 결정적 반올림 (R1 NF / E12 T0) · 확률적 반올림 seed 1–4 (transfer_gap_srframes.py).
  IntPhys 1: Garrido skip2_w32 칸 (시작점마다 C 최소 → 평균), 180 쌍. VOBench split v2 test: 고정 창 c16t32 표준 surprise, 720 쌍.
  정확도 · (PFT − 릴리즈) 를 판마다. 뒤집힘 = 원본 대비 쌍 판정이 바뀐 쌍 (잃음 = 정답 → 오답, 얻음 = 반대), seed 를 모아 부호 검정 (양측 이항).
  체계적 이동 = 영상 surprise 의 (판 − 원본) 평균 (절대 · 상대).
  판정: PFT 의 잃음 : 얻음 몰림 p < 0.05 이고 릴리즈는 아니면 H3-L 쪽. (PFT − 릴리즈) 의 seed 분포가 원본 값 (−6.7, w32 칸) 을 포함하면서
        0 도 포함하면 '하락은 잡음 바닥 안', seed 가 모두 0 아래로 몰리면 '하락은 잡음에 견고'.
출력: z_analysis/transfer_gap/results/q3_r2/E31c_symnoise.json

  /data/hyuntak/anaconda3/envs/vjepa2/bin/python z_research/scripts/analysis/transfer_gap_symnoise.py            # 정본 (frames_src)
  /data/hyuntak/anaconda3/envs/vjepa2/bin/python z_research/scripts/analysis/transfer_gap_symnoise.py --which sr # 첫 판 기록 → E31c_unmatched_noise.json
⚠️ 정정 (2026-10-30): 첫 판 (frames_sr, seed = 파일 경로) 은 matched 쌍 문맥에 서로 다른 잡음을 넣어 문맥 무결성을 깼다 (확인: VOBench 문맥 60 / 60 프레임 불일치).
   그 판은 '대칭 잡음 바닥' 이 아니라 '쌍마다 다른 문맥 잡음' 판이다 → E31c_unmatched_noise.json 으로 따로 남긴다.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
C = Path("/data2/local_datasets/world/world_analysis/cache/transfer_gap")
RUNS = REPO / "z_analysis/transfer_gap/results/q3_r2/runs"
NF = REPO / "z_analysis/transfer_gap/results/q3/runs"
IP1 = REPO / "cvpr/results/eval/intphys1"
OUT = REPO / "z_analysis/transfer_gap/results/q3_r2/E31c_symnoise.json"
SEEDS = (1, 2, 3, 4)
WHICH = "src"   # src = 원본 내용으로 seed (matched 문맥 보존, 정본) · sr = 첫 판 (경로 seed, matched 문맥이 깨짐 — 기록)


def binom_p(k, n):
    if n == 0:
        return None
    pk = [math.comb(n, i) * 0.5 ** n for i in range(n + 1)]
    obs = pk[k]
    return round(min(1.0, sum(p for p in pk if p <= obs + 1e-15)), 5)


def garrido(pw, combo="skip2_w32"):
    out = {}
    for v, lst in pw.items():
        per = defaultdict(list)
        for c, st, Cc, s in lst:
            if c == combo:
                per[st].append(s)
        if per:
            out[v] = float(np.mean([min(x) for x in per.values()]))
    return out


def pairs_ip1():
    import csv
    by = defaultdict(dict)
    for r in csv.DictReader(open("/local_datasets/world/world_analysis/IntPhys1_dev_videos/index.csv")):
        by[(r["block_id"], r["pair_id"])]["pos" if r["plausible"] == "1" else "imp"] = r["video_id"]
    return [(v["pos"], v["imp"]) for v in by.values() if "pos" in v and "imp" in v]


def judge(s, pairs):
    return np.array([1.0 if s[i] > s[p] else (0.5 if s[i] == s[p] else 0.0) for p, i in pairs])


def flips(j0, j1):
    lost = int(((j0 == 1) & (j1 < 1)).sum()); gain = int(((j0 < 1) & (j1 == 1)).sum())
    return lost, gain


def main():
    import argparse
    global WHICH, OUT
    ap = argparse.ArgumentParser(); ap.add_argument("--which", default="src", choices=["src", "sr"]); a = ap.parse_args()
    WHICH = a.which
    if WHICH == "sr":
        OUT = OUT.with_name("E31c_unmatched_noise.json")
    res = {"which": WHICH, "definition": __doc__.split("정의 (CPU, 산출물만 읽음)")[1].split("출력:")[0].strip(), "intphys1_skip2_w32": {}, "vobench_test": {}}
    # ---------- IntPhys 1
    P = pairs_ip1()
    S = {}
    for m, tag in (("release", "vith"), ("vob_pft", "vith_release_vobftg")):
        S[("orig", m)] = garrido(json.load(open(IP1 / f"intphys1_dev_{tag}_wgarrido_w32/per_window.json"))["windows"])
        S[("det", m)] = garrido(json.load(open(NF / f"NF_intphys1_requant_{tag}_w32/per_window.json"))["windows"])
        for s in SEEDS:
            f = RUNS / (f"E31c2_ip1_src{s}_{tag}_skip2w32/per_window.json" if WHICH == "src" else f"E31c_ip1_sr{s}_{tag}_skip2w32/per_window.json")
            if f.exists():
                S[(f"sr{s}", m)] = garrido(json.load(open(f))["windows"])
    R = {"n_pairs": len(P), "versions": {}}
    J = {k: judge(v, P) for k, v in S.items()}
    pooled = {m: [0, 0] for m in ("release", "vob_pft")}
    for ver in ["orig", "det"] + [f"sr{s}" for s in SEEDS]:
        if (ver, "release") not in J or (ver, "vob_pft") not in J:
            continue
        row = {m: round(100 * float(J[(ver, m)].mean()), 2) for m in ("release", "vob_pft")}
        row["pft_minus_release"] = round(row["vob_pft"] - row["release"], 2)
        if ver != "orig":
            for m in ("release", "vob_pft"):
                lo, ga = flips(J[("orig", m)], J[(ver, m)])
                row[f"flips_{m}"] = {"lost": lo, "gained": ga}
                d = np.array([S[(ver, m)][v] - S[("orig", m)][v] for v in S[("orig", m)] if v in S[(ver, m)]])
                o = np.array([S[("orig", m)][v] for v in S[("orig", m)] if v in S[(ver, m)]])
                row[f"shift_{m}"] = {"abs": round(float(d.mean()), 6), "rel_pct": round(100 * float((d / o).mean()), 4), "frac_up": round(float((d > 0).mean()), 3)}
                if ver.startswith("sr"):
                    pooled[m][0] += lo; pooled[m][1] += ga
        R["versions"][ver] = row
    R["pooled_sr_flips"] = {m: {"lost": v[0], "gained": v[1], "binom_p": binom_p(v[0], v[0] + v[1])} for m, v in pooled.items()}
    srs = [R["versions"][f"sr{s}"]["pft_minus_release"] for s in SEEDS if f"sr{s}" in R["versions"]]
    if srs:
        R["sr_pft_minus_release"] = {"values": srs, "min": min(srs), "max": max(srs), "mean": round(float(np.mean(srs)), 2)}
    res["intphys1_skip2_w32"] = R
    # ---------- VOBench test (te)
    def te_load(name):
        d = C / name
        if not (d / "summary.json").exists():
            return None
        m = json.load(open(d / "meta.json"))
        l1 = np.load(d / "l1.npy").mean(2); cp = np.load(d / "copy.npy").mean(1)
        return m, {m["pred_tags"][k]: dict(zip(m["video_id"], l1[:, k])) for k in range(l1.shape[1])} | {"copy": dict(zip(m["video_id"], cp))}
    base = te_load("te_E12_Torig")
    meta = base[0]
    grp = defaultdict(dict)
    for v, b, p, pl in zip(meta["video_id"], meta["block_id"], meta["pair_id"], meta["plausible"]):
        grp[(b, p)]["pos" if str(pl) == "1" else "imp"] = v
    PV = [(g["pos"], g["imp"]) for g in grp.values() if "pos" in g and "imp" in g]
    V = {"orig": base[1]}
    det = te_load("te_E12_T0")
    if det:
        V["det"] = det[1]
    for s in SEEDS:
        x = te_load(f"te_E31c2_src{s}" if WHICH == "src" else f"te_E31c_sr{s}")
        if x:
            V[f"sr{s}"] = x[1]
    R = {"n_pairs": len(PV), "versions": {}}
    pooled = {m: [0, 0] for m in ("release", "vobftg", "copy")}
    for ver, d in V.items():
        row = {m: round(100 * float(judge(d[m], PV).mean()), 2) for m in ("release", "vobftg", "copy")}
        row["pft_minus_release"] = round(row["vobftg"] - row["release"], 2)
        if ver != "orig":
            for m in ("release", "vobftg", "copy"):
                lo, ga = flips(judge(V["orig"][m], PV), judge(d[m], PV))
                row[f"flips_{m}"] = {"lost": lo, "gained": ga, "frac_flipped": round((lo + ga) / len(PV), 4)}
                dd = np.array([d[m][v] - V["orig"][m][v] for v in V["orig"][m]]); oo = np.array([V["orig"][m][v] for v in V["orig"][m]])
                row[f"shift_{m}"] = {"abs": round(float(dd.mean()), 6), "rel_pct": round(100 * float((dd / oo).mean()), 4), "frac_up": round(float((dd > 0).mean()), 3)}
                if ver.startswith("sr"):
                    pooled[m][0] += lo; pooled[m][1] += ga
        R["versions"][ver] = row
    R["pooled_sr_flips"] = {m: {"lost": v[0], "gained": v[1], "binom_p": binom_p(v[0], v[0] + v[1])} for m, v in pooled.items()}
    res["vobench_test"] = R
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(res["intphys1_skip2_w32"], ensure_ascii=False, indent=0)[:3000])
    print(json.dumps({k: v for k, v in res["vobench_test"].items()}, ensure_ascii=False, indent=0)[:3000])
    print("→", OUT)


if __name__ == "__main__":
    main()
