#!/usr/bin/env python3
"""E1 · E2 사후 분석 (paper_lab 진행자 2, 2026-10-21): 운반 보상 · 쌍 정확도를 '물체 이동 크기' 로 나눠 본다.

왜: 다른 팀 측정 — 토큰은 칸 (16 px) 미만 이동에 매끄럽지 않다 (반 칸 이동이면 이웃 칸 평균과도 64–79 % 다름).
운반 오라클은 자리를 가장 가까운 칸으로 반올림해 붙인다. 그래서 "운반 오라클이 평균을 못 넘음" 이
(a) 표적 비국소성 탓인지 (b) 칸보다 작은 이동 탓인지 가르려면 이동 크기별로 나눠야 한다 (작성자 요청).

  python paper_lab/scripts/tspace_disp_bins.py            # TSR_OUT (기본 paper_lab/_scratch/tspace) 의 vobench_r*.npz 를 읽는다
산출: <TSR_OUT>/disp_bins.json + 표 출력. CPU 만.
"""
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "z_research/scripts/analysis"))
import design_headroom as H  # noqa: E402

OUT = Path(os.environ.get("TSR_OUT", REPO / "paper_lab/_scratch/tspace"))
C, K, CELL = 8, 8, 32.0                     # 라벨 좌표 512 px, 한 칸 = 32 라벨 px (= 입력 16 px)
SPACES = ("raw", "fp32", "smooth", "snr64", "snr256", "pca256")
DBINS = ((0, 1), (1, 2), (2, 4), (4, 99))   # 마지막 관측 대비 물체 이동 (칸)
VBINS = ((0, 0.5), (0.5, 1.0), (1.0, 99))   # 블록당 이동 (칸)


def load():
    Z = defaultdict(list)
    for f in sorted(OUT.glob("vobench_r*.npz")):
        z = np.load(f, allow_pickle=True)
        for k in z.files: Z[k].append(z[k])
    return {k: np.concatenate(v) for k, v in Z.items()}


def main():
    Z = load(); vids = list(Z["video_id"])
    rows = {r["video_id"]: r for r in csv.DictReader(open(REPO / "data_csv/vobench_v1/index.csv"))}
    disp = np.full((len(vids), K), np.nan); speed = np.full(len(vids), np.nan)
    for i, v in enumerate(vids):
        t = H.track_vobench(rows[v])
        if t is None: continue
        pos, last_vis, ctx_c = H.plan_positions(*t)
        disp[i] = np.linalg.norm(pos["move_or"] - ctx_c[C - 1][None], axis=1) / CELL
        speed[i] = np.linalg.norm(ctx_c[C - 1] - ctx_c[C - 2]) / CELL
    R = [rows[v] for v in vids]
    pos_ = np.array([r["variant"] in ("pos_a", "pos_b") for r in R])
    mov = np.array([r["motion"] != "flat_static" for r in R]); occ = np.array([r["occlusion"] for r in R])
    res = {"n_clips": len(vids)}
    # (1) 운반 보상을 (clip, 걸음) 단위로 이동 크기별: 1 − ΣL(운반 오라클) / ΣL(8 블록 평균)
    print("운반 보상 (1 − L(운반 오라클)/L(8 블록 평균)) · 가능 · 움직이는 물체, 마지막 관측 대비 이동 칸 구간별  [(clip, 걸음) 수]")
    for cell, sel in (("visible", pos_ & mov & (occ == "visible")), ("all_occ", pos_ & mov)):
        res[cell] = {}
        for sp in SPACES:
            if f"{sp}/mc_or/all" not in Z: continue
            mc, m8, cp, pa = (Z[f"{sp}/{m}/all"] for m in ("mc_or", "mean8", "copy", "paste_or"))
            line = {}
            for lo, hi in DBINS:
                msk = sel[:, None] & (disp >= lo) & (disp < hi)
                if msk.sum() == 0: continue
                line[f"{lo}-{hi}"] = dict(n=int(msk.sum()), carry=float(1 - mc[msk].sum() / m8[msk].sum()),
                                          carry_paste=float(1 - pa[msk].sum() / m8[msk].sum()), avg=float(1 - m8[msk].sum() / cp[msk].sum()))
            res[cell][sp] = line
            print(f"  {cell:8s} {sp:7s} " + " · ".join(f"[{k}] 운반 {100 * d['carry']:5.2f} % (한 블록 {100 * d['carry_paste']:5.2f}) 평균 {100 * d['avg']:4.1f} % n{d['n']}" for k, d in line.items()))
    # (2) 쌍 정확도를 블록당 이동 크기별 (가능 clip 의 문맥 속도)
    bv = {(rows[v]["source_block"], rows[v]["variant"]): i for i, v in enumerate(vids)}
    print("\n쌍 정확도 (전체 토큰) · 움직이는 물체, 블록당 이동 칸 구간별 — copy · mean8 · paste_or · mc_or · mc_cv  [쌍 수]")
    res["pairs"] = {}
    for sp in SPACES:
        if f"{sp}/mc_or/all" not in Z: continue
        A = {m: Z[f"{sp}/{m}/all"].mean(1) for m in ("copy", "mean8", "paste_or", "mc_or", "mc_cv")}
        acc = defaultdict(lambda: defaultdict(list))
        for i, r in enumerate(R):
            if r["variant"] not in ("pos_a", "pos_b") or not mov[i] or np.isnan(speed[i]): continue
            j = bv.get((r["source_block"], {"pos_a": "imp_ab", "pos_b": "imp_ba"}[r["variant"]]))
            if j is None: continue
            b = next(f"{lo}-{hi}" for lo, hi in VBINS if lo <= speed[i] < hi)
            for key in (f"{r['occlusion']}/{b}", f"{r['violation']}/{r['occlusion']}/{b}"):
                for m in A: acc[key][m].append(1.0 if A[m][j] > A[m][i] else (0.5 if A[m][j] == A[m][i] else 0.0))
        res["pairs"][sp] = {k: {m: [float(np.mean(v)), len(v)] for m, v in d.items()} for k, d in acc.items()}
        for key in sorted(k for k in acc if k.count("/") == 1 and k.split("/")[0] in ("visible", "late")):
            print(f"  {sp:7s} {key:16s} " + " ".join(f"{100 * np.mean(acc[key][m]):5.1f}" for m in A) + f"  [{len(acc[key]['copy'])}]")
    json.dump(res, open(OUT / "disp_bins.json", "w"), indent=1)
    print(f"\n블록당 이동 (칸) 분포, 움직이는 물체: 중앙값 {np.nanmedian(speed[mov]):.2f} · 4 분위 {np.nanpercentile(speed[mov], [25, 75]).round(2).tolist()}")


if __name__ == "__main__":
    main()
