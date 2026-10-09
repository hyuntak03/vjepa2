#!/usr/bin/env python3
"""SC2 분석 — 라벨 없는 실영상 지평 곡선. 입력 폴더 (e_ssv2_track.py 산출물) 여럿을 태그와 함께: python e_ssv2_track_analyze.py release=<dir> ariel=<dir> ...
움직임 선별: encoder 궤적 (enc) 이 문맥 끝 질의 칸 q 에서 슬롯 7 까지 ≥ 2 칸 (Chebyshev) 움직인 clip (정방향).
  agree_i = ‖p_i − enc_i‖∞ ≤ 1,  null_i = ‖p_i − enc_i(다른 clip)‖∞ ≤ 1 (clip 순열 20 회 평균),  copy_i = ‖q − enc_i‖∞ ≤ 1 (복사 기준선)
  prog_i = (p_i − q)·(enc_i − q) / |enc_i − q|²  (|enc_i − q| ≥ 1 인 clip, 중앙값)
  두 템플릿: 표준 (T = h_7[q], enc/p) · 인과 (Tc = hc_7[q], encc/pc).  CI = clip bootstrap (B=1000).
"""
import json, os, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "auto_research/exp_results/ssv2"; OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
res = {}


def boot(v, B=1000):
    v = v[np.isfinite(v)]
    if len(v) == 0: return [np.nan] * 3
    bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(B)]
    return [round(float(v.mean()), 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]


for arg in sys.argv[1:]:
    tag, d = arg.split("=", 1); d = Path(d)
    M = np.load(d / "track.npy")[:, 0]; Q = np.load(d / "query.npy")[:, 0]          # 정방향
    q = Q[:, [1, 0]]                                                                   # (n, 2) x, y
    rec = {}
    for tname, (ie, ip) in (("std", (0, 2)), ("causal", (1, 3))):
        enc = M[:, ie, :, [1, 0]].transpose(1, 0, 2) if False else np.stack([M[:, ie, :, 1], M[:, ie, :, 0]], -1)   # (n, 8, 2)
        pp = np.stack([M[:, ip, :, 1], M[:, ip, :, 0]], -1)
        move = np.abs(enc[:, 7] - q).max(-1) >= 2
        n = len(q); rows = []
        for i in range(8):
            ag = (np.abs(pp[:, i] - enc[:, i]).max(-1) <= 1).astype(float)
            nl = np.mean([(np.abs(pp[:, i] - enc[rng.permutation(n), i]).max(-1) <= 1).astype(float) for _ in range(20)], 0)
            cp = (np.abs(q - enc[:, i]).max(-1) <= 1).astype(float)
            dv = enc[:, i] - q; dd = (dv ** 2).sum(-1)
            pr = np.where(dd >= 1, ((pp[:, i] - q) * dv).sum(-1) / np.maximum(dd, 1e-9), np.nan)
            m = move
            rows.append(dict(slot=i, agree=boot(ag[m]), null=boot(nl[m]), agree_minus_null=boot((ag - nl)[m]), copy=boot(cp[m]),
                             agree_minus_copy=boot((ag - cp)[m]), prog_median=round(float(np.nanmedian(pr[m])), 3),
                             enc_disp=round(float(np.sqrt(dd[m]).mean()), 2)))
        rec[tname] = dict(n_moving=int(move.sum()), n=int(n), rows=rows)
    res[tag] = rec
json.dump(res, open(OUT / f"ssv2_track{os.environ.get('OUT_TAG', '')}.json", "w"), indent=1, default=float)
for tag, rec in res.items():
    for tname in ("std", "causal"):
        r = rec[tname]
        print(f"\n== {tag} · 템플릿 {tname} (움직임 clip {r['n_moving']}/{r['n']})  슬롯 0..7")
        print("  agree−null " + " ".join(f"{x['agree_minus_null'][0]:+.2f}" for x in r["rows"]))
        print("  agree      " + " ".join(f"{x['agree'][0]:.2f}" for x in r["rows"]) + "   | copy " + " ".join(f"{x['copy'][0]:.2f}" for x in r["rows"]))
        print("  agree−copy " + " ".join(f"{x['agree_minus_copy'][0]:+.2f}" for x in r["rows"]))
        print("  진행률 중앙 " + " ".join(f"{x['prog_median']:+.2f}" for x in r["rows"]) + "   | enc 변위 " + " ".join(f"{x['enc_disp']:.1f}" for x in r["rows"]))
