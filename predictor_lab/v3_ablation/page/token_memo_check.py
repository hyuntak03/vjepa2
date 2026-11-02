"""v3 · v3t 에서 "다시 나올 자리" 를 토큰 단위로 외울 수 있나 — 다른 세션이 보고한 네 수치의 재확인 (2026-11-01).

팔 라벨 (arms/<split>/<block>/pos_a.npz) 의 center_px (프레임 64 × 2, target 화면 중심 px, 256 px, 가려진 동안도 있음) 를 쓴다.
모델 샘플 s = 0–31 ↔ 원본 프레임 2s. 토큰 = 16 px 칸 (16 × 16). 미래 = 샘플 16–31, 튜블릿 = 샘플 2 개 (미래 8 칸).
  ① 경로 일치: 시험 장면의 미래 토큰 칸 순서가 학습 장면 중 하나와 똑같은가 (튜블릿 8 칸 · 샘플 16 칸 두 가지, 튜블릿 칸 = 두 샘플 중심 평균)
  ② 끝 칸 일치: 샘플 31 의 토큰 칸이 학습 끝 칸 집합에 있는가
  ③ 틀 궤적 오차: 학습에서 (방향 × 운동 종류) 묶음마다 미래 화면 궤적 평균 → 시험 미래와의 평균 거리 (토큰). 속도 크기는 안 쓴다
       방향 = 문맥에서 보인 샘플의 가로 이동 부호 (|Δx| < 4 px 면 정지), 운동 종류 = 장소 × 빗면 방향 × target 움직임
       절대 화면 위치 판과, 문맥 끝 (샘플 15) 위치를 뺀 상대 판 둘 다
  ④ 등속 오차: 마지막으로 보인 두 샘플 (보임 비율 ≥ 0.5) 의 속도로 곧게 이은 미래와의 평균 거리 (토큰)
  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python; $P predictor_lab/v3_ablation/page/token_memo_check.py
"""
import csv
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

R = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
SETS = {"v3": ("data_csv/blender_track_v3/train_arms.csv", "data_csv/blender_track_v3/voe_test/index.csv"),
        "v3t": ("data_csv/blender_track_v3t/train_arms.csv", "data_csv/blender_track_v3t/voe_test/index.csv")}
S = np.arange(32) * 2          # 샘플 → 원본 프레임


def load(row, key):
    p = row.get("labels") or str(Path(row["file"]).with_suffix(".npz"))
    z = np.load(p)
    c, am, vi = z["center_px"], z["amodal_n"], z["visible_n"]
    if c.ndim == 3:                                                      # 원본 라벨 (물체 여럿) → target = 0 번. 팔 라벨 (arms/…/pos_a.npz) 은 target 만
        c, am, vi = c[:, 0], am[:, 0], vi[:, 0]
    c = c[S].astype(np.float64)                                          # (32, 2)
    am, vi = am[S].astype(float), vi[S].astype(float)
    vis = np.where(am > 0, vi / np.maximum(am, 1), 0.0)
    return dict(c=c, vis=vis, kind=f"{row['place']}|{row.get('ramp_mode') or '-'}|{row['target_mode']}",
                static=row["target_mode"] == "static", block=row["block_id"], key=key)


def rows_of(path, test):
    out = []
    for r in csv.DictReader(open(R / path)):
        if r["variant"] != "pos_a":
            continue
        if not test and r.get("render_timing_ok") != "1":
            continue
        out.append(r)
    return out


def feats(d):
    c, vis = d["c"], d["vis"]
    seen = [t for t in range(16) if vis[t] >= 0.5]
    dx = c[seen[-1], 0] - c[seen[0], 0] if len(seen) >= 2 else 0.0
    d["dir"] = 0 if abs(dx) < 4 else int(np.sign(dx))
    d["grp"] = (d["dir"], d["kind"])
    tok = np.clip(np.floor(c / 16), 0, 15).astype(int)
    d["path_s"] = tuple(map(tuple, tok[16:]))
    tub = np.clip(np.floor((c[16::2] + c[17::2]) / 2 / 16), 0, 15).astype(int)
    d["path_t"] = tuple(map(tuple, tub))
    d["end"] = tuple(tok[31])
    if len(seen) >= 2:
        a, b = seen[-2], seen[-1]
        v = (c[b] - c[a]) / (b - a)
        pred = c[b] + v * (np.arange(16, 32) - b)[:, None]
        d["cv_err"] = float(np.linalg.norm(pred - c[16:], axis=1).mean() / 16)
    else:
        d["cv_err"] = np.nan
    return d


def main():
    for name, (tr, te) in SETS.items():
        with ThreadPoolExecutor(32) as ex:
            train = list(ex.map(lambda r: feats(load(r, "tr")), rows_of(tr, False)))
            test = list(ex.map(lambda r: feats(load(r, "te")), rows_of(te, True)))
        ps = {d["path_s"] for d in train}; pt = {d["path_t"] for d in train}; ends = {d["end"] for d in train}
        g_abs, g_rel = defaultdict(list), defaultdict(list)
        for d in train:
            g_abs[d["grp"]].append(d["c"][16:]); g_rel[d["grp"]].append(d["c"][16:] - d["c"][15])
        m_abs = {k: np.mean(v, 0) for k, v in g_abs.items()}; m_rel = {k: np.mean(v, 0) for k, v in g_rel.items()}
        for d in test:
            k = d["grp"]
            d["tpl_abs"] = float(np.linalg.norm(m_abs[k] - d["c"][16:], axis=1).mean() / 16) if k in m_abs else np.nan
            d["tpl_rel"] = float(np.linalg.norm(m_rel[k] + d["c"][15] - d["c"][16:], axis=1).mean() / 16) if k in m_rel else np.nan
            d["in_s"] = d["path_s"] in ps; d["in_t"] = d["path_t"] in pt; d["in_end"] = d["end"] in ends

        def rep(sub, lab):
            f = lambda k: 100 * np.mean([x[k] for x in sub])
            g = lambda k: np.nanmean([x[k] for x in sub])
            print(f"  {lab:14s} n={len(sub):5d} | 경로 일치 튜블릿 {f('in_t'):5.1f} % · 샘플 {f('in_s'):5.1f} % | 끝 칸 {f('in_end'):5.1f} % | "
                  f"틀 궤적 절대 {g('tpl_abs'):.2f} · 상대 {g('tpl_rel'):.2f} 토큰 | 등속 {g('cv_err'):.2f} 토큰")
        hid = np.array([d["c"][15] for d in test])
        print(f"\n== {name}: 학습 장면 {len(train)} · 시험 장면 {len(test)} · 묶음 {len(m_abs)} | 문맥 끝 (샘플 15) target 화면 위치 평균 {hid.mean(0).round(1)} px · 표준편차 {hid.std(0).round(1)} px")
        import pandas as pd
        pd.DataFrame([dict(block=d["block"], static=d["static"], dir=d["dir"], kind=d["kind"], in_t=d["in_t"], in_s=d["in_s"], in_end=d["in_end"],
                           tpl_abs=d["tpl_abs"], tpl_rel=d["tpl_rel"], cv_err=d["cv_err"]) for d in test]).to_csv(
            R / f"predictor_lab/results/v3_ablation/token_memo_{name}.csv", index=False)
        rep(test, "전체")
        rep([d for d in test if not d["static"]], "target 움직임")
        rep([d for d in test if d["static"]], "target 정지")


if __name__ == "__main__":
    main()
