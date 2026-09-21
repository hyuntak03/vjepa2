#!/usr/bin/env python3
"""RollOut_v2 — p 의 물체다운 토큰이 '슬롯' 이 아니라 '마지막 관측 자리에서의 이동 거리' 로 사라지는가. 2026-09-19.

A. 자 (v5 attentive, p) 의 attention 질량 — 진실 물체 3×3 (균등 0.035). flat_v/a/d 가능 clip 전부. 거리 × 슬롯 표 + 회귀.
B. 자 없는 검사 (파라미터 0) — 슬롯 t 의 LN(h) 진실 물체 토큰을 템플릿으로 p 슬롯 t 토큰 256 개와 cos, argmax 가 진실 1 칸 안이면 적중
   (우연 ≈ 9/256). flat_v/a/d 250 + arc/ledge 200 clip (seed 0). ledge 는 진실 (낙하) 자리 vs 등속 (부유) 자리 cos 대비 (3×3 최대 − 전체 중앙값).
거리 = |진실 위치(슬롯 t) − 문맥 마지막 튜블릿 위치| (px, 288 좌표, 1 칸 = 18 px).
입력: exp_results/v5/attentive_pooling/p/preds.npz, 캐시 rollout_v2_vith/{predictor,target}, data_csv/rollout_v2/index_probe.csv.
출력: exp_results/v5/locality/distance_decay.{json,md}
  python z_research/scripts/analysis/rollout2_distance_decay.py
"""
from __future__ import annotations
import csv, json, re
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
C = Path("/local_datasets/world/world_analysis/cache/rollout_v2_vith")
PRED = ROOT / "z_research/RollOutV2/exp_results/v5/attentive_pooling/p/preds.npz"
INDEX = ROOT / "data_csv/rollout_v2/index_probe.csv"
OUT = ROOT / "z_research/RollOutV2/exp_results/v5/locality"
BINS = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 120)]
arr = lambda s: np.array([float(v) for v in re.split(r"[;,\s|]+", s.strip()) if v])
ln = lambda x: (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-6)


def track(r):
    return np.stack([arr(r["px_x_by_sample"]).reshape(16, 2).mean(1), arr(r["px_y_by_sample"]).reshape(16, 2).mean(1)], 1)


def m3(w, x, y):
    cx, cy = int(x // 18), int(y // 18)
    return sum(w[r * 16 + c] for r in range(cy - 1, cy + 2) for c in range(cx - 1, cx + 2) if 0 <= r < 16 and 0 <= c < 16)


def table(d, t, v, lo=30):
    return [[(float(v[(d >= a) & (d < b) & (t == s)].mean()) if ((d >= a) & (d < b) & (t == s)).sum() >= lo else None) for s in range(8)] for a, b in BINS]


def reg(d, t, v):
    X = np.c_[np.ones_like(d), t / 7, d / 100]; return np.linalg.lstsq(X, v, rcond=None)[0].tolist()


def main():
    P = np.load(PRED); idx = {r["video_id"]: r for r in csv.DictReader(INDEX.open())}
    A = P["attn"].astype(np.float32); A /= A.sum(-1, keepdims=True); out = {}
    # ---- A. 자 attention
    rows = []
    for s in ("flat_v", "flat_a", "flat_d"):
        for i in np.where((P["scenario"] == s) & (P["plausible"] == 1))[0]:
            xy = track(idx[P["video_id"][i]]); tr = (P["truth"][i] + 1) * 144
            for t in range(8):
                rows.append((t, np.linalg.norm(tr[t] - xy[7]), m3(A[i, t], *tr[t])))
    t, d, m = (np.array(c, dtype=float) for c in zip(*rows))
    out["A_ruler_attention"] = {"n_rows": len(rows), "table_dist_by_slot": table(d, t, m), "reg_const_slot_dist": reg(d, t, m)}
    # ---- B. 자 없는 검사 (pf.py 와 같은 표본: seed 0, 시나리오 순서대로)
    ids = json.load(open(C / "meta.json"))["video_ids"]; pos = {v: i for i, v in enumerate(ids)}
    Pm = np.load(C / "predictor.npy", mmap_mode="r"); Tm = np.load(C / "target.npy", mmap_mode="r")
    rng = np.random.default_rng(0); R = []
    for s, n in (("flat_v", 250), ("flat_a", 250), ("flat_d", 250), ("arc", 200), ("ledge", 200)):
        ks = np.where((P["scenario"] == s) & (P["plausible"] == 1))[0]; ks = rng.choice(ks, min(n, len(ks)), replace=False)
        for i in ks:
            vid = P["video_id"][i]; j = pos[vid]; xy = track(idx[vid])
            p = np.asarray(Pm[j], dtype=np.float32).reshape(8, 256, -1); h = ln(np.asarray(Tm[j], dtype=np.float32).reshape(16, 256, -1)[8:])
            tr = (P["truth"][i] + 1) * 144; v = xy[7] - xy[6]; pn = p / np.linalg.norm(p, axis=-1, keepdims=True)
            for t_ in range(8):
                c = np.clip((tr[t_] // 18).astype(int), 0, 15); tpl = h[t_, c[1] * 16 + c[0]]; tpl = tpl / np.linalg.norm(tpl)
                sim = pn[t_] @ tpl; ay, ax = divmod(int(sim.argmax()), 16)
                cv = np.clip(((xy[7] + (t_ + 1) * v) // 18).astype(int), 0, 15)
                mx = lambda cc: max(sim[yy * 16 + xx] for yy in range(cc[1] - 1, cc[1] + 2) for xx in range(cc[0] - 1, cc[0] + 2) if 0 <= yy < 16 and 0 <= xx < 16)
                R.append((s, t_, float(np.linalg.norm(tr[t_] - xy[7])), max(abs(ax - c[0]), abs(ay - c[1])) <= 1, mx(c) - np.median(sim), mx(cv) - np.median(sim)))
    S = np.array([r[0] for r in R]); T = np.array([r[1] for r in R], float); D = np.array([r[2] for r in R]); H = np.array([r[3] for r in R], float)
    M = np.array([r[4] for r in R]); MC = np.array([r[5] for r in R]); fl = np.isin(S, ["flat_v", "flat_a", "flat_d"])
    out["B_paramfree"] = {"n_clips": {str(s): int((S == s).sum() // 8) for s in np.unique(S)},
                          "flat_table_dist_by_slot": table(D[fl], T[fl], H[fl], lo=25), "flat_reg_const_slot_dist": reg(D[fl], T[fl], H[fl]),
                          "arc_hit_by_slot": [float(H[(S == "arc") & (T == s)].mean()) for s in range(8)],
                          "ledge_hit_by_slot": [float(H[(S == "ledge") & (T == s)].mean()) for s in range(8)],
                          "ledge_contrast_truth_fell": [float(M[(S == "ledge") & (T == s)].mean()) for s in range(8)],
                          "ledge_contrast_constv_float": [float(MC[(S == "ledge") & (T == s)].mean()) for s in range(8)]}
    OUT.mkdir(parents=True, exist_ok=True); json.dump(out, open(OUT / "distance_decay.json", "w"), indent=1)
    f = lambda x: "  -  " if x is None else f"{x:.2f}"
    L = ["# p 물체다운 토큰 — 거리 × 슬롯 (2026-09-19)\n", "## A. 자 attention 질량 (진실 3×3, 균등 0.035), flat 가능 clip 전부\n",
         "| 거리 (px) | " + " | ".join(f"슬롯 {s}" for s in range(8)) + " |", "|---|" + "---:|" * 8]
    L += [f"| {a}–{b} | " + " | ".join(f(x) for x in row) + " |" for (a, b), row in zip(BINS, out["A_ruler_attention"]["table_dist_by_slot"])]
    c = out["A_ruler_attention"]["reg_const_slot_dist"]; L.append(f"\n회귀: 질량 = {c[0]:.2f} {c[1]:+.2f}·(슬롯/7) {c[2]:+.2f}·(거리/100 px). n = {len(rows)} (clip×슬롯)\n")
    B = out["B_paramfree"]; L += ["## B. 자 없는 검사 — 적중률 (우연 ≈ 0.035)\n", f"표본 clip: {B['n_clips']}\n",
         "| flat 거리 (px) | " + " | ".join(f"슬롯 {s}" for s in range(8)) + " |", "|---|" + "---:|" * 8]
    L += [f"| {a}–{b} | " + " | ".join(f(x) for x in row) + " |" for (a, b), row in zip(BINS, B["flat_table_dist_by_slot"])]
    c = B["flat_reg_const_slot_dist"]; L.append(f"\n회귀 (flat): 적중 = {c[0]:.2f} {c[1]:+.2f}·(슬롯/7) {c[2]:+.2f}·(거리/100 px)\n")
    for k, lab in (("arc_hit_by_slot", "arc 적중"), ("ledge_hit_by_slot", "ledge 적중 (진실 = 낙하)"),
                   ("ledge_contrast_truth_fell", "ledge cos 대비, 낙하 자리"), ("ledge_contrast_constv_float", "ledge cos 대비, 등속 (부유) 자리")):
        L.append(f"- {lab}, 슬롯 0–7: " + " ".join(f"{x:.2f}" for x in B[k]))
    L.append("\n## 재현\n\n```\npython z_research/scripts/analysis/rollout2_distance_decay.py\n```")
    open(OUT / "distance_decay.md", "w").write("\n".join(L)); print("\n".join(L))


if __name__ == "__main__":
    main()
