#!/usr/bin/env python3
"""[T] B′ s 입력 설계 검증 (학습 없음, CPU): IntPhys 2 의 짝을 가르는 문맥 단서가 고정 encoder 블록 토큰에 있나 (auto_research, 2026-10-22 KST).

  OMP_NUM_THREADS=8 /venv/vjepa/bin/python auto_research/scripts/ip2_ctx_cue.py [--nb 6]

[E] 공유 캐시 (intphys2/Main: 블록 토큰 32 개 · meta) 의 장면마다 네 영상 (짝 1 · 2 × 가능 · 불가능).
  짝 사이 차이  Δ_pair(c)   = 블록 0..nb−1 평균 |h(1_가능) − h(2_가능)| (차원 평균)       — 장애물 · 가림막 유무 등 짝을 가르는 문맥
  짝 안 차이    Δ_within(c) = 블록 0..nb−1 평균 |h(k_가능) − h(k_불가능)|, k = 1, 2 평균 — 같은 문맥이면 ≈ 0 (렌더 잡음 바닥)
지표: 비 = mean Δ_pair / mean Δ_within (≫ 1 이면 짝 단서가 토큰에 있다) · 집중도 = Δ_pair 의 상위 10 % 칸 몫 (국소 구조면 ≫ 0.1).
블록별 Δ_within 중앙값도 낸다 (가능 · 불가능이 언제 갈라지나 → 문맥 구간 선택의 근거).
조건 4 × 카메라 로 나눠 낸다."""
import argparse, collections, json, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="/workspace/auto_research_cache/intphys2/Main")
ap.add_argument("--nb", type=int, default=6)
ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "../results/ip2_ctx_cue.json"))
a = ap.parse_args()
meta = json.load(open(os.path.join(a.cache, "meta.json")))
H = np.load(os.path.join(a.cache, "h.npy"), mmap_mode="r")
done = np.zeros(len(meta), bool)
for f in ("done.npy", "done_s0.npy", "done_s1.npy"):
    p = os.path.join(a.cache, f)
    if os.path.exists(p):
        done |= np.load(p).astype(bool)
sc = collections.defaultdict(dict)
for r in meta:
    sc[r["scene_index"]][r["type"]] = r
rows, per_block = [], []
for s, v in sc.items():
    if len(v) != 4 or not all(done[v[t]["row"]] for t in v):
        continue
    g = {t: np.asarray(H[v[t]["row"], :], np.float32) for t in v}          # (32, 256, 1280)
    dpair = np.abs(g["1_Possible"][:a.nb] - g["2_Possible"][:a.nb]).mean((0, 2))
    dw = 0.5 * (np.abs(g["1_Possible"][:a.nb] - g["1_Impossible"][:a.nb]).mean((0, 2)) + np.abs(g["2_Possible"][:a.nb] - g["2_Impossible"][:a.nb]).mean((0, 2)))
    wb = 0.5 * (np.abs(g["1_Possible"] - g["1_Impossible"]).mean((1, 2)) + np.abs(g["2_Possible"] - g["2_Impossible"]).mean((1, 2)))   # (32,)
    pb = np.abs(g["1_Possible"] - g["2_Possible"]).mean((1, 2))
    srt = np.sort(dpair)[::-1]
    rows.append(dict(scene=s, condition=v["1_Possible"]["condition"], camera=v["1_Possible"]["camera"], d_pair=float(dpair.mean()), d_within=float(dw.mean()),
                     top10=float(srt[:26].sum() / srt.sum())))
    per_block.append((wb, pb))
print(f"장면 {len(rows)} (네 영상 모두 캐시됨), 문맥 = 블록 0..{a.nb - 1}")
res = {"n": len(rows), "nb": a.nb}


def summ(rs):
    dp = np.array([r["d_pair"] for r in rs]); dw = np.array([r["d_within"] for r in rs]); t10 = np.array([r["top10"] for r in rs])
    return dict(n=len(rs), d_pair=float(dp.mean()), d_within=float(dw.mean()), ratio=float(dp.mean() / max(dw.mean(), 1e-9)),
                ratio_median=float(np.median(dp / np.maximum(dw, 1e-9))), frac_scene_ratio_gt2=float((dp > 2 * dw).mean()), top10=float(t10.mean()))


res["all"] = summ(rows)
for key in ("condition", "camera"):
    for val in sorted({r[key] for r in rows}):
        res[f"{key}={val}"] = summ([r for r in rows if r[key] == val])
for k, v in res.items():
    if isinstance(v, dict):
        print(f"  {k:24s} n {v['n']:3d} · 짝 사이 {v['d_pair']:.4f} · 짝 안 {v['d_within']:.4f} · 비 {v['ratio']:.2f} (장면 중앙값 {v['ratio_median']:.2f}, 비 > 2 인 장면 {v['frac_scene_ratio_gt2']:.2f}) · 상위 10 % 칸 몫 {v['top10']:.2f}")
wb = np.median(np.stack([p[0] for p in per_block]), 0); pb = np.median(np.stack([p[1] for p in per_block]), 0)
res["block_median_within"] = wb.tolist(); res["block_median_pair"] = pb.tolist()
print("  블록별 중앙값 짝 안: " + " ".join(f"{x:.3f}" for x in wb))
print("  블록별 중앙값 짝 사이: " + " ".join(f"{x:.3f}" for x in pb))
json.dump({"summary": res, "rows": rows}, open(a.out, "w"), indent=1)
