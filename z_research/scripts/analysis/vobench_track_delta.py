"""VOBench 토큰 Δ (e_imp − e_pos) 를 미래 걸음별로 — 물체를 정답 자리에 따라가나, 외형이 흐려지나, 마지막 관측 자리에 남나 (2026-10-12).
  입력 = 채점 실행의 token_surprise.npz (release · AR ep40). 물체 자리 = label.json 중심 ±(반경+1) 토큰 (16×16 격자).
  출력 = z_research/VOBench_v1/ariel_ar_ep40/surprise/track_delta.json + 표 출력. 결과 정리 = 같은 폴더 TRACKING_2026-10-12.md
  python z_research/scripts/analysis/vobench_track_delta.py
"""
import json, csv, numpy as np
from collections import defaultdict
from pathlib import Path
REPO = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2"); R = REPO / "z_research/VOBench_v1/exp_results"
FR = Path("/data2/local_datasets/world/Benchmark/VOBench_v1")
RUNS = {"release": "surprise_c16t32__vobench_v1_vith", "ar": "surprise_c16t32__vobench_v1_vith_ariel_ar_ep40"}
E = {m: dict(zip(*(lambda z: (z["video_id"], z["e"]))(np.load(R / d / "token_surprise.npz")))) for m, d in RUNS.items()}
rows = list(csv.DictReader(open(REPO / "data_csv/vobench_v1/index.csv"))); blocks = defaultdict(dict)
for r in rows: blocks[r["block_id"]][r["variant"]] = r
gy, gx = np.mgrid[0:16, 0:16] + 0.5
lab = {}
def track(row, arm):
    bdir = FR / Path(row["file_name"]).parent
    if bdir not in lab: lab[bdir] = json.load(open(bdir / "label.json"))["labels"]
    L = lab[bdir][arm]
    return [(L[3 * s]["u"] / 32, L[3 * s]["v"] / 32, L[3 * s]["r_px"] / 32, L[3 * s]["hidden"]) for s in range(32)]
def region(cu, cv, rad): return np.hypot(gx - cu, gy - cv) <= rad + 1.0
out = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
for b, v in blocks.items():
    r0 = v["pos_a"]; occ = r0["occlusion"]
    if occ not in ("visible", "late"): continue
    mv = "static" if r0["motion"] == "flat_static" else "moving"
    for p, q, arm in (("pos_a", "imp_ab", "A"), ("pos_b", "imp_ba", "B")):
        if r0["violation"] == "permanence" and arm == "B": continue          # B 는 빈 문맥 (물체 없음) — A 만
        T = track(v[p], arm)
        if T[0] is None: continue
        vis_ctx = [s for s in range(16) if not T[s][3]]
        last = T[vis_ctx[-1]] if vis_ctx else T[15]
        for m in E:
            ep, ei = E[m][v[p]["video_id"]], E[m][v[q]["video_id"]]
            for j in range(8):
                s0, s1 = 16 + 2 * j, 17 + 2 * j
                cu, cv, rad = np.mean([T[s0][0], T[s1][0]]), np.mean([T[s0][1], T[s1][1]]), np.mean([T[s0][2], T[s1][2]])
                ob = region(cu, cv, rad); ls = region(last[0], last[1], last[2]) & ~ob; bg = ~(ob | region(last[0], last[1], last[2]))
                d = ei[j] - ep[j]; dp = np.clip(d, 0, None)
                key = (r0["violation"], occ, mv)
                o = out[key][m]
                o["acc_all"].append(d.mean() > 0); o["acc_obj"].append(d[ob].mean() > 0)
                o["d_obj"].append(d[ob].mean()); o["d_bg"].append(d[bg].mean())
                o["share"].append(dp[ob].sum() / (dp.sum() + 1e-12)); o["frac"].append(ob.mean())
                o["ghost"].append(ep[j][ls].mean() - ep[j][bg].mean() if ls.any() else np.nan)
                o["j"].append(j)
res = {}
for key in sorted(out):
    for m in ("release", "ar"):
        o = {k: np.array(v, float) for k, v in out[key][m].items()}; J = o["j"]
        row = {k: [float(np.nanmean(o[k][J == j])) for j in range(8)] for k in ("acc_all", "acc_obj", "d_obj", "d_bg", "share", "frac", "ghost")}
        row["n_pairs"] = int((J == 0).sum()); res[f"{'/'.join(key)}/{m}"] = row
json.dump(res, open(REPO / "z_research/VOBench_v1/ariel_ar_ep40/surprise/track_delta.json", "w"), indent=1)
for k, r in res.items():
    if "permanence" in k and "static" in k: continue
    print(f"\n{k}  (쌍 {r['n_pairs']}, 물체 자리 토큰 비율 {100*np.mean(r['frac']):.1f}%)")
    print("  걸음                 " + " ".join(f"{'t'+str(8+j):>6s}" for j in range(8)))
    print("  정확도 (전체 토큰)   " + " ".join(f"{100*x:6.1f}" for x in r["acc_all"]))
    print("  정확도 (물체 자리만) " + " ".join(f"{100*x:6.1f}" for x in r["acc_obj"]))
    print("  Δ 물체 자리 (×1e3)  " + " ".join(f"{1e3*x:6.1f}" for x in r["d_obj"]))
    print("  Δ 배경 (×1e3)       " + " ".join(f"{1e3*x:6.1f}" for x in r["d_bg"]))
    print("  양의 Δ 중 물체 자리% " + " ".join(f"{100*x:6.1f}" for x in r["share"]))
    print("  유령: 마지막 관측 자리 e_pos − 배경 (×1e3) " + " ".join(f"{1e3*x:6.1f}" for x in r["ghost"]))

# 대조: 같은 장면의 빈 문맥 영상 (permanence pos_b = 없는데→없는) 에서, A 의 마지막 관측 자리 e − 배경.
# 가림막 · 장면 위치 때문에 생기는 초과 오차라면 빈 영상에서도 같다. 물체 잔상이라면 빈 영상에서는 0 근처.
print("\n대조 — late 이동 permanence: A 마지막 관측 자리 초과 오차 (×1e3), 물체 문맥 pos_a vs 빈 문맥 pos_b")
for m in E:
    g_obj, g_emp = [[] for _ in range(8)], [[] for _ in range(8)]
    for b, v in blocks.items():
        r0 = v["pos_a"]
        if r0["occlusion"] != "late" or r0["violation"] != "permanence" or r0["motion"] == "flat_static": continue
        T = track(v["pos_a"], "A"); vis_ctx = [s for s in range(16) if not T[s][3]]; last = T[vis_ctx[-1]]
        for j in range(8):
            s0, s1 = 16 + 2 * j, 17 + 2 * j
            ob = region(np.mean([T[s0][0], T[s1][0]]), np.mean([T[s0][1], T[s1][1]]), np.mean([T[s0][2], T[s1][2]]))
            ls = region(last[0], last[1], last[2]) & ~ob; bg = ~(ob | region(last[0], last[1], last[2]))
            if not ls.any(): continue
            for vid, acc in ((v["pos_a"]["video_id"], g_obj), (v["pos_b"]["video_id"], g_emp)):
                e = E[m][vid][j]; acc[j].append(e[ls].mean() - e[bg].mean())
    print(f"  {m:7s} 물체 문맥 " + " ".join(f"{1e3*np.mean(x):6.1f}" for x in g_obj))
    print(f"  {m:7s} 빈 문맥   " + " ".join(f"{1e3*np.mean(x):6.1f}" for x in g_emp))
