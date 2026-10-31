"""VOBench 끝 가림 (late) — 블록 AR 의 채점 구간을 미래 첫 블록 (2 샘플) 로 줄이면 어떻게 되나 (2026-10-28, CPU).
사용자: "ar 기본판에서 inference 를 2frame 예측으로 바꾸면 vobench late (k=1) 성능 올라갈려나?"
블록 AR 은 인과적이라 문맥 0–7 → 블록 8 예측은 8 걸음 rollout 의 첫 블록과 같다. 그래서 바뀌는 것은 채점 구간뿐이다.
  k = 가림 길이 (샘플). 문맥 끝 k 장 + 미래 앞 k 장 가림. k=1 이면 블록 8 = (가림, 보임) → 첫 미래 블록에 이미 다시 나타난다.
  k=2 이면 블록 8 전체 가림, k=3 이면 블록 8 전체 + 블록 9 절반 가림.
캐시: /workspace/auto_research_cache/vobench_late/h.npy (vobench_late_rollout_schemes.py 와 같다). 쌍 = (pos_a, imp_ab) · (pos_b, imp_ba), 문맥 블록 0–7 이 같다.
쌍마다 블록 8–15 각각의 surprise (토큰 L1 평균) 를 가능 · 불가능 둘 다 저장 → 구간 [8, 8+w) 의 평균으로 채점. 맞음 = 불가능 > 가능.
  for s in 0 1 2 3; do /venv/vjepa/bin/python -W ignore z_research/scripts/analysis/vobench_late_block_window.py --shard $s --nshard 4 & done; wait
  /venv/vjepa/bin/python -W ignore z_research/scripts/analysis/vobench_late_block_window.py --merge --nshard 4   → z_analysis/results/vobench_late_block_window.json
"""
import argparse, json, os, re, sys
import numpy as np, torch
ap = argparse.ArgumentParser(); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1)
ap.add_argument("--merge", action="store_true"); ap.add_argument("--threads", type=int, default=8); ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
REPO = "/workspace/world_model/vjepa2"; CA = "/workspace/auto_research_cache/vobench_late"; OUTD = "/workspace/auto_research_cache/vobench_late_block_window"
S, D, NB, C = 256, 1280, 16, 8
M = json.load(open(f"{CA}/meta.json"))

if not a.merge:
    torch.set_num_threads(a.threads)
    sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "auto_research/scripts"))
    _argv = list(sys.argv); sys.argv = [sys.argv[0], "--loss", "l1", "--name", "x"]
    import ar_crps_ft as FT
    sys.argv = _argv
    H = np.load(f"{CA}/h.npy", mmap_mode="r")
    ar = FT.load_pred("natural_ar_vw_state", "latest.pt").float().eval(); ix = torch.arange(C * S).unsqueeze(0)
    mine = [m for j, m in enumerate(M) if j % a.nshard == a.shard][: a.limit or None]
    rows = []
    for n, m in enumerate(mine):
        hs = [torch.from_numpy(H[m["i"], k].astype(np.float32)) for k in range(4)]
        for pi, ii in ((0, 2), (1, 3)):
            with torch.no_grad():
                p = ar.rollout_kv(hs[pi][:C].reshape(1, C * S, D), ix, NB - C)[0].view(NB - C, S, D)
            rows.append({"objset": m["objset"], "violation": m["violation"], "motion": "static" if m["motion"] == "flat_static" else "moving",
                         "k": int(re.search(r"late_k(\d)", m["files"]["pos_a"]).group(1)),
                         "s_pos": (p - hs[pi][C:]).abs().mean((1, 2)).tolist(), "s_imp": (p - hs[ii][C:]).abs().mean((1, 2)).tolist()})
        if (n + 1) % 10 == 0:
            print(f"  shard {a.shard}: {n + 1}/{len(mine)}", flush=True)
    os.makedirs(OUTD, exist_ok=True); json.dump(rows, open(f"{OUTD}/shard{a.shard}.json", "w")); print("shard saved", len(rows)); sys.exit(0)

rows = [r for k in range(a.nshard) for r in json.load(open(f"{OUTD}/shard{k}.json"))]
sp, si = np.array([r["s_pos"] for r in rows]), np.array([r["s_imp"] for r in rows])          # (쌍, 8 블록)
acc = lambda sel, w: round(100 * float(np.mean(si[sel, :w].mean(1) > sp[sel, :w].mean(1))), 1)
one = lambda sel, b: round(100 * float(np.mean(si[sel, b] > sp[sel, b])), 1)
groups = {"all": np.ones(len(rows), bool)}
for key in ("k", "violation", "motion"):
    for v in sorted({r[key] for r in rows}):
        groups[f"{key}={v}"] = np.array([r[key] == v for r in rows])
for kv in (1, 2, 3):
    for key in ("violation", "motion"):
        for v in sorted({r[key] for r in rows}):
            groups[f"k={kv}/{key}={v}"] = np.array([r["k"] == kv and r[key] == v for r in rows])
R = {"n_pairs": len(rows), "note": "window w = 미래 블록 8..8+w-1 평균 surprise 로 채점 (w=8 = 지금 채점). single = 그 블록 하나만", "groups": {}}
for g, sel in groups.items():
    R["groups"][g] = {"n": int(sel.sum()), "window": {w: acc(sel, w) for w in (1, 2, 4, 8)}, "single": {8 + b: one(sel, b) for b in range(8)},
                      "margin_rel_mean": [round(float(x), 4) for x in ((si[sel] - sp[sel]) / sp[sel]).mean(0)]}
json.dump(R, open(f"{REPO}/z_analysis/results/vobench_late_block_window.json", "w"), indent=1)
for g, e in R["groups"].items():
    print(f"{g:<32s} n={e['n']:>3d}  w1 {e['window'][1]:5.1f}  w2 {e['window'][2]:5.1f}  w4 {e['window'][4]:5.1f}  w8 {e['window'][8]:5.1f}   single " +
          " ".join(f"{x:5.1f}" for x in e["single"].values()))
print("saved")
