#!/usr/bin/env python3
"""[P] IntPhys 2 칸 2×2 묶음 채점 확인 — Debug 의 Main 겹침 제외 장면 ([비판] 확인 세트 ①), 기준 AR, s = 1 vs 2 (학습 없음) (2026-10-22 KST).
사전 등록: 진행판 [P] 11:18 UTC (KST 20:18). ⚠️ 판별 진단 (관문 4).
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=5 /venv/vjepa/bin/python -W ignore auto_research/scripts/pool_space_confirm.py"""
import collections, csv, json, os, random, sys
import numpy as np, torch, torch.nn.functional as F
REPO = "/workspace/world_model/vjepa2"
sys.path.insert(0, os.path.join(REPO, "auto_research/scripts"))
from crit_ip2_pmi import decomp  # noqa: E402
dev = torch.device("cuda"); S, G = 256, 16
C2 = "/workspace/auto_research_cache/intphys2/Debug/"
meta = json.load(open(C2 + "meta.json")); H = np.load(C2 + "h.npy", mmap_mode="r"); P = np.load(C2 + "p_tf.npy", mmap_mode="r")
mainset = {r["name"] for r in csv.DictReader(open("/workspace/dataset/benchmarks/IntPhys2/Main/metadata.csv"))}
bad = {m["scene_index"] for m in meta if m["name"] in mainset}
meta = [m for m in meta if m["scene_index"] not in bad]
print("Main 과 겹치는 장면", sorted(bad), "→ 영상", len(meta), "· 장면", len({m["scene_index"] for m in meta}), flush=True)


def pool(x, s):
    if s == 1:
        return x
    y = x.reshape(-1, G, G, x.shape[-1]).permute(0, 3, 1, 2)
    return F.avg_pool2d(y, s).permute(0, 2, 3, 1).reshape(x.shape[0], -1, x.shape[-1])


rows = []
for m in meta:
    h = torch.from_numpy(np.asarray(H[m["row"]], np.float32)).to(dev); p = torch.from_numpy(np.asarray(P[m["row"]], np.float32)).to(dev)
    r = dict(scene=m["scene_index"], pair=str(m["pair_id"]), imp=bool(m["is_impossible"]), cond=m["condition"], cam=m["camera"])
    for s in (1, 2):
        r[f"s{s}"] = float((pool(p[4:31], s) - pool(h[5:32], s)).abs().mean())
    rows.append(r)
res = {"n_videos": len(rows), "excluded_scenes": sorted(bad)}
for s in (1, 2):
    res[f"s{s}"] = decomp(rows, f"s{s}", False, random.Random(0))
sc = collections.defaultdict(dict)
for r in rows:
    sc[r["scene"]][r["pair"] + ("I" if r["imp"] else "P")] = r
scenes = [v for v in sc.values() if len(v) == 4]
dv = np.array([np.mean([float(v[q + "I"]["s2"] > v[q + "P"]["s2"]) - float(v[q + "I"]["s1"] > v[q + "P"]["s1"]) for q in ("1", "2")]) for v in scenes]) * 100
rng = np.random.default_rng(0); bs = [dv[rng.integers(0, len(dv), len(dv))].mean() for _ in range(4000)]
res["paired_d2"] = [round(float(dv.mean()), 1), round(float(np.percentile(bs, 2.5)), 1), round(float(np.percentile(bs, 97.5)), 1)]
res["n_scenes"] = len(scenes)
for s in (1, 2):
    e = res[f"s{s}"]; print(f"s{s}: 정확도 {e['acc']:.1f} [{e['acc_ci'][0]:.1f}, {e['acc_ci'][1]:.1f}] 짝 {e['pair1']:.1f}/{e['pair2']:.1f} 결과 항 몫 {e['outcome_share']:.2f} 문맥 항 {e['mean_i']:+.5f} [{e['mean_i_ci'][0]:+.5f}, {e['mean_i_ci'][1]:+.5f}] n {e['n']}", flush=True)
print("짝 대응 차 (s2 − s1)", res["paired_d2"], flush=True)
json.dump(res, open(os.path.join(REPO, "auto_research/results/pool_space_confirm.json"), "w"), indent=1)
