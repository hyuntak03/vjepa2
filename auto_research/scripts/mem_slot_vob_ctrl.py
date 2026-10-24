#!/usr/bin/env python3
"""[물리] 실험 1 VOBench 보충 (학습 없음): 같은 칸 "본 적 있나" 의 대조 주소 (균등 · 무작위 쿼리 head · 3D head) 를 같은 캐시 · 같은 블록 (미래 8–15) 에서,
그리고 mem_slot_vob.json 의 기억 슬롯 판 (A · P 의 e1 · main) 과 block bootstrap (2,000 회) 짝 대응 차. 위반 종류별 (영속 · 모양 · 색).
  CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=8 /venv/vjepa/bin/python -W ignore auto_research/scripts/mem_slot_vob_ctrl.py"""
import json, os, sys, random, collections
import numpy as np, torch
REPO = "/workspace/world_model/vjepa2"
sys.path.insert(0, os.path.join(REPO, "auto_research/scripts")); sys.path.insert(0, REPO)
from crit_ip2_addr_ctrl import load_head  # noqa: E402
C = "/workspace/auto_research_cache/vobench_late"; S = 256
ARMS = ["pos_a", "pos_b", "imp_ab", "imp_ba"]
dev = torch.device("cuda"); torch.set_num_threads(8)
torch.manual_seed(0)
heads = {"head3d": load_head("natural_ar_vw_state_phys3d_single", dev)}; _ = load_head("natural_ar_vw_state_phys2d_single", dev); heads["headrand"] = load_head(None, dev)
meta = json.load(open(os.path.join(C, "meta.json"))); H = np.load(os.path.join(C, "h.npy"), mmap_mode="r")
R = json.load(open(os.path.join(REPO, "auto_research/results/mem_slot_vob.json")))["rows"]
recs = []
for m, rr in zip(meta, R):
    hh = torch.from_numpy(np.asarray(H[m["i"]], dtype=np.float32)).to(dev)
    sc = {}
    with torch.no_grad():
        for ai, v in enumerate(ARMS):
            h = hh[ai]; T = h.shape[0]; ts = list(range(8, T))
            seen = torch.stack([(h[t][None] - h[: t - 1]).abs().mean(-1).min(0).values for t in ts])
            d = {"uniform": float(seen.mean())}
            for nm, hd in heads.items():
                w = hd(h.reshape(1, -1, 1280), S)["attn"][0, :, 0].float()[ts]
                d[nm] = float((w * seen).sum(-1).mean())
            for nm in ("A", "P"):
                d[f"{nm}_e1"] = rr["scores"][nm][v]["e1"]; d[f"{nm}_main"] = rr["scores"][nm][v]["main"]
            sc[v] = d
    mv = "static" if m["motion"] == "flat_static" else "moving"
    for pos, imp in (("pos_a", "imp_ab"), ("pos_b", "imp_ba")):
        recs.append(dict(block=m["i"], mv=mv, vio=m["violation"], ok={k: float(sc[imp][k] > sc[pos][k]) for k in sc[pos]}))
keys = list(recs[0]["ok"])
rng = random.Random(0)
def boot(rs, k, k2=None, n=2000):
    by = collections.defaultdict(list)
    for r in rs:
        by[r["block"]].append(r["ok"][k] - (r["ok"][k2] if k2 else 0.0))
    bl = list(by)
    vals = sorted(100 * np.mean([x for b in [bl[rng.randrange(len(bl))] for _ in bl] for x in by[b]]) for _ in range(n))
    return round(100 * np.mean([x for b in bl for x in by[b]]), 1), (round(vals[int(0.025 * n)], 1), round(vals[int(0.975 * n) - 1], 1))
out = {}
for grp in ("moving", "static"):
    rs = [r for r in recs if r["mv"] == grp]
    out[grp] = {k: boot(rs, k) for k in keys}
    out[grp + "|A_e1-uniform"] = boot(rs, "A_e1", "uniform"); out[grp + "|A_e1-headrand"] = boot(rs, "A_e1", "headrand")
    out[grp + "|A_e1-P_e1"] = boot(rs, "A_e1", "P_e1"); out[grp + "|A_e1-head3d"] = boot(rs, "A_e1", "head3d")
    for vio in sorted({r["vio"] for r in rs}):
        rv = [r for r in rs if r["vio"] == vio]
        out[f"{grp}|{vio}"] = {k: round(100 * np.mean([r["ok"][k] for r in rv]), 1) for k in keys}
for k, v in out.items():
    print(k, v)
json.dump(out, open(os.path.join(REPO, "auto_research/results/mem_slot_vob_ctrl.json"), "w"), indent=1)
