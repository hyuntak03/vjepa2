#!/usr/bin/env python3
"""[물리] 실험 1 보충 (사전 등록 KST 04:07): 옮긴 기억을 칸별로 — VOBench 판정 (1) · IntPhys 2 기술 수치 (2). 학습 없음.
  CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=8 /venv/vjepa/bin/python -W ignore auto_research/scripts/mem_slot_e1cell.py
칸별 옮긴 기억: 기억 = 위치 p_m 둘레 3×3 칸 토큰 9 개, 비교 = 위치 p_t 둘레 3×3 칸 관측 토큰과 같은 상대 칸끼리 L1 (차원 평균) 의 9 칸 평균. 화면 밖 칸은 뺀다."""
import json, os, sys, random, collections
import numpy as np, torch
REPO = "/workspace/world_model/vjepa2"
sys.path.insert(0, os.path.join(REPO, "auto_research/scripts"))
from mem_slot import MemSlot, OUT  # noqa: E402
from crit_ip2_pmi import decomp     # noqa: E402
G, S = 16, 256
dev = torch.device("cuda"); torch.set_num_threads(8)


def load(name):
    if name == "rand":
        torch.manual_seed(0); m = MemSlot()
    else:
        ck = torch.load(os.path.join(OUT, f"{name}.pt"), map_location="cpu", weights_only=False); m = MemSlot(K=ck["args"]["K"]); m.load_state_dict(ck["model"])
    return m.to(dev).eval()


def cells(pos):
    c, r = int(np.clip(np.floor(float(pos[0])), 0, G - 1)), int(np.clip(np.floor(float(pos[1])), 0, G - 1))
    return r, c


def cell_dist(h_m, pm, h_t, pt):
    rm, cm = cells(pm); rt, ct = cells(pt); ds = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            a, b = (rm + dr, cm + dc), (rt + dr, ct + dc)
            if all(0 <= x < G for x in a + b):
                ds.append(float((h_m[a[0] * G + a[1]] - h_t[b[0] * G + b[1]]).abs().mean()))
    return float(np.mean(ds)) if ds else 0.0


@torch.no_grad()
def vob(models):
    C = "/workspace/auto_research_cache/vobench_late"; ARMS = ["pos_a", "pos_b", "imp_ab", "imp_ba"]
    meta = json.load(open(os.path.join(C, "meta.json"))); H = np.load(os.path.join(C, "h.npy"), mmap_mode="r")
    recs = []
    for m in meta:
        hh = torch.from_numpy(np.asarray(H[m["i"]], dtype=np.float32)).to(dev)
        sc = {}
        for nm, md in models.items():
            for ai, v in enumerate(ARMS):
                h = hh[ai]; pri, post, _ = md(h[None])
                e = torch.sigmoid(post["exist"][0]); k = int(e[:8].mean(0).argmax())
                vp = torch.sigmoid(post["vis"][0, :, k]).cpu().numpy(); vq = torch.sigmoid(pri["vis"][0, :, k]).cpu().numpy()
                pp = post["pos"][0, :, k].cpu().numpy(); pq = pri["pos"][0, :, k].cpu().numpy()
                tv = max([t for t in range(8) if vp[t] > 0.5], default=7)
                ts = [t for t in range(8, 16) if vq[t] > 0.5] or list(range(8, 16))
                sc[(nm, v)] = float(np.mean([cell_dist(h[tv], pp[tv], h[t], pq[t]) for t in ts]))
        mv = "static" if m["motion"] == "flat_static" else "moving"
        for pos, imp in (("pos_a", "imp_ab"), ("pos_b", "imp_ba")):
            recs.append(dict(block=m["i"], mv=mv, vio=m["violation"], ok={nm: float(sc[(nm, imp)] > sc[(nm, pos)]) for nm in models}))
    rng = random.Random(0); out = {}
    for grp in ("moving", "static"):
        rs = [r for r in recs if r["mv"] == grp]
        for nm in models:
            by = collections.defaultdict(list)
            for r in rs:
                by[r["block"]].append(r["ok"][nm])
            bl = list(by); est = 100 * np.mean([x for b in bl for x in by[b]])
            bs = sorted(100 * np.mean([x for b in [bl[rng.randrange(len(bl))] for _ in bl] for x in by[b]]) for _ in range(2000))
            out[f"{grp}|{nm}"] = dict(acc=round(float(est), 1), ci=(round(bs[50], 1), round(bs[1949], 1)), n=len(rs))
            for vio in sorted({r["vio"] for r in rs}):
                out[f"{grp}|{vio}|{nm}"] = round(100 * float(np.mean([r["ok"][nm] for r in rs if r["vio"] == vio])), 1)
    return out


@torch.no_grad()
def ip2(md, B0=4):
    C = "/workspace/auto_research_cache/intphys2/Main/"
    meta = json.load(open(C + "meta.json")); H = np.load(C + "h.npy", mmap_mode="r"); rows = []
    for m in meta:
        h = torch.from_numpy(np.asarray(H[m["row"]], dtype=np.float32)).to(dev)
        pri, post, _ = md(h[None]); T = h.shape[0]
        e = torch.sigmoid(post["exist"][0]).cpu().numpy(); vp = torch.sigmoid(post["vis"][0]).cpu().numpy(); vq = torch.sigmoid(pri["vis"][0]).cpu().numpy()
        pp = post["pos"][0].cpu().numpy(); pq = pri["pos"][0].cpu().numpy()
        vals = []
        for t in range(B0 + 1, T):
            num = den = 0.0
            for k in range(e.shape[1]):
                if vq[t, k] <= 0.5:
                    continue
                prev = [u for u in range(t - 1) if vp[u, k] > 0.5]
                if not prev:
                    continue
                u = prev[-1]
                num += e[t, k] * cell_dist(h[u], pp[u, k], h[t], pq[t, k]); den += e[t, k]
            if den > 0:
                vals.append(num / den)
        rows.append(dict(scene=m["scene_index"], pair=str(m["pair_id"]), imp=bool(m["is_impossible"]), cond=m["condition"], cam=m["camera"],
                         moved=float(np.mean(vals)) if vals else 0.0, n_blocks=len(vals)))
    rng = random.Random(0); res = {}
    for cond in ["all", "permanence", "immutability", "continuity", "solidity"]:
        for cam in ["all", "Fixed", "Moving"]:
            xs = [r for r in rows if (cond == "all" or r["cond"] == cond) and (cam == "all" or r["cam"] == cam)]
            res[f"{cond}|{cam}"] = decomp(xs, "moved", False, rng)
    return res, rows


def main():
    models = {nm: load(nm) for nm in ("A", "P", "rand")}
    v = vob(models)
    for k, x in v.items():
        print("VOB", k, x, flush=True)
    r, rows = ip2(models["A"])
    for k, st in r.items():
        print(f"IP2 {k:24s} acc {st['acc']:5.1f} [{st['acc_ci'][0]:5.1f},{st['acc_ci'][1]:5.1f}] 짝1/2 {st['pair1']:5.1f}/{st['pair2']:5.1f} o {st['outcome_share']:.2f} "
              f"i {st['mean_i']:+.5f} [{st['mean_i_ci'][0]:+.5f},{st['mean_i_ci'][1]:+.5f}]", flush=True)
    print("IP2 블록 없는 영상 몫", np.mean([x["n_blocks"] == 0 for x in rows]))
    json.dump(dict(vob=v, ip2=r, ip2_rows=rows), open(os.path.join(REPO, "auto_research/results/mem_slot_e1cell.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
