"""R2 review (round 2): independent recount of the pilot primary measure from the harness error maps.

Does NOT import paper_lab/eval/passline.py. Reads the per-video token-error maps E_* (8 x 16 x 16) saved by the harness,
builds the object-region mask itself from VOBench label.json (union of both arms' object 3x3 over the 8 future blocks),
and recomputes for late-occlusion moving pairs (360) and late static pairs (72):
  - pair accuracy (object-region) for model / copy / mean8 / oracle
  - paired block bootstrap of arm differences (180 source blocks resampled, B = 10000)
  - discordant pairs, direction split (A = context arm with object, B = other arm)
  - continuous margin m = E(imp) - E(pos) over the mask, and the model margin as a share of the oracle margin
  - sensitivity: accuracy of a margin-interpolated predictor m(a) = (1-a) m_A0 + a m_oracle  (approximation; L1 is not linear
    in a prediction blend, so this is a calibration of the decision rule, not of a real model)
Read-only, CPU. Run: /data/hyuntak/anaconda3/envs/vjepa2/bin/python paper_lab/scripts/review_r2/r2_round2_pilot.py
"""
import csv, json, os
from collections import defaultdict
import numpy as np

R = "/data/hyuntak/project/2026/2027_cvpr/vjepa2/"
RES = R + "paper_lab/eval/results/"
FR = "/data2/local_datasets/world/Benchmark/VOBench_v1/"
rows = {r["video_id"]: r for r in csv.DictReader(open(R + "data_csv/vobench_v1/index.csv"))}
blk = defaultdict(dict)
for v, r in rows.items():
    blk[r["source_block"]][r["variant"]] = v

def track(lab_arm):
    if not lab_arm or lab_arm[0] is None:
        return None
    P = np.array([[lab_arm[3 * s]["u"], lab_arm[3 * s]["v"]] for s in range(32)], float)
    return P[16:].reshape(8, 2, 2).mean(1)          # 8 future blocks, 512-px coords

def mask_of(lab, arms=("A", "B")):
    M = np.zeros((8, 16, 16), bool)
    for a in arms:
        p = track(lab.get(a))
        if p is None:
            continue
        for k in range(8):
            u, v = p[k]
            if not (0 <= u < 512 and 0 <= v < 512):
                continue
            y, x = int(v // 32), int(u // 32)
            M[k, max(y - 1, 0):y + 2, max(x - 1, 0):x + 2] = True
    return M

_lab = {}
def pairs(occ, moving):
    out = []
    for sb, arms in blk.items():
        if not all(a in arms for a in ("pos_a", "pos_b", "imp_ab", "imp_ba")):
            continue
        r = rows[arms["pos_a"]]
        if r["occlusion"] != occ or ((r["motion"] != "flat_static") != moving):
            continue
        if sb not in _lab:
            _lab[sb] = json.load(open(FR + os.path.dirname(r["file_name"]) + "/label.json"))["labels"]
        M = mask_of(_lab[sb])
        if not M.any():
            continue
        for pos, imp, d in (("pos_a", "imp_ab", "A"), ("pos_b", "imp_ba", "B")):
            out.append((sb, d, r["violation"], arms[pos], arms[imp], M))
    return out

_Z = {}
def Z(name):
    if name not in _Z:
        z = np.load(RES + name + "/vobench_r0.npz", allow_pickle=True)
        _Z[name] = ({v: i for i, v in enumerate(z["video_id"])}, {k: z[k] for k in z.files if k.startswith("E_")})
    return _Z[name]

def margins(name, P, render="model"):
    idx, E = Z(name)
    m = []
    for sb, d, viol, vp, vi, M in P:
        ep = E["E_" + render][idx[vp]].astype(np.float32)[M].mean()
        ei = E["E_" + render][idx[vi]].astype(np.float32)[M].mean()
        m.append(ei - ep)
    return np.array(m)

def hits(m):
    return np.where(m > 0, 1.0, np.where(m == 0, 0.5, 0.0))

def boot(diff, P, B=10000, seed=0):
    sbs = sorted({p[0] for p in P}); pos = defaultdict(list)
    for i, p in enumerate(P):
        pos[p[0]].append(i)
    groups = [np.array(pos[s]) for s in sbs]
    rng = np.random.default_rng(seed); out = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, len(groups), len(groups))
        out[b] = np.concatenate([diff[groups[j]] for j in pick]).mean()
    return 100 * diff.mean(), 100 * np.percentile(out, 2.5), 100 * np.percentile(out, 97.5)

ARMS = ["pilotA0_e10", "pilotAX_e10", "pilotA1_e10", "pilotA1c_e10", "pilotA2c_e10", "pilotA4_e10", "pilotA0_e5", "pilotAX_e5", "vast70", "ext_ar_vw_shortctx_e30_c8", "ext_ar_vw_shortctx_e30_c2"]
ARMS = [a for a in ARMS if os.path.exists(RES + a + "/vobench_r0.npz")]

for occ, moving, tag in (("late", True, "late/moving (primary)"), ("late", False, "late/static (B1)")):
    P = pairs(occ, moving)
    print(f"\n=== {tag}: n_pairs {len(P)}, n_blocks {len({p[0] for p in P})}")
    base = {r: hits(margins("pilotA0_e10", P, r)) for r in ("copy", "mean8", "oracle")}
    print("  baselines (A0 run): " + "  ".join(f"{r} {100*h.mean():.1f}" for r, h in base.items()))
    mo = margins("pilotA0_e10", P, "oracle")
    H = {}
    for a in ARMS:
        m = margins(a, P); h = hits(m); H[a] = (m, h)
        dA = np.array([p[1] for p in P]); vio = np.array([p[2] for p in P])
        dirs = " ".join(f"{d}:{100*h[dA==d].mean():.0f}" for d in ("A", "B"))
        perm = " ".join(f"perm-{d}:{100*h[(dA==d)&(vio=='permanence')].mean():.0f}" for d in ("A", "B"))
        share = np.mean(m) / np.mean(mo)
        print(f"  {a:28s} acc {100*h.mean():5.1f}  dir {dirs}  {perm}  mean margin {np.mean(m):+.4f} (oracle {np.mean(mo):+.4f}, share {share:+.3f})")
    if moving:
        print("  --- paired differences (block bootstrap, B=10000): diff [95% CI], discordant pairs, margin diff [CI]")
        for b, a in (("pilotAX_e10", "pilotA0_e10"), ("pilotA1_e10", "pilotA0_e10"), ("pilotA1c_e10", "pilotA0_e10"),
                     ("pilotA2c_e10", "pilotA0_e10"), ("pilotA4_e10", "pilotA0_e10"), ("pilotA2c_e10", "pilotAX_e10"),
                     ("pilotA4_e10", "pilotAX_e10"), ("pilotA0_e10", "vast70")):
            if b not in H or a not in H:
                continue
            d = H[b][1] - H[a][1]; disc = int((d != 0).sum())
            md = (H[b][0] - H[a][0]) / np.abs(mo).mean()          # margin change in units of mean |oracle margin|
            print(f"  {b} - {a}: {boot(d, P)[0]:+.1f} [{boot(d, P)[1]:+.1f}, {boot(d, P)[2]:+.1f}]  discordant {disc}/{len(P)}  "
                  f"margin change {100*md.mean():+.2f}% of |oracle margin| [{boot(md, P)[1]:+.2f}, {boot(md, P)[2]:+.2f}]")
        if "pilotAX_e5" in H and "pilotA0_e5" in H:
            dd = (H["pilotAX_e10"][1] - H["pilotA0_e10"][1]) - (H["pilotAX_e5"][1] - H["pilotA0_e5"][1])
            print(f"  diff-in-diff (AX-A0) e10 - e5: {boot(dd, P)[0]:+.1f} [{boot(dd, P)[1]:+.1f}, {boot(dd, P)[2]:+.1f}]")
        print("  --- decision-rule sensitivity: accuracy of m(a) = (1-a) m_A0 + a m_oracle  (approximation)")
        m0 = H["pilotA0_e10"][0]
        line = []
        for al in (0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0):
            line.append(f"a={al}: {100*hits((1-al)*m0 + al*mo).mean():.1f}")
        print("  " + "  ".join(line))
        # minimal detectable effect for a 360-pair paired comparison if discordance were d (two-sided 95%): ~1.96*sqrt(d)/n
        for dfrac in (0.05, 0.10, 0.20, 0.40):
            print(f"  if {int(dfrac*100)}% of pairs were discordant, paired 95% half-width ~ {196*np.sqrt(dfrac/len(P)):.1f} pt")

# ---- continuous trend: (AX - A0) margin change e5 -> e10, in % of mean |oracle margin|
P = pairs("late", True)
mo = margins("pilotA0_e10", P, "oracle"); s = np.abs(mo).mean()
g = {a: margins(a, P) for a in ("pilotA0_e5", "pilotAX_e5", "pilotA0_e10", "pilotAX_e10")}
d5 = (g["pilotAX_e5"] - g["pilotA0_e5"]) / s; d10 = (g["pilotAX_e10"] - g["pilotA0_e10"]) / s
print(f"\n(AX-A0) margin, % of |oracle margin|: e5 {100*d5.mean():+.2f} [{boot(d5, P)[1]:+.2f}, {boot(d5, P)[2]:+.2f}]  "
      f"e10 {100*d10.mean():+.2f} [{boot(d10, P)[1]:+.2f}, {boot(d10, P)[2]:+.2f}]  e10-e5 {100*(d10-d5).mean():+.2f} [{boot(d10-d5, P)[1]:+.2f}, {boot(d10-d5, P)[2]:+.2f}]")
