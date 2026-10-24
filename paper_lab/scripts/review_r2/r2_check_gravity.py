"""R2 review (round 1) — independent future-choice recount on the gravity test split (108 groups).
Adds what EVIDENCE #23 lacks: block-space copy, release-space copy, and a true-trajectory oracle, plus which candidate each picks.
Read-only. No GPU."""
import csv, json, collections, glob
import numpy as np
R = "/data/hyuntak/project/2026/2027_cvpr/vjepa2/"
rows = list(csv.DictReader(open(R + "data_csv/vobench_v1_gravity/index.csv")))
test = {r["block_id"] for r in csv.DictReader(open(R + "data_csv/vobench_v1_gravity_split/index_test.csv"))}
blocks = collections.defaultdict(list)
for r in rows: blocks[r["block_id"]].append(r)

def from_perblock(path):
    S = json.load(open(path))["per_video_surprise"]
    return {k: (v if isinstance(v, (int, float)) else v.get("all")) for k, v in S.items()}

scores = {}
EX = R + "z_research/VOBench_v1/exp_results/surprise_c16t32__vobench_v1_gravity_"
scores["release (clip space)"] = from_perblock(EX + "vith/per_block.json")
scores["base block-AR ep70"] = from_perblock(EX + "vith_vast_ar_state_ep70/per_block.json")
scores["Ariel output-fb ep40"] = from_perblock(EX + "vith_ariel_ar_ep40/per_block.json")
# release-space copy (context-encoder last tubelet copied; clip-wise target)
cp = R + "cvpr/results/eval/testbed/copy/vobench_v1_gravity_vith/"
meta = json.load(open(cp + "meta.json")) if glob.glob(cp + "meta.json") else None
# block-space renders from the 10-13 headroom run (same block-wise LN(target_encoder) space as block-AR)
hs = collections.defaultdict(dict)
for f in sorted(glob.glob(R + "z_research/VOBench_v1/headroom/gravity_r*.npz")):
    z = np.load(f, allow_pickle=True)
    for m in ("copy", "ar", "move_or", "move_cv"):
        for v, s in zip(z["video_id"], z["S_" + m]): hs[m][str(v)] = float(s.mean())
scores["block-space copy"] = hs["copy"]
scores["block-space true-trajectory oracle (move_or)"] = hs["move_or"]
scores["block-space headroom 'ar' (Ariel ep40)"] = hs["ar"]

def pick(S, keep):
    out = collections.defaultdict(lambda: [0, 0, collections.Counter()])
    for b, rs in blocks.items():
        if keep is not None and b not in keep: continue
        if not all(r["video_id"] in S for r in rs): continue
        best = min(rs, key=lambda r: S[r["video_id"]])
        for k in dict.fromkeys(("ALL", rs[0]["law"], rs[0]["law"] + "/" + rs[0]["cell"][:6] if rs[0]["law"] == "arc" else rs[0]["law"])):
            o = out[k]; o[0] += best["plausible"] == "1"; o[1] += 1; o[2][best["variant"]] += 1
    return out

for name, S in scores.items():
    for split, keep in (("test108", test), ("all216", None)):
        o = pick(S, keep)
        print(f"{name:48s} {split:7s}", "  ".join(f"{k}: {v[0]}/{v[1]}" for k, v in sorted(o.items())))
        if split == "test108":
            for k in ("ledge", "arc/apex30", "arc/apex48", "arc/apex66"):
                if k in o: print(f"{'':58s}{k:11s} picks {dict(o[k][2])}")

# ---- release-space copy (te_v11_score copy.npy: mean|LN(z)[7] - h[8+j]|, clip-wise target) and release l1 from the same run
cpm = json.load(open(cp + "meta.json"))
c = np.load(cp + "copy.npy"); l1 = np.load(cp + "l1.npy")
scores2 = {"release-space copy": {v: float(c[i].mean()) for i, v in enumerate(cpm["video_id"])},
           "release (same run as copy)": {v: float(l1[i, 0].mean()) for i, v in enumerate(cpm["video_id"])}}
for name, S in scores2.items():
    o = pick(S, test)
    print(f"{name:48s} test108", "  ".join(f"{k}: {v[0]}/{v[1]}" for k, v in sorted(o.items())))
    for k in ("ledge", "arc/apex30", "arc/apex48", "arc/apex66"):
        print(f"{'':58s}{k:11s} picks {dict(o[k][2])}")
