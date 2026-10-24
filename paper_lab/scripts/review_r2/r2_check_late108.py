"""R2 review (round 1): VOBench late occlusion on split v2 test (108 pairs per object set) for several runs,
plus pair counts and a binomial standard error, to check EVIDENCE #18 #43 #51 and the 2-point reading rule. Read-only."""
import csv, json, math, os
from collections import defaultdict
R = "/data/hyuntak/project/2026/2027_cvpr/vjepa2/"
meta = {r["block_id"]: r for r in csv.DictReader(open(R + "data_csv/vobench_v1_split_v2/index_test.csv"))}
def late(run):
    f = R + run + "/per_block.json"
    if not os.path.exists(f): return None
    acc = defaultdict(lambda: [0.0, 0])
    pb = json.load(open(f))["per_block"]
    if isinstance(pb, dict): pb = next(iter(pb.values()))
    for b in pb:
        r = meta.get(str(b["block_id"]))
        if r is not None and r["occlusion"] == "late":
            acc[r["objset"]][0] += b["acc"] * b["n_pairs"]; acc[r["objset"]][1] += b["n_pairs"]
    return {k: (round(100 * c / n, 1), n) for k, (c, n) in acc.items()}
runs = {
 "base block-AR, cvpr late run": "cvpr/results/eval/testbed/vobench_v1_late_vith_vast_ar_state_ep70",
 "base block-AR, VOBench_v1 full run": "z_research/VOBench_v1/exp_results/surprise_c16t32__vobench_v1_vith_vast_ar_state_ep70",
 "release, cvpr run": "cvpr/results/eval/testbed/vobench_v1_vith",
 "release, VOBench_v1 run": "z_research/VOBench_v1/exp_results/surprise_c16t32__vobench_v1_vith",
 "phys2d single no-label PFT (on base AR)": "cvpr/results/eval/testbed/vobench_v1_late_vith_pft_single_nosup",
 "phys2d single label 0.05 PFT (on base AR)": "cvpr/results/eval/testbed/vobench_v1_late_vith_pft_single_w005",
}
for name, run in runs.items():
    print(f"{name:45s}", late(run), run)
p = 0.76; n = 108
print(f"binomial SE at p={p}, n={n}: {100*math.sqrt(p*(1-p)/n):.1f} pt -> 95% half-width {196*math.sqrt(p*(1-p)/n):.1f} pt; "
      f"SE of a difference of two independent 108-pair cells: {100*math.sqrt(2*p*(1-p)/n):.1f} pt")
