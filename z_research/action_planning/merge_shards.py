"""plan_pusht.py --shard i/N 결과를 합친다.  python z_research/action_planning/merge_shards.py <exp_results 의 tag 접두어 (…_shard 앞까지)>"""
import glob, json, sys, numpy as np
pre = sys.argv[1].rstrip("_")
import re
files = sorted(f for f in glob.glob(f"{pre}_shard*of*/summary.json")
               if re.fullmatch(r"_shard\d+of\d+", f[len(pre):].split("/")[0]))   # 2026-10-12: 보관 폴더 (…_shard0of4_stuck…) 는 뺀다
if not files:
    sys.exit(f"없다: {pre}_shard*of*/summary.json")
rows = []
for f in files:
    d = json.load(open(f))
    rows += list(zip(d["eval_index"], d["successes"], d["state_dist"]))
rows.sort()
idx = [r[0] for r in rows]
n_total = json.load(open(files[0]))["args"].get("n_evals_total", len(idx))
assert idx == list(range(n_total)), f"조각이 빠졌다: {len(idx)}/{n_total}"
sr = float(np.mean([r[1] for r in rows]))
out = {"prefix": pre, "n_shards": len(files), "n_evals": n_total, "success_rate": sr,
       "successes": [r[1] for r in rows], "state_dist": [r[2] for r in rows]}
json.dump(out, open(f"{pre}_merged.json", "w"), indent=1)
print(f"{pre}: success_rate {sr:.3f} ({sum(r[1] for r in rows)}/{n_total}, {len(files)} shards) -> {pre}_merged.json")
