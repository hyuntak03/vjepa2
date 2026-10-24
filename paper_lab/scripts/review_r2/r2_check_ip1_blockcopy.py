"""R2 review (round 1): IntPhys 1 dev, base block-AR (vast ep70) vs block-space copy, same cell, paired block bootstrap.
Read-only. Tie rule: per_block acc as stored (ties 0.5) and ties-as-wrong recomputed from per_video_surprise."""
import json, numpy as np
R = "/data/hyuntak/project/2026/2027_cvpr/vjepa2/cvpr/results/eval/intphys1/"
def blocks(run, key):
    return {b["block_id"]: b["acc"] for b in json.load(open(R + run + "/per_block.json"))["per_block"][key]}
rng = np.random.default_rng(0)
for model, cell in [("vast_ar_state_ep70", "w16"), ("vast_ar_state_ep70", "w32"), ("ariel_ar_ep40", "w16"), ("ariel_ar_ep40", "w32")]:
    key = f"skip2_{cell}/avg"
    m = blocks(f"intphys1_dev_vith_{model}_wgarrido_{cell}", key)
    c = blocks(f"intphys1_dev_vith_ariel_ar_ep40_wgarrido_{cell}_arcopy", key)
    ids = sorted(m, key=int); d = np.array([m[i] - c[i] for i in ids])
    bs = [100 * d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)]
    print(f"{model:20s} {cell}: model {100*np.mean([m[i] for i in ids]):.2f} block-copy {100*np.mean([c[i] for i in ids]):.2f} "
          f"delta {100*d.mean():+.2f} paired block bootstrap 95% [{np.percentile(bs,2.5):+.2f}, {np.percentile(bs,97.5):+.2f}] (n_block {len(ids)})")

# ---- does VOBench+gravity fine-tuning lower IntPhys 1? same cell, paired block bootstrap (EVIDENCE table 1-8a, STORY 2-2)
for cell in ("w16", "w32"):
    key = f"skip2_{cell}/avg"
    a = blocks(f"intphys1_dev_vith_wgarrido_{cell}", key)
    b = blocks(f"intphys1_dev_vith_release_vobftg_wgarrido_{cell}", key)
    ids = sorted(a, key=int); d = np.array([b[i] - a[i] for i in ids])
    bs = [100 * d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)]
    print(f"release -> release fine-tuned (VOBench+gravity) {cell}: {100*np.mean([a[i] for i in ids]):.2f} -> {100*np.mean([b[i] for i in ids]):.2f} "
          f"delta {100*d.mean():+.2f} [{np.percentile(bs,2.5):+.2f}, {np.percentile(bs,97.5):+.2f}]")
