import csv, json
from pathlib import Path
import numpy as np
I = Path("/data2/local_datasets/world/world_analysis/cache/auto_research/v3_p3b"); ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
meta = json.load(open(I / "meta.json")); arms = list(meta["arms"]); ids = meta["video_ids"]; n = len(ids); J = 8
M = np.load(I / "p3b.npy").astype(np.float64); k = {a: i for i, a in enumerate(arms)}
rows = {r["video_id"]: r for r in csv.DictReader(open(ROOT / "data_csv/rollout_v3/index_probe.csv"))}
fa = lambda s: np.array([float(v) for v in s.split()])
X = np.stack([fa(rows[v]["px_x_by_sample"]) for v in ids]); Y = np.stack([fa(rows[v]["px_y_by_sample"]) for v in ids])
cxy = lambda f0: np.stack([(X[:, f0] + X[:, f0 + 1]) / 2, (Y[:, f0] + Y[:, f0 + 1]) / 2], -1) / 18.0
cl = lambda v: np.clip(np.floor(v), 0, 15)
L = cxy(30); T = np.stack([cl(cxy(32 + 2 * j)) for j in range(J)], 1)
traj = np.array(["|".join(map(str, t)) for t in meta["traj"]]); ut = np.unique(traj); tix = {t: np.where(traj == t)[0] for t in ut}
rng = np.random.default_rng(99); BI = [np.concatenate([tix[t] for t in rng.choice(ut, len(ut))]) for _ in range(2000)]
ci = lambda v: [round(float(np.nanmean(v)), 3)] + [round(float(x), 3) for x in np.nanpercentile([np.nanmean(v[b]) for b in BI], [2.5, 97.5])]
out = {}
for t, cols in (("tru", [1, 0]), ("app", [5, 4])):
    A = {a: M[:, :, k[a], cols] for a in arms}
    az = A["ar_z"]; azprev = np.concatenate([az[:, :1], az[:, :-1]], 1)
    # stickiness baseline: ar_z itself equals its own previous argmax, on the same subset used for tf_last copy test
    r = {"ar_z_eq_own_prev": [], "tf_last_eq_fed_pz": [], "tf_first_eq_T_jm1": [], "ar_z_eq_T_jm1": []}
    for j in range(3, J):
        mv = np.abs(T[:, j] - azprev[:, j]).max(-1) > 0
        r["ar_z_eq_own_prev"].append(round(float((az[mv, j] == azprev[mv, j]).all(-1).mean()), 3))
        r["tf_last_eq_fed_pz"].append(round(float((A["tf_last"][mv, j] == azprev[mv, j]).all(-1).mean()), 3))
        mv2 = np.abs(T[:, j] - T[:, j - 1]).max(-1) > 0
        r["tf_first_eq_T_jm1"].append(round(float((A["tf_first"][mv2, j] == T[mv2, j - 1]).all(-1).mean()), 3))
        r["ar_z_eq_T_jm1"].append(round(float((az[mv2, j] == T[mv2, j - 1]).all(-1).mean()), 3))
    out[f"sticky_{t}"] = r
    # Chebyshev distance of argmax to T_j (template-agnostic accuracy), paired diffs j3-7 mean
    dist = {a: np.abs(A[a] - T).max(-1) for a in arms}
    out[f"cheb_dist_mean_j3_7_{t}"] = {a: ci(dist[a][:, 3:].mean(1)) for a in arms}
    out[f"cheb_tf_first_minus_tf_last_{t}"] = ci(dist["tf_first"][:, 3:].mean(1) - dist["tf_last"][:, 3:].mean(1))
    out[f"cheb_tf_first_minus_ar_z_{t}"] = ci(dist["tf_first"][:, 3:].mean(1) - dist["ar_z"][:, 3:].mean(1))
# consensus: (clip,j) where both templates agree on the argmax, per arm, fraction on T_j (Chebyshev<=1)
cons = {}
for a in arms:
    at, aa = M[:, :, k[a], [1, 0]], M[:, :, k[a], [5, 4]]
    agree = (at == aa).all(-1)
    hit = (np.abs(at - T).max(-1) <= 1)
    cons[a] = {"agree_frac_j3_7": round(float(agree[:, 3:].mean()), 3), "hit_given_agree_j3_7": round(float(hit[:, 3:][agree[:, 3:]].mean()), 3),
               "hit_tru_all_j3_7": round(float(hit[:, 3:].mean()), 3), "hit_app_all_j3_7": round(float((np.abs(aa - T).max(-1) <= 1)[:, 3:].mean()), 3)}
out["consensus"] = cons
json.dump(out, open(Path(__file__).resolve().parent / "verify_p3b_extra_out.json", "w"), indent=1)
print(json.dumps(out, indent=1))
