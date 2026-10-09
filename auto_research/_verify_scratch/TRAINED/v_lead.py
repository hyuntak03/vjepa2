# within-slot (fixed t) regression of predicted along-displacement on true along-displacement (velocity varies across clips): multiplicative vs additive lead
import csv, json
from pathlib import Path
import numpy as np
C = Path("/data2/local_datasets/world/world_analysis/cache/auto_research"); ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
meta = json.load(open(C / "v3_p2" / "meta.json")); ids = meta["video_ids"]; arms = meta["arms"]
idx = {r["video_id"]: r for r in csv.DictReader(open(ROOT / "data_csv/rollout_v3/index_probe.csv"))}
fl = lambda s: np.array([float(v) for v in s.split()])
X = np.stack([fl(idx[v]["px_x_by_sample"]) for v in ids]); Y = np.stack([fl(idx[v]["px_y_by_sample"]) for v in ids])
scen = np.array(meta["scenario"]); traj = np.array(["|".join(map(str, t)) for t in meta["traj"]]); ut = np.unique(traj); tix = {u: np.where(traj == u)[0] for u in ut}
rng = np.random.default_rng(5); BI = [np.concatenate([tix[u] for u in rng.choice(ut, len(ut))]) for _ in range(500)]
keep = scen != "arc"
for arm_name in ("s1_C16", "s2_C16"):
    ai = [k for k, A in enumerate(arms) if A["name"] == arm_name][0]; A = arms[ai]; fr, nc, nf = A["frames"], A["n_ctx"], A["n_fut"]
    fa, fb = fr[2*nc-2], fr[2*nc-1]; L = np.stack([(X[:, fa]+X[:, fb])/2, (Y[:, fa]+Y[:, fb])/2], -1)/18.0
    print("==", arm_name)
    for tag in ["", "_pv1", "_ariel", "_v11ft"]:
        Lc = np.load(C / f"v3_p2{tag}" / "loc_app.npy")[:, ai, :nf][..., [1, 0]]
        row = []
        for t in ([2, 4, 6, 8, 10] if nf > 8 else [2, 4, 6]):
            ct = np.stack([(X[:, fr[2*(nc+t)]]+X[:, fr[2*(nc+t)+1]])/2, (Y[:, fr[2*(nc+t)]]+Y[:, fr[2*(nc+t)+1]])/2], -1)/18.0
            d = ct - L; dist = np.linalg.norm(d, axis=-1); u = d/np.maximum(dist, 1e-6)[:, None]
            al = ((Lc[:, t] + 0.5 - L) * u).sum(-1)
            m = keep & (dist > 0.3)
            b, a = np.polyfit(dist[m], al[m], 1)
            bs = []
            for bi in BI:
                mm = bi[m[bi]]; bs.append(np.polyfit(dist[mm], al[mm], 1))
            bs = np.array(bs)
            row.append(f"t{t}: b {b:.2f}[{np.percentile(bs[:,0],2.5):.2f},{np.percentile(bs[:,0],97.5):.2f}] a {a:+.2f}[{np.percentile(bs[:,1],2.5):+.2f},{np.percentile(bs[:,1],97.5):+.2f}] (true {dist[m].mean():.2f})")
        print(tag or "release", " | ".join(row))
