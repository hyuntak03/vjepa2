import json; from pathlib import Path; import numpy as np
I = Path("/data2/local_datasets/world/world_analysis/cache/auto_research/v3_p3b"); H5 = I.parent / "v3_h5"
m = json.load(open(I / "meta.json")); M = np.load(I / "p3b.npy"); a3 = list(m["arms"])
m5 = json.load(open(H5 / "meta.json")); pos = {v: i for i, v in enumerate(m5["video_ids"])}; a5 = list(m5["arms"])
H = np.asarray(np.load(H5 / "h5.npy", mmap_mode="r")[[pos[v] for v in m["video_ids"]], m5["Cs"].index(16)])
for x3, x5 in (("ar_hA", "ar_h"), ("ar_z", "ar_z"), ("ar_pA", "ar_p")):
    ag = [round(float((M[:, j, a3.index(x3), :2] == H[:, j, a5.index(x5), :2]).all(-1).mean()), 3) for j in range(8)]
    l1 = [round(float(np.abs(M[:, j, a3.index(x3), 8] - H[:, j, a5.index(x5), 8]).mean()), 5) for j in range(8)]
    print(x3, "vs H5", x5, "argmax agree", ag, "mean|dL1|", l1)
