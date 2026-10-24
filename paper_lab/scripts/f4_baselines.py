#!/usr/bin/env python
"""F4 (round-1 review, R2 'fix this one thing'): same-space baselines for VOBench pair-accuracy cells.

Block target space (per-block LN(target_encoder), the block-AR scoring space), from the 10-13 assembly artifacts
(z_research/VOBench_v1/headroom/vobench_r*.npz, made by design_headroom.py extract; read-only here):
  renders  copy    = last context block copied to every future step (no prediction)
           ar      = Ariel block-AR (output feedback, ep40) rollout
           move_or = true-trajectory oracle: last fully visible object 3x3 tokens pasted at the true future position
  scores   all-token  : surprise = mean over steps and all 256 cells   (the standard scoring)
           obj-region : surprise = mean over steps and the object cells only
                        (union of both futures' object 3x3 from render labels, same mask as headroom_margin_maps.py)
  pair     correct if S(impossible) > S(possible); ties 0.5 (same rule as design_headroom.analyze)
Context-average copy is NOT in these artifacts (needs tokens) -> filled by E1 / E2b (paper_lab/_scratch/tspace).

Run:  /data/hyuntak/anaconda3/envs/vjepa2/bin/python paper_lab/scripts/f4_baselines.py
Out:  paper_lab/_scratch/f4_baselines.json (+ table on stdout)
"""
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, "z_research/scripts/analysis")
import design_headroom as H  # noqa: E402

RENDERS = ("copy", "ar", "move_or")
Z = H.load("vobench")
vid = {v: i for i, v in enumerate(Z["video_id"])}
rows = {r["video_id"]: r for r in csv.DictReader(open(H.SETS["vobench"]["index"]))}
test_blocks = {r["block_id"] for r in csv.DictReader(open("data_csv/vobench_v1_split_v2/index_test.csv"))}
blk = defaultdict(dict)
for v, r in rows.items():
    blk[r["source_block"]][r["variant"]] = v


def fmask(lab, arm, dil=1):
    M = np.zeros((8, 16, 16), bool)
    L = lab.get(arm)
    if not L or L[0] is None:
        return M
    P = np.array([[L[3 * s]["u"], L[3 * s]["v"]] for s in range(32)])[16:].reshape(8, 2, 2).mean(1)
    for k, p in enumerate(P):
        if not (0 <= p[0] < 512 and 0 <= p[1] < 512):
            continue
        y, x = H.tok(p)
        M[k, max(y - dil, 0):y + dil + 1, max(x - dil, 0):x + dil + 1] = True
    return M


acc = defaultdict(lambda: defaultdict(list))
n_empty_mask = 0
for sb, arms in blk.items():
    if not all(a in arms for a in ("pos_a", "pos_b", "imp_ab", "imp_ba")) or not all(arms[a] in vid for a in arms):
        continue
    r = rows[arms["pos_a"]]
    mot = "static" if r["motion"] == "flat_static" else "moving"
    lab = json.load(open(H.FR / Path(r["file_name"]).parent / "label.json"))["labels"]
    in_test = str(r.get("block_id", "")) in test_blocks or str(sb) in test_blocks
    for pos, imp, d, fa, fb in (("pos_a", "imp_ab", "A", "A", "B"), ("pos_b", "imp_ba", "B", "B", "A")):
        M = fmask(lab, fa) | fmask(lab, fb)
        ip, ii = vid[arms[pos]], vid[arms[imp]]
        keys = ["all", f"{r['occlusion']}/{mot}", f"{r['violation']}/{r['occlusion']}/{mot}", f"{r['objset']}/{r['occlusion']}"]
        if in_test:
            keys += [f"test/{r['objset']}/{r['occlusion']}", f"test/{r['occlusion']}/{mot}"]
        for m in RENDERS:
            # all-token
            sp, si = Z[f"S_{m}"][ip].mean(), Z[f"S_{m}"][ii].mean()
            hit = 1.0 if si > sp else (0.5 if si == sp else 0.0)
            # object region
            if M.any():
                Ep = Z[f"E_{m}"][ip].astype(np.float32)[M].mean()
                Ei = Z[f"E_{m}"][ii].astype(np.float32)[M].mean()
                hit_o = 1.0 if Ei > Ep else (0.5 if Ei == Ep else 0.0)
            else:
                hit_o = np.nan
            for k in keys:
                acc[k][m].append(hit)
                acc[k][m + "@obj"].append(hit_o)
        if not M.any():
            n_empty_mask += 1


def ci(x, B=2000, seed=0):
    x = np.asarray([v for v in x if not np.isnan(v)])
    if len(x) == 0:
        return (float("nan"), float("nan"), float("nan"), 0)
    rng = np.random.default_rng(seed)
    bs = rng.choice(x, size=(B, len(x)), replace=True).mean(1)
    return (100 * x.mean(), 100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5), len(x))


res = {k: {m: ci(v) for m, v in d.items()} for k, d in acc.items()}
res["_meta"] = {"n_pairs_with_empty_object_mask": n_empty_mask,
                "note": "pair bootstrap CI (pairs, not blocks); ties 0.5; block target space; Ariel AR ep40 as the 'ar' render"}
out = Path("paper_lab/_scratch/f4_baselines.json")
out.write_text(json.dumps(res, indent=1, ensure_ascii=False))

order = ["all", "visible/moving", "early/moving", "late/moving", "late/static",
         "permanence/late/moving", "shape/late/moving", "color/late/moving",
         "primitive/late", "realistic/late", "test/primitive/late", "test/realistic/late", "test/late/moving"]
print(f"{'cell':28s} | " + " | ".join(f"{m:>16s}" for m in ("copy", "ar", "move_or", "copy@obj", "ar@obj", "move_or@obj")))
for k in order:
    if k not in res:
        continue
    print(f"{k:28s} | " + " | ".join(f"{res[k][m][0]:5.1f} (n={res[k][m][3]:3d})" for m in ("copy", "ar", "move_or", "copy@obj", "ar@obj", "move_or@obj")))
print("empty object masks:", n_empty_mask, "->", out)
