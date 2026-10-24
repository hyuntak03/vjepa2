#!/usr/bin/env python
"""phys2d_sup boundary 행 검사 (2026-10-18): 문맥 (ctx_start, b) 에 충돌 · 이탈 전환 없음, 사건 post ∈ [b−1, b+2],
예측 [b, end) 에 화면 밖 물체 첫 보임 없음. + 사라짐의 원인 (화면 밖 vs 가림막).

  python z_training/data/verify_phys2d_sup_rows.py /workspace/dataset/training_data/phys2d_sup_v2/train
"""
import sys, csv, json, collections, numpy as np
root = sys.argv[1]
bad = collections.Counter(); n = 0; off = collections.Counter(); kinds = collections.Counter(); cm = []
for r in csv.DictReader(open(root + "/boundary.tsv"), delimiter="\t"):
    n += 1; stem = r["path"][:-4]; js = json.load(open(stem + ".json"))
    b, c0, end, post = int(r["boundary_frame"]), int(r["ctx_start"]), int(r["n_frames"]), int(r["event_frame"])
    off[b - post] += 1; kinds[r["event"]] += 1; cm.append(b - c0)
    role = [o["role"] for o in js["objects"]]
    gate = [e["frame"] for e in js["events"] if role[e["obj"]] != "distractor" and (e["kind"] == "leave" or (e["kind"] == "impact" and e["dv"] >= 60))]
    if any(c0 < p < b and p < post for p in gate): bad["gate transition in context"] += 1
    if not (b - 1 <= post <= b + 2): bad["event not at boundary"] += 1
    vis = np.load(stem + "_states.npz")["vis_frac"]
    for i, ro in enumerate(role):
        if ro == "spawn":
            s = np.where(vis[:, i] > 0.02)[0]
            if len(s) and b <= s[0] < end: bad["spawn first seen in prediction"] += 1
print(f"rows {n} kinds {dict(kinds)} b-post {dict(sorted(off.items()))}  violations: {dict(bad) or 'none'}")
cm = np.array(cm); print("clean ctx >=4/8/16/32: " + " / ".join(f"{100*(cm>=k).mean():.0f}%" for k in (4, 8, 16, 32)))
# 화면 밖 이탈 vs 가림
idx = [l.split()[0] for l in open(root + "/index.csv")]
lost_off = lost_occ = 0; clips_off = 0; frac_off = []
for mp4 in idx:
    stem = mp4[:-4]; js = json.load(open(stem + ".json")); z = np.load(stem + "_states.npz")
    role = [o["role"] for o in js["objects"]]
    seen, inf, hid, pres = z["seen"], z["inframe_frac"], z["hidden"], z["present"]
    any_off = False
    for i, ro in enumerate(role):
        if ro == "distractor": continue
        offm = seen[:, i] & pres[:, i] & (inf[:, i] < 0.5)
        frac_off.append(offm.mean())
        if offm.any(): any_off = True
        # 이탈 사건: 보이던 물체가 사라지는 순간 — 화면 밖 vs 가림막
        vis = (z["vis_frac"][:, i] > 0.3)
        for t in range(1, len(vis)):
            if vis[t - 1] and not vis[t]:
                if inf[t, i] < 0.5: lost_off += 1
                elif hid[t, i] or z["vis_frac"][t, i] < 0.3: lost_occ += 1
    clips_off += any_off
print(f"clips with a seen object offscreen: {clips_off}/{len(idx)} ({clips_off/len(idx):.0%}); object-frames offscreen after seen: {np.mean(frac_off):.1%}")
print(f"disappearances: offscreen {lost_off} vs occluded {lost_occ}")
