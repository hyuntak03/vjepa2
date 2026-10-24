#!/usr/bin/env python
"""EVIDENCE.md v2 (paper_lab, round 1 revision, 2026-10-21) — recompute key numbers from raw artifacts.
v1 checks (unchanged) + v2 checks (F4 same-space baselines, R2 corrections, same-cell IntPhys 1).

Read-only. Prints one line per check: id | recomputed value | value quoted in docs | match?
Run:  /data/hyuntak/anaconda3/envs/vjepa2/bin/python paper_lab/scripts/verify_evidence_v2.py
All paths are relative to the main working tree root.
"""
import csv
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
VOB = "z_research/VOBench_v1"
ROWS = []


def J(p):
    with open(os.path.join(ROOT, p)) as f:
        return json.load(f)


def check(cid, got, quoted, tol=0.051):
    """got / quoted are numbers or tuples of numbers (same length)."""
    g = got if isinstance(got, (tuple, list)) else (got,)
    q = quoted if isinstance(quoted, (tuple, list)) else (quoted,)
    ok = len(g) == len(q) and all(abs(a - b) <= tol for a, b in zip(g, q))
    ROWS.append((cid, tuple(round(x, 3) for x in g), tuple(q), "OK" if ok else "MISMATCH"))


def safe(cid, fn, quoted, tol=0.051):
    try:
        check(cid, fn(), quoted, tol)
    except Exception as e:  # noqa: BLE001
        ROWS.append((cid, ("ERR",), tuple(quoted) if isinstance(quoted, (list, tuple)) else (quoted,), f"ERROR {type(e).__name__}: {e}"))


# ---------------------------------------------------------------- P2: leakage (copy baseline)
def ip1():
    d = J("z_research/previous/TrainingEffects/ip1_copy/ip1_copy.json")["cells"]
    return (d["release|std|skip2_w32"]["macro"][0], d["copy|std|skip2_w32"]["macro"][0])


safe("E-P2-1 IntPhys1 release vs copy (skip2_w32)", ip1, (88.89, 85.0), tol=0.01)


def ip2():
    d = J("auto_research/exp_results/verify/intphys2_permanence_audit.json")
    return (d["release_w48"]["at_best_C"]["overall"][1], d["copy_w48"]["at_best_C"]["overall"][1])


safe("E-P2-2 IntPhys2 Main release vs copy (w48 best C)", ip2, (54.3, 55.5))


def rollout_pos():
    d = J("auto_research/exp_results/scene/rollout2_copy_voe.json")
    return (d["ledge"]["p"], d["ledge"]["copy"], d["wall"]["p"], d["wall"]["copy"])


safe("E-P2-3 position-only pairs ledge p/copy, wall p/copy", rollout_pos, (0.0, 7.9, 17.6, 2.3))


def v11():
    return J("z_research/previous/IntPhysGenV11/exp_results/report.json")["scoring"]["overall"][0]["overall"]


safe("E-P3-0 v11 overall (surprise_c16t32)", v11, 73.37, tol=0.01)


def dinof_v11():
    b = "z_research/previous/IntPhysGenV11/exp_results/surprise_c16t32__v11_dinof_highres"
    return tuple(100 * J(f"{b}{s}/summary.json")["surprise"]["overall"]["block_pairwise"] for s in ("", "_copy"))


safe("E-P2-4 DINO-Foresight v11 predictor vs copy", dinof_v11, (77.10, 76.44))


def v11r():
    d = J("z_research/previous/v11_realistic/exp_results/tables.json")
    return (d["realistic_overall"]["acc"], d["ledge_overall"]["acc"], d["ledge_overall"]["n"])


safe("E-P3-1 v11_realistic vanish overall, ledge acc, ledge n", v11r, (79.58, 0.0, 120))


def ip1_trained():
    d = J("z_research/previous/TrainingEffects/ip1_copy/ip1_copy.json")["cells"]
    return tuple(d[f"{t}|std|skip2_w32"]["macro"][0] for t in ("v11_e10", "ip1_e40", "ariel_ep43"))


safe("E-P3-5 IntPhys1 skip2_w32: v11 post-FT / IntPhys1 post-FT / Ariel block-causal ep43", ip1_trained, (78.33, 93.89, 68.89), tol=0.01)


# ---------------------------------------------------------------- VOBench (release vs AR, copy)
def vob(model, objset, key):
    d = J(f"{VOB}/{model}/surprise/tables.json")[objset]
    if key == "overall":
        return d["overall"]["acc"]
    if key == "copy":
        return d["copy"]["copy"]["acc"]
    viol, occ = key.split("/")
    return d["viol_x_occ"][viol][occ]["acc"]


for model, quoted in [
    ("vjepa2", (71.7, 81.9, 73.8, 81.9, 57.9, 65.7, 55.6, 61.1)),
    ("ariel_ar_ep40", (81.9, 86.4, 77.6, 81.4, 75.0, 78.2, 98.6, 100.0)),
]:
    safe(
        f"E-VOB {model}: overall p/r, copy p/r, late p/r, late-perm p/r",
        lambda m=model: tuple(
            vob(m, o, k)
            for k in ("overall", "copy", "all/late", "permanence/late")
            for o in ("primitive", "realistic")
        ),
        quoted,
    )

safe(
    "E-VOB vast_ar_state_ep70: overall p/r, late p/r, late-perm p/r",
    lambda: tuple(vob("vast_ar_state_ep70", o, k) for k in ("overall", "all/late", "permanence/late") for o in ("primitive", "realistic")),
    (81.5, 85.3, 73.6, 77.3, 93.1, 95.8),
)
safe(
    "E-VOB video-context AR (vll7 ctxt_tgtvideo ep70): overall p/r, late p/r, late-perm p/r",
    lambda: tuple(vob("vll7_ctxt_tgtvideo", o, k) for k in ("overall", "all/late", "permanence/late") for o in ("primitive", "realistic")),
    (71.7, 83.6, 50.9, 61.6, 48.6, 58.3),
)


# ---------------------------------------------------------------- diagnosis chain (vll5, 10-13)
def headroom(cell):
    d = J(f"{VOB}/headroom/headroom.json")["vobench"][cell]
    return tuple(100 * d[m][0] for m in ("copy", "ar", "move_or"))


safe("E-D-assembly all: copy / AR / oracle-move", lambda: headroom("all"), (79.4, 84.5, 82.2))
safe("E-D-assembly late-perm-moving: copy / AR / oracle-move", lambda: headroom("permanence/late/moving"), (53.3, 99.2, 53.3))


def anchor(model):
    d = J(f"{VOB}/bottleneck/anchor/{model}.json")
    f = d["follow_last_index"]
    return (f["stay"][3], f["back"][3], f["double"][3], d["stay6"]["ds"][3], d["back6"]["ds"][3])


safe("E-D-anchor AR ep40 follow k4 stay/back/double, block6 ds k4 stay6/back6", lambda: anchor("ar40"), (0.70, 0.86, 0.47, 0.02, -0.06))
safe("E-D-anchor release follow k4 stay/back/double, block6 ds k4", lambda: anchor("release"), (0.88, 0.70, 0.14, -0.59, -1.18))
safe("E-D-anchor AR-state ep40 follow k4, block6 ds k4", lambda: anchor("ar_state40"), (0.63, 0.74, 0.47, 0.00, -0.06))


def locality(fname):
    d = J(f"{VOB}/headroom/{fname}")
    out = []
    for k in ("primitive/permanence/early/moving", "primitive/color/early/moving"):
        r = d[k]["radius"]["1"]
        out += [r[0], r[1]]
    return tuple(out)


safe("E-D-locality block target: 3x3 area/share perm, color (early moving prim)", lambda: locality("token_locality.json"), (0.035, 0.052, 0.035, 0.034), tol=0.006)
safe("E-D-locality window target: 3x3 area/share perm, color", lambda: locality("token_locality_window.json"), (0.035, 0.054, 0.035, 0.031), tol=0.006)


def hist(name, ctx):
    d = J(f"{VOB}/bottleneck/history/{name}.json")["all"][f"history_value_{ctx}"]
    return (min(d[1:]), max(d[1:]))


safe("E-D-history SSv2 AR 8->2 blocks (k2-k8 min,max)", lambda: hist("ssv2_ar40", "ctx2"), (0.01, 0.02), tol=0.011)
safe("E-D-history VOB late AR 8->1 block (k2-k8 min,max)", lambda: hist("vob_late_ar40", "ctx1"), (0.46, 0.53), tol=0.011)


def vel_fixed():
    d = J(f"{VOB}/bottleneck/velocity_fixed/result.json")
    a = d["ar40"]["st_l"]["speed_ratio_fixed_k1_decoder"]
    s = d["ar_state40"]["st_l"]["speed_ratio_fixed_k1_decoder"]
    return (a[0], a[-1], s[0], s[-1])


safe("E-D-velocity fixed k1 decoder AR obj 0.86->0.31, AR-state 0.81->0.21", vel_fixed, (0.86, 0.31, 0.81, 0.21), tol=0.011)


# ---------------------------------------------------------------- eval matrix: VOBench late on split v2 test
def late_split(run):
    idx = {r["block_id"]: r for r in csv.DictReader(open(os.path.join(ROOT, "data_csv/vobench_v1_split_v2/index_test.csv")))}
    pb = J(f"{run}/eval/surprise_c16t32__vobench_v1_split2_test_latest/per_block.json")["per_block"]
    agg = {}
    for b in pb:
        r = idx.get(str(b["block_id"]))
        if r is None or r.get("occlusion") != "late":
            continue
        s = agg.setdefault(r["objset"], [0.0, 0])
        s[0] += b["acc"] * b["n_pairs"]
        s[1] += b["n_pairs"]
    sc = 100.0 if max(b["acc"] for b in pb) <= 1.0 else 1.0
    return tuple(sc * agg[o][0] / agg[o][1] for o in ("primitive", "realistic"))


safe("E-M release+PFT(g) VOBench late p/r", lambda: late_split("z_training/runs/vobench_release_postft_g"), (88.0, 91.7))
safe("E-M vast AR+PFT(g) VOBench late p/r", lambda: late_split("z_training/runs/vobench_vast_ar_state_postft_g"), (89.8, 89.8))


def ip1_copy_w16():
    c = J("cvpr/results/eval/intphys1/copy/intphys1_dev_vith/copy_summary.json")["cells"]["std"]["skip2_w16"]
    return (c["model"], c["copy"])


safe("E-P2-1b IntPhys1 official-config cell skip2_w16: release vs copy (10-20 rerun)", ip1_copy_w16, (83.33, 73.33), tol=0.01)


# ---------------------------------------------------------------- eval-matrix style rows via the user's sheet logic
def sheet(model, vob_dir=None, grav_dir=None):
    sys.path.insert(0, os.path.join(ROOT, "z_research/scripts/analysis"))
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        import sheet_row_report as S  # noqa: E402
        from pathlib import Path

        g = S.gravity(Path(grav_dir or f"z_research/VOBench_v1/exp_results/surprise_c16t32__vobench_v1_gravity_{model}"))
        ip = S.intphys1(model)
        v = S.vob_late(Path(vob_dir)) if vob_dir else None
        return g, ip, v
    finally:
        os.chdir(cwd)


def row(model, vob_dir=None, grav_dir=None):
    g, ip, v = sheet(model, vob_dir, grav_dir)
    out = [g["primitive"], g["realistic"], ip["acc"]]
    if v:
        out += [v["primitive"], v["realistic"]]
    return tuple(out)


TBD = "cvpr/results/eval/testbed"
safe("E-M row release: gravity p/r, IntPhys1", lambda: row("vith"), (0.0, 0.0, 88.9))
safe("E-M row release+PFT(g): gravity p/r, IntPhys1", lambda: row("vith_release_vobftg"), (100.0, 92.6, 83.3))
safe("E-M row vast AR: gravity p/r, IntPhys1", lambda: row("vith_vast_ar_state_ep70"), (18.5, 11.1, 90.0))
safe("E-M row vast AR+PFT(g): gravity p/r, IntPhys1", lambda: row("vith_vast_ar_state_ep70_vobftg"), (57.4, 59.3, 90.0))
safe(
    "E-T video-context AR (vll7): gravity p/r, IntPhys1 best, VOBench late p/r (split v2 test)",
    lambda: row("vith_vll7_ctxt_tgtvideo", f"{TBD}/vobench_v1_vith_vll7_ctxt_tgtvideo", f"{TBD}/vobench_v1_gravity_vith_vll7_ctxt_tgtvideo"),
    (5.6, 5.6, 47.8, 50.9, 60.2),
)
safe(
    "E-T phys2d-single PFT no-sup: gravity p/r, IntPhys1, VOBench late p/r",
    lambda: row("vith_pft_single_nosup", f"{TBD}/vobench_v1_late_vith_pft_single_nosup", f"{TBD}/vobench_v1_gravity_vith_pft_single_nosup"),
    (16.7, 11.1, 92.2, 68.5, 71.3),
)
safe(
    "E-T phys2d-single PFT state-sup w0.05: gravity p/r, IntPhys1, VOBench late p/r",
    lambda: row("vith_pft_single_w005", f"{TBD}/vobench_v1_late_vith_pft_single_w005", f"{TBD}/vobench_v1_gravity_vith_pft_single_w005"),
    (18.5, 9.3, 91.1, 64.8, 69.4),
)
safe(
    "E-RR release VOBench late p/r, two runs (z_research exp_results vs cvpr testbed)",
    lambda: sheet("vith", "z_research/VOBench_v1/exp_results/surprise_c16t32__vobench_v1_vith")[2]["primitive"]
    - sheet("vith", f"{TBD}/vobench_v1_vith")[2]["primitive"],
    -1.0,
)

# ================================================================ v2 additions (round-1 review fixes)
safe("v2 #2 IntPhys1 skip2_w16 release vs copy, SAME run as the CI (ip1_copy.json)",
     lambda: (J("z_research/previous/TrainingEffects/ip1_copy/ip1_copy.json")["cells"]["release|std|skip2_w16"]["macro"][0],
              J("z_research/previous/TrainingEffects/ip1_copy/ip1_copy.json")["cells"]["copy|std|skip2_w16"]["macro"][0]),
     (83.33, 73.89), tol=0.01)


def ip1_copy_range():
    c = J("z_research/previous/TrainingEffects/ip1_copy/ip1_copy.json")["cells"]
    v = [c[k]["macro"][0] for k in c if k.startswith("copy|")]
    return (min(v), max(v), len(v))


safe("v2 #2 IntPhys1 copy range over all 6 cells, 09-25 run (min, max, n)", ip1_copy_range, (63.33, 85.0, 6), tol=0.01)


def ip1_copy_range_rerun():
    c = J("cvpr/results/eval/intphys1/copy/intphys1_dev_vith/copy_summary.json")["cells"]
    v = [c[t][k]["copy"] for t in c for k in c[t]]
    nb = [c[t][k]["alt_notebook"]["copy"] for t in c for k in c[t]]
    return (min(v), max(v), len(v), max(nb))


safe("v2 #2 IntPhys1 copy range over all 6 cells, 10-20 rerun (min, max, n, notebook-rule max)", ip1_copy_range_rerun, (62.22, 85.0, 6, 88.33), tol=0.01)
safe("v2 #6 position-only pairs, context-mean copy ledge / wall",
     lambda: (J("auto_research/exp_results/scene/rollout2_copy_voe.json")["ledge"]["copy_mean"],
              J("auto_research/exp_results/scene/rollout2_copy_voe.json")["wall"]["copy_mean"]), (1.5, 66.3))
safe("v2 #31 SSv2 AR 8->1 block k2-k8 (min,max)",
     lambda: (min(J(f"{VOB}/bottleneck/history/ssv2_ar40.json")["all"]["history_value_ctx1"][1:]),
              max(J(f"{VOB}/bottleneck/history/ssv2_ar40.json")["all"]["history_value_ctx1"][1:])), (0.021, 0.062), tol=0.0015)
safe("v2 #30 AR ep40 back6 ds k6-k8 and paste ds k6-k8",
     lambda: tuple(J(f"{VOB}/bottleneck/anchor/ar40.json")["back6"]["ds"][5:]) + tuple(J(f"{VOB}/bottleneck/anchor/ar40.json")["paste"]["ds"][5:]),
     (-0.25, -0.34, -0.50, 0.12, 0.14, 0.17), tol=0.011)


def f4(cell, keys):
    d = J("paper_lab/_scratch/f4_baselines.json")[cell]
    return tuple(d[k][0] for k in keys)


K6 = ("copy", "ar", "move_or", "copy@obj", "ar@obj", "move_or@obj")
safe("v2 #57 late/moving: copy, AR, oracle (all-token) | copy, AR, oracle (object region)", lambda: f4("late/moving", K6),
     (54.4, 72.8, 58.3, 51.1, 51.4, 95.3))
safe("v2 #57 permanence/late/moving (all-token | object region)", lambda: f4("permanence/late/moving", K6),
     (53.3, 99.2, 53.3, 50.0, 50.0, 100.0))
safe("v2 #56 visible/moving (all-token | object region)", lambda: f4("visible/moving", K6), (90.0, 87.5, 89.2, 68.3, 92.5, 99.2))
safe("v2 #56 early/moving (all-token | object region)", lambda: f4("early/moving", K6), (89.4, 87.2, 91.4, 63.3, 89.7, 97.8))
safe("v2 #57 late/static object region copy, AR, oracle", lambda: f4("late/static", ("copy@obj", "ar@obj", "move_or@obj")), (63.9, 93.1, 90.3))
safe("v2 #18 split-v2-test late p / r: oracle@obj, copy@obj",
     lambda: f4("test/primitive/late", ("move_or@obj", "copy@obj")) + f4("test/realistic/late", ("move_or@obj", "copy@obj")),
     (92.6, 55.6, 98.1, 54.6))


def grav108():
    sys.path.insert(0, os.path.join(ROOT, "z_research/scripts/analysis"))
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        import design_headroom as H
        from collections import defaultdict
        Zg = H.load("gravity")
        vidg = {v: i for i, v in enumerate(Zg["video_id"])}
        rows = {r["video_id"]: r for r in csv.DictReader(open(H.SETS["gravity"]["index"]))}
        test = {r["block_id"] for r in csv.DictReader(open("data_csv/vobench_v1_gravity_split/index_test.csv"))}
        blk = defaultdict(dict)
        for v, r in rows.items():
            blk[r["source_block"]][r["variant"]] = v
        cnt = defaultdict(int)
        for sb, arms in blk.items():
            r = rows[arms["pos"]]
            if r["block_id"] not in test or not all(a in vidg for a in arms.values()):
                continue
            law = r["law"]
            for m in ("copy", "move_or"):
                sc = {a: Zg[f"S_{m}"][vidg[v]].mean() for a, v in arms.items()}
                cnt[(m, law)] += int(min(sc, key=sc.get) == "pos")
            cnt[("n", law)] += 1
        return (cnt[("copy", "ledge")], cnt[("move_or", "ledge")], cnt[("n", "ledge")],
                cnt[("copy", "arc")], cnt[("move_or", "arc")], cnt[("n", "arc")])
    finally:
        os.chdir(cwd)


safe("v2 #23 gravity test108 block copy / oracle: ledge (copy, oracle, n), arc (copy, oracle, n)", grav108, (3, 7, 72, 4, 16, 36), tol=0.01)


def ip1_cell(model, combo):
    sys.path.insert(0, os.path.join(ROOT, "z_research/scripts/analysis"))
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        import garrido_rescore as G
        from pathlib import Path
        w = combo.split("_")[1]
        cb = G.rescore(Path(f"cvpr/results/eval/intphys1/intphys1_dev_{model}_wgarrido_{w}"))["combos"]
        return G.select({combo: cb[combo]}, "intphys")["accuracy"]
    finally:
        os.chdir(cwd)


for model, q in [("vith_release_vobftg", (83.3, 82.2)), ("vith_pft_ip2pair_lora", (57.8, 64.4)), ("vith_pft_ip2pair", (83.3, 85.6)),
                 ("vith_vast_ar_state_ep70", (90.0, 88.9)), ("vith_pft_single_nosup", (90.6, 92.2)), ("vith_phys2d_futureA", (91.1, 87.8))]:
    safe(f"v2 table 1-8a IntPhys1 same cell skip2_w16 / skip2_w32: {model}",
         lambda m=model: (ip1_cell(m, "skip2_w16"), ip1_cell(m, "skip2_w32")), q)


def e2_pairs(cell):
    import numpy as np
    from collections import defaultdict
    z = np.load(os.path.join(ROOT, "paper_lab/_scratch/tspace/vobench_r0.npz"), allow_pickle=True)
    vid = {v: i for i, v in enumerate(z["video_id"])}
    rows = {r["video_id"]: r for r in csv.DictReader(open(os.path.join(ROOT, "data_csv/vobench_v1/index.csv")))}
    blk = defaultdict(dict)
    for v, r in rows.items():
        blk[r["source_block"]][r["variant"]] = v
    acc = defaultdict(list)
    for arms in blk.values():
        if not all(a in arms and arms[a] in vid for a in ("pos_a", "pos_b", "imp_ab", "imp_ba")):
            continue
        r = rows[arms["pos_a"]]
        mot = "static" if r["motion"] == "flat_static" else "moving"
        if f"{r['violation']}/{r['occlusion']}/{mot}" != cell and f"{r['occlusion']}/{mot}" != cell:
            continue
        for pos, imp in (("pos_a", "imp_ab"), ("pos_b", "imp_ba")):
            for m in ("copy", "mean8", "mc_or"):
                a = z[f"raw/{m}/all"]
                pa, ia = a[vid[arms[pos]]].mean(), a[vid[arms[imp]]].mean()
                acc[m].append(1.0 if ia > pa else (0.5 if ia == pa else 0.0))
            for m in ("copy", "mean8", "mc_or"):
                o = z[f"raw/{m}/obj"]
                po, io = o[vid[arms[pos]]], o[vid[arms[imp]]]
                acc[m + "@obj"].append(1.0 if io > po else (0.5 if io == po else 0.0))
    return tuple(100 * np.mean(acc[m]) for m in ("copy", "mean8", "mc_or", "copy@obj", "mean8@obj", "mc_or@obj"))


safe("v2 #16 E2 permanence/late/moving all-token copy, ctx-mean8, oracle(bg) | obj copy, mean8, oracle",
     lambda: e2_pairs("permanence/late/moving")[:3], (53.3, 95.0, 95.8))
safe("v2 #57 E2 late/moving obj-region (E2 definition): copy, ctx-mean8, oracle(bg)",
     lambda: e2_pairs("late/moving")[3:], (33.6, 34.2, 80.3))


def e2_adv8():
    import numpy as np
    z = np.load(os.path.join(ROOT, "paper_lab/_scratch/tspace/vobench_r0.npz"), allow_pickle=True)
    out = []
    for sp in ("raw", "smooth"):
        a, c = z[f"{sp}/mean8/all"].mean(0), z[f"{sp}/copy/all"].mean(0)
        r = 100 * (1 - a / c)
        out += [r[0], r[-1]]
    a, o = z["raw/mean8/all"].mean(), z["raw/mc_or/all"].mean()
    out.append(100 * (1 - o / a))
    return tuple(out)


safe("v2 #64 context-mean reward k1, k8 (raw), k1, k8 (noise-averaged), carry reward all (raw)", e2_adv8, (-2.3, 20.8, -8.8, 21.2, 1.3))


def e3_vast():
    d = J("paper_lab/_scratch/e3/anchor/vast70.json")
    return (d["stay6"]["ds"][0], d["back6"]["ds"][0], d["stay"]["ds"][0], d["back"]["ds"][0], d["paste"]["ds"][0], d["follow_last_index"]["stay"][0])


safe("v2 #65 E3 baseline block AR k1: stay6, back6, stay, back, paste, follow(stay)", e3_vast, (0.05, 0.06, -1.30, -2.42, -0.02, 0.84), tol=0.011)


def e1b():
    d = J("paper_lab/_scratch/tspace/x2_result.json")["vobench"]
    return tuple(100 * d[k]["adv8"] for k in ("raw>raw", "smooth>raw", "raw>smooth")) + (100 * d["smooth>raw"]["shrink"]["z70"],)


safe("v2 #66 E1b VOBench context-mean reward raw>raw, input-only washed, target-only washed; shrink z70 input-only", e1b, (16.8, 6.8, 24.4, 0.3))


def e2b_apex48():
    g = J("paper_lab/_scratch/tspace/result.json")["gravity"]
    g = g.get("raw", g)
    a = g["arc/apex48"]
    return tuple(round(a[m]["choice_acc"] * a[m]["n"]) for m in ("copy", "mean8", "mc_or"))


safe("v2 table 1-4b E2b apex48 picks copy, ctx-mean8, oracle (of 24)", e2b_apex48, (0, 10, 23), tol=0.01)


def pl(name, cell, keys=("model", "copy", "mean8", "oracle")):
    c = J(f"paper_lab/eval/results/{name}/passline_all.json")["cells"][cell]
    return tuple(c[k][0] for k in keys)


safe("v2 #67 harness vast70 carry gain (model, copy, mean8, upper)", lambda: pl("vast70", "carry gain all", ("model", "copy", "mean8", "truth")), (0.22, -0.02, -0.50, 0.90), tol=0.011)
safe("v2 #67 harness vast70 object-region late/moving", lambda: pl("vast70", "object-region late/moving"), (51.4, 51.1, 50.8, 95.3))
safe("v2 #67 harness vast70 object-region visible/moving", lambda: pl("vast70", "object-region visible/moving"), (79.2, 68.3, 56.7, 99.2))
safe("v2 #68 harness vast70 object-region gravity ledge / arc (model, copy, mean8, oracle)",
     lambda: pl("vast70", "object-region gravity ledge") + pl("vast70", "object-region gravity arc"), (30.6, 31.2, 41.0, 100.0, 11.1, 0.0, 16.7, 100.0))
safe("v2 #69 harness vast70 natural L1 mean (model, copy, mean8)",
     lambda: tuple(sum(J("paper_lab/eval/results/vast70/passline_all.json")["cells"]["natural L1"][k]) / 8 for k in ("model", "copy", "mean8")), (0.561, 0.720, 0.571), tol=0.0011)


safe("v2 table 1-3a harness ariel_ar40: carry gain, obj visible, obj late moving, obj late static",
     lambda: (pl("ariel_ar40", "carry gain all", ("model",))[0], pl("ariel_ar40", "object-region visible/moving")[0],
              pl("ariel_ar40", "object-region late/moving")[0], pl("ariel_ar40", "object-region late/static")[0]), (0.65, 92.5, 51.4, 93.1), tol=0.051)
safe("v2 table 1-3a harness ariel_ar40: obj gravity arc, ledge; vast70 late static (model, mean8, oracle)",
     lambda: (pl("ariel_ar40", "object-region gravity arc")[0], pl("ariel_ar40", "object-region gravity ledge")[0]) + pl("vast70", "object-region late/static", ("model", "mean8", "oracle")),
     (25.0, 7.6, 88.9, 100.0, 90.3))


def cosmos_r2():
    import glob
    d = "cvpr/results/eval/testbed/cosmos_ar4b_latent/"
    S, C = {}, {}
    for f in sorted(glob.glob(ROOT + "/" + d + "vobench_shard*.json")):
        j = json.load(open(f)); S.update(j["surprise"]); C.update(j["pixel_copy"])
    def acc(D, sel):
        ok = n = 0
        for k in D:
            for a, b in (("_pos_a", "_imp_ab"), ("_pos_b", "_imp_ba")):
                if k.endswith(a) and sel(k):
                    kb = k[: -len(a)] + b
                    if kb in D:
                        n += 1; ok += 1.0 if D[kb] > D[k] else 0.5 if D[kb] == D[k] else 0.0  # 원 스크립트와 같은 동점 규칙
        return 100.0 * ok / n, n
    lm = lambda k: "_late_" in k and "static" not in k
    lmp = lambda k: lm(k) and "_perm_" in k
    ls = lambda k: "_late_" in k and "static" in k
    g = J(d + "gravity_pick.json")
    arc = (g["primitive|arc"][0] * 36 + g["realistic|arc"][0] * 36) / 72
    return (acc(S, lambda k: True)[0], acc(S, lm)[0], acc(S, lm)[1], acc(S, lmp)[0], acc(S, ls)[0],
            acc(C, lm)[0], g["primitive|ledge"][0] + g["realistic|ledge"][0], arc)


safe("r2 #71 Cosmos VOBench all, late moving (acc, n), perm late moving, late static, pixel copy late moving; gravity ledge sum, arc",
     cosmos_r2, (75.5, 52.9, 360, 51.7, 73.6, 50.0, 0.0, 31.9))


safe("v2.3 #8 harness ip1late all: vast70, ariel, copy, mean8; moving/occluded vast70, copy, mean8",
     lambda: (pl("vast70", "all-token ip1late all", ("model",))[0], pl("ariel_ar40", "all-token ip1late all", ("model",))[0])
             + pl("vast70", "all-token ip1late all", ("copy", "mean8")) + pl("vast70", "all-token ip1late moving/occluded", ("model", "copy", "mean8")),
     (87.2, 87.8, 87.2, 81.1, 75.0, 71.7, 71.7))
safe("v2.3 table 1-3a natural carry gain all (vast70, ariel, copy, mean8, truth)",
     lambda: (pl("vast70", "natural carry gain all", ("model",))[0], pl("ariel_ar40", "natural carry gain all", ("model",))[0])
             + pl("vast70", "natural carry gain all", ("copy", "mean8", "truth")), (0.45, 0.44, 0.21, 0.29, 0.78), tol=0.011)
safe("v2.3 table 1-3a B2 false carry static / far (vast70 model, ariel model, copy)",
     lambda: tuple(round(sum(J(f"paper_lab/eval/results/{n}/passline_all.json")["cells"][c][k]) / 8, 3)
                   for c in ("false carry static clips", "false carry far cells (moving clips)") for n, k in (("vast70", "model"), ("ariel_ar40", "model"), ("vast70", "copy"))),
     (0.399, 0.397, 0.414, 0.334, 0.336, 0.413), tol=0.0011)


safe("v3 #72 main metric (object-region late/moving): A0 e10, AX e10, A1 e10, A1c e10, shortctx c8",
     lambda: tuple(pl(n, "object-region late/moving", ("model",))[0] for n in ("pilotA0_e10", "pilotAX_e10", "pilotA1_e10", "pilotA1c_e10", "ext_ar_vw_shortctx_e30_c8")),
     (52.2, 52.2, 50.6, 50.6, 52.6))


def _cmp(a, b, cell):
    r = J(f"paper_lab/eval/results/compare_{a}__{b}_all_{cell}.json")["rows"]["all"]
    return (r["diff"], r["ci95"][0], r["ci95"][1])


safe("v3 #72-73 paired: AX-A0 main, AX-A0 B1, A1-A0 main, A1-A0 B1",
     lambda: _cmp("pilotA0_e10", "pilotAX_e10", "late_moving") + _cmp("pilotA0_e10", "pilotAX_e10", "late_static")
             + _cmp("pilotA0_e10", "pilotA1_e10", "late_moving") + _cmp("pilotA0_e10", "pilotA1_e10", "late_static"),
     (0.0, -1.4, 1.4, 9.7, 4.2, 16.7, -1.7, -3.6, 0.3, -23.6, -31.9, -15.3))
safe("v3 #74 A1 training space: main, copy, mean8, oracle; B1; natural L1 model, mean8",
     lambda: pl("pilotA1_e10_tsp", "object-region late/moving") + (pl("pilotA1_e10_tsp", "object-region late/static", ("model",))[0],)
             + tuple(round(sum(J("paper_lab/eval/results/pilotA1_e10_tsp/passline_all.json")["cells"]["natural L1"][k]) / 8, 3) for k in ("model", "mean8")),
     (51.9, 51.1, 51.4, 96.1, 76.4, 0.290, 0.303), tol=0.051)

safe("v3 #75 shortctx c2: main, B1, visible, natural L1; paired main & B1 vs vast70",
     lambda: (pl("ext_ar_vw_shortctx_e30_c2", "object-region late/moving", ("model",))[0], pl("ext_ar_vw_shortctx_e30_c2", "object-region late/static", ("model",))[0],
              pl("ext_ar_vw_shortctx_e30_c2", "object-region visible/moving", ("model",))[0],
              round(sum(J("paper_lab/eval/results/ext_ar_vw_shortctx_e30_c2/passline_all.json")["cells"]["natural L1"]["model"]) / 8, 3))
             + _cmp("vast70", "ext_ar_vw_shortctx_e30_c2", "late_moving") + _cmp("vast70", "ext_ar_vw_shortctx_e30_c2", "late_static"),
     (50.6, 79.2, 82.5, 0.565, -0.8, -2.2, 0.3, -9.7, -16.7, -2.8), tol=0.051)

safe("v3 #74 A1c training space: main, B1, natural L1; paired main & B1 vs A0",
     lambda: (pl("pilotA1c_e10_tsp", "object-region late/moving", ("model",))[0], pl("pilotA1c_e10_tsp", "object-region late/static", ("model",))[0],
              round(sum(J("paper_lab/eval/results/pilotA1c_e10_tsp/passline_all.json")["cells"]["natural L1"]["model"]) / 8, 3))
             + _cmp("pilotA0_e10", "pilotA1c_e10_tsp", "late_moving") + _cmp("pilotA0_e10", "pilotA1c_e10_tsp", "late_static"),
     (50.8, 84.7, 0.288, -1.4, -3.3, 0.6, 0.0, -6.9, 8.3), tol=0.051)

safe("v3 #76 A2c/A4: main, B1; paired main & B1 vs A0 (A2c then A4)",
     lambda: (pl("pilotA2c_e10", "object-region late/moving", ("model",))[0], pl("pilotA4_e10", "object-region late/moving", ("model",))[0],
              pl("pilotA2c_e10", "object-region late/static", ("model",))[0], pl("pilotA4_e10", "object-region late/static", ("model",))[0])
             + _cmp("pilotA0_e10", "pilotA2c_e10", "late_moving") + _cmp("pilotA0_e10", "pilotA2c_e10", "late_static")
             + _cmp("pilotA0_e10", "pilotA4_e10", "late_moving") + _cmp("pilotA0_e10", "pilotA4_e10", "late_static"),
     (52.2, 51.7, 95.8, 94.4, 0.0, -1.4, 1.4, 11.1, 4.2, 19.4, -0.6, -1.9, 0.6, 9.7, 4.2, 16.7), tol=0.051)
safe("v3 #76 A2c/A4 natural L1 mean, natural carry gain all",
     lambda: tuple(round(sum(J(f"paper_lab/eval/results/{n}/passline_all.json")["cells"]["natural L1"]["model"]) / 8, 3) for n in ("pilotA2c_e10", "pilotA4_e10"))
             + tuple(pl(n, "natural carry gain all", ("model",))[0] for n in ("pilotA0_e10", "pilotA2c_e10", "pilotA4_e10")),
     (0.560, 0.560, 0.44, 0.44, 0.44), tol=0.006)


def _cmpm(a, b, cell):
    r = J(f"paper_lab/eval/results/compare_{a}__{b}_all_{cell}.json")["rows"]["all"]["margin_share_pct"]
    return (r["diff"], r["ci95"][0], r["ci95"][1])


safe("v3 #76 continuous margin share vs A0: AX, A2c, A4, A1, A1c (diff); AX ci; DiD",
     lambda: tuple(_cmpm("pilotA0_e10", b, "late_moving")[0] for b in ("pilotAX_e10", "pilotA2c_e10", "pilotA4_e10", "pilotA1_e10", "pilotA1c_e10"))
             + _cmpm("pilotA0_e10", "pilotAX_e10", "late_moving")[1:]
             + (J("paper_lab/eval/results/compare_pilotA0_e5__pilotAX_e5__pilotA0_e10__pilotAX_e10_all_late_moving.json")["rows"]["all"]["margin_share_pct"]["diff"],),
     (1.45, 1.50, 0.99, -3.56, -3.40, 1.01, 1.91, 0.22), tol=0.011)


def _kd(setname, name, svar, y):
    import io, contextlib
    sys.path.insert(0, ROOT + "/paper_lab/scripts")
    import carry_coef as CC
    with contextlib.redirect_stdout(io.StringIO()):
        rows = CC.placebo_table(setname, name, ys=(y,))
    r = [x for x in rows if x["s"] == svar and x["cells"] == "all" and x["y"] == y][0]
    return r["delta"]


safe("v3 #81 G19 post-hoc: data κ_Δ natural pix true, copy; natural12 pix true; models vast70, ariel natural pix",
     lambda: (_kd("natural", "_data", "pix", "true"), _kd("natural", "_data", "pix", "copy"), _kd("natural12", "_data", "pix", "true"),
              _kd("natural", "vast70", "pix", "model"), _kd("natural", "ariel_ar40", "pix", "model")),
     (0.152, 0.007, 0.241, 0.007, 0.011), tol=0.0015)
safe("v3 #82 far-cell margin share AX-A0, A1-A0; binary AX-A0",
     lambda: (J("paper_lab/eval/results/compare_pilotA0_e10__pilotAX_e10_all_late_moving_far48.json")["rows"]["all"]["margin_share_pct"]["diff"],
              J("paper_lab/eval/results/compare_pilotA0_e10__pilotA1_e10_all_late_moving_far48.json")["rows"]["all"]["margin_share_pct"]["diff"],
              J("paper_lab/eval/results/compare_pilotA0_e10__pilotAX_e10_all_late_moving_far48.json")["rows"]["all"]["diff"]),
     (-2.09, 11.44, 0.7), tol=0.051)

safe("v3.1 #83 state ckpts: core contrast bin & margin; sup_v1 vs vast70 bin, margin, far; nosup carry",
     lambda: (J("paper_lab/eval/results/compare_state_nosup__state_w005_all_late_moving.json")["rows"]["all"]["diff"],
              J("paper_lab/eval/results/compare_state_nosup__state_w005_all_late_moving.json")["rows"]["all"]["margin_share_pct"]["diff"],
              J("paper_lab/eval/results/compare_vast70__state_sup_v1_all_late_moving.json")["rows"]["all"]["diff"],
              J("paper_lab/eval/results/compare_vast70__state_sup_v1_all_late_moving.json")["rows"]["all"]["margin_share_pct"]["diff"],
              J("paper_lab/eval/results/compare_vast70__state_sup_v1_all_late_moving_far48.json")["rows"]["all"]["margin_share_pct"]["diff"],
              pl("state_nosup", "carry gain all", ("model",))[0]),
     (-0.3, 0.04, 1.9, 1.14, -7.71, 0.41), tol=0.06)

safe("v3.1 #84 G23 R2 a,b,c,d,e,pos; a-b diff; G23b t1 f_t,f_o; t2 f_t,a; dup jitter natural",
     lambda: tuple(J("paper_lab/_scratch/blockmotion/result.json")["arms"][k]["R2_test"] for k in ("a", "b", "c", "d", "e", "pos"))
             + (J("paper_lab/_scratch/blockmotion/result.json")["paired"]["a-b"]["diff"],
                J("paper_lab/_scratch/blockmotion/result_g23b.json")["target1"]["f_t"]["R2_test"], J("paper_lab/_scratch/blockmotion/result_g23b.json")["target1"]["f_o"]["R2_test"],
                J("paper_lab/_scratch/blockmotion/result_g23b.json")["target2"]["f_t"]["R2_test"], J("paper_lab/_scratch/blockmotion/result_g23b.json")["target2"]["a"]["R2_test"],
                J("paper_lab/_scratch/carry/encprops_natural.json")["dup"]["jitter_over_other_video"]),
     (0.030, 0.000, 0.000, 0.000, 0.001, 0.000, 0.030, 0.047, 0.045, 0.005, 0.002, 0.359), tol=0.0015)

safe("v3.1 #84 G23c MLP R2 pixel, a, b, e",
     lambda: tuple(J("paper_lab/_scratch/blockmotion/result_g23c.json")["arms"][k]["R2_test"] for k in ("pixel", "a", "b", "e")),
     (0.072, -0.027, -0.233, -0.283), tol=0.0015)


safe("v3.2 #81 §11-1 replication + fair e8: natural_s1 data, vast70, fairAXs0/s1, fairB0s0/s1; natural8_s1 data, vast70, fairAXs0, fairB0s1",
     lambda: (_kd("natural_s1", "_data", "pix", "true"), _kd("natural_s1", "vast70", "pix", "model"),
              _kd("natural_s1", "fairAXs0_e8", "pix", "model"), _kd("natural_s1", "fairAXs1_e8", "pix", "model"),
              _kd("natural_s1", "fairB0s0_e8", "pix", "model"), _kd("natural_s1", "fairB0s1_e8", "pix", "model"),
              _kd("natural8_s1", "_data", "pix", "true"), _kd("natural8_s1", "vast70", "pix", "model"),
              _kd("natural8_s1", "fairAXs0_e8", "pix", "model"), _kd("natural8_s1", "fairB0s1_e8", "pix", "model")),
     (0.134, 0.007, 0.007, 0.007, 0.008, 0.008, 0.189, 0.017, 0.018, 0.018), tol=0.0015)


def _fair(s, e, suf, key="margin"):
    d = J(f"paper_lab/eval/results/compare_fairB0s{s}_e{e}__fairAXs{s}_e{e}_all_late_{suf}.json")["rows"]["all"]
    return d["margin_share_pct"]["diff"] if key == "margin" else d["diff"]


safe("v3.2 #85 fair test e8 AX-B0 per seed: margin all s0,s1; far s0,s1; bin all s0,s1; far s0,s1; static s0,s1; far curve e2,e4,e6 (seed mean); B0 reheat s0,s1",
     lambda: (_fair(0, 8, "moving"), _fair(1, 8, "moving"), _fair(0, 8, "moving_far48"), _fair(1, 8, "moving_far48"),
              _fair(0, 8, "moving", "bin"), _fair(1, 8, "moving", "bin"), _fair(0, 8, "moving_far48", "bin"), _fair(1, 8, "moving_far48", "bin"),
              _fair(0, 8, "static", "bin"), _fair(1, 8, "static", "bin"))
             + tuple((_fair(0, e, "moving_far48") + _fair(1, e, "moving_far48")) / 2 for e in (2, 4, 6))
             + tuple(J(f"paper_lab/eval/results/compare_vast70__fairB0s{s}_e8_all_late_moving.json")["rows"]["all"]["margin_share_pct"]["diff"] for s in (0, 1)),
     (0.39, 0.57, -0.45, -0.53, 0.6, 0.0, -0.7, -0.3, 1.4, -1.4, -0.23, -0.49, -0.77, -1.99, -2.47), tol=0.06)


def _obs(cell, arm, key="acc"):
    return J("paper_lab/_scratch/m2/obs_score.json")["rows"][f"{cell}|{arm}"][key]


safe("v3.2 #86 tracking baseline: obs quad late all, far, margin all, far; base AR all, far; diff far ci lo,hi; static obs quad; ledge label quad, obs quad, base AR, far obs quad",
     lambda: (_obs("late_moving|all", "observed track + quadratic"), _obs("late_moving|far48", "observed track + quadratic"),
              100 * _obs("late_moving|all", "observed track + quadratic", "margin_share"), 100 * _obs("late_moving|far48", "observed track + quadratic", "margin_share"),
              _obs("late_moving|all", "base block AR"), _obs("late_moving|far48", "base block AR"),
              100 * _obs("late_moving|far48", "observed track + quadratic", "diff_vs_base")[1], 100 * _obs("late_moving|far48", "observed track + quadratic", "diff_vs_base")[2],
              _obs("late_static|all", "observed track + quadratic"),
              J("paper_lab/_scratch/ledge/score.json")["rows"]["all"]["label track + quadratic"]["acc"],
              J("paper_lab/_scratch/ledge/score.json")["rows"]["all"]["observed track + quadratic"]["acc"],
              J("paper_lab/_scratch/ledge/score.json")["rows"]["all"]["base block AR"]["acc"],
              J("paper_lab/_scratch/ledge/score.json")["rows"]["far48"]["observed track + quadratic"]["acc"]),
     (76.1, 69.0, 42.6, 22.9, 51.4, 41.3, 30.5, 53.9, 63.9, 0.0, 5.6, 30.6, 6.2), tol=0.06)


if __name__ == "__main__":
    w = max(len(r[0]) for r in ROWS)
    for cid, got, q, st in ROWS:
        print(f"{cid:<{w}} | got {got} | quoted {q} | {st}")
    bad = [r for r in ROWS if r[3] != "OK"]
    print(f"\n{len(ROWS)} checks, {len(ROWS) - len(bad)} OK, {len(bad)} not OK")
    sys.exit(0)
