#!/usr/bin/env python
"""EVIDENCE.md v1 (paper_lab, round 1, 2026-10-21) — recompute key numbers from raw artifacts.

Read-only. Prints one line per check: id | recomputed value | value quoted in docs | match?
Run:  /data/hyuntak/anaconda3/envs/vjepa2/bin/python paper_lab/scripts/verify_evidence_v1.py
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


if __name__ == "__main__":
    w = max(len(r[0]) for r in ROWS)
    for cid, got, q, st in ROWS:
        print(f"{cid:<{w}} | got {got} | quoted {q} | {st}")
    bad = [r for r in ROWS if r[3] != "OK"]
    print(f"\n{len(ROWS)} checks, {len(ROWS) - len(bad)} OK, {len(bad)} not OK")
    sys.exit(0)
