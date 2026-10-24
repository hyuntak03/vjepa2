"""R2 review (round 1) — independent read of VOBench tables.json (no reuse of verify_evidence_v1.py).
Read-only. Prints acc / A / B / n / CI for the late-occlusion cells used by EVIDENCE #5 #16."""
import json, os
ROOT = "/data/hyuntak/project/2026/2027_cvpr/vjepa2/z_research/VOBench_v1"
def J(p): return json.load(open(os.path.join(ROOT, p)))
for m in ["vjepa2", "ariel_ar_ep40", "vast_ar_state_ep70", "vll7_ctxt_tgtvideo"]:
    d = J(f"{m}/surprise/tables.json")
    for o in ["primitive", "realistic"]:
        t = d[o]
        ov = t["overall"]; cp = t.get("copy", {}).get("copy")
        late = t["viol_x_occ"]["all"]["late"] if "all" in t["viol_x_occ"] else None
        lp = t["viol_x_occ"]["permanence"]["late"]
        cl = t.get("copy_viol_x_occ", {}).get("all", {}).get("late", {}).get("copy")
        clp = t.get("copy_viol_x_occ", {}).get("permanence", {}).get("late", {}).get("copy")
        f = lambda c: f"{c['acc']:.1f} [{c['lo']:.1f},{c['hi']:.1f}] A{c['A']:.1f} B{c['B']:.1f} n{c['n']}"
        print(m, o)
        print("   overall", f(ov), "| copy", f(cp) if cp else None)
        if late: print("   late all", f(late), "| copy", f(cl) if cl else None)
        print("   late perm", f(lp), "| copy", f(clp) if clp else None)
