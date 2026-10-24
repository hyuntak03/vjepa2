"""IntPhys 2 Main 조건별 쌍 정확도 재계산 (per_video.csv 에서, 학습·모델 없음).

쌍 = (scene_index, pair_id). 정답 = surprise_avg(불가능) > surprise_avg(가능). 동점 0.5.
보고: 창(w)·C 마다 조건 × 카메라 정확도 → (a) overall 최고 C, (b) C 평균, (c) 창 3 × C 6 전부 평균.
사용: python intphys2_by_condition.py <run_dir> [<run_dir> ...]
"""
import sys, csv, json, collections, os, math

def load(path):
    rows = list(csv.DictReader(open(path)))
    by = collections.defaultdict(dict)  # (C) -> {(scene,pair,imp): (surp, cond, diff, cam)}
    for r in rows:
        C = int(r["context_length"])
        key = (r["scene_index"], r["pair_id"], r["is_impossible"] == "True")
        by[C][key] = (float(r["surprise_avg"]), r["condition"], r["difficulty"], r["camera"])
    return by

def pair_scores(d):
    out = []  # (cond, diff, cam, correct, margin_rel)
    pairs = collections.defaultdict(dict)
    for (s, p, imp), v in d.items():
        pairs[(s, p)][imp] = v
    for k, v in pairs.items():
        if True not in v or False not in v:
            continue
        si, cond, diff, cam = v[True]
        sp = v[False][0]
        c = 1.0 if si > sp else (0.5 if si == sp else 0.0)
        out.append((cond, diff, cam, c, (si - sp) / ((si + sp) / 2)))
    return out

def acc(ps, f):
    xs = [p[3] for p in ps if f(p)]
    return (100 * sum(xs) / len(xs), len(xs)) if xs else (float("nan"), 0)

CONDS = ["permanence", "immutability", "continuity", "solidity"]
GROUPS = [("all", lambda p: True)] + [(c, (lambda c: lambda p: p[0] == c)(c)) for c in CONDS] + \
         [(f"{c}|{cam}", (lambda c, cam: lambda p: p[0] == c and p[2] == cam)(c, cam)) for c in CONDS for cam in ["Fixed", "Moving"]] + \
         [("cam=Fixed", lambda p: p[2] == "Fixed"), ("cam=Moving", lambda p: p[2] == "Moving")]

def main():
    res = {}
    for run in sys.argv[1:]:
        runname = run.rstrip("/").replace("/eval","").split("/")[-1]
        per_w = {}
        pooled = collections.defaultdict(list)
        for w in ["w16", "w32", "w48"]:
            cands = [x for x in os.listdir(run) if x.startswith("intphys2_main") and x.endswith(w) or (x.startswith("intphys2_main") and f"_{w}_" in x)]
            cands = [x for x in cands if os.path.exists(os.path.join(run, x, "per_video.csv"))]
            if not cands:
                continue
            path = os.path.join(run, sorted(cands)[0], "per_video.csv")
            by = load(path)
            tab = {}
            for C, d in sorted(by.items()):
                ps = pair_scores(d)
                tab[C] = {g: acc(ps, f) for g, f in GROUPS}
                for g, f in GROUPS:
                    a, n = tab[C][g]
                    if n:
                        pooled[g].append(a)
            bestC = max(tab, key=lambda C: tab[C]["all"][0])
            meanC = {g: sum(tab[C][g][0] for C in tab) / len(tab) for g, _ in GROUPS}
            per_w[w] = {"file": path, "bestC": bestC, "at_bestC": {g: tab[bestC][g] for g, _ in GROUPS},
                        "meanC": meanC, "maxC": {g: max(tab[C][g][0] for C in tab) for g, _ in GROUPS},
                        "n": {g: tab[bestC][g][1] for g, _ in GROUPS}}
        res[runname] = {"per_w": per_w, "pooled_mean": {g: sum(v) / len(v) for g, v in pooled.items()}}
    print(json.dumps(res, indent=1, default=str)[:200])
    for rn, r in res.items():
        print(f"\n=== {rn} ===")
        hdr = f"{'group':24s} {'n':>4s} " + " ".join(f"{w}@best {w}meanC" for w in r['per_w']) + "  pooled"
        print(hdr)
        for g, _ in GROUPS:
            n = next(iter(r["per_w"].values()))["n"][g]
            cells = " ".join(f"{r['per_w'][w]['at_bestC'][g][0]:9.1f} {r['per_w'][w]['meanC'][g]:9.1f}" for w in r["per_w"])
            print(f"{g:24s} {n:4d} {cells}  {r['pooled_mean'].get(g, float('nan')):6.1f}")
    out = os.path.join(os.path.dirname(__file__), "..", "results", "intphys2_by_condition.json")
    json.dump(res, open(out, "w"), indent=1, default=str)
    print("saved", out)

if __name__ == "__main__":
    main()
