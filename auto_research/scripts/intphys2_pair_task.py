"""[물리] IntPhys 2 Main 쌍 정확도를 짝 번호 (1 · 2) 별 · 과제 (game_name) 별로 (per_video.csv, 학습 없음).
3 창 (w16 · w32 · w48) × 문맥 6 개 설정을 모두 모아 평균 (설정끼리 강하게 상관 → 표준오차는 쌍 수로 읽을 것).
사용: python intphys2_pair_task.py <run_eval_dir> [...]"""
import csv, collections, os, sys, json
META = "/workspace/dataset/benchmarks/IntPhys2/Main/metadata.csv"
M = list(csv.DictReader(open(META)))
out = {}
for run in sys.argv[1:]:
    acc = collections.defaultdict(list)
    for w in ("w16", "w32", "w48"):
        c = [x for x in os.listdir(run) if x.startswith("intphys2_main") and (x.endswith(w) or f"_{w}_" in x)]
        if not c:
            continue
        by = collections.defaultdict(dict)
        for r in csv.DictReader(open(os.path.join(run, sorted(c)[0], "per_video.csv"))):
            g = M[int(r["row_index"])]["game_name"]
            by[(w, r["context_length"], r["scene_index"], r["pair_id"], r["condition"], r["camera"], g)][r["is_impossible"] == "True"] = float(r["surprise_avg"])
        for (w, C, s, pid, cond, cam, g), v in by.items():
            if len(v) < 2:
                continue
            x = 1.0 if v[True] > v[False] else (0.5 if v[True] == v[False] else 0.0)
            for key in ((cond, "pair" + pid), ("all", "pair" + pid), ("task", g)):
                acc[key].append(x)
    name = run.rstrip("/").replace("/eval", "").split("/")[-1]
    out[name] = {f"{a}|{b}": round(100 * sum(v) / len(v), 1) for (a, b), v in acc.items()}
    print(name)
    for cond in ("permanence", "immutability", "continuity", "solidity", "all"):
        print(f"  {cond:13s} pair1 {out[name][cond + '|pair1']:5.1f}  pair2 {out[name][cond + '|pair2']:5.1f}")
json.dump(out, open(os.path.join(os.path.dirname(__file__), "../results/intphys2_pair_task.json"), "w"), indent=1)
