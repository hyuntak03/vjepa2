#!/usr/bin/env python3
"""[P] IntPhys 2 물리 공간 surprise 기준선 표 (중간 · 최종 공용, CPU) (auto_research, 2026-10-22 KST).

  /venv/vjepa/bin/python auto_research/scripts/ip2_phys_report.py [--split Main] [--tag 중간]

입력: ip2_phys_surprise.py 의 {split}_rows.jsonl (처리된 영상까지) · {split}_U_*.npy (가능 영상 블록 흐름).
쌍 정확도 = 쌍 (장면, 짝 번호) 마다 불가능 > 가능. 네 영상이 다 처리된 장면만.
표: surprise 넷 (운동 · 외형 × 평균 · 최대) × [전체 · 조건 4 · 짝 1 / 짝 2 · 조건 × 짝 · 난이도 · 카메라].
같은 장면들에서 기준 AR 토큰 surprise (창 16 · 32 · 48, [비판] crit_ip2_xor 의 '최고 C') 정확도도 같이.
결과 항 · 문맥 항 분해 = [비판] crit_ip2_xor.py 를 그대로 import.
흐름 지속 계수 k = [물리] flow_persistence.py 의 coef 를 그대로 (가능 영상, 카메라 Fixed / Moving)."""
import argparse, csv, collections, glob, json, os, random, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="Main")
ap.add_argument("--dir", default=os.path.join(os.path.dirname(__file__), "../results/ip2_phys"))
ap.add_argument("--tag", default="")
a = ap.parse_args()
sys.path.insert(0, os.path.dirname(__file__))
import crit_ip2_xor as X                                                       # noqa: E402
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
rows = [json.loads(l) for l in open(os.path.join(a.dir, f"{a.split}_rows.jsonl"))]
sc = collections.defaultdict(dict)
for r in rows:
    sc[r["scene_index"]][r["type"]] = r
full = {s: v for s, v in sc.items() if len(v) == 4}
print(f"== {a.split} {a.tag}: 영상 {len(rows)} · 네 영상 다 된 장면 {len(full)} (쌍 {2 * len(full)})")
rng = np.random.default_rng(0)


def pairs(key):
    out = []
    for s, v in full.items():
        m = v["1_Possible"]
        for pid in ("1", "2"):
            out.append(dict(scene=s, pid=pid, cond=m["condition"], diff=m["difficulty"], cam=m["camera"],
                            ok=v[f"{pid}_Impossible"][key] > v[f"{pid}_Possible"][key]))
    return out


def acc(P, ci=False):
    if not P:
        return "—"
    v = 100 * np.mean([p["ok"] for p in P])
    if not ci:
        return f"{v:.1f}"
    scn = sorted({p["scene"] for p in P}); byS = collections.defaultdict(list)
    for p in P:
        byS[p["scene"]].append(p["ok"])
    bs = [100 * np.mean(np.concatenate([byS[s] for s in rng.choice(scn, len(scn))])) for _ in range(1000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return f"{v:.1f} [{lo:.1f}, {hi:.1f}]"


COND = ["permanence", "immutability", "continuity", "solidity"]
res = {"n_scenes": len(full)}
for key in ("motion_avg", "appear_avg", "motion_max", "appear_max"):
    P = pairs(key)
    line = {"전체": acc(P, True), "짝 1 · 짝 2": f"{acc([p for p in P if p['pid'] == '1'])} · {acc([p for p in P if p['pid'] == '2'])}"}
    for c in COND:
        sel = [p for p in P if p["cond"] == c]
        line[c] = f"{acc(sel)} ({acc([p for p in sel if p['pid'] == '1'])} / {acc([p for p in sel if p['pid'] == '2'])})"
    for d in ("Easy", "Medium", "Hard"):
        line[d] = acc([p for p in P if p["diff"] == d])
    for cm in ("Fixed", "Moving"):
        sel = [p for p in P if p["cam"] == cm]
        line[f"cam {cm}"] = f"{acc(sel)} ({acc([p for p in sel if p['pid'] == '1'])} / {acc([p for p in sel if p['pid'] == '2'])})"
    res[key] = line
    print(f"\n-- {key}: " + " | ".join(f"{k} {v}" for k, v in line.items()))

# 같은 장면에서 기준 AR 토큰 surprise (문맥 길이 C 고정) 의 같은 표
TOK = ((16, 12), (32, 8), (48, 36)) if a.split == "Main" else ()
TOKKEY_OF = {s_: str(int(s_)) for s_ in full}                                     # 토큰 채점기는 int("196_1") = 1961 로 읽었다 (파이썬 int 가 _ 를 자릿수 구분으로 받음)
TOKKEY = set(TOKKEY_OF.values())
for W, C in TOK:
    src = os.path.join(REPO, f"z_training/runs/natural_ar_vw_state/eval/intphys2_main_latest_w{W}/per_video.csv")
    val = {}
    for r in csv.DictReader(open(src)):
        if int(r["context_length"]) == C and r["scene_index"] in TOKKEY:
            val[(r["scene_index"], r["pair_id"], r["is_impossible"] == "True")] = float(r["surprise_avg"])
    P = []
    for s_, v in full.items():
        m = v["1_Possible"]
        for pid in ("1", "2"):
            P.append(dict(scene=s_, pid=pid, cond=m["condition"], diff=m["difficulty"], cam=m["camera"], ok=val[(TOKKEY_OF[s_], pid, True)] > val[(TOKKEY_OF[s_], pid, False)]))
    line = {"전체": acc(P, True), "짝 1 · 짝 2": f"{acc([p for p in P if p['pid'] == '1'])} · {acc([p for p in P if p['pid'] == '2'])}"}
    for c in COND:
        sel = [p for p in P if p["cond"] == c]
        line[c] = f"{acc(sel)} ({acc([p for p in sel if p['pid'] == '1'])} / {acc([p for p in sel if p['pid'] == '2'])})"
    for d in ("Easy", "Medium", "Hard"):
        line[d] = acc([p for p in P if p["diff"] == d])
    for cm in ("Fixed", "Moving"):
        sel = [p for p in P if p["cam"] == cm]
        line[f"cam {cm}"] = f"{acc(sel)} ({acc([p for p in sel if p['pid'] == '1'])} / {acc([p for p in sel if p['pid'] == '2'])})"
    res[f"token_AR_w{W}_C{C}"] = line
    print(f"\n-- token AR w{W} C{C}: " + " | ".join(f"{k} {v}" for k, v in line.items()))

# 결과 항 · 문맥 항 ([비판] 도구) — 처리된 장면만으로 임시 csv
tmp = os.path.join(a.dir, f"_{a.split}_tmp")
os.makedirs(tmp, exist_ok=True)
paths = []
for key in ("motion_avg", "appear_avg"):
    pth = os.path.join(tmp, f"{key}.csv"); paths.append(pth)
    with open(pth, "w") as f_:
        w = csv.writer(f_); w.writerow(["scene_index", "pair_id", "is_impossible", "condition", "camera", "difficulty", "context_length", "surprise_avg"])
        for s, v in full.items():
            for t, r in v.items():
                w.writerow([s, t.split("_")[0], str(t.endswith("Impossible")), r["condition"], r["camera"], r["difficulty"], 0, r[key]])
# 기준 AR 토큰 surprise, 같은 장면만
for W, _ in TOK:
    src = os.path.join(REPO, f"z_training/runs/natural_ar_vw_state/eval/intphys2_main_latest_w{W}/per_video.csv")
    pth = os.path.join(tmp, f"token_AR_w{W}.csv"); paths.append(pth)
    with open(src) as fi, open(pth, "w") as fo:
        rd = csv.DictReader(fi); w = csv.DictWriter(fo, fieldnames=rd.fieldnames); w.writeheader()
        for r in rd:
            if r["scene_index"] in TOKKEY:
                w.writerow(r)
print("\n== 결과 항 · 문맥 항 분해 ([비판] crit_ip2_xor.py, 장면 bootstrap). 토큰 문맥 길이 C 는 전체 253 장면에서 고른 값으로 고정 (16→12 · 32→8 · 48→36)")
sfx = ('_' + a.tag) if a.tag else ''
sys.argv = [sys.argv[0]] + paths[:2] + ["--out", os.path.join(a.dir, f"{a.split}_xor_phys{sfx}.json")]
X.main()
for pth, C in zip(paths[2:], (12, 8, 36)):
    sys.argv = [sys.argv[0], pth, "--C", str(C), "--out", os.path.join(a.dir, f"{a.split}_xor_{os.path.basename(pth)[:-4]}{sfx}.json")]
    X.main()

# 흐름 지속 계수 k ([물리] coef 그대로)
src_fp = open(os.path.join(os.path.dirname(__file__), "flow_persistence.py")).read()
ns = {}
exec(src_fp[src_fp.index("def cell_of"):src_fp.index("out = {}")], {"np": np, "G": 16, "P0": np.stack([np.divmod(np.arange(256), 16)[1] + 0.5, np.divmod(np.arange(256), 16)[0] + 0.5], 1)}, ns)
cell_of, coef = ns["cell_of"], ns["coef"]
G = 16; cy, cx = np.divmod(np.arange(G * G), G); P0 = np.stack([cx + 0.5, cy + 0.5], 1)
cam_of = {r["file_name"].split("/")[-1][:16]: r["camera"] for r in rows if r["type"].endswith("Possible")}
acc_k = {k: {"Y": [], "Xa": [], "Xb": []} for k in ("all", "Fixed", "Moving")}
for f in glob.glob(os.path.join(a.dir, f"{a.split}_U_*.npy")):
    key = os.path.basename(f)[len(a.split) + 3:-4]
    if key not in cam_of:
        continue
    U = np.load(f).astype(np.float32)                                            # U[k] = 블록 k+1 → k, (nb−1, 256, 2)
    for t in range(1, U.shape[0] - 1):
        u1, u0 = U[t], U[t - 1]                                                  # u1 = 블록 t+1 → t, u0 = 블록 t → t−1
        m1 = np.linalg.norm(u1, axis=1); mov = m1 >= 0.5
        s, ok = cell_of(P0 + u1); sel = np.where(mov & ok)[0]
        if sel.size == 0:
            continue
        for k in ("all", cam_of[key]):
            acc_k[k]["Y"].append(u1[sel]); acc_k[k]["Xa"].append(u0[s[sel]]); acc_k[k]["Xb"].append(u0[sel])
res["persistence"] = {}
print("\n== 흐름 지속 계수 (가능 영상, 블록 대표 장 간격 1/3 초) — 내용 따라가기 / 같은 칸")
for k, d in acc_k.items():
    if not d["Y"]:
        continue
    Y, Xa, Xb = (np.concatenate(d[z]) for z in ("Y", "Xa", "Xb"))
    res["persistence"][k] = {"follow_content": coef(Y, Xa), "same_cell": coef(Y, Xb)}
    print(f"  {k:7s} 내용 따라가기 {coef(Y, Xa)} | 같은 칸 {coef(Y, Xb)}")
json.dump(res, open(os.path.join(a.dir, f"{a.split}_report{('_' + a.tag) if a.tag else ''}.json"), "w"), indent=1, ensure_ascii=False)
