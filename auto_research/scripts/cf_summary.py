#!/usr/bin/env python3
"""[T] 반사실 본 렌더 요약 (학습 없음, CPU) (auto_research, 2026-10-22 KST): 장면 통과율 · 버린 이유 · 사건 시각 · 가림막 · 카메라 · 갈림 전 동일성.

  /venv/vjepa/bin/python auto_research/scripts/cf_summary.py [--splits train,test]

입력: /workspace/dataset/training_data/phys3d_cf_v1/<split>/results.jsonl · scene_*/<변형>.json · _frames · logs/cf_main_<split>_w*.log ("[cf] scene k attempt a: [...]" 줄).
갈림 전 동일성 (test, 사전 등록 기준 = 0): 장 t < t_event 의 PNG 최대 차 W_pos↔W_imp · N_pos↔N_imp.
인코딩이 있으면 (/workspace/auto_research_cache/phys3d_cf_v1/<split>_h.npy) 갈림 전 블록 (6b+3 < t_event) 토큰 차도 낸다 (0 이어야 함)."""
import argparse, collections, glob, json, os, re
import numpy as np
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/workspace/dataset/training_data/phys3d_cf_v1")
ap.add_argument("--cache", default="/workspace/auto_research_cache/phys3d_cf_v1")
ap.add_argument("--logs", default=os.path.join(os.path.dirname(__file__), "../logs"))
ap.add_argument("--splits", default="train,test")
a = ap.parse_args()
out = {}
for sp in a.splits.split(","):
    R = [json.loads(l) for l in open(os.path.join(a.root, sp, "results.jsonl"))] if os.path.exists(os.path.join(a.root, sp, "results.jsonl")) else []
    ok = [r for r in R if r.get("ok")]
    why = collections.Counter()
    for lg in glob.glob(os.path.join(a.logs, f"cf_main_{sp}_w*.log")):
        for line in open(lg, errors="ignore"):
            m = re.search(r"\[cf\] scene \d+ attempt \d+: \[(.*)\]", line)
            if m:
                for w in re.findall(r"'([^']*)'", m.group(1)):
                    why[re.sub(r"[-\d.]+", "#", w)] += 1
    att = collections.Counter(r.get("attempt") for r in ok)
    tev = [r.get("t_event") for r in ok]
    metas = [json.load(open(j)) for j in glob.glob(os.path.join(a.root, sp, "scene_*", "W_pos.json"))]
    s = dict(n_scenes=len(R), n_ok=len(ok), n_fail=len(R) - len(ok), attempts_hist=dict(sorted(att.items())),
             retry_reasons=dict(why.most_common()), t_event_q=[float(np.percentile(tev, q)) for q in (0, 25, 50, 75, 100)] if tev else None,
             screen_frac=float(np.mean([r.get("screen") for r in ok])) if ok else None,
             camera=dict(collections.Counter(m["env"]["camera"]["mode"] for m in metas)), obstacle_kind=dict(collections.Counter(m["obstacle"]["kind"] for m in metas)),
             sec_per_scene=float(np.mean([r.get("sec", 0) for r in ok])) if ok else None)
    if sp != "train":                                                       # 갈림 전 PNG 동일성 (짝 안)
        mx = []
        for sd in sorted(glob.glob(os.path.join(a.root, sp, "scene_*"))):
            if not os.path.exists(os.path.join(sd, "W_imp.json")):
                continue
            te = json.load(open(os.path.join(sd, "W_imp.json")))["t_event"]
            for p_, i_ in (("W_pos", "W_imp"), ("N_pos", "N_imp")):
                fa = sorted(glob.glob(os.path.join(sd, f"{p_}_frames", "f*.png")))[:te]; fb = sorted(glob.glob(os.path.join(sd, f"{i_}_frames", "f*.png")))[:te]
                mx.append(max(int(np.abs(np.asarray(Image.open(x), np.int16) - np.asarray(Image.open(y), np.int16)).max()) for x, y in zip(fa, fb)))
        s["pre_event_png_maxdiff_max"] = max(mx) if mx else None; s["n_pairs_checked"] = len(mx)
    hp, mp = os.path.join(a.cache, f"{sp}_h.npy"), os.path.join(a.cache, f"{sp}_meta.json")
    if os.path.exists(hp) and os.path.exists(mp):
        H = np.load(hp, mmap_mode="r"); M = json.load(open(mp)); by = collections.defaultdict(dict)
        for r in M:
            by[r["scene"]][r["variant"]] = r
        dd = []
        for sc_, v in by.items():
            te = next(iter(v.values()))["t_hit"] if "t_event" not in next(iter(v.values())) else next(iter(v.values()))["t_event"]
            nb = sum(1 for b in range(16) if 6 * b + 3 < (te if te is not None else 0))
            for p_, i_ in (("W_pos", "W_imp"), ("N_pos", "N_imp")):
                if p_ in v and i_ in v and nb > 0:
                    dd.append(float(np.abs(np.asarray(H[v[p_]["row"], :nb], np.float32) - np.asarray(H[v[i_]["row"], :nb], np.float32)).max()))
        s["pre_event_token_maxdiff_max"] = max(dd) if dd else None
    out[sp] = s
    print(f"== {sp}: {json.dumps(s, ensure_ascii=False)}", flush=True)
json.dump(out, open(os.path.join(os.path.dirname(__file__), "../results/cf_summary.json"), "w"), indent=1, ensure_ascii=False)
