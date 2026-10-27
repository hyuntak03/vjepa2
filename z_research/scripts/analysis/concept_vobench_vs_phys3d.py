"""물체 칸 정체 개념 벡터가 VOBench 와 3D 합성 (phys3d_single_v1) 에서 같은 방향인가 (2026-10-26 저녁, CPU).
사용자: "concept vector 자체가 VOBench 랑 3D phys single 이랑 다를 텐데".
VOBench: 끝 가림 캐시 (vobench_late/h.npy, 블록 단위 LN(target_encoder), 4 팔) 의 pos_a (물체 a) · pos_b (물체 b) 문맥 블록 0–7, 라벨의 물체가 보이고 (hidden < 0.5) 상자가 있을 때 물체 칸 평균.
  모양 · 색은 라벨 obj_a / obj_b. 빈 물체 (영속 b) 는 뺌. 3D 합성에 없는 모양 (capsule 등) 은 모양 개념에서 뺌.
3D 합성: objcell_concept_cache.py 의 encoder 물체 칸 평균 (tgt_true, 블록 8–15 합침, train 8,000 편).
개념 = 클래스 평균 − 전체 평균 (같은 클래스 집합으로 각각). cos (클래스마다 · 평균), VOBench 안 반쪽 재현도 (block 짝 / 홀), 라벨 섞은 영 기준.
  /venv/vjepa/bin/python z_research/scripts/analysis/concept_vobench_vs_phys3d.py   → z_analysis/results/concept_vobench_vs_phys3d.json · /workspace/auto_research_cache/objcell_concept/vobench_encoder_concept_{shape,color}.npy
"""
import csv, json, os
from collections import defaultdict
import numpy as np
from PIL import Image
SHAPES = ["sphere", "cube", "cylinder", "cone", "pyramid", "torus"]; COLORS = ["red", "orange", "yellow", "green", "cyan", "blue", "purple", "magenta"]
VROOT = "/workspace/dataset/benchmarks/VOBench_v1"; IDX = list(range(0, 96, 3))[:32]; REPO = "/workspace/world_model/vjepa2"
idx = defaultdict(dict)
for r in csv.DictReader(open(f"{REPO}/data_csv/vobench_v1/index.csv")):
    if r["occlusion"] == "late":
        idx[r["block_id"]][r["variant"]] = r
meta = json.load(open("/workspace/auto_research_cache/vobench_late/meta.json")); H = np.load("/workspace/auto_research_cache/vobench_late/h.npy", mmap_mode="r")
cc = np.arange(16) * 16 + 8.0
X, SH, CO, BLK = [], [], [], []
for mt in meta:
    r = idx[mt["block_id"]]["pos_a"]
    if r["objset"] != "primitive":
        continue
    folder = os.path.dirname(r["file_name"]); W0 = Image.open(os.path.join(VROOT, folder, "pos_a", "000000.png")).size[0]; sc = 256 / W0
    L = json.load(open(os.path.join(VROOT, folder, "label.json")))["labels"]
    for arm, key, obj in ((0, "A", r["obj_a"]), (1, "B", r["obj_b"])):
        if obj in ("", "empty"):
            continue
        shape, color = obj.split("/"); lab = {q["f"]: q for q in L.get(key, [])}; T = np.asarray(H[mt["i"], arm], dtype=np.float32)
        for b in range(8):
            q = [lab.get(IDX[2 * b]), lab.get(IDX[2 * b + 1])]
            if any(t is None for t in q) or max(t.get("hidden", 0) for t in q) >= 0.5:
                continue
            x0 = min(t["u"] - t["ext_u"] for t in q) * sc; x1 = max(t["u"] + t["ext_u"] for t in q) * sc; y0 = min(t["v"] - t["ext_v"] for t in q) * sc; y1 = max(t["v"] + t["ext_v"] for t in q) * sc
            cells = [int(rr * 16 + c) for rr in np.nonzero((cc >= y0) & (cc <= y1))[0] for c in np.nonzero((cc >= x0) & (cc <= x1))[0]]
            if not cells:
                continue
            X.append(T[b, cells].mean(0)); SH.append(shape); CO.append(color); BLK.append(int(mt["block_id"]))
X = np.stack(X); SH = np.array(SH); CO = np.array(CO); BLK = np.array(BLK)
print("VOBench 표본", len(X), "모양", dict(zip(*np.unique(SH, return_counts=True))), "색", dict(zip(*np.unique(CO, return_counts=True))))
Z = np.load("/workspace/auto_research_cache/objcell_concept/ar_train.npz"); tm = {m["i"]: m for m in json.load(open("/workspace/auto_research_cache/ar_layers/train_meta.json"))}
P = Z["tgt_true"].astype(np.float32).reshape(-1, 1280); psh = np.repeat([SHAPES[tm[int(i)]["shape"]] for i in Z["i"]], 8); pco = np.repeat([COLORS[tm[int(i)]["color"]] for i in Z["i"]], 8)
ok = ~np.isnan(P).any(1); P, psh, pco = P[ok], psh[ok], pco[ok]


def concept(X, y, classes):
    m = np.isin(y, classes); X, y = X[m], y[m]; M = np.stack([X[y == c].mean(0) for c in classes]); return M - X.mean(0)


cos = lambda A, B: (A * B).sum(1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1))
rng = np.random.default_rng(0); res = {"n_vobench": int(len(X))}
for fam, yv, yp, ALL in (("shape", SH, psh, SHAPES), ("color", CO, pco, COLORS)):
    classes = [c for c in ALL if (yv == c).sum() >= 10]
    Cv, Cp = concept(X, yv, classes), concept(P, yp, classes); c_ = cos(Cv, Cp)
    half = BLK % 2 == 0; rel = cos(concept(X[half], yv[half], classes), concept(X[~half], yv[~half], classes))
    yperm = rng.permutation(yv); null = cos(concept(X, yperm, classes), Cp)
    res[fam] = {"classes": classes, "n_per_class": [int((yv == c).sum()) for c in classes], "cos_vobench_vs_phys3d": dict(zip(classes, np.round(c_, 3).tolist())), "cos_mean": round(float(c_.mean()), 3),
                "vobench_split_half_reliability": round(float(rel.mean()), 3), "null_cos_mean": round(float(null.mean()), 3),
                "norm_ratio_vobench_over_phys3d": round(float(np.mean(np.linalg.norm(Cv, axis=1) / np.linalg.norm(Cp, axis=1))), 3)}
    print(fam, res[fam], flush=True)
    np.save(f"/workspace/auto_research_cache/objcell_concept/vobench_encoder_concept_{fam}.npy", {c: v for c, v in zip(classes, Cv)}, allow_pickle=True)   # VOBench 자체 개념 (GIF 개입용)
json.dump(res, open(f"{REPO}/z_analysis/results/concept_vobench_vs_phys3d.json", "w"), indent=1)
