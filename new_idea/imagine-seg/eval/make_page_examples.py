#!/usr/bin/env python3
"""결과 페이지 (samples/page_v1) 의 데이터 예시 그림 — 데이터셋마다 한 쌍, 문맥 4 장 | 미래 4 장 (2026-11-01).
글자는 그림에 넣지 않는다 (설명은 페이지 HTML 에). 출력: samples/page_v1/img/*.jpg
"""
import csv, os, sys
import numpy as np, cv2
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
OUT = os.path.join(os.path.dirname(__file__), "../samples/page_v1/img"); os.makedirs(OUT, exist_ok=True)
S = 176                                    # 한 장 크기 (px)
GAP = 14                                   # 문맥 | 미래 사이 틈

def rd(p):
    im = cv2.imread(p, cv2.IMREAD_COLOR)
    if im is None: raise FileNotFoundError(p)
    return cv2.resize(im, (S, S), interpolation=cv2.INTER_AREA)

def strip(ctx, fut):
    gap = np.full((S, GAP, 3), 255, np.uint8)
    pad = np.full((S, 4, 3), 255, np.uint8)
    row = lambda fs: np.concatenate(sum([[f, pad] for f in fs], [])[:-1], 1)
    return np.concatenate([row(ctx), gap, row(fut)], 1)

def save(name, rows):
    vgap = np.full((8, rows[0].shape[1], 3), 255, np.uint8)
    img = np.concatenate(sum([[r, vgap] for r in rows], [])[:-1], 0)
    cv2.imwrite(os.path.join(OUT, name), img, [cv2.IMWRITE_JPEG_QUALITY, 86]); print(name, img.shape)

def idx(ds, index="index.csv"):
    return list(csv.DictReader(open(os.path.join(REPO, "data_csv", ds, index))))

def pair(rows, pred):
    """pred 를 만족하는 첫 (가능, 불가능) 같은 쌍."""
    by = {}
    for r in rows:
        if pred(r): by.setdefault((r["block_id"], r["pair_id"]), {})[r["plausible"]] = r
    for k, d in by.items():
        if "1" in d and "0" in d: return d["1"], d["0"]
    raise RuntimeError("쌍 없음")

# 1) VOBench held-out — 실제 물체 · 영속 · 늦은 가림 · 방향 A (사과)
V = "/workspace/dataset/benchmarks/VOBench_v1"
pos, imp = pair(idx("vobench_v1_split_v2", "index_test.csv"), lambda r: r["condition"] == "realistic/permanence/late" and r["pair_id"] == "A" and "apple" in r["file_name"])
f = lambda r, raws: [rd(f"{V}/{r['file_name']}/{t:06d}.png") for t in raws]
c_raw, f_raw = [0, 15, 30, 45], [51, 63, 75, 93]
save("vobench.jpg", [strip(f(pos, c_raw), f(pos, f_raw)), strip(f(imp, c_raw), f(imp, f_raw))])

# 2) IntPhys 1 late — O1 (영속)
I1 = "/workspace/world_model/local_datasets/world/world_analysis/IntPhys1_late_uni"
pos, imp = pair(idx("intphys1_late_uni"), lambda r: r["block_type"] == "O1" and r["group"] == "moving/occluded")
g = lambda root, r, ts: [rd(f"{root}/{r['video_id']}/{t:06d}.png") for t in ts]
f1 = [16, 17, 18, 19]                     # 두 영상이 갈리는 장 (픽셀 차이 최대 17–19, 그 뒤는 같다)
print("intphys1", pos["video_id"], imp["video_id"])
c1 = [0, 8, 12, 14]                       # 공은 문맥 12–14 장에서만 보인다 (15–16 가림)
save("intphys1.jpg", [strip(g(I1, pos, c1), g(I1, pos, f1)), strip(g(I1, imp, c1), g(I1, imp, f1))])

# 3) IntPhys 2 — 영속 (permanence)
I2 = "/workspace/world_model/local_datasets/world/world_analysis/IntPhys2_violation_uni"
pos, imp = pair(idx("intphys2_violation_uni"), lambda r: r["block_type"] == "permanence")
save("intphys2.jpg", [strip(g(I2, pos, [0, 5, 10, 15]), g(I2, pos, [16, 21, 26, 31])), strip(g(I2, imp, [0, 5, 10, 15]), g(I2, imp, [16, 21, 26, 31]))])

# 4) real_bench — 실제 영상, 방향 A (있던 물체가 나오면 가능, 사라지면 불가능)
RB = "/workspace/dataset/benchmarks/real_bench/rubric_office_quads_c16f16"
pos, imp = pair(idx("real_bench_quads"), lambda r: r["pair_id"] == "A" and r["condition"] == "occlusion" and r["quality"] == "clean")
h = lambda r, ts: [rd(f"{RB}/{r['file_name']}/{t:06d}.png") for t in ts]
save("realbench.jpg", [strip(h(pos, [0, 5, 10, 15]), h(pos, [16, 21, 26, 31])), strip(h(imp, [0, 5, 10, 15]), h(imp, [16, 21, 26, 31]))])

# 5) 학습 데이터 phys3d_perm_v1 — 가려졌다 다시 나오는 클립 하나 + 점유 라벨 겹쳐 그리기 (문맥 16 장)
L = np.load(os.path.join(REPO, "data_csv/phys3d_perm_v1/occ_train.npy"), mmap_mode="r")
tr = [l.split() for l in open(os.path.join(REPO, "data_csv/phys3d_perm_v1/train_occ.csv"))]
want = sys.argv[1] if len(sys.argv) > 1 else "roll_behind"
pick = None
for path, row in tr:
    if f"/{want}/" not in path: continue
    x = np.asarray(L[int(row), 3])                                   # (16 블록, 256 칸)
    hid, vis = (x[8:] == 2).sum(1), (x[8:] == 1).sum(1)
    if hid[:3].min() >= 4 and vis[-2:].min() >= 4 and (x[6:8] == 1).sum() > 0:
        pick = (path, int(row), x); break
path, row, x = pick; print("phys3d", path, row)
cap = cv2.VideoCapture(path); fr = []
while True:
    ok, im = cap.read()
    if not ok: break
    fr.append(cv2.resize(im, (256, 256), interpolation=cv2.INTER_AREA))
print("frames", len(fr))
def over(t):
    im = fr[t].copy(); lab = x[t // 2].reshape(16, 16); ov = im.copy()
    for (yy, xx), v in np.ndenumerate(lab):
        if v == 1: cv2.rectangle(ov, (xx * 16, yy * 16), (xx * 16 + 15, yy * 16 + 15), (122, 175, 27), -1)   # 보이는 물체 (초록)
        if v == 2: cv2.rectangle(ov, (xx * 16, yy * 16), (xx * 16 + 15, yy * 16 + 15), (52, 104, 235), -1)   # 가려진 물체 (주황)
    return cv2.resize(cv2.addWeighted(ov, 0.55, im, 0.45, 0), (S, S), interpolation=cv2.INTER_AREA)
ct, ft = [0, 5, 10, 15], [17, 21, 25, 29]
plain = lambda ts: [cv2.resize(fr[t], (S, S), interpolation=cv2.INTER_AREA) for t in ts]
save("phys3d_train.jpg", [strip(plain(ct), plain(ft)), strip([over(t) for t in ct], [over(t) for t in ft])])
print("label blocks hid", (x == 2).sum(1).tolist(), "vis", (x == 1).sum(1).tolist())
