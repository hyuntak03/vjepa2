"""정체 개념을 키운 AR 예측을 pixel decoder 로 그리기 — VOBench 끝 가림 (2026-10-26 저녁, GPU 1 장 · 4 GB).
사용자: "concept vector magnitude 키운 걸로 pixel decoding 해 봐".
모델: 블록 AR natural_ar_vw_state latest, 문맥 8 블록 → 굴림 8 블록 (identity_subtract_violations.py 와 같음). decoder: z_pixel_decoder vob_grav_i1_i2_vitb_both (VOBench 로 학습, 블록 latent → 2 장).
정체 개념 = 3D 합성 AR 예측의 물체 칸 개념 (모양 + 색, 걸음별, identity_subtract_violations.py 의 CON). 문맥 물체 (obj_a) 의 c 를 VOBench 라벨 A 의 물체 칸 (블록별) 에 β 배 더한다.
그림: 예시마다 한 줄 — [정답 latent 복원 · AR 예측 · 예측 + 4c] × 걸음 1 · 4 · 8 (블록의 첫 장). ○ = 그 블록의 진짜 물체 중심, × = 마지막 문맥 장의 물체 중심.
  CUDA_VISIBLE_DEVICES=0 MEM_GB=4 /venv/vjepa/bin/python -W ignore z_research/scripts/analysis/decode_identity_steer.py
출력: z_analysis/figures/decode_identity_steer.png · z_analysis/results/decode_identity_steer.json (물체 상자 안 픽셀 차이)
"""
import csv, json, os, sys
from collections import defaultdict
import numpy as np, torch
from PIL import Image

REPO = "/workspace/world_model/vjepa2"; sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "auto_research/scripts")); sys.path.insert(0, os.path.join(REPO, "z_pixel_decoder"))
dev = torch.device("cuda")
torch.cuda.set_per_process_memory_fraction(float(os.environ.get("MEM_GB", 4)) * 1024 ** 3 / torch.cuda.get_device_properties(0).total_memory, 0)
_argv = list(sys.argv); sys.argv = [sys.argv[0], "--loss", "l1", "--name", "x"]
import ar_crps_ft as FT
sys.argv = _argv
from model import PixelDecoder
S, C, K, D = 256, 8, 8, 1280
SHAPES = ["sphere", "cube", "cylinder", "cone", "pyramid", "torus"]; COLORS = ["red", "orange", "yellow", "green", "cyan", "blue", "purple", "magenta"]
VROOT = "/workspace/dataset/benchmarks/VOBench_v1"; IDX = list(range(0, 96, 3))[:32]
EXAMPLES = [(29, "static, color violation"), (65, "static, color violation"), (34, "moving (const. vel.), shape violation"), (89, "moving (ramp down), color violation")]
STEPS = [1, 4, 8]; BETA = 4.0

OC = np.load("/workspace/auto_research_cache/objcell_concept/ar_train.npz"); _meta = {m["i"]: m for m in json.load(open("/workspace/auto_research_cache/ar_layers/train_meta.json"))}
_sh = np.array([_meta[int(i)]["shape"] for i in OC["i"]]); _co = np.array([_meta[int(i)]["color"] for i in OC["i"]]); _PT = OC["pred_true"].astype(np.float32)
CON = {}
for k in range(1, 9):
    X = _PT[:, k - 1]; m = ~np.isnan(X).any(1); X, s_, c_ = X[m], _sh[m], _co[m]
    CON[k] = (np.stack([X[s_ == i].mean(0) for i in range(6)]) - X.mean(0), np.stack([X[c_ == i].mean(0) for i in range(8)]) - X.mean(0))


def cvec(obj, k):
    shape, color = obj.split("/"); cs, cc = CON[k]
    return (cs[SHAPES.index(shape)] if shape in SHAPES else 0) + (cc[COLORS.index(color)] if color in COLORS else 0)


def labels(folder, key):
    L_ = json.load(open(os.path.join(VROOT, folder, "label.json")))["labels"].get(key, []); return {q["f"]: q for q in L_}


def obj_cells(lab, sc):
    cc = np.arange(16) * 16 + 8.0; out = []
    for b in range(8, 16):
        q = [lab.get(IDX[2 * b]), lab.get(IDX[2 * b + 1])]
        if any(t is None for t in q):
            out.append([]); continue
        x0 = min(t["u"] - t["ext_u"] for t in q) * sc; x1 = max(t["u"] + t["ext_u"] for t in q) * sc; y0 = min(t["v"] - t["ext_v"] for t in q) * sc; y1 = max(t["v"] + t["ext_v"] for t in q) * sc
        cells = [int(r * 16 + c) for r in np.nonzero((cc >= y0) & (cc <= y1))[0] for c in np.nonzero((cc >= x0) & (cc <= x1))[0]]
        out.append(cells or [int(min(15, np.mean([t["v"] for t in q]) * sc // 16) * 16 + min(15, np.mean([t["u"] for t in q]) * sc // 16))])
    return out


idx = defaultdict(dict)
for r in csv.DictReader(open(f"{REPO}/data_csv/vobench_v1/index.csv")):
    if r["occlusion"] == "late":
        idx[r["block_id"]][r["variant"]] = r
meta = {m["i"]: m for m in json.load(open("/workspace/auto_research_cache/vobench_late/meta.json"))}
H = np.load("/workspace/auto_research_cache/vobench_late/h.npy", mmap_mode="r")
ar = FT.load_pred("natural_ar_vw_state", "latest.pt").to(dev).eval()
st = torch.load(f"{REPO}/z_pixel_decoder/runs/vob_grav_i1_i2_vitb_both/latest.pt", map_location="cpu", weights_only=False); mc = st["cfg"]["model"]
dec = PixelDecoder(in_dim=1280, dim=mc["dim"], depth=mc["depth"], heads=mc["heads"]).to(dev).eval(); dec.load_state_dict(st["decoder"]); del st
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1, 1)
decode = lambda tok: (dec(tok.float()) * STD + MEAN).clamp(0, 1)[:, :, 0].permute(0, 2, 3, 1).cpu().numpy()   # (n, H, W, 3) 블록 첫 장

rows, res = [], {}
for i, desc in EXAMPLES:
    mt = meta[i]; r = idx[mt["block_id"]]["pos_a"]; folder = os.path.dirname(r["file_name"])
    W0 = Image.open(os.path.join(VROOT, folder, "pos_a", "000000.png")).size[0]; sc = 256 / W0; labA = labels(folder, "A")
    hh = torch.from_numpy(np.asarray(H[i, 0], dtype=np.float32)).to(dev)                                   # pos_a (16, 256, 1280)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        p = ar.rollout_kv(hh[:C].reshape(1, C * S, D), torch.arange(C * S, device=dev).unsqueeze(0), K).float()[0].view(K, S, D)
    c = torch.from_numpy(np.stack([cvec(r["obj_a"], k) for k in range(1, 9)])).float().to(dev)
    mask = torch.zeros(K, S, 1, device=dev); cells = obj_cells(labA, sc)
    for j_, cl in enumerate(cells):
        mask[j_, cl] = 1.0
    q = p + BETA * c[:, None, :] * mask
    sel = [k - 1 for k in STEPS]
    with torch.no_grad():
        imgs = {"ground truth (latent decoded)": decode(hh[C:][sel]), "AR prediction": decode(p[sel]), f"AR prediction + {BETA:g}×concept at object cells": decode(q[sel])}
    true_c = [(np.mean([labA[IDX[2 * (7 + k)]]["u"], labA[IDX[2 * (7 + k) + 1]]["u"]]) * sc, np.mean([labA[IDX[2 * (7 + k)]]["v"], labA[IDX[2 * (7 + k) + 1]]["v"]]) * sc) if IDX[2 * (7 + k)] in labA else None for k in STEPS]
    last = labA.get(IDX[15]); last_c = (last["u"] * sc, last["v"] * sc) if last else None
    # 물체 상자 (진짜 자리 칸) 안 픽셀이 정답과 얼마나 다른가
    err = {}
    for nm, im in imgs.items():
        e = []
        for jj, k in enumerate(STEPS):
            cl = cells[k - 1]; mk = np.zeros((256, 256), bool)
            for cc_ in cl:
                rr, cc2 = divmod(cc_, 16); mk[rr * 16:(rr + 1) * 16, cc2 * 16:(cc2 + 1) * 16] = True
            e.append(round(float(np.abs(im[jj] - imgs["ground truth (latent decoded)"][jj])[mk].mean()), 4))
        err[nm] = e
    res[f"{i} {r['obj_a']} {desc}"] = {"object_box_L1_vs_gt_by_step": err, "steps": STEPS}
    rows.append((f"{r['obj_a']} — {desc}", imgs, true_c, last_c))
    print(i, desc, err, flush=True)
json.dump(res, open(f"{REPO}/z_analysis/results/decode_identity_steer.json", "w"), indent=1)

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "DejaVu Sans"
names = list(rows[0][1].keys()); ncol = len(names) * len(STEPS)
fig, axes = plt.subplots(len(rows), ncol, figsize=(1.75 * ncol, 2.0 * len(rows) + 0.6))
for ri, (title, imgs, true_c, last_c) in enumerate(rows):
    for ni, nm in enumerate(names):
        for si, k in enumerate(STEPS):
            ax = axes[ri, ni * len(STEPS) + si]; ax.imshow(imgs[nm][si]); ax.set_xticks([]); ax.set_yticks([])
            if true_c[si] is not None:
                ax.plot(*true_c[si], "o", mfc="none", mec="white", ms=13, mew=1.5)
            if last_c is not None:
                ax.plot(*last_c, "x", color="yellow", ms=8, mew=1.5)
            if ri == 0:
                ax.set_title(f"{nm}\nstep {k} ({0.5 * k:.1f} s)" if si == 1 else f"step {k} ({0.5 * k:.1f} s)", fontsize=8)
    axes[ri, 0].set_ylabel(title, fontsize=8)
fig.suptitle("Decoded AR predictions with the identity concept amplified at object cells (VOBench late occlusion; ○ true object, × last seen)", fontsize=11, y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(f"{REPO}/z_analysis/figures/decode_identity_steer.png", dpi=150, bbox_inches="tight")
print("saved")
